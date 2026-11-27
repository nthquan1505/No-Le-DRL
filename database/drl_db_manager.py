"""
Singleton Database Manager for `drl_database.db` based on SYSTEM_ARCHITECTURE.md.

Tables managed:
1. `events`
2. `users`
3. `event_subscription`
4. `fanpages`
5. `notifications`
6. `rubric`

All timestamps use ISO-8601 format in UTC+7 (Asia/Ho_Chi_Minh).
List-of-values columns are stored as JSON-encoded `TEXT` arrays.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union
from zoneinfo import ZoneInfo

try:
    from config import DRL_DB_PATH
except ImportError:
    DRL_DB_PATH = Path(__file__).resolve().parent.parent / "drl_database.db"

logger = logging.getLogger(__name__)

T = TypeVar("T")

try:
    UTC7_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
except Exception:
    UTC7_TIMEZONE = timezone(timedelta(hours=7))


# JSON-encoded list columns per table for optional automatic serialization/deserialization
TABLE_JSON_LIST_FIELDS: Dict[str, Tuple[str, ...]] = {
    "events": ("keywords", "images_url"),
    "users": ("weights", "credits", "preferred_topics", "preferred_formats"),
    "rubric": ("max_credit",),
}

ALLOWED_TABLES = frozenset(
    {"events", "users", "event_subscription", "fanpages", "notifications", "rubric"}
)

# Expected columns for safe in-place schema migration on existing databases
EXPECTED_NEW_COLUMNS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "events": (("coordinate", "TEXT"),),
    "users": (("default_coordinate", "TEXT"),),
    "event_subscription": (
        ("gain", "REAL CHECK (gain IS NULL OR (gain >= 0.0 AND gain <= 1.0))"),
        ("fit", "REAL CHECK (fit IS NULL OR (fit >= 0.0 AND fit <= 1.0))"),
        ("interested", "REAL CHECK (interested IS NULL OR (interested >= 0.0 AND interested <= 1.0))"),
        ("urgency", "REAL CHECK (urgency IS NULL OR (urgency >= 0.0 AND urgency <= 1.0))"),
    ),
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    keywords TEXT NOT NULL DEFAULT '[]',
    is_online BOOLEAN NOT NULL DEFAULT 0 CHECK (is_online IN (0, 1)),
    reg_start_datetime TEXT,
    reg_end_datetime TEXT,
    event_start_datetime TEXT,
    event_end_datetime TEXT,
    location TEXT,
    coordinate TEXT,
    fee TEXT,
    target_audience TEXT,
    topic TEXT,
    format TEXT,
    award TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    sources TEXT,
    post_url TEXT,
    images_url TEXT NOT NULL DEFAULT '[]',
    category INTEGER,
    min_credit INTEGER
);

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username TEXT UNIQUE,
    password TEXT,
    email TEXT,
    default_coordinate TEXT,
    start_year INTEGER,
    weights TEXT NOT NULL DEFAULT '[0.25, 0.25, 0.25, 0.25]',
    credits TEXT NOT NULL DEFAULT '[0, 0, 0, 0, 0]',
    preferred_topics TEXT NOT NULL DEFAULT '[]',
    preferred_formats TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS event_subscription (
    user_id TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    gain REAL CHECK (gain IS NULL OR (gain >= 0.0 AND gain <= 1.0)),
    fit REAL CHECK (fit IS NULL OR (fit >= 0.0 AND fit <= 1.0)),
    interested REAL CHECK (interested IS NULL OR (interested >= 0.0 AND interested <= 1.0)),
    urgency REAL CHECK (urgency IS NULL OR (urgency >= 0.0 AND urgency <= 1.0)),
    role TEXT,
    status TEXT,
    credit INTEGER,
    other_awards TEXT,
    PRIMARY KEY (user_id, event_id),
    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (event_id) REFERENCES events (event_id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS fanpages (
    fanpage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fanpage_name TEXT,
    number_of_events INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS notifications (
    notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    type TEXT,
    content TEXT,
    event_id INTEGER,
    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (event_id) REFERENCES events (event_id) ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS rubric (
    school_id INTEGER PRIMARY KEY AUTOINCREMENT,
    max_credit TEXT NOT NULL DEFAULT '[20, 20, 20, 20, 20]'
);

CREATE INDEX IF NOT EXISTS idx_events_unclassified ON events (category);
CREATE INDEX IF NOT EXISTS idx_events_uncredited ON events (min_credit);
CREATE INDEX IF NOT EXISTS idx_event_subscription_event_id ON event_subscription (event_id);
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications (user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_event_id ON notifications (event_id);
"""


def now_iso_utc7() -> str:
    """Return the current timestamp in ISO-8601 format with UTC+7 (Asia/Ho_Chi_Minh) offset."""
    return datetime.now(UTC7_TIMEZONE).isoformat()


def to_iso_utc7(value: Union[datetime, str, None]) -> Optional[str]:
    """Normalize a datetime or ISO string to UTC+7 ISO-8601 format."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value.replace(tzinfo=UTC7_TIMEZONE) if value.tzinfo is None else value.astimezone(UTC7_TIMEZONE)
        return dt.isoformat()
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        dt = parsed.replace(tzinfo=UTC7_TIMEZONE) if parsed.tzinfo is None else parsed.astimezone(UTC7_TIMEZONE)
        return dt.isoformat()
    except ValueError:
        return text


def encode_json_list(value: Any, default: Optional[List[Any]] = None) -> str:
    """Safely encode a Python list (or JSON string) into a JSON array string."""
    fallback_list = [] if default is None else default
    if value is None:
        return json.dumps(fallback_list, ensure_ascii=False)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return json.dumps(fallback_list, ensure_ascii=False)
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            return json.dumps([stripped], ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    return json.dumps(fallback_list, ensure_ascii=False)


def decode_json_list(value: Any, default: Optional[List[Any]] = None) -> List[Any]:
    """Safely decode a JSON array string into a Python list with fallback."""
    fallback_list = [] if default is None else list(default)
    if value is None:
        return fallback_list
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return fallback_list
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, list) else fallback_list
        except Exception:
            return fallback_list
    return fallback_list


class DRLDatabaseManager:
    """
    Thread-safe Singleton SQLite manager for `drl_database.db`.

    Provides:
    - Thread-local persistent connections with WAL mode, foreign keys, and busy timeouts
    - Automatic retry on transient SQLite lock contention
    - Safe query helpers (`fetch_one`, `fetch_all`, `fetch_scalar`, `execute_write`, `execute_many`)
      with transaction rollback and configurable fallback return values on errors
    """

    _instance: Optional["DRLDatabaseManager"] = None
    _instance_lock = threading.Lock()

    def __new__(
        cls,
        db_path: Optional[Union[str, Path]] = None,
        busy_timeout_ms: int = 5000,
        max_retries: int = 3,
    ) -> "DRLDatabaseManager":
        with cls._instance_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        busy_timeout_ms: int = 5000,
        max_retries: int = 3,
    ) -> None:
        if getattr(self, "_initialized", False):
            return
        with self._instance_lock:
            if getattr(self, "_initialized", False):
                return
            self.db_path: Path = Path(db_path) if db_path is not None else Path(DRL_DB_PATH)
            self.busy_timeout_ms: int = max(1000, int(busy_timeout_ms))
            self.max_retries: int = max(1, int(max_retries))
            self._local = threading.local()
            self._schema_ready = False
            self.initialize_schema()
            atexit.register(self.close_connection)
            self._initialized = True

    @classmethod
    def get_instance(cls, db_path: Optional[Union[str, Path]] = None) -> "DRLDatabaseManager":
        """Return the singleton DRLDatabaseManager instance."""
        return cls(db_path=db_path)

    @classmethod
    def reset_instance(cls) -> None:
        """Close active thread connection and reset singleton instance (primarily for testing)."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.close_connection()
                cls._instance = None

    def _create_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.busy_timeout_ms / 1000.0,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms};")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _is_connection_usable(self, conn: Optional[sqlite3.Connection]) -> bool:
        if conn is None:
            return False
        try:
            conn.execute("SELECT 1;")
            return True
        except sqlite3.Error:
            return False

    def get_connection(self) -> sqlite3.Connection:
        """Return a healthy thread-local SQLite connection, reconnecting automatically if needed."""
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if not self._is_connection_usable(conn):
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            conn = self._create_connection()
            self._local.conn = conn
        else:
            conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def close_connection(self) -> None:
        """Close the current thread's SQLite connection if open."""
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception as exc:
                logger.debug("Error while closing SQLite connection: %s", exc)
            finally:
                self._local.conn = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """
        Context manager for atomic read/write transactions.
        Commits on success, rolls back cleanly on any exception.
        """
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except sqlite3.Error as rb_exc:
                logger.warning("Failed to roll back transaction cleanly: %s", rb_exc)
                self.close_connection()
            raise

    def _run_with_retry(
        self,
        operation: Callable[[sqlite3.Connection], T],
        fallback: T,
        is_write: bool = False,
        raise_on_error: bool = False,
    ) -> T:
        """
        Execute a database operation with automatic retry on transient lock errors,
        transaction rollback on failure, and safe fallback value return.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                conn = self.get_connection()
                result = operation(conn)
                if is_write:
                    conn.commit()
                return result
            except sqlite3.OperationalError as exc:
                last_error = exc
                self._safe_rollback()
                msg = str(exc).lower()
                is_locked = "locked" in msg or "busy" in msg
                if is_locked and attempt < self.max_retries:
                    backoff = 0.05 * (2 ** (attempt - 1))
                    logger.warning(
                        "SQLite busy/locked (attempt %d/%d), retrying in %.2fs: %s",
                        attempt,
                        self.max_retries,
                        backoff,
                        exc,
                    )
                    time.sleep(backoff)
                    continue
                # Reset potentially stale connection on non-lock operational error
                if not is_locked:
                    self.close_connection()
                break
            except sqlite3.Error as exc:
                last_error = exc
                self._safe_rollback()
                break
            except Exception as exc:
                last_error = exc
                self._safe_rollback()
                break

        logger.error("Database operation failed on %s: %s", self.db_path, last_error)
        if raise_on_error and last_error is not None:
            raise last_error
        return fallback

    def _safe_rollback(self) -> None:
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.rollback()
            except sqlite3.Error:
                self.close_connection()

    def initialize_schema(self) -> bool:
        """Create the 6 DRL architecture tables, migrate missing columns, and create indexes."""
        def _init(conn: sqlite3.Connection) -> bool:
            conn.executescript(SCHEMA_SQL)
            for table, cols in EXPECTED_NEW_COLUMNS.items():
                existing = {
                    row["name"] for row in conn.execute(f"PRAGMA table_info({table});").fetchall()
                }
                for col_name, col_def in cols:
                    if col_name not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def};")
            conn.execute("PRAGMA foreign_keys = ON;")
            self._schema_ready = True
            return True

        return self._run_with_retry(_init, fallback=False, is_write=True, raise_on_error=True)

    @staticmethod
    def _row_to_dict(
        row: sqlite3.Row,
        json_list_fields: Sequence[str] = (),
    ) -> Dict[str, Any]:
        data = dict(row)
        for field in json_list_fields:
            if field in data:
                data[field] = decode_json_list(data[field])
        if "is_online" in data and data["is_online"] is not None:
            data["is_online"] = bool(data["is_online"])
        return data

    # ------------------------------------------------------------------
    # Core Query & Execution Methods (with Safe Error Fallbacks)
    # ------------------------------------------------------------------

    def fetch_one(
        self,
        query: str,
        params: Union[Sequence[Any], Dict[str, Any]] = (),
        fallback: Optional[Dict[str, Any]] = None,
        json_list_fields: Sequence[str] = (),
        raise_on_error: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Execute a SELECT query and return the first row as a dict, or `fallback`."""
        def _op(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
            cur = conn.execute(query, params)
            row = cur.fetchone()
            cur.close()
            if row is None:
                return fallback
            return self._row_to_dict(row, json_list_fields=json_list_fields)

        return self._run_with_retry(
            _op, fallback=fallback, is_write=False, raise_on_error=raise_on_error
        )

    def fetch_all(
        self,
        query: str,
        params: Union[Sequence[Any], Dict[str, Any]] = (),
        fallback: Optional[List[Dict[str, Any]]] = None,
        json_list_fields: Sequence[str] = (),
        raise_on_error: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return all rows as a list of dicts, or `fallback` (defaults to `[]`)."""
        safe_fallback: List[Dict[str, Any]] = [] if fallback is None else fallback

        def _op(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
            cur = conn.execute(query, params)
            rows = cur.fetchall()
            cur.close()
            return [self._row_to_dict(r, json_list_fields=json_list_fields) for r in rows]

        return self._run_with_retry(
            _op, fallback=safe_fallback, is_write=False, raise_on_error=raise_on_error
        )

    def fetch_scalar(
        self,
        query: str,
        params: Union[Sequence[Any], Dict[str, Any]] = (),
        fallback: Any = None,
        raise_on_error: bool = False,
    ) -> Any:
        """Execute a SELECT query and return the first column of the first row, or `fallback`."""
        def _op(conn: sqlite3.Connection) -> Any:
            cur = conn.execute(query, params)
            row = cur.fetchone()
            cur.close()
            return row[0] if row is not None else fallback

        return self._run_with_retry(
            _op, fallback=fallback, is_write=False, raise_on_error=raise_on_error
        )

    def execute_write(
        self,
        query: str,
        params: Union[Sequence[Any], Dict[str, Any]] = (),
        return_lastrowid: bool = False,
        fallback: Optional[int] = None,
        raise_on_error: bool = False,
    ) -> Optional[int]:
        """
        Execute an INSERT/UPDATE/DELETE statement atomically.
        Returns `cursor.lastrowid` when `return_lastrowid=True`, otherwise `cursor.rowcount`,
        or `fallback` on failure.
        """
        def _op(conn: sqlite3.Connection) -> Optional[int]:
            cur = conn.execute(query, params)
            out = cur.lastrowid if return_lastrowid else cur.rowcount
            cur.close()
            return out

        return self._run_with_retry(
            _op, fallback=fallback, is_write=True, raise_on_error=raise_on_error
        )

    def execute_many(
        self,
        query: str,
        seq_of_params: Iterable[Union[Sequence[Any], Dict[str, Any]]],
        fallback: int = 0,
        raise_on_error: bool = False,
    ) -> int:
        """Execute a batch INSERT/UPDATE/DELETE statement atomically and return affected row count."""
        params_list = list(seq_of_params)
        if not params_list:
            return 0

        def _op(conn: sqlite3.Connection) -> int:
            cur = conn.executemany(query, params_list)
            count = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else len(params_list)
            cur.close()
            return count

        return self._run_with_retry(
            _op, fallback=fallback, is_write=True, raise_on_error=raise_on_error
        )

    # ------------------------------------------------------------------
    # Table-Aware Convenience Helpers for System Modules
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_table(table: str) -> str:
        if table not in ALLOWED_TABLES:
            raise ValueError(f"Unsupported DRL table {table!r}. Expected one of {sorted(ALLOWED_TABLES)}.")
        return table

    @staticmethod
    def _validate_identifier(identifier: str) -> str:
        if not identifier or not identifier.replace("_", "").isalnum():
            raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
        return identifier

    def _prepare_payload(self, table: str, data: Dict[str, Any], is_insert: bool = False) -> Dict[str, Any]:
        payload = dict(data)
        json_fields = TABLE_JSON_LIST_FIELDS.get(table, ())
        for field in json_fields:
            if field in payload and not isinstance(payload[field], str):
                payload[field] = encode_json_list(payload[field])

        if table == "events":
            now_str = now_iso_utc7()
            if is_insert:
                payload["created_at"] = to_iso_utc7(payload.get("created_at")) or now_str
            payload["updated_at"] = to_iso_utc7(payload.get("updated_at")) or now_str
            for dt_col in (
                "reg_start_datetime",
                "reg_end_datetime",
                "event_start_datetime",
                "event_end_datetime",
            ):
                if dt_col in payload and payload[dt_col] is not None:
                    payload[dt_col] = to_iso_utc7(payload[dt_col])
            if "is_online" in payload and payload["is_online"] is not None:
                payload["is_online"] = int(bool(payload["is_online"]))

        return payload

    def insert_record(
        self,
        table: str,
        data: Dict[str, Any],
        or_replace: bool = False,
        fallback: Any = None,
        raise_on_error: bool = False,
    ) -> Any:
        """
        Insert a record into one of the DRL tables, handling UTC+7 timestamps and JSON list fields.
        Returns `lastrowid` (or primary key value if provided in `data`), or `fallback` on failure.
        """
        self._validate_table(table)
        payload = self._prepare_payload(table, data, is_insert=True)
        if not payload:
            return fallback

        cols = [self._validate_identifier(k) for k in payload.keys()]
        placeholders = ", ".join(["?"] * len(cols))
        verb = "INSERT OR REPLACE INTO" if or_replace else "INSERT INTO"
        sql = f"{verb} {table} ({', '.join(cols)}) VALUES ({placeholders})"
        values = [payload[c] for c in cols]

        lastrowid = self.execute_write(
            sql,
            values,
            return_lastrowid=True,
            fallback=None,
            raise_on_error=raise_on_error,
        )
        if lastrowid is None:
            return fallback
        if table == "users" and "user_id" in payload:
            return payload["user_id"]
        if table == "rubric" and "school_id" in payload:
            return payload["school_id"]
        if table == "event_subscription" and "user_id" in payload and "event_id" in payload:
            return (payload["user_id"], payload["event_id"])
        return lastrowid

    def update_records(
        self,
        table: str,
        updates: Dict[str, Any],
        where: Dict[str, Any],
        fallback: int = 0,
        raise_on_error: bool = False,
    ) -> int:
        """
        Update matching rows in one of the 5 DRL tables.
        Automatically refreshes `updated_at` on `events` if not explicitly provided.
        """
        self._validate_table(table)
        if not updates or not where:
            return fallback

        payload = self._prepare_payload(table, updates, is_insert=False)
        set_cols = [self._validate_identifier(k) for k in payload.keys()]
        where_cols = [self._validate_identifier(k) for k in where.keys()]

        set_clause = ", ".join(f"{c} = ?" for c in set_cols)
        where_clause = " AND ".join(f"{c} = ?" for c in where_cols)
        sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        params = [payload[c] for c in set_cols] + [where[c] for c in where_cols]

        res = self.execute_write(
            sql,
            params,
            return_lastrowid=False,
            fallback=fallback,
            raise_on_error=raise_on_error,
        )
        return fallback if res is None else res

    def delete_records(
        self,
        table: str,
        where: Dict[str, Any],
        fallback: int = 0,
        raise_on_error: bool = False,
    ) -> int:
        """Delete matching rows in one of the 5 DRL tables and return deleted row count."""
        self._validate_table(table)
        if not where:
            return fallback
        where_cols = [self._validate_identifier(k) for k in where.keys()]
        where_clause = " AND ".join(f"{c} = ?" for c in where_cols)
        sql = f"DELETE FROM {table} WHERE {where_clause}"
        params = [where[c] for c in where_cols]
        res = self.execute_write(
            sql,
            params,
            return_lastrowid=False,
            fallback=fallback,
            raise_on_error=raise_on_error,
        )
        return fallback if res is None else res

    # ------------------------------------------------------------------
    # Introspection & Health Diagnostics
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Verify connection, foreign keys pragma, and existence of all 6 required tables."""
        tables = self.get_table_names()
        fk_enabled = bool(self.fetch_scalar("PRAGMA foreign_keys;", fallback=0))
        missing = sorted(ALLOWED_TABLES - set(tables))
        return {
            "ok": len(missing) == 0 and fk_enabled,
            "db_path": str(self.db_path),
            "foreign_keys": fk_enabled,
            "tables": tables,
            "missing_tables": missing,
            "checked_at": now_iso_utc7(),
        }

    def get_table_names(self) -> List[str]:
        """Return user-defined table names in `drl_database.db`."""
        rows = self.fetch_all(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        )
        return [str(r["name"]) for r in rows]

    def get_table_schema(self, table: str) -> List[Dict[str, Any]]:
        """Return `PRAGMA table_info` column metadata for a given DRL table."""
        self._validate_table(table)
        return self.fetch_all(f"PRAGMA table_info({table});")


def get_drl_db(db_path: Optional[Union[str, Path]] = None) -> DRLDatabaseManager:
    """Module-level convenience accessor for the singleton `DRLDatabaseManager`."""
    return DRLDatabaseManager.get_instance(db_path=db_path)


if __name__ == "__main__":
    db = get_drl_db()
    status = db.health_check()
    print(json.dumps(status, indent=2, ensure_ascii=False))
