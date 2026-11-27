"""
Singleton Database Manager for No-Le-DRL (`drl_database.db` / PostgreSQL) based on SYSTEM_ARCHITECTURE.md.

Supports both:
- **PostgreSQL** (`psycopg` v3) when `DRL_DATABASE_URL` (or `db_url`) uses `postgresql://...` / `postgres://...`
- **SQLite** (`sqlite3`) fallback when using local `drl_database.db` or explicit `db_path` in tests

Tables managed:
1. `events`
2. `users`
3. `event_subscription`
4. `fanpages`
5. `notifications`
6. `rubric`

All timestamps use ISO-8601 format in UTC+7 (Asia/Ho_Chi_Minh).
List-of-values columns are stored as JSON-encoded `TEXT` arrays.
Map coordinates use `"lat,lng"` `TEXT` strings.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union
from urllib.parse import urlparse, urlunparse
from zoneinfo import ZoneInfo

try:
    from config import DRL_DATABASE_URL, DRL_DB_PATH
except ImportError:
    DRL_DB_PATH = Path(__file__).resolve().parent.parent / "drl_database.db"
    DRL_DATABASE_URL = os.environ.get("DRL_DATABASE_URL", "").strip() or f"sqlite:///{DRL_DB_PATH.as_posix()}"

logger = logging.getLogger(__name__)

T = TypeVar("T")

try:
    UTC7_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
except Exception:
    UTC7_TIMEZONE = timezone(timedelta(hours=7))


# JSON-encoded list columns per table for automatic serialization/deserialization
TABLE_JSON_LIST_FIELDS: Dict[str, Tuple[str, ...]] = {
    "events": ("keywords", "target_audience", "topic", "format", "images_url"),
    "users": ("weights", "credits", "preferred_topics", "preferred_formats"),
    "rubric": ("max_credit",),
}

BOOLEAN_FIELDS = frozenset({"is_online", "is_cancelled", "is_read"})

ALLOWED_TABLES = frozenset(
    {"events", "users", "event_subscription", "fanpages", "notifications", "rubric"}
)

TABLE_PRIMARY_KEYS: Dict[str, Tuple[str, ...]] = {
    "rubric": ("school_id",),
    "events": ("event_id",),
    "users": ("user_id",),
    "event_subscription": ("user_id", "event_id"),
    "fanpages": ("fanpage_id",),
    "notifications": ("notification_id",),
}

IDENTITY_PK_TABLES: Dict[str, str] = {
    "rubric": "school_id",
    "events": "event_id",
    "fanpages": "fanpage_id",
    "notifications": "notification_id",
}

# Expected columns for safe in-place schema migration on existing SQLite databases
EXPECTED_NEW_COLUMNS_SQLITE: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "events": (
        ("is_cancelled", "BOOLEAN NOT NULL DEFAULT 0 CHECK (is_cancelled IN (0, 1))"),
        ("coordinate", "TEXT"),
        ("registration_url", "TEXT"),
    ),
    "users": (
        ("school_id", "INTEGER REFERENCES rubric (school_id) ON DELETE SET NULL ON UPDATE CASCADE"),
        ("default_coordinate", "TEXT"),
    ),
    "event_subscription": (
        ("gain", "REAL CHECK (gain IS NULL OR (gain >= 0.0 AND gain <= 1.0))"),
        ("fit", "REAL CHECK (fit IS NULL OR (fit >= 0.0 AND fit <= 1.0))"),
        ("interested", "REAL CHECK (interested IS NULL OR (interested >= 0.0 AND interested <= 1.0))"),
        ("urgency", "REAL CHECK (urgency IS NULL OR (urgency >= 0.0 AND urgency <= 1.0))"),
    ),
    "fanpages": (
        ("facebook_url", "TEXT"),
    ),
    "notifications": (
        ("is_read", "BOOLEAN NOT NULL DEFAULT 0 CHECK (is_read IN (0, 1))"),
        ("created_at", "TEXT"),
    ),
    "rubric": (
        ("school_name", "TEXT"),
    ),
}

# Expected columns for safe in-place schema migration on existing PostgreSQL databases
EXPECTED_NEW_COLUMNS_POSTGRES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "events": (
        ("is_cancelled", "BOOLEAN NOT NULL DEFAULT FALSE"),
        ("coordinate", "TEXT"),
        ("registration_url", "TEXT"),
    ),
    "users": (
        ("school_id", "INTEGER REFERENCES rubric (school_id) ON DELETE SET NULL ON UPDATE CASCADE"),
        ("default_coordinate", "TEXT"),
    ),
    "event_subscription": (
        ("gain", "DOUBLE PRECISION CHECK (gain IS NULL OR (gain >= 0.0 AND gain <= 1.0))"),
        ("fit", "DOUBLE PRECISION CHECK (fit IS NULL OR (fit >= 0.0 AND fit <= 1.0))"),
        ("interested", "DOUBLE PRECISION CHECK (interested IS NULL OR (interested >= 0.0 AND interested <= 1.0))"),
        ("urgency", "DOUBLE PRECISION CHECK (urgency IS NULL OR (urgency >= 0.0 AND urgency <= 1.0))"),
    ),
    "fanpages": (
        ("facebook_url", "TEXT"),
    ),
    "notifications": (
        ("is_read", "BOOLEAN NOT NULL DEFAULT FALSE"),
        ("created_at", "TEXT"),
    ),
    "rubric": (
        ("school_name", "TEXT"),
    ),
}

SCHEMA_SQL_SQLITE = """
CREATE TABLE IF NOT EXISTS rubric (
    school_id INTEGER PRIMARY KEY AUTOINCREMENT,
    school_name TEXT,
    max_credit TEXT NOT NULL DEFAULT '[20, 20, 20, 20, 20]'
);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    keywords TEXT NOT NULL DEFAULT '[]',
    is_online BOOLEAN NOT NULL DEFAULT 0 CHECK (is_online IN (0, 1)),
    is_cancelled BOOLEAN NOT NULL DEFAULT 0 CHECK (is_cancelled IN (0, 1)),
    reg_start_datetime TEXT,
    reg_end_datetime TEXT,
    event_start_datetime TEXT,
    event_end_datetime TEXT,
    location TEXT,
    coordinate TEXT,
    fee TEXT,
    target_audience TEXT NOT NULL DEFAULT '[]',
    topic TEXT NOT NULL DEFAULT '[]',
    format TEXT NOT NULL DEFAULT '[]',
    award TEXT,
    registration_url TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    sources TEXT,
    post_url TEXT,
    images_url TEXT NOT NULL DEFAULT '[]',
    category INTEGER CHECK (category IS NULL OR (category >= 1 AND category <= 5)),
    min_credit INTEGER
);

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username TEXT UNIQUE,
    password TEXT,
    email TEXT,
    school_id INTEGER,
    default_coordinate TEXT,
    start_year INTEGER,
    weights TEXT NOT NULL DEFAULT '[0.25, 0.25, 0.25, 0.25]',
    credits TEXT NOT NULL DEFAULT '[0, 0, 0, 0, 0]',
    preferred_topics TEXT NOT NULL DEFAULT '[]',
    preferred_formats TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (school_id) REFERENCES rubric (school_id) ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS event_subscription (
    user_id TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    gain REAL CHECK (gain IS NULL OR (gain >= 0.0 AND gain <= 1.0)),
    fit REAL CHECK (fit IS NULL OR (fit >= 0.0 AND fit <= 1.0)),
    interested REAL CHECK (interested IS NULL OR (interested >= 0.0 AND interested <= 1.0)),
    urgency REAL CHECK (urgency IS NULL OR (urgency >= 0.0 AND urgency <= 1.0)),
    role TEXT CHECK (role IS NULL OR role IN ('participant', 'host', 'collaborator')),
    status TEXT CHECK (status IS NULL OR status IN ('interested', 'registered', 'in_progress', 'done')),
    credit INTEGER,
    other_awards TEXT,
    PRIMARY KEY (user_id, event_id),
    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (event_id) REFERENCES events (event_id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS fanpages (
    fanpage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fanpage_name TEXT,
    facebook_url TEXT UNIQUE,
    number_of_events INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS notifications (
    notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    type TEXT,
    content TEXT,
    event_id INTEGER,
    is_read BOOLEAN NOT NULL DEFAULT 0 CHECK (is_read IN (0, 1)),
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (event_id) REFERENCES events (event_id) ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_unclassified ON events (category) WHERE is_cancelled = 0;
CREATE INDEX IF NOT EXISTS idx_events_uncredited ON events (min_credit) WHERE is_cancelled = 0;
CREATE INDEX IF NOT EXISTS idx_users_school_id ON users (school_id);
CREATE INDEX IF NOT EXISTS idx_event_subscription_event_id ON event_subscription (event_id);
CREATE INDEX IF NOT EXISTS idx_event_subscription_done_uncredited ON event_subscription (status, credit);
CREATE INDEX IF NOT EXISTS idx_notifications_user_unread ON notifications (user_id, is_read, created_at);
CREATE INDEX IF NOT EXISTS idx_notifications_event_id ON notifications (event_id);
"""

SCHEMA_STATEMENTS_POSTGRES: Tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS rubric (
        school_id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
        school_name TEXT,
        max_credit TEXT NOT NULL DEFAULT '[20, 20, 20, 20, 20]'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        event_id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
        title TEXT,
        description TEXT,
        keywords TEXT NOT NULL DEFAULT '[]',
        is_online BOOLEAN NOT NULL DEFAULT FALSE,
        is_cancelled BOOLEAN NOT NULL DEFAULT FALSE,
        reg_start_datetime TEXT,
        reg_end_datetime TEXT,
        event_start_datetime TEXT,
        event_end_datetime TEXT,
        location TEXT,
        coordinate TEXT,
        fee TEXT,
        target_audience TEXT NOT NULL DEFAULT '[]',
        topic TEXT NOT NULL DEFAULT '[]',
        format TEXT NOT NULL DEFAULT '[]',
        award TEXT,
        registration_url TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        sources TEXT,
        post_url TEXT,
        images_url TEXT NOT NULL DEFAULT '[]',
        category INTEGER CHECK (category IS NULL OR (category >= 1 AND category <= 5)),
        min_credit INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        username TEXT UNIQUE,
        password TEXT,
        email TEXT,
        school_id INTEGER REFERENCES rubric (school_id) ON DELETE SET NULL ON UPDATE CASCADE,
        default_coordinate TEXT,
        start_year INTEGER,
        weights TEXT NOT NULL DEFAULT '[0.25, 0.25, 0.25, 0.25]',
        credits TEXT NOT NULL DEFAULT '[0, 0, 0, 0, 0]',
        preferred_topics TEXT NOT NULL DEFAULT '[]',
        preferred_formats TEXT NOT NULL DEFAULT '[]'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS event_subscription (
        user_id TEXT NOT NULL REFERENCES users (user_id) ON DELETE CASCADE ON UPDATE CASCADE,
        event_id INTEGER NOT NULL REFERENCES events (event_id) ON DELETE CASCADE ON UPDATE CASCADE,
        gain DOUBLE PRECISION CHECK (gain IS NULL OR (gain >= 0.0 AND gain <= 1.0)),
        fit DOUBLE PRECISION CHECK (fit IS NULL OR (fit >= 0.0 AND fit <= 1.0)),
        interested DOUBLE PRECISION CHECK (interested IS NULL OR (interested >= 0.0 AND interested <= 1.0)),
        urgency DOUBLE PRECISION CHECK (urgency IS NULL OR (urgency >= 0.0 AND urgency <= 1.0)),
        role TEXT CHECK (role IS NULL OR role IN ('participant', 'host', 'collaborator')),
        status TEXT CHECK (status IS NULL OR status IN ('interested', 'registered', 'in_progress', 'done')),
        credit INTEGER,
        other_awards TEXT,
        PRIMARY KEY (user_id, event_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fanpages (
        fanpage_id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
        fanpage_name TEXT,
        facebook_url TEXT UNIQUE,
        number_of_events INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS notifications (
        notification_id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users (user_id) ON DELETE CASCADE ON UPDATE CASCADE,
        type TEXT,
        content TEXT,
        event_id INTEGER REFERENCES events (event_id) ON DELETE SET NULL ON UPDATE CASCADE,
        is_read BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_unclassified ON events (category) WHERE is_cancelled = FALSE",
    "CREATE INDEX IF NOT EXISTS idx_events_uncredited ON events (min_credit) WHERE is_cancelled = FALSE",
    "CREATE INDEX IF NOT EXISTS idx_users_school_id ON users (school_id)",
    "CREATE INDEX IF NOT EXISTS idx_event_subscription_event_id ON event_subscription (event_id)",
    "CREATE INDEX IF NOT EXISTS idx_event_subscription_done_uncredited ON event_subscription (status, credit)",
    "CREATE INDEX IF NOT EXISTS idx_notifications_user_unread ON notifications (user_id, is_read, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_notifications_event_id ON notifications (event_id)",
)

# Backwards-compatible alias
SCHEMA_SQL = SCHEMA_SQL_SQLITE

_INSERT_TABLE_RE = re.compile(r"^\s*INSERT\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", re.IGNORECASE)


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


def _is_postgres_url(url: str) -> bool:
    low = (url or "").strip().lower()
    return low.startswith(("postgresql://", "postgres://", "postgresql+psycopg://", "postgresql+psycopg2://"))


def _normalize_postgres_dsn(url: str) -> str:
    cleaned = url.strip()
    for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://", "postgres://"):
        if cleaned.lower().startswith(prefix):
            cleaned = "postgresql://" + cleaned[len(prefix):]
            break
    return cleaned


def _mask_dsn(dsn: str) -> str:
    """Return a safe display string for logs/health_check with password redacted."""
    try:
        parsed = urlparse(dsn)
        if parsed.password:
            user = parsed.username or ""
            host = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            netloc = f"{user}:***@{host}{port}"
            return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
    except Exception:
        pass
    return dsn


def _adapt_sql_for_postgres(query: str, has_params: bool = True) -> str:
    """
    Translate SQLite-style `?` positional placeholders (and `:name` named placeholders)
    to `psycopg` `%s` / `%(name)s` placeholders without touching quoted string literals or comments.
    Also escapes literal `%` characters when parameters are bound so psycopg doesn't misread them.
    """
    if not has_params:
        return query

    out: List[str] = []
    n = len(query)
    i = 0
    in_single = False
    in_double = False

    while i < n:
        ch = query[i]

        if in_single:
            out.append(ch)
            if ch == "'":
                if i + 1 < n and query[i + 1] == "'":
                    out.append("'")
                    i += 2
                    continue
                in_single = False
            elif ch == "%":
                out.append("%")
            i += 1
            continue

        if in_double:
            out.append(ch)
            if ch == '"':
                if i + 1 < n and query[i + 1] == '"':
                    out.append('"')
                    i += 2
                    continue
                in_double = False
            elif ch == "%":
                out.append("%")
            i += 1
            continue

        # Line comment `-- ...`
        if ch == "-" and i + 1 < n and query[i + 1] == "-":
            nl = query.find("\n", i)
            segment = query[i:] if nl == -1 else query[i:nl]
            out.append(segment.replace("%", "%%"))
            if nl == -1:
                break
            i = nl
            continue

        # Block comment `/* ... */`
        if ch == "/" and i + 1 < n and query[i + 1] == "*":
            end = query.find("*/", i + 2)
            segment = query[i:] if end == -1 else query[i : end + 2]
            out.append(segment.replace("%", "%%"))
            if end == -1:
                break
            i = end + 2
            continue

        if ch == "'":
            in_single = True
            out.append(ch)
            i += 1
            continue

        if ch == '"':
            in_double = True
            out.append(ch)
            i += 1
            continue

        if ch == "?":
            out.append("%s")
            i += 1
            continue

        # Named parameter `:param_name` (avoid PostgreSQL `::type` cast)
        if ch == ":":
            if i + 1 < n and query[i + 1] == ":":
                out.append("::")
                i += 2
                continue
            if i + 1 < n and (query[i + 1].isalpha() or query[i + 1] == "_"):
                j = i + 2
                while j < n and (query[j].isalnum() or query[j] == "_"):
                    j += 1
                param_name = query[i + 1 : j]
                out.append(f"%({param_name})s")
                i = j
                continue

        if ch == "%":
            # Preserve existing `%s` or `%(name)s` if caller already wrote psycopg syntax
            if i + 1 < n and (query[i + 1] == "s" or query[i + 1] == "("):
                out.append("%")
            else:
                out.append("%%")
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


class _PostgresConnectionProxy:
    """
    Thin wrapper around `psycopg.Connection` yielded inside `with db.transaction() as conn:`
    so existing callers using `?` placeholders (`conn.execute("... ? ...", params)`) work seamlessly.
    """

    def __init__(self, raw_conn: Any) -> None:
        self._conn = raw_conn

    def execute(self, query: str, params: Optional[Union[Sequence[Any], Dict[str, Any]]] = None) -> Any:
        has_params = params is not None and len(params) > 0 if hasattr(params, "__len__") else params is not None
        adapted = _adapt_sql_for_postgres(query, has_params=has_params)
        if has_params:
            return self._conn.execute(adapted, params)
        return self._conn.execute(query)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class DRLDatabaseManager:
    """
    Thread-safe Singleton Database Manager for No-Le-DRL (`PostgreSQL` or `SQLite`).

    Provides:
    - Automatic dialect selection (`postgresql` via `DRL_DATABASE_URL` or `sqlite` via `db_path`)
    - Thread-local persistent connections with auto-reconnect and retry on transient lock/network errors
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
        db_url: Optional[str] = None,
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
        db_url: Optional[str] = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            return
        with self._instance_lock:
            if getattr(self, "_initialized", False):
                return

            resolved_url, resolved_path, dialect = self._resolve_target(db_path=db_path, db_url=db_url)
            self.dialect: str = dialect
            self.is_postgres: bool = dialect == "postgresql"
            self.is_sqlite: bool = dialect == "sqlite"
            self.db_url: str = resolved_url
            self.db_path: Path = resolved_path
            self.target_label: str = _mask_dsn(resolved_url) if self.is_postgres else str(resolved_path)

            self.busy_timeout_ms: int = max(1000, int(busy_timeout_ms))
            self.max_retries: int = max(1, int(max_retries))
            self._local = threading.local()
            self._schema_ready = False
            self.initialize_schema()
            atexit.register(self.close_connection)
            self._initialized = True

    @staticmethod
    def _resolve_target(
        db_path: Optional[Union[str, Path]],
        db_url: Optional[str],
    ) -> Tuple[str, Path, str]:
        """Resolve `(db_url, db_path, dialect)` from explicit arguments or environment configuration."""
        if db_url and db_url.strip():
            raw = db_url.strip()
            if _is_postgres_url(raw):
                return _normalize_postgres_dsn(raw), Path(DRL_DB_PATH), "postgresql"
            if raw.lower().startswith("sqlite:///"):
                p = Path(raw[len("sqlite:///"):])
                return raw, p, "sqlite"
            p = Path(raw)
            return f"sqlite:///{p.as_posix()}", p, "sqlite"

        if db_path is not None:
            if isinstance(db_path, str) and _is_postgres_url(db_path):
                return _normalize_postgres_dsn(db_path), Path(DRL_DB_PATH), "postgresql"
            if isinstance(db_path, str) and db_path.strip().lower().startswith("sqlite:///"):
                p = Path(db_path.strip()[len("sqlite:///"):])
                return db_path.strip(), p, "sqlite"
            p = Path(db_path)
            return f"sqlite:///{p.as_posix()}", p, "sqlite"

        env_url = os.environ.get("DRL_DATABASE_URL", "").strip() or str(DRL_DATABASE_URL or "").strip()
        if env_url and _is_postgres_url(env_url):
            return _normalize_postgres_dsn(env_url), Path(DRL_DB_PATH), "postgresql"
        if env_url.lower().startswith("sqlite:///"):
            p = Path(env_url[len("sqlite:///"):])
            return env_url, p, "sqlite"

        default_path = Path(DRL_DB_PATH)
        return f"sqlite:///{default_path.as_posix()}", default_path, "sqlite"

    @classmethod
    def get_instance(
        cls,
        db_path: Optional[Union[str, Path]] = None,
        db_url: Optional[str] = None,
    ) -> "DRLDatabaseManager":
        """Return the singleton DRLDatabaseManager instance."""
        return cls(db_path=db_path, db_url=db_url)

    @classmethod
    def reset_instance(cls) -> None:
        """Close active thread connection and reset singleton instance (primarily for testing)."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.close_connection()
                cls._instance = None

    @staticmethod
    def _import_psycopg() -> Tuple[Any, Any]:
        try:
            import psycopg
            from psycopg.rows import dict_row
            return psycopg, dict_row
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL URL configured for DRLDatabaseManager, but `psycopg` is not installed. "
                "Run: pip install \"psycopg[binary]>=3.2.0\""
            ) from exc

    def _create_connection(self) -> Any:
        if self.is_postgres:
            psycopg, dict_row = self._import_psycopg()
            timeout_sec = max(5, int(round(self.busy_timeout_ms / 1000.0)))
            conn = psycopg.connect(
                self.db_url,
                connect_timeout=timeout_sec,
                row_factory=dict_row,
                autocommit=False,
            )
            return conn

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

    def _is_connection_usable(self, conn: Optional[Any]) -> bool:
        if conn is None:
            return False
        try:
            if self.is_postgres:
                if getattr(conn, "closed", False) or getattr(conn, "broken", False):
                    return False
                # Clear aborted transaction state before health probe if idle in error
                info = getattr(conn, "info", None)
                if info is not None and getattr(info, "transaction_status", 0) == 3:  # INERROR
                    conn.rollback()
            conn.execute("SELECT 1;")
            if self.is_postgres:
                info = getattr(conn, "info", None)
                if info is not None and getattr(info, "transaction_status", 0) == 2:  # INTRANS from SELECT 1
                    conn.rollback()
            return True
        except Exception:
            return False

    def get_connection(self) -> Any:
        """Return a healthy thread-local database connection, reconnecting automatically if needed."""
        conn: Optional[Any] = getattr(self._local, "conn", None)
        if not self._is_connection_usable(conn):
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            conn = self._create_connection()
            self._local.conn = conn
        elif self.is_sqlite:
            conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def close_connection(self) -> None:
        """Close the current thread's database connection if open."""
        conn: Optional[Any] = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception as exc:
                logger.debug("Error while closing %s connection: %s", self.dialect, exc)
            finally:
                self._local.conn = None

    def checkpoint(self) -> None:
        """Run a WAL checkpoint on SQLite; safe no-op on PostgreSQL."""
        if self.is_sqlite:
            with self.transaction() as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        """
        Context manager for atomic read/write transactions.
        Commits on success, rolls back cleanly on any exception.
        """
        conn = self.get_connection()
        exposed_conn = _PostgresConnectionProxy(conn) if self.is_postgres else conn
        try:
            yield exposed_conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception as rb_exc:
                logger.warning("Failed to roll back transaction cleanly: %s", rb_exc)
                self.close_connection()
            raise

    def _execute_cursor(
        self,
        conn: Any,
        query: str,
        params: Union[Sequence[Any], Dict[str, Any]] = (),
    ) -> Any:
        has_params = bool(params) if isinstance(params, (list, tuple, dict)) else bool(params)
        if self.is_postgres:
            sql = _adapt_sql_for_postgres(query, has_params=has_params)
            return conn.execute(sql, params) if has_params else conn.execute(query)
        return conn.execute(query, params)

    def _run_with_retry(
        self,
        operation: Callable[[Any], T],
        fallback: T,
        is_write: bool = False,
        raise_on_error: bool = False,
    ) -> T:
        """
        Execute a database operation with automatic retry on transient lock/connection errors,
        transaction rollback on failure, and safe fallback value return.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                conn = self.get_connection()
                result = operation(conn)
                if is_write:
                    conn.commit()
                elif self.is_postgres:
                    # End implicit read-only transaction cleanly so connection doesn't sit 'idle in transaction'
                    info = getattr(conn, "info", None)
                    if info is not None and getattr(info, "transaction_status", 0) == 2:
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
                if not is_locked:
                    self.close_connection()
                break
            except Exception as exc:
                last_error = exc
                self._safe_rollback()
                if self.is_postgres:
                    msg = str(exc).lower()
                    is_transient = any(
                        tok in msg
                        for tok in (
                            "deadlock detected",
                            "could not serialize",
                            "connection is closed",
                            "server closed the connection",
                            "terminating connection",
                            "connection timeout",
                            "SSL SYSCALL error".lower(),
                        )
                    )
                    if is_transient and attempt < self.max_retries:
                        self.close_connection()
                        backoff = 0.1 * (2 ** (attempt - 1))
                        logger.warning(
                            "PostgreSQL transient error (attempt %d/%d), retrying in %.2fs: %s",
                            attempt,
                            self.max_retries,
                            backoff,
                            exc,
                        )
                        time.sleep(backoff)
                        continue
                break

        logger.error("Database operation failed on %s: %s", self.target_label, last_error)
        if raise_on_error and last_error is not None:
            raise last_error
        return fallback

    def _safe_rollback(self) -> None:
        conn: Optional[Any] = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                self.close_connection()

    def initialize_schema(self) -> bool:
        """Create the 6 DRL architecture tables, migrate missing columns, and create indexes."""
        if self.is_postgres:
            def _init_pg(conn: Any) -> bool:
                existing_tables = {
                    row["table_name"]
                    for row in conn.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = current_schema() AND table_type = 'BASE TABLE';"
                    ).fetchall()
                }
                for table, cols in EXPECTED_NEW_COLUMNS_POSTGRES.items():
                    if table in existing_tables:
                        existing_cols = {
                            row["column_name"]
                            for row in conn.execute(
                                "SELECT column_name FROM information_schema.columns "
                                "WHERE table_schema = current_schema() AND table_name = %s;",
                                (table,),
                            ).fetchall()
                        }
                        for col_name, col_def in cols:
                            if col_name not in existing_cols:
                                conn.execute(
                                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col_name} {col_def};"
                                )
                for stmt in SCHEMA_STATEMENTS_POSTGRES:
                    conn.execute(stmt)
                self._schema_ready = True
                return True

            return self._run_with_retry(_init_pg, fallback=False, is_write=True, raise_on_error=True)

        def _init_sqlite(conn: sqlite3.Connection) -> bool:
            existing_tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%';"
                ).fetchall()
            }
            for table, cols in EXPECTED_NEW_COLUMNS_SQLITE.items():
                if table in existing_tables:
                    existing_cols = {
                        row["name"] for row in conn.execute(f"PRAGMA table_info({table});").fetchall()
                    }
                    for col_name, col_def in cols:
                        if col_name not in existing_cols:
                            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def};")
            conn.executescript(SCHEMA_SQL_SQLITE)
            conn.execute("PRAGMA foreign_keys = ON;")
            self._schema_ready = True
            return True

        return self._run_with_retry(_init_sqlite, fallback=False, is_write=True, raise_on_error=True)

    def sync_identity_sequences(self, tables: Optional[Sequence[str]] = None) -> None:
        """
        Synchronize PostgreSQL IDENTITY sequences to `MAX(pk)` after inserting rows with explicit IDs.
        Safe no-op on SQLite.
        """
        if not self.is_postgres:
            return
        target_tables = list(tables) if tables is not None else list(IDENTITY_PK_TABLES.keys())

        def _sync(conn: Any) -> None:
            for tbl in target_tables:
                pk_col = IDENTITY_PK_TABLES.get(tbl)
                if not pk_col:
                    continue
                conn.execute(
                    f"""
                    SELECT setval(
                        pg_get_serial_sequence(%s, %s),
                        GREATEST(COALESCE((SELECT MAX({pk_col}) FROM {tbl}), 1), 1),
                        (SELECT COUNT(*) > 0 FROM {tbl})
                    );
                    """,
                    (tbl, pk_col),
                )

        self._run_with_retry(_sync, fallback=None, is_write=True, raise_on_error=False)

    @classmethod
    def _resolve_json_fields(
        cls,
        table: Optional[str] = None,
        json_list_fields: Sequence[str] = (),
    ) -> Tuple[str, ...]:
        if json_list_fields:
            return tuple(json_list_fields)
        if table:
            return TABLE_JSON_LIST_FIELDS.get(table, ())
        return ()

    @staticmethod
    def _row_to_dict(
        row: Any,
        json_list_fields: Sequence[str] = (),
    ) -> Dict[str, Any]:
        data = dict(row)
        for field in json_list_fields:
            if field in data:
                data[field] = decode_json_list(data[field])
        for bfield in BOOLEAN_FIELDS:
            if bfield in data and data[bfield] is not None:
                data[bfield] = bool(data[bfield])
        return data

    @staticmethod
    def _first_column_value(row: Any) -> Any:
        if isinstance(row, dict):
            return next(iter(row.values())) if row else None
        return row[0]

    # ------------------------------------------------------------------
    # Core Query & Execution Methods (with Safe Error Fallbacks)
    # ------------------------------------------------------------------

    def fetch_one(
        self,
        query: str,
        params: Union[Sequence[Any], Dict[str, Any]] = (),
        fallback: Optional[Dict[str, Any]] = None,
        json_list_fields: Sequence[str] = (),
        table: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Execute a SELECT query and return the first row as a dict, or `fallback`."""
        fields = self._resolve_json_fields(table=table, json_list_fields=json_list_fields)

        def _op(conn: Any) -> Optional[Dict[str, Any]]:
            cur = self._execute_cursor(conn, query, params)
            row = cur.fetchone()
            cur.close()
            if row is None:
                return fallback
            return self._row_to_dict(row, json_list_fields=fields)

        return self._run_with_retry(
            _op, fallback=fallback, is_write=False, raise_on_error=raise_on_error
        )

    def fetch_all(
        self,
        query: str,
        params: Union[Sequence[Any], Dict[str, Any]] = (),
        fallback: Optional[List[Dict[str, Any]]] = None,
        json_list_fields: Sequence[str] = (),
        table: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return all rows as a list of dicts, or `fallback` (defaults to `[]`)."""
        safe_fallback: List[Dict[str, Any]] = [] if fallback is None else fallback
        fields = self._resolve_json_fields(table=table, json_list_fields=json_list_fields)

        def _op(conn: Any) -> List[Dict[str, Any]]:
            cur = self._execute_cursor(conn, query, params)
            rows = cur.fetchall()
            cur.close()
            return [self._row_to_dict(r, json_list_fields=fields) for r in rows]

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
        def _op(conn: Any) -> Any:
            cur = self._execute_cursor(conn, query, params)
            row = cur.fetchone()
            cur.close()
            return self._first_column_value(row) if row is not None else fallback

        return self._run_with_retry(
            _op, fallback=fallback, is_write=False, raise_on_error=raise_on_error
        )

    def execute_write(
        self,
        query: str,
        params: Union[Sequence[Any], Dict[str, Any]] = (),
        return_lastrowid: bool = False,
        fallback: Optional[Any] = None,
        raise_on_error: bool = False,
    ) -> Optional[Any]:
        """
        Execute an INSERT/UPDATE/DELETE statement atomically.
        Returns generated primary key / `lastrowid` when `return_lastrowid=True`,
        otherwise `cursor.rowcount`, or `fallback` on failure.
        """
        def _op(conn: Any) -> Optional[Any]:
            effective_query = query
            if self.is_postgres and return_lastrowid and "returning " not in query.lower():
                m = _INSERT_TABLE_RE.match(query)
                if m:
                    tbl = m.group(1).lower()
                    pk_col = IDENTITY_PK_TABLES.get(tbl)
                    if pk_col:
                        effective_query = query.rstrip().rstrip(";") + f" RETURNING {pk_col}"

            cur = self._execute_cursor(conn, effective_query, params)
            if return_lastrowid:
                if self.is_postgres:
                    if getattr(cur, "description", None):
                        row = cur.fetchone()
                        out = self._first_column_value(row) if row is not None else cur.rowcount
                    else:
                        out = cur.rowcount
                else:
                    out = cur.lastrowid
            else:
                out = cur.rowcount
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

        def _op(conn: Any) -> int:
            effective_query = _adapt_sql_for_postgres(query, has_params=True) if self.is_postgres else query
            cur = conn.cursor()
            cur.executemany(effective_query, params_list)
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

        for bfield in BOOLEAN_FIELDS:
            if bfield in payload and payload[bfield] is not None:
                payload[bfield] = bool(payload[bfield]) if self.is_postgres else int(bool(payload[bfield]))

        now_str = now_iso_utc7()
        if table == "events":
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
        elif table == "notifications":
            if is_insert:
                payload["created_at"] = to_iso_utc7(payload.get("created_at")) or now_str
            elif "created_at" in payload and payload["created_at"] is not None:
                payload["created_at"] = to_iso_utc7(payload["created_at"])

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
        Returns primary key value (or tuple for `event_subscription`), or `fallback` on failure.
        """
        self._validate_table(table)
        payload = self._prepare_payload(table, data, is_insert=True)
        if not payload:
            return fallback

        cols = [self._validate_identifier(k) for k in payload.keys()]
        values = [payload[c] for c in cols]

        if self.is_postgres:
            pk_cols = TABLE_PRIMARY_KEYS[table]
            placeholders = ", ".join(["%s"] * len(cols))
            conflict_sql = ""
            if or_replace:
                non_pk_cols = [c for c in cols if c not in pk_cols]
                if non_pk_cols:
                    updates_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_pk_cols)
                    conflict_sql = f" ON CONFLICT ({', '.join(pk_cols)}) DO UPDATE SET {updates_clause}"
                else:
                    conflict_sql = f" ON CONFLICT ({', '.join(pk_cols)}) DO NOTHING"

            returning_cols = ", ".join(pk_cols)
            sql = (
                f"INSERT INTO {table} ({', '.join(cols)}) "
                f"VALUES ({placeholders}){conflict_sql} RETURNING {returning_cols}"
            )

            def _pg_insert(conn: Any) -> Any:
                cur = conn.execute(sql, values)
                row = cur.fetchone()
                cur.close()
                ident_pk = IDENTITY_PK_TABLES.get(table)
                if ident_pk and ident_pk in payload and payload[ident_pk] is not None:
                    conn.execute(
                        f"""
                        SELECT setval(
                            pg_get_serial_sequence(%s, %s),
                            GREATEST(COALESCE((SELECT MAX({ident_pk}) FROM {table}), 1), 1),
                            (SELECT COUNT(*) > 0 FROM {table})
                        );
                        """,
                        (table, ident_pk),
                    )
                if row is None:
                    if len(pk_cols) == 1 and pk_cols[0] in payload:
                        return payload[pk_cols[0]]
                    if len(pk_cols) == 2 and all(k in payload for k in pk_cols):
                        return (payload[pk_cols[0]], payload[pk_cols[1]])
                    return None
                if len(pk_cols) == 1:
                    return row[pk_cols[0]]
                return tuple(row[k] for k in pk_cols)

            return self._run_with_retry(
                _pg_insert, fallback=fallback, is_write=True, raise_on_error=raise_on_error
            )

        placeholders = ", ".join(["?"] * len(cols))
        verb = "INSERT OR REPLACE INTO" if or_replace else "INSERT INTO"
        sql = f"{verb} {table} ({', '.join(cols)}) VALUES ({placeholders})"

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
        Update matching rows in one of the DRL tables.
        Automatically refreshes `updated_at` on `events` if not explicitly provided.
        """
        self._validate_table(table)
        if not updates or not where:
            return fallback

        payload = self._prepare_payload(table, updates, is_insert=False)
        set_cols = [self._validate_identifier(k) for k in payload.keys()]
        where_cols = [self._validate_identifier(k) for k in where.keys()]
        ph = "%s" if self.is_postgres else "?"

        set_clause = ", ".join(f"{c} = {ph}" for c in set_cols)
        where_clause = " AND ".join(f"{c} = {ph}" for c in where_cols)
        sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        where_vals = [
            (bool(where[c]) if self.is_postgres else int(bool(where[c])))
            if c in BOOLEAN_FIELDS and where[c] is not None
            else where[c]
            for c in where_cols
        ]
        params = [payload[c] for c in set_cols] + where_vals

        res = self.execute_write(
            sql,
            params,
            return_lastrowid=False,
            fallback=fallback,
            raise_on_error=raise_on_error,
        )
        return fallback if res is None else int(res)

    def delete_records(
        self,
        table: str,
        where: Dict[str, Any],
        fallback: int = 0,
        raise_on_error: bool = False,
    ) -> int:
        """Delete matching rows in one of the DRL tables and return deleted row count."""
        self._validate_table(table)
        if not where:
            return fallback
        where_cols = [self._validate_identifier(k) for k in where.keys()]
        ph = "%s" if self.is_postgres else "?"
        where_clause = " AND ".join(f"{c} = {ph}" for c in where_cols)
        sql = f"DELETE FROM {table} WHERE {where_clause}"
        params = [
            (bool(where[c]) if self.is_postgres else int(bool(where[c])))
            if c in BOOLEAN_FIELDS and where[c] is not None
            else where[c]
            for c in where_cols
        ]
        res = self.execute_write(
            sql,
            params,
            return_lastrowid=False,
            fallback=fallback,
            raise_on_error=raise_on_error,
        )
        return fallback if res is None else int(res)

    # ------------------------------------------------------------------
    # Introspection & Health Diagnostics
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Verify connection, foreign key enforcement, and existence of all 6 required tables."""
        tables = self.get_table_names()
        if self.is_postgres:
            fk_enabled = bool(self.fetch_scalar("SELECT 1;", fallback=0))
        else:
            fk_enabled = bool(self.fetch_scalar("PRAGMA foreign_keys;", fallback=0))
        missing = sorted(ALLOWED_TABLES - set(tables))
        return {
            "ok": len(missing) == 0 and fk_enabled,
            "dialect": self.dialect,
            "db_target": self.target_label,
            "db_path": str(self.db_path),
            "foreign_keys": fk_enabled,
            "tables": tables,
            "missing_tables": missing,
            "checked_at": now_iso_utc7(),
        }

    def get_table_names(self) -> List[str]:
        """Return user-defined table names in the DRL database."""
        if self.is_postgres:
            rows = self.fetch_all(
                "SELECT table_name AS name FROM information_schema.tables "
                "WHERE table_schema = current_schema() AND table_type = 'BASE TABLE' "
                "ORDER BY table_name;"
            )
            return [str(r["name"]) for r in rows if r["name"] in ALLOWED_TABLES or not str(r["name"]).startswith("pg_")]

        rows = self.fetch_all(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        )
        return [str(r["name"]) for r in rows]

    def get_table_schema(self, table: str) -> List[Dict[str, Any]]:
        """
        Return column metadata for a given DRL table in `PRAGMA table_info`-compatible format
        (`cid`, `name`, `type`, `notnull`, `dflt_value`, `pk`).
        """
        self._validate_table(table)
        if self.is_postgres:
            pk_cols = set(TABLE_PRIMARY_KEYS.get(table, ()))
            rows = self.fetch_all(
                """
                SELECT
                    ordinal_position - 1 AS cid,
                    column_name AS name,
                    UPPER(data_type) AS type,
                    CASE WHEN is_nullable = 'NO' THEN 1 ELSE 0 END AS notnull,
                    column_default AS dflt_value
                FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = %s
                ORDER BY ordinal_position ASC;
                """,
                (table,),
            )
            for idx, r in enumerate(rows):
                r["pk"] = (idx + 1) if r["name"] in pk_cols else 0
            return rows

        return self.fetch_all(f"PRAGMA table_info({table});")


def get_drl_db(
    db_path: Optional[Union[str, Path]] = None,
    db_url: Optional[str] = None,
) -> DRLDatabaseManager:
    """Module-level convenience accessor for the singleton `DRLDatabaseManager`."""
    return DRLDatabaseManager.get_instance(db_path=db_path, db_url=db_url)


if __name__ == "__main__":
    db = get_drl_db()
    status = db.health_check()
    print(json.dumps(status, indent=2, ensure_ascii=False))
