"""
Idempotent migration utility to copy all 6 DRL tables from `drl_database.db` (SQLite)
into a target PostgreSQL database (`DRL_DATABASE_URL` or `--pg-url`).

Usage:
    # Preview source SQLite counts without connecting to PostgreSQL:
    python -m database.migrate_sqlite_to_postgres --dry-run

    # Migrate using DRL_DATABASE_URL from .env:
    python -m database.migrate_sqlite_to_postgres

    # Or pass an explicit PostgreSQL URL:
    python -m database.migrate_sqlite_to_postgres --pg-url "postgresql://postgres:password@127.0.0.1:5432/nole_drl"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DRL_DATABASE_URL, DRL_DB_PATH  # noqa: E402
from database.drl_db_manager import (  # noqa: E402
    DRLDatabaseManager,
    TABLE_PRIMARY_KEYS,
    _is_postgres_url,
)

# Foreign-key safe migration order
MIGRATION_ORDER: Tuple[str, ...] = (
    "rubric",
    "fanpages",
    "users",
    "events",
    "event_subscription",
    "notifications",
)


def migrate_sqlite_to_postgres(
    sqlite_path: Path,
    pg_url: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"Source SQLite database not found: {sqlite_path}")

    DRLDatabaseManager.reset_instance()
    src_db = DRLDatabaseManager(db_path=sqlite_path)

    source_rows = {}
    for table in MIGRATION_ORDER:
        pk_order = ", ".join(TABLE_PRIMARY_KEYS[table])
        rows = src_db.fetch_all(
            f"SELECT * FROM {table} ORDER BY {pk_order} ASC;",
            table=table,
            raise_on_error=True,
        )
        source_rows[table] = rows

    src_db.close_connection()
    DRLDatabaseManager.reset_instance()

    source_counts = {tbl: len(rows) for tbl, rows in source_rows.items()}
    if dry_run:
        return {
            "dry_run": True,
            "source_sqlite": str(sqlite_path),
            "source_counts": source_counts,
        }

    target_url = (pg_url or os.environ.get("DRL_DATABASE_URL") or DRL_DATABASE_URL or "").strip()
    if not _is_postgres_url(target_url):
        raise ValueError(
            "Target PostgreSQL URL is not set or not a `postgresql://` URL. "
            "Set `DRL_DATABASE_URL=postgresql://user:pass@host:5432/dbname` in `.env` "
            "or pass `--pg-url`."
        )

    dst_db = DRLDatabaseManager(db_url=target_url)
    migrated_counts: Dict[str, int] = {}
    target_counts: Dict[str, int] = {}

    try:
        for table in MIGRATION_ORDER:
            rows = source_rows[table]
            count = 0
            for row in rows:
                dst_db.insert_record(table, row, or_replace=True, raise_on_error=True)
                count += 1
            migrated_counts[table] = count

        dst_db.sync_identity_sequences()

        for table in MIGRATION_ORDER:
            target_counts[table] = int(
                dst_db.fetch_scalar(f"SELECT COUNT(*) FROM {table};", fallback=0, raise_on_error=True)
            )
        target_label = dst_db.target_label
    finally:
        dst_db.close_connection()
        DRLDatabaseManager.reset_instance()

    return {
        "dry_run": False,
        "source_sqlite": str(sqlite_path),
        "target_postgres": target_label,
        "source_counts": source_counts,
        "migrated_counts": migrated_counts,
        "target_counts_after": target_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate all 6 DRL tables from local SQLite (`drl_database.db`) to PostgreSQL."
    )
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=Path(DRL_DB_PATH),
        help="Path to source SQLite database file (default: drl_database.db).",
    )
    parser.add_argument(
        "--pg-url",
        type=str,
        default=None,
        help="Target PostgreSQL connection URL (defaults to DRL_DATABASE_URL in .env).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect source SQLite row counts without connecting to PostgreSQL.",
    )
    args = parser.parse_args()

    summary = migrate_sqlite_to_postgres(
        sqlite_path=args.sqlite,
        pg_url=args.pg_url,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
