"""Unit tests for PostgreSQL dialect resolution, SQL placeholder adaptation, and DRLDatabaseManager."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from database.drl_db_manager import (
    DRLDatabaseManager,
    _PostgresConnectionProxy,
    _adapt_sql_for_postgres,
    _is_postgres_url,
    _mask_dsn,
    _normalize_postgres_dsn,
)


class TestPostgresDialectHelpers(unittest.TestCase):
    def test_url_detection_and_normalization(self) -> None:
        self.assertTrue(_is_postgres_url("postgresql://u:p@localhost:5432/nole_drl"))
        self.assertTrue(_is_postgres_url("postgres://u:p@localhost:5432/nole_drl"))
        self.assertTrue(_is_postgres_url("postgresql+psycopg://u:p@localhost:5432/nole_drl"))
        self.assertFalse(_is_postgres_url("sqlite:///drl_database.db"))

        self.assertEqual(
            _normalize_postgres_dsn("postgresql+psycopg://u:p@127.0.0.1:5432/nole_drl"),
            "postgresql://u:p@127.0.0.1:5432/nole_drl",
        )
        self.assertEqual(
            _normalize_postgres_dsn("postgres://u:p@127.0.0.1:5432/nole_drl"),
            "postgresql://u:p@127.0.0.1:5432/nole_drl",
        )
        self.assertEqual(
            _mask_dsn("postgresql://drl_user:supersecret@127.0.0.1:5432/nole_drl"),
            "postgresql://drl_user:***@127.0.0.1:5432/nole_drl",
        )

    def test_sql_placeholder_adaptation_preserves_quotes_and_casts(self) -> None:
        sql = (
            "SELECT * FROM events "
            "WHERE title LIKE '%AI?%' AND category = ? AND sources::text = :src_name -- ignore ? here\n"
            "AND status = 'done';"
        )
        adapted = _adapt_sql_for_postgres(sql, has_params=True)
        self.assertEqual(
            adapted,
            "SELECT * FROM events "
            "WHERE title LIKE '%%AI?%%' AND category = %s AND sources::text = %(src_name)s -- ignore ? here\n"
            "AND status = 'done';",
        )

    def test_resolve_target_keeps_explicit_db_path_isolated_from_env(self) -> None:
        with patch.dict(os.environ, {"DRL_DATABASE_URL": "postgresql://u:p@localhost:5432/nole_drl"}):
            # When no args passed -> uses env PostgreSQL URL
            url, _, dialect = DRLDatabaseManager._resolve_target(db_path=None, db_url=None)
            self.assertEqual(dialect, "postgresql")
            self.assertEqual(url, "postgresql://u:p@localhost:5432/nole_drl")

            # When explicit SQLite db_path passed (e.g. unit tests) -> stays isolated on SQLite
            tmp_file = Path(tempfile.gettempdir()) / "isolated_test.db"
            _, resolved_path, dialect2 = DRLDatabaseManager._resolve_target(db_path=tmp_file, db_url=None)
            self.assertEqual(dialect2, "sqlite")
            self.assertEqual(resolved_path, tmp_file)

    def test_postgres_connection_proxy_adapts_question_marks(self) -> None:
        raw_conn = MagicMock()
        proxy = _PostgresConnectionProxy(raw_conn)
        proxy.execute("UPDATE users SET credits = ? WHERE user_id = ?;", ("[1,2,3,4,5]", "u_01"))
        raw_conn.execute.assert_called_once_with(
            "UPDATE users SET credits = %s WHERE user_id = %s;",
            ("[1,2,3,4,5]", "u_01"),
        )


class TestMockedPostgresManagerOperations(unittest.TestCase):
    """Verify PostgreSQL DDL, RETURNING, ON CONFLICT upsert, and sequence sync against a mock psycopg connection."""

    def setUp(self) -> None:
        DRLDatabaseManager.reset_instance()

    def tearDown(self) -> None:
        DRLDatabaseManager.reset_instance()

    def test_postgres_schema_and_upsert_sql_generation(self) -> None:
        executed_sql = []

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_conn.broken = False
        mock_conn.info.transaction_status = 0

        def fake_execute(sql: str, params=None):
            executed_sql.append((sql.strip(), params))
            cur = MagicMock()
            cur.description = [("event_id",)]
            if "information_schema" in sql:
                cur.fetchall.return_value = []
            else:
                cur.fetchone.return_value = {"event_id": 42}
                cur.fetchall.return_value = []
            cur.rowcount = 1
            return cur

        mock_conn.execute.side_effect = fake_execute

        with patch.object(
            DRLDatabaseManager,
            "_import_psycopg",
            return_value=(MagicMock(connect=MagicMock(return_value=mock_conn)), MagicMock()),
        ):
            db = DRLDatabaseManager(db_url="postgresql://drl_user:pw@127.0.0.1:5432/nole_drl")
            self.assertTrue(db.is_postgres)
            self.assertFalse(db.is_sqlite)
            self.assertEqual(db.target_label, "postgresql://drl_user:***@127.0.0.1:5432/nole_drl")

            pk = db.insert_record(
                "events",
                {
                    "event_id": 42,
                    "title": "Test Event",
                    "is_online": True,
                    "topic": ["AI", "Hackathon"],
                },
                or_replace=True,
                raise_on_error=True,
            )
            self.assertEqual(pk, 42)

            upsert_calls = [s for s, _ in executed_sql if s.startswith("INSERT INTO events")]
            self.assertEqual(len(upsert_calls), 1)
            self.assertIn("ON CONFLICT (event_id) DO UPDATE SET", upsert_calls[0])
            self.assertIn("RETURNING event_id", upsert_calls[0])

            seq_calls = [s for s, _ in executed_sql if "pg_get_serial_sequence" in s]
            self.assertEqual(len(seq_calls), 1)


if __name__ == "__main__":
    unittest.main()
