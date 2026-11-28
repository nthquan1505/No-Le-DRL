"""SQLite cache for reusable Fit data. Calendar text and addresses stay in memory."""

from contextlib import closing
from dataclasses import asdict
from datetime import timezone
import json
from pathlib import Path
import sqlite3
import time

from .engine import FitAssessment, TravelLeg


class FitCache:
    def __init__(self, path=Path("runtime/module4/fit_cache.sqlite3"), *, retention_days=30, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.retention = retention_days * 86400
        with closing(self._connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS fit (
                    user_id TEXT NOT NULL, event_id TEXT NOT NULL, signature TEXT NOT NULL,
                    expires REAL, accessed REAL NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY (user_id, event_id)
                );
                CREATE TABLE IF NOT EXISTS cleanup (id INTEGER PRIMARY KEY, last_run REAL NOT NULL);
            """)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=5)

    def get(self, user_id, event_id, signature):
        now = self.clock()
        with closing(self._connect()) as db, db:
            row = db.execute(
                "SELECT payload FROM fit WHERE user_id=? AND event_id=? AND signature=? "
                "AND (expires IS NULL OR expires>?) AND accessed>?",
                (user_id, event_id, signature, now, now - self.retention),
            ).fetchone()
            if row is None:
                return None
            db.execute("UPDATE fit SET accessed=? WHERE user_id=? AND event_id=? AND signature=?",
                       (now, user_id, event_id, signature))
        payload = json.loads(row[0])
        for field in ("inbound", "outbound"):
            if payload.get(field) is not None:
                payload[field] = TravelLeg(**payload[field])
        return FitAssessment(**payload)

    def put(self, user_id, event_id, signature, result, *, end=None):
        if result.provisional or any(leg and leg.source == "google" for leg in (result.inbound, result.outbound)):
            return
        payload = asdict(result)
        payload["flexible_overlaps"] = ()
        payload["conflict_detail"] = None
        if result.has_conflict:
            payload["reasons"] = []
        expires = end.astimezone(timezone.utc).timestamp() if end else None
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO fit VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(user_id, event_id) DO UPDATE SET "
                "signature=excluded.signature, expires=excluded.expires, accessed=excluded.accessed, "
                "payload=excluded.payload",
                (user_id, event_id, signature, expires, self.clock(), json.dumps(payload, ensure_ascii=False)),
            )

    def delete_user(self, user_id):
        with closing(self._connect()) as db, db:
            return db.execute("DELETE FROM fit WHERE user_id=?", (user_id,)).rowcount

    def delete_event(self, event_id):
        with closing(self._connect()) as db, db:
            return db.execute("DELETE FROM fit WHERE event_id=?", (event_id,)).rowcount

    def clean(self):
        now = self.clock()
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            last = db.execute("SELECT last_run FROM cleanup WHERE id=1").fetchone()
            if last and now - last[0] < 86400:
                return 0
            deleted = db.execute("DELETE FROM fit WHERE expires<=? OR accessed<=?",
                                 (now, now - self.retention)).rowcount
            db.execute("INSERT OR REPLACE INTO cleanup VALUES (1, ?)", (now,))
            return deleted
