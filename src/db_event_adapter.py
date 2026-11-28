"""Database Input/Output Adapter bridging `database/drl_db_manager.py` with Module 3.

Reads scraped events (and optional user/subscription/rubric records) directly from
`drl_database.db` via `DRLDatabaseManager`, replacing hardcoded `events.json` inputs
while preserving full compatibility with `SYSTEM_ARCHITECTURE.md`'s 6-table schema
and handling an empty database (0 rows) gracefully.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from database.drl_db_manager import DRLDatabaseManager, decode_json_list, get_drl_db
from src.schemas import EventNormalized, TargetAudienceStructured

GROUP_ID_TO_DB_CATEGORY: Dict[str, int] = {
    "C1": 1,
    "C2": 2,
    "C3": 3,
    "C4": 4,
    "C5": 5,
}
DB_CATEGORY_TO_GROUP_ID: Dict[int, str] = {v: k for k, v in GROUP_ID_TO_DB_CATEGORY.items()}

_ALL_STUDENT_CUES = (
    "toàn thể sinh viên",
    "tất cả sinh viên",
    "mọi sinh viên",
    "sinh viên đang theo học",
    "sinh viên hcmus",
    "sinh viên trường đại học khoa học tự nhiên",
    "sinh viên",
)
_COHORT_RE = re.compile(r"\b(?:k|khóa|khoa)\s*(\d{2,4})\b", re.IGNORECASE)
_FEE_NUM_RE = re.compile(r"(\d[\d\.,\s]*)")


def parse_fee_to_vnd(fee_raw: Any) -> Optional[int]:
    """Parse free-text `events.fee` (e.g. '25.000 VNĐ', 'Miễn phí', '0') into integer VND."""
    if fee_raw is None:
        return None
    if isinstance(fee_raw, (int, float)):
        return max(0, int(round(fee_raw)))
    text = str(fee_raw).strip()
    if not text:
        return None
    low = text.lower()
    if any(tok in low for tok in ("miễn phí", "mien phi", "free", "không thu phí", "0đ", "0 đ")):
        return 0
    m = _FEE_NUM_RE.search(text)
    if not m:
        return None
    digits = re.sub(r"[^\d]", "", m.group(1))
    if not digits:
        return None
    val = int(digits)
    # Handle shorthand like "25k" / "30K"
    if re.search(r"\d\s*k\b", low) and val < 1000:
        val *= 1000
    return val


def parse_target_audience_list(raw_list: Sequence[Any]) -> TargetAudienceStructured:
    """Convert `events.target_audience` JSON string array into `TargetAudienceStructured`."""
    raw_texts = [str(x).strip() for x in raw_list if x is not None and str(x).strip()]
    if not raw_texts:
        return TargetAudienceStructured(raw_texts=[], all_students=False)

    cohorts: List[str] = []
    faculties: List[str] = []
    classes: List[str] = []
    all_students = False

    for text in raw_texts:
        low = text.lower()
        found_cohort = False
        for m in _COHORT_RE.finditer(text):
            num = m.group(1)
            code = f"K{num[-2:]}" if len(num) == 4 else f"K{num}"
            if code not in cohorts:
                cohorts.append(code)
            found_cohort = True

        # Detect faculty mentions (e.g., "Khoa CNTT", "Khoa Toán - Tin học")
        if "khoa " in low and not found_cohort:
            fac_match = re.search(r"khoa\s+([^\.,;()]+)", text, re.IGNORECASE)
            if fac_match:
                fac = fac_match.group(1).strip()
                if fac and fac.lower() not in ("học tự nhiên", "học tự nhiên, đhqg-hcm"):
                    if fac not in faculties:
                        faculties.append(fac)

        if any(cue in low for cue in _ALL_STUDENT_CUES) and not cohorts and not faculties:
            all_students = True

    return TargetAudienceStructured(
        raw_texts=raw_texts,
        cohorts=cohorts,
        faculties=faculties,
        classes=classes,
        all_students=all_students,
    )


def parse_coordinate_string(coord: Optional[str]) -> Optional[Tuple[float, float]]:
    """Parse `'lat,lng'` coordinate string from `events.coordinate` / `users.default_coordinate`."""
    if not coord or not isinstance(coord, str):
        return None
    parts = [p.strip() for p in coord.split(",")]
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def fetch_events_from_db(
    db: Optional[DRLDatabaseManager] = None,
    *,
    db_path: Optional[Union[str, Path]] = None,
    only_unclassified: bool = False,
    only_classified: bool = False,
    exclude_cancelled: bool = True,
    event_ids: Optional[Sequence[int]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch event rows from `drl_database.db` (`events` table) safely, even when empty."""
    manager = db if db is not None else get_drl_db(db_path=db_path)
    clauses: List[str] = []
    params: List[Any] = []

    if exclude_cancelled:
        clauses.append("is_cancelled = 0")
    if only_unclassified:
        clauses.append("category IS NULL")
    if only_classified:
        clauses.append("category IS NOT NULL")
    if event_ids:
        placeholders = ", ".join(["?"] * len(event_ids))
        clauses.append(f"event_id IN ({placeholders})")
        params.extend(int(eid) for eid in event_ids)

    sql = "SELECT * FROM events"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY event_id ASC"
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params.append(int(limit))

    return manager.fetch_all(sql, params=params, table="events", fallback=[])


def fetch_user_from_db(
    user_id: str,
    db: Optional[DRLDatabaseManager] = None,
    *,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch one user row from `users` table (JSON list columns decoded automatically)."""
    manager = db if db is not None else get_drl_db(db_path=db_path)
    return manager.fetch_one(
        "SELECT * FROM users WHERE user_id = ?;",
        params=(str(user_id),),
        table="users",
        fallback=None,
    )


def fetch_user_subscriptions_from_db(
    user_id: str,
    db: Optional[DRLDatabaseManager] = None,
    *,
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Fetch subscription rows for a user from `event_subscription`."""
    manager = db if db is not None else get_drl_db(db_path=db_path)
    return manager.fetch_all(
        "SELECT * FROM event_subscription WHERE user_id = ? ORDER BY event_id ASC;",
        params=(str(user_id),),
        table="event_subscription",
        fallback=[],
    )


def fetch_school_rubric_from_db(
    school_id: int,
    db: Optional[DRLDatabaseManager] = None,
    *,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch one school's max_credit array from `rubric` table."""
    manager = db if db is not None else get_drl_db(db_path=db_path)
    return manager.fetch_one(
        "SELECT * FROM rubric WHERE school_id = ?;",
        params=(int(school_id),),
        table="rubric",
        fallback=None,
    )


def _extract_organizer_from_sources(sources_raw: Any) -> str:
    """Extract clean organizer name from `events.sources` (handles `'... | organizer: <Name>'`)."""
    text = str(sources_raw or "").strip()
    if not text:
        return ""
    m = re.search(r"\borganizer\s*:\s*(.+)$", text, re.IGNORECASE)
    if m:
        extracted = m.group(1).strip()
        if extracted:
            return extracted
    return text


def db_row_to_classification_event(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map one `events` SQLite row into the canonical Classification input dictionary."""
    eid = row.get("event_id")
    is_online = bool(row.get("is_online", False))
    is_cancelled = bool(row.get("is_cancelled", False))
    topics = decode_json_list(row.get("topic"))
    formats = decode_json_list(row.get("format"))
    keywords = decode_json_list(row.get("keywords"))
    target_audience = decode_json_list(row.get("target_audience"))
    images_url = decode_json_list(row.get("images_url"))
    start_dt = row.get("event_start_datetime")
    date_str = str(start_dt)[:10] if isinstance(start_dt, str) and len(start_dt) >= 10 else None
    organizer = _extract_organizer_from_sources(row.get("sources"))

    return {
        "event_id": int(eid) if eid is not None else None,
        "post_id": f"db_{eid}" if eid is not None else None,
        "is_event": not is_cancelled,
        "is_cancelled": is_cancelled,
        "title": str(row.get("title") or ""),
        "description": str(row.get("description") or ""),
        "keywords": [str(k) for k in keywords if k is not None],
        "topic": [str(t) for t in topics if t is not None],
        "format": [str(f) for f in formats if f is not None],
        "category": [str(t) for t in topics if t is not None],  # compatibility alias
        "organizer": organizer,
        "sources": row.get("sources"),
        "date": date_str,
        "is_online": is_online,
        "online_mode": "online" if is_online else "offline",
        "reg_start_datetime": row.get("reg_start_datetime"),
        "reg_end_datetime": row.get("reg_end_datetime"),
        "start_datetime": start_dt,
        "end_datetime": row.get("event_end_datetime"),
        "location": row.get("location"),
        "coordinate": row.get("coordinate"),
        "fee": row.get("fee"),
        "target_audience": [str(a) for a in target_audience if a is not None],
        "award": row.get("award"),
        "registration_url": row.get("registration_url"),
        "post_url": row.get("post_url"),
        "source_url": row.get("post_url"),
        "images_url": [str(u) for u in images_url if u is not None],
        "db_category": row.get("category"),
        "db_min_credit": row.get("min_credit"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "source_type": "database",
    }


def db_row_to_event_normalized(row: Dict[str, Any]) -> EventNormalized:
    """Map one `events` SQLite row into `src.schemas.EventNormalized` for Recommendation & M4/M6."""
    eid = row.get("event_id")
    is_online = bool(row.get("is_online", False))
    topics = [str(t) for t in decode_json_list(row.get("topic")) if t is not None and str(t).strip()]
    formats = [str(f) for f in decode_json_list(row.get("format")) if f is not None and str(f).strip()]
    keywords = [str(k) for k in decode_json_list(row.get("keywords")) if k is not None and str(k).strip()]
    ta_raw = decode_json_list(row.get("target_audience"))
    images_url = [str(u) for u in decode_json_list(row.get("images_url")) if u is not None]

    fee_raw = str(row.get("fee")).strip() if row.get("fee") is not None else None
    fee_vnd = parse_fee_to_vnd(fee_raw)

    missing_flags: List[str] = []
    if not row.get("event_start_datetime"):
        missing_flags.append("start_datetime_missing")
    if not row.get("event_end_datetime"):
        missing_flags.append("end_datetime_missing")
    if not row.get("reg_end_datetime"):
        missing_flags.append("reg_end_datetime_missing")
    if not row.get("sources"):
        missing_flags.append("organizer_missing")
    if fee_raw is None:
        missing_flags.append("fee_missing")
    elif fee_vnd is None:
        missing_flags.append("fee_unparseable")

    return EventNormalized(
        event_id=f"db_{eid}" if eid is not None else "db_unknown",
        title=str(row.get("title") or ""),
        description=str(row.get("description") or ""),
        organizer=_extract_organizer_from_sources(row.get("sources")),
        start_datetime=row.get("event_start_datetime"),
        end_datetime=row.get("event_end_datetime"),
        reg_start_datetime=row.get("reg_start_datetime"),
        reg_end_datetime=row.get("reg_end_datetime"),
        online_mode="online" if is_online else "offline",
        is_online=is_online,
        location=row.get("location"),
        fee_vnd=fee_vnd,
        fee_raw=fee_raw,
        target_audience=parse_target_audience_list(ta_raw),
        capacity=None,
        seats_left=None,
        registration_url=row.get("registration_url"),
        source_type="database",
        source_url=row.get("post_url"),
        categories=topics,
        missing_field_flags=missing_flags,
        metadata={
            "db_event_id": int(eid) if eid is not None else None,
            "topic": topics,
            "format": formats,
            "keywords": keywords,
            "coordinate": row.get("coordinate"),
            "award": row.get("award"),
            "images_url": images_url,
            "is_cancelled": bool(row.get("is_cancelled", False)),
            "db_category": row.get("category"),
            "db_min_credit": row.get("min_credit"),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        },
    )


def compute_events_fingerprint(events: Sequence[Dict[str, Any]]) -> str:
    """Compute a deterministic SHA-256 fingerprint across DB or JSON event records for caching."""
    canonical: List[Dict[str, Any]] = []
    for idx, ev in enumerate(events):
        canonical.append(
            {
                "idx": idx,
                "id": ev.get("event_id", ev.get("post_id")),
                "title": ev.get("title"),
                "description": ev.get("description"),
                "topic": ev.get("topic", ev.get("category")),
                "format": ev.get("format"),
                "is_event": ev.get("is_event", True),
            }
        )
    raw = json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_classification_category_to_db(
    event_id: int,
    predicted_group_id: Optional[str],
    *,
    db: Optional[DRLDatabaseManager] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Write predicted major rubric category (1..5) back to `events.category` per SYSTEM_ARCHITECTURE."""
    if not predicted_group_id:
        return False
    cat_int = GROUP_ID_TO_DB_CATEGORY.get(str(predicted_group_id).strip().upper())
    if cat_int is None:
        return False
    manager = db if db is not None else get_drl_db(db_path=db_path)
    updated = manager.update_records(
        "events",
        {"category": cat_int},
        {"event_id": int(event_id)},
        fallback=0,
    )
    return updated > 0
