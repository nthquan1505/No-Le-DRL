"""Stage-4 `drl_database.db` deduplication, updates & soft-cancellation (`src/extraction/deduplicator.py`).

Implements the exact `EXT` database contract from `SYSTEM_ARCHITECTURE.md`:
- `created`: Inserts into `events` with `category = NULL` and `min_credit = NULL`; updates `fanpages.number_of_events`.
- `updated`: Non-destructively merges updated fields into `events` and notifies subscribed users in `notifications`.
- `cancelled`: Soft-cancels (`is_cancelled = 1`—never deletes `event_id`) and notifies subscribed users in `notifications`.
- `duplicate`: Appends cross-channel provenance to `sources` without touching `category` or `min_credit`.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from database import DRLDatabaseManager, now_iso_utc7

from .contracts import ExtractedEventDraft, PersistenceAction, RawIngestionItem
from .prefilter import normalize_vi_ascii

TITLE_TAG_STRIP_RE = re.compile(
    r"\b(?:thong\s*bao|cap\s*nhat|update|nhac\s*nho|reminder|tam\s*hoan|huy\s*to\s*chuc|huy\s*su\s*kien|gia\s*han)\b",
    re.IGNORECASE,
)


def _canonical_title_for_match(title: str) -> str:
    ascii_t = normalize_vi_ascii(title or "")
    stripped = TITLE_TAG_STRIP_RE.sub(" ", ascii_t)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", stripped)
    return re.sub(r"\s+", " ", cleaned).strip()


def compute_title_similarity(title_a: str, title_b: str) -> float:
    """Combine SequenceMatcher ratio and token containment for resilient Vietnamese title matching."""
    ca = _canonical_title_for_match(title_a)
    cb = _canonical_title_for_match(title_b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0

    seq_ratio = difflib.SequenceMatcher(None, ca, cb).ratio()
    tokens_a = {t for t in ca.split() if len(t) > 1}
    tokens_b = {t for t in cb.split() if len(t) > 1}
    if not tokens_a or not tokens_b:
        return seq_ratio

    union_sz = len(tokens_a | tokens_b)
    inter_sz = len(tokens_a & tokens_b)
    jaccard = inter_sz / union_sz if union_sz else 0.0

    # Also handle cases where a short cancellation notice embeds the full event title or vice versa
    min_sz = min(len(tokens_a), len(tokens_b))
    containment = (inter_sz / min_sz) if min_sz >= 3 else 0.0
    return max(seq_ratio, jaccard, containment * 0.92)


def format_provenance_source(item: RawIngestionItem, organizer: str = "") -> str:
    """Format `events.sources` consistently (compatible with `CRD` level inference)."""
    parts: List[str] = []
    channel_label = f"{item.source_type} ({item.source_name})" if item.source_name else item.source_type
    parts.append(channel_label)
    org = (organizer or "").strip()
    if org and org.lower() not in channel_label.lower():
        parts.append(f"organizer: {org}")
    return " | ".join(parts)


def merge_sources_strings(existing: str, incoming: str) -> str:
    tokens: List[str] = []
    seen_low: set[str] = set()
    for chunk in [existing, incoming]:
        for part in (chunk or "").split("|"):
            p = part.strip()
            if p and p.lower() not in seen_low:
                seen_low.add(p.lower())
                tokens.append(p)
    return " | ".join(tokens)


def _merge_unique_lists(existing: Sequence[str], incoming: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in list(existing or []) + list(incoming or []):
        s = str(item or "").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def _event_date_prefix(iso_str: Optional[str]) -> Optional[str]:
    if iso_str and len(iso_str) >= 10:
        return iso_str[:10]
    return None


def find_matching_event(
    db: DRLDatabaseManager,
    draft: ExtractedEventDraft,
    candidate_limit: int = 200,
) -> Optional[Dict[str, Any]]:
    """Find an existing row in `events` matching `draft` via URL or title+date anchors."""
    # 1. Exact post_url match
    if draft.post_url:
        exact_post = db.fetch_one(
            "SELECT * FROM events WHERE post_url = ? ORDER BY event_id DESC LIMIT 1;",
            (draft.post_url,),
            table="events",
        )
        if exact_post:
            return exact_post

    candidates = db.fetch_all(
        "SELECT * FROM events ORDER BY event_id DESC LIMIT ?;",
        (candidate_limit,),
        table="events",
    )
    if not candidates:
        return None

    # 2. Exact registration_url match (when title shows reasonable overlap or update/cancel)
    if draft.registration_url:
        for row in candidates:
            if row.get("registration_url") and row["registration_url"] == draft.registration_url:
                sim = compute_title_similarity(draft.title, str(row.get("title") or ""))
                if sim >= 0.55 or draft.classification in ("event_update", "event_cancellation"):
                    return row

    # 3. Anchor similarity match (Title + Date/Lifecycle)
    draft_date = _event_date_prefix(draft.event_start_datetime)
    best_row: Optional[Dict[str, Any]] = None
    best_score = 0.0

    for row in candidates:
        sim = compute_title_similarity(draft.title, str(row.get("title") or ""))
        row_date = _event_date_prefix(row.get("event_start_datetime"))

        if draft.classification in ("event_update", "event_cancellation"):
            # Rescheduled or cancelled events may announce a new date or omit the date
            if sim >= 0.72 and sim > best_score:
                best_score = sim
                best_row = row
        else:
            same_date = bool(draft_date and row_date and draft_date == row_date)
            missing_one_date = draft_date is None or row_date is None
            same_loc = bool(
                draft.location
                and row.get("location")
                and normalize_vi_ascii(draft.location) == normalize_vi_ascii(str(row.get("location")))
            )
            if sim >= 0.78 and (same_date or (missing_one_date and (same_loc or sim >= 0.88))):
                if sim > best_score:
                    best_score = sim
                    best_row = row

    return best_row


def _notify_subscribers(
    db: DRLDatabaseManager,
    event_id: int,
    notif_type: str,
    content: str,
) -> int:
    """Create a notification row in `notifications` for every user subscribed to `event_id`."""
    subs = db.fetch_all(
        "SELECT user_id FROM event_subscription WHERE event_id = ?;",
        (event_id,),
        table="event_subscription",
    )
    created = 0
    now = now_iso_utc7()
    for sub in subs:
        uid = sub.get("user_id")
        if not uid:
            continue
        nid = db.insert_record(
            "notifications",
            {
                "user_id": uid,
                "type": notif_type,
                "content": content,
                "event_id": event_id,
                "is_read": False,
                "created_at": now,
            },
        )
        if nid is not None:
            created += 1
    return created


def _refresh_fanpage_event_count(db: DRLDatabaseManager, fanpage_id: Optional[int], delta: int) -> None:
    if fanpage_id is None or delta == 0:
        return
    fp = db.fetch_one(
        "SELECT fanpage_id, number_of_events FROM fanpages WHERE fanpage_id = ?;",
        (fanpage_id,),
        table="fanpages",
    )
    if fp is None:
        return
    new_count = max(0, int(fp.get("number_of_events") or 0) + delta)
    db.update_records("fanpages", {"number_of_events": new_count}, {"fanpage_id": fanpage_id})


def persist_extracted_draft(
    db: DRLDatabaseManager,
    draft: ExtractedEventDraft,
    item: RawIngestionItem,
) -> Tuple[PersistenceAction, Optional[int], List[str], int]:
    """Upsert or soft-cancel `draft` in `drl_database.db`.

    Returns `(action, event_id, changed_fields, notifications_created)`.
    """
    if not draft.is_actionable_event:
        return "ignored", None, [], 0

    draft.sources = format_provenance_source(item, draft.organizer)
    existing = find_matching_event(db, draft)

    # Case A: Cancellation without a previously tracked event -> ignore cleanly (do not create dead rows)
    if existing is None and draft.classification == "event_cancellation":
        return "ignored", None, [], 0

    # Case B: Brand-new event (or standalone update whose original post wasn't captured yet)
    if existing is None:
        payload = draft.to_event_db_payload(is_cancelled=False)
        new_id = db.insert_record("events", payload, raise_on_error=True)
        _refresh_fanpage_event_count(db, item.fanpage_id, delta=1)
        return "created", int(new_id), ["*"], 0

    event_id = int(existing["event_id"])
    merged_sources = merge_sources_strings(str(existing.get("sources") or ""), draft.sources)

    # Case C: Soft-cancellation of an existing event (`is_cancelled = 1`)
    if draft.classification == "event_cancellation":
        if bool(existing.get("is_cancelled", False)):
            if merged_sources != existing.get("sources"):
                db.update_records("events", {"sources": merged_sources}, {"event_id": event_id})
            return "duplicate", event_id, [], 0

        db.update_records(
            "events",
            {
                "is_cancelled": True,
                "sources": merged_sources,
                "updated_at": now_iso_utc7(),
            },
            {"event_id": event_id},
        )
        _refresh_fanpage_event_count(db, item.fanpage_id, delta=-1)
        notif_count = _notify_subscribers(
            db,
            event_id=event_id,
            notif_type="cancellation",
            content=f"Sự kiện \"{existing.get('title')}\" đã có thông báo tạm hoãn hoặc hủy tổ chức.",
        )
        return "cancelled", event_id, ["is_cancelled"], notif_count

    # Case D: Existing event -> check whether meaningful fields changed (`updated` vs `duplicate`)
    updates: Dict[str, Any] = {}
    changed_fields: List[str] = []

    scalar_candidates: Tuple[Tuple[str, Any], ...] = (
        ("reg_start_datetime", draft.reg_start_datetime),
        ("reg_end_datetime", draft.reg_end_datetime),
        ("event_start_datetime", draft.event_start_datetime),
        ("event_end_datetime", draft.event_end_datetime),
        ("location", draft.location),
        ("coordinate", draft.coordinate),
        ("fee", draft.fee),
        ("award", draft.award),
        ("registration_url", draft.registration_url),
    )

    is_explicit_update = draft.classification == "event_update"
    incoming_is_reminder = bool(
        re.search(
            r"\b(nhac\s+hen|nhac\s+lich|chi\s+con\s+\d+|24\s*gio\s*cuoi)\b",
            normalize_vi_ascii(draft.title or ""),
        )
    )
    # If an earlier teaser/venue-intro had no registration_url and this incoming post brings a real registration_url,
    # promote its canonical title & description so teaser wording never sticks on the record!
    is_primary_launch_upgrade = (
        not existing.get("registration_url")
        and bool(draft.registration_url)
        and not incoming_is_reminder
    )
    if is_primary_launch_upgrade:
        if draft.title and draft.title != existing.get("title"):
            updates["title"] = draft.title
        if draft.description and draft.description != existing.get("description"):
            updates["description"] = draft.description

    for col, new_val in scalar_candidates:
        if new_val in (None, ""):
            continue
        old_val = existing.get(col)
        # Fill missing fields on duplicates, or overwrite changed schedule/venue on explicit updates
        if old_val in (None, "") or (is_explicit_update and str(new_val) != str(old_val)):
            if str(new_val) != str(old_val):
                updates[col] = new_val
                changed_fields.append(col)

    # Reactivate if a previously soft-cancelled event announces a rescheduled date!
    if is_explicit_update and bool(existing.get("is_cancelled", False)) and draft.event_start_datetime:
        updates["is_cancelled"] = False
        changed_fields.append("is_cancelled")
        _refresh_fanpage_event_count(db, item.fanpage_id, delta=1)

    for list_col, incoming_list in (
        ("keywords", draft.keywords),
        ("target_audience", draft.target_audience),
        ("topic", draft.topic),
        ("format", draft.format),
        ("images_url", draft.images_url),
    ):
        old_list = existing.get(list_col) or []
        merged_list = _merge_unique_lists(old_list, incoming_list)
        if merged_list != old_list and list_col in ("images_url", "topic", "format"):
            updates[list_col] = merged_list
        elif not old_list and merged_list:
            updates[list_col] = merged_list

    if not existing.get("post_url") and draft.post_url:
        updates["post_url"] = draft.post_url

    if merged_sources != (existing.get("sources") or ""):
        updates["sources"] = merged_sources

    if not changed_fields:
        # Pure cross-post duplicate (only provenance/images/tags enriched, no schedule/venue change)
        if updates:
            db.update_records("events", updates, {"event_id": event_id})
        return "duplicate", event_id, [], 0

    updates["updated_at"] = now_iso_utc7()
    db.update_records("events", updates, {"event_id": event_id})

    notif_count = _notify_subscribers(
        db,
        event_id=event_id,
        notif_type="update",
        content=(
            f"Sự kiện \"{existing.get('title')}\" vừa cập nhật thông tin: "
            f"{', '.join(changed_fields)}."
        ),
    )
    return "updated", event_id, changed_fields, notif_count
