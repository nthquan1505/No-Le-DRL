"""Model-layer repository helpers for events, users, subscriptions, and distributed feed inspection."""
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import FEEDS_DIR, LOCAL_TIMEZONE, PUBLIC_NGROK_URL
from database import (
    EmailMessage,
    EventRecord,
    EventSubscription,
    Fanpage,
    Post,
    PostAnalysis,
    User,
    UserSubscription,
    add_fanpage,
    create_user,
    ensure_local_tz,
    get_all_events,
    get_all_recent_posts,
    get_event_by_id,
    get_inbox_messages,
    get_or_create_user,
    get_posts_for_fanpage,
    get_posts_for_user,
    get_session,
    get_subscribers_for_fanpage,
    get_user_by_id,
    get_user_by_username,
    get_user_event_subscriptions,
    get_user_subscriptions,
    get_webapp_session,
    init_db,
    list_all_event_subscriptions,
    list_fanpages,
    list_users,
    local_now_iso,
    remove_fanpage,
    save_posts,
    subscribe_user_to_event,
    subscribe_user_to_fanpages,
    unsubscribe_user_from_event,
    unsubscribe_user_from_fanpage,
    update_event_subscription,
    update_fanpage_metadata,
)
from events import sync_all_event_folders


def query_filtered_events(
    *,
    source_type: Optional[str] = None,
    client_id: Optional[str] = None,
    fanpage_id: Optional[int] = None,
    category: Optional[str] = None,
    format_tag: Optional[str] = None,
    is_online: Optional[bool] = None,
    q: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Query unified `EventRecord` rows from `webapp.db` and apply classification/search filters."""
    rows = get_all_events(source_type=source_type, client_id=client_id, fanpage_id=fanpage_id, limit=limit)
    items = [r.to_dict() for r in rows]
    if category:
        cat_l = category.strip().lower()
        items = [e for e in items if any(cat_l == str(c).lower() for c in (e.get("category") or []))]
    if format_tag:
        fmt_l = format_tag.strip().lower()
        items = [e for e in items if any(fmt_l == str(f).lower() for f in (e.get("format") or []))]
    if is_online is not None:
        items = [e for e in items if bool(e.get("is_online")) == is_online]
    if q and q.strip():
        needle = q.strip().lower()
        items = [
            e
            for e in items
            if needle in (e.get("title") or "").lower()
            or needle in (e.get("description") or "").lower()
            or needle in (e.get("organizer") or "").lower()
            or needle in (e.get("location") or "").lower()
        ]
    return items


def count_fanpage_events_file(slug: str) -> int:
    """Count extracted events in a fanpage's `events.json` feed file."""
    for candidate in (FEEDS_DIR / "fanpages" / slug / "events.json", FEEDS_DIR / slug / "events.json"):
        if candidate.is_file():
            try:
                return int(json.loads(candidate.read_text(encoding="utf-8")).get("count", 0))
            except Exception:
                continue
    return 0


def inspect_feed_file(file_path: Path) -> Dict[str, Any]:
    """Inspect a single distributed JSON/XML feed file for metadata."""
    rel = file_path.relative_to(FEEDS_DIR).as_posix()
    stat = file_path.stat()
    mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=LOCAL_TIMEZONE).isoformat()
    item_count: Optional[int] = None
    if file_path.suffix.lower() == ".json":
        try:
            doc = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                item_count = doc.get("count")
        except Exception:
            pass
    return {
        "name": file_path.name,
        "path": f"/feeds/{rel}",
        "relative_path": rel,
        "type": file_path.suffix.lstrip(".").lower() or "file",
        "size_bytes": stat.st_size,
        "updated_at": mtime_iso,
        "count": item_count,
    }


def build_feeds_directory_tree() -> Dict[str, Any]:
    """Build a structured directory tree of `feeds/` organized into logical folders."""
    sync_all_event_folders()
    folders: Dict[str, List[Dict[str, Any]]] = {
        "all": [],
        "fanpages": [],
        "emails": [],
        "inbox": [],
    }

    for p in sorted(FEEDS_DIR.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".json":
            continue
        rel_parts = p.relative_to(FEEDS_DIR).parts
        info = inspect_feed_file(p)

        if rel_parts[0] == "all":
            folders["all"].append(info)
        elif rel_parts[0] == "fanpages":
            folders["fanpages"].append(info)
        elif rel_parts[0] == "emails":
            folders["emails"].append(info)
        elif rel_parts[0] == "inbox":
            folders["inbox"].append(info)
        elif len(rel_parts) == 2 and p.name == "events.json":
            continue

    return {
        "public_url": PUBLIC_NGROK_URL,
        "updated_at": datetime.now(LOCAL_TIMEZONE).isoformat(),
        "folders": [
            {
                "folder": "/feeds/all/",
                "key": "all",
                "description": "Combined deduplicated events.json across all sources",
                "files": folders["all"],
            },
            {
                "folder": "/feeds/fanpages/",
                "key": "fanpages",
                "description": "Extracted events.json distributed per Facebook fanpage folder (<slug>/events.json)",
                "files": folders["fanpages"],
            },
            {
                "folder": "/feeds/emails/",
                "key": "emails",
                "description": "Extracted events.json from client email submissions (global & per-client folders)",
                "files": folders["emails"],
            },
            {
                "folder": "/feeds/inbox/",
                "key": "inbox",
                "description": "Client Email Inbox messages log and extraction statuses",
                "files": folders["inbox"],
            },
        ],
    }
