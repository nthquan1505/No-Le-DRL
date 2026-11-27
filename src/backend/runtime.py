"""Shared WebSocket connection manager, scheduler state, and scrape/extraction orchestration."""
import asyncio
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import HTTPException, WebSocket

from config import DEFAULT_SCRAPE_HOURS, DEFAULT_SCRAPE_MINUTE, SCRAPE_TIMEZONE
from events import extract_single_post_url
from scheduler import run_scrape_cycle
from src.model import Fanpage, Post, get_subscribers_for_fanpage


@dataclass
class RuntimeSettings:
    interval_seconds: Optional[int] = None
    scrape_hours: str = DEFAULT_SCRAPE_HOURS
    scrape_minute: int = DEFAULT_SCRAPE_MINUTE
    auto_scrape: bool = True


class ConnectionManager:
    """Manages active WebSocket connections mapped by user `client_id`."""

    def __init__(self) -> None:
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.scrape_lock = asyncio.Lock()

    async def connect(self, client_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.setdefault(client_id, set()).add(websocket)

    def disconnect(self, client_id: str, websocket: WebSocket) -> None:
        if client_id in self.active_connections:
            self.active_connections[client_id].discard(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]

    async def send_to_user(self, client_id: str, payload: dict) -> None:
        sockets = list(self.active_connections.get(client_id, []))
        dead_sockets: List[WebSocket] = []
        for ws in sockets:
            try:
                await ws.send_json(payload)
            except Exception:
                dead_sockets.append(ws)
        for ws in dead_sockets:
            self.disconnect(client_id, ws)

    async def broadcast_to_all(self, payload: dict) -> None:
        for client_id in list(self.active_connections.keys()):
            await self.send_to_user(client_id, payload)

    async def on_new_posts_scraped(self, fanpage: Fanpage, new_posts: List[Post]) -> None:
        payload = {
            "event": "new_posts",
            "fanpage": fanpage.to_dict(),
            "posts": [p.to_dict() for p in new_posts],
        }
        await self.send_to_user("__public__", payload)

        subscriber_ids = get_subscribers_for_fanpage(fanpage.id)
        for client_id in subscriber_ids:
            if client_id in self.active_connections:
                await self.send_to_user(client_id, payload)

    async def broadcast_event_updated(
        self,
        event_data: Dict[str, Any],
        changes: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "event": "event_updated",
            "event_id": event_data.get("event_id"),
            "event_data": event_data,
            "changes": changes or {},
            "refresh_tree": True,
        }
        await self.broadcast_to_all(payload)

    async def on_events_updated(
        self,
        fanpage: Fanpage,
        updated_events: List[Dict[str, Any]],
    ) -> None:
        for item in updated_events:
            ev_data = item.get("event")
            if ev_data:
                await self.broadcast_event_updated(ev_data, item.get("changes"))


manager = ConnectionManager()
scheduler = AsyncIOScheduler(timezone=SCRAPE_TIMEZONE)
runtime_settings = RuntimeSettings()


def sync_settings_from_server_module() -> RuntimeSettings:
    """Keep `runtime_settings` synchronized with `server.SERVER_*` attributes set by `main.py`."""
    try:
        import sys

        srv = sys.modules.get("server")
        if srv is not None:
            runtime_settings.interval_seconds = getattr(srv, "SERVER_INTERVAL_SECONDS", runtime_settings.interval_seconds)
            runtime_settings.scrape_hours = getattr(srv, "SERVER_SCRAPE_HOURS", runtime_settings.scrape_hours)
            runtime_settings.scrape_minute = getattr(srv, "SERVER_SCRAPE_MINUTE", runtime_settings.scrape_minute)
            runtime_settings.auto_scrape = getattr(srv, "SERVER_AUTO_SCRAPE", runtime_settings.auto_scrape)
    except Exception:
        pass
    return runtime_settings


def trigger_github_actions_crawl(target_slug_or_id: Optional[str] = None) -> bool:
    """Trigger GitHub Actions `live_server_demo.yml` via `workflow_dispatch` when `GITHUB_TOKEN` is set."""
    gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not gh_token:
        return False
    repo = os.environ.get("GITHUB_REPO", "vttaan/fb-rss-bot").strip()
    ref = os.environ.get("GITHUB_REF", "main").strip() or "main"
    url = f"https://api.github.com/repos/{repo}/actions/workflows/live_server_demo.yml/dispatches"
    payload: dict = {"ref": ref}
    if target_slug_or_id:
        payload["inputs"] = {"target_fanpage": str(target_slug_or_id)}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {gh_token}",
            "Content-Type": "application/json",
            "User-Agent": "fb-rss-bot-admin",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 201, 204)
    except Exception as exc:
        print(f"[admin] GitHub workflow_dispatch trigger failed: {exc}")
        return False


async def scheduled_scrape_task() -> None:
    if manager.scrape_lock.locked():
        return
    async with manager.scrape_lock:
        await manager.broadcast_to_all({"event": "status", "message": "Running scheduled fanpage scrape & extractor..."})
        new_count = await run_scrape_cycle(
            on_new_posts=manager.on_new_posts_scraped,
            on_events_updated=manager.on_events_updated,
        )
        await manager.broadcast_to_all(
            {
                "event": "status",
                "message": f"Scrape cycle finished ({new_count} new post(s) discovered).",
                "refresh_tree": True,
            }
        )


async def trigger_background_scrape(target_slug_or_id: str | None = None) -> None:
    async with manager.scrape_lock:
        await manager.broadcast_to_all(
            {"event": "status", "message": f"Scraping fanpage(s) ({target_slug_or_id or 'all'})..."}
        )
        new_count = await run_scrape_cycle(
            target_slug_or_id=target_slug_or_id,
            on_new_posts=manager.on_new_posts_scraped,
            on_events_updated=manager.on_events_updated,
        )
        await manager.broadcast_to_all(
            {
                "event": "status",
                "message": f"Scrape finished! Found {new_count} new post(s).",
                "refresh_tree": True,
            }
        )


async def trigger_unified_scrape(target_slug_or_id: str | None = None, *, force_local: bool = False) -> Dict[str, Any]:
    cfg = sync_settings_from_server_module()
    use_actions_env = os.environ.get("USE_GITHUB_ACTIONS_CRAWLER", "").strip().lower() in ("1", "true", "yes")
    prefer_actions = (not force_local) and (not cfg.auto_scrape or use_actions_env)

    if prefer_actions:
        dispatched = await asyncio.to_thread(trigger_github_actions_crawl, target_slug_or_id)
        if dispatched:
            scope = f" for '{target_slug_or_id}'" if target_slug_or_id else ""
            msg = f"Dispatched GitHub Actions crawler{scope} (results will sync back automatically)."
            await manager.broadcast_to_all({"event": "status", "message": msg})
            return {"status": "dispatched", "mode": "github_actions_dispatched", "message": msg}

        if not cfg.auto_scrape:
            msg = (
                "GitHub Actions dispatch unavailable (set GITHUB_TOKEN on the Ngrok host); "
                "queued for the next scheduled GitHub Actions crawl."
            )
            await manager.broadcast_to_all({"event": "status", "message": msg})
            return {"status": "queued_for_actions", "mode": "queued_for_actions", "message": msg}

    if manager.scrape_lock.locked():
        msg = "A local scrape cycle is already in progress..."
        return {"status": "busy", "mode": "local_background", "message": msg}

    asyncio.create_task(trigger_background_scrape(target_slug_or_id))
    msg = f"Started VM scrape & extraction ({target_slug_or_id or 'all'})."
    return {"status": "started", "mode": "local_background", "message": msg}


async def process_single_post_url(url: str) -> Dict[str, Any]:
    """Scrape a single Facebook post URL, extract event metadata, and auto-add fanpage if applicable."""
    clean_url = (url or "").strip()
    if not clean_url:
        raise HTTPException(status_code=400, detail="Post URL is required.")

    result = await extract_single_post_url(clean_url)
    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result.get("detail") or "Failed to extract post.")

    if result.get("is_event"):
        ev_title = (result.get("event") or {}).get("title") or "Event"
        fp = result.get("fanpage")
        if result.get("fanpage_added") and fp:
            msg = f"Event found: '{ev_title}'. Added fanpage '{fp.get('name') or fp.get('slug')}' to tracked list."
        elif fp:
            msg = f"Event found: '{ev_title}' (fanpage '{fp.get('name') or fp.get('slug')}' already tracked)."
        else:
            msg = f"Event found: '{ev_title}' (fanpage link could not be resolved)."

        await manager.broadcast_to_all({"event": "status", "message": msg, "refresh_tree": True})
        if result.get("status") == "updated" and result.get("event"):
            await manager.broadcast_event_updated(result["event"], result.get("changes"))
        if result.get("post") and fp:
            await manager.send_to_user(
                "__public__",
                {"event": "new_posts", "fanpage": fp, "posts": [result["post"]]},
            )
    else:
        msg = result.get("detail") or "Post fetched and analyzed, but it is not an event (fanpage not added)."

    return {**result, "message": msg}
