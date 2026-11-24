"""Events, email inbox, distributed feeds, and WebSocket controller (Admin-only)."""
import asyncio
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from config import FEEDS_DIR, SCRAPE_TIMEZONE
from events import ingest_email_message
from rss import generate_user_rss_xml
from scheduler import backup_state_to_git, parse_scrape_hours, run_scrape_cycle
from src.control.runtime import (
    manager,
    process_single_post_url,
    scheduler,
    sync_settings_from_server_module,
    trigger_unified_scrape,
)
from src.model import (
    ADMIN_SESSION_COOKIE,
    BatchEmailIngestRequest,
    EmailIngestRequest,
    ExtractPostRequest,
    SubscribeRequest,
    build_feeds_directory_tree,
    get_all_recent_posts,
    get_inbox_messages,
    get_or_create_user,
    get_posts_for_user,
    get_user_subscriptions,
    inspect_feed_file,
    list_fanpages,
    query_filtered_events,
    require_admin_session,
    subscribe_user_to_fanpages,
    unsubscribe_user_from_fanpage,
    verify_admin_token_value,
    verify_session_token,
)

# HTTP routes router — every endpoint requires an active Admin session/token
router = APIRouter(tags=["admin-functions"], dependencies=[Depends(require_admin_session)])

# WebSocket router — verifies Admin session cookie/header before accepting connections
ws_router = APIRouter(tags=["admin-websockets"])


def _is_ws_admin_authenticated(websocket: WebSocket) -> bool:
    hdr_token = websocket.headers.get("x-admin-token", "")
    if hdr_token and verify_admin_token_value(hdr_token):
        return True
    cookie_tok = websocket.cookies.get(ADMIN_SESSION_COOKIE)
    data = verify_session_token(cookie_tok)
    return bool(data and data.get("role") == "admin")


@router.get("/api/tree")
def api_get_directory_tree():
    """Return the complete folder tree of distributed JSON and XML files (Admin only)."""
    return build_feeds_directory_tree()


@router.get("/api/schedule")
def api_get_schedule():
    """Return the active scrape schedule configuration and next run timestamp (Admin only)."""
    cfg = sync_settings_from_server_module()
    job = scheduler.get_job("server_scheduled_scrape") if scheduler.running else None
    next_run = job.next_run_time.isoformat() if job and job.next_run_time else None
    parsed_hours = parse_scrape_hours(cfg.scrape_hours)
    mode = (
        "external_push"
        if not cfg.auto_scrape
        else ("interval" if cfg.interval_seconds is not None else "daily_cron")
    )
    return {
        "mode": mode,
        "auto_scrape": cfg.auto_scrape,
        "timezone": str(SCRAPE_TIMEZONE),
        "interval_seconds": cfg.interval_seconds,
        "scheduled_times_ict": [f"{h:02d}:{cfg.scrape_minute:02d}" for h in parsed_hours],
        "next_run_at": next_run,
    }


@router.post("/api/inbox")
@router.post("/api/events/ingest-email")
async def api_ingest_email(body: EmailIngestRequest):
    """Receive an event-related email entry and run unified extraction (Admin only)."""
    if not body.content or not body.content.strip():
        raise HTTPException(status_code=400, detail="Email content must not be empty.")

    try:
        get_or_create_user(body.client_id or "anonymous")
        result = await asyncio.to_thread(
            ingest_email_message,
            content=body.content,
            subject=body.subject,
            client_id=body.client_id,
            message_id=body.message_id,
            sender=body.sender,
            received_at=body.received_at,
            media_urls=body.media_urls,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inbox extraction failed: {exc}") from exc

    await manager.broadcast_to_all(
        {
            "event": "inbox_updated",
            "result": result,
            "refresh_tree": True,
        }
    )
    if result.get("is_updated") and result.get("event"):
        await manager.broadcast_event_updated(result["event"], result.get("changes"))
    return result


@router.post("/api/inbox/batch")
async def api_ingest_email_batch(body: BatchEmailIngestRequest):
    """Batch upload endpoint for filtered emails (Admin only)."""
    results = []
    for item in body.emails:
        if not item.content or not item.content.strip():
            continue
        get_or_create_user(item.client_id or "anonymous")
        res = await asyncio.to_thread(
            ingest_email_message,
            content=item.content,
            subject=item.subject,
            client_id=item.client_id,
            message_id=item.message_id,
            sender=item.sender,
            received_at=item.received_at,
            media_urls=item.media_urls,
        )
        results.append(res)

    await manager.broadcast_to_all({"event": "inbox_updated", "refresh_tree": True})
    for res in results:
        if res.get("is_updated") and res.get("event"):
            await manager.broadcast_event_updated(res["event"], res.get("changes"))
    return {
        "processed": len(results),
        "created": sum(1 for r in results if r["status"] == "created"),
        "updated": sum(1 for r in results if r["status"] == "updated"),
        "duplicates": sum(1 for r in results if r["status"] == "duplicate"),
        "ignored": sum(1 for r in results if r["status"] == "ignored"),
        "results": results,
    }


@router.get("/api/inbox")
def api_list_inbox_messages(
    client_id: Optional[str] = Query(None, description="Optional client_id filter"),
    limit: int = Query(50, ge=1, le=200),
):
    """List email entries in the server Inbox along with extraction statuses (Admin only)."""
    rows = get_inbox_messages(client_id=client_id, limit=limit)
    items = [r.to_dict() for r in rows]
    return {
        "count": len(items),
        "messages": items,
    }


@router.get("/api/events")
def api_list_unified_events(
    source_type: Optional[str] = Query(None, description="Filter by 'facebook' or 'email'"),
    client_id: Optional[str] = Query(None, description="Filter by client_id"),
    fanpage_id: Optional[int] = Query(None, description="Filter by fanpage_id"),
    category: Optional[str] = Query(None, description="Filter by category tag"),
    format: Optional[str] = Query(None, description="Filter by format tag"),
    is_online: Optional[bool] = Query(None, description="Filter by online/offline mode"),
    q: Optional[str] = Query(None, description="Search title, description, organizer, or location"),
    limit: int = Query(100, ge=1, le=500),
):
    """Return deduplicated, classified & updated events from `webapp.db` (Admin only)."""
    items = query_filtered_events(
        source_type=source_type,
        client_id=client_id,
        fanpage_id=fanpage_id,
        category=category,
        format_tag=format,
        is_online=is_online,
        q=q,
        limit=limit,
    )
    return {
        "count": len(items),
        "events": items,
    }


@router.get("/api/fanpages")
def api_list_fanpages():
    rows = list_fanpages(active_only=False)
    return [{**fp.to_dict(), "post_count": count} for fp, count in rows]


@router.get("/api/posts")
def api_list_all_posts(limit: int = Query(50, ge=1, le=200)):
    """Return latest scraped posts across all tracked Facebook fanpages (Admin only)."""
    posts = get_all_recent_posts(limit=limit)
    return {
        "count": len(posts),
        "posts": [p.to_dict() for p in posts],
    }


@router.post("/api/scrape")
async def api_run_scrape_now(target: Optional[str] = Query(None, description="Optional fanpage slug/ID/URL")):
    """Unified scrape trigger (Admin only)."""
    cfg = sync_settings_from_server_module()
    use_actions_env = os.environ.get("USE_GITHUB_ACTIONS_CRAWLER", "").strip().lower() in ("1", "true", "yes")
    if not cfg.auto_scrape or use_actions_env:
        return await trigger_unified_scrape(target_slug_or_id=target)

    async with manager.scrape_lock:
        await manager.broadcast_to_all({"event": "status", "message": "Scraping fanpages for new posts..."})
        new_count = await run_scrape_cycle(
            target_slug_or_id=target,
            on_new_posts=manager.on_new_posts_scraped,
            on_events_updated=manager.on_events_updated,
        )
        await asyncio.to_thread(backup_state_to_git, "API VM crawl")
        await manager.broadcast_to_all(
            {
                "event": "status",
                "message": f"Scrape finished! Found {new_count} new post(s).",
                "refresh_tree": True,
            }
        )
    return {"status": "ok", "mode": "local", "new_posts_count": new_count}


@router.post("/api/extract-post")
async def api_extract_single_post(body: ExtractPostRequest):
    """Fetch a single Facebook post URL, feed it to the extractor, and add its fanpage if event (Admin only)."""
    return await process_single_post_url(body.url)


@router.get("/api/users/{client_id}/subscriptions")
def api_get_user_subscriptions(client_id: str):
    get_or_create_user(client_id)
    subs = get_user_subscriptions(client_id)
    return {
        "client_id": client_id,
        "rss_url": f"/users/{client_id}/rss.xml",
        "subscriptions": [fp.to_dict() for fp in subs],
    }


@router.post("/api/users/{client_id}/subscriptions")
async def api_subscribe_user(client_id: str, body: SubscribeRequest):
    try:
        subs, needs_scrape = subscribe_user_to_fanpages(client_id, body.urls)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    generate_user_rss_xml(client_id)

    posts = get_posts_for_user(client_id, limit=50)
    await manager.send_to_user(
        client_id,
        {
            "event": "subscriptions_updated",
            "subscriptions": [fp.to_dict() for fp in subs],
            "posts": [p.to_dict() for p in posts],
        },
    )

    for fp in needs_scrape:
        await trigger_unified_scrape(fp.slug)

    return {
        "client_id": client_id,
        "subscriptions": [fp.to_dict() for fp in subs],
        "scraping_queued": [fp.url for fp in needs_scrape],
    }


@router.delete("/api/users/{client_id}/subscriptions")
async def api_unsubscribe_user(client_id: str, url: str = Query(..., description="Fanpage URL, slug, or ID")):
    removed = unsubscribe_user_from_fanpage(client_id, url)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Subscription '{url}' not found for user '{client_id}'.")

    subs = get_user_subscriptions(client_id)
    posts = get_posts_for_user(client_id, limit=50)
    generate_user_rss_xml(client_id)

    await manager.send_to_user(
        client_id,
        {
            "event": "subscriptions_updated",
            "subscriptions": [fp.to_dict() for fp in subs],
            "posts": [p.to_dict() for p in posts],
        },
    )
    return {"client_id": client_id, "removed": removed.to_dict(), "subscriptions": [fp.to_dict() for fp in subs]}


@router.get("/api/users/{client_id}/posts")
def api_get_user_posts(client_id: str, limit: int = Query(50, ge=1, le=200)):
    posts = get_posts_for_user(client_id, limit=limit)
    return {
        "client_id": client_id,
        "count": len(posts),
        "posts": [p.to_dict() for p in posts],
    }


@router.get("/users/{client_id}/rss.xml")
def api_get_user_rss(client_id: str):
    """Return personalized RSS 2.0 XML feed (Admin only)."""
    xml_content = generate_user_rss_xml(client_id)
    return Response(content=xml_content, media_type="application/rss+xml; charset=utf-8")


@router.get("/feeds")
@router.get("/feeds/")
def serve_feeds_root():
    """Return the JSON directory tree index of `/feeds/` (Admin only)."""
    return build_feeds_directory_tree()


@router.get("/feeds/{file_path:path}")
def serve_feed_file(file_path: str):
    """Serve distributed `.json` and `.xml` files from `feeds/` (Admin only)."""
    base = FEEDS_DIR.resolve()
    target = (FEEDS_DIR / file_path).resolve()
    if base not in target.parents and target != base:
        raise HTTPException(status_code=403, detail="Access denied.")

    if target.is_dir():
        files = [
            inspect_feed_file(p)
            for p in sorted(target.rglob("*"))
            if p.is_file() and p.suffix.lower() in (".json", ".xml")
        ]
        return JSONResponse(
            content={
                "folder": f"/feeds/{file_path.strip('/')}/",
                "count": len(files),
                "files": files,
            }
        )

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Feed file '{file_path}' not found.")

    if target.suffix.lower() == ".json":
        return FileResponse(target, media_type="application/json; charset=utf-8")
    return FileResponse(target, media_type="application/rss+xml; charset=utf-8")


# --- Admin WebSocket Endpoints ---


@ws_router.websocket("/ws")
async def admin_websocket_endpoint(websocket: WebSocket):
    """Real-time WebSocket endpoint for the Admin Console."""
    if not _is_ws_admin_authenticated(websocket):
        await websocket.close(code=1008)
        return

    await manager.connect("__public__", websocket)
    fanpages = [{**fp.to_dict(), "post_count": count} for fp, count in list_fanpages(active_only=False)]
    posts = get_all_recent_posts(limit=50)
    await websocket.send_json(
        {
            "event": "init",
            "fanpages": fanpages,
            "posts": [p.to_dict() for p in posts],
        }
    )

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("action") == "refresh":
                res = await trigger_unified_scrape()
                if res.get("status") == "busy":
                    await websocket.send_json({"event": "status", "message": res["message"]})
    except WebSocketDisconnect:
        manager.disconnect("__public__", websocket)
    except Exception:
        manager.disconnect("__public__", websocket)


@ws_router.websocket("/ws/{client_id}")
async def admin_client_websocket_endpoint(websocket: WebSocket, client_id: str):
    """Admin-scoped real-time WebSocket endpoint for inspecting/managing `client_id` feeds."""
    if not _is_ws_admin_authenticated(websocket):
        await websocket.close(code=1008)
        return

    get_or_create_user(client_id)
    await manager.connect(client_id, websocket)

    subs = get_user_subscriptions(client_id)
    posts = get_posts_for_user(client_id, limit=50)
    await websocket.send_json(
        {
            "event": "init",
            "client_id": client_id,
            "rss_url": f"/users/{client_id}/rss.xml",
            "subscriptions": [fp.to_dict() for fp in subs],
            "posts": [p.to_dict() for p in posts],
        }
    )

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "subscribe":
                raw_urls = data.get("urls") or []
                if isinstance(raw_urls, str):
                    raw_urls = [raw_urls]
                try:
                    subs, needs_scrape = subscribe_user_to_fanpages(client_id, raw_urls)
                    generate_user_rss_xml(client_id)
                    posts = get_posts_for_user(client_id, limit=50)
                    await manager.send_to_user(
                        client_id,
                        {
                            "event": "subscriptions_updated",
                            "subscriptions": [fp.to_dict() for fp in subs],
                            "posts": [p.to_dict() for p in posts],
                        },
                    )
                    for fp in needs_scrape:
                        await trigger_unified_scrape(fp.slug)
                except ValueError as exc:
                    await websocket.send_json({"event": "error", "message": str(exc)})

            elif action == "unsubscribe":
                target_url = data.get("url", "")
                unsubscribe_user_from_fanpage(client_id, target_url)
                generate_user_rss_xml(client_id)
                subs = get_user_subscriptions(client_id)
                posts = get_posts_for_user(client_id, limit=50)
                await manager.send_to_user(
                    client_id,
                    {
                        "event": "subscriptions_updated",
                        "subscriptions": [fp.to_dict() for fp in subs],
                        "posts": [p.to_dict() for p in posts],
                    },
                )

            elif action == "refresh":
                res = await trigger_unified_scrape()
                if res.get("status") == "busy":
                    await websocket.send_json({"event": "status", "message": res["message"]})

    except WebSocketDisconnect:
        manager.disconnect(client_id, websocket)
    except Exception:
        manager.disconnect(client_id, websocket)
