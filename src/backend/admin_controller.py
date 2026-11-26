"""Admin session & management controller (`/api/admin/*`)."""
import asyncio
from contextlib import closing
from datetime import datetime
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response

from config import FEEDS_DIR, SCROLL_COUNT, WEBAPP_DB_PATH
from events import analyze_fanpage_posts, purge_fanpage_events, reclassify_all_events, sync_all_event_folders
from scraper import is_facebook_post_url, scrape_fanpage
from src.backend.runtime import manager, process_single_post_url, trigger_unified_scrape
from src.model import (
    AddFanpageRequest,
    AdminLoginRequest,
    ClassifyRequest,
    ImportCrawlRequest,
    RefetchExtractRequest,
    add_fanpage,
    count_fanpage_events_file,
    end_admin_session,
    ensure_local_tz,
    init_db,
    is_admin_authenticated,
    is_admin_configured,
    list_all_event_subscriptions,
    list_fanpages,
    list_users,
    remove_fanpage,
    require_admin_session,
    save_posts,
    start_admin_session,
    update_fanpage_metadata,
    verify_admin_token_value,
)

router = APIRouter(tags=["admin"])


def require_admin(request: Request, x_admin_token: str = Header(default="")) -> None:
    require_admin_session(request, x_admin_token=x_admin_token)


@router.get("/api/admin/session")
def admin_session_status(request: Request):
    """Check whether server ADMIN_TOKEN is configured and if the current cookie/session is authenticated."""
    return {
        "configured": is_admin_configured(),
        "authenticated": is_admin_authenticated(request),
    }


@router.post("/api/admin/login")
def admin_login(body: AdminLoginRequest, response: Response):
    """Authenticate an admin session via ADMIN_TOKEN and set an HTTP-only session cookie."""
    if not is_admin_configured():
        raise HTTPException(status_code=503, detail="Admin is disabled: ADMIN_TOKEN is not configured on the server.")
    if not verify_admin_token_value(body.token):
        raise HTTPException(status_code=401, detail="Invalid admin token.")
    start_admin_session(response)
    return {"ok": True, "role": "admin", "authenticated": True}


@router.post("/api/admin/logout")
def admin_logout(response: Response):
    """Clear the admin session cookie."""
    end_admin_session(response)
    return {"ok": True, "authenticated": False}


def _create_webapp_db_snapshot_bytes() -> bytes:
    """Create a consistent SQLite online backup of `webapp.db`, read its bytes, and clean up the temp file."""
    init_db()
    if not WEBAPP_DB_PATH.is_file():
        raise FileNotFoundError("webapp.db does not exist on the server.")
    fd, tmp_str = tempfile.mkstemp(prefix="webapp_snapshot_", suffix=".db")
    os.close(fd)
    tmp_path = Path(tmp_str)
    try:
        with closing(sqlite3.connect(str(WEBAPP_DB_PATH))) as src_conn, closing(
            sqlite3.connect(str(tmp_path))
        ) as dst_conn:
            src_conn.backup(dst_conn)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/api/admin/download-webapp-db", dependencies=[Depends(require_admin)])
async def admin_download_webapp_db():
    """Download a consistent snapshot of the current `webapp.db` SQLite database."""
    try:
        db_bytes = await asyncio.to_thread(_create_webapp_db_snapshot_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to snapshot webapp.db: {exc}") from exc

    return Response(
        content=db_bytes,
        media_type="application/vnd.sqlite3",
        headers={
            "Content-Disposition": 'attachment; filename="webapp.db"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/api/admin/verify", dependencies=[Depends(require_admin)])
def admin_verify_token():
    """Verify admin session or X-Admin-Token."""
    return {"ok": True}


@router.post("/api/admin/classify", dependencies=[Depends(require_admin)])
async def admin_classify_events(body: Optional[ClassifyRequest] = None):
    """Run the classification module on all (or selected) events in the database."""
    event_ids = body.event_ids if body else None
    result = await asyncio.to_thread(reclassify_all_events, event_ids)
    await asyncio.to_thread(sync_all_event_folders)
    return {"status": "ok", **result}


@router.get("/api/admin/users", dependencies=[Depends(require_admin)])
def admin_list_users():
    """List registered client users with event subscription counts and total credits."""
    users = list_users()
    return {"count": len(users), "users": users}


@router.get("/api/admin/subscriptions", dependencies=[Depends(require_admin)])
def admin_list_subscriptions(limit: int = Query(200, ge=1, le=1000)):
    """List event subscriptions across all users."""
    subs = list_all_event_subscriptions(limit=limit)
    return {"count": len(subs), "subscriptions": subs}


@router.post("/api/admin/scrape", dependencies=[Depends(require_admin)])
async def admin_trigger_scrape(target: Optional[str] = Query(None, description="Optional fanpage slug/ID/URL")):
    """Unified scrape trigger for admin."""
    return await trigger_unified_scrape(target_slug_or_id=target)


@router.post("/api/admin/refetch-extract", dependencies=[Depends(require_admin)])
async def admin_refetch_and_extract(body: RefetchExtractRequest):
    """Refetch the N latest posts from a selected fanpage (or all fanpages) and refeed to Event Extractor."""
    rows = list_fanpages(active_only=True)
    target = (body.fanpage or "all").strip()
    if target and target.lower() != "all":
        exact_rows = [
            (fp, cnt)
            for fp, cnt in rows
            if str(fp.id) == target
            or fp.slug.lower() == target.lower()
            or fp.url.lower().rstrip("/") == target.lower().rstrip("/")
        ]
        rows = exact_rows or [
            (fp, cnt)
            for fp, cnt in rows
            if not target.isdigit() and target.lower() in fp.url.lower()
        ]
    if not rows:
        raise HTTPException(status_code=404, detail=f"No active fanpage matching '{target}'.")

    scroll_count = max(SCROLL_COUNT, min(12, (body.limit // 3) + 2))
    total_fetched = 0
    total_new = 0
    total_events = 0
    total_llm = 0
    total_updated = 0
    total_errors = 0
    error_details: List[str] = []

    async with manager.scrape_lock:
        await manager.broadcast_to_all(
            {
                "event": "status",
                "message": f"Refetching up to {body.limit} latest post(s) for {len(rows)} fanpage(s)...",
            }
        )
        for idx, (fp, _) in enumerate(rows):
            try:
                page_title, scraped_items = await scrape_fanpage(fp.url, scroll_count=scroll_count)
                latest_items = scraped_items[: body.limit]
                total_fetched += len(latest_items)

                resolved_name = update_fanpage_metadata(fp.id, name=page_title)
                if resolved_name:
                    fp.name = resolved_name

                new_posts = save_posts(fp.id, latest_items)
                total_new += len(new_posts)

                ev_stats = await asyncio.to_thread(
                    analyze_fanpage_posts,
                    fp.id,
                    limit=body.limit,
                    force_recheck=body.force_recheck,
                )
                total_events += int(ev_stats.get("events", 0))
                total_llm += int(ev_stats.get("llm_calls", 0))
                total_updated += int(ev_stats.get("updated", 0))
                ev_err_count = int(ev_stats.get("errors", 0))
                total_errors += ev_err_count
                if ev_err_count > 0:
                    error_details.append(f"{fp.slug}: {ev_err_count} LLM/extraction error(s)")

                if new_posts:
                    await manager.on_new_posts_scraped(fp, new_posts)
                if ev_stats.get("updated_events"):
                    await manager.on_events_updated(fp, ev_stats["updated_events"])
            except Exception as exc:
                total_errors += 1
                error_details.append(f"{fp.slug}: {exc}")
                print(f"[admin] Error refetching {fp.url}: {exc}")

            if idx < len(rows) - 1:
                await asyncio.sleep(2.5)

        await asyncio.to_thread(sync_all_event_folders)

    msg = (
        f"Refetched {total_fetched} latest post(s) across {len(rows)} fanpage(s): "
        f"+{total_new} new post(s), {total_llm} LLM call(s), {total_events} event(s) found"
        + (f", {total_updated} updated" if total_updated else "")
        + (f", {total_errors} error(s)" if total_errors else "")
        + "."
        + (f" Details: {' | '.join(error_details)}" if error_details else "")
    )
    await manager.broadcast_to_all({"event": "status", "message": msg, "refresh_tree": True})

    return {
        "status": "ok",
        "fanpages_processed": len(rows),
        "posts_fetched": total_fetched,
        "new_posts": total_new,
        "llm_calls": total_llm,
        "events_found": total_events,
        "updated": total_updated,
        "errors": total_errors,
        "error_details": error_details,
        "message": msg,
    }


@router.post("/api/admin/import-crawl", dependencies=[Depends(require_admin)])
async def admin_import_crawl(body: ImportCrawlRequest):
    """Receive scraped fanpage posts pushed from external crawlers and run unified extraction."""
    total_new_posts = 0
    total_events = 0
    fanpages_synced = 0

    for fp_payload in body.fanpages:
        try:
            fp, _ = add_fanpage(fp_payload.url, fp_payload.name)
        except ValueError:
            continue

        resolved_name = update_fanpage_metadata(fp.id, name=fp_payload.name)
        if resolved_name:
            fp.name = resolved_name

        scraped_items = []
        for p in fp_payload.posts:
            pub_dt = None
            if p.published_at:
                try:
                    pub_dt = ensure_local_tz(datetime.fromisoformat(p.published_at.replace("Z", "+00:00")))
                except Exception:
                    pub_dt = None
            scraped_items.append(
                {
                    "post_id": p.post_id,
                    "post_url": p.post_url,
                    "content": p.content,
                    "media_urls": p.media_urls,
                    "published_at": pub_dt,
                }
            )

        new_posts = save_posts(fp.id, scraped_items)
        total_new_posts += len(new_posts)

        ev_stats = await asyncio.to_thread(
            analyze_fanpage_posts,
            fp.id,
            limit=body.extract_limit,
            force_recheck=body.force_recheck,
        )
        total_events += int(ev_stats.get("events", 0))
        fanpages_synced += 1

        if new_posts:
            await manager.on_new_posts_scraped(fp, new_posts)
        if ev_stats.get("updated_events"):
            await manager.on_events_updated(fp, ev_stats["updated_events"])

    await asyncio.to_thread(sync_all_event_folders)
    await manager.broadcast_to_all(
        {
            "event": "status",
            "message": f"Synced scheduled crawl: {fanpages_synced} fanpage(s), {total_new_posts} new post(s).",
            "refresh_tree": True,
        }
    )
    return {
        "status": "ok",
        "fanpages_synced": fanpages_synced,
        "new_posts": total_new_posts,
        "events_found": total_events,
    }


@router.get("/api/admin/fanpages", dependencies=[Depends(require_admin)])
def admin_list_fanpages():
    return [
        {**fp.to_dict(), "post_count": count, "event_count": count_fanpage_events_file(fp.slug)}
        for fp, count in list_fanpages(active_only=False)
    ]


@router.post("/api/admin/extract-post", dependencies=[Depends(require_admin)])
async def admin_extract_post(body: AddFanpageRequest):
    """Fetch a single Facebook post URL, feed it to the extractor, and add its fanpage if it is an event."""
    return await process_single_post_url(body.url)


@router.post("/api/admin/fanpages", dependencies=[Depends(require_admin)])
async def admin_add_fanpage(body: AddFanpageRequest):
    if is_facebook_post_url(body.url):
        res = await process_single_post_url(body.url)
        return {
            "created": bool(res.get("fanpage_added")),
            "scrape_mode": "single_post_extract",
            **res,
        }

    try:
        fp, created = add_fanpage(body.url, body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    scrape_mode = "already_tracked"
    message = "Fanpage already tracked."
    if created:
        res = await trigger_unified_scrape(fp.slug)
        scrape_mode = res.get("mode", "queued_for_actions")
        message = f"Added. {res.get('message', '')}".strip()
        await manager.broadcast_to_all(
            {"event": "status", "message": f"Added fanpage: {fp.name or fp.slug}", "refresh_tree": True}
        )

    return {"created": created, "scrape_mode": scrape_mode, "message": message, "fanpage": fp.to_dict()}


@router.delete("/api/admin/fanpages", dependencies=[Depends(require_admin)])
async def admin_remove_fanpage(identifier: str = Query(..., description="Fanpage id, slug or URL")):
    removed = remove_fanpage(identifier)
    if not removed:
        raise HTTPException(status_code=404, detail="Fanpage not found.")
    (FEEDS_DIR / f"{removed.slug}.xml").unlink(missing_ok=True)
    purge_fanpage_events(removed.id, removed.slug, fanpage_name=removed.name)
    await manager.broadcast_to_all(
        {"event": "status", "message": f"Removed fanpage: {removed.name or removed.slug}", "refresh_tree": True}
    )
    return {"removed": removed.to_dict()}
