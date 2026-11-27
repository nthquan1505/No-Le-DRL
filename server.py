"""FastAPI MVC Web Server entrypoint (`src.model`, `src.frontend`, `src.backend`)."""
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import (
    DEFAULT_SCRAPE_HOURS,
    DEFAULT_SCRAPE_MINUTE,
    SCRAPE_MISFIRE_GRACE_SECONDS,
)
from events import sync_all_event_folders
from scheduler import build_daily_cron_trigger
from src.backend import (
    admin_router,
    auth_router,
    event_router,
    manager,
    page_router,
    process_single_post_url,
    scheduled_scrape_task,
    scheduler,
    sync_settings_from_server_module,
    trigger_background_scrape,
    trigger_unified_scrape,
    ws_router,
)
from src.model import init_db

# Runtime schedule globals (configurable via `main.py serve`)
SERVER_INTERVAL_SECONDS: Optional[int] = None
SERVER_SCRAPE_HOURS: str = DEFAULT_SCRAPE_HOURS
SERVER_SCRAPE_MINUTE: int = DEFAULT_SCRAPE_MINUTE
SERVER_AUTO_SCRAPE: bool = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    await asyncio.to_thread(sync_all_event_folders)
    cfg = sync_settings_from_server_module()
    if cfg.auto_scrape:
        if cfg.interval_seconds is not None:
            scheduler.add_job(
                scheduled_scrape_task,
                trigger="interval",
                seconds=cfg.interval_seconds,
                id="server_scheduled_scrape",
                max_instances=1,
                coalesce=True,
                misfire_grace_time=SCRAPE_MISFIRE_GRACE_SECONDS,
            )
        else:
            scheduler.add_job(
                scheduled_scrape_task,
                trigger=build_daily_cron_trigger(hours_str=cfg.scrape_hours, minute=cfg.scrape_minute),
                id="server_scheduled_scrape",
                max_instances=1,
                coalesce=True,
                misfire_grace_time=SCRAPE_MISFIRE_GRACE_SECONDS,
            )
        scheduler.start()
    yield
    if scheduler.running:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="Facebook Fanpage & Email Event Extractor Server (MVC)",
    description=(
        "MVC FastAPI Web App with session-authenticated User & Admin portals, "
        "unified event extraction + classification, and JSON/XML feed distribution."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Ensure API and dashboard endpoints always return valid JSON on unexpected errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )


app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(event_router)
app.include_router(ws_router)
app.include_router(page_router)

__all__ = [
    "SERVER_AUTO_SCRAPE",
    "SERVER_INTERVAL_SECONDS",
    "SERVER_SCRAPE_HOURS",
    "SERVER_SCRAPE_MINUTE",
    "app",
    "manager",
    "process_single_post_url",
    "scheduled_scrape_task",
    "scheduler",
    "trigger_background_scrape",
    "trigger_unified_scrape",
]
