"""MVC Controller layer: FastAPI routers and runtime orchestration."""
from .admin_controller import router as admin_router
from .auth_controller import router as auth_router
from .event_controller import router as event_router
from .event_controller import ws_router
from .page_controller import router as page_router
from .runtime import (
    ConnectionManager,
    RuntimeSettings,
    manager,
    process_single_post_url,
    runtime_settings,
    scheduled_scrape_task,
    scheduler,
    sync_settings_from_server_module,
    trigger_background_scrape,
    trigger_github_actions_crawl,
    trigger_unified_scrape,
)

__all__ = [
    "ConnectionManager",
    "RuntimeSettings",
    "admin_router",
    "auth_router",
    "event_router",
    "manager",
    "page_router",
    "process_single_post_url",
    "runtime_settings",
    "scheduled_scrape_task",
    "scheduler",
    "sync_settings_from_server_module",
    "trigger_background_scrape",
    "trigger_github_actions_crawl",
    "trigger_unified_scrape",
    "ws_router",
]
