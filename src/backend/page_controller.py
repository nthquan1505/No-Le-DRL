"""Page view controller: `/` serves Login first, `/client` serves blank client page, `/admin` serves full Admin Console."""
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from src.model import get_current_user_optional, is_admin_authenticated
from src.frontend import render_view

router = APIRouter(tags=["pages"])


@router.get("/")
def serve_root_login(request: Request):
    """Entrypoint: redirect active sessions to their role workspace, otherwise render Login page."""
    if is_admin_authenticated(request):
        return RedirectResponse(url="/admin", status_code=302)
    if get_current_user_optional(request) is not None:
        return RedirectResponse(url="/client", status_code=302)
    return render_view("login.html")


@router.get("/login")
@router.get("/signin")
@router.get("/register")
def serve_login_page():
    """Render the User Sign In / Sign Up & Admin Login page."""
    return render_view("login.html")


@router.get("/client")
def serve_client_page(request: Request):
    """Render the blank Client workspace (requires an active user session)."""
    if get_current_user_optional(request) is None:
        return RedirectResponse(url="/login", status_code=302)
    return render_view("client.html")


@router.get("/admin")
def serve_admin_page(request: Request):
    """Render the full Admin Console with all current functions (requires an active admin session)."""
    if not is_admin_authenticated(request):
        return RedirectResponse(url="/login?tab=admin", status_code=302)
    return render_view("admin.html")
