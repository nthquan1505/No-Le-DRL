"""Authentication & session controller for client user Sign In / Sign Up and session inspection."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from src.model import (
    User,
    UserLoginRequest,
    UserRegisterRequest,
    authenticate_user_credentials,
    end_user_session,
    get_current_user_optional,
    is_admin_authenticated,
    is_admin_configured,
    register_user_account,
    start_user_session,
)

router = APIRouter(tags=["auth"])


@router.post("/api/auth/register")
@router.post("/api/auth/signup")
def api_auth_register(body: UserRegisterRequest, response: Response):
    """Register a new client user account and start an HTTP-only user session."""
    try:
        user = register_user_account(
            username=body.username,
            password=body.password,
            start_year=body.start_year,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    start_user_session(response, user)
    return {"ok": True, "role": "user", "user": user.to_dict()}


@router.post("/api/auth/login")
@router.post("/api/auth/signin")
def api_auth_login(body: UserLoginRequest, response: Response):
    """Authenticate a client user and set an HTTP-only user session cookie."""
    user = authenticate_user_credentials(body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    start_user_session(response, user)
    return {"ok": True, "role": "user", "user": user.to_dict()}


@router.post("/api/auth/logout")
def api_auth_logout(response: Response):
    """End the current client user session."""
    end_user_session(response)
    return {"ok": True}


@router.get("/api/auth/me")
def api_auth_me(user: Optional[User] = Depends(get_current_user_optional)):
    """Return the currently authenticated client user, or null if not logged in."""
    if not user:
        return {"authenticated": False, "user": None}
    return {"authenticated": True, "user": user.to_dict()}


@router.get("/api/auth/session")
def api_auth_session_overview(
    request: Request,
    user: Optional[User] = Depends(get_current_user_optional),
):
    """Return combined session status for both client user (`rss_user_session`) and admin (`rss_admin_session`)."""
    admin_auth = is_admin_authenticated(request)
    return {
        "user_authenticated": bool(user),
        "user": user.to_dict() if user else None,
        "admin_configured": is_admin_configured(),
        "admin_authenticated": admin_auth,
    }
