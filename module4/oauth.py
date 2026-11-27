"""OAuth setup and service loading for Google Calendar."""

import argparse
import os
from pathlib import Path
import tempfile

from .calendar import CalendarReadError


CALENDAR_SCOPES = ("https://www.googleapis.com/auth/calendar.events",)


def token_path():
    configured = os.getenv("MODULE4_CALENDAR_TOKEN")
    return Path(configured).expanduser() if configured else Path.home() / ".no-le-drl" / "calendar-token.json"


def _save_token(path, credentials):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, encoding="utf-8", mode="w", delete=False) as file:
        temporary = Path(file.name)
        file.write(credentials.to_json())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def connect(client_secrets, token=None):
    from google_auth_oauthlib.flow import InstalledAppFlow

    path = Path(token) if token else token_path()
    flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), scopes=CALENDAR_SCOPES)
    credentials = flow.run_local_server(host="localhost", port=0)
    _save_token(path, credentials)
    return path


def load_calendar_service(token=None):
    """Load a separate account token supplied by the caller, refreshing when needed."""
    from google.auth.transport.requests import Request
    from google.auth.exceptions import GoogleAuthError
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    path = Path(token) if token else token_path()
    try:
        credentials = Credentials.from_authorized_user_file(str(path))
        if not credentials.has_scopes(CALENDAR_SCOPES):
            raise ValueError("Cần quyền đọc và ghi sự kiện")
        if not credentials.valid and credentials.refresh_token:
            credentials.refresh(Request())
            _save_token(path, credentials)
        if not credentials.valid:
            raise ValueError("Token Calendar không hợp lệ")
        return build("calendar", "v3", credentials=credentials, cache_discovery=False)
    except (GoogleAuthError, OSError, ValueError) as error:
        raise CalendarReadError("Hãy kết nối lại Google Calendar") from error


def main():
    parser = argparse.ArgumentParser(description="Connect a Google Calendar account to Module 4.")
    parser.add_argument("--client-secrets", required=True, type=Path)
    parser.add_argument("--token", type=Path)
    args = parser.parse_args()
    connect(args.client_secrets, args.token)
    print("Google Calendar connection saved.")


if __name__ == "__main__":
    main()
