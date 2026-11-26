from datetime import datetime
import json
import re
from urllib.parse import parse_qs, urlparse
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, synonym, validates
from sqlalchemy.types import TypeDecorator

from config import LOCAL_TIMEZONE


def local_now() -> datetime:
    return datetime.now(LOCAL_TIMEZONE)


def local_now_iso() -> str:
    return local_now().isoformat()


def ensure_local_tz(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=LOCAL_TIMEZONE)
    return dt.astimezone(LOCAL_TIMEZONE)


def _to_iso_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, datetime):
        dt = ensure_local_tz(val)
        return dt.isoformat() if dt else None
    text = str(val).strip()
    return text or None


class StringListType(TypeDecorator):
    """SQLite Text column that transparently serializes/deserializes a list of strings."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Optional[List[str]], dialect: Any) -> str:
        if not value:
            return "[]"
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return json.dumps([str(x).strip() for x in parsed if str(x).strip()], ensure_ascii=False)
            except Exception:
                return json.dumps([value.strip()] if value.strip() else [], ensure_ascii=False)
        return json.dumps([str(x).strip() for x in value if str(x).strip()], ensure_ascii=False)

    def process_result_value(self, value: Optional[str], dialect: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(x) for x in value]
        try:
            parsed = json.loads(value)
            return [str(x) for x in parsed] if isinstance(parsed, list) else []
        except Exception:
            return []


class Base(DeclarativeBase):
    """Ingestion & RSS cache tables stored in rss_bot.db."""
    pass


class WebAppBase(DeclarativeBase):
    """Web app tables (events, users, event_subscriptions, fanpages) stored in webapp.db."""
    pass


def _derive_slug_from_fb_url(raw_url: str) -> str:
    """Derive a deterministic file-safe slug from a Facebook fanpage URL."""
    u = (raw_url or "").strip()
    if not u:
        return "unknown"
    if not u.startswith(("http://", "https://")):
        if "facebook.com" in u or "fb.com" in u:
            u = "https://" + u
        else:
            return re.sub(r"[^a-zA-Z0-9._-]", "", u.strip("/")) or "unknown"

    parsed = urlparse(u)
    path = parsed.path.strip("/")
    if path == "profile.php":
        qs = parse_qs(parsed.query)
        page_id = qs.get("id", ["unknown"])[0]
        return f"profile_{page_id}"

    parts = [p for p in path.split("/") if p]
    if not parts:
        return "unknown"
    if parts[0] in ("pages", "people") and len(parts) >= 3:
        slug = f"{parts[1]}_{parts[2]}"
    else:
        slug = parts[0]
    return re.sub(r"[^a-zA-Z0-9._-]", "_", slug) or "unknown"


class User(WebAppBase):
    """Client user account table (`users` in webapp.db)."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(
        String(128), primary_key=True, default=lambda: uuid.uuid4().hex
    )
    username: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    password: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    start_year: Mapped[int] = mapped_column(Integer, default=2024, nullable=False)

    # Compatibility aliases
    id = synonym("user_id")
    client_id = synonym("username")

    event_subscriptions: Mapped[List["EventSubscription"]] = relationship(
        "EventSubscription", back_populates="user", cascade="all, delete-orphan"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "start_year": self.start_year,
        }


class EventRecord(WebAppBase):
    """Unified classified event table (`events` in webapp.db) with exact requested fields."""

    __tablename__ = "events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    organizer: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    is_online: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    event_start_datetime: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reg_start_datetime: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reg_end_datetime: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    event_end_datetime: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    location: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    source_type: Mapped[str] = mapped_column(String(255), default="facebook", nullable=False, index=True)
    created_at: Mapped[str] = mapped_column(String(64), default=local_now_iso, nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), default=local_now_iso, nullable=False)
    fee: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    target_audience: Mapped[List[str]] = mapped_column(StringListType, default=list, nullable=False)
    award: Mapped[str] = mapped_column(Text, default="", nullable=False)
    registration: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    category: Mapped[List[str]] = mapped_column(StringListType, default=list, nullable=False)
    format: Mapped[List[str]] = mapped_column(StringListType, default=list, nullable=False)
    post_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    id = synonym("event_id")

    subscriptions: Mapped[List["EventSubscription"]] = relationship(
        "EventSubscription", back_populates="event", cascade="all, delete-orphan"
    )

    @validates("created_at", "updated_at")
    def _coerce_iso_datetime(self, key: str, value: Any) -> str:
        return _to_iso_str(value) or local_now_iso()

    @property
    def event_date(self) -> Optional[str]:
        if self.event_start_datetime and len(self.event_start_datetime) >= 10:
            return self.event_start_datetime[:10]
        return None

    @property
    def event_time(self) -> Optional[str]:
        if self.event_start_datetime and len(self.event_start_datetime) >= 16:
            t = self.event_start_datetime[11:16]
            return t if ":" in t else None
        return None

    @property
    def start_datetime(self) -> Optional[str]:
        return self.event_start_datetime

    @property
    def end_datetime(self) -> Optional[str]:
        return self.event_end_datetime

    @property
    def online_mode(self) -> str:
        return "online" if self.is_online else "offline"

    @property
    def registration_url(self) -> Optional[str]:
        return self.registration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "description": self.description or "",
            "organizer": self.organizer or "",
            "is_online": bool(self.is_online),
            "event_start_datetime": self.event_start_datetime,
            "reg_start_datetime": self.reg_start_datetime,
            "reg_end_datetime": self.reg_end_datetime,
            "event_end_datetime": self.event_end_datetime,
            "location": self.location or "",
            "source_type": self.source_type,
            "created_at": _to_iso_str(self.created_at),
            "updated_at": _to_iso_str(self.updated_at),
            "fee": self.fee or "",
            "target_audience": self.target_audience or [],
            "award": self.award or "",
            "registration": self.registration,
            "category": self.category or [],
            "format": self.format or [],
            "post_url": self.post_url,
        }


class EventSubscription(WebAppBase):
    """Event subscription table (`event_subscriptions` in webapp.db) connecting users & events."""

    __tablename__ = "event_subscriptions"

    user_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True
    )
    event_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("events.event_id", ondelete="CASCADE"), primary_key=True
    )
    audience: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    progress: Mapped[str] = mapped_column(String(64), default="interested", nullable=False)
    evidence: Mapped[str] = mapped_column(Text, default="", nullable=False)
    credit: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="event_subscriptions")
    event: Mapped["EventRecord"] = relationship("EventRecord", back_populates="subscriptions")

    def to_dict(self, include_event: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "user_id": self.user_id,
            "event_id": self.event_id,
            "audience": self.audience,
            "progress": self.progress,
            "evidence": self.evidence,
            "credit": self.credit,
        }
        if include_event and self.event is not None:
            data["event"] = self.event.to_dict()
        return data


class Fanpage(WebAppBase):
    """Tracked Facebook fanpage list table (`fanpages` in webapp.db) for the crawler."""

    __tablename__ = "fanpages"

    fanpage_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fanpage_name: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    facebook_url: Mapped[str] = mapped_column(String(512), unique=True, nullable=False, index=True)

    # Compatibility aliases for scraper, RSS generator, and controllers
    id = synonym("fanpage_id")
    name = synonym("fanpage_name")
    url = synonym("facebook_url")

    @property
    def slug(self) -> str:
        return _derive_slug_from_fb_url(self.facebook_url)

    @property
    def is_active(self) -> bool:
        return True

    @property
    def last_checked_at(self) -> Optional[datetime]:
        return getattr(self, "_last_checked_at", None)

    @last_checked_at.setter
    def last_checked_at(self, value: Optional[datetime]) -> None:
        self._last_checked_at = value

    def to_dict(self) -> Dict[str, Any]:
        last_checked = ensure_local_tz(self.last_checked_at)
        slug = self.slug
        display_name = self.fanpage_name or slug
        return {
            "fanpage_id": self.fanpage_id,
            "fanpage_name": display_name,
            "facebook_url": self.facebook_url,
            "id": self.fanpage_id,
            "url": self.facebook_url,
            "slug": slug,
            "name": display_name,
            "is_active": True,
            "last_checked_at": last_checked.isoformat() if last_checked else None,
            "rss_file": f"feeds/{slug}.xml",
        }


class UserSubscription(Base):
    """Fanpage RSS feed subscription per client_id (stored in rss_bot.db)."""

    __tablename__ = "user_subscriptions"
    __table_args__ = (
        UniqueConstraint("client_id", "fanpage_id", name="uq_client_fanpage"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    fanpage_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=local_now)


class Post(Base):
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    fanpage_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    post_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    media_urls_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=local_now, index=True)
    scraped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=local_now)

    @property
    def fanpage(self) -> Optional["Fanpage"]:
        return getattr(self, "_fanpage", None)

    @fanpage.setter
    def fanpage(self, value: Optional["Fanpage"]) -> None:
        self._fanpage = value

    @property
    def media_urls(self) -> List[str]:
        try:
            return json.loads(self.media_urls_json)
        except Exception:
            return []

    @media_urls.setter
    def media_urls(self, urls: List[str]) -> None:
        self.media_urls_json = json.dumps(urls, ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        pub_dt = ensure_local_tz(self.published_at)
        scr_dt = ensure_local_tz(self.scraped_at)
        fp = self.fanpage
        return {
            "id": self.id,
            "post_id": self.post_id,
            "fanpage_id": self.fanpage_id,
            "fanpage_name": (fp.name or fp.slug) if fp else None,
            "fanpage_url": fp.url if fp else None,
            "post_url": self.post_url,
            "content": self.content,
            "media_urls": self.media_urls,
            "published_at": pub_dt.isoformat() if pub_dt else None,
            "scraped_at": scr_dt.isoformat() if scr_dt else None,
        }


class PostAnalysis(Base):
    """LLM event-analysis result cache so each post is analyzed only once."""

    __tablename__ = "post_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    fanpage_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    is_event: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    event_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    analyzed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=local_now)


class EmailMessage(Base):
    """Client-submitted email entries sent to the server inbox for event extraction."""

    __tablename__ = "email_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    client_id: Mapped[str] = mapped_column(String(128), default="anonymous", nullable=False, index=True)
    subject: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    sender: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    media_urls_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=local_now, index=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=local_now)
    status: Mapped[str] = mapped_column(String(64), default="pending", nullable=False)
    is_event: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    event_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    changes_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    analysis_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)

    @property
    def media_urls(self) -> List[str]:
        try:
            return json.loads(self.media_urls_json)
        except Exception:
            return []

    def to_dict(self) -> Dict[str, Any]:
        rec_dt = ensure_local_tz(self.received_at)
        ing_dt = ensure_local_tz(self.ingested_at)
        try:
            changes = json.loads(self.changes_json or "{}")
        except Exception:
            changes = {}
        try:
            analysis = json.loads(self.analysis_json or "{}")
        except Exception:
            analysis = {}
        return {
            "id": self.id,
            "message_id": self.message_id,
            "client_id": self.client_id,
            "subject": self.subject,
            "sender": self.sender,
            "content": self.content,
            "media_urls": self.media_urls,
            "received_at": rec_dt.isoformat() if rec_dt else None,
            "ingested_at": ing_dt.isoformat() if ing_dt else None,
            "status": self.status,
            "is_event": self.is_event,
            "event_id": self.event_id,
            "changes": changes,
            "event": analysis if self.is_event else None,
        }
