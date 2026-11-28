"""Data contracts and controlled vocabularies for `[EXT]` (`src/extraction`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence
import unicodedata
from zoneinfo import ZoneInfo

from database import now_iso_utc7, to_iso_utc7

try:
    UTC7 = ZoneInfo("Asia/Ho_Chi_Minh")
except Exception:
    UTC7 = timezone(timedelta(hours=7))

SourceChannel = Literal["facebook", "email", "manual"]
EventClassification = Literal[
    "new_or_active_event",
    "event_update",
    "event_cancellation",
    "not_event",
]
ExtractionMethod = Literal["prefilter", "gemini", "groq", "heuristic"]
PersistenceAction = Literal[
    "created",
    "updated",
    "cancelled",
    "duplicate",
    "ignored",
    "skipped_seen",
    "error",
]

# Controlled vocabularies aligned with downstream CLF, CRD, and RCM modules
ALLOWED_TOPICS: tuple[str, ...] = (
    "học thuật",
    "khoa học",
    "lập trình",
    "AI",
    "hướng nghiệp",
    "kỹ năng",
    "cộng đồng",
    "thiện nguyện",
    "quyên góp",
    "thể thao",
    "văn nghệ",
    "pháp luật",
    "chính trị - xã hội",
    "hội nhập",
    "khởi nghiệp",
)

ALLOWED_FORMATS: tuple[str, ...] = (
    "workshop",
    "seminar",
    "talkshow",
    "tọa đàm",
    "hội nghị",
    "cuộc thi",
    "ngày hội",
    "tình nguyện",
    "hiến máu",
    "khảo sát",
    "triển lãm",
    "tham quan",
    "tuyển cộng tác viên",
    "online",
    "offline",
    "cá nhân",
    "đồng đội",
)


def coerce_utc7_datetime(value: Any) -> datetime:
    """Coerce datetime, ISO string, or epoch into a timezone-aware UTC+7 `datetime`."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC7)
        return value.astimezone(UTC7)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC7)
    if isinstance(value, str) and value.strip():
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC7)
            return dt.astimezone(UTC7)
        except ValueError:
            pass
    return datetime.now(UTC7)


def _clean_str_list(items: Any) -> List[str]:
    if not items:
        return []
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, Sequence):
        return []
    seen: set[str] = set()
    out: List[str] = []
    for raw in items:
        text = unicodedata.normalize("NFC", str(raw or "")).strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out


@dataclass(frozen=True)
class RawIngestionItem:
    """Channel-agnostic raw announcement passed from `ING` into `EXT`."""

    source_type: SourceChannel
    source_id: str
    source_name: str
    content: str
    published_at: datetime = field(default_factory=lambda: datetime.now(UTC7))
    subject: str = ""
    post_url: Optional[str] = None
    images_url: List[str] = field(default_factory=list)
    fanpage_id: Optional[int] = None

    @property
    def published_at_iso(self) -> str:
        return coerce_utc7_datetime(self.published_at).isoformat()

    @property
    def full_text(self) -> str:
        subj = unicodedata.normalize("NFC", (self.subject or "").strip())
        body = unicodedata.normalize("NFC", (self.content or "").strip())
        if subj and subj.lower() not in body.lower():
            return f"{subj}\n{body}".strip()
        return body

    @property
    def content_hash(self) -> str:
        payload = f"{self.source_type}|{self.source_id}|{self.full_text}|{'|'.join(self.images_url)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_scraped_post(
        cls,
        post: Mapping[str, Any],
        fanpage_name: str = "",
        fanpage_id: Optional[int] = None,
    ) -> "RawIngestionItem":
        raw_url = str(post.get("post_url") or "").strip() or None
        raw_id = str(post.get("post_id") or raw_url or "").strip()
        if not raw_id:
            raw_id = hashlib.sha1(str(post.get("content") or "").encode("utf-8")).hexdigest()[:16]
        source_label = (
            str(fanpage_name or post.get("fanpage_name") or post.get("author_name") or "").strip()
            or "Facebook Fanpage"
        )
        media = post.get("media_urls") or post.get("images_url") or []
        return cls(
            source_type="facebook",
            source_id=raw_id,
            source_name=source_label,
            content=unicodedata.normalize("NFC", str(post.get("content") or "")).strip(),
            published_at=coerce_utc7_datetime(post.get("published_at")),
            subject="",
            post_url=raw_url,
            images_url=_clean_str_list(media),
            fanpage_id=fanpage_id if fanpage_id is not None else post.get("fanpage_id"),
        )

    @classmethod
    def from_email(
        cls,
        content: str,
        subject: str = "",
        sender: str = "",
        message_id: Optional[str] = None,
        received_at: Optional[Any] = None,
        images_url: Optional[Sequence[str]] = None,
    ) -> "RawIngestionItem":
        norm_body = unicodedata.normalize("NFC", content or "").strip()
        norm_subj = unicodedata.normalize("NFC", subject or "").strip()
        msg_id = (message_id or "").strip()
        if not msg_id:
            digest = hashlib.sha1(f"{sender}|{norm_subj}|{norm_body}".encode("utf-8")).hexdigest()[:16]
            msg_id = f"email_{digest}"
        return cls(
            source_type="email",
            source_id=msg_id,
            source_name=(sender or "").strip() or "Email Inbox",
            content=norm_body,
            published_at=coerce_utc7_datetime(received_at),
            subject=norm_subj,
            post_url=None,
            images_url=_clean_str_list(images_url or []),
            fanpage_id=None,
        )


@dataclass
class PreFilterDecision:
    """Result of Stage-1 deterministic pre-filtering."""

    is_candidate: bool
    reason: str
    matched_signals: List[str] = field(default_factory=list)
    is_lifecycle_signal: bool = False


@dataclass
class ExtractedEventDraft:
    """Normalized extraction output ready for enrichment and DB deduplication."""

    source_id: str
    classification: EventClassification
    reason: str = ""
    title: str = ""
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    is_online: bool = False
    reg_start_datetime: Optional[str] = None
    reg_end_datetime: Optional[str] = None
    event_start_datetime: Optional[str] = None
    event_end_datetime: Optional[str] = None
    location: str = ""
    coordinate: Optional[str] = None
    fee: str = ""
    target_audience: List[str] = field(default_factory=list)
    topic: List[str] = field(default_factory=list)
    format: List[str] = field(default_factory=list)
    award: str = ""
    organizer: str = ""
    registration_url: Optional[str] = None
    sources: str = ""
    post_url: Optional[str] = None
    images_url: List[str] = field(default_factory=list)
    extraction_method: ExtractionMethod = "heuristic"

    @property
    def is_actionable_event(self) -> bool:
        return self.classification in (
            "new_or_active_event",
            "event_update",
            "event_cancellation",
        )

    def to_event_db_payload(self, is_cancelled: bool = False) -> Dict[str, Any]:
        """Build a clean column dict for `events` in `drl_database.db` (`category` & `min_credit` left None)."""
        now = now_iso_utc7()
        return {
            "title": self.title.strip(),
            "description": self.description.strip(),
            "keywords": _clean_str_list(self.keywords),
            "is_online": bool(self.is_online),
            "is_cancelled": bool(is_cancelled),
            "reg_start_datetime": to_iso_utc7(self.reg_start_datetime),
            "reg_end_datetime": to_iso_utc7(self.reg_end_datetime),
            "event_start_datetime": to_iso_utc7(self.event_start_datetime),
            "event_end_datetime": to_iso_utc7(self.event_end_datetime),
            "location": self.location.strip(),
            "coordinate": self.coordinate,
            "fee": self.fee.strip(),
            "target_audience": _clean_str_list(self.target_audience),
            "topic": _clean_str_list(self.topic),
            "format": _clean_str_list(self.format),
            "award": self.award.strip(),
            "registration_url": self.registration_url,
            "created_at": now,
            "updated_at": now,
            "sources": self.sources.strip(),
            "post_url": self.post_url,
            "images_url": _clean_str_list(self.images_url),
            "category": None,
            "min_credit": None,
        }


@dataclass
class ItemExtractionResult:
    """Outcome for a single ingested post or email."""

    source_id: str
    action: PersistenceAction
    classification: EventClassification
    extraction_method: ExtractionMethod
    reason: str
    event_id: Optional[int] = None
    changed_fields: List[str] = field(default_factory=list)
    notifications_created: int = 0
    qr_decoded_url: Optional[str] = None
    coordinate_reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BatchExtractionSummary:
    """Aggregated outcome across a batch of ingested items."""

    total_items: int = 0
    prefilter_rejected: int = 0
    skipped_seen: int = 0
    llm_candidates: int = 0
    created: int = 0
    updated: int = 0
    cancelled: int = 0
    duplicates: int = 0
    ignored_not_event: int = 0
    errors: int = 0
    results: List[ItemExtractionResult] = field(default_factory=list)
