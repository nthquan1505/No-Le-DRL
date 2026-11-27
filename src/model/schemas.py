"""Pydantic request and response schemas for user/admin sessions, events, inbox, and admin APIs."""
from typing import List, Optional

from pydantic import BaseModel, Field


# --- Auth & Session Schemas ---


class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=64, description="Unique client username")
    password: str = Field(..., min_length=4, max_length=128, description="Plaintext password (min 4 chars)")
    start_year: int = Field(2024, ge=2000, le=2035, description="Cohort / enrollment start year")


class UserLoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class AdminLoginRequest(BaseModel):
    token: str = Field(..., min_length=1, description="Server ADMIN_TOKEN")


# --- Event & Subscription Schemas ---


class EventSubscriptionRequest(BaseModel):
    audience: str = Field("", max_length=255)
    progress: str = Field("interested", max_length=64)
    evidence: str = Field("", max_length=2000)
    credit: int = Field(0, ge=0, le=1000)


class EventSubscriptionUpdateRequest(BaseModel):
    audience: Optional[str] = Field(None, max_length=255)
    progress: Optional[str] = Field(None, max_length=64)
    evidence: Optional[str] = Field(None, max_length=2000)
    credit: Optional[int] = Field(None, ge=0, le=1000)


class SubscribeRequest(BaseModel):
    urls: List[str]


class ExtractPostRequest(BaseModel):
    url: str = Field(..., description="Public Facebook post URL to scrape and analyze")


# --- Inbox Ingestion Schemas ---


class EmailIngestRequest(BaseModel):
    client_id: str = Field("anonymous", description="Client or user identifier submitting the email")
    message_id: Optional[str] = Field(None, description="Unique Email Message-ID header or UID")
    subject: str = Field("", description="Email subject line")
    sender: Optional[str] = Field(None, description="Sender email address or display name")
    received_at: Optional[str] = Field(None, description="ISO 8601 timestamp when email was received")
    content: str = Field(..., description="Plain-text email body content")
    media_urls: List[str] = Field(default_factory=list, description="Optional poster/attachment image URLs")


class BatchEmailIngestRequest(BaseModel):
    emails: List[EmailIngestRequest]


# --- Admin Management Schemas ---


class AddFanpageRequest(BaseModel):
    url: str
    name: Optional[str] = None


class CrawledPostItem(BaseModel):
    post_id: str
    post_url: str = ""
    content: str = ""
    media_urls: List[str] = Field(default_factory=list)
    published_at: Optional[str] = None


class CrawledFanpagePayload(BaseModel):
    url: str
    name: Optional[str] = None
    posts: List[CrawledPostItem] = Field(default_factory=list)


class ImportCrawlRequest(BaseModel):
    fanpages: List[CrawledFanpagePayload] = Field(default_factory=list)
    extract_limit: Optional[int] = None
    force_recheck: bool = False


class RefetchExtractRequest(BaseModel):
    fanpage: str = Field("all", description="Fanpage ID, slug, URL, or 'all'")
    limit: int = Field(5, ge=1, le=50, description="Number of latest posts per fanpage to refetch & extract")
    force_recheck: bool = Field(True, description="Re-evaluate posts previously marked non-event within the limit")


class ClassifyRequest(BaseModel):
    event_ids: Optional[List[int]] = Field(None, description="Optional list of event_ids to reclassify; omit for all")
