"""Leakage-free query text builders for HCMUS DRL event classification.

Supports 3 distinct query embedding profiles for dual-branch kNN evaluation:
  1. `title`                   : `query: {title}`
  2. `title_description`       : `query: {title} {description}` (default legacy profile)
  3. `title_description_format`: `query: {title}. Hình thức: {format_text}. {description}`

All profiles strictly exclude labels, extraction reasons, criterion IDs, group IDs,
and artificial rubric keywords to prevent data leakage.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_QUERY_PREFIX = "query: "

CLASSIFICATION_VECTOR_PROFILES: Tuple[str, ...] = (
    "title",
    "title_description",
    "title_description_format",
)

# Transport/delivery modes that belong to Schedule/Fit rather than semantic activity format
_DELIVERY_ONLY_FORMATS = frozenset({"offline", "online", "in-person", "in person", "trực tiếp", "trực tuyến"})


def normalize_event_text(value: Any) -> str:
    """Normalize Unicode NFC and collapse whitespace safely for optional/missing fields."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def normalize_string_list_field(
    value: Any,
    *,
    exclude_tokens: Sequence[str] = (),
) -> List[str]:
    """Normalize a list-or-string field (e.g., `format`, `topic`) into deduplicated clean tokens."""
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [part.strip() for part in re.split(r"[,;/|]+", value) if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        raw_items = [str(x) for x in value if x is not None]
    else:
        raw_items = [str(value)]

    excluded = {t.strip().lower() for t in exclude_tokens}
    out: List[str] = []
    seen = set()
    for item in raw_items:
        cleaned = normalize_event_text(item)
        if not cleaned:
            continue
        low = cleaned.lower()
        if low in excluded or low in seen:
            continue
        seen.add(low)
        out.append(cleaned)
    return out


def extract_activity_format_tokens(event: Dict[str, Any]) -> List[str]:
    """Extract semantic activity format tokens from `events.format` (excluding pure online/offline)."""
    raw_fmt = event.get("format")
    if raw_fmt is None and isinstance(event.get("metadata"), dict):
        raw_fmt = event["metadata"].get("format")
    return normalize_string_list_field(raw_fmt, exclude_tokens=tuple(_DELIVERY_ONLY_FORMATS))


def check_event_classifiable(event: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Determine whether an input record is a classifiable event with usable text.

    Returns:
        (True, None) if classifiable, otherwise (False, skip_reason).
    """
    if event.get("is_cancelled") is True:
        return False, "record_cancelled (is_cancelled=true)"

    is_event = event.get("is_event")
    if is_event is False:
        return False, "record_marked_not_event (is_event=false)"

    title = normalize_event_text(event.get("title"))
    description = normalize_event_text(event.get("description"))
    if not title and not description:
        return False, "missing_usable_text (both title and description empty)"

    return True, None


def build_event_query_text(
    event: Dict[str, Any],
    query_prefix: str = DEFAULT_QUERY_PREFIX,
) -> str:
    """Construct leakage-free E5 query text from event title and description only.

    Format:
        `query: [event title] [event description]`

    Preserved for full backward compatibility (profile = `title_description`).
    """
    return build_classification_profile_text(
        event,
        profile="title_description",
        query_prefix=query_prefix,
    )


def build_classification_profile_text(
    event: Dict[str, Any],
    *,
    profile: str = "title_description",
    query_prefix: str = DEFAULT_QUERY_PREFIX,
) -> str:
    """Build E5 query text for one of the 3 Classification vector profiles.

    Profiles:
      - `title`                   : `query: {title}` (fallback to description if title empty)
      - `title_description`       : `query: {title} {description}`
      - `title_description_format`: `query: {title}. Hình thức: {format}. {description}`
    """
    if profile not in CLASSIFICATION_VECTOR_PROFILES:
        raise ValueError(
            f"Unsupported classification vector profile {profile!r}. "
            f"Expected one of {CLASSIFICATION_VECTOR_PROFILES}."
        )

    title = normalize_event_text(event.get("title"))
    description = normalize_event_text(event.get("description"))
    if not title and not description:
        eid = event.get("event_id", event.get("post_id"))
        raise ValueError(
            f"Cannot build query text for event id={eid!r}: both title and description are empty."
        )

    if profile == "title":
        body = title if title else description
        return f"{query_prefix}{body}"

    if profile == "title_description":
        parts = [part for part in (title, description) if part]
        return f"{query_prefix}{' '.join(parts)}"

    # profile == "title_description_format"
    fmt_tokens = extract_activity_format_tokens(event)
    segments: List[str] = []
    if title:
        segments.append(title)
    if fmt_tokens:
        segments.append(f"Hình thức: {', '.join(fmt_tokens)}")
    if description:
        segments.append(description)
    return f"{query_prefix}{'. '.join(segments)}"


def build_all_classification_profile_texts(
    event: Dict[str, Any],
    *,
    profiles: Sequence[str] = CLASSIFICATION_VECTOR_PROFILES,
    query_prefix: str = DEFAULT_QUERY_PREFIX,
) -> Dict[str, str]:
    """Build formatted E5 query texts for all requested classification profiles."""
    return {
        p: build_classification_profile_text(event, profile=p, query_prefix=query_prefix)
        for p in profiles
    }
