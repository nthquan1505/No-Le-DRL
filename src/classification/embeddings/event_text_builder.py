"""Leakage-free query text builder for HCMUS DRL event classification."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Optional, Tuple


DEFAULT_QUERY_PREFIX = "query: "


def normalize_event_text(value: Any) -> str:
    """Normalize Unicode NFC and collapse whitespace safely for optional/missing fields."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def check_event_classifiable(event: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Determine whether an input record is a classifiable event with usable text.

    Returns:
        (True, None) if classifiable, otherwise (False, skip_reason).
    """
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

    Explicitly excludes labels, categories, extraction reasons, criterion IDs,
    group IDs, and artificial rubric keywords to prevent leakage.
    """
    title = normalize_event_text(event.get("title"))
    description = normalize_event_text(event.get("description"))

    parts = [part for part in (title, description) if part]
    if not parts:
        raise ValueError(
            f"Cannot build query text for event post_id={event.get('post_id')!r}: "
            "both title and description are empty."
        )

    body = " ".join(parts)
    return f"{query_prefix}{body}"
