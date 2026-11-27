"""Dedicated Recommendation Event Embedder (`title + description + topic + format`).

Constructs passage embeddings specifically for P6 Interest matching against `user_vec`
(`query: `), using the 4 required fields from the `events` table:
  - `title`
  - `description`
  - `topic`
  - `format` (semantic activity formats; excludes pure transport flags online/offline)

Supports both DB-sourced events (`drl_database.db` via `DRLDatabaseManager`) and
normalized `EventNormalized` records, with disk caching and empty-pool safety (`N == 0`).
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.classification.embeddings.event_text_builder import (
    extract_activity_format_tokens,
    normalize_event_text,
    normalize_string_list_field,
)
from src.embedder import BaseEmbedder
from src.schemas import EventNormalized
from src.vector_ops import l2_normalize

DEFAULT_REC_PASSAGE_PREFIX = "passage: "
REC_EMBEDDING_FIELDS: Tuple[str, ...] = ("title", "description", "topic", "format")
_GENERIC_TOPIC_NOISE = frozenset({"khác", "khac", "other", "activity"})


@dataclass(frozen=True)
class RecommendationEmbeddingSet:
    """Cached or freshly computed 4-field Recommendation embeddings (`title+description+topic+format`)."""

    embeddings: np.ndarray
    metadata: Dict[str, Any]
    event_id_to_row: Dict[str, int]
    reused_from_cache: bool

    def get_vector(self, event_id: str) -> Optional[List[float]]:
        row = self.event_id_to_row.get(str(event_id))
        if row is None or row < 0 or row >= self.embeddings.shape[0]:
            return None
        return [float(v) for v in self.embeddings[row].tolist()]


def _extract_event_fields(
    event: Union[EventNormalized, Dict[str, Any]],
) -> Tuple[str, str, str, List[str], List[str]]:
    """Extract `(event_id, title, description, topics, formats)` uniformly from EventNormalized or dict."""
    if isinstance(event, EventNormalized):
        meta = event.metadata or {}
        raw_topics = meta.get("topic") if meta.get("topic") is not None else event.categories
        raw_formats = meta.get("format")
        eid = str(event.event_id)
        title = normalize_event_text(event.title)
        desc = normalize_event_text(event.description)
        topics = normalize_string_list_field(raw_topics, exclude_tokens=tuple(_GENERIC_TOPIC_NOISE))
        formats = extract_activity_format_tokens({"format": raw_formats})
        return eid, title, desc, topics, formats

    eid = str(event.get("event_id") or event.get("post_id") or "")
    title = normalize_event_text(event.get("title"))
    desc = normalize_event_text(event.get("description"))
    raw_topics = event.get("topic") if event.get("topic") is not None else event.get("category")
    topics = normalize_string_list_field(raw_topics, exclude_tokens=tuple(_GENERIC_TOPIC_NOISE))
    formats = extract_activity_format_tokens(event)
    return eid, title, desc, topics, formats


def build_recommendation_event_text(
    event: Union[EventNormalized, Dict[str, Any]],
    *,
    include_prefix: bool = True,
    passage_prefix: str = DEFAULT_REC_PASSAGE_PREFIX,
) -> str:
    """Build Recommendation text from `title + description + topic + format`.

    Format (when all 4 fields are populated):
        `passage: {title}. Chủ đề: {topic}. Hình thức: {format}. {description}`

    If `include_prefix=False`, returns the unprefixed body suitable for `BaseEmbedder.encode_passage()`.
    """
    _, title, desc, topics, formats = _extract_event_fields(event)

    segments: List[str] = []
    if title:
        segments.append(title)
    if topics:
        segments.append(f"Chủ đề: {', '.join(topics)}")
    if formats:
        segments.append(f"Hình thức: {', '.join(formats)}")
    if desc:
        segments.append(desc)

    body = ". ".join(segments).strip()
    if not body:
        body = "Hoạt động sinh viên"
    return f"{passage_prefix}{body}" if include_prefix else body


def _can_reuse_rec_cache(
    embeddings_path: Path,
    metadata_path: Path,
    source_fingerprint: str,
    expected_count: int,
    expected_dim: int,
    passage_prefix: str,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    if not embeddings_path.is_file() or not metadata_path.is_file():
        return None
    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        arr = np.load(embeddings_path, mmap_mode="r")
    except Exception:
        return None

    if (
        meta.get("schema_version") != "events-rec-embeddings-v1"
        or meta.get("source_fingerprint") != source_fingerprint
        or meta.get("embedded_events_count") != expected_count
        or meta.get("embedding_dimension") != expected_dim
        or meta.get("passage_prefix") != passage_prefix
        or tuple(meta.get("fields") or ()) != REC_EMBEDDING_FIELDS
    ):
        return None
    if arr.ndim != 2 or arr.shape != (expected_count, expected_dim) or arr.dtype != np.float32:
        return None
    if not np.isfinite(arr).all():
        return None
    return np.array(arr, dtype=np.float32, copy=True), meta


def generate_or_load_recommendation_embeddings(
    events: Sequence[Union[EventNormalized, Dict[str, Any]]],
    *,
    embedder: BaseEmbedder,
    source_fingerprint: str,
    embeddings_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    reuse_existing: bool = True,
    passage_prefix: str = DEFAULT_REC_PASSAGE_PREFIX,
) -> RecommendationEmbeddingSet:
    """Encode (or load cached) 4-field Recommendation embeddings (`title+description+topic+format`).

    Handles an empty event sequence (`len(events) == 0`, e.g., empty SQLite DB) cleanly.
    """
    dim = int(embedder.dimension)
    if (
        reuse_existing
        and embeddings_path is not None
        and metadata_path is not None
    ):
        reused = _can_reuse_rec_cache(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            source_fingerprint=source_fingerprint,
            expected_count=len(events),
            expected_dim=dim,
            passage_prefix=passage_prefix,
        )
        if reused is not None:
            arr, meta = reused
            id_map = {str(r["event_id"]): int(r["embedding_row"]) for r in (meta.get("records") or [])}
            return RecommendationEmbeddingSet(
                embeddings=arr,
                metadata=meta,
                event_id_to_row=id_map,
                reused_from_cache=True,
            )

    records: List[Dict[str, Any]] = []
    bodies: List[str] = []
    id_map: Dict[str, int] = {}

    for idx, ev in enumerate(events):
        eid, title, _, topics, formats = _extract_event_fields(ev)
        body = build_recommendation_event_text(ev, include_prefix=False, passage_prefix=passage_prefix)
        full_text = f"{passage_prefix}{body}"
        records.append(
            {
                "embedding_row": idx,
                "event_id": eid,
                "event_title": title,
                "topic": topics,
                "format": formats,
                "embedding_text": full_text,
            }
        )
        bodies.append(body)
        if eid:
            id_map[eid] = idx

    if bodies:
        encoded = embedder.encode_passages(bodies)
        normed = [l2_normalize(vec) for vec in encoded]
        arr = np.asarray(normed, dtype=np.float32)
    else:
        arr = np.zeros((0, dim), dtype=np.float32)

    meta: Dict[str, Any] = {
        "schema_version": "events-rec-embeddings-v1",
        "source_fingerprint": source_fingerprint,
        "fields": list(REC_EMBEDDING_FIELDS),
        "passage_prefix": passage_prefix,
        "embedding_dimension": dim,
        "dtype": "float32",
        "normalization": "l2",
        "embedded_events_count": len(records),
        "generation_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "records": records,
    }

    if embeddings_path is not None and metadata_path is not None:
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, arr)
        metadata_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    return RecommendationEmbeddingSet(
        embeddings=arr,
        metadata=meta,
        event_id_to_row=id_map,
        reused_from_cache=False,
    )
