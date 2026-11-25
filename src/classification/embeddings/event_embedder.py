"""Shared query embedding generator and cache validator for events.json.

Generates exactly ONE query embedding vector per classifiable event using the
exact model and configuration recorded in `embedding_config.json`, then shares
that vector across both Sample and Rubric branches and across Stage 1 and Stage 2.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.classification.embeddings.event_text_builder import (
    build_event_query_text,
    check_event_classifiable,
    normalize_event_text,
)


@dataclass(frozen=True)
class SharedQueryEmbeddingSet:
    """Loaded or freshly generated shared event query embeddings and metadata."""

    embeddings: np.ndarray
    metadata: Dict[str, Any]
    embeddings_path: Path
    metadata_path: Path
    reused_from_cache: bool
    event_index_to_row: Dict[int, int]
    skipped_by_event_index: Dict[int, Dict[str, Any]]


def _build_index_maps(
    metadata: Dict[str, Any],
) -> Tuple[Dict[int, int], Dict[int, Dict[str, Any]]]:
    event_index_to_row = {
        int(rec["source_event_index"]): int(rec["embedding_row"])
        for rec in (metadata.get("records") or [])
    }
    skipped_by_event_index = {
        int(rec["source_event_index"]): rec
        for rec in (metadata.get("skipped_records") or [])
    }
    return event_index_to_row, skipped_by_event_index


def _can_reuse_existing(
    embeddings_path: Path,
    metadata_path: Path,
    events_sha256: str,
    expected_event_count: int,
    embedding_config: Dict[str, Any],
    l2_tolerance: float,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """Return (embeddings, metadata) if existing query artifacts match source and config."""
    if not embeddings_path.exists() or not metadata_path.exists():
        return None
    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        vecs = np.load(embeddings_path, mmap_mode="r")
    except Exception:
        return None

    expected_model = embedding_config.get("model_name")
    expected_rev = embedding_config.get("model_revision")
    expected_dim = int(embedding_config.get("embedding_dimension", 384))
    expected_prefix = (embedding_config.get("prefixes") or {}).get("query", "query: ")

    if (
        meta.get("source_events_sha256") != events_sha256
        or meta.get("total_input_events") != expected_event_count
        or meta.get("embedding_model") != expected_model
        or meta.get("model_revision") != expected_rev
        or meta.get("embedding_dimension") != expected_dim
        or meta.get("query_prefix") != expected_prefix
        or meta.get("normalization") != "l2"
        or meta.get("dtype") != "float32"
    ):
        return None

    records = meta.get("records") or []
    if vecs.ndim != 2 or vecs.shape != (len(records), expected_dim) or vecs.dtype != np.float32:
        return None
    if not np.isfinite(vecs).all():
        return None
    if len(records) > 0:
        norms = np.linalg.norm(np.asarray(vecs, dtype=np.float64), axis=1)
        if float(np.max(np.abs(norms - 1.0))) > l2_tolerance:
            return None

    return np.array(vecs, dtype=np.float32, copy=True), meta


def _load_matching_e5_model(embedding_config: Dict[str, Any], device: str = "cpu"):
    """Load the exact SentenceTransformer model and cached revision specified in embedding_config.json."""
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    model_name = str(embedding_config["model_name"])
    expected_rev = embedding_config.get("model_revision")

    snapshot_path: Optional[str] = None
    try:
        snapshot_path = snapshot_download(
            model_name,
            revision=expected_rev if expected_rev else None,
            local_files_only=True,
        )
    except Exception:
        snapshot_path = None

    if snapshot_path:
        model = SentenceTransformer(snapshot_path, device=device, local_files_only=True)
    else:
        model = SentenceTransformer(model_name, device=device)

    model.eval()
    actual_dim = (
        int(model.get_embedding_dimension())
        if hasattr(model, "get_embedding_dimension")
        else int(model.get_sentence_embedding_dimension())
    )
    expected_dim = int(embedding_config["embedding_dimension"])
    if actual_dim != expected_dim:
        raise ValueError(
            f"Loaded model '{model_name}' dimension={actual_dim}, expected {expected_dim}"
        )
    return model


def generate_or_load_shared_query_embeddings(
    *,
    events: List[Dict[str, Any]],
    events_path: Path,
    events_sha256: str,
    embedding_config: Dict[str, Any],
    query_embeddings_path: Path,
    query_metadata_path: Path,
    reuse_existing: bool = True,
    batch_size: int = 32,
    device: str = "cpu",
    l2_tolerance: float = 1e-4,
) -> SharedQueryEmbeddingSet:
    """Generate (or verify and reuse) exactly one shared query embedding per classifiable event."""
    if reuse_existing:
        reused = _can_reuse_existing(
            embeddings_path=query_embeddings_path,
            metadata_path=query_metadata_path,
            events_sha256=events_sha256,
            expected_event_count=len(events),
            embedding_config=embedding_config,
            l2_tolerance=l2_tolerance,
        )
        if reused is not None:
            vecs, meta = reused
            idx_to_row, skipped_map = _build_index_maps(meta)
            return SharedQueryEmbeddingSet(
                embeddings=vecs,
                metadata=meta,
                embeddings_path=query_embeddings_path,
                metadata_path=query_metadata_path,
                reused_from_cache=True,
                event_index_to_row=idx_to_row,
                skipped_by_event_index=skipped_map,
            )

    model_name = str(embedding_config["model_name"])
    model_revision = embedding_config.get("model_revision")
    expected_dim = int(embedding_config["embedding_dimension"])
    query_prefix = str((embedding_config.get("prefixes") or {}).get("query", "query: "))
    pooling_method = str(embedding_config.get("pooling_method", "mean"))

    records: List[Dict[str, Any]] = []
    skipped_records: List[Dict[str, Any]] = []
    texts_to_encode: List[str] = []

    for source_idx, ev in enumerate(events):
        post_id = ev.get("post_id")
        event_title = normalize_event_text(ev.get("title"))
        classifiable, skip_reason = check_event_classifiable(ev)
        if not classifiable:
            skipped_records.append(
                {
                    "source_event_index": source_idx,
                    "post_id": post_id,
                    "event_title": event_title,
                    "skip_reason": skip_reason,
                }
            )
            continue

        query_text = build_event_query_text(ev, query_prefix=query_prefix)
        emb_row = len(records)
        records.append(
            {
                "embedding_row": emb_row,
                "post_id": post_id,
                "source_event_index": source_idx,
                "event_title": event_title,
                "embedding_text": query_text,
                "embedding_model": model_name,
                "embedding_dimension": expected_dim,
                "normalization": "l2",
            }
        )
        texts_to_encode.append(query_text)

    if texts_to_encode:
        model = _load_matching_e5_model(embedding_config, device=device)
        tok = model.tokenizer
        max_seq_len = int(model.max_seq_length)
        tok_lengths = [
            len(ids)
            for ids in tok(texts_to_encode, add_special_tokens=True, truncation=False)["input_ids"]
        ]
        token_stats = {
            "count": len(texts_to_encode),
            "min_tokens": int(min(tok_lengths)),
            "median_tokens": float(np.median(tok_lengths)),
            "max_tokens": int(max(tok_lengths)),
            "max_seq_length": max_seq_len,
            "truncated_count": int(sum(1 for l in tok_lengths if l > max_seq_len)),
        }

        raw_vectors = model.encode(
            texts_to_encode,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors = np.asarray(raw_vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Zero-norm query embedding encountered.")
        vectors = (vectors / norms).astype(np.float32)
    else:
        vectors = np.zeros((0, expected_dim), dtype=np.float32)
        token_stats = {
            "count": 0,
            "min_tokens": 0,
            "median_tokens": 0.0,
            "max_tokens": 0,
            "max_seq_length": 512,
            "truncated_count": 0,
        }

    metadata: Dict[str, Any] = {
        "schema_version": "events-query-embeddings-v1",
        "source_events_path": events_path.as_posix(),
        "source_events_sha256": events_sha256,
        "embedding_model": model_name,
        "model_revision": model_revision,
        "embedding_dimension": expected_dim,
        "dtype": "float32",
        "normalization": "l2",
        "pooling_method": pooling_method,
        "query_prefix": query_prefix,
        "shared_across_branches": True,
        "total_input_events": len(events),
        "embedded_events_count": len(records),
        "skipped_events_count": len(skipped_records),
        "token_stats": token_stats,
        "generation_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "records": records,
        "skipped_records": skipped_records,
    }

    query_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    query_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(query_embeddings_path, vectors)
    query_metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    idx_to_row, skipped_map = _build_index_maps(metadata)
    return SharedQueryEmbeddingSet(
        embeddings=vectors,
        metadata=metadata,
        embeddings_path=query_embeddings_path,
        metadata_path=query_metadata_path,
        reused_from_cache=False,
        event_index_to_row=idx_to_row,
        skipped_by_event_index=skipped_map,
    )
