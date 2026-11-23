"""DATA · CACHE — Event Profile Cache storing 384d vectors, P2 labels, and Module 6 expected points."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from configs.model_config import (
    EMBEDDING_DIMENSION,
    EVENT_EMBEDDINGS_PATH,
    EVENT_METADATA_PATH,
)
from src.schemas import ClassificationResult, CreditEstimate, EventNormalized
from src.vector_ops import l2_normalize, load_npy_float32, save_npy_float32


@dataclass
class CachedEventProfile:
    """One cached entry in Event Profile Cache (vector + P2 labels + M6 credit estimate)."""

    event: EventNormalized
    vector: List[float]
    classification: ClassificationResult
    credit_estimate: Optional[CreditEstimate] = None

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "classification": self.classification.to_dict(),
            "credit_estimate": (
                self.credit_estimate.to_dict() if self.credit_estimate is not None else None
            ),
        }


class EventProfileCache:
    """Local cache backed by event_embeddings.npy (float32) and event_metadata.json."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension
        self._profiles: Dict[str, CachedEventProfile] = {}
        self._order: List[str] = []

    def __len__(self) -> int:
        return len(self._order)

    def upsert(
        self,
        event: EventNormalized,
        vector: Sequence[float],
        classification: ClassificationResult,
        credit_estimate: Optional[CreditEstimate] = None,
    ) -> None:
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector for event '{event.event_id}' has dim={len(vector)}, expected {self.dimension}"
            )
        if classification.event_id != event.event_id:
            raise ValueError(
                f"Mismatched event_id: event='{event.event_id}' vs classification='{classification.event_id}'"
            )
        if event.event_id not in self._profiles:
            self._order.append(event.event_id)
        self._profiles[event.event_id] = CachedEventProfile(
            event=event,
            vector=l2_normalize(vector),
            classification=classification,
            credit_estimate=credit_estimate,
        )

    def get(self, event_id: str) -> Optional[CachedEventProfile]:
        return self._profiles.get(event_id)

    def all_profiles(self) -> List[CachedEventProfile]:
        return [self._profiles[eid] for eid in self._order]

    def save(
        self,
        embeddings_path: Path = EVENT_EMBEDDINGS_PATH,
        metadata_path: Path = EVENT_METADATA_PATH,
    ) -> None:
        vectors = [self._profiles[eid].vector for eid in self._order]
        save_npy_float32(embeddings_path, vectors, self.dimension)

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "event-profile-cache-v1",
            "dimension": self.dimension,
            "count": len(self._order),
            "embeddings_file": embeddings_path.name,
            "profiles": [self._profiles[eid].to_metadata_dict() for eid in self._order],
        }
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        embeddings_path: Path = EVENT_EMBEDDINGS_PATH,
        metadata_path: Path = EVENT_METADATA_PATH,
    ) -> "EventProfileCache":
        with open(metadata_path, encoding="utf-8") as handle:
            payload = json.load(handle)

        dimension = int(payload["dimension"])
        vectors, shape = load_npy_float32(embeddings_path)
        profiles_raw = payload.get("profiles") or []
        if shape[0] != len(profiles_raw):
            raise ValueError(
                f"Embeddings rows ({shape[0]}) != metadata profiles ({len(profiles_raw)})"
            )

        cache = cls(dimension=dimension)
        for raw, vec in zip(profiles_raw, vectors):
            ev = EventNormalized.from_dict(raw["event"])
            cls_res = ClassificationResult.from_dict(raw["classification"])
            credit_raw = raw.get("credit_estimate")
            credit = CreditEstimate(**credit_raw) if credit_raw is not None else None
            cache.upsert(ev, vec, cls_res, credit_estimate=credit)
        return cache
