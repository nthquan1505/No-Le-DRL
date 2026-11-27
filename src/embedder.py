"""P2.1 Embedding interface and `intfloat/multilingual-e5-small` implementation for Module 3.

Provides:
- BaseEmbedder: common interface enforcing 384d L2-normalized outputs and E5 prefixes.
- E5Embedder: production wrapper around `intfloat/multilingual-e5-small` via sentence-transformers.
- create_embedder / is_e5_runtime_available / EmbedderStatus: unified e5-small factory.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from configs.model_config import (
    E5_PASSAGE_PREFIX,
    E5_QUERY_PREFIX,
    EMBEDDING_DIMENSION,
    MODEL_DEVICE,
    MODEL_ID,
)
from src.vector_ops import l2_normalize


class BaseEmbedder(ABC):
    """Abstract embedder interface enforcing 384d L2-normalized E5 embeddings."""

    dimension: int = EMBEDDING_DIMENSION
    model_version: str = MODEL_ID

    @abstractmethod
    def encode_queries(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode query texts (e.g. incoming events) into L2-normalized vectors."""

    @abstractmethod
    def encode_passages(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode reference/passage texts into L2-normalized vectors."""

    def encode_query(self, text: str) -> List[float]:
        return self.encode_queries([text])[0]

    def encode_passage(self, text: str) -> List[float]:
        return self.encode_passages([text])[0]


class E5Embedder(BaseEmbedder):
    """Production embedder wrapping `intfloat/multilingual-e5-small` (P2.1)."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = MODEL_DEVICE,
        dimension: int = EMBEDDING_DIMENSION,
        query_prefix: str = E5_QUERY_PREFIX,
        passage_prefix: str = E5_PASSAGE_PREFIX,
    ) -> None:
        self.model_id = model_id
        self.model_version = model_id
        self.device = device
        self.dimension = dimension
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for E5Embedder. "
                    "Install with `pip install 'no-le-drl[ml]'` or `pip install sentence-transformers torch`."
                ) from exc

            # Resolve local HuggingFace cache snapshot path first so offline startup works
            # without unnecessary HuggingFace Hub network metadata calls.
            local_snapshot: Optional[str] = None
            try:
                from huggingface_hub import snapshot_download  # type: ignore

                local_snapshot = snapshot_download(self.model_id, local_files_only=True)
            except Exception:
                local_snapshot = None

            if local_snapshot:
                self._model = SentenceTransformer(
                    local_snapshot,
                    device=self.device,
                    local_files_only=True,
                )
            else:
                self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model

    def _encode_prefixed(self, texts: Sequence[str], prefix: str) -> List[List[float]]:
        if not texts:
            return []
        model = self._load_model()
        prefixed = [f"{prefix}{(t or '').strip()}" for t in texts]
        raw = model.encode(prefixed, normalize_embeddings=True)
        vectors = [[float(v) for v in row] for row in raw.tolist()]
        for vec in vectors:
            if len(vec) != self.dimension:
                raise ValueError(
                    f"Model '{self.model_id}' produced dim={len(vec)}, expected {self.dimension}"
                )
        return [l2_normalize(v) for v in vectors]

    def encode_queries(self, texts: Sequence[str]) -> List[List[float]]:
        return self._encode_prefixed(texts, self.query_prefix)

    def encode_passages(self, texts: Sequence[str]) -> List[List[float]]:
        return self._encode_prefixed(texts, self.passage_prefix)


@dataclass(frozen=True)
class EmbedderStatus:
    """Metadata describing the active P2.1 embedder backend."""

    requested_model: str
    active_model: str
    is_e5_active: bool
    dimension: int
    device: str


def is_e5_runtime_available() -> bool:
    """Return True if sentence-transformers is importable in the current environment."""
    return importlib.util.find_spec("sentence_transformers") is not None


def create_embedder(
    model_id: str = MODEL_ID,
    device: str = MODEL_DEVICE,
) -> Tuple[E5Embedder, EmbedderStatus]:
    """Instantiate the unified `intfloat/multilingual-e5-small` embedder."""
    if not is_e5_runtime_available():
        raise ImportError(
            "Chưa cài đặt `sentence-transformers` và `torch` trong môi trường Python hiện tại. "
            f"Hãy cài đặt bằng `pip install sentence-transformers torch` để chạy `{model_id}`."
        )

    emb_e5 = E5Embedder(model_id=model_id, device=device, dimension=EMBEDDING_DIMENSION)
    return emb_e5, EmbedderStatus(
        requested_model=model_id,
        active_model=model_id,
        is_e5_active=True,
        dimension=emb_e5.dimension,
        device=device,
    )
