"""P2.1 Embedding interfaces and implementations for Module 3.

Provides:
- BaseEmbedder: common interface enforcing 384d L2-normalized outputs and E5 prefixes.
- E5Embedder: production wrapper around `intfloat/multilingual-e5-small` via sentence-transformers.
- HashingDemoEmbedder: deterministic, zero-dependency 384d L2-normalized embedder for fast
  offline unit tests and smoke runs without downloading HuggingFace weights.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from abc import ABC, abstractmethod
from typing import List, Sequence

from configs.model_config import (
    E5_PASSAGE_PREFIX,
    E5_QUERY_PREFIX,
    EMBEDDING_DIMENSION,
    MODEL_DEVICE,
    MODEL_ID,
)
from src.vector_ops import l2_normalize


class BaseEmbedder(ABC):
    """Abstract embedder interface for P2.1 and P4 user profile vectors."""

    dimension: int = EMBEDDING_DIMENSION
    model_version: str = MODEL_ID

    @abstractmethod
    def encode_queries(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode query texts (e.g. incoming events) into L2-normalized vectors."""

    @abstractmethod
    def encode_passages(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode reference/index texts (e.g. ClsIndex samples) into L2-normalized vectors."""

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
                    "Install with `pip install 'no-le-drl[ml]'` or `pip install sentence-transformers`."
                ) from exc
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


class HashingDemoEmbedder(BaseEmbedder):
    """Deterministic zero-dependency 384d L2-normalized embedder for offline tests/demos.

    Hashes accent-stripped word unigrams, bigrams, and character trigrams into a fixed
    384-dimensional space so lexically similar Vietnamese/English event texts share high
    cosine similarity without requiring PyTorch or HuggingFace downloads.
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension
        self.model_version = f"hashing-demo-v1-{dimension}d"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        norm = unicodedata.normalize("NFD", (text or "").lower().replace("đ", "d"))
        norm = "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")
        words = re.findall(r"[a-z0-9]+", norm)
        features: List[str] = []
        features.extend(f"w:{w}" for w in words)
        features.extend(f"b:{words[i]}_{words[i + 1]}" for i in range(len(words) - 1))
        joined = " ".join(words)
        if len(joined) >= 3:
            features.extend(f"c:{joined[i : i + 3]}" for i in range(len(joined) - 2))
        return features or ["__empty__"]

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        for feat in self._tokenize(text):
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            weight = 2.0 if feat.startswith("w:") else (1.5 if feat.startswith("b:") else 0.5)
            vec[idx] += sign * weight
        return l2_normalize(vec)

    def encode_queries(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def encode_passages(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]
