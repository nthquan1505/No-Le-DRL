"""Independent Sample branch retriever over labeled activity embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

import numpy as np
import pandas as pd

from src.classification.data.loaders import ReferenceBranchDataset


@dataclass(frozen=True)
class SampleNeighbor:
    """One retrieved labeled activity sample and its cosine similarity."""

    embedding_row: int
    sample_id: str
    event_title: str
    criterion_ref: str
    criterion_key: str
    group_id: str
    label_status: str
    similarity: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "embedding_row": self.embedding_row,
            "sample_id": self.sample_id,
            "event_title": self.event_title,
            "criterion_ref": self.criterion_ref,
            "criterion_key": self.criterion_key,
            "group_id": self.group_id,
            "label_status": self.label_status,
            "similarity": round(float(self.similarity), 6),
        }


class SampleRetriever:
    """Independent cosine-similarity retriever for the Sample reference branch.

    Shares ONLY the query vector with the Rubric branch; reference vectors and
    metadata remain strictly isolated.
    """

    def __init__(self, dataset: ReferenceBranchDataset) -> None:
        self._dataset = dataset
        self._embeddings = np.asarray(dataset.embeddings, dtype=np.float32)
        self._metadata = dataset.metadata.reset_index(drop=True)

        # Restrict strictly to valid positive criterion samples
        eligible_mask = self._metadata["label_status"].astype(str) == "valid_event_matchable"
        self._eligible_df = self._metadata.loc[eligible_mask].reset_index(drop=True)
        self._eligible_rows = self._eligible_df["embedding_row"].to_numpy(dtype=np.int64)
        self._eligible_matrix = self._embeddings[self._eligible_rows]

        # Pre-index eligible rows by group_id for fast Stage 2 slicing
        self._group_to_positions: Dict[str, np.ndarray] = {}
        self._group_to_criterion_refs: Dict[str, Set[str]] = {}
        for grp, grp_df in self._eligible_df.groupby("group_id", sort=False):
            grp_str = str(grp)
            self._group_to_positions[grp_str] = grp_df.index.to_numpy(dtype=np.int64)
            self._group_to_criterion_refs[grp_str] = set(grp_df["criterion_ref"].astype(str).unique())

        self._criterion_ref_counts: Dict[str, int] = (
            self._eligible_df["criterion_ref"].astype(str).value_counts().to_dict()
        )
        self.stage1_query_calls: int = 0
        self.stage2_query_calls: int = 0

    @property
    def eligible_count(self) -> int:
        return int(len(self._eligible_df))

    @property
    def available_groups(self) -> List[str]:
        return sorted(self._group_to_positions.keys())

    def get_available_criterion_refs(self, candidate_groups: Sequence[str]) -> Set[str]:
        """Return set of criterion_refs that have >= 1 reference sample within candidate_groups."""
        refs: Set[str] = set()
        for grp in candidate_groups:
            refs.update(self._group_to_criterion_refs.get(str(grp), set()))
        return refs

    def criterion_sample_count(self, criterion_ref: str) -> int:
        return int(self._criterion_ref_counts.get(str(criterion_ref), 0))

    def _build_neighbors(
        self,
        positions: np.ndarray,
        sims: np.ndarray,
        top_k: int,
    ) -> List[SampleNeighbor]:
        if len(positions) == 0 or top_k <= 0:
            return []

        # Deterministic descending similarity sort with sample_id tie-breaker
        records: List[SampleNeighbor] = []
        for pos, sim in zip(positions.tolist(), sims.tolist()):
            row = self._eligible_df.iloc[pos]
            records.append(
                SampleNeighbor(
                    embedding_row=int(row["embedding_row"]),
                    sample_id=str(row["sample_id"]),
                    event_title=str(row["event_title"]),
                    criterion_ref=str(row["criterion_ref"]),
                    criterion_key=str(row["criterion_key"]),
                    group_id=str(row["group_id"]),
                    label_status=str(row["label_status"]),
                    similarity=float(sim),
                )
            )
        records.sort(key=lambda r: (-r.similarity, r.sample_id))
        return records[: min(top_k, len(records))]

    def retrieve_stage1(self, query_vector: np.ndarray, top_k: int) -> List[SampleNeighbor]:
        """Stage 1: cosine similarity against all eligible positive samples, returning Top-K1."""
        self.stage1_query_calls += 1
        if len(self._eligible_rows) == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        sims = self._eligible_matrix @ q
        all_positions = np.arange(len(self._eligible_df), dtype=np.int64)
        return self._build_neighbors(all_positions, sims, top_k=top_k)

    def retrieve_stage2(
        self,
        query_vector: np.ndarray,
        candidate_groups: Sequence[str],
        top_k: int,
    ) -> List[SampleNeighbor]:
        """Stage 2: filter strictly to shared candidate_groups and retrieve Top-K2 samples."""
        self.stage2_query_calls += 1
        unique_groups = list(dict.fromkeys(str(g) for g in candidate_groups))
        pos_chunks = [
            self._group_to_positions[g]
            for g in unique_groups
            if g in self._group_to_positions and len(self._group_to_positions[g]) > 0
        ]
        if not pos_chunks:
            return []

        subset_positions = np.concatenate(pos_chunks)
        emb_rows = self._eligible_df.iloc[subset_positions]["embedding_row"].to_numpy(dtype=np.int64)
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        sims = self._embeddings[emb_rows] @ q
        return self._build_neighbors(subset_positions, sims, top_k=top_k)
