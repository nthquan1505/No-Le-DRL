"""Independent Rubric branch retriever over semantic DRL rubric prototypes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from src.classification.data.loaders import ReferenceBranchDataset


@dataclass(frozen=True)
class RubricPrototypeMatch:
    """One retrieved semantic rubric prototype and its cosine similarity."""

    embedding_row: int
    prototype_id: str
    prototype_type: str
    stage: str
    group_id: str
    criterion_ref: Optional[str]
    criterion_key: Optional[str]
    anchor_status: Optional[str]
    criterion_anchors_status: Optional[str]
    event_matchable: bool
    is_positive_target: bool
    has_uncertain: bool
    embedding_text: str
    similarity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding_row": self.embedding_row,
            "prototype_id": self.prototype_id,
            "prototype_type": self.prototype_type,
            "stage": self.stage,
            "group_id": self.group_id,
            "criterion_ref": self.criterion_ref,
            "criterion_key": self.criterion_key,
            "anchor_status": self.anchor_status,
            "criterion_anchors_status": self.criterion_anchors_status,
            "event_matchable": self.event_matchable,
            "is_positive_target": self.is_positive_target,
            "has_uncertain": self.has_uncertain,
            "similarity": round(float(self.similarity), 6),
        }


class RubricRetriever:
    """Independent cosine-similarity retriever for the Rubric semantic prototype branch.

    Keeps rubric vectors, metadata, and rubric.yaml completely separate from the Sample branch.
    """

    def __init__(
        self,
        dataset: ReferenceBranchDataset,
        rubric_data: Dict[str, Any],
    ) -> None:
        self._dataset = dataset
        self._rubric = rubric_data
        self._embeddings = np.asarray(dataset.embeddings, dtype=np.float32)
        self._metadata = dataset.metadata.reset_index(drop=True)

        # Stage 1: stage == "group" and event_matchable == True
        s1_mask = (self._metadata["stage"].astype(str) == "group") & (
            self._metadata["event_matchable"].astype(bool)
        )
        self._stage1_df = self._metadata.loc[s1_mask].reset_index(drop=True)
        self._stage1_rows = self._stage1_df["embedding_row"].to_numpy(dtype=np.int64)
        self._stage1_matrix = self._embeddings[self._stage1_rows]

        # Stage 2: stage == "criterion" and event_matchable == True
        s2_mask = (self._metadata["stage"].astype(str) == "criterion") & (
            self._metadata["event_matchable"].astype(bool)
        )
        self._stage2_df = self._metadata.loc[s2_mask].reset_index(drop=True)

        self._stage2_group_to_positions: Dict[str, np.ndarray] = {}
        self._stage2_group_to_refs: Dict[str, Set[str]] = {}
        for grp, grp_df in self._stage2_df.groupby("group_id", sort=False):
            grp_str = str(grp)
            self._stage2_group_to_positions[grp_str] = grp_df.index.to_numpy(dtype=np.int64)
            self._stage2_group_to_refs[grp_str] = set(grp_df["criterion_ref"].astype(str).unique())

        # Index rubric.yaml items by criterion_ref for authoritative metadata lookup
        self._ref_to_rubric_item: Dict[str, Dict[str, Any]] = {}
        self._ref_to_criterion_key: Dict[str, str] = {}
        for key, item in (rubric_data.get("items") or {}).items():
            if isinstance(item, dict) and "ref" in item:
                ref = str(item["ref"]).strip()
                self._ref_to_rubric_item[ref] = item
                self._ref_to_criterion_key[ref] = key

        self._open_questions: List[Dict[str, Any]] = list(rubric_data.get("open_questions") or [])
        self.stage1_query_calls: int = 0
        self.stage2_query_calls: int = 0

    @property
    def eligible_stage1_groups(self) -> List[str]:
        return sorted(self._stage1_df["group_id"].astype(str).unique().tolist())

    def get_available_criterion_refs(self, candidate_groups: Sequence[str]) -> Set[str]:
        refs: Set[str] = set()
        for grp in candidate_groups:
            refs.update(self._stage2_group_to_refs.get(str(grp), set()))
        return refs

    def get_criterion_info(self, criterion_ref: str) -> Dict[str, Any]:
        """Return authoritative rubric metadata (name, key, group, scope_note, uncertain, etc.)."""
        ref = str(criterion_ref).strip()
        item = self._ref_to_rubric_item.get(ref)
        if not item:
            return {}
        return {
            "criterion_ref": ref,
            "criterion_key": self._ref_to_criterion_key.get(ref),
            "group_id": str(item.get("group", "")),
            "tieu_chi": item.get("tieu_chi"),
            "name": str(item.get("name", "")),
            "kind": str(item.get("kind", "")),
            "event_matchable": bool(item.get("event_matchable", False)),
            "mvp_scope": item.get("mvp_scope"),
            "anchors_status": item.get("anchors_status"),
            "scope_note": item.get("scope_note"),
            "uncertain": list(item.get("uncertain") or []),
            "keywords": list(item.get("keywords") or []),
            "inclusion_rules": list(item.get("inclusion_rules") or []),
            "exclusion_rules": list(item.get("exclusion_rules") or []),
            "confusable_with": list(item.get("confusable_with") or []),
        }

    def get_group_name(self, group_id: str) -> str:
        grp = (self._rubric.get("groups") or {}).get(str(group_id)) or {}
        return str(grp.get("name", group_id))

    def get_relevant_open_questions(self, criterion_ref: str) -> List[Dict[str, str]]:
        """Extract open_questions from rubric.yaml referencing this criterion_ref."""
        ref = str(criterion_ref).strip()
        matched: List[Dict[str, str]] = []
        for q in self._open_questions:
            topic = str(q.get("topic", ""))
            question = str(q.get("question", ""))
            if ref in topic or ref in question:
                matched.append(
                    {
                        "id": str(q.get("id", "")),
                        "topic": topic,
                        "question": question,
                    }
                )
        return matched

    @staticmethod
    def _opt_str(val: Any) -> Optional[str]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        if pd.isna(val):
            return None
        return str(val)

    def _row_to_match(self, row: pd.Series, sim: float) -> RubricPrototypeMatch:
        is_pos = (
            bool(row["is_positive_target"])
            if "is_positive_target" in row.index and not pd.isna(row["is_positive_target"])
            else (str(row["group_id"]) != "P")
        )
        has_unc = (
            bool(row["has_uncertain"])
            if "has_uncertain" in row.index and not pd.isna(row["has_uncertain"])
            else False
        )
        crit_ast = (
            self._opt_str(row["criterion_anchors_status"])
            if "criterion_anchors_status" in row.index
            else self._opt_str(row["anchor_status"])
        )
        return RubricPrototypeMatch(
            embedding_row=int(row["embedding_row"]),
            prototype_id=str(row["prototype_id"]),
            prototype_type=str(row["prototype_type"]),
            stage=str(row["stage"]),
            group_id=str(row["group_id"]),
            criterion_ref=self._opt_str(row["criterion_ref"]),
            criterion_key=self._opt_str(row["criterion_key"]),
            anchor_status=self._opt_str(row["anchor_status"]),
            criterion_anchors_status=crit_ast,
            event_matchable=bool(row["event_matchable"]),
            is_positive_target=is_pos,
            has_uncertain=has_unc,
            embedding_text=str(row.get("embedding_text", "")),
            similarity=float(sim),
        )

    def retrieve_stage1_groups(self, query_vector: np.ndarray) -> List[RubricPrototypeMatch]:
        """Stage 1: cosine similarity against all eligible `stage=group` prototypes."""
        self.stage1_query_calls += 1
        if len(self._stage1_rows) == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        sims = self._stage1_matrix @ q
        matches = [
            self._row_to_match(self._stage1_df.iloc[pos], float(sim))
            for pos, sim in enumerate(sims.tolist())
        ]
        matches.sort(key=lambda m: (-m.similarity, m.prototype_id))
        return matches

    def retrieve_stage2_criteria(
        self,
        query_vector: np.ndarray,
        candidate_groups: Sequence[str],
    ) -> List[RubricPrototypeMatch]:
        """Stage 2: cosine similarity against `stage=criterion`, `event_matchable=true` in candidate_groups."""
        self.stage2_query_calls += 1
        unique_groups = list(dict.fromkeys(str(g) for g in candidate_groups))
        pos_chunks = [
            self._stage2_group_to_positions[g]
            for g in unique_groups
            if g in self._stage2_group_to_positions and len(self._stage2_group_to_positions[g]) > 0
        ]
        if not pos_chunks:
            return []

        subset_positions = np.concatenate(pos_chunks)
        emb_rows = self._stage2_df.iloc[subset_positions]["embedding_row"].to_numpy(dtype=np.int64)
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        sims = self._embeddings[emb_rows] @ q

        matches = [
            self._row_to_match(self._stage2_df.iloc[pos], float(sim))
            for pos, sim in zip(subset_positions.tolist(), sims.tolist())
        ]
        matches.sort(key=lambda m: (-m.similarity, m.prototype_id))
        return matches
