"""Hierarchical ClsIndex storing Stage 1 coarse vectors and per-group Stage 2 datasets (D-2).

Architecture guarantees:
- Data-driven: automatically partitions whatever parent groups (C1..C5, or all groups) and
  canonical '<major>.<minor>' rubric items appear in the input dataset (demo or full).
- Explicit parent-child partitioning: Stage 2 datasets are stored strictly per parent_group_id
  and retrieved as a union across ALL Stage 1 selected candidate parent groups.
- Supports incremental user-corrected samples (Phase 7 feedback loop) with lineage validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from configs.model_config import EMBEDDING_DIMENSION
from src.contract_validator import validate_cls_sample
from src.embedder import BaseEmbedder
from src.schemas import ClsSample, VALID_PARENT_GROUPS, parent_group_from_item_id
from src.vector_ops import l2_normalize, load_npy_float32, save_npy_float32


@dataclass(frozen=True)
class IndexedSample:
    """A validated ClsSample paired with its L2-normalized embedding vector."""

    sample: ClsSample
    vector: List[float]


class HierarchicalClsIndex:
    """Two-level KNN vector index (Stage 1 parent groups + Stage 2 per-group fine datasets)."""

    def __init__(
        self,
        dimension: int = EMBEDDING_DIMENSION,
        model_version: str = "",
        allowed_parent_groups: Sequence[str] = VALID_PARENT_GROUPS,
        index_version: int = 1,
    ) -> None:
        self.dimension = dimension
        self.model_version = model_version
        self.allowed_parent_groups: Tuple[str, ...] = tuple(allowed_parent_groups)
        self.index_version = index_version

        self._stage1_entries: List[IndexedSample] = []
        self._stage2_by_parent_group: Dict[str, List[IndexedSample]] = {}
        self._sample_ids: set = set()

    @property
    def size(self) -> int:
        return len(self._stage1_entries)

    @property
    def parent_groups(self) -> List[str]:
        return sorted(self._stage2_by_parent_group.keys())

    def add_sample(self, sample: ClsSample, vector: Sequence[float]) -> None:
        """Validate and insert one sample into both Stage 1 and its parent group's Stage 2 pool."""
        report = validate_cls_sample(sample, allowed_parent_groups=self.allowed_parent_groups)
        if not report.is_valid:
            msgs = "; ".join(i.message for i in report.errors)
            raise ValueError(f"Invalid ClsSample '{sample.sample_id}': {msgs}")

        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector for '{sample.sample_id}' has dim={len(vector)}, expected {self.dimension}"
            )
        if sample.sample_id in self._sample_ids:
            raise ValueError(f"Duplicate sample_id '{sample.sample_id}' in ClsIndex.")

        expected_parent = parent_group_from_item_id(sample.rubric_item_id)
        if sample.parent_group_id != expected_parent:
            raise ValueError(
                f"Lineage violation: item '{sample.rubric_item_id}' must belong to '{expected_parent}'."
            )

        entry = IndexedSample(sample=sample, vector=l2_normalize(vector))
        self._stage1_entries.append(entry)
        self._stage2_by_parent_group.setdefault(sample.parent_group_id, []).append(entry)
        self._sample_ids.add(sample.sample_id)

    def add_samples_with_embedder(
        self,
        samples: Sequence[ClsSample],
        embedder: BaseEmbedder,
    ) -> None:
        """Encode samples as passages and add them to the hierarchical index."""
        if not samples:
            return
        if not self.model_version:
            self.model_version = embedder.model_version
        vectors = embedder.encode_passages([s.text for s in samples])
        for sample, vec in zip(samples, vectors):
            self.add_sample(sample, vec)

    def get_stage1_entries(self) -> List[IndexedSample]:
        """Return all Stage 1 coarse entries across available parent groups."""
        return list(self._stage1_entries)

    def get_group_entries(self, parent_group_id: str) -> List[IndexedSample]:
        """Return the fine-grained Stage 2 dataset for a single parent group."""
        return list(self._stage2_by_parent_group.get(parent_group_id, []))

    def get_stage2_union_entries(self, selected_groups: Sequence[str]) -> List[IndexedSample]:
        """Retrieve and union the fine-grained datasets for ALL selected Stage 1 parent groups (D-2)."""
        union_entries: List[IndexedSample] = []
        seen_groups: set = set()
        for grp in selected_groups:
            if grp in seen_groups:
                continue
            seen_groups.add(grp)
            union_entries.extend(self._stage2_by_parent_group.get(grp, []))
        return union_entries

    def summary(self) -> Dict[str, Any]:
        item_counts: Dict[str, int] = {}
        for entry in self._stage1_entries:
            iid = entry.sample.rubric_item_id
            item_counts[iid] = item_counts.get(iid, 0) + 1
        return {
            "index_version": self.index_version,
            "model_version": self.model_version,
            "dimension": self.dimension,
            "total_samples": self.size,
            "counts_by_parent_group": {
                grp: len(entries) for grp, entries in sorted(self._stage2_by_parent_group.items())
            },
            "unique_rubric_items": len(item_counts),
        }

    @classmethod
    def from_hierarchical_dataset(
        cls,
        dataset_path: Path,
        embedder: BaseEmbedder,
        allowed_parent_groups: Optional[Sequence[str]] = None,
    ) -> "HierarchicalClsIndex":
        """Build a HierarchicalClsIndex from a JSON generated by scripts/prepare_cls_dataset.py."""
        with open(dataset_path, encoding="utf-8") as handle:
            payload = json.load(handle)

        stage2_map: Dict[str, List[Dict[str, Any]]] = payload.get("stage2_by_parent_group") or {}
        if allowed_parent_groups is None:
            excluded = set(payload.get("excluded_parent_groups") or [])
            allowed = tuple(g for g in VALID_PARENT_GROUPS if g not in excluded)
        else:
            allowed = tuple(allowed_parent_groups)

        index = cls(
            dimension=embedder.dimension,
            model_version=embedder.model_version,
            allowed_parent_groups=allowed,
        )

        all_samples: List[ClsSample] = []
        for parent_grp in sorted(stage2_map.keys()):
            for raw in stage2_map[parent_grp]:
                all_samples.append(ClsSample.from_dict(raw))

        index.add_samples_with_embedder(all_samples, embedder)
        return index

    def save(self, directory: Path) -> None:
        """Persist index metadata and per-group + coarse float32 .npy matrices."""
        directory.mkdir(parents=True, exist_ok=True)
        stage1_vectors = [e.vector for e in self._stage1_entries]
        save_npy_float32(directory / "stage1_embeddings.npy", stage1_vectors, self.dimension)

        group_files: Dict[str, str] = {}
        stage2_meta: Dict[str, List[Dict[str, Any]]] = {}
        for grp, entries in sorted(self._stage2_by_parent_group.items()):
            npy_name = f"stage2_{grp}_embeddings.npy"
            save_npy_float32(
                directory / npy_name,
                [e.vector for e in entries],
                self.dimension,
            )
            group_files[grp] = npy_name
            stage2_meta[grp] = [e.sample.to_dict() for e in entries]

        metadata = {
            "schema_version": "hierarchical-cls-index-v1",
            "index_version": self.index_version,
            "model_version": self.model_version,
            "dimension": self.dimension,
            "allowed_parent_groups": list(self.allowed_parent_groups),
            "stage1_embeddings_file": "stage1_embeddings.npy",
            "stage2_embedding_files": group_files,
            "stage2_by_parent_group": stage2_meta,
        }
        with open(directory / "index_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: Path) -> "HierarchicalClsIndex":
        """Load a persisted HierarchicalClsIndex from disk."""
        with open(directory / "index_metadata.json", encoding="utf-8") as handle:
            meta = json.load(handle)

        index = cls(
            dimension=int(meta["dimension"]),
            model_version=str(meta.get("model_version") or ""),
            allowed_parent_groups=tuple(meta.get("allowed_parent_groups") or VALID_PARENT_GROUPS),
            index_version=int(meta.get("index_version") or 1),
        )

        group_files: Dict[str, str] = meta["stage2_embedding_files"]
        stage2_meta: Dict[str, List[Dict[str, Any]]] = meta["stage2_by_parent_group"]
        for grp in sorted(stage2_meta.keys()):
            vectors, shape = load_npy_float32(directory / group_files[grp])
            raw_samples = stage2_meta[grp]
            if shape[0] != len(raw_samples):
                raise ValueError(
                    f"Group '{grp}' vector count ({shape[0]}) != metadata count ({len(raw_samples)})"
                )
            for raw_sample, vec in zip(raw_samples, vectors):
                index.add_sample(ClsSample.from_dict(raw_sample), vec)

        return index
