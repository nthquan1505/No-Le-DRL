"""P2 Two-Stage Hierarchical KNN Classifier (Approved Decisions D-1, D-2, D-3, T-1..T-4).

Architecture:
- P2.1: Embed normalized event text into a 384d L2-normalized vector.
- P2.2 (Stage 1 — Coarse Parent-Group KNN): Rank parent groups (C1..C5) via KNN over ClsIndex
  and select the Top-k candidate parent groups (optionally filtered by stage1_min_score_threshold).
  Stage 1 selects candidate parent groups ONLY; it never determines the final rubric item.
- P2.3 (Stage 2 — Fine-Grained Multi-Group Union KNN): Retrieve the fine-grained dataset for
  EVERY selected Stage 1 parent group, run the second KNN over their UNION, and rank specific
  canonical '<major>.<minor>' rubric items paired with their parent group.

Mandatory D-2 invariants enforced in code:
1. Open hyperparameters (stage1_top_k, stage1_neighbors, stage2_neighbors, etc.) must be
   explicitly provided via HierarchicalKNNConfig — never silently hardcoded in production.
2. stage1_scores and stage2_scores are preserved as strictly distinct fields.
3. Stage 1 and Stage 2 scores are NEVER blindly multiplied (stage1 * stage2).
4. Raw KNN scores are NEVER exposed as calibrated probabilities unless a fitted calibrator
   is explicitly attached (`is_calibrated=False`, `calibrated_confidence=None`).
5. Classification does NOT decide DRL credit eligibility (Module 6 owns eligibility per D-3).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from configs.model_config import HierarchicalKNNConfig
from src.cls_index import HierarchicalClsIndex, IndexedSample
from src.contract_validator import validate_classification_result
from src.embedder import BaseEmbedder
from src.schemas import ClassificationResult, EventNormalized, parent_group_from_item_id
from src.vector_ops import batch_dot, l2_normalize

# Custom aggregation hook: (stage1_scores, stage2_scores, item_to_parent) -> aggregated_scores
ScoreAggregatorFn = Callable[[Dict[str, float], Dict[str, float], Dict[str, str]], Dict[str, float]]
# Custom calibration hook: (top_item_id, stage1_scores, stage2_scores, aggregated_scores) -> float in [0, 1]
CalibratorFn = Callable[
    [str, Dict[str, float], Dict[str, float], Optional[Dict[str, float]]],
    float,
]


def _knn_similarity_scores(
    query_vec: Sequence[float],
    entries: Sequence[IndexedSample],
    label_getter: Callable[[IndexedSample], str],
    k_neighbors: int,
) -> Dict[str, float]:
    """Compute similarity-weighted KNN scores over the top-k_neighbors nearest entries.

    Note: Outputs are raw similarity-weighted scores (NOT calibrated probabilities).
    """
    if not entries:
        return {}
    if k_neighbors <= 0:
        raise ValueError(f"k_neighbors must be > 0, got {k_neighbors}")

    sims = batch_dot(query_vec, [e.vector for e in entries])
    scored_pairs = sorted(
        zip(sims, entries),
        key=lambda pair: (-pair[0], pair[1].sample.sample_id),
    )
    top_neighbors = scored_pairs[: min(k_neighbors, len(scored_pairs))]

    # Shift similarities into non-negative weights via max(sim, 0.0) and normalize by total weight
    # so scores reflect local neighbor affinity while remaining clearly flagged as raw KNN scores.
    raw_by_label: Dict[str, float] = {}
    for sim, entry in top_neighbors:
        label = label_getter(entry)
        weight = max(float(sim), 0.0)
        raw_by_label[label] = raw_by_label.get(label, 0.0) + weight

    total_weight = sum(raw_by_label.values())
    if total_weight <= 0.0:
        # Fallback to best raw cosine similarity per label among top neighbors
        best_by_label: Dict[str, float] = {}
        for sim, entry in top_neighbors:
            label = label_getter(entry)
            if label not in best_by_label or sim > best_by_label[label]:
                best_by_label[label] = float(sim)
        return best_by_label

    return {label: score / total_weight for label, score in raw_by_label.items()}


class TwoStageHierarchicalKNNClassifier:
    """Two-stage hierarchical KNN classifier implementing P2.1, P2.2, and P2.3 (D-2)."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        cls_index: HierarchicalClsIndex,
        config: HierarchicalKNNConfig,
        score_aggregator: Optional[ScoreAggregatorFn] = None,
        calibrator: Optional[CalibratorFn] = None,
    ) -> None:
        self.embedder = embedder
        self.cls_index = cls_index
        self.config = config
        self.score_aggregator = score_aggregator
        self.calibrator = calibrator

    def _require_runtime_params(self) -> Tuple[int, int, int]:
        """Ensure open D-2 KNN parameters are explicitly configured before inference."""
        missing: List[str] = []
        if self.config.stage1_top_k is None:
            missing.append("stage1_top_k")
        if self.config.stage1_neighbors is None:
            missing.append("stage1_neighbors")
        if self.config.stage2_neighbors is None:
            missing.append("stage2_neighbors")
        if missing:
            raise ValueError(
                "Open D-2 hyperparameter(s) must be explicitly specified before running "
                f"TwoStageHierarchicalKNNClassifier: {', '.join(missing)}. "
                "Pass a configured HierarchicalKNNConfig (no silent production defaults)."
            )
        assert self.config.stage1_top_k is not None
        assert self.config.stage1_neighbors is not None
        assert self.config.stage2_neighbors is not None
        if self.config.stage1_top_k <= 0:
            raise ValueError(f"stage1_top_k must be > 0, got {self.config.stage1_top_k}")
        return (
            self.config.stage1_top_k,
            self.config.stage1_neighbors,
            self.config.stage2_neighbors,
        )

    def embed_event(self, event: EventNormalized, role: Optional[str] = None) -> List[float]:
        """P2.1: Embed normalized event into a 384d L2-normalized vector."""
        text = event.to_classification_text(role=role)
        return l2_normalize(self.embedder.encode_query(text))

    def run_stage1_coarse(
        self,
        query_vec: Sequence[float],
    ) -> Tuple[List[str], Dict[str, float]]:
        """P2.2 (Stage 1): Rank parent groups (C1..C5) and select Top-k candidates."""
        stage1_top_k, stage1_neighbors, _ = self._require_runtime_params()
        entries = self.cls_index.get_stage1_entries()
        if not entries:
            return [], {}

        stage1_scores = _knn_similarity_scores(
            query_vec=query_vec,
            entries=entries,
            label_getter=lambda e: e.sample.parent_group_id,
            k_neighbors=stage1_neighbors,
        )
        ranked_groups = sorted(stage1_scores.items(), key=lambda kv: (-kv[1], kv[0]))

        min_thresh = self.config.stage1_min_score_threshold
        selected: List[str] = []
        for grp, score in ranked_groups:
            if min_thresh is not None and score < min_thresh:
                continue
            selected.append(grp)
            if len(selected) >= stage1_top_k:
                break

        return selected, stage1_scores

    def run_stage2_fine_union(
        self,
        query_vec: Sequence[float],
        selected_candidate_groups: Sequence[str],
    ) -> Dict[str, float]:
        """P2.3 (Stage 2): Retrieve datasets for ALL selected groups and run KNN over their UNION."""
        _, _, stage2_neighbors = self._require_runtime_params()
        union_entries = self.cls_index.get_stage2_union_entries(selected_candidate_groups)
        if not union_entries:
            return {}

        return _knn_similarity_scores(
            query_vec=query_vec,
            entries=union_entries,
            label_getter=lambda e: e.sample.rubric_item_id,
            k_neighbors=stage2_neighbors,
        )

    def classify_vector(
        self,
        event_id: str,
        query_vec: Sequence[float],
        top_n_items: int = 5,
    ) -> ClassificationResult:
        """Run full two-stage hierarchical KNN on an already-computed 384d query vector."""
        norm_vec = l2_normalize(query_vec)
        selected_groups, stage1_scores = self.run_stage1_coarse(norm_vec)

        if not selected_groups:
            return ClassificationResult(
                event_id=event_id,
                rubric_item_id=None,
                parent_group_id=None,
                selected_candidate_groups=[],
                stage1_scores=stage1_scores,
                stage2_scores={},
                aggregated_scores=None,
                calibrated_confidence=None,
                is_calibrated=False,
                status="rejected" if self.config.rejection_policy else "uncalibrated",
                top_k_items=[],
                model_version=self.embedder.model_version,
            )

        stage2_scores = self.run_stage2_fine_union(norm_vec, selected_groups)
        if not stage2_scores:
            return ClassificationResult(
                event_id=event_id,
                rubric_item_id=None,
                parent_group_id=None,
                selected_candidate_groups=selected_groups,
                stage1_scores=stage1_scores,
                stage2_scores={},
                aggregated_scores=None,
                calibrated_confidence=None,
                is_calibrated=False,
                status="rejected" if self.config.rejection_policy else "uncalibrated",
                top_k_items=[],
                model_version=self.embedder.model_version,
            )

        item_to_parent = {
            item_id: parent_group_from_item_id(item_id) for item_id in stage2_scores
        }

        # Score aggregation: never blindly multiply stage1 * stage2 (D-2 Requirement 6).
        aggregated_scores: Optional[Dict[str, float]] = None
        if self.score_aggregator is not None:
            aggregated_scores = self.score_aggregator(stage1_scores, stage2_scores, item_to_parent)
            ranking_source = aggregated_scores
        else:
            # Explicit Stage-2 union ranking when no custom fusion strategy is attached.
            ranking_source = stage2_scores

        ranked_items = sorted(ranking_source.items(), key=lambda kv: (-kv[1], kv[0]))
        top_item_id = ranked_items[0][0]
        top_parent_group = item_to_parent[top_item_id]

        top_k_items = [
            {
                "rubric_item_id": iid,
                "parent_group_id": item_to_parent[iid],
                "stage2_score": float(stage2_scores[iid]),
                "stage1_parent_score": float(stage1_scores.get(item_to_parent[iid], 0.0)),
                "ranking_score": float(score),
            }
            for iid, score in ranked_items[:top_n_items]
        ]

        # Calibration & acceptance gate (D-2 Requirement 5: raw scores != calibrated probabilities)
        is_calibrated = False
        calibrated_confidence: Optional[float] = None
        status = "uncalibrated"

        if self.calibrator is not None:
            prob = float(
                self.calibrator(top_item_id, stage1_scores, stage2_scores, aggregated_scores)
            )
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"Calibrator returned out-of-range probability {prob}")
            is_calibrated = True
            calibrated_confidence = prob
            if self.config.acceptance_threshold is not None:
                status = (
                    "accepted"
                    if calibrated_confidence >= self.config.acceptance_threshold
                    else "pending_review"
                )
            else:
                status = "pending_review"

        result = ClassificationResult(
            event_id=event_id,
            rubric_item_id=top_item_id,
            parent_group_id=top_parent_group,
            selected_candidate_groups=selected_groups,
            stage1_scores=stage1_scores,
            stage2_scores=stage2_scores,
            aggregated_scores=aggregated_scores,
            calibrated_confidence=calibrated_confidence,
            is_calibrated=is_calibrated,
            status=status,
            top_k_items=top_k_items,
            model_version=self.embedder.model_version,
        )

        validation = validate_classification_result(result)
        if not validation.is_valid:
            msgs = "; ".join(i.message for i in validation.errors)
            raise RuntimeError(f"Classifier produced invalid ClassificationResult: {msgs}")

        return result

    def classify_event(
        self,
        event: EventNormalized,
        role: Optional[str] = None,
        top_n_items: int = 5,
    ) -> Tuple[ClassificationResult, List[float]]:
        """Run P2.1 -> P2.2 -> P2.3 on a normalized event, returning (result, query_vector)."""
        query_vec = self.embed_event(event, role=role)
        result = self.classify_vector(
            event_id=event.event_id,
            query_vec=query_vec,
            top_n_items=top_n_items,
        )
        return result, query_vec
