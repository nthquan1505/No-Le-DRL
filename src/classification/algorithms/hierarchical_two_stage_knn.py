"""Hierarchical 2-Stage Dual-Branch kNN Classification Algorithm with Stage-wise Fusion.

Implements the complete dual-branch retrieval, aggregation, and Stage-wise Fusion logic:
  - Stage 1:
      * Sample branch kNN (Top-K1 weighted voting by group_id)
      * Rubric branch group retrieval (stage = group, unbiased per group_id)
      * Stage 1 Fusion: S1(g) = beta * S_sample(g) + (1 - beta) * S_rubric(g)
      * Shared Top-2 major group selection
  - Stage 2:
      * Sample branch kNN filtered strictly to shared Top-2 groups (Top-K2 weighted voting by criterion_ref)
      * Rubric branch prototype retrieval filtered strictly to stage = criterion, event_matchable = true,
        group_id in shared Top-2 groups, aggregated via type-stratified max + evidence-tier weights
        (zero anchor-count bias; verified vs partial vs draft evidence preserved)
      * Stage 2 Fusion: S2(c) = alpha * S_sample(c) + (1 - alpha) * S_rubric(c) over union of
        candidate criterion_refs with explicit missing-branch representation

All numerical parameters are passed in via `params` from `configs/classification_config.yaml`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.classification.algorithms.base_algorithm import (
    AlgorithmPrediction,
    BaseClassificationAlgorithm,
)
from src.classification.fusion.score_fusion import (
    fuse_stage1_groups,
    fuse_stage2_criteria,
    similarity_to_weight,
)
from src.classification.retrieval.rubric_retriever import RubricPrototypeMatch
from src.classification.retrieval.sample_retriever import SampleNeighbor


class HierarchicalTwoStageKNN(BaseClassificationAlgorithm):
    """Dual-branch hierarchical 2-stage kNN classifier with Stage-wise Fusion."""

    name: str = "hierarchical_two_stage_knn"

    def _require_param(self, key: str) -> Any:
        if key not in self.params or self.params[key] is None:
            raise ValueError(
                f"Missing required algorithm parameter '{key}' for '{self.name}'. "
                "Specify all parameters in configs/classification_config.yaml."
            )
        return self.params[key]

    # ------------------------------------------------------------------
    # Stage 1 Aggregation Helpers
    # ------------------------------------------------------------------

    def _aggregate_stage1_sample(
        self,
        neighbors: Sequence[SampleNeighbor],
        weighting_mode: str,
        temperature: float,
        power: float,
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Similarity-weighted voting by group_id over Top-K1 nearest samples."""
        if not neighbors:
            return {}, []

        weighted_sum: Dict[str, float] = {}
        neighbor_counts: Dict[str, int] = {}
        best_sim_by_group: Dict[str, float] = {}
        total_weight = 0.0

        for nb in neighbors:
            w = similarity_to_weight(
                nb.similarity,
                mode=weighting_mode,
                temperature=temperature,
                power=power,
            )
            weighted_sum[nb.group_id] = weighted_sum.get(nb.group_id, 0.0) + w
            neighbor_counts[nb.group_id] = neighbor_counts.get(nb.group_id, 0) + 1
            if nb.group_id not in best_sim_by_group or nb.similarity > best_sim_by_group[nb.group_id]:
                best_sim_by_group[nb.group_id] = nb.similarity
            total_weight += w

        scores: Dict[str, float] = {}
        if total_weight > 0.0:
            for grp, w_sum in weighted_sum.items():
                scores[grp] = w_sum / total_weight
        else:
            for grp, best_s in best_sim_by_group.items():
                scores[grp] = best_s

        ranked = [
            {
                "group_id": grp,
                "score": round(float(score), 6),
                "neighbor_count": int(neighbor_counts[grp]),
                "max_similarity": round(float(best_sim_by_group[grp]), 6),
            }
            for grp, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        return scores, ranked

    @staticmethod
    def _aggregate_stage1_rubric(
        group_matches: Sequence[RubricPrototypeMatch],
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Aggregate Stage 1 group prototypes by group_id using max similarity (zero count bias)."""
        best_by_group: Dict[str, RubricPrototypeMatch] = {}
        for m in group_matches:
            if m.group_id not in best_by_group or m.similarity > best_by_group[m.group_id].similarity:
                best_by_group[m.group_id] = m

        scores = {grp: m.similarity for grp, m in best_by_group.items()}
        ranked = [
            {
                "group_id": grp,
                "score": round(float(m.similarity), 6),
                "prototype_id": m.prototype_id,
                "prototype_type": m.prototype_type,
            }
            for grp, m in sorted(best_by_group.items(), key=lambda kv: (-kv[1].similarity, kv[0]))
        ]
        return scores, ranked

    # ------------------------------------------------------------------
    # Stage 2 Aggregation Helpers
    # ------------------------------------------------------------------

    def _aggregate_stage2_sample(
        self,
        neighbors: Sequence[SampleNeighbor],
        weighting_mode: str,
        temperature: float,
        power: float,
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Similarity-weighted voting by criterion_ref over Top-K2 nearest samples."""
        if not neighbors:
            return {}, []

        weighted_sum: Dict[str, float] = {}
        neighbor_counts: Dict[str, int] = {}
        best_sim_by_ref: Dict[str, float] = {}
        ref_to_group: Dict[str, str] = {}
        ref_to_key: Dict[str, str] = {}
        total_weight = 0.0

        for nb in neighbors:
            w = similarity_to_weight(
                nb.similarity,
                mode=weighting_mode,
                temperature=temperature,
                power=power,
            )
            ref = nb.criterion_ref
            weighted_sum[ref] = weighted_sum.get(ref, 0.0) + w
            neighbor_counts[ref] = neighbor_counts.get(ref, 0) + 1
            ref_to_group[ref] = nb.group_id
            ref_to_key[ref] = nb.criterion_key
            if ref not in best_sim_by_ref or nb.similarity > best_sim_by_ref[ref]:
                best_sim_by_ref[ref] = nb.similarity
            total_weight += w

        scores: Dict[str, float] = {}
        if total_weight > 0.0:
            for ref, w_sum in weighted_sum.items():
                scores[ref] = w_sum / total_weight
        else:
            for ref, best_s in best_sim_by_ref.items():
                scores[ref] = best_s

        ranked = [
            {
                "criterion_ref": ref,
                "criterion_key": ref_to_key[ref],
                "group_id": ref_to_group[ref],
                "score": round(float(score), 6),
                "neighbor_count": int(neighbor_counts[ref]),
                "max_similarity": round(float(best_sim_by_ref[ref]), 6),
            }
            for ref, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        return scores, ranked

    @staticmethod
    def _classify_evidence_verification(
        by_type_best: Dict[str, RubricPrototypeMatch],
        criterion_anchors_status: Optional[str],
    ) -> str:
        """Determine transparent verification tier without presenting draft evidence as verified."""
        if criterion_anchors_status == "verified_partial":
            return "verified_partial"
        if "verified_anchor" in by_type_best:
            return "verified_from_school_list"
        if "partial_anchor" in by_type_best:
            return "verified_partial"
        if criterion_anchors_status == "draft" or "draft_anchor" in by_type_best:
            return "draft_only"
        return "definition_only"

    def _aggregate_stage2_rubric(
        self,
        prototype_matches: Sequence[RubricPrototypeMatch],
        type_weights: Dict[str, float],
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Type-stratified max + evidence-tier weighted aggregation by criterion_ref.

        Guarantees:
          - Never sums all prototype similarities.
          - Takes MAX per prototype_type so criteria with 18 anchors gain ZERO count advantage
            over criteria with 3 anchors.
          - Weights primary evidence (criterion_definition, verified_anchor) higher than
            partial_anchor, keyword_context, and draft_anchor.
          - Preserves prototype_type, anchor_status, and verification tier explicitly.
        """
        grouped: Dict[str, List[RubricPrototypeMatch]] = {}
        for m in prototype_matches:
            if not m.criterion_ref or not m.event_matchable:
                continue
            grouped.setdefault(m.criterion_ref, []).append(m)

        scores: Dict[str, float] = {}
        ranked_entries: List[Dict[str, Any]] = []
        details_by_ref: Dict[str, Dict[str, Any]] = {}

        for ref, matches in grouped.items():
            # Max similarity within each prototype_type (eliminates anchor-count bias)
            by_type_best: Dict[str, RubricPrototypeMatch] = {}
            for m in matches:
                ptype = m.prototype_type
                if ptype not in by_type_best or m.similarity > by_type_best[ptype].similarity:
                    by_type_best[ptype] = m

            weighted_num = 0.0
            weighted_den = 0.0
            type_breakdown: Dict[str, Dict[str, Any]] = {}
            for ptype, best_m in sorted(by_type_best.items()):
                w = float(type_weights.get(ptype, 0.5))
                weighted_num += w * best_m.similarity
                weighted_den += w
                type_breakdown[ptype] = {
                    "prototype_id": best_m.prototype_id,
                    "similarity": round(float(best_m.similarity), 6),
                    "anchor_status": best_m.anchor_status,
                    "type_weight": w,
                }

            agg_score = (weighted_num / weighted_den) if weighted_den > 0.0 else 0.0
            best_overall = max(matches, key=lambda x: (x.similarity, x.prototype_id))
            crit_anchors_status = best_overall.criterion_anchors_status
            verification_tier = self._classify_evidence_verification(by_type_best, crit_anchors_status)

            scores[ref] = agg_score
            entry = {
                "criterion_ref": ref,
                "criterion_key": best_overall.criterion_key,
                "group_id": best_overall.group_id,
                "score": round(float(agg_score), 6),
                "max_prototype_similarity": round(float(best_overall.similarity), 6),
                "best_prototype_id": best_overall.prototype_id,
                "best_prototype_type": best_overall.prototype_type,
                "best_anchor_status": best_overall.anchor_status,
                "criterion_anchors_status": crit_anchors_status,
                "evidence_verification_status": verification_tier,
                "prototype_type_breakdown": type_breakdown,
            }
            ranked_entries.append(entry)
            details_by_ref[ref] = entry

        ranked_entries.sort(
            key=lambda e: (-e["score"], -e["max_prototype_similarity"], e["criterion_ref"])
        )
        return scores, ranked_entries, details_by_ref

    # ------------------------------------------------------------------
    # Main Classification Entrypoint
    # ------------------------------------------------------------------

    def _resolve_branch_vector(
        self,
        profile_name: str,
        fallback_vec: np.ndarray,
        query_vectors_by_profile: Optional[Dict[str, np.ndarray]],
    ) -> Tuple[np.ndarray, str]:
        if query_vectors_by_profile and profile_name in query_vectors_by_profile:
            vec = np.asarray(query_vectors_by_profile[profile_name], dtype=np.float32).reshape(-1)
            return vec, profile_name
        return fallback_vec, "shared_default"

    def classify(
        self,
        query_vector: np.ndarray,
        event_context: Optional[Dict[str, Any]] = None,
        query_vectors_by_profile: Optional[Dict[str, np.ndarray]] = None,
    ) -> AlgorithmPrediction:
        """Run independent dual-branch Stage 1 + Fusion -> shared Top-2 -> Stage 2 + Fusion.

        Supports configurable vector profile selection per branch (and optional per-stage
        override) across the 3 classification profiles: `title`, `title_description`, and
        `title_description_format`.
        """
        k1 = int(self._require_param("k1"))
        k2 = int(self._require_param("k2"))
        candidate_group_count = int(self._require_param("candidate_group_count"))
        beta = float(self._require_param("stage1_fusion_beta"))
        alpha = float(self._require_param("stage2_fusion_alpha"))

        s1_weight_mode = str(self._require_param("stage1_similarity_weighting"))
        s1_temp = float(self.params.get("stage1_weight_temperature", 0.1))
        s1_power = float(self.params.get("stage1_weight_power", 2.0))
        s1_norm_method = str(self._require_param("stage1_score_normalization"))

        s2_weight_mode = str(self._require_param("stage2_similarity_weighting"))
        s2_temp = float(self.params.get("stage2_weight_temperature", 0.1))
        s2_power = float(self.params.get("stage2_weight_power", 2.0))
        s2_norm_method = str(self._require_param("stage2_score_normalization"))

        type_weights = dict(self._require_param("rubric_prototype_type_weights"))
        missing_policy = str(self._require_param("missing_branch_reference_policy"))
        top_sample_out = int(self.params.get("top_sample_neighbors_in_output", 5))
        top_rubric_out = int(self.params.get("top_rubric_prototypes_in_output", 5))

        # Resolve per-branch / per-stage vector routing (falls back cleanly to `query_vector`)
        shared_q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        sample_branch_pref = str(self.params.get("sample_branch_vector") or "title_description")
        rubric_branch_pref = str(self.params.get("rubric_branch_vector") or "title_description")

        sample_s1_pref = str(self.params.get("sample_stage1_vector") or sample_branch_pref)
        sample_s2_pref = str(self.params.get("sample_stage2_vector") or sample_branch_pref)
        rubric_s1_pref = str(self.params.get("rubric_stage1_vector") or rubric_branch_pref)
        rubric_s2_pref = str(self.params.get("rubric_stage2_vector") or rubric_branch_pref)

        q_sample_s1, used_sample_s1 = self._resolve_branch_vector(
            sample_s1_pref, shared_q, query_vectors_by_profile
        )
        q_sample_s2, used_sample_s2 = self._resolve_branch_vector(
            sample_s2_pref, shared_q, query_vectors_by_profile
        )
        q_rubric_s1, used_rubric_s1 = self._resolve_branch_vector(
            rubric_s1_pref, shared_q, query_vectors_by_profile
        )
        q_rubric_s2, used_rubric_s2 = self._resolve_branch_vector(
            rubric_s2_pref, shared_q, query_vectors_by_profile
        )

        # ==============================================================
        # STAGE 1: Independent Sample & Rubric Group Retrieval + Fusion
        # ==============================================================
        s1_sample_neighbors = self.sample_retriever.retrieve_stage1(q_sample_s1, top_k=k1)
        s1_sample_raw, s1_sample_ranked = self._aggregate_stage1_sample(
            s1_sample_neighbors,
            weighting_mode=s1_weight_mode,
            temperature=s1_temp,
            power=s1_power,
        )

        s1_rubric_matches = self.rubric_retriever.retrieve_stage1_groups(q_rubric_s1)
        s1_rubric_raw, s1_rubric_ranked = self._aggregate_stage1_rubric(s1_rubric_matches)

        eligible_groups = sorted(
            set(self.rubric_retriever.eligible_stage1_groups)
            | set(self.sample_retriever.available_groups)
        )

        s1_fused, _, _ = fuse_stage1_groups(
            eligible_groups=eligible_groups,
            sample_raw_scores=s1_sample_raw,
            rubric_raw_scores=s1_rubric_raw,
            beta=beta,
            normalization_method=s1_norm_method,
            top_k_groups=candidate_group_count,
        )

        # Shared Top-2 major groups passed identically to BOTH branches in Stage 2
        selected_groups = [entry["group_id"] for entry in s1_fused if entry["selected_for_stage2"]]

        group_margin = (
            float(s1_fused[0]["fused_score"] - s1_fused[1]["fused_score"])
            if len(s1_fused) >= 2
            else (float(s1_fused[0]["fused_score"]) if s1_fused else 0.0)
        )

        # ==============================================================
        # STAGE 2: Independent Sample & Rubric Criterion Retrieval + Fusion
        # ==============================================================
        s2_sample_neighbors = self.sample_retriever.retrieve_stage2(
            q_sample_s2,
            candidate_groups=selected_groups,
            top_k=k2,
        )
        s2_sample_raw, s2_sample_ranked = self._aggregate_stage2_sample(
            s2_sample_neighbors,
            weighting_mode=s2_weight_mode,
            temperature=s2_temp,
            power=s2_power,
        )

        s2_rubric_matches = self.rubric_retriever.retrieve_stage2_criteria(
            q_rubric_s2,
            candidate_groups=selected_groups,
        )
        s2_rubric_raw, s2_rubric_ranked, rubric_details_by_ref = self._aggregate_stage2_rubric(
            s2_rubric_matches,
            type_weights=type_weights,
        )

        sample_avail_refs = self.sample_retriever.get_available_criterion_refs(selected_groups)
        rubric_avail_refs = self.rubric_retriever.get_available_criterion_refs(selected_groups)
        candidate_union_refs = sorted(set(s2_sample_raw.keys()) | set(s2_rubric_raw.keys()))

        ref_to_meta = {
            ref: self.rubric_retriever.get_criterion_info(ref) for ref in candidate_union_refs
        }

        s2_fused, _, _ = fuse_stage2_criteria(
            candidate_refs=candidate_union_refs,
            sample_raw_scores=s2_sample_raw,
            rubric_raw_scores=s2_rubric_raw,
            sample_available_refs=sample_avail_refs,
            rubric_available_refs=rubric_avail_refs,
            ref_to_meta=ref_to_meta,
            alpha=alpha,
            normalization_method=s2_norm_method,
            missing_branch_policy=missing_policy,
        )

        # Attach rubric verification status onto fused Stage 2 candidates for traceability
        for entry in s2_fused:
            ref = entry["criterion_ref"]
            r_detail = rubric_details_by_ref.get(ref) or {}
            entry["evidence_verification_status"] = r_detail.get("evidence_verification_status")
            entry["criterion_anchors_status"] = r_detail.get("criterion_anchors_status")

        criterion_margin = (
            float(s2_fused[0]["fused_score"] - s2_fused[1]["fused_score"])
            if len(s2_fused) >= 2
            else (float(s2_fused[0]["fused_score"]) if s2_fused else 0.0)
        )

        top_sample_ref = s2_sample_ranked[0]["criterion_ref"] if s2_sample_ranked else None
        top_rubric_ref = s2_rubric_ranked[0]["criterion_ref"] if s2_rubric_ranked else None

        if top_sample_ref is not None and top_rubric_ref is not None:
            branch_agreement: Optional[bool] = top_sample_ref == top_rubric_ref
        else:
            branch_agreement = None

        top_raw_sample_sim = (
            float(s2_sample_neighbors[0].similarity) if s2_sample_neighbors else None
        )
        top_raw_rubric_sim = (
            float(s2_rubric_matches[0].similarity) if s2_rubric_matches else None
        )

        return AlgorithmPrediction(
            algorithm_name=self.name,
            selected_groups=selected_groups,
            stage1_sample_scores=s1_sample_ranked,
            stage1_rubric_scores=s1_rubric_ranked,
            stage1_fused_scores=s1_fused,
            stage2_sample_scores=s2_sample_ranked,
            stage2_rubric_scores=s2_rubric_ranked,
            stage2_fused_scores=s2_fused,
            sample_neighbors=[nb.to_dict() for nb in s2_sample_neighbors[:top_sample_out]],
            rubric_prototypes=[m.to_dict() for m in s2_rubric_matches[:top_rubric_out]],
            group_margin=round(group_margin, 6),
            criterion_margin=round(criterion_margin, 6),
            top_raw_sample_similarity=(
                round(top_raw_sample_sim, 6) if top_raw_sample_sim is not None else None
            ),
            top_raw_rubric_similarity=(
                round(top_raw_rubric_sim, 6) if top_raw_rubric_sim is not None else None
            ),
            branch_top1_sample_criterion=top_sample_ref,
            branch_top1_rubric_criterion=top_rubric_ref,
            branch_agreement=branch_agreement,
            top_candidate=s2_fused[0] if s2_fused else None,
            criterion_rubric_details=rubric_details_by_ref,
            algorithm_metadata={
                "parameter_status": "provisional_not_evaluated",
                "k1": k1,
                "k2": k2,
                "beta": beta,
                "alpha": alpha,
                "candidate_group_count": candidate_group_count,
                "missing_branch_reference_policy": missing_policy,
                "vector_routing_used": {
                    "sample_stage1": used_sample_s1,
                    "sample_stage2": used_sample_s2,
                    "rubric_stage1": used_rubric_s1,
                    "rubric_stage2": used_rubric_s2,
                },
            },
        )
