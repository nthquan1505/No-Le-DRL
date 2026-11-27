"""Provisional confidence handling, abstention, and rubric boundary guards.

Statuses supported:
  - PREDICTED
  - NEEDS_REVIEW
  - NO_MATCH
  - SKIPPED_NOT_EVENT

IMPORTANT:
  - All thresholds are provisional (`provisional_not_evaluated`) and NOT calibrated.
  - Raw similarity and fusion scores are NEVER treated as calibrated probabilities.
  - Event classification is strictly separated from participation verification,
    organizer/collaborator role eligibility, and DRL point calculation.
  - `NONE` is NEVER auto-predicted without an explicit validated negative classifier.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.classification.algorithms.base_algorithm import AlgorithmPrediction
from src.classification.embeddings.event_text_builder import normalize_event_text
from src.classification.retrieval.rubric_retriever import RubricRetriever


# Explicit mobilization cues derived directly from rubric.yaml item 3.3 keywords & inclusion_rules
DEFAULT_MOBILIZATION_CUES = (
    "huy động lực lượng",
    "huy động sinh viên",
    "lực lượng hỗ trợ",
    "kêu gọi sinh viên tham gia hỗ trợ",
    "cổ vũ theo phân công",
    "huy động cổ vũ",
    "triệu tập tham dự",
    "triệu tập sinh viên",
)


class ConfidenceAndDecisionHandler:
    """Converts standardized AlgorithmPrediction outputs into final classification payloads."""

    def __init__(
        self,
        *,
        rubric_retriever: RubricRetriever,
        decision_config: Dict[str, Any],
        max_suggested_candidates: int = 5,
    ) -> None:
        self.rubric_retriever = rubric_retriever
        self.cfg = dict(decision_config)
        self.max_suggested_candidates = int(max_suggested_candidates)

        # Pull explicit 3.3 keywords from rubric.yaml to augment default mobilization cues
        info_3_3 = self.rubric_retriever.get_criterion_info("3.3")
        rubric_cues = [normalize_event_text(k).lower() for k in info_3_3.get("keywords") or [] if k]
        self._mobilization_cues = tuple(dict.fromkeys([*DEFAULT_MOBILIZATION_CUES, *rubric_cues]))

    @staticmethod
    def build_skipped_result(
        *,
        algorithm_name: str,
        skip_reason: str,
    ) -> Dict[str, Any]:
        """Build classification payload for non-events or records without usable text."""
        return {
            "status": "SKIPPED_NOT_EVENT",
            "algorithm": algorithm_name,
            "predicted_group_id": None,
            "predicted_criterion_ref": None,
            "predicted_criterion_key": None,
            "predicted_criterion_name": None,
            "suggested_candidates": [],
            "scores": {
                "stage1": {"sample": [], "rubric": [], "fused": []},
                "stage2": {"sample": [], "rubric": [], "fused": []},
            },
            "confidence": {
                "group_margin": 0.0,
                "criterion_margin": 0.0,
                "top_group_score": None,
                "top_criterion_score": None,
                "top_raw_sample_similarity": None,
                "top_raw_rubric_similarity": None,
                "branch_agreement": None,
                "evidence_verification_status": None,
                "calibrated": False,
                "threshold_status": "provisional_not_evaluated",
            },
            "evidence": {
                "sample_neighbors": [],
                "rubric_prototypes": [],
                "rubric_context": None,
            },
            "review_reasons": [skip_reason],
        }

    def _has_explicit_mobilization_cue(self, event: Dict[str, Any]) -> bool:
        title = normalize_event_text(event.get("title")).lower()
        desc = normalize_event_text(event.get("description")).lower()
        combined = f"{title} {desc}"
        return any(cue in combined for cue in self._mobilization_cues)

    def _build_rubric_context(self, criterion_ref: Optional[str]) -> Optional[Dict[str, Any]]:
        if not criterion_ref:
            return None
        info = self.rubric_retriever.get_criterion_info(criterion_ref)
        if not info:
            return None
        return {
            "criterion_ref": criterion_ref,
            "criterion_key": info.get("criterion_key"),
            "group_id": info.get("group_id"),
            "tieu_chi": info.get("tieu_chi"),
            "event_matchable": info.get("event_matchable"),
            "mvp_scope": info.get("mvp_scope"),
            "anchors_status": info.get("anchors_status"),
            "scope_note": info.get("scope_note"),
            "uncertain": info.get("uncertain", []),
            "open_questions": self.rubric_retriever.get_relevant_open_questions(criterion_ref),
        }

    def _format_suggested_candidates(
        self,
        stage2_fused: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        suggestions: List[Dict[str, Any]] = []
        for entry in stage2_fused[: self.max_suggested_candidates]:
            ref = entry["criterion_ref"]
            info = self.rubric_retriever.get_criterion_info(ref)
            suggestions.append(
                {
                    "rank": entry["rank"],
                    "group_id": entry["group_id"],
                    "criterion_ref": ref,
                    "criterion_key": entry["criterion_key"],
                    "criterion_name": entry.get("criterion_name") or info.get("name"),
                    "fused_score": entry["fused_score"],
                    "sample_raw_score": entry["sample_raw_score"],
                    "sample_evidence_status": entry["sample_evidence_status"],
                    "rubric_raw_score": entry["rubric_raw_score"],
                    "rubric_evidence_status": entry["rubric_evidence_status"],
                    "evidence_verification_status": entry.get("evidence_verification_status"),
                    "criterion_anchors_status": entry.get("criterion_anchors_status"),
                    "missing_branches": entry.get("missing_branches", []),
                }
            )
        return suggestions

    def decide(
        self,
        *,
        prediction: AlgorithmPrediction,
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply provisional confidence and rubric boundary checks to produce final decision."""
        no_match_thresh = float(self.cfg.get("no_match_max_raw_similarity", 0.70))
        min_raw_sim_predict = float(self.cfg.get("min_top_raw_similarity_for_predict", 0.78))
        min_group_margin = float(self.cfg.get("min_group_margin", 0.05))
        min_crit_margin = float(self.cfg.get("min_criterion_margin", 0.08))
        require_agreement = bool(self.cfg.get("require_branch_agreement_when_both_available", True))
        route_draft_only = bool(self.cfg.get("route_draft_only_to_review", True))
        boundary_31_35_margin = float(self.cfg.get("boundary_3_1_vs_3_5_margin", 0.18))
        guard_33 = bool(self.cfg.get("guard_3_3_requires_mobilization_cue", True))
        guard_19_margin = float(self.cfg.get("guard_1_9_catch_all_margin", 0.12))
        threshold_status = str(self.cfg.get("threshold_status", "provisional_not_evaluated"))

        s1_fused = prediction.stage1_fused_scores
        s2_fused = prediction.stage2_fused_scores
        top_group_score = float(s1_fused[0]["fused_score"]) if s1_fused else None
        top_crit_score = float(s2_fused[0]["fused_score"]) if s2_fused else None

        raw_sims = [
            s
            for s in (prediction.top_raw_sample_similarity, prediction.top_raw_rubric_similarity)
            if s is not None
        ]
        best_raw_sim = max(raw_sims) if raw_sims else 0.0

        top_cand = prediction.top_candidate
        second_cand = s2_fused[1] if len(s2_fused) >= 2 else None
        suggested = self._format_suggested_candidates(s2_fused)

        scores_block = {
            "stage1": {
                "sample": prediction.stage1_sample_scores,
                "rubric": prediction.stage1_rubric_scores,
                "fused": s1_fused,
            },
            "stage2": {
                "sample": prediction.stage2_sample_scores,
                "rubric": prediction.stage2_rubric_scores,
                "fused": s2_fused,
            },
        }

        # Case 1: No candidate retrieved OR max raw similarity below NO_MATCH threshold
        if top_cand is None or best_raw_sim < no_match_thresh:
            reason = (
                "no_eligible_criterion_candidates_retrieved"
                if top_cand is None
                else (
                    f"no_sufficiently_supported_criterion "
                    f"(max_raw_similarity={best_raw_sim:.4f} < provisional_no_match_threshold={no_match_thresh:.2f}; "
                    "NONE not auto-assigned without validated negative classifier)"
                )
            )
            return {
                "status": "NO_MATCH",
                "algorithm": prediction.algorithm_name,
                "predicted_group_id": None,
                "predicted_criterion_ref": None,
                "predicted_criterion_key": None,
                "predicted_criterion_name": None,
                "vector_routing_used": (prediction.algorithm_metadata or {}).get("vector_routing_used"),
                "suggested_candidates": suggested,
                "scores": scores_block,
                "confidence": {
                    "group_margin": prediction.group_margin,
                    "criterion_margin": prediction.criterion_margin,
                    "top_group_score": top_group_score,
                    "top_criterion_score": top_crit_score,
                    "top_raw_sample_similarity": prediction.top_raw_sample_similarity,
                    "top_raw_rubric_similarity": prediction.top_raw_rubric_similarity,
                    "branch_agreement": prediction.branch_agreement,
                    "evidence_verification_status": (
                        top_cand.get("evidence_verification_status") if top_cand else None
                    ),
                    "calibrated": False,
                    "threshold_status": threshold_status,
                },
                "evidence": {
                    "sample_neighbors": prediction.sample_neighbors,
                    "rubric_prototypes": prediction.rubric_prototypes,
                    "rubric_context": self._build_rubric_context(
                        top_cand["criterion_ref"] if top_cand else None
                    ),
                },
                "review_reasons": [reason],
            }

        # Evaluate eligibility, provisional confidence, and rubric-specific boundary guards
        top_ref = str(top_cand["criterion_ref"])
        top_info = self.rubric_retriever.get_criterion_info(top_ref)
        top_verification = top_cand.get("evidence_verification_status")
        review_reasons: List[str] = []

        # Guard A: Strict rubric event_matchable=true & positive target enforcement
        if not top_info or not top_info.get("event_matchable") or top_info.get("group_id") == "P":
            review_reasons.append(
                f"top_candidate_not_positive_event_matchable_criterion ({top_ref})"
            )

        # Guard B: Draft-only or missing-sample evidence must not be presented as verified certainty
        if route_draft_only and top_verification == "draft_only":
            review_reasons.append(
                f"draft_only_rubric_evidence_unverified (criterion {top_ref} has anchors_status=draft "
                "and no verified school-list samples)"
            )

        # Guard C: Raw semantic similarity support
        if best_raw_sim < min_raw_sim_predict:
            review_reasons.append(
                f"low_raw_similarity_support (max_raw_similarity={best_raw_sim:.4f} < {min_raw_sim_predict:.2f})"
            )

        # Guard D: Stage 1 & Stage 2 margin checks
        if prediction.group_margin < min_group_margin:
            review_reasons.append(
                f"narrow_stage1_group_margin (group_margin={prediction.group_margin:.4f} < {min_group_margin:.2f})"
            )
        if prediction.criterion_margin < min_crit_margin:
            review_reasons.append(
                f"narrow_stage2_criterion_margin (criterion_margin={prediction.criterion_margin:.4f} < {min_crit_margin:.2f})"
            )

        # Guard E: Branch agreement when both branches have reference coverage for top candidate
        if (
            require_agreement
            and not top_cand.get("missing_branches")
            and prediction.branch_agreement is False
        ):
            review_reasons.append(
                f"branch_disagreement (sample_top1={prediction.branch_top1_sample_criterion}, "
                f"rubric_top1={prediction.branch_top1_rubric_criterion})"
            )

        # Guard F: Rubric-specific boundary — 3.1 vs 3.5 (rubric.yaml open question Q10)
        pair_refs = {top_ref, str(second_cand["criterion_ref"])} if second_cand else {top_ref}
        branch_pair = {
            prediction.branch_top1_sample_criterion,
            prediction.branch_top1_rubric_criterion,
        }
        if (
            (pair_refs == {"3.1", "3.5"} and prediction.criterion_margin < boundary_31_35_margin)
            or branch_pair == {"3.1", "3.5"}
        ):
            review_reasons.append(
                "unresolved_rubric_boundary_3_1_vs_3_5 (Q10: requires human confirmation of competition level/scope)"
            )

        # Guard G: Rubric-specific constraint — 3.3 requires explicit student mobilization evidence
        if guard_33 and top_ref == "3.3" and not self._has_explicit_mobilization_cue(event):
            review_reasons.append(
                "criterion_3_3_missing_explicit_mobilization_cue "
                "(3.3 requires explicit 'huy động lực lượng / huy động sinh viên' signal, not just a similar event type)"
            )

        # Guard H: Rubric-specific constraint — 1.9 must not act as a catch-all (open question Q23)
        if top_ref == "1.9" and second_cand is not None:
            second_ref = str(second_cand["criterion_ref"])
            if prediction.criterion_margin < guard_19_margin:
                review_reasons.append(
                    f"criterion_1_9_catch_all_guard (Q23: narrow margin={prediction.criterion_margin:.4f} "
                    f"against candidate {second_ref}; verify academic/career scope vs movement activity)"
                )

        # Build final status payload
        rubric_context = self._build_rubric_context(top_ref)
        confidence_block = {
            "group_margin": prediction.group_margin,
            "criterion_margin": prediction.criterion_margin,
            "top_group_score": top_group_score,
            "top_criterion_score": top_crit_score,
            "top_raw_sample_similarity": prediction.top_raw_sample_similarity,
            "top_raw_rubric_similarity": prediction.top_raw_rubric_similarity,
            "branch_agreement": prediction.branch_agreement,
            "evidence_verification_status": top_verification,
            "calibrated": False,
            "threshold_status": threshold_status,
        }
        evidence_block = {
            "sample_neighbors": prediction.sample_neighbors,
            "rubric_prototypes": prediction.rubric_prototypes,
            "rubric_context": rubric_context,
        }

        vector_routing = (prediction.algorithm_metadata or {}).get("vector_routing_used")

        if review_reasons:
            res = {
                "status": "NEEDS_REVIEW",
                "algorithm": prediction.algorithm_name,
                "predicted_group_id": None,
                "predicted_criterion_ref": None,
                "predicted_criterion_key": None,
                "predicted_criterion_name": None,
                "suggested_candidates": suggested,
                "scores": scores_block,
                "confidence": confidence_block,
                "evidence": evidence_block,
                "review_reasons": review_reasons,
            }
            if vector_routing:
                res["vector_routing_used"] = vector_routing
            return res

        res = {
            "status": "PREDICTED",
            "algorithm": prediction.algorithm_name,
            "predicted_group_id": top_info["group_id"],
            "predicted_criterion_ref": top_info["criterion_ref"],
            "predicted_criterion_key": top_info["criterion_key"],
            "predicted_criterion_name": top_info["name"],
            "suggested_candidates": suggested,
            "scores": scores_block,
            "confidence": confidence_block,
            "evidence": evidence_block,
            "review_reasons": [],
        }
        if vector_routing:
            res["vector_routing_used"] = vector_routing
        return res
