"""Unit & integration tests for P7 MMR List Selection (`src/recommendation/selector.py`)."""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence

import numpy as np

from src.embedder import BaseEmbedder
from src.recommendation import (
    DEMO_PERSONAS,
    DemoCreditProvider,
    EligibleCandidate,
    MMRConfig,
    MMRSelector,
    MandatoryPenaltyInfo,
    ScoredCandidate,
    build_similarity_matrices,
    compute_intra_list_diversity,
    compute_rubric_similarity,
    load_recommendation_candidate_pool,
    run_recommendation_demo,
    select_top_n_mmr,
)
from src.recommendation.pipeline import DEFAULT_SAMPLE_EMB_PATH
from src.recommendation.schedule_adapter import ConflictCheckResult, FitAssessment
from src.schemas import CreditEstimate, EventNormalized, ScoredEvent, ScoringWeights
from src.vector_ops import l2_normalize

TZ = timezone(timedelta(hours=7))


class _DeterministicStubEmbedder(BaseEmbedder):
    """Fast deterministic 384d embedder for offline integration tests."""

    @property
    def dimension(self) -> int:
        return 384

    def encode_query(self, text: str) -> List[float]:
        vec = [0.05] * 384
        for idx, ch in enumerate(text.encode("utf-8")):
            vec[idx % 384] += ((ch % 19) + 1) * 0.1
        return l2_normalize(vec)

    def encode_queries(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.encode_query(t) for t in texts]

    def encode_passage(self, text: str) -> List[float]:
        return self.encode_query(text)

    def encode_passages(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.encode_query(t) for t in texts]


def _make_scored_candidate(
    event_id: str,
    title: str,
    *,
    score_total: float,
    rubric_item_id: str = "1.9",
    parent_group_id: str = "C1",
    gain: float = 0.50,
    interest: float = 0.50,
    fit: float = 0.80,
    urgency: float = 0.50,
    penalty: float = 0.0,
    is_mandatory: bool = False,
    event_vec: Optional[List[float]] = None,
) -> ScoredCandidate:
    ev = EventNormalized(
        event_id=event_id,
        title=title,
        online_mode="online",
        fee_vnd=0,
    )
    ce = CreditEstimate(
        event_id=event_id,
        rubric_item_id=rubric_item_id,
        group_id=parent_group_id,
        is_credit_eligible=not is_mandatory,
        expected_points=0.0 if is_mandatory else 5.0,
        min_points=0.0 if is_mandatory else 5.0,
        max_points=0.0 if is_mandatory else 5.0,
        effective_points=0.0 if is_mandatory else 5.0,
        source="rule",
    )
    mi = MandatoryPenaltyInfo(
        is_mandatory=is_mandatory,
        penalty_if_missed=-20.0 if is_mandatory else None,
        reason="Bắt buộc 6.4" if is_mandatory else "",
    )
    cand = EligibleCandidate(
        event=ev,
        classification={
            "status": "PREDICTED",
            "predicted_group_id": parent_group_id,
            "predicted_criterion_ref": rubric_item_id,
        },
        event_vec=event_vec,
        rubric_item_id=rubric_item_id,
        parent_group_id=parent_group_id,
        credit_estimate=ce,
        mandatory_info=mi,
        conflict_result=ConflictCheckResult(has_conflict=False),
        estimated_travel_min=0,
        effective_deadline=datetime(2026, 11, 10, 12, 0, tzinfo=TZ),
    )
    se = ScoredEvent(
        event_id=event_id,
        score_total=score_total,
        gain=gain,
        interest=interest,
        fit=fit,
        urgency=urgency,
        penalty=penalty,
        weights=ScoringWeights(),
        rubric_item_id=rubric_item_id,
        parent_group_id=parent_group_id,
    )
    fit_res = FitAssessment(
        fit_score=fit,
        has_conflict=False,
        in_exam_week=False,
        estimated_travel_min=0,
    )
    return ScoredCandidate(
        scored_event=se,
        candidate=cand,
        fit_assessment=fit_res,
        interest_cos_raw=0.80 if event_vec is not None else None,
        days_to_deadline=5.0,
    )


def _vec_with_cos_to_e0(cos_val: float, orth_axis: int, dim: int = 8) -> List[float]:
    """Return unit vector having exact dot product `cos_val` with axis 0."""
    v = [0.0] * dim
    v[0] = cos_val
    v[orth_axis] = math.sqrt(max(0.0, 1.0 - cos_val * cos_val))
    return v


class TestMMRSelectorCore(unittest.TestCase):
    def test_mmr_diversifies_homogeneous_top_p6_and_improves_ild(self) -> None:
        """Pure P6 Top-3 picks [E1, E2, E3] (all C1 AI); MMR picks [E1, E4, E5] (C1, C4, C3)."""
        e1 = _make_scored_candidate(
            "E1", "Hội thảo AI & ML", score_total=0.85, rubric_item_id="1.9", parent_group_id="C1",
            event_vec=_vec_with_cos_to_e0(1.0, 1),
        )
        e2 = _make_scored_candidate(
            "E2", "Workshop Deep Learning", score_total=0.83, rubric_item_id="1.9", parent_group_id="C1",
            event_vec=_vec_with_cos_to_e0(0.90, 1),
        )
        e3 = _make_scored_candidate(
            "E3", "Seminar Ứng dụng LLM", score_total=0.80, rubric_item_id="1.9", parent_group_id="C1",
            event_vec=_vec_with_cos_to_e0(0.85, 2),
        )
        e4 = _make_scored_candidate(
            "E4", "Ngày hội Hiến máu", score_total=0.76, rubric_item_id="4.3", parent_group_id="C4",
            event_vec=_vec_with_cos_to_e0(0.15, 3),
        )
        e5 = _make_scored_candidate(
            "E5", "Giải chạy bộ Sinh viên khỏe", score_total=0.72, rubric_item_id="3.5", parent_group_id="C3",
            event_vec=_vec_with_cos_to_e0(0.20, 4),
        )

        res = select_top_n_mmr(
            [e1, e2, e3, e4, e5],
            top_n=3,
            lambda_param=0.70,
            alpha_semantic=1.0,
            cosine_rescale_mode="none",
            enable_exploration_slot=False,
        )

        ids = [item.scored.scored_event.event_id for item in res.selected]
        self.assertEqual(ids, ["E1", "E4", "E5"])

        # Exact mathematical checks
        self.assertAlmostEqual(res.selected[0].trace.mmr_score or 0.0, 0.70 * 0.85, places=5)
        self.assertEqual(res.selected[0].trace.rank_delta, 0)

        # E4 promoted from P6 #4 -> MMR #2 (delta = +2)
        self.assertAlmostEqual(
            res.selected[1].trace.mmr_score or 0.0,
            0.70 * 0.76 - 0.30 * 0.15,
            places=5,
        )
        self.assertEqual(res.selected[1].trace.p6_rank, 4)
        self.assertEqual(res.selected[1].trace.mmr_rank, 2)
        self.assertEqual(res.selected[1].trace.rank_delta, 2)
        self.assertEqual(res.selected[1].trace.most_similar_selected_id, "E1")

        # Diversity & group coverage strictly improved over Pure P6 Top-3
        self.assertGreater(res.metrics.mmr_ild, res.metrics.p6_topn_ild)
        self.assertGreater(res.metrics.ild_gain, 0.0)
        self.assertEqual(res.metrics.p6_topn_group_coverage, 1)
        self.assertEqual(res.metrics.mmr_group_coverage, 3)

    def test_e5_cosine_anisotropy_rescaling_restores_diversity(self) -> None:
        """Verify why cosine rescaling is required under E5-small compression ([0.75, 0.90])."""
        e1 = _make_scored_candidate(
            "E1", "Hội thảo AI", score_total=0.85, rubric_item_id="1.9", parent_group_id="C1",
            event_vec=_vec_with_cos_to_e0(1.0, 1),
        )
        # Near-duplicate AI event: raw cos with E1 = 0.90
        e2 = _make_scored_candidate(
            "E2", "Workshop AI nâng cao", score_total=0.83, rubric_item_id="1.9", parent_group_id="C1",
            event_vec=_vec_with_cos_to_e0(0.90, 1),
        )
        # Unrelated event under E5 anisotropy: raw cos with E1 = 0.75 (compressed!)
        e4 = _make_scored_candidate(
            "E4", "Hiến máu tình nguyện", score_total=0.76, rubric_item_id="4.3", parent_group_id="C4",
            event_vec=_vec_with_cos_to_e0(0.75, 2),
        )

        # Without rescaling: 0.7*0.83 - 0.3*0.90 = 0.311 > 0.7*0.76 - 0.3*0.75 = 0.307 -> E2 wrongly wins!
        res_raw = select_top_n_mmr(
            [e1, e2, e4],
            top_n=2,
            lambda_param=0.70,
            alpha_semantic=1.0,
            cosine_rescale_mode="none",
            enable_exploration_slot=False,
        )
        self.assertEqual([x.scored.scored_event.event_id for x in res_raw.selected], ["E1", "E2"])
        self.assertEqual(res_raw.similarity_calibration.mode_applied, "raw")

        # With pool_minmax rescaling: raw [0.75..0.90] stretches -> E4 beats E2 decisively!
        res_scaled = select_top_n_mmr(
            [e1, e2, e4],
            top_n=2,
            lambda_param=0.70,
            alpha_semantic=1.0,
            cosine_rescale_mode="pool_minmax",
            enable_exploration_slot=False,
        )
        self.assertEqual([x.scored.scored_event.event_id for x in res_scaled.selected], ["E1", "E4"])
        self.assertEqual(res_scaled.similarity_calibration.mode_applied, "pool_minmax")

    def test_homogeneous_tiny_span_pool_uses_fixed_fallback(self) -> None:
        """Three near-identical E5 events (all cosines >= 0.91) must NOT stretch 0.91 down to 0.0."""
        e1 = _make_scored_candidate("E1", "AI 1", score_total=0.85, event_vec=_vec_with_cos_to_e0(1.0, 1))
        e2 = _make_scored_candidate("E2", "AI 2", score_total=0.84, event_vec=_vec_with_cos_to_e0(0.96, 1))
        e3 = _make_scored_candidate("E3", "AI 3", score_total=0.83, event_vec=_vec_with_cos_to_e0(0.97, 2))

        cfg = MMRConfig(cosine_rescale_mode="pool_minmax", min_pool_cosine_span=0.05)
        _, sem_sim, _, _, cal = build_similarity_matrices([e1, e2, e3], cfg)
        self.assertEqual(cal.mode_applied, "fixed_fallback")
        # All pairwise semantic similarities remain high (> 0.90) instead of collapsing to 0.0
        self.assertGreater(sem_sim[0, 1], 0.90)
        self.assertGreater(sem_sim[0, 2], 0.90)
        self.assertGreater(sem_sim[1, 2], 0.90)

    def test_lambda_one_and_disabled_mode_preserve_pure_p6_order(self) -> None:
        cands = [
            _make_scored_candidate(f"E{i}", f"Event {i}", score_total=0.90 - i * 0.05, event_vec=[1.0, 0.0, 0.0, 0.0])
            for i in range(5)
        ]
        res_lam1 = select_top_n_mmr(cands, top_n=4, lambda_param=1.0, enable_exploration_slot=False)
        self.assertEqual([x.scored.scored_event.event_id for x in res_lam1.selected], ["E0", "E1", "E2", "E3"])

        selector = MMRSelector()
        res_off = selector.select(cands, overrides={"enabled": False, "top_n": 4})
        self.assertEqual([x.scored.scored_event.event_id for x in res_off.selected], ["E0", "E1", "E2", "E3"])
        self.assertEqual(res_off.exploration_count, 0)
        self.assertIsNone(res_off.selected[1].trace.mmr_score)

    def test_mandatory_pinning_and_exploration_slot(self) -> None:
        """Mandatory 6.4 is pinned at #1; final slot picks unseen-group Exploration candidate."""
        shcd = _make_scored_candidate(
            "demo_mandatory_6_4", "Tuần SHCD", score_total=1.45,
            rubric_item_id="6.4", parent_group_id="P", gain=0.0, penalty=1.0, is_mandatory=True,
            event_vec=_vec_with_cos_to_e0(0.5, 5),
        )
        e1 = _make_scored_candidate(
            "E1", "Hội thảo AI", score_total=0.85, rubric_item_id="1.9", parent_group_id="C1",
            gain=0.60, fit=0.90, event_vec=_vec_with_cos_to_e0(1.0, 1),
        )
        e2 = _make_scored_candidate(
            "E2", "Workshop AI", score_total=0.82, rubric_item_id="1.9", parent_group_id="C1",
            gain=0.55, fit=0.85, event_vec=_vec_with_cos_to_e0(0.95, 1),
        )
        e3 = _make_scored_candidate(
            "E3", "Hội thao SV", score_total=0.75, rubric_item_id="3.5", parent_group_id="C3",
            gain=0.50, fit=0.80, event_vec=_vec_with_cos_to_e0(0.10, 2),
        )
        # Unseen group C4 with strong Gain & Fit -> ideal exploration slot!
        e4 = _make_scored_candidate(
            "E4", "Hiến máu cộng đồng", score_total=0.58, rubric_item_id="4.3", parent_group_id="C4",
            gain=0.80, fit=0.95, event_vec=_vec_with_cos_to_e0(0.05, 3),
        )

        res = select_top_n_mmr(
            [shcd, e1, e2, e3, e4],
            top_n=4,
            lambda_param=0.70,
            alpha_semantic=0.70,
            cosine_rescale_mode="none",
            enable_exploration_slot=True,
            pin_mandatory=True,
        )

        self.assertEqual(len(res.selected), 4)
        self.assertEqual(res.mandatory_count, 1)
        self.assertEqual(res.relevance_count, 2)
        self.assertEqual(res.exploration_count, 1)

        # #1 is mandatory SHCD
        self.assertEqual(res.selected[0].scored.scored_event.event_id, "demo_mandatory_6_4")
        self.assertEqual(res.selected[0].trace.slot_type, "mandatory")

        # #2 is top regular event E1
        self.assertEqual(res.selected[1].scored.scored_event.event_id, "E1")
        self.assertEqual(res.selected[1].trace.slot_type, "relevance")

        # #3 is diverse MMR pick E3 (C3)
        self.assertEqual(res.selected[2].scored.scored_event.event_id, "E3")
        self.assertEqual(res.selected[2].trace.slot_type, "relevance")

        # #4 is Exploration slot picking unseen group C4 (E4)
        self.assertEqual(res.selected[3].scored.scored_event.event_id, "E4")
        self.assertEqual(res.selected[3].trace.slot_type, "exploration")
        self.assertTrue(res.selected[3].trace.is_exploration_slot)
        self.assertTrue(res.selected[3].trace.exploration_is_unseen_group)
        self.assertEqual(res.exploration_event_id, "E4")
        self.assertIn("SEL_EXPLORATION", res.selected[3].trace.selection_reason)

    def test_empty_and_missing_vectors_graceful_fallback(self) -> None:
        # Empty input
        empty_res = select_top_n_mmr([], top_n=5)
        self.assertEqual(empty_res.selected, [])
        self.assertEqual(empty_res.metrics.mmr_ild, 0.0)
        self.assertEqual(empty_res.similarity_calibration.mode_applied, "empty")

        # Missing vectors -> falls back cleanly to rubric similarity
        a1 = _make_scored_candidate("A1", "C1 item 1", score_total=0.85, rubric_item_id="1.9", parent_group_id="C1", event_vec=None)
        a2 = _make_scored_candidate("A2", "C1 item 2", score_total=0.84, rubric_item_id="1.9", parent_group_id="C1", event_vec=None)
        b1 = _make_scored_candidate("B1", "C3 item 1", score_total=0.75, rubric_item_id="3.5", parent_group_id="C3", event_vec=None)

        self.assertEqual(compute_rubric_similarity(a1, a2), 1.0)
        self.assertEqual(compute_rubric_similarity(a1, b1), 0.0)

        res = select_top_n_mmr([a1, a2, b1], top_n=2, lambda_param=0.70, enable_exploration_slot=False)
        self.assertEqual([x.scored.scored_event.event_id for x in res.selected], ["A1", "B1"])
        self.assertTrue(math.isfinite(res.metrics.mmr_ild))
        self.assertAlmostEqual(res.metrics.mmr_ild, 1.0, places=5)

    def test_compute_ild_helper(self) -> None:
        sim = np.asarray([[1.0, 0.2, 0.8], [0.2, 1.0, 0.5], [0.8, 0.5, 1.0]], dtype=np.float64)
        # Pairwise distances: (1-0.2)=0.8, (1-0.8)=0.2, (1-0.5)=0.5 -> mean = 0.5
        self.assertAlmostEqual(compute_intra_list_diversity([0, 1, 2], sim), 0.5, places=6)
        self.assertEqual(compute_intra_list_diversity([0], sim), 0.0)


class TestMMRRealDatasetPipelineIntegration(unittest.TestCase):
    """End-to-end integration of P7 MMR on the real 108-event fixture pool (39 PREDICTED)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.candidates = load_recommendation_candidate_pool()
        cls.sample_embeddings = np.load(DEFAULT_SAMPLE_EMB_PATH)
        cls.embedder = _DeterministicStubEmbedder()
        cls.credit_provider = DemoCreditProvider()

    def test_pipeline_emits_mmr_selection_for_all_demo_personas(self) -> None:
        for pid in DEMO_PERSONAS:
            res = run_recommendation_demo(
                {"persona_id": pid, "mmr": {"top_n": 5, "lambda_param": 0.70}},
                candidates=self.candidates,
                sample_embeddings=self.sample_embeddings,
                embedder=self.embedder,
                credit_provider=self.credit_provider,
            )
            self.assertIn("mmr_selection", res)
            self.assertIn("selected_events", res)
            mmr = res["mmr_selection"]
            sel = res["selected_events"]
            ranked = res["ranked_events"]

            expected_len = min(5, len(ranked))
            self.assertEqual(len(sel), expected_len)
            self.assertEqual(res["pipeline_summary"]["selected_by_p7_mmr"], expected_len)
            self.assertEqual(mmr["selected_count"], expected_len)

            # Unique event IDs, valid traces, strictly sequential MMR ranks 1..N
            seen_ids = set()
            for expected_rank, ev in enumerate(sel, start=1):
                self.assertEqual(ev["rank"], expected_rank)
                self.assertNotIn(ev["event_id"], seen_ids)
                seen_ids.add(ev["event_id"])
                trace = ev["selection_trace"]
                self.assertIn(trace["slot_type"], {"mandatory", "relevance", "exploration"})
                self.assertTrue(len(trace["selection_reason"]) > 0)

            # Mandatory persona pins 6.4 at #1 in both P6 and P7 MMR
            if pid == "mandatory_shcd_test":
                self.assertEqual(sel[0]["event_id"], "demo_mandatory_6_4")
                self.assertEqual(sel[0]["selection_trace"]["slot_type"], "mandatory")

            # C1-capped persona never emits C1 in P7 MMR either
            if pid == "c1_capped_need_points":
                for ev in sel:
                    self.assertNotEqual(ev["parent_group_id"], "C1")

    def test_lambda_sweep_improves_diversity_on_real_pool(self) -> None:
        """Compare lambda=1.0 (no diversity) vs lambda=0.55 on balanced persona at 2026-04-01 (36 PREDICTED)."""
        base_payload = {
            "persona_id": "balanced_cs1",
            "reference_now": "2026-04-01T08:00:00+07:00",
        }
        res_pure = run_recommendation_demo(
            {**base_payload, "mmr": {"top_n": 5, "lambda_param": 1.0, "enable_exploration_slot": False}},
            candidates=self.candidates,
            sample_embeddings=self.sample_embeddings,
            embedder=self.embedder,
            credit_provider=self.credit_provider,
        )
        res_div = run_recommendation_demo(
            {**base_payload, "mmr": {"top_n": 5, "lambda_param": 0.55, "enable_exploration_slot": True}},
            candidates=self.candidates,
            sample_embeddings=self.sample_embeddings,
            embedder=self.embedder,
            credit_provider=self.credit_provider,
        )
        ild_pure = res_pure["mmr_selection"]["metrics"]["mmr_ild"]
        ild_div = res_div["mmr_selection"]["metrics"]["mmr_ild"]
        self.assertGreaterEqual(ild_div, ild_pure)
        self.assertEqual(res_div["mmr_selection"]["exploration_count"], 1)


if __name__ == "__main__":
    unittest.main()
