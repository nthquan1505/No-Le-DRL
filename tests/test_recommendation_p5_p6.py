"""Unit & integration tests for Demo M4, Demo M6, P4 Demo Profile, P5 Hard Filter, and P6 Scoring."""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone
from typing import List, Sequence

import numpy as np

from src.embedder import BaseEmbedder
from src.recommendation import (
    DEMO_PERSONAS,
    DemoCreditProvider,
    DemoScheduleProvider,
    DemoUserProfile,
    HardConstraintFilter,
    MultiCriteriaScorer,
    RecommendationCandidateInput,
    WeeklyBusySlot,
    compute_interest_calibration_range,
    create_mandatory_shcd_demo_event,
    load_recommendation_candidate_pool,
    resolve_effective_deadline,
    run_recommendation_demo,
    scale_interest_clipped_minmax,
)
from src.recommendation.pipeline import DEFAULT_SAMPLE_EMB_PATH
from src.schemas import BusyInterval, EventNormalized, ScheduleContext, ScoringWeights, TargetAudienceStructured
from src.vector_ops import l2_normalize

TZ = timezone(timedelta(hours=7))


class _DeterministicStubEmbedder(BaseEmbedder):
    """Fast deterministic 384d embedder for unit tests (no model download/loading)."""

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


class TestDemoModule4ComputeFitSpec(unittest.TestCase):
    """Verify all 8 acceptance test cases (TC1-TC8) from `docs/module4_fit_handoff_spec.md`."""

    def _make_provider(
        self,
        *,
        weekly_slots: Sequence[WeeklyBusySlot] = (),
        busy_intervals: Sequence[BusyInterval] = (),
        exam_weeks: Sequence[tuple[str, str]] = (),
        travel_overrides: dict | None = None,
    ) -> DemoScheduleProvider:
        ctx = ScheduleContext(
            busy_intervals=list(busy_intervals),
            exam_weeks=list(exam_weeks),
            default_travel_buffer_min=30,
        )
        return DemoScheduleProvider(
            context=ctx,
            weekly_slots=list(weekly_slots),
            travel_overrides=travel_overrides,
        )

    def test_tc1_hard_conflict_returns_zero(self) -> None:
        # Saturday 14/11/2026 13:00-15:00 class conflicts with 14:00-16:00 event
        provider = self._make_provider(
            weekly_slots=[WeeklyBusySlot(day_of_week=5, start_hour=13.0, end_hour=15.0, label="Class")]
        )
        ev = EventNormalized(
            event_id="tc1",
            title="Offline event",
            start_datetime="2026-11-14T14:00:00+07:00",
            end_datetime="2026-11-14T16:00:00+07:00",
            online_mode="offline",
            location="227 Nguyễn Văn Cừ",
        )
        res = provider.compute_fit(ev, home_campus="cs1_nvc", max_travel_min=45, pref_hours=(8.0, 21.0))
        self.assertTrue(res.has_conflict)
        self.assertEqual(res.fit_score, 0.0)

    def test_tc2_online_nice_hours_non_exam(self) -> None:
        provider = self._make_provider()
        ev = EventNormalized(
            event_id="tc2",
            title="Online webinar",
            start_datetime="2026-11-14T19:00:00+07:00",
            end_datetime="2026-11-14T20:30:00+07:00",
            online_mode="online",
        )
        res = provider.compute_fit(ev, home_campus="cs1_nvc", max_travel_min=45, pref_hours=(8.0, 21.0))
        self.assertFalse(res.has_conflict)
        self.assertAlmostEqual(res.fit_score, 1.0, places=6)

    def test_tc3_offline_same_campus_comfortable_slack(self) -> None:
        # Class ends at 13:00, event starts at 14:00 -> slack=60p, buffer=30p, excess=30p -> f_slack=1.0
        provider = self._make_provider(
            weekly_slots=[WeeklyBusySlot(day_of_week=5, start_hour=9.0, end_hour=13.0, label="Morning")]
        )
        ev = EventNormalized(
            event_id="tc3",
            title="CS1 seminar",
            start_datetime="2026-11-14T14:00:00+07:00",
            end_datetime="2026-11-14T16:00:00+07:00",
            online_mode="offline",
            location="227 Nguyễn Văn Cừ, Cơ sở 1",
        )
        res = provider.compute_fit(ev, home_campus="cs1_nvc", max_travel_min=45, pref_hours=(8.0, 21.0))
        self.assertEqual(res.estimated_travel_min, 10)
        self.assertEqual(res.nearest_slack_min, 60)
        self.assertAlmostEqual(res.fit_score, 0.92, places=6)

    def test_tc4_offline_cross_campus_tight_slack(self) -> None:
        # Class ends at 13:30, event starts at 14:00 -> slack=30p, buffer=30p, excess=0p -> f_slack=0.4
        provider = self._make_provider(
            weekly_slots=[WeeklyBusySlot(day_of_week=5, start_hour=10.0, end_hour=13.5, label="Class")]
        )
        ev = EventNormalized(
            event_id="tc4",
            title="CS2 activity",
            start_datetime="2026-11-14T14:00:00+07:00",
            end_datetime="2026-11-14T16:00:00+07:00",
            online_mode="offline",
            location="Cơ sở 2 Linh Trung Thủ Đức",
        )
        res = provider.compute_fit(ev, home_campus="cs1_nvc", max_travel_min=45, pref_hours=(8.0, 21.0))
        self.assertFalse(res.has_conflict)
        self.assertEqual(res.estimated_travel_min, 45)
        self.assertEqual(res.nearest_slack_min, 30)
        self.assertAlmostEqual(res.fit_score, 0.49, places=6)

    def test_tc5_offline_cross_campus_exam_week(self) -> None:
        provider = self._make_provider(exam_weeks=[("2026-11-09", "2026-11-22")])
        ev = EventNormalized(
            event_id="tc5",
            title="CS2 exam week",
            start_datetime="2026-11-14T14:00:00+07:00",
            end_datetime="2026-11-14T16:00:00+07:00",
            online_mode="offline",
            location="Linh Trung, Thủ Đức",
        )
        res = provider.compute_fit(ev, home_campus="cs1_nvc", max_travel_min=45, pref_hours=(8.0, 21.0))
        self.assertTrue(res.in_exam_week)
        self.assertAlmostEqual(res.fit_score, 0.288, places=6)

    def test_tc6_online_exam_week(self) -> None:
        provider = self._make_provider(exam_weeks=[("2026-11-09", "2026-11-22")])
        ev = EventNormalized(
            event_id="tc6",
            title="Online exam week",
            start_datetime="2026-11-14T19:00:00+07:00",
            end_datetime="2026-11-14T20:30:00+07:00",
            online_mode="online",
        )
        res = provider.compute_fit(ev, home_campus="cs1_nvc", max_travel_min=45, pref_hours=(8.0, 21.0))
        self.assertTrue(res.in_exam_week)
        self.assertAlmostEqual(res.fit_score, 0.90, places=6)

    def test_tc7_offline_over_max_travel_and_near_pref_hours(self) -> None:
        provider = self._make_provider(travel_overrides={"cross_campus": 60})
        ev = EventNormalized(
            event_id="tc7",
            title="Far early event",
            start_datetime="2026-11-14T07:30:00+07:00",
            end_datetime="2026-11-14T10:00:00+07:00",
            online_mode="offline",
            location="Linh Trung",
        )
        res = provider.compute_fit(ev, home_campus="cs1_nvc", max_travel_min=45, pref_hours=(8.0, 21.0))
        self.assertEqual(res.estimated_travel_min, 60)
        self.assertAlmostEqual(res.fit_score, 0.52, places=6)

    def test_tc8_missing_start_datetime_unknown_location(self) -> None:
        provider = self._make_provider()
        ev = EventNormalized(
            event_id="tc8",
            title="Missing time",
            start_datetime=None,
            online_mode="offline",
            location="Địa điểm chưa xác định",
        )
        res = provider.compute_fit(ev, home_campus="cs1_nvc", max_travel_min=45, pref_hours=(8.0, 21.0))
        self.assertFalse(res.has_conflict)
        self.assertAlmostEqual(res.fit_score, 0.61, places=6)

    def test_fit_has_no_fee_coupling(self) -> None:
        provider = self._make_provider()
        ev_free = EventNormalized(
            event_id="f0",
            title="Seminar",
            start_datetime="2026-11-14T14:00:00+07:00",
            end_datetime="2026-11-14T16:00:00+07:00",
            online_mode="offline",
            location="227 Nguyễn Văn Cừ",
            fee_vnd=0,
        )
        ev_paid = EventNormalized(
            event_id="f1",
            title="Seminar",
            start_datetime="2026-11-14T14:00:00+07:00",
            end_datetime="2026-11-14T16:00:00+07:00",
            online_mode="offline",
            location="227 Nguyễn Văn Cừ",
            fee_vnd=50000,
        )
        self.assertEqual(
            provider.compute_fit(ev_free, home_campus="cs1_nvc").fit_score,
            provider.compute_fit(ev_paid, home_campus="cs1_nvc").fit_score,
        )


class TestDemoModule6AndP5P6(unittest.TestCase):
    def setUp(self) -> None:
        self.credit_provider = DemoCreditProvider()
        self.hard_filter = HardConstraintFilter()
        self.scorer = MultiCriteriaScorer()
        self.ref_now = datetime(2026, 11, 1, 8, 0, tzinfo=TZ)

    def test_p5_strictly_accepts_predicted_only(self) -> None:
        profile = DemoUserProfile(reference_now=self.ref_now.isoformat())
        sched = DemoScheduleProvider.from_preset("light_schedule")
        ledger = self.credit_provider.build_ledger_from_group_totals(profile.current_by_group)

        ev = EventNormalized(
            event_id="e_nr",
            title="Uncertain event",
            start_datetime="2026-11-10T14:00:00+07:00",
            end_datetime="2026-11-10T16:00:00+07:00",
            online_mode="online",
        )
        report = self.hard_filter.filter_candidates(
            [
                RecommendationCandidateInput(
                    event=ev,
                    classification={"status": "NEEDS_REVIEW", "predicted_criterion_ref": None},
                ),
                RecommendationCandidateInput(
                    event=ev,
                    classification={
                        "status": "PREDICTED",
                        "predicted_group_id": "C1",
                        "predicted_criterion_ref": "1.9",
                    },
                ),
            ],
            profile=profile,
            schedule_provider=sched,
            credit_provider=self.credit_provider,
            ledger=ledger,
            reference_now=self.ref_now,
        )
        self.assertEqual(report.total_input, 2)
        self.assertEqual(report.predicted_input, 1)
        self.assertEqual(report.excluded_not_predicted, 1)
        self.assertEqual(report.excluded_status_counts["NEEDS_REVIEW"], 1)
        self.assertEqual(len(report.eligible), 1)

    def test_p5_records_all_violated_reasons_without_short_circuit(self) -> None:
        profile = DemoUserProfile(
            cohort="K24",
            home_campus="cs1_nvc",
            max_travel_min=20,
            strict_max_travel=True,
            max_budget_vnd=10000,
            participated_event_ids=["e_bad"],
            hidden_event_ids=["e_bad"],
            current_by_group={"C1": 30.0, "C2": 15.0, "C3": 10.0, "C4": 10.0, "C5": 0.0},
            reference_now=self.ref_now.isoformat(),
        )
        sched = DemoScheduleProvider.from_preset("busy_weekend")
        ledger = self.credit_provider.build_ledger_from_group_totals(profile.current_by_group)

        # Violates: F1 participated, F2 hidden, F3 expired, F4 cohort, F5 conflict (Sat 14:00),
        # F6 budget (50k > 10k), F7 too_far (45p > 20p), F8 C1 capped
        ev_bad = EventNormalized(
            event_id="e_bad",
            title="All violations",
            start_datetime="2026-10-31T14:00:00+07:00",  # Saturday before ref_now
            end_datetime="2026-10-31T16:00:00+07:00",
            reg_end_datetime="2026-10-29T23:59:00+07:00",
            online_mode="offline",
            location="Cơ sở 2 Linh Trung",
            fee_vnd=50000,
            target_audience=TargetAudienceStructured(cohorts=["K26"], all_students=False),
        )
        report = self.hard_filter.filter_candidates(
            [
                RecommendationCandidateInput(
                    event=ev_bad,
                    classification={
                        "status": "PREDICTED",
                        "predicted_group_id": "C1",
                        "predicted_criterion_ref": "1.9",
                    },
                )
            ],
            profile=profile,
            schedule_provider=sched,
            credit_provider=self.credit_provider,
            ledger=ledger,
            reference_now=self.ref_now,
        )
        self.assertEqual(len(report.eligible), 0)
        self.assertEqual(len(report.dropped), 1)
        codes = {r.code for r in report.dropped[0].reasons}
        self.assertEqual(
            codes,
            {
                "already_participated",
                "hidden_by_user",
                "expired",
                "audience_mismatch",
                "schedule_conflict",
                "over_budget",
                "too_far",
                "no_credit_gain",
            },
        )

    def test_p5_and_p6_mandatory_shcd_bypasses_zero_gain_and_ranks_first(self) -> None:
        profile = DemoUserProfile(
            user_vec=[1.0] + [0.0] * 383,
            hidden_event_ids=["demo_mandatory_6_4"],
            current_by_group={"C1": 22.0, "C2": 15.0, "C3": 10.0, "C4": 10.0, "C5": 0.0},
            target_total=85.0,
            target_by_group={"C1": 25.0, "C2": 18.0, "C3": 22.0, "C4": 20.0, "C5": 0.0},
            reference_now=self.ref_now.isoformat(),
        )
        sched = DemoScheduleProvider.from_preset("light_schedule")
        ledger = self.credit_provider.build_ledger_from_group_totals(
            profile.current_by_group, target_total=profile.target_total, target_by_group=profile.target_by_group
        )
        shortfall = self.credit_provider.compute_group_shortfall(ledger)

        shcd_ev, shcd_cls = create_mandatory_shcd_demo_event(self.ref_now)
        reg_ev = EventNormalized(
            event_id="reg_1",
            title="Hội thảo AI",
            organizer="Khoa CNTT",
            start_datetime="2026-11-05T14:00:00+07:00",
            end_datetime="2026-11-05T16:00:00+07:00",
            reg_end_datetime="2026-11-03T08:00:00+07:00",
            online_mode="online",
            fee_vnd=0,
            seats_left=1,  # Must NOT boost Urgency (scarcity omitted!)
            capacity=100,
        )
        reg_cls = {"status": "PREDICTED", "predicted_group_id": "C1", "predicted_criterion_ref": "1.6"}

        # Sample embeddings giving exact range [0.70, 0.90] for user_vec=[1, 0, ...]
        sample_emb = np.zeros((3, 384), dtype=np.float32)
        sample_emb[0, 0] = 0.70
        sample_emb[0, 1] = math.sqrt(1 - 0.70**2)
        sample_emb[1, 0] = 0.80
        sample_emb[1, 1] = math.sqrt(1 - 0.80**2)
        sample_emb[2, 0] = 0.90
        sample_emb[2, 1] = math.sqrt(1 - 0.90**2)

        # Give regular event high cosine (0.90 -> Interest=1.0) and SHCD low cosine (0.70 -> Interest=0.0)
        reg_vec = sample_emb[2].tolist()
        shcd_vec = sample_emb[0].tolist()

        report = self.hard_filter.filter_candidates(
            [
                RecommendationCandidateInput(event=reg_ev, classification=reg_cls, event_vec=reg_vec),
                RecommendationCandidateInput(event=shcd_ev, classification=shcd_cls, event_vec=shcd_vec),
            ],
            profile=profile,
            schedule_provider=sched,
            credit_provider=self.credit_provider,
            ledger=ledger,
            reference_now=self.ref_now,
        )
        # Both pass P5 (SHCD bypasses zero-gain & hidden_by_user!)
        self.assertEqual(len(report.eligible), 2)

        scored, cal = self.scorer.score_candidates(
            report.eligible,
            profile=profile,
            schedule_provider=sched,
            credit_provider=self.credit_provider,
            shortfall=shortfall,
            sample_embeddings=sample_emb,
            reference_now=self.ref_now,
        )
        self.assertAlmostEqual(cal.c_min, 0.70, places=5)
        self.assertAlmostEqual(cal.c_max, 0.90, places=5)

        # Mandatory 6.4 MUST be #1 with Gain=0.0 and Penalty=+1.0
        self.assertEqual(scored[0].scored_event.event_id, "demo_mandatory_6_4")
        self.assertEqual(scored[0].scored_event.gain, 0.0)
        self.assertAlmostEqual(scored[0].scored_event.interest, 0.0, places=5)
        self.assertAlmostEqual(scored[0].scored_event.penalty, 1.0, places=5)
        self.assertGreater(scored[0].scored_event.score_total, scored[1].scored_event.score_total)

        # Regular event has Penalty=0.0, Interest=1.0, Urgency = exp(-2/5) (NO seat scarcity bonus!)
        reg_sc = scored[1].scored_event
        self.assertEqual(reg_sc.event_id, "reg_1")
        self.assertEqual(reg_sc.penalty, 0.0)
        self.assertAlmostEqual(reg_sc.interest, 1.0, places=5)
        self.assertAlmostEqual(reg_sc.urgency, math.exp(-2.0 / 5.0), places=5)
        # Gain check: 1.6 Khoa = 5 pts, C1 gap = 3/30 = 0.1 -> m_gap=0.55, source=rule (0.9) -> 0.5*0.55*0.9 = 0.2475
        self.assertAlmostEqual(reg_sc.gain, 0.2475, places=5)

        # Grounded reasoning verification (strict math audit + no hallucination)
        for sc in scored:
            r = sc.to_dict()["reasoning"]
            self.assertIn("headline", r)
            self.assertIn("formula_check", r)
            self.assertEqual(len(r["criteria"]), 5)
            sum_weighted = sum(c["weighted_score"] for c in r["criteria"])
            self.assertAlmostEqual(sum_weighted, sc.scored_event.score_total, places=5)
            sum_share = sum(c["share_pct"] for c in r["criteria"])
            self.assertAlmostEqual(sum_share, 100.0, delta=0.5)
        self.assertIn("Bắt buộc", scored[0].to_dict()["reasoning"]["headline"])
        self.assertIn("không cộng điểm", scored[0].to_dict()["reasoning"]["criteria"][0]["reason"])

    def test_effective_deadline_handles_inverted_reg_end(self) -> None:
        ev = EventNormalized(
            event_id="inv",
            title="Inverted deadline",
            start_datetime="2026-11-10T08:00:00+07:00",
            end_datetime="2026-11-10T11:00:00+07:00",
            reg_end_datetime="2026-11-12T23:59:00+07:00",
        )
        dl = resolve_effective_deadline(ev)
        self.assertIsNotNone(dl)
        self.assertEqual(dl.day, 10)  # Falls back to start_datetime

    def test_interest_clipped_minmax_clamps_out_of_range(self) -> None:
        sample_emb = np.asarray([[0.75, math.sqrt(1 - 0.75**2)], [0.85, math.sqrt(1 - 0.85**2)]], dtype=np.float32)
        cal = compute_interest_calibration_range([1.0, 0.0], sample_emb, mode="per_user_samples_minmax")
        self.assertEqual(scale_interest_clipped_minmax(0.60, cal), 0.0)
        self.assertEqual(scale_interest_clipped_minmax(0.95, cal), 1.0)
        self.assertAlmostEqual(scale_interest_clipped_minmax(0.80, cal), 0.5, places=5)


class TestRealDatasetPipelineIntegration(unittest.TestCase):
    """Run real 108-event batch pool (39 PREDICTED + 908 sample embeddings) across all 4 demo personas."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.candidates = load_recommendation_candidate_pool()
        cls.sample_embeddings = np.load(DEFAULT_SAMPLE_EMB_PATH)
        cls.embedder = _DeterministicStubEmbedder()
        cls.credit_provider = DemoCreditProvider()

    def test_candidate_pool_has_108_total_and_39_predicted(self) -> None:
        self.assertEqual(len(self.candidates), 108)
        pred_count = sum(1 for c in self.candidates if c.classification.get("status") == "PREDICTED")
        self.assertEqual(pred_count, 39)
        self.assertEqual(self.sample_embeddings.shape, (908, 384))

    def test_all_four_demo_personas_execute_cleanly(self) -> None:
        for pid in DEMO_PERSONAS:
            res = run_recommendation_demo(
                {"persona_id": pid},
                candidates=self.candidates,
                sample_embeddings=self.sample_embeddings,
                embedder=self.embedder,
                credit_provider=self.credit_provider,
            )
            ps = res["pipeline_summary"]
            self.assertEqual(ps["excluded_not_predicted"], 69)
            expected_pred = 40 if pid == "mandatory_shcd_test" else 39
            self.assertEqual(ps["predicted_events_entered_p5"], expected_pred)
            self.assertEqual(ps["eligible_after_p5"] + ps["dropped_by_p5"], expected_pred)

            # Verify c1_capped drops all C1 events
            if pid == "c1_capped_need_points":
                for ev in res["ranked_events"]:
                    self.assertNotEqual(ev["parent_group_id"], "C1")

            # Verify mandatory persona ranks 6.4 at #1
            if pid == "mandatory_shcd_test":
                self.assertGreater(len(res["ranked_events"]), 0)
                top1 = res["ranked_events"][0]
                self.assertEqual(top1["event_id"], "demo_mandatory_6_4")
                self.assertEqual(top1["components"]["gain"], 0.0)
                self.assertAlmostEqual(top1["components"]["penalty"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
