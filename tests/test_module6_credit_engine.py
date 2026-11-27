"""Unit tests for isolated replaceable Module 6 (RubricCreditEngine & AttributeExtractor)."""

from __future__ import annotations

import unittest

from module6 import (
    AttributeExtractor,
    CreditCalculatorProtocol,
    LedgerEntry,
    RubricCreditEngine,
    UserLedger,
)
from src.schemas import EventNormalized


class TestAttributeExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ext = AttributeExtractor()

    def test_duration_buckets_match_rubric_yaml_keys(self) -> None:
        self.assertEqual(self.ext.extract_duration_bucket("Tình nguyện trong 4 giờ")[0], "lt_24h")
        self.assertEqual(self.ext.extract_duration_bucket("Chiến dịch 1 ngày")[0], "lt_24h")
        self.assertEqual(
            self.ext.extract_duration_bucket("Tình nguyện 3 ngày 2 đêm")[0],
            "24h_to_lt_1w",
        )
        self.assertEqual(self.ext.extract_duration_bucket("Chiến dịch 2 tuần")[0], "ge_1w")
        self.assertEqual(self.ext.extract_duration_bucket("Mùa hè xanh 2026"), ("ge_1w", True))

    def test_level_role_and_explicit_points(self) -> None:
        lv, _ = self.ext.extract_level("Hội thảo học thuật cấp Trường", "Đoàn Trường")
        self.assertEqual(lv, "truong")

        roles, recruiting = self.ext.extract_role("Tuyển cộng tác viên ban tổ chức chương trình")
        self.assertIn("collaborator", roles)
        self.assertTrue(recruiting)

        ep_pos = self.ext.extract_explicit_points("Tham gia được cộng 5 điểm rèn luyện")
        self.assertEqual(ep_pos["points"], 5)
        self.assertFalse(ep_pos["negated"])

        ep_neg = self.ext.extract_explicit_points("Sự kiện này không cộng điểm rèn luyện")
        self.assertTrue(ep_neg["negated"])


class TestRubricCreditEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = AttributeExtractor()
        cls.engine = RubricCreditEngine(attribute_extractor=cls.extractor)

    def test_implements_protocol(self) -> None:
        self.assertIsInstance(self.engine, CreditCalculatorProtocol)

    def test_baseline_defaults_50_points_and_toggle(self) -> None:
        eval_default = self.engine.evaluate_ledger(UserLedger())
        self.assertEqual(eval_default.capped_by_group["C1"], 15.0)
        self.assertEqual(eval_default.capped_by_group["C2"], 15.0)
        self.assertEqual(eval_default.capped_by_group["C3"], 10.0)
        self.assertEqual(eval_default.capped_by_group["C4"], 10.0)
        self.assertEqual(eval_default.capped_by_group["C5"], 0.0)
        self.assertEqual(eval_default.total_score, 50.0)

        eval_disabled = self.engine.evaluate_ledger(
            UserLedger(use_default_compliance_baseline=False)
        )
        self.assertEqual(eval_disabled.total_score, 0.0)

    def test_group_caps_overflow_to_100_and_penalties_applied_after_caps(self) -> None:
        # Push every positive group above its cap (30 + 20 + 25 + 25 + 30 = 130 -> capped at 100)
        entries = [
            LedgerEntry("1.6", 40.0),  # C1: 15 baseline + 40 = 55 -> capped at 30
            LedgerEntry("2.3", 20.0),  # C2: 15 + 20 = 35 -> capped at 20
            LedgerEntry("3.5", 30.0),  # C3: 10 + 30 = 40 -> capped at 25
            LedgerEntry("4.3", 30.0),  # C4: 10 + 30 = 40 -> capped at 25
            LedgerEntry("5.4", 40.0),  # C5: 40 -> capped at 30
        ]
        eval_full = self.engine.evaluate_ledger(UserLedger(entries=entries))
        self.assertEqual(eval_full.capped_by_group, {"C1": 30, "C2": 20, "C3": 25, "C4": 25, "C5": 30})
        self.assertEqual(eval_full.total_before_clamp, 130.0)
        self.assertEqual(eval_full.total_score, 100.0)

        # Apply -20 penalty (6.4) AFTER group caps: 130 - 20 = 110 -> clamped to 100
        # Apply -50 total penalties: 130 - 50 = 80
        eval_penalized = self.engine.evaluate_ledger(
            UserLedger(entries=entries + [LedgerEntry("6.4", -20.0), LedgerEntry("6.6", -30.0)])
        )
        self.assertEqual(eval_penalized.total_before_clamp, 80.0)
        self.assertEqual(eval_penalized.total_score, 80.0)

        # Heavy penalties floor at 0 (Q2)
        eval_floor = self.engine.evaluate_ledger(
            UserLedger(
                use_default_compliance_baseline=False,
                entries=[LedgerEntry("6.4", -20.0), LedgerEntry("6.6", -20.0)],
            )
        )
        self.assertEqual(eval_floor.total_score, 0.0)

    def test_role_redirection_and_lop_collaborator_zero_points(self) -> None:
        ev = EventNormalized(event_id="ev1", title="Cuộc thi học thuật CTF 2026")
        est_part = self.engine.estimate_event_credit(ev, "1.6", role="participant")
        self.assertEqual(est_part.rubric_item_id, "1.6")
        self.assertEqual(est_part.expected_points, 5.0)

        est_org = self.engine.estimate_event_credit(ev, "1.6", role="organizer")
        self.assertEqual(est_org.rubric_item_id, "1.7")
        self.assertEqual(est_org.expected_points, 10.0)

        # Q6: C3 collaborator at 'lop' level receives 0 points
        ev_c3 = EventNormalized(event_id="ev2", title="Hội thao cấp lớp")
        est_collab_lop = self.engine.estimate_event_credit(
            ev_c3, "3.14", role="collaborator", inferred_level="lop"
        )
        self.assertEqual(est_collab_lop.expected_points, 0.0)
        self.assertFalse(est_collab_lop.is_credit_eligible)

    def test_level_buckets_and_c4_duration_buckets(self) -> None:
        ev_comp = EventNormalized(event_id="ev3", title="Hội diễn văn nghệ cấp Quốc gia")
        est_nat = self.engine.estimate_event_credit(
            ev_comp, "3.1", role="participant", inferred_level="quoc_gia"
        )
        self.assertEqual(est_nat.expected_points, 10.0)

        ev_vol = EventNormalized(event_id="ev4", title="Chiến dịch tình nguyện 3 ngày")
        est_mid = self.engine.estimate_event_credit(ev_vol, "4.3", role="participant")
        self.assertEqual(est_mid.expected_points, 10.0)  # 24h_to_lt_1w via fixed extractor

        ev_vol_long = EventNormalized(event_id="ev5", title="Mùa hè xanh 2026")
        est_long = self.engine.estimate_event_credit(ev_vol_long, "4.3", role="participant")
        self.assertEqual(est_long.expected_points, 15.0)  # ge_1w prior

    def test_group_cap_reached_zeroes_effective_points(self) -> None:
        # C3 baseline = 10; adding 15 reaches cap 25
        ledger = UserLedger(entries=[LedgerEntry("3.5", 15.0)])
        ev = EventNormalized(event_id="ev6", title="Hoạt động phong trào")
        est = self.engine.estimate_event_credit(ev, "3.5", ledger=ledger)
        self.assertEqual(est.expected_points, 5.0)
        self.assertEqual(est.effective_points, 0.0)
        self.assertEqual(est.capped_by, "group")
        self.assertFalse(est.is_credit_eligible)

    def test_mandatory_event_6_4_never_awards_positive_credit(self) -> None:
        ev = EventNormalized(event_id="ev7", title="Tuần sinh hoạt công dân đầu năm học")
        est = self.engine.estimate_event_credit(ev, "6.4")
        self.assertFalse(est.is_credit_eligible)
        self.assertEqual(est.expected_points, 0.0)
        self.assertIn("mandatory_event_avoid_penalty", est.warning_flags)

    def test_exclusive_with_prevents_double_counting(self) -> None:
        # 5.11 (C5.mien_giam_hoc_phi) is exclusive_with 5.8 (C5.mo_coi) — verified symmetrically
        ledger = UserLedger(entries=[LedgerEntry("5.8", 20.0)])
        ev = EventNormalized(event_id="ev8", title="Miễn giảm học phí")
        est = self.engine.estimate_event_credit(ev, "5.11", ledger=ledger)
        self.assertFalse(est.is_credit_eligible)
        self.assertEqual(est.capped_by, "exclusive_with")

        # Reverse direction: 3.13 (tự ứng tuyển) is exclusive_with 3.12 (được tuyển chọn)
        ledger_rev = UserLedger(entries=[LedgerEntry("3.13", 10.0)])
        ev_rev = EventNormalized(event_id="ev9", title="Đại biểu được tuyển chọn")
        est_rev = self.engine.estimate_event_credit(ev_rev, "3.12", ledger=ledger_rev)
        self.assertFalse(est_rev.is_credit_eligible)
        self.assertEqual(est_rev.capped_by, "exclusive_with")

    def test_group_shortfall_computation(self) -> None:
        ledger = UserLedger(
            entries=[LedgerEntry("1.6", 5.0)],
            target_total=85.0,
            target_by_group={"C1": 25.0, "C3": 20.0},
        )
        sf = self.engine.compute_group_shortfall(ledger)
        self.assertEqual(sf.current_by_group["C1"], 20.0)  # 15 baseline + 5
        self.assertEqual(sf.remaining_cap_by_group["C1"], 10.0)  # 30 - 20
        self.assertEqual(sf.target_gap_by_group["C1"], 5.0)  # 25 - 20
        self.assertEqual(sf.current_total, 55.0)


if __name__ == "__main__":
    unittest.main()
