"""Phase 1 unit tests for schemas, canonical IDs (T-4), and contract validators."""

from __future__ import annotations

import unittest

from src.contract_validator import (
    validate_classification_result,
    validate_cls_sample,
    validate_event_normalized,
    validate_feedback_event,
    validate_scoring_weights,
)
from src.schemas import (
    ClassificationResult,
    ClsSample,
    EventNormalized,
    FeedbackEvent,
    ScoringWeights,
    TargetAudienceStructured,
    is_valid_item_id,
    parent_group_from_item_id,
)


class TestCanonicalTaxonomyIDs(unittest.TestCase):
    def test_valid_item_ids_and_parent_mapping(self) -> None:
        cases = {
            "1.1": "C1",
            "1.6": "C1",
            "1.7": "C1",
            "2.3": "C2",
            "3.14": "C3",
            "4.8": "C4",
            "5.11": "C5",
            "6.4": "P",
        }
        for item_id, expected_group in cases.items():
            self.assertTrue(is_valid_item_id(item_id))
            self.assertEqual(parent_group_from_item_id(item_id), expected_group)

    def test_invalid_item_ids_rejected(self) -> None:
        for bad in ("", "C1.cuoc_thi", "0.1", "7.1", "1.0", "1", "1.6.1"):
            self.assertFalse(is_valid_item_id(bad), f"Expected '{bad}' to be invalid")
            with self.assertRaises(ValueError):
                parent_group_from_item_id(bad)


class TestEventNormalizedContract(unittest.TestCase):
    def test_valid_event_roundtrip_and_validation(self) -> None:
        ev = EventNormalized(
            event_id="evt_001",
            title="Hội thảo AI 2026",
            description="Chia sẻ nghiên cứu khoa học cấp Khoa",
            organizer="Khoa CNTT",
            start_datetime="2026-12-01T08:00:00+07:00",
            end_datetime="2026-12-01T11:30:00+07:00",
            reg_end_datetime="2026-11-30T23:59:00+07:00",
            online_mode="offline",
            is_online=False,
            fee_vnd=0,
            target_audience=TargetAudienceStructured(all_students=True),
            capacity=100,
            seats_left=40,
        )
        restored = EventNormalized.from_dict(ev.to_dict())
        self.assertEqual(restored, ev)
        report = validate_event_normalized(restored)
        self.assertTrue(report.is_valid)
        self.assertEqual(len(report.warnings), 0)
        self.assertIn("Vai trò: Ban tổ chức", ev.to_classification_text(role="Ban tổ chức"))

    def test_event_validation_catches_contract_violations(self) -> None:
        bad = EventNormalized(
            event_id="",
            title="",
            start_datetime="2026-12-02T10:00:00+07:00",
            end_datetime="2026-12-01T10:00:00+07:00",
            online_mode="online",
            is_online=False,
            fee_vnd=-5000,
            capacity=50,
            seats_left=60,
        )
        report = validate_event_normalized(bad)
        self.assertFalse(report.is_valid)
        codes = {issue.code for issue in report.errors}
        self.assertIn("missing_required", codes)
        self.assertIn("end_before_start", codes)
        self.assertIn("mode_mismatch", codes)
        self.assertIn("negative_value", codes)
        self.assertIn("exceeds_capacity", codes)


class TestClassificationResultInvariants(unittest.TestCase):
    def test_valid_uncalibrated_two_stage_result(self) -> None:
        res = ClassificationResult(
            event_id="evt_001",
            rubric_item_id="3.5",
            parent_group_id="C3",
            selected_candidate_groups=["C1", "C3"],
            stage1_scores={"C1": 0.62, "C3": 0.58},
            stage2_scores={"1.9": 0.61, "3.5": 0.74},
            calibrated_confidence=None,
            is_calibrated=False,
            status="uncalibrated",
        )
        restored = ClassificationResult.from_dict(res.to_dict())
        self.assertEqual(restored, res)
        report = validate_classification_result(restored)
        self.assertTrue(report.is_valid)

    def test_rejects_uncalibrated_probability_leak_and_parent_mismatch(self) -> None:
        bad = ClassificationResult(
            event_id="evt_002",
            rubric_item_id="1.6",
            parent_group_id="C3",  # wrong parent (should be C1)
            selected_candidate_groups=["C2", "C3"],  # C1 missing
            stage1_scores={"C2": 0.5, "C3": 0.6},
            stage2_scores={"1.6": 0.8},
            calibrated_confidence=0.8,  # forbidden while is_calibrated=False
            is_calibrated=False,
            status="uncalibrated",
        )
        report = validate_classification_result(bad)
        self.assertFalse(report.is_valid)
        codes = {i.code for i in report.errors}
        self.assertIn("parent_child_mismatch", codes)
        self.assertIn("parent_not_in_stage1_candidates", codes)
        self.assertIn("uncalibrated_probability_leak", codes)


class TestWeightsSampleAndFeedbackValidation(unittest.TestCase):
    def test_scoring_weights_d5(self) -> None:
        self.assertTrue(validate_scoring_weights(ScoringWeights()).is_valid)
        bad = ScoringWeights(w1_gain=0.5, w2_interest=0.5, w3_fit=0.5, w4_urgency=0.0)
        self.assertFalse(validate_scoring_weights(bad).is_valid)

    def test_cls_sample_validation(self) -> None:
        ok = ClsSample(
            sample_id="demo_0001",
            text="Cuộc thi Coding Challenge | Vai trò: Ban tổ chức",
            title="Cuộc thi Coding Challenge",
            rubric_item_id="1.7",
            parent_group_id="C1",
            role="Ban tổ chức",
        )
        self.assertTrue(validate_cls_sample(ok).is_valid)

        bad = ClsSample(
            sample_id="demo_0002",
            text="Cuộc thi Coding Challenge",
            title="Cuộc thi Coding Challenge",
            rubric_item_id="1.7",
            parent_group_id="C3",
        )
        self.assertFalse(validate_cls_sample(bad).is_valid)

    def test_feedback_label_correction_lineage(self) -> None:
        ok = FeedbackEvent(
            user_id="u1",
            event_id="evt_1",
            action="label_correction",
            timestamp="2026-11-24T00:00:00+07:00",
            corrected_item_id="1.7",
            corrected_group_id="C1",
        )
        self.assertTrue(validate_feedback_event(ok).is_valid)

        bad = FeedbackEvent(
            user_id="u1",
            event_id="evt_1",
            action="label_correction",
            timestamp="2026-11-24T00:00:00+07:00",
            corrected_item_id="1.7",
            corrected_group_id="C4",
        )
        self.assertFalse(validate_feedback_event(bad).is_valid)


if __name__ == "__main__":
    unittest.main()
