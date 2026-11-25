"""Technical validation tests for the modular HCMUS DRL Event Classification System.

Validates architectural contracts, shared query embedding reuse, dual-branch isolation,
Stage 1 -> Stage 2 shared Top-2 coordination, algorithm registry extensibility,
rubric boundary guards, and output artifact integrity.
(Does NOT perform model accuracy evaluation or hyperparameter tuning.)
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.classification.algorithms.algorithm_registry import build_default_registry
from src.classification.algorithms.base_algorithm import (
    AlgorithmPrediction,
    BaseClassificationAlgorithm,
)
from src.classification.data.loaders import load_all_assets
from src.classification.data.validators import (
    validate_loaded_assets,
    validate_output_records,
    validate_query_embeddings,
)
from src.classification.decision.confidence_handler import ConfidenceAndDecisionHandler
from src.classification.embeddings.event_text_builder import (
    build_event_query_text,
    check_event_classifiable,
)
from src.classification.fusion.score_fusion import fuse_stage2_criteria
from src.classification.pipeline.classification_pipeline import ClassificationPipeline
from src.classification.retrieval.rubric_retriever import RubricRetriever
from src.classification.retrieval.sample_retriever import SampleRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "classification_config.yaml"


class _DummyRegisteredAlgorithm(BaseClassificationAlgorithm):
    """Minimal pluggable algorithm verifying AlgorithmRegistry extensibility."""

    name = "dummy_test_algorithm"

    def classify(
        self,
        query_vector: np.ndarray,
        event_context: Optional[Dict[str, Any]] = None,
    ) -> AlgorithmPrediction:
        return AlgorithmPrediction(
            algorithm_name=self.name,
            selected_groups=["C1", "C3"],
            stage1_sample_scores=[],
            stage1_rubric_scores=[],
            stage1_fused_scores=[
                {"group_id": "C1", "fused_score": 0.9, "rank": 1, "selected_for_stage2": True},
                {"group_id": "C3", "fused_score": 0.4, "rank": 2, "selected_for_stage2": True},
            ],
            stage2_sample_scores=[],
            stage2_rubric_scores=[],
            stage2_fused_scores=[],
            sample_neighbors=[],
            rubric_prototypes=[],
            group_margin=0.5,
            criterion_margin=0.0,
            top_raw_sample_similarity=None,
            top_raw_rubric_similarity=None,
            branch_top1_sample_criterion=None,
            branch_top1_rubric_criterion=None,
            branch_agreement=None,
            top_candidate=None,
        )


class TestQueryTextAndSkipping(unittest.TestCase):
    def test_build_event_query_text_uses_only_title_and_description(self) -> None:
        ev = {
            "is_event": True,
            "title": "  Cuộc thi Lập trình  ",
            "description": "Học thuật dành cho sinh viên HCMUS",
            "category": ["Academic"],
            "reason": "Some extraction reason",
            "predicted_criterion_ref": "1.6",
        }
        q = build_event_query_text(ev)
        self.assertEqual(q, "query: Cuộc thi Lập trình Học thuật dành cho sinh viên HCMUS")
        self.assertNotIn("Academic", q)
        self.assertNotIn("1.6", q)
        self.assertNotIn("extraction reason", q)

    def test_skip_non_event_and_empty_text(self) -> None:
        ok1, r1 = check_event_classifiable({"is_event": False, "title": "Not an event", "description": "desc"})
        self.assertFalse(ok1)
        self.assertIn("is_event=false", r1 or "")

        ok2, r2 = check_event_classifiable({"is_event": True, "title": "   ", "description": None})
        self.assertFalse(ok2)
        self.assertIn("missing_usable_text", r2 or "")

        ok3, r3 = check_event_classifiable({"is_event": True, "title": None, "description": "Valid description"})
        self.assertTrue(ok3)
        self.assertIsNone(r3)


class TestFusionAndRubricGuards(unittest.TestCase):
    def test_stage2_fusion_represents_missing_branch_without_zero_penalty(self) -> None:
        fused, s_norm, r_norm = fuse_stage2_criteria(
            candidate_refs=["4.2", "4.3"],
            sample_raw_scores={"4.3": 1.0},
            rubric_raw_scores={"4.2": 0.92, "4.3": 0.80},
            sample_available_refs={"4.3"},          # 4.2 has NO reference samples in events_full.csv
            rubric_available_refs={"4.2", "4.3"},
            ref_to_meta={
                "4.2": {"criterion_key": "C4.hien_mau", "group_id": "C4", "name": "Hiến máu"},
                "4.3": {"criterion_key": "C4.tinh_nguyen", "group_id": "C4", "name": "Tình nguyện"},
            },
            alpha=0.5,
            normalization_method="min_max",
            missing_branch_policy="reweight_available_branch",
        )
        by_ref = {e["criterion_ref"]: e for e in fused}
        self.assertEqual(by_ref["4.2"]["sample_evidence_status"], "no_reference_samples")
        self.assertEqual(by_ref["4.2"]["missing_branches"], ["sample"])
        self.assertIsNone(by_ref["4.2"]["sample_raw_score"])
        self.assertEqual(by_ref["4.2"]["fused_score"], 1.0)
        self.assertEqual(by_ref["4.2"]["effective_alpha"], 0.0)
        self.assertIsNone(s_norm["4.2"])
        self.assertEqual(r_norm["4.2"], 1.0)

    def test_algorithm_registry_extensibility(self) -> None:
        registry = build_default_registry()
        self.assertIn("hierarchical_two_stage_knn", registry.list_algorithms())
        registry.register("dummy_test_algorithm", _DummyRegisteredAlgorithm)
        self.assertIn("dummy_test_algorithm", registry.list_algorithms())

    def test_decision_handler_enforces_3_3_mobilization_guard(self) -> None:
        assets = load_all_assets(CONFIG_PATH, project_root=PROJECT_ROOT)
        rubric_ret = RubricRetriever(assets.rubric_branch, assets.rubric)
        handler = ConfidenceAndDecisionHandler(
            rubric_retriever=rubric_ret,
            decision_config=assets.config["decision"],
        )
        fake_pred = AlgorithmPrediction(
            algorithm_name="hierarchical_two_stage_knn",
            selected_groups=["C3", "C1"],
            stage1_sample_scores=[],
            stage1_rubric_scores=[],
            stage1_fused_scores=[
                {"group_id": "C3", "fused_score": 0.95, "rank": 1, "selected_for_stage2": True},
                {"group_id": "C1", "fused_score": 0.30, "rank": 2, "selected_for_stage2": True},
            ],
            stage2_sample_scores=[],
            stage2_rubric_scores=[],
            stage2_fused_scores=[
                {
                    "rank": 1,
                    "criterion_ref": "3.3",
                    "criterion_key": "C3.huy_dong_luc_luong",
                    "group_id": "C3",
                    "criterion_name": "Tham gia các đợt huy động lực lượng cấp Trường, cấp Khoa",
                    "fused_score": 0.95,
                    "sample_raw_score": 0.9,
                    "sample_evidence_status": "retrieved",
                    "rubric_raw_score": 0.88,
                    "rubric_evidence_status": "retrieved",
                    "evidence_verification_status": "verified_from_school_list",
                    "criterion_anchors_status": "verified_from_school_list",
                    "missing_branches": [],
                }
            ],
            sample_neighbors=[],
            rubric_prototypes=[],
            group_margin=0.65,
            criterion_margin=0.50,
            top_raw_sample_similarity=0.90,
            top_raw_rubric_similarity=0.88,
            branch_top1_sample_criterion="3.3",
            branch_top1_rubric_criterion="3.3",
            branch_agreement=True,
            top_candidate={
                "criterion_ref": "3.3",
                "criterion_key": "C3.huy_dong_luc_luong",
                "group_id": "C3",
                "fused_score": 0.95,
                "evidence_verification_status": "verified_from_school_list",
                "missing_branches": [],
            },
        )
        # Event WITHOUT explicit mobilization cues -> must route to NEEDS_REVIEW
        res_no_cue = handler.decide(
            prediction=fake_pred,
            event={"title": "Giải Bóng đá Futsal sinh viên", "description": "Mời các bạn đăng ký đội bóng."},
        )
        self.assertEqual(res_no_cue["status"], "NEEDS_REVIEW")
        self.assertIsNone(res_no_cue["predicted_criterion_ref"])
        self.assertTrue(any("3_3" in r for r in res_no_cue["review_reasons"]))

        # Event WITH explicit mobilization cue -> allowed to PREDICT
        res_with_cue = handler.decide(
            prediction=fake_pred,
            event={"title": "Thông báo huy động lực lượng cổ vũ Futsal", "description": "Huy động sinh viên tham dự."},
        )
        self.assertEqual(res_with_cue["status"], "PREDICTED")
        self.assertEqual(res_with_cue["predicted_criterion_ref"], "3.3")


class TestGeneratedArtifactsIntegrity(unittest.TestCase):
    """Technical validation of all generated artifacts on disk (Section 14)."""

    def test_generated_query_embeddings_and_classification_outputs(self) -> None:
        assets = load_all_assets(CONFIG_PATH, project_root=PROJECT_ROOT)
        asset_rep = validate_loaded_assets(assets)
        self.assertTrue(asset_rep.all_passed)

        q_emb_path = PROJECT_ROOT / assets.config["paths"]["query_embeddings"]
        q_meta_path = PROJECT_ROOT / assets.config["paths"]["query_metadata"]
        out_json_path = PROJECT_ROOT / assets.config["paths"]["output_classification_json"]
        out_report_path = PROJECT_ROOT / assets.config["paths"]["output_run_report_md"]
        future_work_path = PROJECT_ROOT / "FUTURE_WORK.md"

        for fp in (q_emb_path, q_meta_path, out_json_path, out_report_path, future_work_path):
            self.assertTrue(fp.exists(), f"Missing deliverable: {fp}")

        q_vecs = np.load(q_emb_path, mmap_mode="r")
        q_meta = json.loads(q_meta_path.read_text(encoding="utf-8"))
        q_rep = validate_query_embeddings(q_vecs, q_meta, assets.events)
        self.assertTrue(q_rep.all_passed, q_rep.failures)

        out_events = json.loads(out_json_path.read_text(encoding="utf-8"))
        out_rep = validate_output_records(
            input_events=assets.events,
            output_events=out_events,
            rubric=assets.rubric,
            expected_algorithm="hierarchical_two_stage_knn",
        )
        self.assertTrue(out_rep.all_passed, out_rep.failures)

    def test_pipeline_reuses_shared_query_embeddings_cleanly(self) -> None:
        pipeline = ClassificationPipeline(CONFIG_PATH, project_root=PROJECT_ROOT)
        summary = pipeline.run()
        self.assertTrue(summary.query_embeddings_reused)
        self.assertEqual(summary.total_input_events, 108)
        self.assertEqual(summary.output_events_count, 108)
        self.assertEqual(summary.sample_stage1_calls, summary.embedded_events_count)
        self.assertEqual(summary.rubric_stage1_calls, summary.embedded_events_count)
        self.assertEqual(summary.sample_stage2_calls, summary.embedded_events_count)
        self.assertEqual(summary.rubric_stage2_calls, summary.embedded_events_count)
        self.assertTrue(summary.asset_validation.all_passed)
        self.assertTrue(summary.query_validation.all_passed)
        self.assertTrue(summary.output_validation.all_passed)


if __name__ == "__main__":
    unittest.main()
