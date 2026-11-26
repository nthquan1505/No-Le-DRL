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


class TestRubricPrototypeV2Consistency(unittest.TestCase):
    """Technical validation of Schema v2.0 rubric prototypes & cross-artifact sync (Section 11)."""

    def test_cross_artifact_jsonl_parquet_npy_sync_and_v2_semantics(self) -> None:
        cls_dir = PROJECT_ROOT / "data" / "processed" / "cls_index"
        jsonl_path = cls_dir / "rubric_prototypes.jsonl"
        parquet_path = cls_dir / "rubric_metadata.parquet"
        npy_path = cls_dir / "rubric_embeddings.npy"
        audit_md_path = PROJECT_ROOT / "outputs" / "rubric_prototype_audit.md"

        for fp in (jsonl_path, parquet_path, npy_path, audit_md_path):
            self.assertTrue(fp.exists(), f"Missing deliverable: {fp}")

        jsonl_rows = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        df = pd.read_parquet(parquet_path)
        vecs = np.load(npy_path, mmap_mode="r")

        # 1. Exact 1:1 count, dtype, finite, L2-normalized, and row ordering
        self.assertEqual(len(jsonl_rows), 176)
        self.assertEqual(len(df), 176)
        self.assertEqual(tuple(vecs.shape), (176, 384))
        self.assertEqual(vecs.dtype, np.float32)
        self.assertTrue(np.isfinite(vecs).all())
        norms = np.linalg.norm(np.asarray(vecs, dtype=np.float64), axis=1)
        self.assertLess(float(np.max(np.abs(norms - 1.0))), 1e-5)

        self.assertEqual(df["embedding_row"].tolist(), list(range(176)))
        self.assertEqual(df["prototype_id"].tolist(), [r["prototype_id"] for r in jsonl_rows])
        self.assertEqual(df["embedding_text"].tolist(), [r["embedding_text"] for r in jsonl_rows])

        # 2. Single passage prefix & zero truncated group prototypes
        for r in jsonl_rows:
            txt = r["embedding_text"]
            self.assertTrue(txt.startswith("passage: "))
            self.assertFalse(txt.startswith("passage: passage:"))
            self.assertEqual(r["schema_version"], "2.0")

        stage1 = [r for r in jsonl_rows if r["stage"] == "group"]
        self.assertEqual([r["group_id"] for r in stage1], ["C1", "C2", "C3", "C4"])
        for r in stage1:
            txt = r["embedding_text"]
            self.assertIn("Phạm vi:", txt)
            self.assertIn("Các loại hoạt động tiêu biểu:", txt)
            self.assertEqual(txt.count("("), txt.count(")"))
            self.assertNotIn("tiểu l;", txt)
            self.assertNotIn("đơn vị tron;", txt)

        # 3. Stage 2 isolation: 20 eligible C1-C4 criteria only
        stage2 = [r for r in jsonl_rows if r["stage"] == "criterion"]
        self.assertEqual(len(stage2), 161)
        self.assertEqual(len({r["criterion_ref"] for r in stage2}), 20)
        for r in stage2:
            self.assertIn(r["group_id"], {"C1", "C2", "C3", "C4"})
            self.assertTrue(r["event_matchable"])
            self.assertIn(r["eligibility_class"], {"positive_candidate", "conditional_candidate"})
            self.assertIsInstance(r.get("boundary_metadata"), dict)

        # 4. Reminder (6.4), Special (NONE), and 0 NEEDS_REVIEW prototypes
        reminders = [r for r in jsonl_rows if r["stage"] == "reminder"]
        self.assertEqual(len(reminders), 5)
        self.assertTrue(all(r["criterion_ref"] == "6.4" and r["group_id"] == "P" for r in reminders))
        self.assertTrue(all(r["eligibility_class"] == "reminder_only" for r in reminders))

        specials = [r for r in jsonl_rows if r["stage"] == "special"]
        self.assertEqual(len(specials), 6)
        self.assertTrue(all(r["criterion_ref"] == "NONE" and r["group_id"] is None for r in specials))
        self.assertTrue(all(r["prototype_type"] == "draft_negative_reference" for r in specials))
        self.assertTrue(all(r["eligibility_class"] == "negative_reference_only" for r in specials))
        self.assertTrue(all(r.get("can_decide_none") is False for r in specials))

        self.assertEqual([r for r in jsonl_rows if r.get("criterion_ref") == "NEEDS_REVIEW"], [])

        # 5. Criterion 3.3 mobilization context & required_signals
        va_33 = [r for r in stage2 if r["criterion_ref"] == "3.3" and r["prototype_type"] == "verified_anchor"]
        self.assertEqual(len(va_33), 3)
        for r in va_33:
            self.assertIn("huy động", r["embedding_text"].lower())
            self.assertTrue(bool(r.get("anchor_text")))
            self.assertIn("explicit_student_mobilization_signal", r.get("required_signals") or [])

        # 6. Criterion 3.8 normalized verified vs draft with criterion_anchor_status=verified_partial
        p_38 = [r for r in stage2 if r["criterion_ref"] == "3.8"]
        va_38 = [r for r in p_38 if r["prototype_type"] == "verified_anchor"]
        da_38 = [r for r in p_38 if r["prototype_type"] == "draft_anchor"]
        self.assertEqual(len(va_38), 2)
        self.assertEqual(len(da_38), 2)
        self.assertTrue(all(r["anchor_status"] == "verified_from_school_list" for r in va_38))
        self.assertTrue(all(r["anchor_status"] == "draft" for r in da_38))
        self.assertTrue(all(r["criterion_anchor_status"] == "verified_partial" for r in p_38))

        # 7. Retriever isolation: Stage 1 & Stage 2 never expose 6.4 or NONE
        assets = load_all_assets(CONFIG_PATH, project_root=PROJECT_ROOT)
        retriever = RubricRetriever(assets.rubric_branch, assets.rubric)
        self.assertEqual(len(retriever._stage1_df), 4)
        self.assertEqual(len(retriever._stage2_df), 161)
        self.assertNotIn("6.4", set(retriever._stage2_df["criterion_ref"].dropna().astype(str)))
        self.assertNotIn("NONE", set(retriever._stage2_df["criterion_ref"].dropna().astype(str)))


class TestUpdatedSampleArtifactsConsistency(unittest.TestCase):
    """Technical validation of regenerated Sample Branch artifacts from activities_sample.csv."""

    def test_sample_embeddings_metadata_alignment_and_dual_branch_compatibility(self) -> None:
        cls_dir = PROJECT_ROOT / "data" / "processed" / "cls_index"
        sample_npy_path = cls_dir / "sample_embeddings.npy"
        sample_pq_path = cls_dir / "sample_metadata.parquet"
        report_path = PROJECT_ROOT / "outputs" / "sample_embedding_generation_report.md"

        for fp in (sample_npy_path, sample_pq_path, report_path):
            self.assertTrue(fp.exists(), f"Missing sample deliverable: {fp}")

        vecs = np.load(sample_npy_path, mmap_mode="r")
        df = pd.read_parquet(sample_pq_path)

        # 1. Exact count, shape, dtype, finite, L2-normalized
        self.assertEqual(tuple(vecs.shape), (908, 384))
        self.assertEqual(vecs.dtype, np.float32)
        self.assertTrue(np.isfinite(vecs).all())
        norms = np.linalg.norm(np.asarray(vecs, dtype=np.float64), axis=1)
        self.assertLess(float(np.max(np.abs(norms - 1.0))), 1e-5)

        # 2. Metadata alignment, provenance, and schema completeness
        self.assertEqual(len(df), 908)
        self.assertEqual(df["embedding_row"].tolist(), list(range(908)))
        self.assertTrue(df["sample_id"].is_unique)
        src_rows = df["source_row_index"].tolist()
        self.assertTrue(all(a < b for a, b in zip(src_rows, src_rows[1:])))
        self.assertTrue((df["source_dataset"] == "data/raw/activities_sample.csv").all())
        self.assertTrue((df["embedding_model"] == "intfloat/multilingual-e5-small").all())
        self.assertTrue((df["embedding_dimension"] == 384).all())
        self.assertTrue((df["normalization"] == "l2").all())
        for col in ("event_description", "role", "organizer", "embedding_text", "original_label"):
            self.assertIn(col, df.columns)
        self.assertFalse(any("vector" in c.lower() for c in df.columns))

        # 3. Leakage-free passage prefix & zero label leakage
        self.assertTrue(df["embedding_text"].str.startswith("passage: ").all())
        self.assertFalse(df["embedding_text"].str.startswith("passage: passage:").any())
        self.assertTrue((df["embedding_text"] == ("passage: " + df["event_title"])).all())
        self.assertTrue((df["original_label"] == df["criterion_ref"]).all())
        self.assertEqual(sorted(df["group_id"].unique().tolist()), ["C1", "C2", "C3", "C4"])
        self.assertNotIn("6.4", set(df["criterion_ref"].astype(str)))

        # 4. End-to-end dual-branch compatibility (loaders -> validators -> retrievers -> 2-stage kNN)
        assets = load_all_assets(CONFIG_PATH, project_root=PROJECT_ROOT)
        asset_rep = validate_loaded_assets(assets)
        self.assertTrue(asset_rep.all_passed, asset_rep.failures)

        sample_ret = SampleRetriever(assets.sample_branch)
        rubric_ret = RubricRetriever(assets.rubric_branch, assets.rubric)
        self.assertEqual(sample_ret.eligible_count, 908)
        self.assertEqual(sample_ret.available_groups, ["C1", "C2", "C3", "C4"])

        registry = build_default_registry()
        alg = registry.create(
            assets.config["algorithm"]["name"],
            sample_retriever=sample_ret,
            rubric_retriever=rubric_ret,
            rubric_data=assets.rubric,
            params=dict(assets.config["algorithm"]["params"]),
        )
        handler = ConfidenceAndDecisionHandler(
            rubric_retriever=rubric_ret,
            decision_config=assets.config["decision"],
        )

        # Classify real cached shared query vectors without mutating any disk files
        q_vecs = np.load(PROJECT_ROOT / assets.config["paths"]["query_embeddings"], mmap_mode="r")
        for idx in (0, len(q_vecs) // 2, len(q_vecs) - 1):
            pred = alg.classify(q_vecs[idx], event_context=assets.events[idx])
            self.assertEqual(len(pred.selected_groups), 2)
            self.assertEqual(len(pred.stage1_fused_scores), 4)
            self.assertGreater(len(pred.sample_neighbors), 0)
            self.assertTrue(all(nb["group_id"] in pred.selected_groups for nb in pred.sample_neighbors))
            dec = handler.decide(prediction=pred, event=assets.events[idx])
            self.assertIn(dec["status"], {"PREDICTED", "NEEDS_REVIEW", "NO_MATCH"})


if __name__ == "__main__":
    unittest.main()
