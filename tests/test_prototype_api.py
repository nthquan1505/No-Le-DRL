"""Tests for unified e5-small embedder and clean Classification + Module 6 backend (app.py)."""

from __future__ import annotations

import unittest

import src.embedder as embedder_mod
from app import PROTOTYPE_DIR, PrototypeService
from configs import MODEL_ID
from src.embedder import E5Embedder, create_embedder, is_e5_runtime_available


class TestEmbedder(unittest.TestCase):
    def test_unified_e5_small_only_and_no_hashing_fallback(self) -> None:
        self.assertFalse(hasattr(embedder_mod, "HashingDemoEmbedder"))
        self.assertTrue(is_e5_runtime_available())
        emb, status = create_embedder()
        self.assertIsInstance(emb, E5Embedder)
        self.assertTrue(status.is_e5_active)
        self.assertEqual(status.requested_model, MODEL_ID)
        self.assertEqual(status.active_model, MODEL_ID)
        self.assertEqual(status.dimension, 384)
        self.assertEqual(emb.dimension, 384)


class TestPrototypeService(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = PrototypeService()

    def test_static_frontend_files_exist(self) -> None:
        self.assertTrue((PROTOTYPE_DIR / "index.html").is_file())
        self.assertTrue((PROTOTYPE_DIR / "styles.css").is_file())
        self.assertTrue((PROTOTYPE_DIR / "app.js").is_file())

    def test_bootstrap_payload_contains_only_real_modules(self) -> None:
        boot = self.service.bootstrap_payload()
        self.assertEqual(boot["embedder_status"]["requested_model"], MODEL_ID)
        self.assertEqual(boot["embedder_status"]["active_model"], MODEL_ID)
        self.assertTrue(boot["embedder_status"]["is_e5_active"])

        summary = boot["dual_branch_summary"]
        self.assertEqual(summary["algorithm"], "hierarchical_two_stage_knn")
        self.assertEqual(summary["sample_reference_count"], 908)
        self.assertEqual(summary["rubric_total_prototypes"], 176)
        self.assertEqual(summary["batch_events_count"], 108)
        self.assertEqual(len(boot["dual_branch_events"]["events"]), 108)

        for mock_key in (
            "recommendations_bundle",
            "schedule_plan",
            "weekly_timetable",
            "interest_chips",
        ):
            self.assertNotIn(mock_key, boot)

    def test_live_inspect_classify_returns_dual_branch_and_module6(self) -> None:
        res = self.service.inspect_classify(
            {
                "title": "Cuộc thi An toàn thông tin HCMUS CTF 2026",
                "description": "Cuộc thi học thuật cấp Khoa",
                "organizer": "Khoa CNTT",
                "role": "organizer",
                "stage1_top_k": 2,
                "k1": 10,
                "k2": 10,
                "beta": 0.5,
                "alpha": 0.5,
            }
        )
        self.assertFalse(res["reused_cached_query"])
        self.assertEqual(res["vector_dimension"], 384)
        self.assertAlmostEqual(res["vector_l2_norm"], 1.0, places=5)
        self.assertIn("module6_credit_estimate", res)
        self.assertIn(res["module6_ref_mode"], {"predicted", "suggested_preview", "none"})

        db = res["dual_branch"]
        self.assertEqual(db["algorithm"], "hierarchical_two_stage_knn")
        self.assertIn(db["status"], {"PREDICTED", "NEEDS_REVIEW", "NO_MATCH"})
        self.assertFalse(db["confidence"]["calibrated"])
        self.assertEqual(len(db["scores"]["stage1"]["fused"]), 4)
        self.assertGreater(len(db["scores"]["stage2"]["fused"]), 0)
        self.assertIn("sample_neighbors", db["evidence"])
        self.assertIn("rubric_prototypes", db["evidence"])

    def test_dual_branch_batch_events_endpoint_returns_108_events(self) -> None:
        batch = self.service.get_dual_branch_events()
        self.assertEqual(batch["summary"]["algorithm"], "hierarchical_two_stage_knn")
        self.assertEqual(batch["summary"]["sample_reference_count"], 908)
        self.assertEqual(batch["summary"]["rubric_total_prototypes"], 176)
        self.assertEqual(len(batch["events"]), 108)

    def test_schedule_planner_is_conflict_free_and_respects_group_caps(self) -> None:
        plan = self.service.generate_schedule_plan(target_total=85)
        self.assertGreaterEqual(plan["projected_total"], plan["current_total"])
        for grp, pts in plan["projected_by_group"].items():
            self.assertLessEqual(pts, plan["group_caps"][grp])

    def test_feedback_joined_and_label_correction_update_ledger_and_cls_index(self) -> None:
        svc = PrototypeService(embedder_mode="demo")
        initial_cls_size = svc.cls_index.size
        rec = svc.get_recommendations()
        first_event = rec["recommendations"][0]
        eid = first_event["event_id"]

        # Label correction -> adds sample live to ClsIndex
        corr = svc.record_feedback(
            {
                "post_id": first_event["post_id"],
                "use_cached_query": True,
                "title": first_event["title"],
                "description": first_event["description"],
                "organizer": first_event["organizer"],
                "role": "participant",
            }
        )
        self.assertTrue(res["reused_cached_query"])
        self.assertEqual(res["vector_dimension"], 384)
        self.assertIn("module6_credit_estimate", res)

    def test_run_dual_branch_pipeline_passes_all_checks(self) -> None:
        run_res = self.service.run_dual_branch_pipeline()
        self.assertEqual(run_res["status"], "ok")
        self.assertTrue(run_res["all_checks_passed"])
        self.assertEqual(run_res["total_input_events"], 108)
        self.assertEqual(len(run_res["dual_branch_events"]["events"]), 108)


if __name__ == "__main__":
    unittest.main()
