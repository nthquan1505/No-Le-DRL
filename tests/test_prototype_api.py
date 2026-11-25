"""Tests for P2.1 e5-small embedder factory and interactive prototype backend (app.py)."""

from __future__ import annotations

import unittest

from app import PROTOTYPE_DIR, PrototypeService
from configs import MODEL_ID
from src.embedder import create_embedder, is_e5_runtime_available


class TestEmbedderFactory(unittest.TestCase):
    def test_e5_mode_reports_requested_model_and_falls_back_cleanly_if_uninstalled(self) -> None:
        emb, status = create_embedder(mode="e5", allow_fallback=True)
        self.assertEqual(status.requested_model, MODEL_ID)
        self.assertEqual(status.dimension, 384)
        self.assertEqual(emb.dimension, 384)
        if is_e5_runtime_available():
            self.assertTrue(status.is_e5_active)
            self.assertEqual(status.active_model, MODEL_ID)
        else:
            self.assertFalse(status.is_e5_active)
            self.assertIsNotNone(status.fallback_reason)
            with self.assertRaises(ImportError):
                create_embedder(mode="e5", allow_fallback=False)


class TestPrototypeService(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = PrototypeService(embedder_mode="e5", allow_fallback=True)

    def test_static_frontend_files_exist(self) -> None:
        self.assertTrue((PROTOTYPE_DIR / "index.html").is_file())
        self.assertTrue((PROTOTYPE_DIR / "styles.css").is_file())
        self.assertTrue((PROTOTYPE_DIR / "app.js").is_file())

    def test_bootstrap_and_recommendations_include_exploration_slot_and_drop_reasons(self) -> None:
        boot = self.service.bootstrap_payload()
        self.assertEqual(boot["embedder_status"]["requested_model"], MODEL_ID)
        self.assertEqual(len(boot["rubric_catalog"]), 51)

        rec = boot["recommendations_bundle"]
        self.assertEqual(rec["stats"]["total_cached_events"], 108)
        self.assertGreater(len(rec["recommendations"]), 0)
        self.assertGreater(len(rec["dropped_events"]), 0)
        self.assertTrue(any(c["is_exploration_slot"] for c in rec["recommendations"]))

    def test_live_p2_inspector_returns_distinct_stage1_stage2_and_module6(self) -> None:
        res = self.service.inspect_classify(
            {
                "title": "Cuộc thi An toàn thông tin HCMUS CTF 2026",
                "description": "Cuộc thi học thuật cấp Khoa",
                "organizer": "Khoa CNTT",
                "role": "organizer",
                "stage1_top_k": 2,
                "stage1_neighbors": 5,
                "stage2_neighbors": 3,
            }
        )
        self.assertEqual(res["vector_dimension"], 384)
        self.assertAlmostEqual(res["vector_l2_norm"], 1.0, places=5)
        self.assertEqual(len(res["classification"]["selected_candidate_groups"]), 2)
        self.assertFalse(res["classification"]["is_calibrated"])
        self.assertIsNone(res["classification"]["calibrated_confidence"])
        self.assertIn("module6_credit_estimate", res)

        # Verify real dual-branch payload is connected to /api/classify
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
        self.assertEqual(batch["summary"]["sample_reference_count"], 127)
        self.assertEqual(batch["summary"]["rubric_total_prototypes"], 170)
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
                "action": "label_correction",
                "event_id": eid,
                "corrected_item_id": "1.6",
            }
        )
        self.assertEqual(svc.cls_index.size, initial_cls_size + 1)
        self.assertEqual(corr["entry"]["corrected_group_id"], "C1")

        # Joined -> updates Module 6 ledger score
        score_before = svc.credit_engine.evaluate_ledger(svc.ledger).total_score
        svc.record_feedback({"action": "joined", "event_id": eid})
        score_after = svc.credit_engine.evaluate_ledger(svc.ledger).total_score
        self.assertGreater(score_after, score_before)


if __name__ == "__main__":
    unittest.main()
