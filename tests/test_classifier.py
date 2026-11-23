"""Phase 3 (Stage 2B) tests for P2 embedder, HierarchicalClsIndex, TwoStageHierarchicalKNNClassifier, and EventProfileCache."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Sequence

from configs import CLS_INDEX_DIR, EMBEDDING_DIMENSION, PROCESSED_DATA_DIR, HierarchicalKNNConfig
from module6 import AttributeExtractor, RubricCreditEngine
from src.classifier import TwoStageHierarchicalKNNClassifier
from src.cls_index import HierarchicalClsIndex
from src.embedder import BaseEmbedder, HashingDemoEmbedder
from src.event_cache import EventProfileCache
from src.schemas import ClsSample, EventNormalized, parent_group_from_item_id
from src.vector_ops import l2_norm, l2_normalize


class _StubEmbedder(BaseEmbedder):
    """Deterministic lookup embedder for controlled synthetic D-2 invariant tests."""

    def __init__(self, vectors: Dict[str, List[float]], dimension: int = 4) -> None:
        self._vectors = {k: l2_normalize(v) for k, v in vectors.items()}
        self.dimension = dimension
        self.model_version = "stub-unit-test-v1"

    def encode_queries(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vectors[t] for t in texts]

    def encode_passages(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vectors[t] for t in texts]


class TestEmbedderAndVectorOps(unittest.TestCase):
    def test_hashing_embedder_384d_l2_normalized(self) -> None:
        embedder = HashingDemoEmbedder()
        vecs = embedder.encode_queries(
            [
                "Hội thảo học thuật về AI và Machine Learning",
                "Chiến dịch tình nguyện Mùa Hè Xanh 2026",
            ]
        )
        self.assertEqual(len(vecs), 2)
        for v in vecs:
            self.assertEqual(len(v), EMBEDDING_DIMENSION)
            self.assertAlmostEqual(l2_norm(v), 1.0, places=6)


class TestTwoStageHierarchicalKNNInvariants(unittest.TestCase):
    def test_unset_d2_open_parameters_raise_value_error(self) -> None:
        embedder = HashingDemoEmbedder()
        index = HierarchicalClsIndex(dimension=embedder.dimension)
        clf = TwoStageHierarchicalKNNClassifier(
            embedder=embedder,
            cls_index=index,
            config=HierarchicalKNNConfig(),  # all D-2 params intentionally None
        )
        ev = EventNormalized(event_id="e1", title="Seminar AI")
        with self.assertRaises(ValueError) as ctx:
            clf.classify_event(ev)
        self.assertIn("stage1_top_k", str(ctx.exception))

    def test_stage2_searches_union_of_all_selected_stage1_groups_and_allows_rank2_group_to_win(
        self,
    ) -> None:
        """Critical D-2 test: Stage 1 ranks C1 #1 and C3 #2, but C3's fine item '3.5' wins in Stage 2 union!"""
        # 4D orthogonal-ish synthetic space:
        # Query = [1, 0, 0, 0]
        # C1 has 3 samples moderately aligned with query (sim ~ 0.71) split across 1.6 and 1.9
        # C3 has 1 sample nearly identical to query (sim = 0.99) for 3.5 -> Stage 1 (k=4) totals:
        #   C1 weight = 3 * 0.71 = 2.13 (Stage 1 #1)
        #   C3 weight = 1 * 0.99 = 0.99 (Stage 1 #2)
        # In Stage 2 union (k=1 or k=2) over {C1, C3}, C3's '3.5' (sim=0.99) beats C1's '1.6'/'1.9' (0.71)!
        vec_map = {
            "q": [1.0, 0.0, 0.0, 0.0],
            "c1_a": [0.71, 0.70, 0.0, 0.0],
            "c1_b": [0.71, 0.0, 0.70, 0.0],
            "c1_c": [0.71, 0.0, 0.0, 0.70],
            "c3_win": [0.99, 0.14, 0.0, 0.0],
            "c4_far": [0.0, 0.0, 0.0, 1.0],
        }
        embedder = _StubEmbedder(vec_map, dimension=4)
        index = HierarchicalClsIndex(dimension=4, model_version=embedder.model_version)
        samples = [
            ClsSample("s1", "c1_a", "c1_a", "1.6", "C1"),
            ClsSample("s2", "c1_b", "c1_b", "1.9", "C1"),
            ClsSample("s3", "c1_c", "c1_c", "1.9", "C1"),
            ClsSample("s4", "c3_win", "c3_win", "3.5", "C3"),
            ClsSample("s5", "c4_far", "c4_far", "4.3", "C4"),
        ]
        index.add_samples_with_embedder(samples, embedder)

        cfg = HierarchicalKNNConfig(
            stage1_top_k=2,
            stage1_neighbors=4,
            stage2_neighbors=1,
        )
        clf = TwoStageHierarchicalKNNClassifier(embedder=embedder, cls_index=index, config=cfg)

        result = clf.classify_vector("evt_union_test", vec_map["q"])

        # Verify Stage 1 ranked C1 first, C3 second
        self.assertEqual(result.selected_candidate_groups, ["C1", "C3"])
        self.assertGreater(result.stage1_scores["C1"], result.stage1_scores["C3"])

        # Verify Stage 2 searched the UNION of C1 + C3 and selected C3's '3.5'
        self.assertEqual(result.rubric_item_id, "3.5")
        self.assertEqual(result.parent_group_id, "C3")
        self.assertEqual(parent_group_from_item_id(result.rubric_item_id), result.parent_group_id)

        # Verify Stage 1 and Stage 2 scores remain strictly distinct
        self.assertNotEqual(set(result.stage1_scores.keys()), set(result.stage2_scores.keys()))
        self.assertIn("C1", result.stage1_scores)
        self.assertIn("3.5", result.stage2_scores)

        # Verify uncalibrated guard (D-2 Requirement 5)
        self.assertFalse(result.is_calibrated)
        self.assertIsNone(result.calibrated_confidence)
        self.assertEqual(result.status, "uncalibrated")

    def test_custom_calibrator_sets_accepted_or_pending_review(self) -> None:
        vec_map = {
            "q": [1.0, 0.0, 0.0, 0.0],
            "s1": [1.0, 0.0, 0.0, 0.0],
        }
        embedder = _StubEmbedder(vec_map, dimension=4)
        index = HierarchicalClsIndex(dimension=4, model_version=embedder.model_version)
        index.add_samples_with_embedder([ClsSample("s1", "s1", "s1", "1.6", "C1")], embedder)

        cfg_accept = HierarchicalKNNConfig(
            stage1_top_k=1,
            stage1_neighbors=1,
            stage2_neighbors=1,
            acceptance_threshold=0.75,
        )
        clf_accept = TwoStageHierarchicalKNNClassifier(
            embedder=embedder,
            cls_index=index,
            config=cfg_accept,
            calibrator=lambda item, s1, s2, agg: 0.85,
        )
        res_accept = clf_accept.classify_vector("e_acc", vec_map["q"])
        self.assertTrue(res_accept.is_calibrated)
        self.assertAlmostEqual(res_accept.calibrated_confidence, 0.85)
        self.assertEqual(res_accept.status, "accepted")

        clf_review = TwoStageHierarchicalKNNClassifier(
            embedder=embedder,
            cls_index=index,
            config=cfg_accept,
            calibrator=lambda item, s1, s2, agg: 0.60,
        )
        res_review = clf_review.classify_vector("e_rev", vec_map["q"])
        self.assertTrue(res_review.is_calibrated)
        self.assertEqual(res_review.status, "pending_review")


class TestDemoDatasetIntegrationAndPersistence(unittest.TestCase):
    def test_cls_index_save_load_and_incremental_user_correction(self) -> None:
        demo_dataset_path = CLS_INDEX_DIR / "hierarchical_dataset_demo.json"
        self.assertTrue(demo_dataset_path.is_file(), "Run scripts.prepare_cls_dataset first")

        embedder = HashingDemoEmbedder()
        index = HierarchicalClsIndex.from_hierarchical_dataset(demo_dataset_path, embedder)
        self.assertEqual(index.size, 145)
        self.assertEqual(set(index.parent_groups), {"C1", "C2", "C3", "C4", "C5"})

        # Add incremental user-corrected sample
        corrected = ClsSample(
            sample_id="user_corr_0001",
            text="Giải chạy Marathon Sinh viên HCMUS 2026 | Vai trò: Tham gia",
            title="Giải chạy Marathon Sinh viên HCMUS 2026",
            rubric_item_id="3.1",
            parent_group_id="C3",
            role="Tham gia",
            source="user_corrected",
        )
        index.add_samples_with_embedder([corrected], embedder)
        self.assertEqual(index.size, 146)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "cls_index"
            index.save(save_dir)
            reloaded = HierarchicalClsIndex.load(save_dir)
            self.assertEqual(reloaded.size, 146)
            self.assertEqual(reloaded.parent_groups, index.parent_groups)
            self.assertEqual(
                len(reloaded.get_stage2_union_entries(["C1", "C3"])),
                len(reloaded.get_group_entries("C1")) + len(reloaded.get_group_entries("C3")),
            )

    def test_end_to_end_p2_to_module6_to_event_profile_cache(self) -> None:
        demo_dataset_path = CLS_INDEX_DIR / "hierarchical_dataset_demo.json"
        fixtures_path = PROCESSED_DATA_DIR / "test_fixtures_events.json"
        self.assertTrue(fixtures_path.is_file(), "Run scripts.prepare_test_fixtures first")

        embedder = HashingDemoEmbedder()
        index = HierarchicalClsIndex.from_hierarchical_dataset(demo_dataset_path, embedder)
        cfg = HierarchicalKNNConfig(
            stage1_top_k=2,
            stage1_neighbors=5,
            stage2_neighbors=3,
        )
        clf = TwoStageHierarchicalKNNClassifier(embedder=embedder, cls_index=index, config=cfg)
        credit_engine = RubricCreditEngine(attribute_extractor=AttributeExtractor())

        with open(fixtures_path, encoding="utf-8") as handle:
            raw_fixtures = json.load(handle)["events"]

        cache = EventProfileCache(dimension=embedder.dimension)
        for raw_ev in raw_fixtures[:10]:
            ev = EventNormalized.from_dict(raw_ev)
            cls_result, vec = clf.classify_event(ev)
            credit_est = credit_engine.estimate_event_credit(
                event=ev,
                rubric_item_id=cls_result.rubric_item_id,
            )
            cache.upsert(ev, vec, cls_result, credit_estimate=credit_est)

        self.assertEqual(len(cache), 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            emb_path = Path(tmpdir) / "event_embeddings.npy"
            meta_path = Path(tmpdir) / "event_metadata.json"
            cache.save(embeddings_path=emb_path, metadata_path=meta_path)
            loaded = EventProfileCache.load(embeddings_path=emb_path, metadata_path=meta_path)
            self.assertEqual(len(loaded), 10)
            first = loaded.all_profiles()[0]
            self.assertEqual(len(first.vector), EMBEDDING_DIMENSION)
            self.assertAlmostEqual(l2_norm(first.vector), 1.0, places=5)
            self.assertIsNotNone(first.classification.rubric_item_id)
            self.assertIsNotNone(first.credit_estimate)


if __name__ == "__main__":
    unittest.main()
