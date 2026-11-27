"""Unit and integration tests for DB-backed input, 3-vector dual-branch Classification,
and 4-field Recommendation embeddings (`title + description + topic + format`).

All database tests use isolated temporary SQLite databases with
`DRLDatabaseManager.reset_instance()` so the workspace's empty `drl_database.db`
is never mutated.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from configs.model_config import PROJECT_ROOT
from database.drl_db_manager import DRLDatabaseManager
from src.classification.algorithms.hierarchical_two_stage_knn import HierarchicalTwoStageKNN
from src.classification.data.loaders import load_all_assets
from src.classification.embeddings.event_embedder import (
    generate_or_load_multi_profile_query_embeddings,
)
from src.classification.embeddings.event_text_builder import (
    CLASSIFICATION_VECTOR_PROFILES,
    build_all_classification_profile_texts,
    build_classification_profile_text,
    build_event_query_text,
    extract_activity_format_tokens,
)
from src.classification.pipeline.classification_pipeline import ClassificationPipeline
from src.classification.retrieval.rubric_retriever import RubricRetriever
from src.classification.retrieval.sample_retriever import SampleRetriever
from src.db_event_adapter import (
    compute_events_fingerprint,
    db_row_to_classification_event,
    db_row_to_event_normalized,
    fetch_events_from_db,
    parse_coordinate_string,
    parse_fee_to_vnd,
    parse_target_audience_list,
    write_classification_category_to_db,
)
from src.embedder import BaseEmbedder
from src.recommendation.demo_module6 import DemoCreditProvider
from src.recommendation.pipeline import (
    build_demo_profile_from_db_user,
    load_recommendation_candidate_pool_from_db,
    run_recommendation_demo,
)
from src.recommendation.rec_embedder import (
    DEFAULT_REC_PASSAGE_PREFIX,
    REC_EMBEDDING_FIELDS,
    build_recommendation_event_text,
    generate_or_load_recommendation_embeddings,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "classification_config.yaml"


class DeterministicTestEmbedder(BaseEmbedder):
    """Fast deterministic 384-d unit-norm embedder sensitive to every input word."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self.encoded_queries: List[str] = []
        self.encoded_passages: List[str] = []

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_vec(self, text: str) -> List[float]:
        vec = np.zeros(self._dim, dtype=np.float64)
        tokens = [t.strip().lower() for t in str(text or "").split() if t.strip()]
        if not tokens:
            vec[0] = 1.0
            return vec.tolist()
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            for k in range(4):
                idx = int.from_bytes(digest[k * 4 : k * 4 + 2], "little") % self._dim
                sign = 1.0 if (digest[k * 4 + 2] & 1) == 0 else -1.0
                vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        else:
            vec[0] = 1.0
        return vec.astype(np.float32).tolist()

    def encode_query(self, text: str) -> List[float]:
        self.encoded_queries.append(text)
        return self._hash_vec(f"query: {text}")

    def encode_queries(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.encode_query(t) for t in texts]

    def encode_passage(self, text: str) -> List[float]:
        self.encoded_passages.append(text)
        return self._hash_vec(f"passage: {text}")

    def encode_passages(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.encode_passage(t) for t in texts]


class SpySampleRetriever(SampleRetriever):
    """Records exact query vectors passed into Sample Branch Stage 1 & Stage 2."""

    def __init__(self, base: SampleRetriever) -> None:
        self.__dict__.update(base.__dict__)
        self.stage1_vectors: List[np.ndarray] = []
        self.stage2_vectors: List[np.ndarray] = []

    def retrieve_stage1(self, query_vector: np.ndarray, top_k: int) -> List[Any]:
        self.stage1_vectors.append(np.asarray(query_vector, dtype=np.float32).copy())
        return super().retrieve_stage1(query_vector, top_k=top_k)

    def retrieve_stage2(
        self,
        query_vector: np.ndarray,
        candidate_groups: Sequence[str],
        top_k: int,
    ) -> List[Any]:
        self.stage2_vectors.append(np.asarray(query_vector, dtype=np.float32).copy())
        return super().retrieve_stage2(query_vector, candidate_groups=candidate_groups, top_k=top_k)


class SpyRubricRetriever(RubricRetriever):
    """Records exact query vectors passed into Rubric Branch Stage 1 & Stage 2."""

    def __init__(self, base: RubricRetriever) -> None:
        self.__dict__.update(base.__dict__)
        self.stage1_vectors: List[np.ndarray] = []
        self.stage2_vectors: List[np.ndarray] = []

    def retrieve_stage1_groups(self, query_vector: np.ndarray) -> List[Any]:
        self.stage1_vectors.append(np.asarray(query_vector, dtype=np.float32).copy())
        return super().retrieve_stage1_groups(query_vector)

    def retrieve_stage2_criteria(
        self, query_vector: np.ndarray, candidate_groups: Sequence[str]
    ) -> List[Any]:
        self.stage2_vectors.append(np.asarray(query_vector, dtype=np.float32).copy())
        return super().retrieve_stage2_criteria(query_vector, candidate_groups=candidate_groups)


class TestDBMultiVectorArchitecture(unittest.TestCase):
    def setUp(self) -> None:
        DRLDatabaseManager.reset_instance()
        self._tmp_dir = tempfile.TemporaryDirectory()
        # LIFO cleanup: reset_instance closes SQLite connection BEFORE TemporaryDirectory deletes files
        self.addCleanup(self._tmp_dir.cleanup)
        self.addCleanup(DRLDatabaseManager.reset_instance)
        self.tmp_path = Path(self._tmp_dir.name)
        self.tmp_db_path = self.tmp_path / "test_drl_database.db"
        self.db = DRLDatabaseManager(db_path=self.tmp_db_path)
        self.embedder = DeterministicTestEmbedder(dim=384)

    # ------------------------------------------------------------------
    # 1. Empty Database Graceful Handling
    # ------------------------------------------------------------------

    def test_empty_database_classification_and_recommendation_succeed_gracefully(self) -> None:
        rows = fetch_events_from_db(db=self.db)
        self.assertEqual(rows, [])

        pipeline = ClassificationPipeline(CONFIG_PATH, project_root=PROJECT_ROOT)
        cls_res = pipeline.run_from_database(
            db=self.db,
            branch_vector_selection={
                "sample_branch": "title",
                "rubric_branch": "title_description_format",
            },
            write_back_db=True,
            write_output_files=False,
            embedder=self.embedder,
        )
        self.assertEqual(cls_res["source_mode"], "database")
        self.assertEqual(cls_res["total_input_events"], 0)
        self.assertEqual(cls_res["embedded_events_count"], 0)
        self.assertEqual(cls_res["db_categories_updated"], 0)
        self.assertTrue(cls_res["asset_validation_passed"])
        self.assertEqual(cls_res["events"], [])
        self.assertEqual(cls_res["branch_vector_selection"]["sample_branch"], "title")
        self.assertEqual(
            cls_res["branch_vector_selection"]["rubric_branch"],
            "title_description_format",
        )
        for p in CLASSIFICATION_VECTOR_PROFILES:
            self.assertEqual(cls_res["multi_profile_embeddings"].vectors_by_profile[p].shape, (0, 384))

        rec_pool = load_recommendation_candidate_pool_from_db(
            db=self.db,
            embedder=self.embedder,
        )
        self.assertEqual(rec_pool, [])

        sample_embs = np.load(PROJECT_ROOT / "data" / "processed" / "cls_index" / "sample_embeddings.npy")
        rec_out = run_recommendation_demo(
            {"include_mandatory_demo": False, "interest_chips": ["ai_ml"]},
            candidates=rec_pool,
            sample_embeddings=sample_embs,
            embedder=self.embedder,
        )
        self.assertEqual(rec_out["pipeline_summary"]["total_batch_events"], 0)
        self.assertEqual(rec_out["pipeline_summary"]["predicted_events_entered_p5"], 0)
        self.assertEqual(rec_out["ranked_events"], [])
        self.assertEqual(rec_out["dropped_events"], [])
        self.assertEqual(
            rec_out["interest_calibration"]["event_embedding_fields"],
            list(REC_EMBEDDING_FIELDS),
        )
        self.assertEqual(
            rec_out["interest_calibration"]["event_embedding_prefix"],
            DEFAULT_REC_PASSAGE_PREFIX,
        )

    # ------------------------------------------------------------------
    # 2. Populated Database Adapter, Filtering, User Hydration & Write-back
    # ------------------------------------------------------------------

    def test_db_adapter_schema_mapping_and_category_writeback(self) -> None:
        id_a = self.db.insert_record(
            "events",
            {
                "title": "Cuộc thi An toàn thông tin HCMUS CTF 2026",
                "description": "Cuộc thi học thuật về bảo mật mạng dành cho sinh viên.",
                "keywords": ["CTF", "an toàn thông tin"],
                "is_online": False,
                "is_cancelled": False,
                "reg_start_datetime": "2026-10-01T08:00:00+07:00",
                "reg_end_datetime": "2026-10-10T23:59:00+07:00",
                "event_start_datetime": "2026-10-15T08:30:00+07:00",
                "event_end_datetime": "2026-10-15T16:30:00+07:00",
                "location": "227 Nguyễn Văn Cừ, Q.5",
                "coordinate": "10.7629,106.6822",
                "fee": "25.000 VNĐ",
                "target_audience": ["Sinh viên K24, K25", "Khoa CNTT"],
                "topic": ["An toàn thông tin", "Lập trình"],
                "format": ["offline", "contest"],
                "award": "Giải nhất 3.000.000 VNĐ",
                "registration_url": "https://example.hcmus.edu.vn/ctf",
                "sources": "Khoa CNTT - HCMUS",
                "post_url": "https://fb.com/fit.hcmus/posts/101",
                "images_url": ["https://img.example/1.jpg"],
                "category": None,
                "min_credit": 4,
            },
            raise_on_error=True,
        )
        id_b = self.db.insert_record(
            "events",
            {
                "title": "Seminar AI & Large Language Models",
                "description": "Chia sẻ kiến thức nghiên cứu AI hiện đại.",
                "is_online": True,
                "is_cancelled": False,
                "reg_end_datetime": "2026-10-12T20:00:00+07:00",
                "event_start_datetime": "2026-10-14T18:30:00+07:00",
                "event_end_datetime": "2026-10-14T20:30:00+07:00",
                "location": "Zoom Online",
                "fee": "Miễn phí",
                "target_audience": ["Toàn thể sinh viên HCMUS"],
                "topic": ["Trí tuệ nhân tạo"],
                "format": ["online", "seminar"],
                "sources": "CLB AI HCMUS",
                "category": 1,
                "min_credit": 3,
            },
            raise_on_error=True,
        )
        self.db.insert_record(
            "events",
            {
                "title": "Sự kiện đã hủy",
                "description": "Hoãn tổ chức.",
                "is_cancelled": True,
                "category": None,
            },
            raise_on_error=True,
        )

        # Cancelled events excluded by default
        active_rows = fetch_events_from_db(db=self.db, exclude_cancelled=True)
        self.assertEqual([r["event_id"] for r in active_rows], [id_a, id_b])

        unclassified_rows = fetch_events_from_db(db=self.db, only_unclassified=True)
        self.assertEqual([r["event_id"] for r in unclassified_rows], [id_a])

        # Mapping checks
        cls_ev_a = db_row_to_classification_event(active_rows[0])
        self.assertEqual(cls_ev_a["event_id"], id_a)
        self.assertEqual(cls_ev_a["post_id"], f"db_{id_a}")
        self.assertEqual(cls_ev_a["format"], ["offline", "contest"])
        self.assertEqual(cls_ev_a["topic"], ["An toàn thông tin", "Lập trình"])
        self.assertEqual(cls_ev_a["organizer"], "Khoa CNTT - HCMUS")

        norm_a = db_row_to_event_normalized(active_rows[0])
        self.assertEqual(norm_a.event_id, f"db_{id_a}")
        self.assertEqual(norm_a.fee_vnd, 25000)
        self.assertEqual(norm_a.online_mode, "offline")
        self.assertEqual(norm_a.target_audience.cohorts, ["K24", "K25"])
        self.assertEqual(norm_a.target_audience.faculties, ["CNTT"])
        self.assertEqual(norm_a.metadata["db_min_credit"], 4)
        self.assertEqual(parse_coordinate_string(norm_a.metadata["coordinate"]), (10.7629, 106.6822))

        norm_b = db_row_to_event_normalized(active_rows[1])
        self.assertEqual(norm_b.fee_vnd, 0)
        self.assertEqual(norm_b.online_mode, "online")
        self.assertTrue(norm_b.target_audience.all_students)

        # Fee & audience helpers edge cases
        self.assertEqual(parse_fee_to_vnd("30k"), 30000)
        self.assertEqual(parse_fee_to_vnd("Miễn phí"), 0)
        self.assertIsNone(parse_fee_to_vnd(""))
        ta = parse_target_audience_list(["Khóa 2024", "K25"])
        self.assertEqual(ta.cohorts, ["K24", "K25"])

        # Category write-back (C1 -> 1)
        ok = write_classification_category_to_db(id_a, "C1", db=self.db)
        self.assertTrue(ok)
        self.assertEqual(fetch_events_from_db(db=self.db, only_unclassified=True), [])
        updated_row = self.db.fetch_one("SELECT category FROM events WHERE event_id = ?;", (id_a,))
        self.assertEqual(updated_row["category"], 1)

        # User + rubric + subscription hydration
        self.db.insert_record(
            "rubric",
            {"school_id": 1, "max_credit": [30, 20, 25, 25, 30]},
            raise_on_error=True,
        )
        self.db.insert_record(
            "users",
            {
                "user_id": "24120001",
                "school_id": 1,
                "start_year": 2024,
                "credits": [18, 20, 12, 15, 5],
                "preferred_topics": ["AI", "bảo mật"],
                "preferred_formats": ["seminar", "contest"],
                "weights": [0.35, 0.20, 0.30, 0.15],  # [gain, fit, interested, urgency]
                "default_coordinate": "10.7629,106.6822",
            },
            raise_on_error=True,
        )
        self.db.insert_record(
            "event_subscription",
            {"user_id": "24120001", "event_id": id_b, "status": "done", "credit": 3},
            raise_on_error=True,
        )
        profile = build_demo_profile_from_db_user("24120001", db=self.db, embedder=self.embedder)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertAlmostEqual(profile.weights.w1_gain, 0.35, places=5)
        self.assertAlmostEqual(profile.weights.w2_interest, 0.30, places=5)
        self.assertAlmostEqual(profile.weights.w3_fit, 0.20, places=5)
        self.assertAlmostEqual(profile.weights.w4_urgency, 0.15, places=5)
        self.assertEqual(profile.current_by_group["C1"], 18.0)
        self.assertEqual(profile.cohort, "K24")
        self.assertEqual(profile.participated_event_ids, [f"db_{id_b}"])
        self.assertIn("AI", profile.free_text or "")
        self.assertIn("seminar", profile.free_text or "")

    # ------------------------------------------------------------------
    # 3. Classification 3-Vector Profiles & Independent Branch Routing
    # ------------------------------------------------------------------

    def test_three_classification_profiles_and_format_token_filtering(self) -> None:
        ev = {
            "is_event": True,
            "title": "Cuộc thi lập trình HCMUS CodeFest",
            "description": "Giải đấu thuật toán dành cho sinh viên CNTT.",
            "format": ["offline", "in-person", "contest", "hackathon"],
        }
        # Delivery-only tokens stripped, activity tokens preserved
        self.assertEqual(extract_activity_format_tokens(ev), ["contest", "hackathon"])

        texts = build_all_classification_profile_texts(ev)
        self.assertEqual(tuple(texts.keys()), CLASSIFICATION_VECTOR_PROFILES)
        self.assertEqual(texts["title"], "query: Cuộc thi lập trình HCMUS CodeFest")
        self.assertEqual(texts["title_description"], build_event_query_text(ev))
        self.assertIn("Hình thức: contest, hackathon.", texts["title_description_format"])
        self.assertNotIn("offline", texts["title_description_format"].lower())
        self.assertNotIn("in-person", texts["title_description_format"].lower())

        # Pure delivery-only format gracefully degrades without empty 'Hình thức:' label
        ev_delivery_only = {
            "title": "Workshop Python",
            "description": "Hướng dẫn thực hành",
            "format": ["online", "offline"],
        }
        tdf_only = build_classification_profile_text(
            ev_delivery_only, profile="title_description_format"
        )
        self.assertNotIn("Hình thức:", tdf_only)

    def test_multi_profile_embedding_generation_and_cache_reuse(self) -> None:
        events = [
            {
                "event_id": 1,
                "post_id": "db_1",
                "is_event": True,
                "title": "Cuộc thi CTF 2026",
                "description": "Thi an toàn thông tin cấp Khoa.",
                "format": ["offline", "contest"],
            },
            {
                "event_id": 2,
                "post_id": "db_2",
                "is_event": True,
                "title": "Ngày hội Hiến máu tình nguyện",
                "description": "Hoạt động thiện nguyện cộng đồng.",
                "format": ["volunteer"],
            },
        ]
        fp = compute_events_fingerprint(events)
        emb_path = self.tmp_path / "mp_embeddings.npz"
        meta_path = self.tmp_path / "mp_metadata.json"
        emb_cfg = {"embedding_dimension": 384, "query_prefix": "query: ", "model_name": "test-e5"}

        mp1 = generate_or_load_multi_profile_query_embeddings(
            events=events,
            events_path=self.tmp_db_path,
            events_sha256=fp,
            embedding_config=emb_cfg,
            embeddings_path=emb_path,
            metadata_path=meta_path,
            reuse_existing=True,
            embedder=self.embedder,
        )
        self.assertFalse(mp1.reused_from_cache)
        for prof in CLASSIFICATION_VECTOR_PROFILES:
            mat = mp1.vectors_by_profile[prof]
            self.assertEqual(mat.shape, (2, 384))
            norms = np.linalg.norm(mat, axis=1)
            np.testing.assert_allclose(norms, np.ones(2), atol=1e-4)

        # Confirm the 3 vectors for event 0 are genuinely distinct
        v0 = mp1.get_vectors_for_row(0)
        self.assertFalse(np.allclose(v0["title"], v0["title_description"]))
        self.assertFalse(np.allclose(v0["title_description"], v0["title_description_format"]))

        # Second call reuses .npz cache
        mp2 = generate_or_load_multi_profile_query_embeddings(
            events=events,
            events_path=self.tmp_db_path,
            events_sha256=fp,
            embedding_config=emb_cfg,
            embeddings_path=emb_path,
            metadata_path=meta_path,
            reuse_existing=True,
            embedder=self.embedder,
        )
        self.assertTrue(mp2.reused_from_cache)
        for prof in CLASSIFICATION_VECTOR_PROFILES:
            np.testing.assert_allclose(
                mp1.vectors_by_profile[prof],
                mp2.vectors_by_profile[prof],
                atol=1e-6,
            )

    def test_hierarchical_knn_routes_chosen_vectors_independently_to_each_branch(self) -> None:
        assets = load_all_assets(CONFIG_PATH, project_root=PROJECT_ROOT)
        base_sample = SampleRetriever(assets.sample_branch)
        base_rubric = RubricRetriever(assets.rubric_branch, assets.rubric)

        # Build 3 clearly distinct unit vectors
        dim = int(assets.embedding_config["embedding_dimension"])
        rng = np.random.default_rng(2026)
        vecs: Dict[str, np.ndarray] = {}
        for prof in CLASSIFICATION_VECTOR_PROFILES:
            raw = rng.standard_normal(dim).astype(np.float32)
            vecs[prof] = raw / np.linalg.norm(raw)

        # Case A: Branch-level selection (Sample -> title, Rubric -> title_description_format)
        base_params = dict((assets.config.get("algorithm") or {}).get("params") or {})
        spy_s = SpySampleRetriever(base_sample)
        spy_r = SpyRubricRetriever(base_rubric)
        alg_a = HierarchicalTwoStageKNN(
            sample_retriever=spy_s,
            rubric_retriever=spy_r,
            rubric_data=assets.rubric,
            params={
                **base_params,
                "sample_branch_vector": "title",
                "rubric_branch_vector": "title_description_format",
            },
        )
        pred_a = alg_a.classify(vecs["title_description"], query_vectors_by_profile=vecs)
        np.testing.assert_allclose(spy_s.stage1_vectors[0], vecs["title"], atol=1e-6)
        np.testing.assert_allclose(spy_s.stage2_vectors[0], vecs["title"], atol=1e-6)
        np.testing.assert_allclose(spy_r.stage1_vectors[0], vecs["title_description_format"], atol=1e-6)
        np.testing.assert_allclose(spy_r.stage2_vectors[0], vecs["title_description_format"], atol=1e-6)
        self.assertEqual(
            pred_a.algorithm_metadata["vector_routing_used"],
            {
                "sample_stage1": "title",
                "sample_stage2": "title",
                "rubric_stage1": "title_description_format",
                "rubric_stage2": "title_description_format",
            },
        )

        # Case B: Per-stage override across all 4 branch-stage slots
        spy_s2 = SpySampleRetriever(base_sample)
        spy_r2 = SpyRubricRetriever(base_rubric)
        alg_b = HierarchicalTwoStageKNN(
            sample_retriever=spy_s2,
            rubric_retriever=spy_r2,
            rubric_data=assets.rubric,
            params={
                **base_params,
                "sample_stage1_vector": "title",
                "sample_stage2_vector": "title_description",
                "rubric_stage1_vector": "title_description",
                "rubric_stage2_vector": "title_description_format",
            },
        )
        pred_b = alg_b.classify(vecs["title_description"], query_vectors_by_profile=vecs)
        np.testing.assert_allclose(spy_s2.stage1_vectors[0], vecs["title"], atol=1e-6)
        np.testing.assert_allclose(spy_s2.stage2_vectors[0], vecs["title_description"], atol=1e-6)
        np.testing.assert_allclose(spy_r2.stage1_vectors[0], vecs["title_description"], atol=1e-6)
        np.testing.assert_allclose(spy_r2.stage2_vectors[0], vecs["title_description_format"], atol=1e-6)
        self.assertEqual(
            pred_b.algorithm_metadata["vector_routing_used"],
            {
                "sample_stage1": "title",
                "sample_stage2": "title_description",
                "rubric_stage1": "title_description",
                "rubric_stage2": "title_description_format",
            },
        )

    def test_classification_pipeline_run_from_database_with_custom_branch_vectors(self) -> None:
        self.db.insert_record(
            "events",
            {
                "title": "Cuộc thi lập trình HCMUS Hackathon 2026",
                "description": "Cuộc thi học thuật lập trình cho sinh viên CNTT.",
                "format": ["offline", "contest"],
                "topic": ["Lập trình"],
                "is_cancelled": False,
                "category": None,
            },
            raise_on_error=True,
        )
        pipeline = ClassificationPipeline(CONFIG_PATH, project_root=PROJECT_ROOT)
        res = pipeline.run_from_database(
            db=self.db,
            branch_vector_selection={
                "sample_branch": "title_description",
                "rubric_branch": "title_description_format",
            },
            write_back_db=True,
            write_output_files=False,
            embedder=self.embedder,
        )
        self.assertEqual(res["total_input_events"], 1)
        self.assertEqual(res["embedded_events_count"], 1)
        ev_out = res["events"][0]
        cls = ev_out["classification"]
        self.assertIn(cls["status"], {"PREDICTED", "NEEDS_REVIEW", "NO_MATCH"})
        self.assertEqual(
            cls["vector_routing_used"],
            {
                "sample_stage1": "title_description",
                "sample_stage2": "title_description",
                "rubric_stage1": "title_description_format",
                "rubric_stage2": "title_description_format",
            },
        )

    # ------------------------------------------------------------------
    # 4. Recommendation 4-Field Embedding (`title + description + topic + format`)
    # ------------------------------------------------------------------

    def test_recommendation_4field_embedding_and_db_pipeline_end_to_end(self) -> None:
        ev1 = {
            "event_id": "db_10",
            "title": "Ngày hội Khởi nghiệp Công nghệ",
            "description": "Kết nối sinh viên với doanh nghiệp.",
            "topic": ["Khởi nghiệp", "AI"],
            "format": ["offline", "workshop", "networking"],
        }
        ev2 = {
            "event_id": "db_11",
            "title": "Ngày hội Khởi nghiệp Công nghệ",
            "description": "Kết nối sinh viên với doanh nghiệp.",
            "topic": ["Tình nguyện cộng đồng"],
            "format": ["volunteer"],
        }
        txt1 = build_recommendation_event_text(ev1)
        txt2 = build_recommendation_event_text(ev2)
        self.assertTrue(txt1.startswith(DEFAULT_REC_PASSAGE_PREFIX))
        self.assertIn("Chủ đề: Khởi nghiệp, AI.", txt1)
        self.assertIn("Hình thức: workshop, networking.", txt1)
        self.assertNotIn("offline", txt1.lower())
        self.assertNotEqual(txt1, txt2)

        emb_set = generate_or_load_recommendation_embeddings(
            [ev1, ev2],
            embedder=self.embedder,
            source_fingerprint=compute_events_fingerprint([ev1, ev2]),
            reuse_existing=False,
        )
        self.assertEqual(emb_set.embeddings.shape, (2, 384))
        v1 = np.asarray(emb_set.get_vector("db_10"), dtype=np.float32)
        v2 = np.asarray(emb_set.get_vector("db_11"), dtype=np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(v1)), 1.0, places=4)
        # Same title & description, different topic & format -> different vector!
        self.assertFalse(np.allclose(v1, v2))

        # End-to-end DB -> Recommendation (P5 + P6) with DB-only category + min_credit
        self.db.insert_record(
            "events",
            {
                "title": "Workshop Machine Learning & AI ứng dụng",
                "description": "Thực hành xây dựng mô hình học máy với Python.",
                "is_online": True,
                "is_cancelled": False,
                "reg_end_datetime": "2026-10-15T23:59:00+07:00",
                "event_start_datetime": "2026-10-18T09:00:00+07:00",
                "event_end_datetime": "2026-10-18T11:30:00+07:00",
                "location": "Online Zoom",
                "fee": "Miễn phí",
                "target_audience": ["Toàn thể sinh viên"],
                "topic": ["AI", "Machine Learning"],
                "format": ["online", "workshop"],
                "sources": "Khoa CNTT",
                "category": 1,
                "min_credit": 4,
            },
            raise_on_error=True,
        )
        pool = load_recommendation_candidate_pool_from_db(
            db=self.db,
            embedder=self.embedder,
            reuse_existing_embeddings=False,
        )
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0].classification["status"], "PREDICTED")
        self.assertEqual(pool[0].classification["predicted_group_id"], "C1")
        self.assertIsNotNone(pool[0].event_vec)
        self.assertTrue(
            str(pool[0].event.metadata.get("rec_embedding_text", "")).startswith("passage: ")
        )
        self.assertIn("AI, Machine Learning", pool[0].event.metadata["rec_embedding_text"])
        self.assertIn("workshop", pool[0].event.metadata["rec_embedding_text"])

        sample_embs = np.load(PROJECT_ROOT / "data" / "processed" / "cls_index" / "sample_embeddings.npy")
        out = run_recommendation_demo(
            {
                "interest_chips": ["ai_ml"],
                "reference_now": "2026-10-01T08:00:00+07:00",
                "include_mandatory_demo": False,
            },
            candidates=pool,
            sample_embeddings=sample_embs,
            embedder=self.embedder,
            credit_provider=DemoCreditProvider(),
        )
        self.assertEqual(out["pipeline_summary"]["total_batch_events"], 1)
        self.assertEqual(out["pipeline_summary"]["eligible_after_p5"], 1)
        self.assertEqual(len(out["ranked_events"]), 1)
        top = out["ranked_events"][0]
        self.assertEqual(top["parent_group_id"], "C1")
        self.assertEqual(top["credit_estimate"]["effective_points"], 4.0)
        self.assertEqual(top["credit_estimate"]["source"], "explicit")
        self.assertIn("db_min_credit_group_level", top["credit_estimate"]["warning_flags"])
        self.assertTrue(str(top.get("rec_embedding_text") or "").startswith("passage: "))


if __name__ == "__main__":
    unittest.main()
