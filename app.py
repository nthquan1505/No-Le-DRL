"""Interactive Prototype Server & Service for No-Le-DRL Module 3.

Runs with zero external web-framework dependencies (Python stdlib ThreadingHTTPServer)
and connects directly to the real implemented Module 3 & Module 6 components:
- P2.1: E5Embedder (`intfloat/multilingual-e5-small`, 384d L2) by default, with transparent
  fallback to HashingDemoEmbedder when `sentence-transformers` is not yet installed.
- P2.2 & P2.3: TwoStageHierarchicalKNNClassifier + HierarchicalClsIndex (145 demo seeds).
- Module 6: Isolated `module6.RubricCreditEngine` driven by `data/raw/rubric.yaml`.
- Event Fixtures: 108 real HCMUS events from `data/processed/test_fixtures_events.json`.
- Prototype Simulation: P4 profile, Module 4 mock timetable, P5 hard filter, P6 scoring
  (w1..w4 per D-5), P7 MMR + 1 exploration slot & conflict-free planner, P8 grounded
  explanations, and live Feedback Store (including live ClsIndex label corrections).

Usage:
    python3 app.py                # Starts on http://127.0.0.1:8765 (prefers e5-small)
    python3 app.py --port 8080    # Custom port
    python3 app.py --embedder demo
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from configs.model_config import (
    CLS_INDEX_DIR,
    DEMO_EVENTS_FULL_CSV_PATH,
    MODEL_DEVICE,
    MODEL_ID,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_EVENTS_JSON_PATH,
    HierarchicalKNNConfig,
)
from module6 import AttributeExtractor, LedgerEntry, RubricCreditEngine, UserLedger
from scripts.prepare_cls_dataset import build_hierarchical_dataset
from scripts.prepare_test_fixtures import convert_events_file
from src.classification.algorithms.algorithm_registry import build_default_registry
from src.classification.data.loaders import load_all_assets
from src.classification.data.validators import validate_loaded_assets
from src.classification.decision.confidence_handler import ConfidenceAndDecisionHandler
from src.classification.embeddings.event_text_builder import build_event_query_text
from src.classification.pipeline.classification_pipeline import ClassificationPipeline
from src.classification.retrieval.rubric_retriever import RubricRetriever
from src.classification.retrieval.sample_retriever import SampleRetriever
from src.classifier import TwoStageHierarchicalKNNClassifier
from src.cls_index import HierarchicalClsIndex
from src.contract_validator import parse_iso_datetime
from src.embedder import create_embedder
from src.event_cache import EventProfileCache
from src.schemas import (
    ClassificationResult,
    ClsSample,
    EventNormalized,
    ScoringWeights,
    is_valid_item_id,
    parent_group_from_item_id,
)
from src.vector_ops import cosine_similarity, l2_norm, l2_normalize

import numpy as np

PROTOTYPE_DIR = PROJECT_ROOT / "prototype"
CLASSIFICATION_CONFIG_PATH = PROJECT_ROOT / "configs" / "classification_config.yaml"
DEFAULT_REFERENCE_NOW = "2026-10-15T09:00:00+07:00"

INTEREST_CHIPS: List[Dict[str, str]] = [
    {
        "id": "ai_tech",
        "label": "AI & Công nghệ",
        "text": "Hội thảo AI Machine Learning lập trình an toàn thông tin CTF Hackathon khoa CNTT",
    },
    {
        "id": "academic_nckh",
        "label": "Học thuật & NCKH",
        "text": "Seminar học thuật nghiên cứu khoa học sinh viên Olympic Toán Tin học ICPC",
    },
    {
        "id": "soft_skills",
        "label": "Kỹ năng mềm & Hướng nghiệp",
        "text": "Workshop kỹ năng mềm viết CV phỏng vấn tuyển dụng người trong nghề hành trang sinh viên",
    },
    {
        "id": "volunteering",
        "label": "Tình nguyện & Cộng đồng",
        "text": "Chiến dịch tình nguyện Mùa Hè Xanh Xuân Tình Nguyện thiện nguyện hiến máu quyên góp cộng đồng",
    },
    {
        "id": "arts_sports",
        "label": "Văn nghệ & Thể thao",
        "text": "Hội thao bóng đá chạy bộ giải thể thao văn nghệ âm nhạc sinh viên khỏe",
    },
    {
        "id": "english_global",
        "label": "Ngoại ngữ & Hội nhập",
        "text": "Thi thử TOEIC tiếng Anh Star Awards CLB Anh văn hội nhập quốc tế sinh viên 5 tốt",
    },
]

# Mock Module 4 weekly student timetable (day_of_week: 0=Mon .. 6=Sun, start_h, end_h)
MOCK_WEEKLY_TIMETABLE: List[Dict[str, Any]] = [
    {"day": 0, "day_label": "Thứ 2", "start_hour": 7.5, "end_hour": 11.5, "course": "Nhập môn Tư duy Tính toán", "campus": "CS1 - Nguyễn Văn Cừ"},
    {"day": 1, "day_label": "Thứ 3", "start_hour": 13.0, "end_hour": 16.5, "course": "Cấu trúc dữ liệu & Giải thuật", "campus": "CS2 - Linh Trung"},
    {"day": 2, "day_label": "Thứ 4", "start_hour": 7.5, "end_hour": 11.0, "course": "Xác suất Thống kê", "campus": "CS1 - Nguyễn Văn Cừ"},
    {"day": 3, "day_label": "Thứ 5", "start_hour": 13.0, "end_hour": 17.0, "course": "Thực hành Lập trình", "campus": "CS2 - Linh Trung"},
    {"day": 4, "day_label": "Thứ 6", "start_hour": 7.5, "end_hour": 10.5, "course": "Cơ sở Trí tuệ Nhân tạo", "campus": "CS1 - Nguyễn Văn Cừ"},
]

WEIGHT_PRESETS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "label": "Cân bằng",
        "weights": {"w1_gain": 0.40, "w2_interest": 0.25, "w3_fit": 0.20, "w4_urgency": 0.15},
    },
    "need_points": {
        "label": "Cần đủ điểm ĐRL",
        "weights": {"w1_gain": 0.55, "w2_interest": 0.15, "w3_fit": 0.15, "w4_urgency": 0.15},
    },
    "interest_first": {
        "label": "Ưu tiên sở thích",
        "weights": {"w1_gain": 0.20, "w2_interest": 0.50, "w3_fit": 0.20, "w4_urgency": 0.10},
    },
    "exam_week": {
        "label": "Tuần thi / Ít thời gian",
        "weights": {"w1_gain": 0.25, "w2_interest": 0.15, "w3_fit": 0.45, "w4_urgency": 0.15},
    },
}


class PrototypeService:
    """Stateful prototype engine backing all UI views & REST API endpoints."""

    def __init__(
        self,
        embedder_mode: str = "e5",
        allow_fallback: bool = True,
        reference_now: str = DEFAULT_REFERENCE_NOW,
    ) -> None:
        self.reference_now_str = reference_now
        self.reference_now = parse_iso_datetime(reference_now)
        assert self.reference_now is not None

        # 1. P2.1 Embedder (defaults to intfloat/multilingual-e5-small)
        self.embedder, self.embedder_status = create_embedder(
            mode=embedder_mode,
            model_id=MODEL_ID,
            device=MODEL_DEVICE,
            allow_fallback=allow_fallback,
        )

        # 2. Ensure Stage 1 processed demo artifacts exist
        demo_dataset_path = CLS_INDEX_DIR / "hierarchical_dataset_demo.json"
        if not demo_dataset_path.is_file():
            build_hierarchical_dataset(
                csv_path=DEMO_EVENTS_FULL_CSV_PATH,
                events_json_path=RAW_EVENTS_JSON_PATH,
                output_path=demo_dataset_path,
            )

        fixtures_path = PROCESSED_DATA_DIR / "test_fixtures_events.json"
        if not fixtures_path.is_file():
            convert_events_file(
                input_path=RAW_EVENTS_JSON_PATH,
                output_path=fixtures_path,
            )

        # 3. Build HierarchicalClsIndex (for legacy feedback loop tests) + Dual-Branch 2-Stage kNN System
        self.cls_index = HierarchicalClsIndex.from_hierarchical_dataset(
            demo_dataset_path,
            self.embedder,
        )
        self.preview_knn_config = HierarchicalKNNConfig(
            stage1_top_k=2,
            stage1_neighbors=5,
            stage2_neighbors=3,
        )
        self.classifier = TwoStageHierarchicalKNNClassifier(
            embedder=self.embedder,
            cls_index=self.cls_index,
            config=self.preview_knn_config,
        )

        # 3b. Initialize real Dual-Branch Hierarchical 2-Stage kNN (src/classification)
        self.dual_branch_config_path = CLASSIFICATION_CONFIG_PATH
        self.dual_assets = load_all_assets(self.dual_branch_config_path, project_root=PROJECT_ROOT)
        validate_loaded_assets(self.dual_assets)
        self.dual_sample_retriever = SampleRetriever(self.dual_assets.sample_branch)
        self.dual_rubric_retriever = RubricRetriever(
            self.dual_assets.rubric_branch,
            self.dual_assets.rubric,
        )
        self.dual_registry = build_default_registry()
        self.dual_algorithm_name = str((self.dual_assets.config.get("algorithm") or {}).get("name", "hierarchical_two_stage_knn"))
        self.dual_default_params = dict((self.dual_assets.config.get("algorithm") or {}).get("params") or {})
        self.dual_decision_cfg = dict(self.dual_assets.config.get("decision") or {})
        self.dual_algorithm = self.dual_registry.create(
            self.dual_algorithm_name,
            sample_retriever=self.dual_sample_retriever,
            rubric_retriever=self.dual_rubric_retriever,
            rubric_data=self.dual_assets.rubric,
            params=self.dual_default_params,
        )
        self.dual_decision_handler = ConfidenceAndDecisionHandler(
            rubric_retriever=self.dual_rubric_retriever,
            decision_config=self.dual_decision_cfg,
            max_suggested_candidates=int(self.dual_default_params.get("max_suggested_candidates", 5)),
        )
        self.dual_batch_events: List[Dict[str, Any]] = []
        self.dual_classification_by_post_id: Dict[str, Dict[str, Any]] = {}
        self.dual_query_vectors_by_post_id: Dict[str, List[float]] = {}
        self._load_dual_branch_batch_outputs()

        # 4. Isolated Module 6 Shared Library (replaceable)
        self.attribute_extractor = AttributeExtractor()
        self.credit_engine = RubricCreditEngine(attribute_extractor=self.attribute_extractor)

        # Build canonical rubric catalog (all 51 items) for UI display & label correction dropdown
        self.rubric_catalog: List[Dict[str, Any]] = []
        self.rubric_name_by_id: Dict[str, str] = {}
        for _, spec in self.credit_engine.rubric["items"].items():
            ref = str(spec["ref"])
            name = str(spec["name"])
            grp = str(spec["group"])
            self.rubric_name_by_id[ref] = name
            self.rubric_catalog.append(
                {
                    "rubric_item_id": ref,
                    "parent_group_id": grp,
                    "name": name,
                    "kind": str(spec.get("kind") or ""),
                }
            )
        self.rubric_catalog.sort(key=lambda x: [int(p) for p in x["rubric_item_id"].split(".")])

        # 5. Load 108 real normalized HCMUS events & populate live EventProfileCache
        with open(fixtures_path, encoding="utf-8") as handle:
            raw_events = json.load(handle)["events"]
        self.events: List[EventNormalized] = [EventNormalized.from_dict(r) for r in raw_events]
        self.cache = EventProfileCache(dimension=self.embedder.dimension)
        self._rebuild_event_cache()

        # 6. Pre-embed Cold-Start interest chips with the same P2.1 embedder
        self.chip_vectors: Dict[str, List[float]] = {
            chip["id"]: self.embedder.encode_query(chip["text"]) for chip in INTEREST_CHIPS
        }

        # 7. Prototype Student State (P4 + Module 6 Ledger + Feedback Store)
        self.student_profile: Dict[str, Any] = {
            "user_id": "sv_23120001",
            "name": "Nguyễn Minh Khoa",
            "cohort": "K23",
            "faculty": "Khoa Công nghệ Thông tin",
            "class_id": "23CTT1",
            "selected_chips": ["ai_tech", "soft_skills", "volunteering"],
            "target_total": 85.0,
            "target_by_group": {"C1": 25.0, "C2": 20.0, "C3": 20.0, "C4": 20.0, "C5": 10.0},
            "max_budget_vnd": 50000,
            "preset": "balanced",
            "weights": dict(WEIGHT_PRESETS["balanced"]["weights"]),
            "exam_week_mode": False,
            "hide_schedule_conflicts": True,
            "hide_group_cap_reached": True,
            "saved_event_ids": [],
            "dismissed_event_ids": [],
            "joined_event_ids": [],
            "label_overrides": {},
        }
        self.ledger = UserLedger(
            entries=[
                LedgerEntry(rubric_item_id="1.9", points=5.0, event_id="past_demo_1"),
            ],
            use_default_compliance_baseline=True,
            target_total=85.0,
            target_by_group=dict(self.student_profile["target_by_group"]),
        )
        self.feedback_log: List[Dict[str, Any]] = []
        self._user_correction_seq = 0

    def _load_dual_branch_batch_outputs(self) -> None:
        """Load precomputed dual-branch classification_events.json and shared query vectors."""
        paths_cfg = self.dual_assets.config.get("paths") or {}
        out_json = PROJECT_ROOT / str(paths_cfg.get("output_classification_json", "outputs/classification_events.json"))
        q_emb_path = PROJECT_ROOT / str(paths_cfg.get("query_embeddings", "data/generated/events_query_embeddings.npy"))
        q_meta_path = PROJECT_ROOT / str(paths_cfg.get("query_metadata", "data/generated/events_query_metadata.json"))

        if not out_json.is_file() or not q_emb_path.is_file() or not q_meta_path.is_file():
            # Generate once if missing (uses cached model if e5 is active)
            if self.embedder_status.is_e5_active:
                pipeline = ClassificationPipeline(self.dual_branch_config_path, project_root=PROJECT_ROOT)
                pipeline.run()

        self.dual_batch_events = []
        self.dual_classification_by_post_id = {}
        self.dual_query_vectors_by_post_id = {}

        if out_json.is_file():
            self.dual_batch_events = json.loads(out_json.read_text(encoding="utf-8"))
            for rec in self.dual_batch_events:
                pid = str(rec.get("post_id") or "").strip()
                cls_obj = rec.get("classification")
                if pid and isinstance(cls_obj, dict):
                    self.dual_classification_by_post_id[pid] = cls_obj

        if q_emb_path.is_file() and q_meta_path.is_file():
            q_vecs = np.load(q_emb_path, mmap_mode="r")
            q_meta = json.loads(q_meta_path.read_text(encoding="utf-8"))
            for row_meta in q_meta.get("records") or []:
                pid = str(row_meta.get("post_id") or "").strip()
                r_idx = int(row_meta["embedding_row"])
                if pid and 0 <= r_idx < q_vecs.shape[0]:
                    self.dual_query_vectors_by_post_id[pid] = [float(v) for v in q_vecs[r_idx].tolist()]

    def _dual_branch_summary(self) -> Dict[str, Any]:
        status_counts: Dict[str, int] = {
            "PREDICTED": 0,
            "NEEDS_REVIEW": 0,
            "NO_MATCH": 0,
            "SKIPPED_NOT_EVENT": 0,
        }
        for rec in self.dual_batch_events:
            st = str((rec.get("classification") or {}).get("status") or "")
            if st in status_counts:
                status_counts[st] += 1
        return {
            "enabled": True,
            "algorithm": self.dual_algorithm_name,
            "registered_algorithms": self.dual_registry.list_algorithms(),
            "parameter_status": "provisional_not_evaluated",
            "sample_reference_count": self.dual_sample_retriever.eligible_count,
            "rubric_stage1_count": len(self.dual_rubric_retriever.eligible_stage1_groups),
            "rubric_total_prototypes": int(len(self.dual_assets.rubric_branch.metadata)),
            "batch_events_count": len(self.dual_batch_events),
            "status_counts": status_counts,
            "default_params": {
                "stage1_top_k": int(self.dual_default_params.get("candidate_group_count", 2)),
                "k1": int(self.dual_default_params.get("k1", 10)),
                "k2": int(self.dual_default_params.get("k2", 10)),
                "beta": float(self.dual_default_params.get("stage1_fusion_beta", 0.5)),
                "alpha": float(self.dual_default_params.get("stage2_fusion_alpha", 0.5)),
            },
        }

    def get_dual_branch_events(self) -> Dict[str, Any]:
        """Return all 108 batch-classified events from the dual-branch pipeline for UI inspection."""
        items = []
        for rec in self.dual_batch_events:
            cls = rec.get("classification") or {}
            top_sugg = (cls.get("suggested_candidates") or [None])[0]
            items.append(
                {
                    "post_id": rec.get("post_id"),
                    "title": rec.get("title") or "",
                    "description": rec.get("description") or "",
                    "organizer": rec.get("organizer") or "",
                    "date": rec.get("date"),
                    "status": cls.get("status"),
                    "predicted_group_id": cls.get("predicted_group_id"),
                    "predicted_criterion_ref": cls.get("predicted_criterion_ref"),
                    "predicted_criterion_name": cls.get("predicted_criterion_name"),
                    "top_suggested_ref": top_sugg.get("criterion_ref") if top_sugg else None,
                    "top_suggested_group": top_sugg.get("group_id") if top_sugg else None,
                    "top_suggested_score": top_sugg.get("fused_score") if top_sugg else None,
                    "branch_agreement": (cls.get("confidence") or {}).get("branch_agreement"),
                    "review_reasons": cls.get("review_reasons") or [],
                    "classification": cls,
                }
            )
        return {
            "summary": self._dual_branch_summary(),
            "events": items,
        }

    def run_dual_branch_pipeline(self) -> Dict[str, Any]:
        """Re-run ClassificationPipeline (reusing verified shared query cache) and refresh UI state."""
        pipeline = ClassificationPipeline(self.dual_branch_config_path, project_root=PROJECT_ROOT)
        run_summary = pipeline.run()
        self._load_dual_branch_batch_outputs()
        self._rebuild_event_cache()
        return {
            "status": "ok",
            "algorithm": run_summary.algorithm_name,
            "query_embeddings_reused": run_summary.query_embeddings_reused,
            "total_input_events": run_summary.total_input_events,
            "embedded_events_count": run_summary.embedded_events_count,
            "status_counts": run_summary.status_counts,
            "all_checks_passed": (
                run_summary.asset_validation.all_passed
                and run_summary.query_validation.all_passed
                and run_summary.output_validation.all_passed
            ),
            "dual_branch_events": self.get_dual_branch_events(),
            "recommendations_bundle": self.get_recommendations(),
            "schedule_plan": self.generate_schedule_plan(),
        }

    def _dual_to_legacy_classification_result(
        self,
        event_id: str,
        dual_cls: Dict[str, Any],
    ) -> ClassificationResult:
        """Adapt a dual-branch classification payload into ClassificationResult for EventProfileCache."""
        scores = dual_cls.get("scores") or {}
        s1_fused = (scores.get("stage1") or {}).get("fused") or []
        s2_fused = (scores.get("stage2") or {}).get("fused") or []
        selected_groups = [g["group_id"] for g in s1_fused if g.get("selected_for_stage2")]
        stage1_scores = {g["group_id"]: float(g["fused_score"]) for g in s1_fused}
        stage2_scores = {c["criterion_ref"]: float(c["fused_score"]) for c in s2_fused}

        status_raw = str(dual_cls.get("status") or "NEEDS_REVIEW")
        # For EventProfileCache / Module 6 preview: use predicted ref if PREDICTED,
        # otherwise fall back to rank-1 suggested candidate while keeping status='pending_review'.
        top_ref = dual_cls.get("predicted_criterion_ref")
        top_grp = dual_cls.get("predicted_group_id")
        if top_ref is None and s2_fused:
            top_ref = s2_fused[0].get("criterion_ref")
            top_grp = s2_fused[0].get("group_id")

        status_map = {
            "PREDICTED": "uncalibrated",
            "NEEDS_REVIEW": "pending_review",
            "NO_MATCH": "rejected",
            "SKIPPED_NOT_EVENT": "rejected",
        }
        mapped_status = status_map.get(status_raw, "uncalibrated")

        top_k_items = [
            {
                "rubric_item_id": c["criterion_ref"],
                "parent_group_id": c["group_id"],
                "stage2_score": float(c["fused_score"]),
                "stage1_parent_score": float(stage1_scores.get(c["group_id"], 0.0)),
                "ranking_score": float(c["fused_score"]),
            }
            for c in s2_fused[:6]
        ]

        return ClassificationResult(
            event_id=event_id,
            rubric_item_id=top_ref,
            parent_group_id=top_grp,
            selected_candidate_groups=selected_groups,
            stage1_scores=stage1_scores,
            stage2_scores=stage2_scores,
            aggregated_scores=stage2_scores,
            calibrated_confidence=None,
            is_calibrated=False,
            status=mapped_status,
            top_k_items=top_k_items,
            model_version=f"{self.embedder.model_version} ({self.dual_algorithm_name})",
        )

    def _rebuild_event_cache(self) -> None:
        """Populate EventProfileCache from Dual-Branch 2-Stage kNN + Module 6 baseline estimate."""
        for ev in self.events:
            dual_cls = self.dual_classification_by_post_id.get(ev.event_id)
            shared_vec = self.dual_query_vectors_by_post_id.get(ev.event_id)
            if dual_cls is not None and shared_vec is not None and self.embedder_status.is_e5_active:
                cls_res = self._dual_to_legacy_classification_result(ev.event_id, dual_cls)
                vec = shared_vec
            else:
                cls_res, vec = self.classifier.classify_event(ev)

            credit = self.credit_engine.estimate_event_credit(
                event=ev,
                rubric_item_id=cls_res.rubric_item_id,
            )
            self.cache.upsert(ev, vec, cls_res, credit_estimate=credit)

    def _compute_user_vector(self) -> List[float]:
        """P4 adaptive user_vec = normalize(0.7 * u_declared + 0.3 * u_learned)."""
        dim = self.embedder.dimension
        declared = [0.0] * dim
        selected_chips = self.student_profile["selected_chips"] or ["ai_tech"]
        for chip_id in selected_chips:
            vec = self.chip_vectors.get(chip_id)
            if vec:
                for i in range(dim):
                    declared[i] += vec[i]
        u_declared = l2_normalize(declared)

        learned = [0.0] * dim
        has_learned = False
        for eid in self.student_profile["saved_event_ids"]:
            prof = self.cache.get(eid)
            if prof:
                has_learned = True
                for i in range(dim):
                    learned[i] += 1.0 * prof.vector[i]
        for eid in self.student_profile["joined_event_ids"]:
            prof = self.cache.get(eid)
            if prof:
                has_learned = True
                for i in range(dim):
                    learned[i] += 1.5 * prof.vector[i]
        for eid in self.student_profile["dismissed_event_ids"]:
            prof = self.cache.get(eid)
            if prof:
                has_learned = True
                for i in range(dim):
                    learned[i] -= 0.4 * prof.vector[i]

        if not has_learned:
            return u_declared
        u_learned = l2_normalize(learned)
        blended = [0.7 * u_declared[i] + 0.3 * u_learned[i] for i in range(dim)]
        return l2_normalize(blended)

    def _check_timetable_conflict(self, ev: EventNormalized) -> Tuple[bool, Optional[str]]:
        """Mock Module 4 schedule overlap check against MOCK_WEEKLY_TIMETABLE."""
        if not ev.start_datetime:
            return False, None
        start_dt = parse_iso_datetime(ev.start_datetime)
        end_dt = parse_iso_datetime(ev.end_datetime) if ev.end_datetime else None
        if start_dt is None:
            return False, None
        if end_dt is None:
            end_dt = start_dt + timedelta(hours=2)

        # Online asynchronous events do not trigger physical travel/class conflict
        if ev.online_mode == "online":
            return False, None

        ev_start_h = start_dt.hour + start_dt.minute / 60.0
        ev_end_h = end_dt.hour + end_dt.minute / 60.0
        if ev_end_h <= ev_start_h:
            ev_end_h = ev_start_h + 2.0

        day = start_dt.weekday()
        for slot in MOCK_WEEKLY_TIMETABLE:
            if slot["day"] != day:
                continue
            # 30-min (0.5h) travel buffer around classes
            if ev_start_h < (slot["end_hour"] + 0.5) and ev_end_h > (slot["start_hour"] - 0.5):
                return True, f"Trùng {slot['course']} ({slot['day_label']} {slot['start_hour']:.1f}h-{slot['end_hour']:.1f}h)"
        return False, None

    def _compute_fit_score(self, ev: EventNormalized, has_conflict: bool) -> float:
        """Prototype Module 4 Fit score in [0, 1]."""
        if has_conflict:
            return 0.10
        score = 0.70
        if ev.online_mode == "online":
            score += 0.25
        elif ev.online_mode == "hybrid":
            score += 0.15
        elif ev.location and ("nguyễn văn cừ" in ev.location.lower() or "cơ sở 1" in ev.location.lower()):
            score += 0.12
        if ev.fee_vnd == 0:
            score += 0.05
        if self.student_profile.get("exam_week_mode") and ev.online_mode == "offline":
            score -= 0.25
        return max(0.0, min(1.0, score))

    def _compute_urgency_score(self, ev: EventNormalized) -> Tuple[float, Optional[str]]:
        """Compute Urgency in [0, 1] and optional Module 5 near-deadline alert."""
        deadline_str = ev.reg_end_datetime or ev.start_datetime
        if not deadline_str:
            return 0.25, None
        dt = parse_iso_datetime(deadline_str)
        if dt is None:
            return 0.25, None
        days_left = (dt - self.reference_now).total_seconds() / 86400.0
        if days_left <= 0:
            return 0.0, None
        if days_left <= 3:
            return 1.0, f"Sắp hết hạn ĐK (còn {max(1, int(math.ceil(days_left)))} ngày)"
        if days_left <= 7:
            return 0.80, f"Hạn ĐK trong {int(math.ceil(days_left))} ngày"
        if days_left <= 14:
            return 0.55, None
        if days_left <= 30:
            return 0.35, None
        return 0.15, None

    def _get_effective_weights(self) -> ScoringWeights:
        w = self.student_profile["weights"]
        raw = [
            max(0.0, float(w.get("w1_gain", 0.40))),
            max(0.0, float(w.get("w2_interest", 0.25))),
            max(0.0, float(w.get("w3_fit", 0.20))),
            max(0.0, float(w.get("w4_urgency", 0.15))),
        ]
        total = sum(raw) or 1.0
        return ScoringWeights(
            w1_gain=raw[0] / total,
            w2_interest=raw[1] / total,
            w3_fit=raw[2] / total,
            w4_urgency=raw[3] / total,
        )

    def get_recommendations(self, top_n: int = 12) -> Dict[str, Any]:
        """Run P5 (Filter) -> P6 (Scoring w1..w4) -> P7 (MMR + 1 Explore Slot) -> P8 (Explain)."""
        weights = self._get_effective_weights()
        user_vec = self._compute_user_vector()
        ledger_eval = self.credit_engine.evaluate_ledger(self.ledger)
        shortfall = self.credit_engine.compute_group_shortfall(self.ledger)

        eligible: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []

        dismissed_set = set(self.student_profile["dismissed_event_ids"])
        joined_set = set(self.student_profile["joined_event_ids"])
        saved_set = set(self.student_profile["saved_event_ids"])
        max_budget = self.student_profile.get("max_budget_vnd")
        hide_conflicts = bool(self.student_profile.get("hide_schedule_conflicts", True))
        hide_capped = bool(self.student_profile.get("hide_group_cap_reached", True))

        for prof in self.cache.all_profiles():
            ev = prof.event
            cls_res = prof.classification
            effective_item_id = self.student_profile["label_overrides"].get(
                ev.event_id,
                cls_res.rubric_item_id,
            )
            effective_group_id = (
                parent_group_from_item_id(effective_item_id)
                if effective_item_id and is_valid_item_id(effective_item_id)
                else cls_res.parent_group_id
            )

            credit = self.credit_engine.estimate_event_credit(
                event=ev,
                rubric_item_id=effective_item_id,
                ledger=self.ledger,
            )
            reasons: List[Dict[str, str]] = []

            if ev.event_id in joined_set:
                reasons.append({"code": "already_joined", "label": "Đã tham gia"})
            if ev.event_id in dismissed_set:
                reasons.append({"code": "dismissed_by_user", "label": "Bạn đã ẩn sự kiện này"})

            # Deadline / start check relative to reference_now
            start_dt = parse_iso_datetime(ev.start_datetime) if ev.start_datetime else None
            reg_end_dt = parse_iso_datetime(ev.reg_end_datetime) if ev.reg_end_datetime else None
            if start_dt and start_dt < self.reference_now:
                reasons.append({"code": "event_expired", "label": "Sự kiện đã diễn ra"})
            elif reg_end_dt and reg_end_dt < self.reference_now:
                reasons.append({"code": "registration_closed", "label": "Hết hạn đăng ký"})

            # Budget check
            if max_budget is not None and ev.fee_vnd is not None and ev.fee_vnd > max_budget:
                reasons.append(
                    {
                        "code": "exceeds_budget",
                        "label": f"Phí {ev.fee_vnd:,}đ vượt ngân sách {max_budget:,}đ",
                    }
                )

            # Module 4 schedule overlap check
            has_conflict, conflict_msg = self._check_timetable_conflict(ev)
            if has_conflict and hide_conflicts:
                reasons.append({"code": "schedule_conflict", "label": conflict_msg or "Trùng lịch TKB (M4)"})

            # Module 6 group cap check
            if hide_capped and credit.capped_by in ("group", "total"):
                reasons.append(
                    {
                        "code": "group_cap_reached",
                        "label": f"Nhóm {credit.group_id} đã chạm trần điểm (M6)",
                    }
                )

            if reasons:
                dropped.append(
                    {
                        "event_id": ev.event_id,
                        "title": ev.title,
                        "start_datetime": ev.start_datetime,
                        "rubric_item_id": effective_item_id,
                        "parent_group_id": effective_group_id,
                        "expected_points": credit.expected_points,
                        "drop_reasons": reasons,
                    }
                )
                continue

            raw_interest = cosine_similarity(user_vec, prof.vector)
            fit_score = self._compute_fit_score(ev, has_conflict)
            urgency_score, deadline_alert = self._compute_urgency_score(ev)
            # Gain normalized by max 15 pts + small boost if group still has target gap
            gap_boost = 1.0 if shortfall.target_gap_by_group.get(effective_group_id or "", 0.0) > 0 else 0.65
            gain_norm = min(1.0, (credit.effective_points / 15.0) * gap_boost)
            penalty = -0.15 if has_conflict else 0.0

            eligible.append(
                {
                    "profile": prof,
                    "event": ev,
                    "rubric_item_id": effective_item_id,
                    "parent_group_id": effective_group_id,
                    "credit": credit,
                    "raw_interest": raw_interest,
                    "gain": gain_norm,
                    "fit": fit_score,
                    "urgency": urgency_score,
                    "penalty": penalty,
                    "deadline_alert": deadline_alert,
                    "has_conflict": has_conflict,
                    "conflict_msg": conflict_msg,
                    "is_saved": ev.event_id in saved_set,
                }
            )

        # Percentile-rank normalize Interest across eligible pool (per roadmap Phase 5)
        if eligible:
            sorted_ints = sorted(item["raw_interest"] for item in eligible)
            denom = max(1, len(sorted_ints) - 1)
            for item in eligible:
                rank_idx = sorted_ints.index(item["raw_interest"])
                item["interest"] = rank_idx / denom if denom > 0 else 0.5
                item["score_total"] = (
                    weights.w1_gain * item["gain"]
                    + weights.w2_interest * item["interest"]
                    + weights.w3_fit * item["fit"]
                    + weights.w4_urgency * item["urgency"]
                    + item["penalty"]
                )

        # P7 Selection: MMR for (top_n - 1) items + 1 Exploration Slot from under-represented group
        selected: List[Dict[str, Any]] = []
        remaining = list(eligible)
        mmr_lambda = 0.78
        main_slots = max(0, top_n - 1) if len(eligible) >= top_n else len(eligible)

        while remaining and len(selected) < main_slots:
            best_item = None
            best_mmr = -1e9
            for cand in remaining:
                if not selected:
                    mmr = cand["score_total"]
                else:
                    max_sim = max(
                        cosine_similarity(cand["profile"].vector, s["profile"].vector)
                        for s in selected
                    )
                    mmr = mmr_lambda * cand["score_total"] - (1.0 - mmr_lambda) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_item = cand
            assert best_item is not None
            best_item["is_exploration_slot"] = False
            selected.append(best_item)
            remaining.remove(best_item)

        # Reserve 1 exploration slot (prefer a parent group not dominant in selected, high Gain + Fit)
        if remaining and len(selected) < top_n:
            selected_groups = {s["parent_group_id"] for s in selected}
            novel = [c for c in remaining if c["parent_group_id"] not in selected_groups]
            pool = novel if novel else remaining
            explore_pick = max(
                pool,
                key=lambda c: (0.5 * c["gain"] + 0.5 * c["fit"] - 0.1 * c["interest"]),
            )
            explore_pick["is_exploration_slot"] = True
            selected.append(explore_pick)

        # P8 Explain: Grounded Vietnamese breakdown reasons
        cards: List[Dict[str, Any]] = []
        for item in selected:
            ev: EventNormalized = item["event"]
            credit = item["credit"]
            cls_res = item["profile"].classification
            grp = item["parent_group_id"] or "N/A"
            iid = item["rubric_item_id"] or "N/A"
            reasons_vi: List[str] = []

            if credit.effective_points > 0:
                gap = shortfall.target_gap_by_group.get(grp, 0.0)
                if gap > 0:
                    reasons_vi.append(
                        f"Cộng dự kiến +{credit.effective_points:g}đ vào nhóm {grp} (đang thiếu {gap:g}đ so với mục tiêu)."
                    )
                else:
                    reasons_vi.append(
                        f"Cộng dự kiến +{credit.effective_points:g}đ nhóm {grp} (mục {iid})."
                    )
            if item["interest"] >= 0.65:
                reasons_vi.append(
                    f"Độ khớp sở thích cao ({int(round(item['interest'] * 100))}%) theo hồ sơ vector P4."
                )
            if not item["has_conflict"]:
                mode_vi = (
                    "trực tuyến (linh hoạt)"
                    if ev.online_mode == "online"
                    else ("kết hợp online/offline" if ev.online_mode == "hybrid" else "không trùng lịch học TKB")
                )
                reasons_vi.append(f"Thuận tiện tham gia: {mode_vi}.")
            if item["is_exploration_slot"]:
                reasons_vi.append(
                    f"Slot khám phá (P7): mở rộng trải nghiệm sang nhóm {grp} ngoài vùng sở thích quen thuộc."
                )

            dual_cls = self.dual_classification_by_post_id.get(ev.event_id)
            dual_status = dual_cls.get("status") if dual_cls else None
            dual_conf = (dual_cls.get("confidence") or {}) if dual_cls else {}
            dual_reasons = (dual_cls.get("review_reasons") or []) if dual_cls else []

            cards.append(
                {
                    "event_id": ev.event_id,
                    "title": ev.title,
                    "description": ev.description,
                    "organizer": ev.organizer,
                    "start_datetime": ev.start_datetime,
                    "end_datetime": ev.end_datetime,
                    "reg_end_datetime": ev.reg_end_datetime,
                    "online_mode": ev.online_mode,
                    "location": ev.location,
                    "fee_vnd": ev.fee_vnd,
                    "fee_raw": ev.fee_raw,
                    "registration_url": ev.registration_url,
                    "rubric_item_id": iid,
                    "rubric_item_name": self.rubric_name_by_id.get(iid, ""),
                    "parent_group_id": grp,
                    "is_label_overridden": ev.event_id in self.student_profile["label_overrides"],
                    "classification_status": cls_res.status,
                    "dual_branch_status": dual_status,
                    "dual_branch_agreement": dual_conf.get("branch_agreement"),
                    "dual_branch_verification": dual_conf.get("evidence_verification_status"),
                    "dual_branch_review_reasons": dual_reasons,
                    "selected_candidate_groups": cls_res.selected_candidate_groups,
                    "expected_points": credit.expected_points,
                    "effective_points": credit.effective_points,
                    "is_credit_eligible": credit.is_credit_eligible,
                    "score_total": round(item["score_total"], 4),
                    "score_breakdown": {
                        "gain": round(item["gain"], 3),
                        "interest": round(item["interest"], 3),
                        "fit": round(item["fit"], 3),
                        "urgency": round(item["urgency"], 3),
                        "penalty": round(item["penalty"], 3),
                    },
                    "top_reasons": reasons_vi[:3],
                    "deadline_alert": item["deadline_alert"],
                    "is_exploration_slot": item["is_exploration_slot"],
                    "is_saved": item["is_saved"],
                    "has_conflict": item["has_conflict"],
                    "conflict_msg": item["conflict_msg"],
                    "missing_field_flags": ev.missing_field_flags,
                }
            )

        # Build Module 5 high-urgency reminders
        reminders = [
            {
                "event_id": c["event_id"],
                "title": c["title"],
                "deadline_alert": c["deadline_alert"],
                "is_saved": c["is_saved"],
                "rubric_item_id": c["rubric_item_id"],
                "effective_points": c["effective_points"],
            }
            for c in cards
            if c["deadline_alert"] or c["is_saved"]
        ][:5]

        return {
            "reference_now": self.reference_now_str,
            "weights_applied": weights.to_dict(),
            "ledger_summary": {
                "capped_by_group": ledger_eval.capped_by_group,
                "group_caps": self.credit_engine.group_caps,
                "remaining_cap_by_group": ledger_eval.remaining_cap_by_group,
                "target_gap_by_group": shortfall.target_gap_by_group,
                "total_score": ledger_eval.total_score,
                "target_total": self.ledger.target_total,
                "label_vi": "Ước tính — chưa xác nhận",
            },
            "stats": {
                "total_cached_events": len(self.cache),
                "eligible_count": len(eligible),
                "dropped_count": len(dropped),
                "returned_count": len(cards),
            },
            "recommendations": cards,
            "dropped_events": dropped[:20],
            "reminders": reminders,
        }

    def generate_schedule_plan(self, target_total: Optional[float] = None) -> Dict[str, Any]:
        """P7 Schedule Planner prototype: conflict-free, cap-respecting subset hitting DRL target."""
        rec_data = self.get_recommendations(top_n=40)
        candidates = rec_data["recommendations"]
        eval_now = self.credit_engine.evaluate_ledger(self.ledger)

        goal = float(target_total if target_total is not None else self.ledger.target_total)
        goal = min(100.0, max(eval_now.total_score, goal))

        rem_caps = dict(eval_now.remaining_cap_by_group)
        projected_total = eval_now.total_score
        planned: List[Dict[str, Any]] = []
        chosen_intervals: List[Tuple[datetime, datetime]] = []

        # Prioritize high effective_points + high score_total while enforcing pairwise non-overlap & caps
        ordered = sorted(
            candidates,
            key=lambda c: (
                -min(c["effective_points"], rem_caps.get(c["parent_group_id"], 0.0)),
                -c["score_total"],
            ),
        )

        for cand in ordered:
            if projected_total >= goal:
                break
            grp = cand["parent_group_id"]
            rem_g = rem_caps.get(grp, 0.0)
            gain = min(float(cand["expected_points"]), rem_g, 100.0 - projected_total)
            if gain <= 0:
                continue
            if cand["has_conflict"]:
                continue

            # Pairwise event time non-overlap check
            st = parse_iso_datetime(cand["start_datetime"]) if cand["start_datetime"] else None
            en = parse_iso_datetime(cand["end_datetime"]) if cand["end_datetime"] else None
            if st is not None:
                if en is None or en <= st:
                    en = st + timedelta(hours=2)
                overlaps = any(
                    st < (prev_en + timedelta(minutes=30)) and en > (prev_st - timedelta(minutes=30))
                    for prev_st, prev_en in chosen_intervals
                )
                if overlaps:
                    continue
                chosen_intervals.append((st, en))

            rem_caps[grp] = max(0.0, rem_g - gain)
            projected_total += gain
            planned.append({**cand, "planned_gain": gain})

        projected_by_group = {
            grp: eval_now.capped_by_group[grp] + (eval_now.remaining_cap_by_group[grp] - rem_caps[grp])
            for grp in eval_now.capped_by_group
        }

        return {
            "solver_mode": "prototype_cap_aware_greedy",
            "solver_note": "Prototype preview — sẽ nâng cấp lên OR-Tools CP-SAT ở Chặng 3 (Phase 6).",
            "current_total": eval_now.total_score,
            "target_total": goal,
            "projected_total": round(projected_total, 2),
            "target_reached": projected_total >= goal,
            "current_by_group": eval_now.capped_by_group,
            "projected_by_group": projected_by_group,
            "group_caps": self.credit_engine.group_caps,
            "planned_events": planned,
            "weekly_timetable": MOCK_WEEKLY_TIMETABLE,
        }

    def inspect_classify(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Live Dual-Branch Hierarchical 2-Stage kNN + Stage-wise Fusion + Module 6 trace."""
        title = str(payload.get("title") or "").strip()
        description = str(payload.get("description") or "").strip()
        organizer = str(payload.get("organizer") or "").strip()
        role = str(payload.get("role") or "participant").strip()
        post_id = str(payload.get("post_id") or "").strip()

        stage1_top_k = max(1, min(4, int(payload.get("stage1_top_k") or self.dual_default_params.get("candidate_group_count", 2))))
        k1 = max(1, min(30, int(payload.get("k1") or payload.get("stage1_neighbors") or self.dual_default_params.get("k1", 10))))
        k2 = max(1, min(30, int(payload.get("k2") or payload.get("stage2_neighbors") or self.dual_default_params.get("k2", 10))))
        beta = max(0.0, min(1.0, float(payload.get("beta") if payload.get("beta") not in (None, "") else self.dual_default_params.get("stage1_fusion_beta", 0.5))))
        alpha = max(0.0, min(1.0, float(payload.get("alpha") if payload.get("alpha") not in (None, "") else self.dual_default_params.get("stage2_fusion_alpha", 0.5))))

        raw_event = {
            "is_event": True,
            "post_id": post_id or "live_inspect",
            "title": title or "Sự kiện thử nghiệm",
            "description": description,
            "organizer": organizer,
        }
        query_text = build_event_query_text(raw_event, query_prefix="query: ")
        query_body = query_text[len("query: "):] if query_text.startswith("query: ") else query_text

        t0 = time.perf_counter()
        # Reuse precomputed shared query vector if user clicked one of the 108 events and didn't edit text
        reused_query_vec = False
        if (
            post_id
            and post_id in self.dual_query_vectors_by_post_id
            and self.embedder_status.is_e5_active
            and payload.get("use_cached_query") is True
        ):
            vec = self.dual_query_vectors_by_post_id[post_id]
            reused_query_vec = True
        else:
            vec = l2_normalize(self.embedder.encode_query(query_body))

        # Run real dual-branch HierarchicalTwoStageKNN resolved from AlgorithmRegistry
        runtime_params = dict(self.dual_default_params)
        runtime_params["candidate_group_count"] = stage1_top_k
        runtime_params["k1"] = k1
        runtime_params["k2"] = k2
        runtime_params["stage1_fusion_beta"] = beta
        runtime_params["stage2_fusion_alpha"] = alpha

        algorithm = self.dual_registry.create(
            self.dual_algorithm_name,
            sample_retriever=self.dual_sample_retriever,
            rubric_retriever=self.dual_rubric_retriever,
            rubric_data=self.dual_assets.rubric,
            params=runtime_params,
        )
        shared_q_np = np.asarray(vec, dtype=np.float32)
        prediction = algorithm.classify(shared_q_np, event_context=raw_event)
        dual_cls = self.dual_decision_handler.decide(prediction=prediction, event=raw_event)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Adapt to legacy classification block so existing callers & tests remain 100% compatible
        cls_result = self._dual_to_legacy_classification_result(raw_event["post_id"], dual_cls)

        ev = EventNormalized(
            event_id=raw_event["post_id"],
            title=raw_event["title"],
            description=description,
            organizer=organizer,
        )
        credit_est = self.credit_engine.estimate_event_credit(
            event=ev,
            rubric_item_id=cls_result.rubric_item_id,
            ledger=self.ledger,
            role=role,
        )

        # Count filtered vectors in Stage 2 for both branches
        selected_groups = cls_result.selected_candidate_groups
        sample_stage2_pool = int(
            self.dual_assets.sample_branch.metadata["group_id"].isin(selected_groups).sum()
        )
        rm = self.dual_assets.rubric_branch.metadata
        rubric_stage2_pool = int(
            ((rm["stage"] == "criterion") & (rm["group_id"].isin(selected_groups)) & (rm["event_matchable"])).sum()
        )

        return {
            "embedder_status": asdict(self.embedder_status),
            "formatted_query_text": query_text,
            "reused_cached_query": reused_query_vec,
            "vector_dimension": len(vec),
            "vector_l2_norm": round(l2_norm(vec), 6),
            "elapsed_ms": round(elapsed_ms, 2),
            "knn_params_used": {
                "algorithm": self.dual_algorithm_name,
                "parameter_status": "provisional_not_evaluated",
                "stage1_top_k": stage1_top_k,
                "k1": k1,
                "k2": k2,
                "stage1_neighbors": k1,
                "stage2_neighbors": k2,
                "beta": beta,
                "alpha": alpha,
                "note": "Tham số tạm thời (provisional_not_evaluated); chưa đánh giá hay tinh chỉnh.",
            },
            "cls_index_summary": self.cls_index.summary(),
            "reference_pool_sizes": {
                "sample_stage1_total": self.dual_sample_retriever.eligible_count,
                "sample_stage2_filtered": sample_stage2_pool,
                "rubric_stage1_total": len(self.dual_rubric_retriever.eligible_stage1_groups),
                "rubric_stage2_filtered": rubric_stage2_pool,
            },
            "stage2_union_pool_size": sample_stage2_pool + rubric_stage2_pool,
            "dual_branch": dual_cls,
            "classification": {
                **cls_result.to_dict(),
                "rubric_item_name": self.rubric_name_by_id.get(
                    cls_result.rubric_item_id or "",
                    "",
                ),
                "top_k_items_named": [
                    {
                        **row,
                        "rubric_item_name": self.rubric_name_by_id.get(row["rubric_item_id"], ""),
                    }
                    for row in cls_result.top_k_items
                ],
            },
            "module6_credit_estimate": credit_est.to_dict(),
        }

    def update_profile(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Update P4 profile, weights w1..w4, cold-start chips, or filter toggles."""
        for key in (
            "cohort",
            "faculty",
            "class_id",
            "selected_chips",
            "max_budget_vnd",
            "exam_week_mode",
            "hide_schedule_conflicts",
            "hide_group_cap_reached",
        ):
            if key in payload:
                self.student_profile[key] = payload[key]

        if "target_total" in payload:
            t = float(payload["target_total"])
            self.student_profile["target_total"] = t
            self.ledger.target_total = t

        if "preset" in payload and payload["preset"] in WEIGHT_PRESETS:
            preset = payload["preset"]
            self.student_profile["preset"] = preset
            self.student_profile["weights"] = dict(WEIGHT_PRESETS[preset]["weights"])
        elif "weights" in payload and isinstance(payload["weights"], dict):
            self.student_profile["preset"] = "custom"
            self.student_profile["weights"] = dict(payload["weights"])

        return {"profile": self.student_profile, "recommendations": self.get_recommendations()}

    def record_feedback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle UI actions (save, dismiss, joined, label_correction, reset) into Feedback Store."""
        action = str(payload.get("action") or "").strip()
        event_id = str(payload.get("event_id") or "").strip()
        ts = datetime.now().isoformat(timespec="seconds")

        if action == "reset_demo":
            self.student_profile["saved_event_ids"] = []
            self.student_profile["dismissed_event_ids"] = []
            self.student_profile["joined_event_ids"] = []
            self.student_profile["label_overrides"] = {}
            self.ledger.entries = [LedgerEntry(rubric_item_id="1.9", points=5.0, event_id="past_demo_1")]
            self.feedback_log.clear()
            return {"status": "reset", "feedback_log": self.feedback_log, "recommendations": self.get_recommendations()}

        prof = self.cache.get(event_id)
        if prof is None:
            raise ValueError(f"Unknown event_id '{event_id}'")

        entry: Dict[str, Any] = {
            "timestamp": ts,
            "action": action,
            "event_id": event_id,
            "event_title": prof.event.title,
        }

        if action == "save":
            saved = self.student_profile["saved_event_ids"]
            if event_id in saved:
                saved.remove(event_id)
                entry["detail"] = "Bỏ lưu sự kiện"
            else:
                saved.append(event_id)
                entry["detail"] = "Đã lưu -> cập nhật vector sở thích P4 & nhắc hạn M5"
        elif action == "dismiss":
            if event_id not in self.student_profile["dismissed_event_ids"]:
                self.student_profile["dismissed_event_ids"].append(event_id)
            entry["detail"] = "Đã ẩn -> giảm trọng số sở thích P4 & lọc ở P5"
        elif action == "joined":
            if event_id not in self.student_profile["joined_event_ids"]:
                self.student_profile["joined_event_ids"].append(event_id)
                eff_id = self.student_profile["label_overrides"].get(
                    event_id,
                    prof.classification.rubric_item_id,
                )
                credit = self.credit_engine.estimate_event_credit(
                    event=prof.event,
                    rubric_item_id=eff_id,
                    ledger=self.ledger,
                )
                if eff_id and credit.expected_points > 0:
                    self.ledger.entries.append(
                        LedgerEntry(
                            rubric_item_id=eff_id,
                            points=credit.expected_points,
                            event_id=event_id,
                            source="system_detected",
                        )
                    )
                entry["detail"] = (
                    f"Xác nhận tham gia (M7) -> cộng +{credit.effective_points:g}đ vào nhóm {credit.group_id} (M6)"
                )
        elif action == "label_correction":
            corrected_item = str(payload.get("corrected_item_id") or "").strip()
            if not is_valid_item_id(corrected_item):
                raise ValueError(f"Invalid canonical rubric item_id '{corrected_item}'")
            corrected_group = parent_group_from_item_id(corrected_item)
            self.student_profile["label_overrides"][event_id] = corrected_item

            # LIVE Feedback Loop -> ClsIndex.add_sample()!
            self._user_correction_seq += 1
            new_sample = ClsSample(
                sample_id=f"user_corr_{self._user_correction_seq:04d}",
                text=prof.event.to_classification_text(),
                title=prof.event.title,
                rubric_item_id=corrected_item,
                parent_group_id=corrected_group,
                organizer=prof.event.organizer,
                source="user_corrected",
            )
            self.cls_index.add_sample(new_sample, prof.vector)
            entry["corrected_item_id"] = corrected_item
            entry["corrected_group_id"] = corrected_group
            entry["detail"] = (
                f"Sửa nhãn -> {corrected_item} ({corrected_group}) & bổ sung vector mẫu vào ClsIndex (tổng {self.cls_index.size} mẫu)"
            )
        else:
            raise ValueError(f"Unsupported feedback action '{action}'")

        self.feedback_log.insert(0, entry)
        return {
            "entry": entry,
            "cls_index_summary": self.cls_index.summary(),
            "feedback_log": self.feedback_log[:30],
            "recommendations": self.get_recommendations(),
        }

    def bootstrap_payload(self) -> Dict[str, Any]:
        """Initial payload powering all 4 tabs of the prototype UI."""
        return {
            "embedder_status": asdict(self.embedder_status),
            "cls_index_summary": self.cls_index.summary(),
            "dual_branch_summary": self._dual_branch_summary(),
            "dual_branch_events": self.get_dual_branch_events(),
            "interest_chips": INTEREST_CHIPS,
            "weight_presets": WEIGHT_PRESETS,
            "weekly_timetable": MOCK_WEEKLY_TIMETABLE,
            "rubric_catalog": self.rubric_catalog,
            "profile": self.student_profile,
            "feedback_log": self.feedback_log,
            "recommendations_bundle": self.get_recommendations(),
            "schedule_plan": self.generate_schedule_plan(),
        }


def make_handler(service: PrototypeService):
    """Create HTTP request handler bound to a PrototypeService instance."""

    class PrototypeHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Clean concise server log
            print(f"[prototype] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

        def _send_json(self, data: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, file_path: Path, content_type: str) -> None:
            if not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            data = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index.html"):
                self._send_file(PROTOTYPE_DIR / "index.html", "text/html; charset=utf-8")
            elif path == "/styles.css":
                self._send_file(PROTOTYPE_DIR / "styles.css", "text/css; charset=utf-8")
            elif path == "/app.js":
                self._send_file(PROTOTYPE_DIR / "app.js", "application/javascript; charset=utf-8")
            elif path == "/api/bootstrap":
                self._send_json(service.bootstrap_payload())
            elif path == "/api/recommendations":
                self._send_json(service.get_recommendations())
            elif path == "/api/plan":
                self._send_json(service.generate_schedule_plan())
            elif path == "/api/dual-branch/events":
                self._send_json(service.get_dual_branch_events())
            else:
                self.send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {path}")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                body = self._read_json_body()
                if path == "/api/classify":
                    self._send_json(service.inspect_classify(body))
                elif path == "/api/dual-branch/run":
                    self._send_json(service.run_dual_branch_pipeline())
                elif path == "/api/profile":
                    self._send_json(service.update_profile(body))
                elif path == "/api/plan":
                    self._send_json(
                        service.generate_schedule_plan(target_total=body.get("target_total"))
                    )
                elif path == "/api/feedback":
                    self._send_json(service.record_feedback(body))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {path}")
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    return PrototypeHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run No-Le-DRL Module 3 Interactive Prototype.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--embedder",
        choices=("e5", "demo"),
        default="e5",
        help="P2.1 embedder mode ('e5' uses intfloat/multilingual-e5-small with graceful fallback).",
    )
    parser.add_argument(
        "--require-e5",
        action="store_true",
        help="Fail immediately if sentence-transformers is not installed.",
    )
    args = parser.parse_args()

    print("Đang khởi tạo No-Le-DRL Module 3 Prototype...")
    service = PrototypeService(
        embedder_mode=args.embedder,
        allow_fallback=not args.require_e5,
    )
    st = service.embedder_status
    db = service._dual_branch_summary()
    print(f"  • P2.1 Embedder : {st.active_model} (e5_active={st.is_e5_active}, dim={st.dimension})")
    if st.fallback_reason:
        print(f"  • Ghi chú       : {st.fallback_reason}")
    print(
        f"  • Dual-Branch   : {db['algorithm']} "
        f"(Sample={db['sample_reference_count']} mẫu, Rubric={db['rubric_total_prototypes']} prototypes, "
        f"β={db['default_params']['beta']}, α={db['default_params']['alpha']}, provisional)"
    )
    print(f"  • Batch Events  : {db['batch_events_count']} sự kiện -> {db['status_counts']}")
    print(f"  • Event Cache   : {len(service.cache)} sự kiện HCMUS đã nạp")
    print(f"  • URL           : http://{args.host}:{args.port}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server prototype.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
