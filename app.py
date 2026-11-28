"""Interactive Web Server & Service for No-Le-DRL (DB-Backed Classification, Demo M6, Scoring & MMR).

Exposes strictly the real, implemented backend modules backed by `drl_database.db`:
- Embedding: Unified `intfloat/multilingual-e5-small` (`384d` L2-normalized) via `src.embedder`.
- Classification: Dual-Branch Hierarchical 2-Stage kNN + Stage-wise Fusion (`src.classification`),
  computing 3 query vectors (`title`, `title_description`, `title_description_format`) from `drl_database.db`.
- Credit Estimation: Isolated deterministic Demo Module 6 (`DemoCreditProvider` wrapping
  `src.credit_calculation.RubricCreditEngine` driven by `data/raw/rubric.yaml`; does NOT use `credit_calculator.py`).
- Recommendation: 4-field `passage:` event embeddings (`title + description + topic + format`)
  from `drl_database.db`, P5 Hard Filter, P6 Multi-Criteria Scoring, and separate P7 MMR Selector.

Usage:
    python3 app.py                 # Starts on http://127.0.0.1:8765
    python3 app.py --port 8080     # Custom port
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np

from configs.model_config import MODEL_DEVICE, MODEL_ID, PROJECT_ROOT
from src.credit_calculation import AttributeExtractor, RubricCreditEngine
from src.classification.algorithms.algorithm_registry import build_default_registry
from src.classification.data.loaders import load_all_assets
from src.classification.data.validators import validate_loaded_assets
from src.classification.decision.confidence_handler import ConfidenceAndDecisionHandler
from src.classification.embeddings.event_text_builder import (
    CLASSIFICATION_VECTOR_PROFILES,
    build_all_classification_profile_texts,
    normalize_string_list_field,
)
from src.classification.pipeline.classification_pipeline import ClassificationPipeline
from src.classification.retrieval.rubric_retriever import RubricRetriever
from src.classification.retrieval.sample_retriever import SampleRetriever
from src.db_event_adapter import fetch_events_from_db
from src.embedder import create_embedder
from src.recommendation import (
    REC_EMBEDDING_FIELDS,
    DemoCreditProvider,
    DemoScheduleProvider,
    MMRConfig,
    available_demo_personas,
    load_recommendation_candidate_pool_from_db,
    load_recommendation_config,
    parse_reference_now,
    resolve_effective_deadline,
    run_recommendation_demo,
)
from src.recommendation.pipeline import DEFAULT_REC_EMB_PATH, DEFAULT_REC_META_PATH
from src.schemas import EventNormalized
from src.vector_ops import l2_norm, l2_normalize

PROTOTYPE_DIR = PROJECT_ROOT / "prototype"
CLASSIFICATION_CONFIG_PATH = PROJECT_ROOT / "configs" / "classification_config.yaml"


class PrototypeService:
    """Backend service backing the No-Le-DRL DB-backed Classification, Scoring & MMR UI."""

    def __init__(self) -> None:
        # 1. Unified P2.1 Embedder (intfloat/multilingual-e5-small, 384d L2-normalized)
        self.embedder, self.embedder_status = create_embedder(
            model_id=MODEL_ID,
            device=MODEL_DEVICE,
        )

        # 2. Dual-Branch Hierarchical 2-Stage kNN System (src/classification) — DB-backed
        self.dual_branch_config_path = CLASSIFICATION_CONFIG_PATH
        self.dual_assets = load_all_assets(
            self.dual_branch_config_path,
            project_root=PROJECT_ROOT,
            input_source_override="database",
        )
        validate_loaded_assets(self.dual_assets)
        paths_cfg = self.dual_assets.config.get("paths") or {}
        raw_db = Path(str(paths_cfg.get("db_path", "drl_database.db")))
        self.db_path = raw_db if raw_db.is_absolute() else PROJECT_ROOT / raw_db

        self.dual_sample_retriever = SampleRetriever(self.dual_assets.sample_branch)
        self.dual_rubric_retriever = RubricRetriever(
            self.dual_assets.rubric_branch,
            self.dual_assets.rubric,
        )
        self.dual_registry = build_default_registry()
        alg_cfg = self.dual_assets.config.get("algorithm") or {}
        self.dual_algorithm_name = str(alg_cfg.get("name", "hierarchical_two_stage_knn"))
        self.dual_default_params = dict(alg_cfg.get("params") or {})
        self.dual_decision_cfg = dict(self.dual_assets.config.get("decision") or {})

        self.dual_decision_handler = ConfidenceAndDecisionHandler(
            rubric_retriever=self.dual_rubric_retriever,
            decision_config=self.dual_decision_cfg,
            max_suggested_candidates=int(self.dual_default_params.get("max_suggested_candidates", 5)),
        )

        # 3. Isolated Demo Module 6 Shared Credit Engine (driven by data/raw/rubric.yaml)
        self.attribute_extractor = AttributeExtractor()
        self.credit_engine = RubricCreditEngine(attribute_extractor=self.attribute_extractor)
        self.rubric_name_by_id: Dict[str, str] = {
            str(spec["ref"]): str(spec["name"])
            for spec in self.credit_engine.rubric["items"].values()
        }

        # 4. Recommendation Subsystem (P4 Demo + Module 4 ScheduleEngine + Demo M6 + P5 + P6 Scoring + P7 MMR)
        self.rec_config = load_recommendation_config()
        pen_cfg = (self.rec_config.get("scoring") or {}).get("penalty_mandatory") or {}
        self.demo_credit_provider = DemoCreditProvider(
            credit_engine=self.credit_engine,
            enable_item_2_2_penalty_link=bool(pen_cfg.get("enable_item_2_2_penalty_link", False)),
        )
        self.rec_chip_vec_cache: Dict[str, List[float]] = {}

        # 5. Load & embed ALL active events strictly from `drl_database.db` (no events.json)
        self.dual_batch_events: List[Dict[str, Any]] = []
        self.dual_classification_by_post_id: Dict[str, Dict[str, Any]] = {}
        self.dual_classification_by_event_id: Dict[int, Dict[str, Any]] = {}
        self.dual_query_vectors_by_post_id: Dict[str, List[float]] = {}
        self.dual_multi_vectors_by_post_id: Dict[str, Dict[str, List[float]]] = {}
        self.dual_last_vector_routing: Dict[str, str] = {}
        self.dual_embeddings_reused: bool = False
        self.rec_candidates: List[Any] = []

        self._load_dual_branch_batch_from_db()

    def _reload_recommendation_pool_from_db(self) -> None:
        """Build Recommendation candidate pool strictly from `drl_database.db` using 4-field passage embeddings."""
        self.rec_candidates = load_recommendation_candidate_pool_from_db(
            db_path=self.db_path,
            embedder=self.embedder,
            classification_by_event_id=self.dual_classification_by_event_id,
            exclude_cancelled=True,
            rec_embeddings_path=DEFAULT_REC_EMB_PATH,
            rec_metadata_path=DEFAULT_REC_META_PATH,
            reuse_existing_embeddings=True,
        )

    def _load_dual_branch_batch_from_db(
        self,
        *,
        branch_vector_selection: Optional[Dict[str, Optional[str]]] = None,
        only_unclassified: bool = False,
        write_back_db: bool = True,
        write_output_files: bool = True,
    ) -> Dict[str, Any]:
        """Run 3-vector DB Classification pipeline and sync both Tab 2 and Recommendation pool."""
        pipeline = ClassificationPipeline(self.dual_branch_config_path, project_root=PROJECT_ROOT)
        result = pipeline.run_from_database(
            db_path=self.db_path,
            only_unclassified=only_unclassified,
            exclude_cancelled=True,
            branch_vector_selection=branch_vector_selection,
            write_back_db=write_back_db,
            write_output_files=write_output_files,
            embedder=self.embedder,
        )

        self.dual_batch_events = list(result.get("events") or [])
        self.dual_classification_by_post_id = {}
        self.dual_classification_by_event_id = {}
        self.dual_query_vectors_by_post_id = {}
        self.dual_multi_vectors_by_post_id = {}
        self.dual_last_vector_routing = dict(result.get("branch_vector_selection") or {})
        self.dual_embeddings_reused = bool(result.get("query_embeddings_reused", False))

        for rec in self.dual_batch_events:
            pid = str(rec.get("post_id") or "").strip()
            cls_obj = rec.get("classification")
            if pid and isinstance(cls_obj, dict):
                self.dual_classification_by_post_id[pid] = cls_obj
            if rec.get("event_id") is not None and isinstance(cls_obj, dict):
                self.dual_classification_by_event_id[int(rec["event_id"])] = cls_obj

        multi_set = result.get("multi_profile_embeddings")
        if multi_set is not None:
            for row_meta in (multi_set.metadata or {}).get("records") or []:
                pid = str(row_meta.get("post_id") or "").strip()
                r_idx = int(row_meta["embedding_row"])
                if not pid:
                    continue
                prof_map: Dict[str, List[float]] = {}
                for prof, mat in multi_set.vectors_by_profile.items():
                    if 0 <= r_idx < mat.shape[0]:
                        prof_map[prof] = [float(v) for v in mat[r_idx].tolist()]
                if prof_map:
                    self.dual_multi_vectors_by_post_id[pid] = prof_map
                    if "title_description" in prof_map:
                        self.dual_query_vectors_by_post_id[pid] = prof_map["title_description"]

        self._reload_recommendation_pool_from_db()
        return result

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
        emb_cfg = self.dual_assets.config.get("embedding") or {}
        bvs = emb_cfg.get("branch_vector_selection") or {}
        active_sample = self.dual_last_vector_routing.get(
            "sample_branch", str(bvs.get("sample_branch", "title_description"))
        )
        active_rubric = self.dual_last_vector_routing.get(
            "rubric_branch", str(bvs.get("rubric_branch", "title_description"))
        )
        return {
            "enabled": True,
            "source_mode": "database",
            "db_path": self.db_path.name,
            "algorithm": self.dual_algorithm_name,
            "registered_algorithms": self.dual_registry.list_algorithms(),
            "parameter_status": "provisional_not_evaluated",
            "sample_reference_count": self.dual_sample_retriever.eligible_count,
            "rubric_stage1_count": len(self.dual_rubric_retriever.eligible_stage1_groups),
            "rubric_total_prototypes": int(len(self.dual_assets.rubric_branch.metadata)),
            "batch_events_count": len(self.dual_batch_events),
            "status_counts": status_counts,
            "classification_vector_profiles": list(CLASSIFICATION_VECTOR_PROFILES),
            "vector_routing_active": dict(self.dual_last_vector_routing),
            "default_params": {
                "stage1_top_k": int(self.dual_default_params.get("candidate_group_count", 2)),
                "k1": int(self.dual_default_params.get("k1", 10)),
                "k2": int(self.dual_default_params.get("k2", 10)),
                "beta": float(self.dual_default_params.get("stage1_fusion_beta", 0.5)),
                "alpha": float(self.dual_default_params.get("stage2_fusion_alpha", 0.5)),
                "sample_branch_vector": str(active_sample),
                "rubric_branch_vector": str(active_rubric),
            },
        }

    def get_dual_branch_events(self) -> Dict[str, Any]:
        """Return all DB-sourced batch-classified events for UI inspection."""
        items = []
        for rec in self.dual_batch_events:
            cls = rec.get("classification") or {}
            top_sugg = (cls.get("suggested_candidates") or [None])[0]
            date_val = rec.get("date")
            if not date_val and isinstance(rec.get("start_datetime"), str):
                date_val = rec["start_datetime"][:10]
            items.append(
                {
                    "event_id": rec.get("event_id"),
                    "post_id": rec.get("post_id"),
                    "title": rec.get("title") or "",
                    "description": rec.get("description") or "",
                    "topic": rec.get("topic") or [],
                    "format": rec.get("format") or [],
                    "organizer": rec.get("organizer") or "",
                    "date": date_val,
                    "status": cls.get("status"),
                    "predicted_group_id": cls.get("predicted_group_id"),
                    "predicted_criterion_ref": cls.get("predicted_criterion_ref"),
                    "predicted_criterion_name": cls.get("predicted_criterion_name"),
                    "top_suggested_ref": top_sugg.get("criterion_ref") if top_sugg else None,
                    "top_suggested_group": top_sugg.get("group_id") if top_sugg else None,
                    "top_suggested_score": top_sugg.get("fused_score") if top_sugg else None,
                    "branch_agreement": (cls.get("confidence") or {}).get("branch_agreement"),
                    "review_reasons": cls.get("review_reasons") or [],
                    "vector_routing_used": cls.get("vector_routing_used"),
                    "classification": cls,
                }
            )
        return {
            "summary": self._dual_branch_summary(),
            "events": items,
        }

    def run_dual_branch_pipeline(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Re-run 3-vector ClassificationPipeline directly on `drl_database.db` and refresh Rec pool."""
        req = dict(payload or {})
        bvs: Dict[str, Optional[str]] = {}
        for k in (
            "sample_branch",
            "rubric_branch",
            "sample_stage1",
            "sample_stage2",
            "rubric_stage1",
            "rubric_stage2",
        ):
            if req.get(k) not in (None, ""):
                bvs[k] = str(req[k]).strip()

        result = self._load_dual_branch_batch_from_db(
            branch_vector_selection=bvs or None,
            only_unclassified=bool(req.get("only_unclassified", False)),
            write_back_db=bool(req.get("write_back_db", True)),
            write_output_files=bool(req.get("write_output_files", True)),
        )
        return {
            "status": "ok",
            "source_mode": result["source_mode"],
            "db_path": result["db_path"],
            "algorithm": result["algorithm"],
            "profiles_computed": result["profiles_computed"],
            "branch_vector_selection": result["branch_vector_selection"],
            "query_embeddings_reused": bool(result.get("query_embeddings_reused", False)),
            "total_input_events": result["total_input_events"],
            "embedded_events_count": result["embedded_events_count"],
            "skipped_events_count": result["skipped_events_count"],
            "status_counts": result["status_counts"],
            "db_categories_updated": result["db_categories_updated"],
            "all_checks_passed": bool(result.get("asset_validation_passed", True)),
            "dual_branch_events": self.get_dual_branch_events(),
        }

    def run_dual_branch_pipeline_from_db(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Alias for DB-backed dual-branch pipeline execution (`POST /api/dual-branch/run-db`)."""
        res = self.run_dual_branch_pipeline(payload)
        res["events"] = list(self.dual_batch_events)
        return res

    def inspect_classify(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run Dual-Branch Hierarchical 2-Stage kNN + Stage-wise Fusion + Demo Module 6 estimation."""
        title = str(payload.get("title") or "").strip()
        description = str(payload.get("description") or "").strip()
        organizer = str(payload.get("organizer") or "").strip()
        role = str(payload.get("role") or "participant").strip()
        post_id = str(payload.get("post_id") or "").strip()
        formats = normalize_string_list_field(payload.get("format"))

        if not title and not description:
            raise ValueError("Vui lòng nhập ít nhất tên hoạt động hoặc mô tả ngắn.")

        emb_cfg = self.dual_assets.config.get("embedding") or {}
        default_bvs = emb_cfg.get("branch_vector_selection") or {}
        sample_branch_vector = str(
            payload.get("sample_branch_vector")
            or default_bvs.get("sample_branch", "title_description")
        ).strip()
        rubric_branch_vector = str(
            payload.get("rubric_branch_vector")
            or default_bvs.get("rubric_branch", "title_description")
        ).strip()
        for bname, pval in (
            ("sample_branch_vector", sample_branch_vector),
            ("rubric_branch_vector", rubric_branch_vector),
        ):
            if pval not in CLASSIFICATION_VECTOR_PROFILES:
                raise ValueError(
                    f"Invalid {bname}={pval!r}. Expected one of {CLASSIFICATION_VECTOR_PROFILES}."
                )

        stage_overrides: Dict[str, str] = {}
        for skey in (
            "sample_stage1_vector",
            "sample_stage2_vector",
            "rubric_stage1_vector",
            "rubric_stage2_vector",
        ):
            sval = payload.get(skey)
            if sval not in (None, ""):
                sval_clean = str(sval).strip()
                if sval_clean not in CLASSIFICATION_VECTOR_PROFILES:
                    raise ValueError(
                        f"Invalid {skey}={sval_clean!r}. Expected one of {CLASSIFICATION_VECTOR_PROFILES}."
                    )
                stage_overrides[skey] = sval_clean

        stage1_top_k = max(
            1,
            min(
                4,
                int(
                    payload.get("stage1_top_k")
                    if payload.get("stage1_top_k") not in (None, "")
                    else self.dual_default_params.get("candidate_group_count", 2)
                ),
            ),
        )
        k1 = max(
            1,
            min(
                30,
                int(
                    payload.get("k1")
                    if payload.get("k1") not in (None, "")
                    else self.dual_default_params.get("k1", 10)
                ),
            ),
        )
        k2 = max(
            1,
            min(
                30,
                int(
                    payload.get("k2")
                    if payload.get("k2") not in (None, "")
                    else self.dual_default_params.get("k2", 10)
                ),
            ),
        )
        beta = max(
            0.0,
            min(
                1.0,
                float(
                    payload.get("beta")
                    if payload.get("beta") not in (None, "")
                    else self.dual_default_params.get("stage1_fusion_beta", 0.5)
                ),
            ),
        )
        alpha = max(
            0.0,
            min(
                1.0,
                float(
                    payload.get("alpha")
                    if payload.get("alpha") not in (None, "")
                    else self.dual_default_params.get("stage2_fusion_alpha", 0.5)
                ),
            ),
        )

        raw_event = {
            "is_event": True,
            "post_id": post_id or "live_inspect",
            "title": title,
            "description": description,
            "organizer": organizer,
            "format": formats,
        }
        profile_texts = build_all_classification_profile_texts(raw_event, query_prefix="query: ")
        query_text = profile_texts["title_description"]

        t0 = time.perf_counter()
        reused_query_vec = False
        vectors_by_profile: Dict[str, np.ndarray] = {}
        needed_profiles = sorted(
            {
                sample_branch_vector,
                rubric_branch_vector,
                *stage_overrides.values(),
                "title_description",
            }
        )
        cached_map = self.dual_multi_vectors_by_post_id.get(post_id) if post_id else None
        for prof in needed_profiles:
            if (
                payload.get("use_cached_query") is True
                and cached_map is not None
                and prof in cached_map
            ):
                p_vec = cached_map[prof]
                reused_query_vec = True
            else:
                ptxt = profile_texts[prof]
                pbody = ptxt[len("query: ") :] if ptxt.startswith("query: ") else ptxt
                p_vec = l2_normalize(self.embedder.encode_query(pbody))
            vectors_by_profile[prof] = np.asarray(p_vec, dtype=np.float32)

        vec = vectors_by_profile["title_description"].tolist()

        runtime_params = dict(self.dual_default_params)
        runtime_params["candidate_group_count"] = stage1_top_k
        runtime_params["k1"] = k1
        runtime_params["k2"] = k2
        runtime_params["stage1_fusion_beta"] = beta
        runtime_params["stage2_fusion_alpha"] = alpha
        runtime_params["sample_branch_vector"] = sample_branch_vector
        runtime_params["rubric_branch_vector"] = rubric_branch_vector
        for skey, sval in stage_overrides.items():
            runtime_params[skey] = sval

        algorithm = self.dual_registry.create(
            self.dual_algorithm_name,
            sample_retriever=self.dual_sample_retriever,
            rubric_retriever=self.dual_rubric_retriever,
            rubric_data=self.dual_assets.rubric,
            params=runtime_params,
        )
        shared_q_np = vectors_by_profile["title_description"]
        prediction = algorithm.classify(
            shared_q_np,
            event_context=raw_event,
            query_vectors_by_profile=vectors_by_profile,
        )
        dual_cls = self.dual_decision_handler.decide(prediction=prediction, event=raw_event)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Resolve rubric criterion for Demo Module 6 rule-based point estimation
        s1_fused = (dual_cls.get("scores") or {}).get("stage1", {}).get("fused") or []
        s2_fused = (dual_cls.get("scores") or {}).get("stage2", {}).get("fused") or []
        selected_groups = [g["group_id"] for g in s1_fused if g.get("selected_for_stage2")]

        status = dual_cls.get("status")
        if status == "PREDICTED" and dual_cls.get("predicted_criterion_ref"):
            m6_ref = str(dual_cls["predicted_criterion_ref"])
            m6_ref_mode = "predicted"
        elif status == "NEEDS_REVIEW" and s2_fused:
            m6_ref = str(s2_fused[0]["criterion_ref"])
            m6_ref_mode = "suggested_preview"
        else:
            m6_ref = None
            m6_ref_mode = "none"

        ev = EventNormalized(
            event_id=raw_event["post_id"],
            title=title or description,
            description=description,
            organizer=organizer,
        )
        credit_est = self.credit_engine.estimate_event_credit(
            event=ev,
            rubric_item_id=m6_ref,
            ledger=None,
            role=role,
        )

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
            "profile_query_texts": profile_texts,
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
                "beta": beta,
                "alpha": alpha,
                "sample_branch_vector": sample_branch_vector,
                "rubric_branch_vector": rubric_branch_vector,
            },
            "reference_pool_sizes": {
                "sample_stage1_total": self.dual_sample_retriever.eligible_count,
                "sample_stage2_filtered": sample_stage2_pool,
                "rubric_stage1_total": len(self.dual_rubric_retriever.eligible_stage1_groups),
                "rubric_stage2_filtered": rubric_stage2_pool,
            },
            "dual_branch": dual_cls,
            "module6_credit_estimate": credit_est.to_dict(),
            "module6_ref_mode": m6_ref_mode,
            "rubric_item_name": self.rubric_name_by_id.get(credit_est.rubric_item_id or "", ""),
        }

    def bootstrap_payload(self) -> Dict[str, Any]:
        """Initial payload powering Tab 1 & Tab 2 (Classification & DB Events Browser)."""
        return {
            "embedder_status": asdict(self.embedder_status),
            "dual_branch_summary": self._dual_branch_summary(),
            "dual_branch_events": self.get_dual_branch_events(),
        }

    def _build_reference_now_presets(self) -> List[Dict[str, str]]:
        """Compute live PREDICTED upcoming counts for reference_now quick presets on DB pool."""
        preset_specs = [
            ("2026-10-01T08:00:00+07:00", "01/10/2026"),
            ("2026-11-01T08:00:00+07:00", "01/11/2026"),
            ("2026-04-01T08:00:00+07:00", "01/04/2026"),
        ]
        predicted_cands = [
            c for c in self.rec_candidates
            if str((c.classification or {}).get("status") or "") == "PREDICTED"
        ]
        out: List[Dict[str, str]] = []
        for iso_val, date_label in preset_specs:
            ref_dt = parse_reference_now(iso_val)
            upcoming = 0
            for c in predicted_cands:
                dl = resolve_effective_deadline(c.event)
                if dl is None or dl >= ref_dt:
                    upcoming += 1
            out.append({"value": iso_val, "label": f"{date_label} ({upcoming} PREDICTED)"})
        return out

    def get_recommendation_bootstrap(self) -> Dict[str, Any]:
        """Bootstrap metadata powering Tab 3 (Scoring P5–P6) & Tab 4 (MMR Selector P7)."""
        chips_raw = self.rec_config.get("interest_chips") or {}
        interest_chips = [
            {"id": str(k), "label": str(v["label"]), "text": str(v["text"])}
            for k, v in chips_raw.items()
        ]
        summary = self._dual_branch_summary()
        status_counts = summary.get("status_counts") or {}
        db_rows = fetch_events_from_db(db_path=self.db_path, exclude_cancelled=True)
        return {
            "data_source": "database",
            "db_path": self.db_path.name,
            "interest_chips": interest_chips,
            "personas": available_demo_personas(),
            "weight_presets": dict(self.rec_config.get("weight_presets") or {}),
            "schedule_presets": DemoScheduleProvider.available_presets(),
            "credit_presets": DemoCreditProvider.available_presets(),
            "group_caps": self.demo_credit_provider.group_caps,
            "reference_now_presets": self._build_reference_now_presets(),
            "rec_embedding_fields": list(REC_EMBEDDING_FIELDS),
            "mmr_defaults": MMRConfig.from_config(self.rec_config).to_dict(),
            "pool_stats": {
                "total_events": len(self.rec_candidates),
                "predicted_count": int(status_counts.get("PREDICTED", 0)),
                "needs_review_count": int(status_counts.get("NEEDS_REVIEW", 0)),
                "no_match_count": int(status_counts.get("NO_MATCH", 0)),
                "sample_embeddings_count": int(self.dual_assets.sample_branch.embeddings.shape[0]),
                "db_active_events_count": len(db_rows),
            },
        }

    def run_recommendation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Recommendation pipeline strictly on `drl_database.db` (Module 4 + Demo M6 + P5 + P6 [+ optional P7])."""
        req = dict(payload or {})
        if bool(req.get("refresh_db", False)):
            self._load_dual_branch_batch_from_db(write_back_db=True, write_output_files=True)

        result = run_recommendation_demo(
            req,
            candidates=self.rec_candidates,
            sample_embeddings=self.dual_assets.sample_branch.embeddings,
            embedder=self.embedder,
            credit_provider=self.demo_credit_provider,
            config=self.rec_config,
            chip_vec_cache=self.rec_chip_vec_cache,
        )
        result["pipeline_summary"]["data_source"] = "database"
        result["pipeline_summary"]["db_path"] = self.db_path.name
        return result

    def run_mmr_selection(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dedicated P7 MMR Selector endpoint (`POST /api/recommendation/mmr`) on `drl_database.db`."""
        req = dict(payload or {})
        mmr_cfg = dict(req.get("mmr") or {})
        mmr_cfg["enabled"] = True
        req["mmr"] = mmr_cfg
        return self.run_recommendation(req)


def make_handler(service: PrototypeService):
    """Create HTTP request handler bound to a PrototypeService instance."""

    class PrototypeHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            print(f"[server] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

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
            path = urlparse(self.path).path

            if path in ("/", "/index.html"):
                self._send_file(PROTOTYPE_DIR / "index.html", "text/html; charset=utf-8")
            elif path == "/styles.css":
                self._send_file(PROTOTYPE_DIR / "styles.css", "text/css; charset=utf-8")
            elif path == "/app.js":
                self._send_file(PROTOTYPE_DIR / "app.js", "application/javascript; charset=utf-8")
            elif path == "/api/bootstrap":
                self._send_json(service.bootstrap_payload())
            elif path == "/api/dual-branch/events":
                self._send_json(service.get_dual_branch_events())
            elif path == "/api/recommendation/bootstrap":
                self._send_json(service.get_recommendation_bootstrap())
            else:
                self.send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {path}")

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                body = self._read_json_body()
                if path == "/api/classify":
                    self._send_json(service.inspect_classify(body))
                elif path == "/api/dual-branch/run":
                    self._send_json(service.run_dual_branch_pipeline(body))
                elif path == "/api/dual-branch/run-db":
                    self._send_json(service.run_dual_branch_pipeline_from_db(body))
                elif path == "/api/recommendation/run":
                    self._send_json(service.run_recommendation(body))
                elif path == "/api/recommendation/mmr":
                    self._send_json(service.run_mmr_selection(body))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {path}")
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    return PrototypeHandler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run No-Le-DRL DB-Backed Classification, Demo Module 6, Scoring & MMR UI Server."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print("Đang khởi tạo No-Le-DRL (DB-Backed Classification + Demo M6 + Scoring + MMR)...")
    service = PrototypeService()
    st = service.embedder_status
    db = service._dual_branch_summary()
    print(f"  • Embedder    : {st.active_model} ({st.dimension}d, device={st.device})")
    print(f"  • Database    : {service.db_path} ({db['batch_events_count']} sự kiện)")
    print(
        f"  • Dual-Branch : {db['algorithm']} "
        f"(Sample={db['sample_reference_count']}, Rubric={db['rubric_total_prototypes']}, "
        f"β={db['default_params']['beta']}, α={db['default_params']['alpha']}, provisional)"
    )
    print(f"  • DB Events   : {db['batch_events_count']} sự kiện -> {db['status_counts']}")
    print(f"  • URL         : http://{args.host}:{args.port}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
