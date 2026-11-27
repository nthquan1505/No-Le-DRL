"""End-to-end Demo Recommendation Orchestrator (P4 + Demo M4/M6 -> P5 -> P6 -> P7 MMR).

Consumes ONLY `PREDICTED` events from `outputs/classification_events.json` merged
with normalized event contracts (`data/processed/test_fixtures_events.json`) and
shared query vectors (`data/generated/events_query_embeddings.npy`), plus optional
synthetic mandatory event (`6.4` Tuần SHCD) for testing the Penalty branch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from configs.model_config import PROJECT_ROOT
from src.db_event_adapter import (
    DB_CATEGORY_TO_GROUP_ID,
    compute_events_fingerprint,
    db_row_to_event_normalized,
    fetch_events_from_db,
    fetch_school_rubric_from_db,
    fetch_user_from_db,
    fetch_user_subscriptions_from_db,
)
from src.embedder import BaseEmbedder
from src.recommendation.config import load_recommendation_config
from src.recommendation.demo_module4 import DemoScheduleProvider
from src.recommendation.demo_module6 import DemoCreditProvider
from src.recommendation.demo_profile import (
    DEMO_PERSONAS,
    DemoUserProfile,
    build_demo_user_vector,
    create_mandatory_shcd_demo_event,
    resolve_demo_weights,
)
from src.recommendation.filter import (
    HardConstraintFilter,
    RecommendationCandidateInput,
    parse_reference_now,
)
from src.recommendation.rec_embedder import (
    DEFAULT_REC_PASSAGE_PREFIX,
    REC_EMBEDDING_FIELDS,
    build_recommendation_event_text,
    generate_or_load_recommendation_embeddings,
)
from src.recommendation.scoring import MultiCriteriaScorer
from src.recommendation.selector import MMRSelector
from src.schemas import EventNormalized
from src.vector_ops import l2_normalize

DEFAULT_FIXTURES_PATH = PROJECT_ROOT / "data" / "processed" / "test_fixtures_events.json"
DEFAULT_CLASSIFICATION_PATH = PROJECT_ROOT / "outputs" / "classification_events.json"
DEFAULT_QUERY_EMB_PATH = PROJECT_ROOT / "data" / "generated" / "events_query_embeddings.npy"
DEFAULT_QUERY_META_PATH = PROJECT_ROOT / "data" / "generated" / "events_query_metadata.json"
DEFAULT_REC_EMB_PATH = PROJECT_ROOT / "data" / "generated" / "events_rec_embeddings.npy"
DEFAULT_REC_META_PATH = PROJECT_ROOT / "data" / "generated" / "events_rec_metadata.json"
DEFAULT_SAMPLE_EMB_PATH = PROJECT_ROOT / "data" / "processed" / "cls_index" / "sample_embeddings.npy"
DEFAULT_DB_PATH = PROJECT_ROOT / "drl_database.db"


def load_recommendation_candidate_pool_from_db(
    *,
    db: Any = None,
    db_path: Optional[Union[str, Path]] = None,
    embedder: Optional[BaseEmbedder] = None,
    classification_by_event_id: Optional[Dict[int, Dict[str, Any]]] = None,
    exclude_cancelled: bool = True,
    only_classified_in_db: bool = False,
    rec_embeddings_path: Optional[Path] = None,
    rec_metadata_path: Optional[Path] = None,
    reuse_existing_embeddings: bool = True,
) -> List[RecommendationCandidateInput]:
    """Load Recommendation candidates directly from `drl_database.db` (`events` table).

    Embeds each event from the 4 required fields (`title + description + topic + format`)
    using `passage: ` prefix for `Interest` matching against `user_vec`.
    Returns `[]` cleanly when the database is empty.
    """
    rows = fetch_events_from_db(
        db=db,
        db_path=db_path or DEFAULT_DB_PATH,
        only_classified=only_classified_in_db,
        exclude_cancelled=exclude_cancelled,
    )
    if not rows:
        return []

    normalized_events: List[EventNormalized] = [db_row_to_event_normalized(r) for r in rows]
    rec_vecs_by_id: Dict[str, List[float]] = {}
    if embedder is not None:
        fp = compute_events_fingerprint(rows)
        rec_emb_set = generate_or_load_recommendation_embeddings(
            normalized_events,
            embedder=embedder,
            source_fingerprint=fp,
            embeddings_path=rec_embeddings_path,
            metadata_path=rec_metadata_path,
            reuse_existing=reuse_existing_embeddings,
        )
        for ev in normalized_events:
            vec = rec_emb_set.get_vector(ev.event_id)
            if vec is not None:
                rec_vecs_by_id[ev.event_id] = vec

    cls_lookup = classification_by_event_id or {}
    pool: List[RecommendationCandidateInput] = []
    for row, ev in zip(rows, normalized_events):
        db_eid = int(row["event_id"])
        cls_payload = cls_lookup.get(db_eid)
        if cls_payload is None:
            db_cat = row.get("category")
            if db_cat is not None and int(db_cat) in DB_CATEGORY_TO_GROUP_ID:
                grp = DB_CATEGORY_TO_GROUP_ID[int(db_cat)]
                cls_payload = {
                    "status": "PREDICTED",
                    "predicted_group_id": grp,
                    "predicted_criterion_ref": None,
                    "source": "db_category_column",
                }
            else:
                cls_payload = {
                    "status": "UNCLASSIFIED",
                    "predicted_group_id": None,
                    "predicted_criterion_ref": None,
                }

        ev.metadata["rec_embedding_text"] = build_recommendation_event_text(
            ev, include_prefix=True, passage_prefix=DEFAULT_REC_PASSAGE_PREFIX
        )
        pool.append(
            RecommendationCandidateInput(
                event=ev,
                classification=dict(cls_payload),
                event_vec=rec_vecs_by_id.get(ev.event_id),
            )
        )
    return pool


def build_demo_profile_from_db_user(
    user_id: str,
    *,
    db: Any = None,
    db_path: Optional[Union[str, Path]] = None,
    embedder: Optional[BaseEmbedder] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[DemoUserProfile]:
    """Hydrate a `DemoUserProfile` from `users` + `event_subscription` + `rubric` tables."""
    user_row = fetch_user_from_db(user_id, db=db, db_path=db_path or DEFAULT_DB_PATH)
    if user_row is None:
        return None

    subs = fetch_user_subscriptions_from_db(user_id, db=db, db_path=db_path or DEFAULT_DB_PATH)
    participated = [f"db_{s['event_id']}" for s in subs if s.get("status") == "done"]

    credits_list = user_row.get("credits") or [0, 0, 0, 0, 0]
    current_by_group = {
        f"C{i + 1}": float(credits_list[i]) if i < len(credits_list) else 0.0
        for i in range(5)
    }

    target_by_group: Dict[str, float] = {}
    if user_row.get("school_id") is not None:
        rubric_row = fetch_school_rubric_from_db(
            int(user_row["school_id"]), db=db, db_path=db_path or DEFAULT_DB_PATH
        )
        if rubric_row and isinstance(rubric_row.get("max_credit"), list):
            mc = rubric_row["max_credit"]
            target_by_group = {
                f"C{i + 1}": float(mc[i]) if i < len(mc) else 20.0
                for i in range(5)
            }

    pref_topics = [str(t) for t in (user_row.get("preferred_topics") or []) if str(t).strip()]
    pref_formats = [str(f) for f in (user_row.get("preferred_formats") or []) if str(f).strip()]
    pref_parts: List[str] = []
    if pref_topics:
        pref_parts.append(f"Chủ đề yêu thích: {', '.join(pref_topics)}")
    if pref_formats:
        pref_parts.append(f"Hình thức yêu thích: {', '.join(pref_formats)}")
    free_text = ". ".join(pref_parts) if pref_parts else None

    user_vec = (
        l2_normalize(embedder.encode_query(free_text))
        if (embedder is not None and free_text)
        else None
    )

    w_raw = user_row.get("weights") or [0.40, 0.25, 0.20, 0.15]
    # Note: SYSTEM_ARCHITECTURE users.weights order is [w1_gain, w2_fit, w3_interested, w4_urgency]
    custom_w = {
        "w1_gain": float(w_raw[0]) if len(w_raw) > 0 else 0.40,
        "w2_interest": float(w_raw[2]) if len(w_raw) > 2 else 0.25,
        "w3_fit": float(w_raw[1]) if len(w_raw) > 1 else 0.20,
        "w4_urgency": float(w_raw[3]) if len(w_raw) > 3 else 0.15,
    }
    preset_name, resolved_w = resolve_demo_weights("custom", custom_w, config=config)

    start_year = user_row.get("start_year")
    cohort = f"K{str(start_year)[-2:]}" if start_year else None

    return DemoUserProfile(
        user_id=str(user_row["user_id"]),
        cohort=cohort,
        interest_chips=[],
        free_text=free_text,
        user_vec=user_vec,
        participated_event_ids=participated,
        current_by_group=current_by_group,
        target_by_group=target_by_group,
        weight_preset=preset_name,
        weights=resolved_w,
    )


def load_recommendation_candidate_pool(
    *,
    fixtures_path: Path = DEFAULT_FIXTURES_PATH,
    classification_path: Path = DEFAULT_CLASSIFICATION_PATH,
    query_embeddings_path: Path = DEFAULT_QUERY_EMB_PATH,
    query_metadata_path: Path = DEFAULT_QUERY_META_PATH,
    embedder: Optional[BaseEmbedder] = None,
    use_rec_4field_embeddings: bool = False,
    rec_embeddings_path: Path = DEFAULT_REC_EMB_PATH,
    rec_metadata_path: Path = DEFAULT_REC_META_PATH,
) -> List[RecommendationCandidateInput]:
    """Load batch events from legacy JSON fixtures (optionally computing 4-field Rec embeddings)."""
    fix_raw = json.loads(fixtures_path.read_text(encoding="utf-8"))
    fix_map: Dict[str, EventNormalized] = {
        str(rec["event_id"]): EventNormalized.from_dict(rec)
        for rec in (fix_raw.get("events") or [])
    }

    cls_records = json.loads(classification_path.read_text(encoding="utf-8"))

    # Hydrate `format` & `topic` into EventNormalized.metadata from raw classification records
    ordered_events: List[EventNormalized] = []
    ordered_cls: List[Dict[str, Any]] = []
    for rec in cls_records:
        pid = str(rec.get("post_id") or "").strip()
        if not pid:
            continue
        ev = fix_map.get(pid)
        if ev is None:
            ev = EventNormalized(
                event_id=pid,
                title=str(rec.get("title") or ""),
                description=str(rec.get("description") or ""),
                organizer=str(rec.get("organizer") or ""),
                start_datetime=rec.get("start_datetime"),
                end_datetime=rec.get("end_datetime"),
                reg_end_datetime=rec.get("reg_end_datetime"),
                online_mode=rec.get("online_mode"),
                is_online=rec.get("is_online"),
                location=rec.get("location"),
            )
        if "topic" not in ev.metadata:
            ev.metadata["topic"] = list(rec.get("topic") or rec.get("category") or ev.categories or [])
        if "format" not in ev.metadata:
            ev.metadata["format"] = list(rec.get("format") or [])
        ev.metadata["rec_embedding_text"] = build_recommendation_event_text(
            ev, include_prefix=True, passage_prefix=DEFAULT_REC_PASSAGE_PREFIX
        )
        ordered_events.append(ev)
        ordered_cls.append(dict(rec.get("classification") or {}))

    vecs_by_id: Dict[str, List[float]] = {}
    if use_rec_4field_embeddings and embedder is not None:
        fp = compute_events_fingerprint(cls_records)
        rec_set = generate_or_load_recommendation_embeddings(
            ordered_events,
            embedder=embedder,
            source_fingerprint=fp,
            embeddings_path=rec_embeddings_path,
            metadata_path=rec_metadata_path,
            reuse_existing=True,
        )
        for ev in ordered_events:
            v = rec_set.get_vector(ev.event_id)
            if v is not None:
                vecs_by_id[ev.event_id] = v
    elif query_embeddings_path.is_file() and query_metadata_path.is_file():
        q_arr = np.load(query_embeddings_path, mmap_mode="r")
        q_meta = json.loads(query_metadata_path.read_text(encoding="utf-8"))
        for row in q_meta.get("records") or []:
            pid = str(row.get("post_id") or "").strip()
            idx = int(row["embedding_row"])
            if pid and 0 <= idx < q_arr.shape[0]:
                vecs_by_id[pid] = [float(v) for v in q_arr[idx].tolist()]

    return [
        RecommendationCandidateInput(
            event=ev,
            classification=cls_obj,
            event_vec=vecs_by_id.get(ev.event_id),
        )
        for ev, cls_obj in zip(ordered_events, ordered_cls)
    ]


def run_recommendation_demo(
    payload: Dict[str, Any],
    *,
    candidates: Sequence[RecommendationCandidateInput],
    sample_embeddings: np.ndarray,
    embedder: BaseEmbedder,
    credit_provider: Optional[DemoCreditProvider] = None,
    config: Optional[Dict[str, Any]] = None,
    chip_vec_cache: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """Run full Demo Recommendation pipeline (P4 -> Demo M4/M6 -> P5 -> P6) and return JSON dict."""
    t0 = time.perf_counter()
    cfg = config if config is not None else load_recommendation_config()
    pen_cfg = (cfg.get("scoring") or {}).get("penalty_mandatory") or {}

    if credit_provider is None:
        credit_provider = DemoCreditProvider(
            enable_item_2_2_penalty_link=bool(pen_cfg.get("enable_item_2_2_penalty_link", False))
        )

    # 1. Resolve DemoUserProfile (optionally starting from a named persona)
    persona_id = str(payload.get("persona_id") or "").strip()
    base = dict(DEMO_PERSONAS[persona_id]) if persona_id in DEMO_PERSONAS else {}

    def _pick(key: str, default: Any) -> Any:
        if key in payload and payload[key] is not None:
            return payload[key]
        if key in base:
            return base[key]
        return default

    interest_chips = list(_pick("interest_chips", ["ai_tech", "academic_nckh"]))
    free_text = _pick("free_text", "")
    weight_preset_req = str(_pick("weight_preset", "balanced"))
    custom_weights = payload.get("custom_weights") or payload.get("weights")
    resolved_preset, weights = resolve_demo_weights(
        weight_preset_req,
        custom_weights if weight_preset_req == "custom" else None,
        config=cfg,
    )

    # Build static 384d E5 user vector
    user_vec = build_demo_user_vector(
        interest_chips,
        free_text,
        embedder,
        config=cfg,
        chip_vec_cache=chip_vec_cache,
    )

    pref_raw = _pick("pref_hours", [8.0, 21.0])
    pref_hours = (float(pref_raw[0]), float(pref_raw[1])) if len(pref_raw) == 2 else (8.0, 21.0)

    max_budget_raw = _pick("max_budget_vnd", 30000)
    max_budget_vnd = None if max_budget_raw in ("", None) else max(0, int(max_budget_raw))

    max_travel_raw = _pick("max_travel_min", 45)
    max_travel_min = None if max_travel_raw in ("", None) else max(0, int(max_travel_raw))

    current_by_group_raw = _pick(
        "current_by_group",
        {"C1": 22.0, "C2": 15.0, "C3": 10.0, "C4": 10.0, "C5": 0.0},
    )
    current_by_group = {k: max(0.0, float(v)) for k, v in dict(current_by_group_raw).items()}
    target_total = float(_pick("target_total", 85.0))
    target_by_group_raw = _pick("target_by_group", {}) or {}
    target_by_group = {
        k: max(0.0, float(v)) for k, v in dict(target_by_group_raw).items() if v is not None
    }

    ref_now_str = str(_pick("reference_now", "2026-10-01T08:00:00+07:00"))
    ref_now_dt = parse_reference_now(ref_now_str)

    profile = DemoUserProfile(
        user_id=str(_pick("user_id", persona_id or "demo_sv_01")),
        cohort=str(_pick("cohort", "")).strip() or None,
        faculty=str(_pick("faculty", "")).strip() or None,
        class_id=str(_pick("class_id", "")).strip() or None,
        interest_chips=interest_chips,
        free_text=str(free_text or "").strip() or None,
        user_vec=user_vec,
        home_campus=str(_pick("home_campus", "cs1_nvc")).strip() or None,
        max_travel_min=max_travel_min,
        strict_max_travel=bool(_pick("strict_max_travel", True)),
        max_budget_vnd=max_budget_vnd,
        pref_hours=pref_hours,
        participated_event_ids=list(payload.get("participated_event_ids") or []),
        hidden_event_ids=list(payload.get("hidden_event_ids") or []),
        credit_preset=str(_pick("credit_preset", "mid_semester")),
        current_by_group=current_by_group,
        target_total=target_total,
        target_by_group=target_by_group,
        schedule_preset=str(_pick("schedule_preset", "light_schedule")),
        weight_preset=resolved_preset,
        weights=weights,
        reference_now=ref_now_dt.isoformat(),
    )

    # 2. Build Demo Module 4 provider
    sched_payload = payload.get("schedule")
    if isinstance(sched_payload, dict) and sched_payload:
        sched_provider = DemoScheduleProvider.from_demo_payload(sched_payload, config=cfg)
    else:
        sched_provider = DemoScheduleProvider.from_preset(profile.schedule_preset, config=cfg)

    # 3. Build Demo Module 6 ledger & shortfall
    ledger = credit_provider.build_ledger_from_group_totals(
        profile.current_by_group,
        target_total=profile.target_total,
        target_by_group=profile.target_by_group or None,
    )
    ledger_eval = credit_provider.evaluate_ledger(ledger)
    shortfall = credit_provider.compute_group_shortfall(ledger)

    # 4. Prepare candidate pool (+ optional synthetic 6.4 mandatory event for testing Penalty)
    active_candidates = list(candidates)
    include_mandatory = bool(_pick("include_mandatory_demo", False))
    if include_mandatory:
        shcd_ev, shcd_cls = create_mandatory_shcd_demo_event(ref_now_dt)
        shcd_ev.metadata["topic"] = ["Orientation", "Mandatory"]
        shcd_ev.metadata["format"] = ["Sinh hoạt công dân"]
        shcd_body = build_recommendation_event_text(shcd_ev, include_prefix=False)
        shcd_ev.metadata["rec_embedding_text"] = f"{DEFAULT_REC_PASSAGE_PREFIX}{shcd_body}"
        shcd_vec = l2_normalize(embedder.encode_passage(shcd_body))
        active_candidates.append(
            RecommendationCandidateInput(
                event=shcd_ev,
                classification=shcd_cls,
                event_vec=shcd_vec,
            )
        )

    # 5. Run P5 Hard Filter (PREDICTED-only + 8 rules)
    hard_filter = HardConstraintFilter(config=cfg)
    filter_report = hard_filter.filter_candidates(
        active_candidates,
        profile=profile,
        schedule_provider=sched_provider,
        credit_provider=credit_provider,
        ledger=ledger,
        reference_now=ref_now_dt,
    )

    # 6. Run P6 Multi-Criteria Scoring
    interest_mode_override = payload.get("interest_calibration_mode")
    scorer = MultiCriteriaScorer(config=cfg)
    scored_candidates, cal_range = scorer.score_candidates(
        filter_report.eligible,
        profile=profile,
        schedule_provider=sched_provider,
        credit_provider=credit_provider,
        shortfall=shortfall,
        sample_embeddings=sample_embeddings,
        reference_now=ref_now_dt,
        interest_mode_override=interest_mode_override,
    )

    # 7. Run P7 List Selection via MMR (+ 1 Exploration Slot; CP-SAT deferred)
    mmr_overrides: Dict[str, Any] = {}
    if isinstance(payload.get("mmr"), dict):
        mmr_overrides.update(payload["mmr"])
    flat_mmr_map = {
        "mmr_enabled": "enabled",
        "top_n": "top_n",
        "mmr_top_n": "top_n",
        "mmr_lambda": "lambda_param",
        "mmr_alpha_semantic": "alpha_semantic",
        "mmr_cosine_rescale_mode": "cosine_rescale_mode",
        "mmr_enable_exploration_slot": "enable_exploration_slot",
        "mmr_pin_mandatory": "pin_mandatory",
    }
    for flat_k, target_k in flat_mmr_map.items():
        if flat_k in payload and payload[flat_k] is not None:
            mmr_overrides[target_k] = payload[flat_k]

    selector = MMRSelector(config=cfg)
    mmr_result = selector.select(scored_candidates, overrides=mmr_overrides or None)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    ranked_dicts = []
    for idx, sc in enumerate(scored_candidates, start=1):
        d = sc.to_dict()
        d["rank"] = idx
        ranked_dicts.append(d)

    selected_dicts = [sc.to_dict() for sc in mmr_result.selected]

    return {
        "elapsed_ms": round(elapsed_ms, 2),
        "pipeline_summary": {
            "total_batch_events": filter_report.total_input,
            "predicted_events_entered_p5": filter_report.predicted_input,
            "excluded_not_predicted": filter_report.excluded_not_predicted,
            "excluded_status_counts": filter_report.excluded_status_counts,
            "eligible_after_p5": len(filter_report.eligible),
            "dropped_by_p5": len(filter_report.dropped),
            "selected_by_p7_mmr": len(selected_dicts),
            "drop_reason_counts": filter_report.reason_counts,
            "include_mandatory_demo": include_mandatory,
            "reference_now": ref_now_dt.isoformat(),
        },
        "profile_used": profile.to_dict(),
        "demo_module6_state": {
            "group_caps": credit_provider.group_caps,
            "current_by_group": ledger_eval.capped_by_group,
            "remaining_cap_by_group": ledger_eval.remaining_cap_by_group,
            "target_gap_by_group": shortfall.target_gap_by_group,
            "current_total": ledger_eval.total_score,
            "target_total": shortfall.target_total,
        },
        "demo_module4_state": {
            "preset_id": sched_provider.preset_id,
            "weekly_slots_count": len(sched_provider.weekly_slots),
            "busy_intervals_count": len(sched_provider.get_schedule_context().busy_intervals),
            "exam_weeks": [list(p) for p in sched_provider.get_schedule_context().exam_weeks],
            "default_travel_buffer_min": sched_provider.get_schedule_context().default_travel_buffer_min,
        },
        "interest_calibration": {
            **cal_range.to_dict(),
            "event_embedding_fields": list(REC_EMBEDDING_FIELDS),
            "event_embedding_prefix": DEFAULT_REC_PASSAGE_PREFIX,
        },
        "mmr_selection": mmr_result.to_summary_dict(),
        "selected_events": selected_dicts,
        "ranked_events": ranked_dicts,
        "dropped_events": [d.to_dict() for d in filter_report.dropped],
    }
