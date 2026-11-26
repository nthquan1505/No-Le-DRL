"""End-to-end Demo Recommendation Orchestrator (P4 + Demo M4/M6 -> P5 -> P6).

Consumes ONLY `PREDICTED` events from `outputs/classification_events.json` merged
with normalized event contracts (`data/processed/test_fixtures_events.json`) and
shared query vectors (`data/generated/events_query_embeddings.npy`), plus optional
synthetic mandatory event (`6.4` Tuần SHCD) for testing the Penalty branch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from configs.model_config import PROJECT_ROOT
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
from src.recommendation.scoring import MultiCriteriaScorer
from src.schemas import EventNormalized
from src.vector_ops import l2_normalize

DEFAULT_FIXTURES_PATH = PROJECT_ROOT / "data" / "processed" / "test_fixtures_events.json"
DEFAULT_CLASSIFICATION_PATH = PROJECT_ROOT / "outputs" / "classification_events.json"
DEFAULT_QUERY_EMB_PATH = PROJECT_ROOT / "data" / "generated" / "events_query_embeddings.npy"
DEFAULT_QUERY_META_PATH = PROJECT_ROOT / "data" / "generated" / "events_query_metadata.json"
DEFAULT_SAMPLE_EMB_PATH = PROJECT_ROOT / "data" / "processed" / "cls_index" / "sample_embeddings.npy"


def load_recommendation_candidate_pool(
    *,
    fixtures_path: Path = DEFAULT_FIXTURES_PATH,
    classification_path: Path = DEFAULT_CLASSIFICATION_PATH,
    query_embeddings_path: Path = DEFAULT_QUERY_EMB_PATH,
    query_metadata_path: Path = DEFAULT_QUERY_META_PATH,
) -> List[RecommendationCandidateInput]:
    """Load all batch events merged with normalized fixtures, P2 classifications, and E5 query vectors."""
    fix_raw = json.loads(fixtures_path.read_text(encoding="utf-8"))
    fix_map: Dict[str, EventNormalized] = {
        str(rec["event_id"]): EventNormalized.from_dict(rec)
        for rec in (fix_raw.get("events") or [])
    }

    cls_records = json.loads(classification_path.read_text(encoding="utf-8"))

    q_vecs_by_id: Dict[str, List[float]] = {}
    if query_embeddings_path.is_file() and query_metadata_path.is_file():
        q_arr = np.load(query_embeddings_path, mmap_mode="r")
        q_meta = json.loads(query_metadata_path.read_text(encoding="utf-8"))
        for row in q_meta.get("records") or []:
            pid = str(row.get("post_id") or "").strip()
            idx = int(row["embedding_row"])
            if pid and 0 <= idx < q_arr.shape[0]:
                q_vecs_by_id[pid] = [float(v) for v in q_arr[idx].tolist()]

    pool: List[RecommendationCandidateInput] = []
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
        pool.append(
            RecommendationCandidateInput(
                event=ev,
                classification=dict(rec.get("classification") or {}),
                event_vec=q_vecs_by_id.get(pid),
            )
        )
    return pool


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
        shcd_vec = l2_normalize(embedder.encode_query(f"{shcd_ev.title} {shcd_ev.description}"))
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

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    ranked_dicts = []
    for idx, sc in enumerate(scored_candidates, start=1):
        d = sc.to_dict()
        d["rank"] = idx
        ranked_dicts.append(d)

    return {
        "elapsed_ms": round(elapsed_ms, 2),
        "pipeline_summary": {
            "total_batch_events": filter_report.total_input,
            "predicted_events_entered_p5": filter_report.predicted_input,
            "excluded_not_predicted": filter_report.excluded_not_predicted,
            "excluded_status_counts": filter_report.excluded_status_counts,
            "eligible_after_p5": len(filter_report.eligible),
            "dropped_by_p5": len(filter_report.dropped),
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
        "interest_calibration": cal_range.to_dict(),
        "ranked_events": ranked_dicts,
        "dropped_events": [d.to_dict() for d in filter_report.dropped],
    }
