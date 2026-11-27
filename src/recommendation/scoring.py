"""P6 Multi-Criteria Event Scoring (`src/recommendation/scoring.py`).

Implements Decision D-5 and the approved v2.0 scoring design (`docs/module3_p4_p5_p6_design.md`):
    Score(u, e) = w1 * Gain + w2 * Interest + w3 * Fit + w4 * Urgency + Penalty

Component definitions:
- Gain     in [0, 1] : Derived strictly from Module 6 (`effective_points`, `target_gap_by_group`, `source`).
- Interest in [0, 1] : Clipped Min-Max of E5 cosine similarity scaled using empirical `[c_min, c_max]`
                       computed by scanning the full reference sample embeddings (`sample_embeddings.npy`).
- Fit      in [0, 1] : Supplied by Module 4 `compute_fit(...)` (travel, slack, pref_hours, exam_week).
- Urgency  in [0, 1] : Time-only exponential decay `exp(-days_to_deadline / tau)` (scarcity omitted).
- Penalty  >= 0      : Pure mandatory-activity intervention boost (`> 0` for high-mandatory events like
                       6.4 Tuần Sinh hoạt Công dân to avoid DRL penalty; `0.0` for regular events).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.recommendation.config import load_recommendation_config
from src.recommendation.demo_module4 import DemoFitAssessment, DemoScheduleProvider
from src.recommendation.demo_module6 import DemoCreditProvider, MandatoryPenaltyInfo
from src.recommendation.demo_profile import DemoUserProfile
from src.recommendation.filter import EligibleCandidate, parse_reference_now
from src.recommendation.schedule_adapter import FitAssessment, ScheduleProvider
from src.schemas import CreditEstimate, GroupShortfall, ScoredEvent, ScoringWeights

DEFAULT_TZ = timezone(timedelta(hours=7))


@dataclass(frozen=True)
class InterestCalibrationRange:
    """Empirical `[c_min, c_max]` range used for E5 Clipped Min-Max Interest scaling."""

    c_min: float
    c_max: float
    mode: str
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "c_min": round(self.c_min, 6),
            "c_max": round(self.c_max, 6),
            "mode": self.mode,
            "sample_count": int(self.sample_count),
        }


@dataclass
class ScoredCandidate:
    """Rich scored candidate produced by P6 (wraps `ScoredEvent` with full explainability details)."""

    scored_event: ScoredEvent
    candidate: EligibleCandidate
    fit_assessment: FitAssessment
    interest_cos_raw: Optional[float]
    days_to_deadline: Optional[float]
    gain_details: Dict[str, Any] = field(default_factory=dict)
    penalty_details: Dict[str, Any] = field(default_factory=dict)
    reasoning: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        ev = self.candidate.event
        se = self.scored_event
        ce = self.candidate.credit_estimate
        mi = self.candidate.mandatory_info
        fit_dict = (
            self.fit_assessment.to_dict()
            if isinstance(self.fit_assessment, DemoFitAssessment)
            else {
                "fit_score": self.fit_assessment.fit_score,
                "has_conflict": self.fit_assessment.has_conflict,
                "in_exam_week": self.fit_assessment.in_exam_week,
                "estimated_travel_min": self.fit_assessment.estimated_travel_min,
                "nearest_slack_min": None,
                "conflict_detail": self.fit_assessment.conflict_detail,
                "sub_scores": {},
                "reasons": list(self.fit_assessment.reasons),
            }
        )
        return {
            "event_id": ev.event_id,
            "title": ev.title,
            "description": ev.description,
            "organizer": ev.organizer,
            "start_datetime": ev.start_datetime,
            "end_datetime": ev.end_datetime,
            "reg_end_datetime": ev.reg_end_datetime,
            "effective_deadline": (
                self.candidate.effective_deadline.isoformat()
                if self.candidate.effective_deadline is not None
                else None
            ),
            "online_mode": ev.online_mode,
            "location": ev.location,
            "fee_vnd": ev.fee_vnd,
            "fee_raw": ev.fee_raw,
            "registration_url": ev.registration_url,
            "source_url": ev.source_url,
            "rubric_item_id": se.rubric_item_id,
            "parent_group_id": se.parent_group_id,
            "criterion_name": self.candidate.classification.get("predicted_criterion_name") or "",
            "is_mandatory": mi.is_mandatory,
            "mandatory_info": mi.to_dict(),
            "score_total": se.score_total,
            "components": {
                "gain": se.gain,
                "interest": se.interest,
                "fit": se.fit,
                "urgency": se.urgency,
                "penalty": se.penalty,
            },
            "weighted_contributions": {
                "gain": round(se.weights.w1_gain * se.gain, 6),
                "interest": round(se.weights.w2_interest * se.interest, 6),
                "fit": round(se.weights.w3_fit * se.fit, 6),
                "urgency": round(se.weights.w4_urgency * se.urgency, 6),
                "penalty": round(se.penalty, 6),
            },
            "weights": se.weights.to_dict(),
            "credit_estimate": ce.to_dict(),
            "gain_details": dict(self.gain_details),
            "interest_cos_raw": (
                round(self.interest_cos_raw, 6) if self.interest_cos_raw is not None else None
            ),
            "fit_assessment": fit_dict,
            "days_to_deadline": (
                round(self.days_to_deadline, 3) if self.days_to_deadline is not None else None
            ),
            "penalty_details": dict(self.penalty_details),
            "reasoning": dict(self.reasoning),
        }


def compute_interest_calibration_range(
    user_vec: Optional[Sequence[float]],
    sample_embeddings: Optional[np.ndarray],
    *,
    mode: str = "per_user_samples_minmax",
    fixed_c_min: float = 0.72,
    fixed_c_max: float = 0.90,
) -> InterestCalibrationRange:
    """Scan all reference sample embeddings to determine empirical `[c_min, c_max]`."""
    if mode == "fixed" or user_vec is None or sample_embeddings is None or sample_embeddings.size == 0:
        c_lo = float(fixed_c_min)
        c_hi = float(fixed_c_max)
        if c_hi <= c_lo:
            c_lo, c_hi = 0.72, 0.90
        return InterestCalibrationRange(
            c_min=c_lo,
            c_max=c_hi,
            mode="fixed" if mode == "fixed" else "fixed_fallback",
            sample_count=int(sample_embeddings.shape[0]) if sample_embeddings is not None else 0,
        )

    u = np.asarray(user_vec, dtype=np.float32)
    u_norm = float(np.linalg.norm(u))
    if u_norm <= 1e-12:
        return InterestCalibrationRange(
            c_min=float(fixed_c_min),
            c_max=float(fixed_c_max),
            mode="fixed_fallback",
            sample_count=int(sample_embeddings.shape[0]),
        )
    u = u / u_norm

    sims = sample_embeddings @ u
    if mode == "per_user_samples_p1_p99":
        c_lo = float(np.percentile(sims, 1))
        c_hi = float(np.percentile(sims, 99))
    else:
        c_lo = float(np.min(sims))
        c_hi = float(np.max(sims))

    if c_hi - c_lo < 1e-6:
        c_lo, c_hi = float(fixed_c_min), float(fixed_c_max)

    return InterestCalibrationRange(
        c_min=c_lo,
        c_max=c_hi,
        mode=mode,
        sample_count=int(sample_embeddings.shape[0]),
    )


def scale_interest_clipped_minmax(
    cos_sim: float,
    cal_range: InterestCalibrationRange,
) -> float:
    """Clipped Min-Max scaling: `clip((cos - c_min) / (c_max - c_min), 0, 1)`."""
    denom = cal_range.c_max - cal_range.c_min
    if denom <= 1e-9:
        return 0.5
    scaled = (float(cos_sim) - cal_range.c_min) / denom
    return round(max(0.0, min(1.0, scaled)), 6)


class MultiCriteriaScorer:
    """P6 Multi-Criteria Event Scorer (`w1·Gain + w2·Interest + w3·Fit + w4·Urgency + Penalty`)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config if config is not None else load_recommendation_config()
        self._scoring_cfg: Dict[str, Any] = dict(cfg.get("scoring") or {})
        self._interest_chips_cfg: Dict[str, Any] = dict(cfg.get("interest_chips") or {})
        sched_cfg = cfg.get("schedule") or {}
        tz_hours = int(sched_cfg.get("default_timezone_offset_hours", 7))
        self._default_tz = timezone(timedelta(hours=tz_hours))

    def compute_gain(
        self,
        credit: CreditEstimate,
        shortfall: GroupShortfall,
        group_caps: Dict[str, float],
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute normalized Gain in [0, 1] purely from Module 6 outputs."""
        gain_cfg = self._scoring_cfg.get("gain") or {}
        p_ref = max(1e-6, float(gain_cfg.get("p_ref", 10.0)))
        gap_floor = max(0.0, min(1.0, float(gain_cfg.get("gap_floor", 0.5))))
        fallback_min_w = max(0.0, min(1.0, float(gain_cfg.get("fallback_min_weight", 0.7))))
        src_rel_map = gain_cfg.get("source_reliability") or {}

        if (not credit.is_credit_eligible) or credit.effective_points <= 0.0:
            return 0.0, {
                "p_use": 0.0,
                "g_pts": 0.0,
                "gap_ratio": 0.0,
                "m_gap": 0.0,
                "r_src": 0.0,
                "reason": "no_positive_credit_points",
            }

        if credit.source == "rule_fallback":
            conservative = fallback_min_w * credit.min_points + (1.0 - fallback_min_w) * credit.expected_points
            p_use = min(credit.effective_points, conservative)
        else:
            p_use = credit.effective_points

        g_pts = min(1.0, max(0.0, p_use / p_ref))

        grp = credit.group_id or ""
        grp_cap = max(1.0, float(group_caps.get(grp, 25.0)))
        grp_gap = max(0.0, float(shortfall.target_gap_by_group.get(grp, 0.0)))
        gap_ratio = max(0.0, min(1.0, grp_gap / grp_cap))
        m_gap = gap_floor + (1.0 - gap_floor) * gap_ratio

        default_rel = {"explicit": 1.00, "rule": 0.90, "rule_fallback": 0.80}
        r_src = float(src_rel_map.get(credit.source, default_rel.get(credit.source, 0.90)))

        gain = round(max(0.0, min(1.0, g_pts * m_gap * r_src)), 6)
        return gain, {
            "p_use": round(p_use, 4),
            "g_pts": round(g_pts, 6),
            "group_gap_points": round(grp_gap, 4),
            "group_cap": round(grp_cap, 4),
            "gap_ratio": round(gap_ratio, 6),
            "m_gap": round(m_gap, 6),
            "r_src": round(r_src, 6),
        }

    def compute_interest(
        self,
        user_vec: Optional[Sequence[float]],
        event_vec: Optional[Sequence[float]],
        cal_range: InterestCalibrationRange,
    ) -> Tuple[float, Optional[float]]:
        """Compute Clipped Min-Max Interest in [0, 1] from E5 cosine similarity."""
        int_cfg = self._scoring_cfg.get("interest") or {}
        cold_default = float(int_cfg.get("cold_start_default", 0.50))

        if user_vec is None or event_vec is None:
            return cold_default, None

        u = np.asarray(user_vec, dtype=np.float32)
        e = np.asarray(event_vec, dtype=np.float32)
        u_n = float(np.linalg.norm(u))
        e_n = float(np.linalg.norm(e))
        if u_n <= 1e-12 or e_n <= 1e-12:
            return cold_default, None

        cos_sim = float(np.dot(u, e) / (u_n * e_n))
        return scale_interest_clipped_minmax(cos_sim, cal_range), cos_sim

    def compute_urgency(
        self,
        effective_deadline: Optional[datetime],
        reference_now: datetime,
    ) -> Tuple[float, Optional[float]]:
        """Compute time-only Urgency in [0, 1]: `exp(-days_left / tau)` (no seat scarcity)."""
        urg_cfg = self._scoring_cfg.get("urgency") or {}
        tau = max(1e-6, float(urg_cfg.get("tau_days", 5.0)))
        missing_default = float(urg_cfg.get("missing_deadline_default", 0.20))

        if effective_deadline is None:
            return round(missing_default, 6), None

        delta_sec = (effective_deadline - reference_now).total_seconds()
        days_left = max(0.0, delta_sec / 86400.0)
        urg = math.exp(-days_left / tau)
        return round(max(0.0, min(1.0, urg)), 6), days_left

    def compute_penalty_mandatory(
        self,
        mandatory_info: MandatoryPenaltyInfo,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute additive Penalty intervention strictly for high-mandatory events."""
        if not mandatory_info.is_mandatory:
            return 0.0, {"is_mandatory": False, "boost_applied": 0.0}

        pen_cfg = self._scoring_cfg.get("penalty_mandatory") or {}
        ref_pts = max(1e-6, float(pen_cfg.get("penalty_ref_points", 20.0)))
        max_boost = max(0.0, float(pen_cfg.get("max_boost", 1.00)))
        default_boost = max(0.0, float(pen_cfg.get("default_mandatory_boost", 0.50)))

        if mandatory_info.penalty_if_missed is not None and abs(mandatory_info.penalty_if_missed) > 0:
            boost = min(max_boost, abs(float(mandatory_info.penalty_if_missed)) / ref_pts)
        else:
            boost = min(max_boost, default_boost)

        boost = round(boost, 6)
        return boost, {
            "is_mandatory": True,
            "penalty_if_missed": mandatory_info.penalty_if_missed,
            "boost_applied": boost,
            "reason": mandatory_info.reason,
        }

    @staticmethod
    def _level_tag(score: float) -> str:
        if score >= 0.75:
            return "cao"
        if score >= 0.50:
            return "khá"
        if score >= 0.25:
            return "trung bình"
        return "thấp"

    def _build_grounded_reasoning(
        self,
        *,
        scored_event: ScoredEvent,
        cand: EligibleCandidate,
        profile: DemoUserProfile,
        cal_range: InterestCalibrationRange,
        gain_details: Dict[str, Any],
        cos_raw: Optional[float],
        fit_res: FitAssessment,
        days_left: Optional[float],
        pen_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build strictly data-grounded, deterministic reasoning for each criterion & total score."""
        w = scored_event.weights
        ce = cand.credit_estimate
        mi = cand.mandatory_info
        total = scored_event.score_total

        wc_gain = round(w.w1_gain * scored_event.gain, 6)
        wc_int = round(w.w2_interest * scored_event.interest, 6)
        wc_fit = round(w.w3_fit * scored_event.fit, 6)
        wc_urg = round(w.w4_urgency * scored_event.urgency, 6)
        wc_pen = round(scored_event.penalty, 6)

        def _share(val: float) -> float:
            if total <= 1e-9:
                return 0.0
            return round((val / total) * 100.0, 1)

        # 1. Gain reasoning (grounded in Module 6 CreditEstimate + GroupShortfall)
        if mi.is_mandatory and ce.effective_points <= 0.0:
            gain_reason = (
                f"Mục {ce.rubric_item_id or '6.4'} thuộc nhóm P không cộng điểm (+0đ); "
                "giá trị ưu tiên được xử lý riêng ở cột Penalty bắt buộc."
            )
            gain_verdict = "không cộng điểm"
        elif ce.effective_points <= 0.0 or not ce.is_credit_eligible:
            gain_reason = f"Điểm thực nhận bằng 0đ (capped_by={ce.capped_by or 'ineligible'})."
            gain_verdict = "thấp"
        else:
            grp = ce.group_id or "?"
            ref = ce.rubric_item_id or "?"
            gap_pts = gain_details.get("group_gap_points", 0.0)
            grp_cap = gain_details.get("group_cap", 0.0)
            r_src = gain_details.get("r_src", 0.9)
            cap_note = f", bị chạm trần {ce.capped_by}" if ce.capped_by else ""
            gap_note = (
                f"nhóm {grp} còn thiếu {gap_pts:g}/{grp_cap:g}đ so với mục tiêu"
                if gap_pts > 0
                else f"nhóm {grp} đã đủ mục tiêu (nhưng chưa chạm trần {grp_cap:g}đ nên vẫn cộng được)"
            )
            gain_reason = (
                f"Cộng dự kiến +{ce.effective_points:g}đ vào mục {ref} ({grp}{cap_note}); "
                f"{gap_note}; nguồn {ce.source} (×{r_src:.2f})."
            )
            gain_verdict = self._level_tag(scored_event.gain)

        # 2. Interest reasoning (grounded in E5 cosine + 908-sample range)
        chip_labels = [
            str((self._interest_chips_cfg.get(cid) or {}).get("label", cid))
            for cid in (profile.interest_chips or [])
        ]
        pref_desc = ", ".join(chip_labels)
        if profile.free_text:
            pref_desc = f"{pref_desc} + '{profile.free_text}'" if pref_desc else f"'{profile.free_text}'"

        if cos_raw is None:
            int_reason = "Chưa khai báo sở thích (hoặc thiếu vector sự kiện) → áp mức trung tính 0.50."
            int_verdict = "trung tính"
        else:
            clip_note = ""
            if cos_raw <= cal_range.c_min:
                clip_note = " (bị chặn dưới về 0.00)"
            elif cos_raw >= cal_range.c_max:
                clip_note = " (chạm trần 1.00)"
            int_verdict = self._level_tag(scored_event.interest)
            target_txt = f" với sở thích [{pref_desc}]" if pref_desc else ""
            int_reason = (
                f"Tương đồng E5 đạt cos={cos_raw:.4f}{target_txt}, ở mức {int_verdict} trong dải "
                f"[{cal_range.c_min:.3f}, {cal_range.c_max:.3f}] của {cal_range.sample_count} mẫu{clip_note}."
            )

        # 3. Fit reasoning (grounded in Module 4 FitAssessment & sub_scores)
        sub = fit_res.sub_scores if isinstance(fit_res, DemoFitAssessment) else {}
        slack_min = fit_res.nearest_slack_min if isinstance(fit_res, DemoFitAssessment) else None
        fit_parts: List[str] = []
        if cand.event.online_mode == "online" or cand.event.is_online is True:
            fit_parts.append("trực tuyến (0 phút di chuyển)")
        else:
            t_max = profile.max_travel_min or 45
            cmp_txt = "≤" if fit_res.estimated_travel_min <= t_max else ">"
            fit_parts.append(f"di chuyển ~{fit_res.estimated_travel_min}p ({cmp_txt} {t_max}p)")
            if slack_min is None:
                fit_parts.append("không có tiết kề cận trong ngày")
            else:
                fit_parts.append(f"cách tiết gần nhất {slack_min}p")
        f_pref_val = sub.get("f_pref")
        if f_pref_val is not None:
            if f_pref_val >= 0.99:
                fit_parts.append("trong khung giờ rảnh")
            elif f_pref_val >= 0.65:
                fit_parts.append("lệch nhẹ ≤1h so với khung giờ rảnh")
            elif abs(f_pref_val - 0.5) < 1e-6:
                fit_parts.append("thiếu giờ diễn ra (trung tính)")
            else:
                fit_parts.append("ngoài khung giờ rảnh")
        if fit_res.in_exam_week:
            f_exam_val = sub.get("f_exam", 1.0)
            fit_parts.append(f"diễn ra trong tuần thi (×{f_exam_val:.2f})")
        else:
            fit_parts.append("ngoài tuần thi")
        fit_reason = "Không trùng lịch; " + "; ".join(fit_parts) + "."
        fit_verdict = self._level_tag(scored_event.fit)

        # 4. Urgency reasoning (grounded in time-only decay exp(-d / tau))
        tau = float((self._scoring_cfg.get("urgency") or {}).get("tau_days", 5.0))
        if days_left is None:
            urg_reason = f"Thiếu cả hạn đăng ký lẫn giờ bắt đầu → áp mức mặc định {scored_event.urgency:.2f}."
            urg_verdict = "trung tính"
        else:
            dl_txt = (
                cand.effective_deadline.strftime("%d/%m/%Y %H:%M")
                if cand.effective_deadline is not None
                else "—"
            )
            if days_left <= 2.0:
                urg_stage = "rất gấp"
            elif days_left <= 7.0:
                urg_stage = "sắp tới hạn"
            else:
                urg_stage = "còn khá xa"
            urg_reason = (
                f"Còn {days_left:.1f} ngày đến hạn ({dl_txt}) — {urg_stage}; "
                f"tính thuần thời gian exp(-{days_left:.1f}/{tau:g}) = {scored_event.urgency:.3f}."
            )
            urg_verdict = self._level_tag(scored_event.urgency)

        # 5. Penalty reasoning (grounded in mandatory flag only)
        if mi.is_mandatory and scored_event.penalty > 0.0:
            pen_reason = (
                f"{mi.reason} → cộng thẳng +{scored_event.penalty:.3f} (không nhân trọng số) "
                "để ưu tiên tránh bị trừ điểm ĐRL."
            )
            pen_verdict = "bắt buộc"
        else:
            pen_reason = "Sự kiện thường (không bắt buộc) → không can thiệp (+0.000)."
            pen_verdict = "không áp dụng"

        criteria = [
            {
                "key": "gain",
                "label": "Gain (Điểm ĐRL)",
                "raw_score": scored_event.gain,
                "weight": w.w1_gain,
                "weighted_score": wc_gain,
                "share_pct": _share(wc_gain),
                "verdict": gain_verdict,
                "reason": gain_reason,
            },
            {
                "key": "interest",
                "label": "Interest (Sở thích)",
                "raw_score": scored_event.interest,
                "weight": w.w2_interest,
                "weighted_score": wc_int,
                "share_pct": _share(wc_int),
                "verdict": int_verdict,
                "reason": int_reason,
            },
            {
                "key": "fit",
                "label": "Fit (Lịch & Di chuyển)",
                "raw_score": scored_event.fit,
                "weight": w.w3_fit,
                "weighted_score": wc_fit,
                "share_pct": _share(wc_fit),
                "verdict": fit_verdict,
                "reason": fit_reason,
            },
            {
                "key": "urgency",
                "label": "Urgency (Độ gấp)",
                "raw_score": scored_event.urgency,
                "weight": w.w4_urgency,
                "weighted_score": wc_urg,
                "share_pct": _share(wc_urg),
                "verdict": urg_verdict,
                "reason": urg_reason,
            },
            {
                "key": "penalty",
                "label": "Penalty (Bắt buộc)",
                "raw_score": scored_event.penalty,
                "weight": None,
                "weighted_score": wc_pen,
                "share_pct": _share(wc_pen),
                "verdict": pen_verdict,
                "reason": pen_reason,
            },
        ]

        # Concise grounded headline explaining the top 1-2 actual score drivers
        if mi.is_mandatory and wc_pen > 0.0:
            headline = (
                f"Ưu tiên hàng đầu nhờ can thiệp Bắt buộc (+{wc_pen:.3f}, chiếm {_share(wc_pen):.0f}% điểm) "
                f"để tránh bị trừ {abs(mi.penalty_if_missed or 20):.0f}đ ĐRL."
            )
        else:
            ranked_drivers = sorted(
                [c for c in criteria if c["key"] != "penalty" and c["weighted_score"] > 0.0],
                key=lambda x: -x["weighted_score"],
            )
            if len(ranked_drivers) >= 2:
                d1, d2 = ranked_drivers[0], ranked_drivers[1]
                headline = (
                    f"Động lực chính ({profile.weight_preset}): "
                    f"{d1['label'].split(' ')[0]} (+{d1['weighted_score']:.3f} · {d1['share_pct']:.0f}%) "
                    f"và {d2['label'].split(' ')[0]} (+{d2['weighted_score']:.3f} · {d2['share_pct']:.0f}%)."
                )
            elif len(ranked_drivers) == 1:
                d1 = ranked_drivers[0]
                headline = (
                    f"Động lực chính ({profile.weight_preset}): "
                    f"{d1['label'].split(' ')[0]} (+{d1['weighted_score']:.3f} · {d1['share_pct']:.0f}%)."
                )
            else:
                headline = "Tất cả thành phần điểm đều bằng 0."

        formula_check = (
            f"({w.w1_gain:.2f}×{scored_event.gain:.3f}) + "
            f"({w.w2_interest:.2f}×{scored_event.interest:.3f}) + "
            f"({w.w3_fit:.2f}×{scored_event.fit:.3f}) + "
            f"({w.w4_urgency:.2f}×{scored_event.urgency:.3f}) + "
            f"{scored_event.penalty:.3f} = {total:.3f}"
        )

        return {
            "headline": headline,
            "formula_check": formula_check,
            "criteria": criteria,
        }

    def score_candidates(
        self,
        eligible_candidates: Sequence[EligibleCandidate],
        *,
        profile: DemoUserProfile,
        schedule_provider: ScheduleProvider,
        credit_provider: DemoCreditProvider,
        shortfall: GroupShortfall,
        sample_embeddings: Optional[np.ndarray],
        reference_now: Optional[datetime] = None,
        interest_mode_override: Optional[str] = None,
    ) -> Tuple[List[ScoredCandidate], InterestCalibrationRange]:
        ref_now = (
            parse_reference_now(reference_now, self._default_tz)
            if reference_now is not None
            else parse_reference_now(profile.reference_now, self._default_tz)
        )

        int_cfg = self._scoring_cfg.get("interest") or {}
        cal_mode = interest_mode_override or str(int_cfg.get("calibration_mode", "per_user_samples_minmax"))
        cal_range = compute_interest_calibration_range(
            user_vec=profile.user_vec,
            sample_embeddings=sample_embeddings,
            mode=cal_mode,
            fixed_c_min=float(int_cfg.get("fixed_c_min", 0.72)),
            fixed_c_max=float(int_cfg.get("fixed_c_max", 0.90)),
        )

        weights: ScoringWeights = profile.weights
        group_caps = credit_provider.group_caps
        scored_list: List[ScoredCandidate] = []

        for cand in eligible_candidates:
            gain, gain_details = self.compute_gain(cand.credit_estimate, shortfall, group_caps)
            interest, cos_raw = self.compute_interest(profile.user_vec, cand.event_vec, cal_range)

            if isinstance(schedule_provider, DemoScheduleProvider):
                fit_res: FitAssessment = schedule_provider.compute_fit(
                    cand.event,
                    home_campus=profile.home_campus,
                    max_travel_min=profile.max_travel_min,
                    pref_hours=profile.pref_hours,
                )
            else:
                fit_res = schedule_provider.compute_fit(
                    cand.event,
                    home_campus=profile.home_campus,
                    max_travel_min=profile.max_travel_min,
                )
            fit = round(max(0.0, min(1.0, float(fit_res.fit_score))), 6)

            urgency, days_left = self.compute_urgency(cand.effective_deadline, ref_now)
            penalty, pen_details = self.compute_penalty_mandatory(cand.mandatory_info)

            score_total = round(
                weights.w1_gain * gain
                + weights.w2_interest * interest
                + weights.w3_fit * fit
                + weights.w4_urgency * urgency
                + penalty,
                6,
            )

            se = ScoredEvent(
                event_id=cand.event.event_id,
                score_total=score_total,
                gain=gain,
                interest=interest,
                fit=fit,
                urgency=urgency,
                penalty=penalty,
                weights=weights,
                rubric_item_id=cand.rubric_item_id,
                parent_group_id=cand.parent_group_id,
            )
            reasoning = self._build_grounded_reasoning(
                scored_event=se,
                cand=cand,
                profile=profile,
                cal_range=cal_range,
                gain_details=gain_details,
                cos_raw=cos_raw,
                fit_res=fit_res,
                days_left=days_left,
                pen_details=pen_details,
            )
            scored_list.append(
                ScoredCandidate(
                    scored_event=se,
                    candidate=cand,
                    fit_assessment=fit_res,
                    interest_cos_raw=cos_raw,
                    days_to_deadline=days_left,
                    gain_details=gain_details,
                    penalty_details=pen_details,
                    reasoning=reasoning,
                )
            )

        scored_list.sort(
            key=lambda x: (
                -x.scored_event.score_total,
                -x.scored_event.gain,
                -x.scored_event.urgency,
                x.scored_event.event_id,
            )
        )
        return scored_list, cal_range
