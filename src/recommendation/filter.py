"""P5 Hard Constraint Filter (`src/recommendation/filter.py`).

Enforces:
1. Strict Classification Gate: accepts ONLY `classification.status == "PREDICTED"`
   (excluding `NEEDS_REVIEW`, `NO_MATCH`, `SKIPPED_NOT_EVENT`).
2. Non-short-circuiting evaluation of all 8 hard constraints (F1..F8), recording
   every violated `DropReason` for transparent "Why not?" diagnostics.
3. Explicit mandatory-event exceptions (e.g. item `6.4` Tuần Sinh hoạt Công dân)
   so high-mandatory activities pass P5 even when `effective_points == 0`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from module6.interface import UserLedger
from src.contract_validator import parse_iso_datetime
from src.recommendation.config import load_recommendation_config
from src.recommendation.demo_module4 import DemoScheduleProvider
from src.recommendation.demo_module6 import DemoCreditProvider, MandatoryPenaltyInfo
from src.recommendation.demo_profile import DemoUserProfile
from src.recommendation.schedule_adapter import ConflictCheckResult, ScheduleProvider
from src.schemas import CreditEstimate, DropReason, EventNormalized

DEFAULT_TZ = timezone(timedelta(hours=7))


@dataclass(frozen=True)
class RecommendationCandidateInput:
    """Single event paired with its P2 classification payload and 384d E5 query vector."""

    event: EventNormalized
    classification: Dict[str, Any]
    event_vec: Optional[List[float]] = None


@dataclass
class EligibleCandidate:
    """Event that survived P5, carrying precomputed M4/M6 metadata for P6."""

    event: EventNormalized
    classification: Dict[str, Any]
    event_vec: Optional[List[float]]
    rubric_item_id: Optional[str]
    parent_group_id: Optional[str]
    credit_estimate: CreditEstimate
    mandatory_info: MandatoryPenaltyInfo
    conflict_result: ConflictCheckResult
    estimated_travel_min: int
    effective_deadline: Optional[datetime]


@dataclass
class DroppedCandidate:
    """Event filtered out by P5 together with ALL violated `DropReason`s."""

    event: EventNormalized
    classification: Dict[str, Any]
    rubric_item_id: Optional[str]
    parent_group_id: Optional[str]
    reasons: List[DropReason] = field(default_factory=list)
    credit_estimate: Optional[CreditEstimate] = None
    estimated_travel_min: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event.event_id,
            "title": self.event.title,
            "description": self.event.description,
            "organizer": self.event.organizer,
            "start_datetime": self.event.start_datetime,
            "reg_end_datetime": self.event.reg_end_datetime,
            "online_mode": self.event.online_mode,
            "location": self.event.location,
            "fee_vnd": self.event.fee_vnd,
            "rubric_item_id": self.rubric_item_id,
            "parent_group_id": self.parent_group_id,
            "estimated_travel_min": self.estimated_travel_min,
            "effective_points": (
                self.credit_estimate.effective_points if self.credit_estimate is not None else None
            ),
            "capped_by": self.credit_estimate.capped_by if self.credit_estimate is not None else None,
            "reasons": [
                {"code": r.code, "message": r.message, "details": dict(r.details)}
                for r in self.reasons
            ],
        }


@dataclass
class FilterReport:
    """Complete output of P5 Hard Constraint Filtering."""

    total_input: int
    predicted_input: int
    excluded_not_predicted: int
    excluded_status_counts: Dict[str, int]
    eligible: List[EligibleCandidate]
    dropped: List[DroppedCandidate]
    reason_counts: Dict[str, int]


def _ensure_aware(dt: datetime, tz: timezone = DEFAULT_TZ) -> datetime:
    return dt.replace(tzinfo=tz) if dt.tzinfo is None else dt


def parse_reference_now(raw: Any, default_tz: timezone = DEFAULT_TZ) -> datetime:
    if isinstance(raw, datetime):
        return _ensure_aware(raw, default_tz)
    if isinstance(raw, str) and raw.strip():
        text = raw.strip()
        if len(text) == 10 and text[4] == "-" and text[7] == "-":
            text = f"{text}T08:00:00+07:00"
        parsed = parse_iso_datetime(text)
        if parsed is not None:
            return _ensure_aware(parsed, default_tz)
    return datetime.now(tz=default_tz)


def resolve_effective_deadline(
    event: EventNormalized,
    default_tz: timezone = DEFAULT_TZ,
) -> Optional[datetime]:
    """Deterministically compute effective deadline handling missing & inverted dates.

    Handles real fixture quirks (`47` missing `reg_end_datetime`, `16` `reg_end_after_start`):
    - If both `reg_end_datetime` and `start_datetime` exist:
      - If `reg_end <= start`: return `reg_end`.
      - If `reg_end > start`:
        - If multi-day event and `end_datetime` exists with `reg_end <= end_datetime`: return `reg_end`.
        - Otherwise return `start_datetime` (guards against registration-date extraction noise).
    - If `reg_end_datetime` missing: fallback to `start_datetime`.
    - If both missing: return `None`.
    """
    reg_end: Optional[datetime] = None
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None

    if event.reg_end_datetime:
        try:
            parsed = parse_iso_datetime(event.reg_end_datetime)
            if parsed is not None:
                reg_end = _ensure_aware(parsed, default_tz)
        except ValueError:
            reg_end = None

    if event.start_datetime:
        try:
            parsed = parse_iso_datetime(event.start_datetime)
            if parsed is not None:
                start_dt = _ensure_aware(parsed, default_tz)
        except ValueError:
            start_dt = None

    if event.end_datetime:
        try:
            parsed = parse_iso_datetime(event.end_datetime)
            if parsed is not None:
                end_dt = _ensure_aware(parsed, default_tz)
        except ValueError:
            end_dt = None

    if reg_end is not None and start_dt is not None:
        if reg_end <= start_dt:
            return reg_end
        if end_dt is not None and end_dt.date() > start_dt.date() and reg_end <= end_dt:
            return reg_end
        return start_dt

    return reg_end if reg_end is not None else start_dt


def _normalize_cohort(raw: str) -> str:
    """Normalize cohort labels so 'K26', 'KHOA2026', 'Khóa 2026', '26' compare consistently."""
    cleaned = re.sub(r"[^a-z0-9]", "", str(raw or "").strip().lower())
    m = re.search(r"(\d{2,4})", cleaned)
    if not m:
        return cleaned
    digits = m.group(1)
    return digits[-2:] if len(digits) == 4 else digits


def _normalize_text_token(raw: str) -> str:
    return re.sub(r"\s+", " ", str(raw or "").strip().lower())


def _is_online_event(event: EventNormalized) -> bool:
    if event.online_mode is not None:
        return event.online_mode == "online"
    return event.is_online is True


class HardConstraintFilter:
    """P5 Hard Constraint Filter with strict `PREDICTED`-only gate and full `DropReason` reporting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config if config is not None else load_recommendation_config()
        self._filter_cfg: Dict[str, Any] = dict(cfg.get("filter") or {})
        sched_cfg = cfg.get("schedule") or {}
        tz_hours = int(sched_cfg.get("default_timezone_offset_hours", 7))
        self._default_tz = timezone(timedelta(hours=tz_hours))

    def filter_candidates(
        self,
        candidates: Sequence[RecommendationCandidateInput],
        *,
        profile: DemoUserProfile,
        schedule_provider: ScheduleProvider,
        credit_provider: DemoCreditProvider,
        ledger: UserLedger,
        reference_now: Optional[datetime] = None,
    ) -> FilterReport:
        ref_now = (
            parse_reference_now(reference_now, self._default_tz)
            if reference_now is not None
            else parse_reference_now(profile.reference_now, self._default_tz)
        )

        require_predicted = bool(self._filter_cfg.get("require_predicted_status", True))
        mandatory_bypass_zero_gain = bool(self._filter_cfg.get("mandatory_bypass_zero_gain", True))
        mandatory_bypass_budget = bool(self._filter_cfg.get("mandatory_bypass_budget", True))
        mandatory_bypass_travel = bool(self._filter_cfg.get("mandatory_bypass_travel", True))
        mandatory_bypass_hidden = bool(self._filter_cfg.get("mandatory_bypass_hidden", True))
        allow_zero_in_interest = bool(self._filter_cfg.get("allow_zero_gain_in_interest_mode", False))

        participated_set = set(profile.participated_event_ids or [])
        hidden_set = set(profile.hidden_event_ids or [])

        excluded_not_predicted = 0
        excluded_status_counts: Dict[str, int] = {}
        eligible: List[EligibleCandidate] = []
        dropped: List[DroppedCandidate] = []
        reason_counts: Dict[str, int] = {}

        for item in candidates:
            cls = item.classification or {}
            status = str(cls.get("status") or "UNKNOWN")
            if require_predicted and status != "PREDICTED":
                excluded_not_predicted += 1
                excluded_status_counts[status] = excluded_status_counts.get(status, 0) + 1
                continue

            ev = item.event
            rubric_item_id = cls.get("predicted_criterion_ref")
            parent_group_id = cls.get("predicted_group_id")

            # Evaluate M6 credit & mandatory status
            credit = credit_provider.estimate_event_credit(
                event=ev,
                rubric_item_id=rubric_item_id,
                ledger=ledger,
                role="participant",
            )
            mandatory_info = credit_provider.get_mandatory_penalty_info(ev, credit)

            # Evaluate M4 conflict & travel
            conflict = schedule_provider.check_conflict(ev)
            travel_min = int(
                schedule_provider.estimate_travel_minutes(ev, home_campus=profile.home_campus)
            )
            eff_deadline = resolve_effective_deadline(ev, self._default_tz)

            reasons: List[DropReason] = []

            # F1: already_participated
            if ev.event_id in participated_set:
                reasons.append(
                    DropReason(
                        code="already_participated",
                        message="Bạn đã đánh dấu tham gia hoạt động này.",
                        details={"event_id": ev.event_id},
                    )
                )

            # F2: hidden_by_user (mandatory events bypass unless disabled)
            if ev.event_id in hidden_set and not (mandatory_info.is_mandatory and mandatory_bypass_hidden):
                reasons.append(
                    DropReason(
                        code="hidden_by_user",
                        message="Hoạt động nằm trong danh sách bạn đã ẩn.",
                        details={"event_id": ev.event_id},
                    )
                )

            # F3: expired (check effective deadline & end_datetime against reference_now)
            end_dt_parsed: Optional[datetime] = None
            if ev.end_datetime:
                try:
                    p_end = parse_iso_datetime(ev.end_datetime)
                    if p_end is not None:
                        end_dt_parsed = _ensure_aware(p_end, self._default_tz)
                except ValueError:
                    end_dt_parsed = None

            if eff_deadline is not None and eff_deadline < ref_now:
                reasons.append(
                    DropReason(
                        code="expired",
                        message=f"Đã quá hạn đăng ký/diễn ra ({eff_deadline.strftime('%d/%m/%Y %H:%M')}).",
                        details={
                            "effective_deadline": eff_deadline.isoformat(),
                            "reference_now": ref_now.isoformat(),
                        },
                    )
                )
            elif end_dt_parsed is not None and end_dt_parsed < ref_now:
                reasons.append(
                    DropReason(
                        code="expired",
                        message=f"Sự kiện đã kết thúc ({end_dt_parsed.strftime('%d/%m/%Y %H:%M')}).",
                        details={
                            "end_datetime": end_dt_parsed.isoformat(),
                            "reference_now": ref_now.isoformat(),
                        },
                    )
                )

            # F4: audience_mismatch (conservative: only filter when structured list is non-empty)
            ta = ev.target_audience
            if ta is not None and not ta.all_students:
                mismatch_parts: List[str] = []
                if ta.cohorts and profile.cohort:
                    ev_cohorts = {_normalize_cohort(c) for c in ta.cohorts if str(c).strip()}
                    u_cohort = _normalize_cohort(profile.cohort)
                    if ev_cohorts and u_cohort not in ev_cohorts:
                        mismatch_parts.append(f"khóa yêu cầu {ta.cohorts}, bạn là {profile.cohort}")
                if ta.faculties and profile.faculty:
                    ev_facs = {_normalize_text_token(f) for f in ta.faculties if str(f).strip()}
                    u_fac = _normalize_text_token(profile.faculty)
                    if ev_facs and u_fac not in ev_facs:
                        mismatch_parts.append(f"khoa yêu cầu {ta.faculties}, bạn là {profile.faculty}")
                if ta.classes and profile.class_id:
                    ev_cls = {_normalize_text_token(c) for c in ta.classes if str(c).strip()}
                    u_cls = _normalize_text_token(profile.class_id)
                    if ev_cls and u_cls not in ev_cls:
                        mismatch_parts.append(f"lớp yêu cầu {ta.classes}, bạn là {profile.class_id}")
                if mismatch_parts:
                    reasons.append(
                        DropReason(
                            code="audience_mismatch",
                            message="Không đúng đối tượng tham gia (" + "; ".join(mismatch_parts) + ").",
                            details={
                                "event_cohorts": list(ta.cohorts),
                                "event_faculties": list(ta.faculties),
                                "event_classes": list(ta.classes),
                                "user_cohort": profile.cohort,
                                "user_faculty": profile.faculty,
                                "user_class": profile.class_id,
                            },
                        )
                    )

            # F5: schedule_conflict (Demo Module 4)
            if conflict.has_conflict:
                reasons.append(
                    DropReason(
                        code="schedule_conflict",
                        message=conflict.overlap_detail or "Trùng lịch học / lịch thi (Module 4).",
                        details={
                            "conflict_type": conflict.conflict_type,
                            "conflicting_label": conflict.conflicting_label,
                            "travel_buffer_min": conflict.travel_buffer_min,
                        },
                    )
                )

            # F6: over_budget (mandatory events bypass when configured)
            if (
                profile.max_budget_vnd is not None
                and ev.fee_vnd is not None
                and ev.fee_vnd > profile.max_budget_vnd
                and not (mandatory_info.is_mandatory and mandatory_bypass_budget)
            ):
                reasons.append(
                    DropReason(
                        code="over_budget",
                        message=(
                            f"Lệ phí {ev.fee_vnd:,}đ vượt ngân sách tối đa "
                            f"{profile.max_budget_vnd:,}đ."
                        ),
                        details={
                            "fee_vnd": ev.fee_vnd,
                            "max_budget_vnd": profile.max_budget_vnd,
                        },
                    )
                )

            # F7: too_far (when strict_max_travel is enabled; mandatory events bypass)
            if (
                profile.strict_max_travel
                and not _is_online_event(ev)
                and profile.max_travel_min is not None
                and travel_min > profile.max_travel_min
                and not (mandatory_info.is_mandatory and mandatory_bypass_travel)
            ):
                reasons.append(
                    DropReason(
                        code="too_far",
                        message=(
                            f"Thời gian di chuyển (~{travel_min} phút) vượt giới hạn "
                            f"{profile.max_travel_min} phút."
                        ),
                        details={
                            "estimated_travel_min": travel_min,
                            "max_travel_min": profile.max_travel_min,
                        },
                    )
                )

            # F8: no_credit_gain (mandatory events ALWAYS pass when configured)
            zero_gain = (not credit.is_credit_eligible) or (credit.effective_points <= 0.0)
            interest_bypass = allow_zero_in_interest and profile.weight_preset == "interest_first"
            mandatory_bypass = mandatory_info.is_mandatory and mandatory_bypass_zero_gain
            if zero_gain and not mandatory_bypass and not interest_bypass:
                cap_msg = (
                    f"Nhóm {credit.group_id} đã chạm trần điểm ĐRL."
                    if credit.capped_by == "group"
                    else "Tổng điểm ĐRL đã chạm trần 100."
                    if credit.capped_by == "total"
                    else f"Không còn điểm cộng khả dụng (lý do: {credit.capped_by or 'ineligible'})."
                )
                reasons.append(
                    DropReason(
                        code="no_credit_gain",
                        message=cap_msg,
                        details={
                            "group_id": credit.group_id,
                            "rubric_item_id": credit.rubric_item_id,
                            "expected_points": credit.expected_points,
                            "effective_points": credit.effective_points,
                            "capped_by": credit.capped_by,
                        },
                    )
                )

            if reasons:
                for r in reasons:
                    reason_counts[r.code] = reason_counts.get(r.code, 0) + 1
                dropped.append(
                    DroppedCandidate(
                        event=ev,
                        classification=cls,
                        rubric_item_id=rubric_item_id,
                        parent_group_id=parent_group_id,
                        reasons=reasons,
                        credit_estimate=credit,
                        estimated_travel_min=travel_min,
                    )
                )
            else:
                eligible.append(
                    EligibleCandidate(
                        event=ev,
                        classification=cls,
                        event_vec=item.event_vec,
                        rubric_item_id=rubric_item_id,
                        parent_group_id=parent_group_id,
                        credit_estimate=credit,
                        mandatory_info=mandatory_info,
                        conflict_result=conflict,
                        estimated_travel_min=travel_min,
                        effective_deadline=eff_deadline,
                    )
                )

        predicted_input = len(candidates) - excluded_not_predicted
        return FilterReport(
            total_input=len(candidates),
            predicted_input=predicted_input,
            excluded_not_predicted=excluded_not_predicted,
            excluded_status_counts=excluded_status_counts,
            eligible=eligible,
            dropped=dropped,
            reason_counts=reason_counts,
        )
