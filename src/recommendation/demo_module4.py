"""Demo Module 4 (`DemoScheduleProvider`) — Replaceable Schedule & Fit Engine.

Implements the `ScheduleProvider` protocol and the 5-step continuous `compute_fit`
specification handed off to Module 4 (`docs/module4_fit_handoff_spec.md`):
- Hard conflict detection (`check_conflict`) with travel buffer & exam-week awareness
- Campus/configurable travel-time estimation (`estimate_travel_minutes`)
- Nearest same-day schedule slack (`nearest_slack_minutes`)
- Continuous 5-step `compute_fit` returning `DemoFitAssessment` (subclass of `FitAssessment`)
- Ready-to-use demo timetable presets (`light_schedule`, `busy_weekend`, `exam_week_heavy`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.contract_validator import parse_iso_datetime
from src.recommendation.config import load_recommendation_config
from src.recommendation.schedule_adapter import (
    ConflictCheckResult,
    FitAssessment,
    LocalScheduleAdapter,
    WeeklyBusySlot,
    _ensure_aware,
    _parse_hour_value,
    _parse_optional_date,
    detect_campus,
)
from src.schemas import BusyInterval, EventNormalized, ScheduleContext


@dataclass(frozen=True)
class DemoFitAssessment(FitAssessment):
    """Extended `FitAssessment` carrying `nearest_slack_min` and `sub_scores` per M4 handoff spec."""

    nearest_slack_min: Optional[int] = None
    sub_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fit_score": self.fit_score,
            "has_conflict": self.has_conflict,
            "in_exam_week": self.in_exam_week,
            "estimated_travel_min": self.estimated_travel_min,
            "nearest_slack_min": self.nearest_slack_min,
            "conflict_detail": self.conflict_detail,
            "sub_scores": dict(self.sub_scores),
            "reasons": list(self.reasons),
        }


DEMO_SCHEDULE_PRESETS: Dict[str, Dict[str, Any]] = {
    "light_schedule": {
        "id": "light_schedule",
        "label": "Lịch học nhẹ (Sáng T2 & T4)",
        "description": "Rảnh phần lớn các buổi chiều và cuối tuần, không nằm trong tuần thi.",
        "weekly_slots": [
            {
                "day_of_week": 0,
                "start_hour": 7.5,
                "end_hour": 11.5,
                "label": "Cấu trúc dữ liệu & Giải thuật",
                "location": "cs1_nvc",
                "is_exam": False,
            },
            {
                "day_of_week": 2,
                "start_hour": 7.5,
                "end_hour": 11.5,
                "label": "Cơ sở dữ liệu",
                "location": "cs1_nvc",
                "is_exam": False,
            },
        ],
        "busy_intervals": [],
        "exam_weeks": [],
        "default_travel_buffer_min": 30,
    },
    "busy_weekend": {
        "id": "busy_weekend",
        "label": "Bận cuối tuần & chiều Thứ 2",
        "description": "Bận chiều Thứ 7 (12:00–17:00), sáng Chủ Nhật (08:00–12:00) và chiều Thứ 2 (13:00–17:00).",
        "weekly_slots": [
            {
                "day_of_week": 0,
                "start_hour": 13.0,
                "end_hour": 17.0,
                "label": "Thực hành Mạng máy tính",
                "location": "cs1_nvc",
                "is_exam": False,
            },
            {
                "day_of_week": 5,
                "start_hour": 12.0,
                "end_hour": 17.0,
                "label": "Lớp chuyên đề chiều Thứ 7",
                "location": "cs1_nvc",
                "is_exam": False,
            },
            {
                "day_of_week": 6,
                "start_hour": 8.0,
                "end_hour": 12.0,
                "label": "Học nhóm / Làm thêm sáng CN",
                "location": "cs1_nvc",
                "is_exam": False,
            },
        ],
        "busy_intervals": [],
        "exam_weeks": [],
        "default_travel_buffer_min": 30,
    },
    "exam_week_heavy": {
        "id": "exam_week_heavy",
        "label": "Tuần thi (09/11–22/11) + lịch dày",
        "description": "Đang trong tuần thi giữa kỳ 09/11–22/11/2026 kèm ca thi chiều 16/11 và lớp chiều Thứ 7.",
        "weekly_slots": [
            {
                "day_of_week": 1,
                "start_hour": 13.0,
                "end_hour": 16.5,
                "label": "Hệ điều hành",
                "location": "cs1_nvc",
                "is_exam": False,
            },
            {
                "day_of_week": 5,
                "start_hour": 12.0,
                "end_hour": 17.0,
                "label": "Ôn thi tập trung chiều T7",
                "location": "cs1_nvc",
                "is_exam": False,
            },
        ],
        "busy_intervals": [
            {
                "start_datetime": "2026-11-16T13:00:00+07:00",
                "end_datetime": "2026-11-16T15:00:00+07:00",
                "label": "Thi giữa kỳ Nhập môn AI",
                "location": "cs1_nvc",
                "is_exam": True,
            },
        ],
        "exam_weeks": [["2026-11-09", "2026-11-22"]],
        "default_travel_buffer_min": 30,
    },
}


class DemoScheduleProvider(LocalScheduleAdapter):
    """Demo Module 4 provider implementing the exact 5-step continuous `compute_fit` spec."""

    def __init__(
        self,
        context: Optional[ScheduleContext] = None,
        weekly_slots: Sequence[WeeklyBusySlot] = (),
        config: Optional[Dict[str, Any]] = None,
        travel_overrides: Optional[Dict[str, int]] = None,
        preset_id: str = "custom",
    ) -> None:
        cfg = config if config is not None else load_recommendation_config()
        super().__init__(context=context, weekly_slots=weekly_slots, config=cfg)
        self._m4_cfg: Dict[str, Any] = dict(cfg.get("demo_module4") or {})
        self._travel_overrides: Dict[str, int] = dict(travel_overrides or {})
        self.preset_id = preset_id

    @classmethod
    def from_preset(
        cls,
        preset_id: str = "light_schedule",
        *,
        config: Optional[Dict[str, Any]] = None,
        travel_overrides: Optional[Dict[str, int]] = None,
    ) -> "DemoScheduleProvider":
        if preset_id not in DEMO_SCHEDULE_PRESETS:
            raise ValueError(
                f"Unknown demo schedule preset '{preset_id}'. "
                f"Available: {sorted(DEMO_SCHEDULE_PRESETS.keys())}"
            )
        return cls.from_demo_payload(
            DEMO_SCHEDULE_PRESETS[preset_id],
            config=config,
            travel_overrides=travel_overrides,
            preset_id=preset_id,
        )

    @classmethod
    def from_demo_payload(
        cls,
        payload: Dict[str, Any],
        *,
        config: Optional[Dict[str, Any]] = None,
        travel_overrides: Optional[Dict[str, int]] = None,
        preset_id: Optional[str] = None,
    ) -> "DemoScheduleProvider":
        cfg = config if config is not None else load_recommendation_config()
        sched_cfg = cfg.get("schedule") or {}

        # Allow starting from a preset and optionally overriding fields
        base_preset_id = str(payload.get("preset_id") or payload.get("id") or preset_id or "custom")
        if (
            base_preset_id in DEMO_SCHEDULE_PRESETS
            and "weekly_slots" not in payload
            and "exam_weeks" not in payload
            and "busy_intervals" not in payload
        ):
            payload = dict(DEMO_SCHEDULE_PRESETS[base_preset_id])

        default_buf = int(
            payload.get("default_travel_buffer_min", sched_cfg.get("default_travel_buffer_min", 30))
        )

        busy_intervals: List[BusyInterval] = []
        for raw in payload.get("busy_intervals") or []:
            if not raw.get("start_datetime") or not raw.get("end_datetime"):
                continue
            busy_intervals.append(
                BusyInterval(
                    start_datetime=str(raw["start_datetime"]),
                    end_datetime=str(raw["end_datetime"]),
                    label=str(raw.get("label") or ""),
                    location=raw.get("location"),
                    is_exam=bool(raw.get("is_exam", False)),
                )
            )

        exam_weeks: List[Tuple[str, str]] = []
        for pair in payload.get("exam_weeks") or []:
            if isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[0] and pair[1]:
                exam_weeks.append((str(pair[0]), str(pair[1])))
            elif isinstance(pair, dict) and pair.get("start") and pair.get("end"):
                exam_weeks.append((str(pair["start"]), str(pair["end"])))

        weekly_slots: List[WeeklyBusySlot] = []
        for raw in payload.get("weekly_slots") or []:
            dow = int(raw.get("day_of_week", raw.get("day", 0)))
            start_h = _parse_hour_value(raw.get("start_hour", raw.get("start", 8.0)))
            end_h = _parse_hour_value(raw.get("end_hour", raw.get("end", start_h + 2.0)))
            if end_h <= start_h:
                continue
            weekly_slots.append(
                WeeklyBusySlot(
                    day_of_week=dow % 7,
                    start_hour=start_h,
                    end_hour=end_h,
                    label=str(raw.get("label") or "Lịch bận"),
                    location=raw.get("location"),
                    is_exam=bool(raw.get("is_exam", False)),
                    valid_from=raw.get("valid_from"),
                    valid_until=raw.get("valid_until"),
                )
            )

        overrides = dict(travel_overrides or {})
        if isinstance(payload.get("travel_overrides"), dict):
            for k, v in payload["travel_overrides"].items():
                if v is not None:
                    overrides[str(k)] = max(0, int(v))

        context = ScheduleContext(
            busy_intervals=busy_intervals,
            exam_weeks=exam_weeks,
            default_travel_buffer_min=default_buf,
        )
        return cls(
            context=context,
            weekly_slots=weekly_slots,
            config=cfg,
            travel_overrides=overrides,
            preset_id=base_preset_id,
        )

    @staticmethod
    def available_presets() -> List[Dict[str, Any]]:
        return [dict(v) for v in DEMO_SCHEDULE_PRESETS.values()]

    def estimate_travel_minutes(
        self,
        event: EventNormalized,
        home_campus: Optional[str] = None,
    ) -> int:
        if self._is_online_event(event):
            return 0
        travel_cfg = dict(self._schedule_cfg.get("campus_travel_minutes") or {})
        travel_cfg.update(self._travel_overrides)

        ev_campus = detect_campus(event.location)
        if ev_campus is not None and home_campus:
            if ev_campus == home_campus:
                return max(0, int(travel_cfg.get("same_campus", 10)))
            return max(0, int(travel_cfg.get("cross_campus", 45)))
        return max(0, int(travel_cfg.get("unknown", 30)))

    def nearest_slack_minutes(self, event: EventNormalized) -> Optional[int]:
        """Compute smallest non-overlapping gap (minutes) to any same-day busy slot."""
        if self._is_online_event(event):
            return None
        window = self._parse_event_window(event)
        if window is None:
            return None

        start_local = window[0].astimezone(self._default_tz)
        end_local = window[1].astimezone(self._default_tz)
        ev_date = start_local.date()

        gaps: List[float] = []

        # 1. Concrete BusyInterval entries touching ev_date
        for iv in self._context.busy_intervals:
            try:
                iv_s_raw = parse_iso_datetime(iv.start_datetime)
                iv_e_raw = parse_iso_datetime(iv.end_datetime)
            except ValueError:
                continue
            if iv_s_raw is None or iv_e_raw is None:
                continue
            iv_s = _ensure_aware(iv_s_raw, self._default_tz).astimezone(self._default_tz)
            iv_e = _ensure_aware(iv_e_raw, self._default_tz).astimezone(self._default_tz)
            if iv_s.date() != ev_date and iv_e.date() != ev_date:
                continue
            if iv_e <= start_local:
                gaps.append((start_local - iv_e).total_seconds() / 60.0)
            elif iv_s >= end_local:
                gaps.append((iv_s - end_local).total_seconds() / 60.0)

        # 2. Recurring WeeklyBusySlot entries on ev_date
        dow = ev_date.weekday()
        day_midnight = datetime(ev_date.year, ev_date.month, ev_date.day, tzinfo=self._default_tz)
        for slot in self._weekly_slots:
            if slot.day_of_week != dow:
                continue
            v_from = _parse_optional_date(slot.valid_from)
            v_until = _parse_optional_date(slot.valid_until)
            if v_from is not None and ev_date < v_from:
                continue
            if v_until is not None and ev_date > v_until:
                continue
            slot_s = day_midnight + timedelta(hours=slot.start_hour)
            slot_e = day_midnight + timedelta(hours=slot.end_hour)
            if slot_e <= start_local:
                gaps.append((start_local - slot_e).total_seconds() / 60.0)
            elif slot_s >= end_local:
                gaps.append((slot_s - end_local).total_seconds() / 60.0)

        if not gaps:
            return None
        return max(0, int(round(min(gaps))))

    def compute_fit(
        self,
        event: EventNormalized,
        *,
        home_campus: Optional[str] = None,
        max_travel_min: Optional[int] = None,
        pref_hours: Tuple[float, float] = (8.0, 21.0),
    ) -> DemoFitAssessment:
        """Compute continuous 5-step Fit score per `docs/module4_fit_handoff_spec.md`."""
        conflict: ConflictCheckResult = self.check_conflict(event)
        travel_min = max(0, int(self.estimate_travel_minutes(event, home_campus=home_campus)))
        window = self._parse_event_window(event)
        in_exam = self.is_in_exam_week(window[0]) if window is not None else False

        # Step 0: Hard schedule conflict
        if conflict.has_conflict:
            return DemoFitAssessment(
                fit_score=0.0,
                has_conflict=True,
                in_exam_week=in_exam,
                estimated_travel_min=travel_min,
                nearest_slack_min=0,
                conflict_detail=conflict.overlap_detail,
                sub_scores={"f_travel": 0.0, "f_slack": 0.0, "f_pref": 0.0, "f_exam": 0.0},
                reasons=[conflict.overlap_detail or "Trùng thời khóa biểu (Demo Module 4)"],
            )

        is_online = self._is_online_event(event)
        is_hybrid = event.online_mode == "hybrid"

        default_t_max = float(self._m4_cfg.get("default_max_travel_min", 45.0))
        t_max = float(max_travel_min) if (max_travel_min is not None and max_travel_min > 0) else default_t_max
        comfort_slack = float(self._m4_cfg.get("comfort_slack_min", 30.0))

        # Step 1: Travel component f_travel in [0, 1]
        if is_online:
            f_travel = 1.0
        else:
            if travel_min <= t_max:
                f_travel_offline = 1.0 - 0.8 * (travel_min / t_max)
            else:
                f_travel_offline = max(0.0, 0.2 - 0.2 * ((travel_min - t_max) / t_max))
            f_travel = 0.5 * (1.0 + f_travel_offline) if is_hybrid else f_travel_offline

        # Step 2: Same-day schedule slack f_slack in [0.4, 1.0]
        slack_min = None if (is_online or window is None) else self.nearest_slack_minutes(event)
        if is_online or window is None or slack_min is None:
            f_slack = 1.0
        else:
            excess = max(0.0, float(slack_min - conflict.travel_buffer_min))
            f_slack = min(1.0, 0.4 + 0.6 * (excess / max(1.0, comfort_slack)))

        # Step 3: Preferred hours f_pref in {0.3, 0.5, 0.7, 1.0}
        pref_cfg = self._m4_cfg.get("pref_hours_scores") or {}
        if window is None:
            f_pref = float(pref_cfg.get("missing_time", 0.5))
        else:
            start_local = window[0].astimezone(self._default_tz)
            end_local = window[1].astimezone(self._default_tz)
            h_s = start_local.hour + start_local.minute / 60.0
            h_e = end_local.hour + end_local.minute / 60.0
            p_start, p_end = float(pref_hours[0]), float(pref_hours[1])
            if end_local.date() > start_local.date():
                overflow = max(0.0, p_start - h_s)
            else:
                overflow = max(0.0, p_start - h_s) + max(0.0, h_e - p_end)
            if overflow <= 0.0:
                f_pref = float(pref_cfg.get("inside", 1.0))
            elif overflow <= 1.0:
                f_pref = float(pref_cfg.get("near_1h", 0.7))
            else:
                f_pref = float(pref_cfg.get("outside", 0.3))

        # Step 4: Exam-week multiplier f_exam
        exam_cfg = self._m4_cfg.get("exam_week_multipliers") or {}
        if not in_exam:
            f_exam = 1.0
        elif is_online:
            f_exam = float(exam_cfg.get("online", 0.90))
        elif is_hybrid:
            f_exam = float(exam_cfg.get("hybrid", 0.70))
        else:
            f_exam = float(exam_cfg.get("offline", 0.45))

        # Step 5: Weighted combination
        fw = self._m4_cfg.get("fit_weights") or {}
        w_t = float(fw.get("travel", 0.45))
        w_s = float(fw.get("slack", 0.25))
        w_p = float(fw.get("pref_hours", 0.30))

        raw_score = (w_t * f_travel + w_s * f_slack + w_p * f_pref) * f_exam
        fit_score = round(max(0.0, min(1.0, raw_score)), 6)

        reasons: List[str] = []
        if is_online:
            reasons.append("Hình thức trực tuyến (không mất thời gian di chuyển)")
        elif is_hybrid:
            reasons.append(f"Hình thức kết hợp online/offline (di chuyển ~{travel_min} phút)")
        else:
            if travel_min <= 15:
                reasons.append(f"Không trùng lịch, di chuyển ngắn (~{travel_min} phút)")
            elif travel_min <= t_max:
                reasons.append(f"Không trùng lịch, ước tính di chuyển ~{travel_min} phút")
            else:
                reasons.append(f"Di chuyển ~{travel_min} phút (vượt mức mong muốn {int(t_max)} phút)")

        if slack_min is not None and not is_online:
            if slack_min - conflict.travel_buffer_min < 15:
                reasons.append(f"Khá sát tiết học kề cạnh (trống {slack_min} phút)")
            else:
                reasons.append(f"Khoảng đệm rộng với tiết học ({slack_min} phút)")

        if in_exam:
            reasons.append("Diễn ra trong tuần thi")

        return DemoFitAssessment(
            fit_score=fit_score,
            has_conflict=False,
            in_exam_week=in_exam,
            estimated_travel_min=travel_min,
            nearest_slack_min=slack_min,
            conflict_detail=None,
            sub_scores={
                "f_travel": round(f_travel, 6),
                "f_slack": round(f_slack, 6),
                "f_pref": round(f_pref, 6),
                "f_exam": round(f_exam, 6),
            },
            reasons=reasons,
        )
