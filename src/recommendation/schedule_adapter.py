"""Module 4 Schedule Adapter (`ScheduleProvider` & `LocalScheduleAdapter`).

Provides timetable conflict detection, campus travel estimation, exam-week detection,
and Fit scoring behind the replaceable `ScheduleProvider` protocol, supporting both
JSON schedule payloads and pure-Python `.ics` (iCalendar) exports.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

from src.contract_validator import parse_iso_datetime
from src.recommendation.config import load_recommendation_config
from src.schemas import BusyInterval, EventNormalized, ScheduleContext

ICS_WEEKDAY_MAP: Dict[str, int] = {
    "MO": 0,
    "TU": 1,
    "WE": 2,
    "TH": 3,
    "FR": 4,
    "SA": 5,
    "SU": 6,
}


@dataclass(frozen=True)
class WeeklyBusySlot:
    """Recurring weekly busy class/activity slot."""

    day_of_week: int  # 0=Mon .. 6=Sun
    start_hour: float
    end_hour: float
    label: str = ""
    location: Optional[str] = None
    is_exam: bool = False
    valid_from: Optional[str] = None  # YYYY-MM-DD
    valid_until: Optional[str] = None  # YYYY-MM-DD


@dataclass(frozen=True)
class ConflictCheckResult:
    """Result of checking an event against student timetable & exams."""

    has_conflict: bool
    conflicting_label: Optional[str] = None
    conflict_type: Optional[str] = None  # "exam" | "busy_interval" | "weekly_class" | None
    overlap_detail: Optional[str] = None
    travel_buffer_min: int = 30


@dataclass(frozen=True)
class FitAssessment:
    """Module 4 convenience/fit evaluation in [0.0, 1.0]."""

    fit_score: float
    has_conflict: bool
    in_exam_week: bool
    estimated_travel_min: int
    conflict_detail: Optional[str] = None
    reasons: List[str] = field(default_factory=list)


@runtime_checkable
class ScheduleProvider(Protocol):
    """Replaceable protocol for Module 4 timetable/schedule integration."""

    def get_schedule_context(self) -> ScheduleContext:
        ...

    def check_conflict(self, event: EventNormalized) -> ConflictCheckResult:
        ...

    def is_in_exam_week(self, dt: datetime) -> bool:
        ...

    def estimate_travel_minutes(
        self,
        event: EventNormalized,
        home_campus: Optional[str] = None,
    ) -> int:
        ...

    def compute_fit(
        self,
        event: EventNormalized,
        *,
        home_campus: Optional[str] = None,
        max_travel_min: Optional[int] = None,
    ) -> FitAssessment:
        ...


def _ensure_aware(dt: datetime, default_tz: timezone) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=default_tz)
    return dt


def _parse_hour_value(val: Union[int, float, str]) -> float:
    """Convert float hour (7.5) or 'HH:MM' string ('07:30') to decimal hours."""
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip()
    if ":" in text:
        parts = text.split(":", 1)
        return int(parts[0]) + int(parts[1]) / 60.0
    return float(text)


def _format_hour(hour_val: float) -> str:
    total_mins = int(round(hour_val * 60))
    hh, mm = divmod(total_mins, 60)
    return f"{hh:02d}:{mm:02d}"


def _parse_optional_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return date.fromisoformat(text)
    dt = parse_iso_datetime(text)
    return dt.date() if dt is not None else None


def detect_campus(location: Optional[str]) -> Optional[str]:
    """Infer HCMUS campus ('cs1_nvc' vs 'cs2_lt') from a free-text location string."""
    if not location:
        return None
    loc = str(location).strip().lower()
    if not loc:
        return None

    cs1_tokens = ("nguyễn văn cừ", "nguyen van cu", "cơ sở 1", "co so 1", "cs1", "quận 5", "quan 5", "227")
    cs2_tokens = ("linh trung", "cơ sở 2", "co so 2", "cs2", "thủ đức", "thu duc", "đhqg", "dhqg", "làng đại học", "lang dai hoc")

    if any(tok in loc for tok in cs1_tokens):
        return "cs1_nvc"
    if any(tok in loc for tok in cs2_tokens):
        return "cs2_lt"
    return None


def _is_exam_label(summary: str) -> bool:
    low = (summary or "").strip().lower()
    return any(tok in low for tok in ("thi ", "kiểm tra", "kiem tra", "exam", "cuối kỳ", "cuoi ky", "giữa kỳ", "giua ky"))


def _unescape_ics_text(text: str) -> str:
    return (
        text.replace("\\n", " ")
        .replace("\\N", " ")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\\\", "\\")
        .strip()
    )


def _parse_ics_datetime(
    raw_val: str,
    params: Dict[str, str],
    default_tz: timezone,
) -> Tuple[datetime, bool]:
    """Parse RFC 5545 DTSTART/DTEND value into (aware_datetime, is_date_only)."""
    val = raw_val.strip()
    if val.endswith("Z") and len(val) == 16:
        dt = datetime.strptime(val, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return dt.astimezone(default_tz), False
    if "T" in val and len(val) >= 15:
        dt = datetime.strptime(val[:15], "%Y%m%dT%H%M%S")
        tzid = params.get("TZID", "").upper()
        src_tz = timezone.utc if tzid == "UTC" else default_tz
        return dt.replace(tzinfo=src_tz).astimezone(default_tz), False
    if len(val) == 8 and val.isdigit():
        d = datetime.strptime(val, "%Y%m%d")
        return d.replace(tzinfo=default_tz), True
    parsed = parse_iso_datetime(val)
    if parsed is None:
        raise ValueError(f"Unsupported iCalendar datetime '{raw_val}'")
    return _ensure_aware(parsed, default_tz).astimezone(default_tz), False


def parse_ics_schedule(
    ics_text: str,
    default_tz: timezone = timezone(timedelta(hours=7)),
) -> Tuple[ScheduleContext, List[WeeklyBusySlot]]:
    """Parse RFC 5545 `.ics` text into concrete `BusyInterval`s and recurring `WeeklyBusySlot`s."""
    unfolded = re.sub(r"\r?\n[ \t]", "", ics_text or "")
    lines = re.split(r"\r?\n", unfolded)

    busy_intervals: List[BusyInterval] = []
    weekly_slots: List[WeeklyBusySlot] = []

    in_event = False
    props: Dict[str, Tuple[Dict[str, str], str]] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line == "BEGIN:VEVENT":
            in_event = True
            props = {}
            continue
        if line == "END:VEVENT":
            in_event = False
            if "DTSTART" not in props:
                continue

            dtstart_params, dtstart_val = props["DTSTART"]
            start_dt, start_is_date = _parse_ics_datetime(dtstart_val, dtstart_params, default_tz)

            if "DTEND" in props:
                dtend_params, dtend_val = props["DTEND"]
                end_dt, _ = _parse_ics_datetime(dtend_val, dtend_params, default_tz)
            else:
                end_dt = start_dt + (timedelta(days=1) if start_is_date else timedelta(hours=1))

            if end_dt <= start_dt:
                end_dt = start_dt + timedelta(hours=1)

            summary = _unescape_ics_text(props["SUMMARY"][1]) if "SUMMARY" in props else "Lịch bận"
            location = _unescape_ics_text(props["LOCATION"][1]) if "LOCATION" in props else None
            is_exam = _is_exam_label(summary)

            rrule_val = props["RRULE"][1] if "RRULE" in props else ""
            rrule_map: Dict[str, str] = {}
            for part in rrule_val.split(";"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    rrule_map[k.strip().upper()] = v.strip().upper()

            if rrule_map.get("FREQ") == "WEEKLY":
                byday_raw = rrule_map.get("BYDAY", "")
                days: List[int] = []
                for token in byday_raw.split(","):
                    code = token.strip()[-2:]
                    if code in ICS_WEEKDAY_MAP:
                        days.append(ICS_WEEKDAY_MAP[code])
                if not days:
                    days = [start_dt.weekday()]

                valid_until: Optional[str] = None
                if "UNTIL" in rrule_map:
                    until_dt, _ = _parse_ics_datetime(rrule_map["UNTIL"], {}, default_tz)
                    valid_until = until_dt.date().isoformat()

                start_h = start_dt.hour + start_dt.minute / 60.0 + start_dt.second / 3600.0
                end_h = end_dt.hour + end_dt.minute / 60.0 + end_dt.second / 3600.0
                if end_h <= start_h:
                    end_h = min(24.0, start_h + 1.0)

                for dow in days:
                    weekly_slots.append(
                        WeeklyBusySlot(
                            day_of_week=dow,
                            start_hour=start_h,
                            end_hour=end_h,
                            label=summary,
                            location=location or None,
                            is_exam=is_exam,
                            valid_from=start_dt.date().isoformat(),
                            valid_until=valid_until,
                        )
                    )
            else:
                busy_intervals.append(
                    BusyInterval(
                        start_datetime=start_dt.isoformat(),
                        end_datetime=end_dt.isoformat(),
                        label=summary,
                        location=location or None,
                        is_exam=is_exam,
                    )
                )
            continue

        if in_event and ":" in line:
            head, val = line.split(":", 1)
            parts = head.split(";")
            key = parts[0].strip().upper()
            params: Dict[str, str] = {}
            for p in parts[1:]:
                if "=" in p:
                    pk, pv = p.split("=", 1)
                    params[pk.strip().upper()] = pv.strip()
            props[key] = (params, val)

    return ScheduleContext(busy_intervals=busy_intervals), weekly_slots


class LocalScheduleAdapter:
    """Default local implementation of `ScheduleProvider` for JSON and `.ics` schedules."""

    def __init__(
        self,
        context: Optional[ScheduleContext] = None,
        weekly_slots: Sequence[WeeklyBusySlot] = (),
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config if config is not None else load_recommendation_config()
        self._schedule_cfg: Dict[str, Any] = dict(cfg.get("schedule") or {})
        tz_hours = int(self._schedule_cfg.get("default_timezone_offset_hours", 7))
        self._default_tz = timezone(timedelta(hours=tz_hours))
        self._default_duration_min = int(self._schedule_cfg.get("default_event_duration_min", 120))
        default_buf = int(self._schedule_cfg.get("default_travel_buffer_min", 30))

        self._context = context if context is not None else ScheduleContext(default_travel_buffer_min=default_buf)
        self._weekly_slots = list(weekly_slots)

    @classmethod
    def from_json_dict(
        cls,
        data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> "LocalScheduleAdapter":
        cfg = config if config is not None else load_recommendation_config()
        sched_cfg = cfg.get("schedule") or {}
        default_buf = int(data.get("default_travel_buffer_min", sched_cfg.get("default_travel_buffer_min", 30)))

        busy_intervals: List[BusyInterval] = []
        for raw in data.get("busy_intervals") or []:
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
        for pair in data.get("exam_weeks") or []:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                exam_weeks.append((str(pair[0]), str(pair[1])))
            elif isinstance(pair, dict) and "start" in pair and "end" in pair:
                exam_weeks.append((str(pair["start"]), str(pair["end"])))

        raw_slots = data.get("weekly_slots") or data.get("weekly_timetable") or []
        weekly_slots: List[WeeklyBusySlot] = []
        for raw in raw_slots:
            dow = int(raw.get("day_of_week", raw.get("day")))
            start_h = _parse_hour_value(raw.get("start_hour", raw.get("start")))
            end_h = _parse_hour_value(raw.get("end_hour", raw.get("end")))
            weekly_slots.append(
                WeeklyBusySlot(
                    day_of_week=dow,
                    start_hour=start_h,
                    end_hour=end_h,
                    label=str(raw.get("label") or raw.get("course") or ""),
                    location=raw.get("location") or raw.get("campus"),
                    is_exam=bool(raw.get("is_exam", False)),
                    valid_from=raw.get("valid_from"),
                    valid_until=raw.get("valid_until"),
                )
            )

        context = ScheduleContext(
            busy_intervals=busy_intervals,
            exam_weeks=exam_weeks,
            default_travel_buffer_min=default_buf,
        )
        return cls(context=context, weekly_slots=weekly_slots, config=cfg)

    @classmethod
    def from_json_file(
        cls,
        path: Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> "LocalScheduleAdapter":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json_dict(payload, config=config)

    @classmethod
    def from_ics_text(
        cls,
        ics_text: str,
        config: Optional[Dict[str, Any]] = None,
        exam_weeks: Sequence[Tuple[str, str]] = (),
        default_travel_buffer_min: Optional[int] = None,
    ) -> "LocalScheduleAdapter":
        cfg = config if config is not None else load_recommendation_config()
        sched_cfg = cfg.get("schedule") or {}
        tz = timezone(timedelta(hours=int(sched_cfg.get("default_timezone_offset_hours", 7))))
        ctx, slots = parse_ics_schedule(ics_text, default_tz=tz)
        if exam_weeks:
            ctx.exam_weeks = [(str(s), str(e)) for s, e in exam_weeks]
        ctx.default_travel_buffer_min = (
            int(default_travel_buffer_min)
            if default_travel_buffer_min is not None
            else int(sched_cfg.get("default_travel_buffer_min", 30))
        )
        return cls(context=ctx, weekly_slots=slots, config=cfg)

    @classmethod
    def from_ics_file(
        cls,
        path: Path,
        config: Optional[Dict[str, Any]] = None,
        exam_weeks: Sequence[Tuple[str, str]] = (),
        default_travel_buffer_min: Optional[int] = None,
    ) -> "LocalScheduleAdapter":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_ics_text(
            text,
            config=config,
            exam_weeks=exam_weeks,
            default_travel_buffer_min=default_travel_buffer_min,
        )

    def get_schedule_context(self) -> ScheduleContext:
        return self._context

    @property
    def weekly_slots(self) -> List[WeeklyBusySlot]:
        return list(self._weekly_slots)

    @staticmethod
    def _is_online_event(event: EventNormalized) -> bool:
        if event.online_mode is not None:
            return event.online_mode == "online"
        return event.is_online is True

    def _parse_event_window(self, event: EventNormalized) -> Optional[Tuple[datetime, datetime]]:
        if not event.start_datetime:
            return None
        try:
            start_dt = parse_iso_datetime(event.start_datetime)
        except ValueError:
            return None
        if start_dt is None:
            return None
        start_dt = _ensure_aware(start_dt, self._default_tz)

        end_dt: Optional[datetime] = None
        if event.end_datetime:
            try:
                parsed_end = parse_iso_datetime(event.end_datetime)
                if parsed_end is not None:
                    end_dt = _ensure_aware(parsed_end, self._default_tz)
            except ValueError:
                end_dt = None

        if end_dt is None or end_dt <= start_dt:
            end_dt = start_dt + timedelta(minutes=self._default_duration_min)
        return start_dt, end_dt

    def check_conflict(self, event: EventNormalized) -> ConflictCheckResult:
        window = self._parse_event_window(event)
        if window is None:
            return ConflictCheckResult(has_conflict=False, travel_buffer_min=0)

        start_dt, end_dt = window
        is_online = self._is_online_event(event)
        buffer_min = 0 if is_online else self._context.default_travel_buffer_min
        buffer_td = timedelta(minutes=buffer_min)
        online_regular_conflict = bool(
            self._schedule_cfg.get("online_conflicts_with_regular_classes", False)
        )

        # 1. Check concrete BusyInterval entries
        for iv in self._context.busy_intervals:
            if is_online and not iv.is_exam and not online_regular_conflict:
                continue
            try:
                iv_s_raw = parse_iso_datetime(iv.start_datetime)
                iv_e_raw = parse_iso_datetime(iv.end_datetime)
            except ValueError:
                continue
            if iv_s_raw is None or iv_e_raw is None:
                continue
            iv_s = _ensure_aware(iv_s_raw, self._default_tz)
            iv_e = _ensure_aware(iv_e_raw, self._default_tz)
            if start_dt < (iv_e + buffer_td) and end_dt > (iv_s - buffer_td):
                label = iv.label or ("Lịch thi" if iv.is_exam else "Lịch bận")
                ctype = "exam" if iv.is_exam else "busy_interval"
                detail = f"Trùng {label} ({iv_s.strftime('%d/%m %H:%M')}–{iv_e.strftime('%H:%M')})"
                return ConflictCheckResult(
                    has_conflict=True,
                    conflicting_label=label,
                    conflict_type=ctype,
                    overlap_detail=detail,
                    travel_buffer_min=buffer_min,
                )

        # 2. Check recurring WeeklyBusySlot entries across dates touched by [start_dt, end_dt]
        start_local = start_dt.astimezone(self._default_tz)
        end_local = end_dt.astimezone(self._default_tz)
        cur_day = start_local.date()
        last_day = end_local.date()

        while cur_day <= last_day:
            dow = cur_day.weekday()
            day_midnight = datetime(cur_day.year, cur_day.month, cur_day.day, tzinfo=self._default_tz)
            for slot in self._weekly_slots:
                if slot.day_of_week != dow:
                    continue
                if is_online and not slot.is_exam and not online_regular_conflict:
                    continue

                v_from = _parse_optional_date(slot.valid_from)
                v_until = _parse_optional_date(slot.valid_until)
                if v_from is not None and cur_day < v_from:
                    continue
                if v_until is not None and cur_day > v_until:
                    continue

                slot_s = day_midnight + timedelta(hours=slot.start_hour)
                slot_e = day_midnight + timedelta(hours=slot.end_hour)
                if start_local < (slot_e + buffer_td) and end_local > (slot_s - buffer_td):
                    label = slot.label or "Lịch học TKB"
                    ctype = "exam" if slot.is_exam else "weekly_class"
                    detail = (
                        f"Trùng {label} ({cur_day.strftime('%d/%m')} "
                        f"{_format_hour(slot.start_hour)}–{_format_hour(slot.end_hour)})"
                    )
                    return ConflictCheckResult(
                        has_conflict=True,
                        conflicting_label=label,
                        conflict_type=ctype,
                        overlap_detail=detail,
                        travel_buffer_min=buffer_min,
                    )
            cur_day += timedelta(days=1)

        return ConflictCheckResult(has_conflict=False, travel_buffer_min=buffer_min)

    def is_in_exam_week(self, dt: datetime) -> bool:
        target = _ensure_aware(dt, self._default_tz)
        for raw_start, raw_end in self._context.exam_weeks:
            s_text = str(raw_start).strip()
            e_text = str(raw_end).strip()
            try:
                if len(s_text) == 10 and s_text[4] == "-":
                    d_s = date.fromisoformat(s_text)
                    s_dt = datetime(d_s.year, d_s.month, d_s.day, 0, 0, 0, tzinfo=self._default_tz)
                else:
                    parsed_s = parse_iso_datetime(s_text)
                    if parsed_s is None:
                        continue
                    s_dt = _ensure_aware(parsed_s, self._default_tz)

                if len(e_text) == 10 and e_text[4] == "-":
                    d_e = date.fromisoformat(e_text)
                    e_dt = datetime(d_e.year, d_e.month, d_e.day, 23, 59, 59, tzinfo=self._default_tz)
                else:
                    parsed_e = parse_iso_datetime(e_text)
                    if parsed_e is None:
                        continue
                    e_dt = _ensure_aware(parsed_e, self._default_tz)
            except ValueError:
                continue

            if s_dt <= target <= e_dt:
                return True
        return False

    def estimate_travel_minutes(
        self,
        event: EventNormalized,
        home_campus: Optional[str] = None,
    ) -> int:
        travel_cfg = self._schedule_cfg.get("campus_travel_minutes") or {}
        if self._is_online_event(event):
            return int(travel_cfg.get("online", 0))

        ev_campus = detect_campus(event.location)
        if ev_campus is not None and home_campus:
            if ev_campus == home_campus:
                return int(travel_cfg.get("same_campus", 10))
            return int(travel_cfg.get("cross_campus", 45))
        return int(travel_cfg.get("unknown", 30))

    def compute_fit(
        self,
        event: EventNormalized,
        *,
        home_campus: Optional[str] = None,
        max_travel_min: Optional[int] = None,
    ) -> FitAssessment:
        fit_cfg = self._schedule_cfg.get("fit_scoring") or {}
        conflict = self.check_conflict(event)
        travel_min = self.estimate_travel_minutes(event, home_campus=home_campus)

        window = self._parse_event_window(event)
        in_exam = self.is_in_exam_week(window[0]) if window is not None else False

        if conflict.has_conflict:
            return FitAssessment(
                fit_score=float(fit_cfg.get("conflict_score", 0.0)),
                has_conflict=True,
                in_exam_week=in_exam,
                estimated_travel_min=travel_min,
                conflict_detail=conflict.overlap_detail,
                reasons=[conflict.overlap_detail or "Trùng thời khóa biểu (Module 4)"],
            )

        is_online = self._is_online_event(event)
        is_hybrid = event.online_mode == "hybrid"
        reasons: List[str] = []

        score = float(fit_cfg.get("base_offline", 0.70))
        if is_online:
            score += float(fit_cfg.get("online_bonus", 0.25))
            reasons.append("Hình thức trực tuyến (linh hoạt thời gian)")
        elif is_hybrid:
            score += float(fit_cfg.get("hybrid_bonus", 0.15))
            reasons.append("Hình thức kết hợp online/offline")
        else:
            reasons.append("Không trùng lịch học TKB")
            ev_campus = detect_campus(event.location)
            if ev_campus is not None and home_campus:
                if ev_campus == home_campus:
                    score += float(fit_cfg.get("same_campus_bonus", 0.10))
                    reasons.append("Cùng cơ sở học tập")
                else:
                    score += float(fit_cfg.get("cross_campus_penalty", -0.15))
                    reasons.append(f"Khác cơ sở (ước tính di chuyển ~{travel_min} phút)")

        if event.fee_vnd == 0:
            score += float(fit_cfg.get("free_event_bonus", 0.05))

        if in_exam:
            if is_online:
                reasons.append("Diễn ra trong tuần thi (hình thức online phù hợp)")
            elif is_hybrid:
                score += float(fit_cfg.get("exam_week_hybrid_penalty", -0.10))
                reasons.append("Diễn ra trong tuần thi")
            else:
                score += float(fit_cfg.get("exam_week_offline_penalty", -0.25))
                reasons.append("Diễn ra trong tuần thi (offline bất tiện hơn)")

        if max_travel_min is not None and travel_min > max_travel_min:
            score += float(fit_cfg.get("travel_over_max_penalty", -0.20))
            reasons.append(f"Thời gian di chuyển (~{travel_min} phút) vượt mức mong muốn ({max_travel_min} phút)")

        clamped = round(max(0.0, min(1.0, score)), 6)
        return FitAssessment(
            fit_score=clamped,
            has_conflict=False,
            in_exam_week=in_exam,
            estimated_travel_min=travel_min,
            conflict_detail=None,
            reasons=reasons,
        )
