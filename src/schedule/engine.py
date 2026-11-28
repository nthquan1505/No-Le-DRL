"""Deterministic timetable and Fit evaluation, independent of Module 3."""

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from math import asin, cos, isfinite, radians, sin, sqrt
from typing import Callable, Optional, Protocol
import re
import unicodedata

CAMPUS_COORDS: dict[str, tuple[float, float]] = {
    "cs1_nvc": (10.762622, 106.682172),
    "cs2_lt": (10.875809, 106.799178),
}


class Event(Protocol):
    event_id: str
    start_datetime: Optional[str]
    end_datetime: Optional[str]
    online_mode: Optional[str]
    is_online: Optional[bool]
    location: Optional[str]


@dataclass(frozen=True)
class BusyInterval:
    start_datetime: str
    end_datetime: str
    label: str = ""
    location: Optional[str] = None
    is_exam: bool = False


@dataclass(frozen=True)
class WeeklyBusySlot:
    day_of_week: int
    start_hour: float
    end_hour: float
    label: str = ""
    location: Optional[str] = None
    is_exam: bool = False
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None


@dataclass
class ScheduleContext:
    busy_intervals: list[BusyInterval] = field(default_factory=list)
    exam_weeks: list[tuple[str, str]] = field(default_factory=list)
    default_travel_buffer_min: int = 30
    weekly_slots: list[WeeklyBusySlot] = field(default_factory=list)
    flexible_intervals: list[BusyInterval] = field(default_factory=list)


@dataclass(frozen=True)
class ConflictCheckResult:
    has_conflict: bool
    conflicting_label: Optional[str] = None
    conflict_type: Optional[str] = None
    overlap_detail: Optional[str] = None
    travel_buffer_min: int = 30


@dataclass(frozen=True)
class FitAssessment:
    fit_score: float
    has_conflict: bool
    in_exam_week: bool
    estimated_travel_min: int
    nearest_slack_min: Optional[int] = None
    conflict_detail: Optional[str] = None
    sub_scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    flexible_overlaps: tuple[str, ...] = ()
    inbound: Optional["TravelLeg"] = None
    outbound: Optional["TravelLeg"] = None
    calendar_status: str = "local"
    provisional: bool = False


@dataclass(frozen=True)
class TravelLeg:
    minutes: int
    gap_min: Optional[int]
    shortage_min: Optional[int]
    source: str
    location_inferred: bool = False


@dataclass(frozen=True)
class _Busy:
    start: datetime
    end: datetime
    label: str
    location: Optional[str]
    is_exam: bool
    kind: str


def _parse_lat_lng(raw: Optional[str]) -> Optional[tuple[float, float]]:
    if not raw or not isinstance(raw, str) or "," not in raw:
        return None
    parts = raw.split(",")
    if len(parts) != 2:
        return None
    try:
        lat, lng = float(parts[0].strip()), float(parts[1].strip())
    except ValueError:
        return None
    if not (isfinite(lat) and isfinite(lng) and -90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return None
    return lat, lng


def _haversine_km(c1: tuple[float, float], c2: tuple[float, float]) -> float:
    lat1, lon1 = radians(c1[0]), radians(c1[1])
    lat2, lon2 = radians(c2[0]), radians(c2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0088 * 2 * asin(min(1.0, sqrt(a)))


def _campus(location: Optional[str]) -> Optional[str]:
    if not location:
        return None
    coord = _parse_lat_lng(location)
    if coord is not None:
        for cid, cpos in CAMPUS_COORDS.items():
            if _haversine_km(coord, cpos) <= 0.45:
                return cid
        return None
    folded = (location or "").casefold().translate(str.maketrans({"đ": "d"}))
    text = unicodedata.normalize("NFKD", folded)
    text = "".join(char for char in text if not unicodedata.combining(char))
    if re.search(r"\b(?:cs1_nvc|(?:cs|co so)\s*0?1|227\s+nguyen van cu|nguyen van cu|cho quan)\b", text):
        return "cs1_nvc"
    if re.search(r"\b(?:cs2_lt|(?:cs|co so)\s*0?2|linh trung|dong hoa)\b", text):
        return "cs2_lt"
    return None


def _extract_slot_coord(slot) -> Optional[tuple[float, float]]:
    if slot is None:
        return None
    raw_coord = getattr(slot, "coordinate", None)
    if not raw_coord:
        meta = getattr(slot, "metadata", None)
        if isinstance(meta, dict):
            raw_coord = meta.get("coordinate")
    parsed = _parse_lat_lng(raw_coord)
    if parsed is not None:
        return parsed
    loc = getattr(slot, "location", None)
    parsed_loc = _parse_lat_lng(loc)
    if parsed_loc is not None:
        return parsed_loc
    camp = _campus(loc)
    return CAMPUS_COORDS.get(camp) if camp else None


class ScheduleEngine:
    def __init__(
        self,
        context: Optional[ScheduleContext] = None,
        *,
        weekly_slots: tuple[WeeklyBusySlot, ...] = (),
        config: Optional[dict] = None,
        route_minutes: Optional[Callable[[str, str], Optional[int]]] = None,
        home_address: Optional[str] = None,
    ):
        cfg = (config or {}).get("schedule", {})
        self._tz = timezone(timedelta(hours=cfg.get("default_timezone_offset_hours", 7)))
        self._duration = cfg.get("default_event_duration_min", 120)
        self._comfort = cfg.get("comfort_slack_min", 30)
        self._max_travel = cfg.get("default_max_travel_min", 45)
        self._pref = tuple(cfg.get("default_pref_hours", (8.0, 21.0)))
        if min(self._duration, self._comfort, self._max_travel) <= 0:
            raise ValueError("Schedule duration, comfort and travel limit must be positive")
        self._online_conflicts = cfg.get("online_conflicts_with_regular_classes", False)
        self._campus_minutes = {"online": 0, "same_campus": 10, "cross_campus": 45, "unknown": 30}
        self._campus_minutes.update(cfg.get("campus_travel_minutes", {}))
        scoring = cfg.get("fit_scoring", {})
        self._weights = {"travel": 0.45, "slack": 0.25, "pref_hours": 0.30}
        self._weights.update(scoring.get("weights", {}))
        self._pref_scores = {"inside": 1.0, "near_1h": 0.7, "outside": 0.3, "missing_time": 0.5}
        self._pref_scores.update(scoring.get("pref_hours_scores", {}))
        self._exam_scores = {"online": 0.90, "hybrid": 0.70, "offline": 0.45}
        self._exam_scores.update(scoring.get("exam_week_multipliers", {}))
        self._context = context if context is not None else ScheduleContext(
            default_travel_buffer_min=cfg.get("default_travel_buffer_min", 30)
        )
        self.weekly_slots = tuple(weekly_slots) or tuple(getattr(self._context, "weekly_slots", ()))
        self._route = route_minutes
        self._home_address = home_address

    def get_schedule_context(self) -> ScheduleContext:
        return self._context

    def _parse(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.strip())
            return dt.astimezone(self._tz) if dt.tzinfo else dt.replace(tzinfo=self._tz)
        except ValueError:
            return None

    def _window(self, event: Event) -> Optional[tuple[datetime, datetime]]:
        start = self._parse(event.start_datetime)
        if start is None:
            return None
        end = self._parse(event.end_datetime)
        return start, end if end and end > start else start + timedelta(minutes=self._duration)

    @staticmethod
    def _mode(event: Event) -> str:
        if event.online_mode in {"online", "offline", "hybrid"}:
            return event.online_mode
        return "online" if event.is_online is True else "offline"

    def _intervals(self, start: datetime, end: datetime, buffer: int) -> list[_Busy]:
        intervals = []
        for item in self._context.busy_intervals:
            busy_start, busy_end = self._parse(item.start_datetime), self._parse(item.end_datetime)
            if busy_start and busy_end and busy_start < busy_end:
                intervals.append(_Busy(busy_start, busy_end, item.label, item.location, item.is_exam, "busy_interval"))

        padding = timedelta(minutes=buffer)
        day = (start - padding).date() - timedelta(days=1)
        last = (end + padding).date()
        while day <= last:
            midnight = datetime.combine(day, time.min, self._tz)
            for slot in self.weekly_slots:
                if slot.day_of_week != day.weekday():
                    continue
                if (slot.valid_from and day < date.fromisoformat(slot.valid_from)) or (
                    slot.valid_until and day > date.fromisoformat(slot.valid_until)
                ):
                    continue
                if not 0 <= slot.start_hour < 24 or not 0 <= slot.end_hour <= 24:
                    continue
                if slot.start_hour == slot.end_hour:
                    continue
                end_hour = slot.end_hour + (24 if slot.end_hour < slot.start_hour else 0)
                intervals.append(_Busy(
                    midnight + timedelta(hours=slot.start_hour), midnight + timedelta(hours=end_hour),
                    slot.label, slot.location, slot.is_exam, "weekly_class",
                ))
            day += timedelta(days=1)
        return intervals

    def _event_schedule(self, event: Event):
        window = self._window(event)
        mode = self._mode(event)
        buffer = 0 if mode == "online" else max(0, self._context.default_travel_buffer_min)
        return window, mode, buffer, self._intervals(*window, buffer) if window else []

    def _conflict(self, window, mode, buffer, intervals) -> ConflictCheckResult:
        if window:
            start, end = window
            padding = timedelta(minutes=buffer)
            for slot in intervals:
                if mode == "online" and not (slot.is_exam or self._online_conflicts):
                    continue
                if start < slot.end + padding and end > slot.start - padding:
                    label = slot.label or ("lịch thi" if slot.is_exam else "lịch bận")
                    actual_overlap = start < slot.end and end > slot.start
                    reason = "Trùng" if actual_overlap else f"Không đủ đệm {buffer} phút với"
                    detail = f"{reason} {label} ({slot.start:%d/%m %H:%M}–{slot.end:%d/%m %H:%M})"
                    return ConflictCheckResult(True, label, "exam" if slot.is_exam else slot.kind, detail, buffer)
        return ConflictCheckResult(False, travel_buffer_min=buffer)

    def check_conflict(self, event: Event) -> ConflictCheckResult:
        return self._conflict(*self._event_schedule(event))

    def is_in_exam_week(self, dt: datetime) -> bool:
        day = (dt.astimezone(self._tz) if dt.tzinfo else dt).date()
        for pair in self._context.exam_weeks:
            if not pair or len(pair) != 2:
                continue
            start, end = str(pair[0] or "").strip(), str(pair[1] or "").strip()
            if len(start) < 10 or len(end) < 10:
                continue
            try:
                if date.fromisoformat(start[:10]) <= day <= date.fromisoformat(end[:10]):
                    return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _neighbors(window, intervals):
        if not window:
            return None, None
        start, end = window
        before = max((item for item in intervals if item.end <= start and item.end.date() == start.date()),
                     key=lambda item: item.end, default=None)
        after = min((item for item in intervals if item.start >= end and item.start.date() == end.date()),
                    key=lambda item: item.start, default=None)
        return before, after

    @staticmethod
    def _slack(window, before, after) -> Optional[int]:
        if not window:
            return None
        start, end = window
        gaps = []
        if before:
            gaps.append(start - before.end)
        if after:
            gaps.append(after.start - end)
        return max(0, int(round(min(gaps).total_seconds() / 60))) if gaps else None

    def nearest_slack_minutes(self, event: Event) -> Optional[int]:
        window, mode, buffer, intervals = self._event_schedule(event)
        if mode == "online" or self._conflict(window, mode, buffer, intervals).has_conflict:
            return None
        return self._slack(window, *self._neighbors(window, intervals))

    def flexible_overlaps(self, event: Event) -> tuple[str, ...]:
        window = self._window(event)
        if window is None:
            return ()
        start, end = window
        labels = []
        for item in getattr(self._context, "flexible_intervals", ()):
            first, last = self._parse(item.start_datetime), self._parse(item.end_datetime)
            if first and last and first < last and first < end and start < last:
                labels.append(item.label or "Lịch linh hoạt")
        return tuple(labels)

    def _location(self, slot, intervals, home, *, inherit=True):
        """Resolve a slot location and report whether the location was inferred."""
        if slot is None:
            return home, False
        if slot.location:
            if slot.location.strip().casefold() in {"ở nhà", "nhà", "home", "o nha"}:
                return self._home_address, False
            return slot.location, False
        if not inherit:
            return None, False
        earlier = max(
            (item for item in intervals if item.location and item.end <= slot.start
             and item.end.date() == slot.start.date()),
            key=lambda item: item.end, default=None,
        )
        location, _ = self._location(earlier, (), home)
        return location, True

    def _coord_for_endpoint(self, label: Optional[str], slot=None) -> Optional[tuple[float, float]]:
        if slot is not None:
            sc = _extract_slot_coord(slot)
            if sc is not None:
                return sc
        parsed = _parse_lat_lng(label)
        if parsed is not None:
            return parsed
        camp = _campus(label)
        return CAMPUS_COORDS.get(camp) if camp else None

    def _travel(self, origin, target, mode, *, origin_coord=None, target_coord=None):
        if mode == "online":
            return 0, "online"
        first, second = _campus(origin), _campus(target)
        if first and second:
            key = "same_campus" if first == second else "cross_campus"
            return max(0, int(self._campus_minutes[key])), "campus"
        route_origin = first or (f"{origin_coord[0]:.6f},{origin_coord[1]:.6f}" if not origin and origin_coord else origin)
        route_target = second or (f"{target_coord[0]:.6f},{target_coord[1]:.6f}" if not target and target_coord else target)
        if self._route and route_origin and route_target:
            estimate = self._route(route_origin, route_target)
            if estimate is not None:
                return max(0, int(getattr(estimate, "minutes", estimate))), getattr(estimate, "source", "route")
            if origin_coord is not None and target_coord is not None:
                km = _haversine_km(origin_coord, target_coord)
                est_min = 0 if km < 0.15 else max(5, int(round(km * 1.25 / 30.0 * 60.0)))
                return est_min, "coordinate"
            return max(0, int(self._campus_minutes["unknown"])), "fallback"
        if origin_coord is not None and target_coord is not None:
            km = _haversine_km(origin_coord, target_coord)
            est_min = 0 if km < 0.15 else max(5, int(round(km * 1.25 / 30.0 * 60.0)))
            return est_min, "coordinate"
        return max(0, int(self._campus_minutes["unknown"])), "unverified"

    def _inbound(self, event, mode, before, intervals, home_campus, window):
        home_ref = self._home_address or home_campus
        origin, inferred = self._location(before, intervals, home_ref)
        target, _ = self._location(event, (), None, inherit=False)
        origin_coord = self._coord_for_endpoint(origin, before if not inferred else None)
        target_coord = self._coord_for_endpoint(target, event)
        if mode != "online" and event.online_mode is None and event.is_online is None:
            origin, target = None, None
            origin_coord, target_coord = None, None
        travel = self._travel(origin, target, mode, origin_coord=origin_coord, target_coord=target_coord)
        return self._leg(travel, window[0] - before.end if before else None, inferred)

    @staticmethod
    def _leg(travel, gap, inferred=False):
        minutes, source = travel
        gap_min = None if gap is None else int(gap.total_seconds() // 60)
        short = None if gap_min is None or source in {"fallback", "unverified"} else max(0, minutes - gap_min)
        return TravelLeg(minutes, gap_min, short, source, inferred)

    def estimate_travel_minutes(self, event: Event, home_campus: Optional[str] = None) -> int:
        window, mode, _, intervals = self._event_schedule(event)
        before, _ = self._neighbors(window, intervals)
        return self._inbound(event, mode, before, intervals, home_campus, window).minutes

    def _preference(self, window, pref_hours):
        if window is None:
            return self._pref_scores["missing_time"]
        start, end = window
        pref_start, pref_end = pref_hours
        start_hour = start.hour + start.minute / 60
        end_hour = end.hour + end.minute / 60
        overflow = max(0, pref_start - start_hour)
        if start.date() == end.date():
            overflow += max(0, end_hour - pref_end)
        key = "inside" if overflow <= 0 else "near_1h" if overflow <= 1 else "outside"
        return self._pref_scores[key]

    def compute_fit(
        self, event: Event, *, home_campus: Optional[str] = None,
        max_travel_min: Optional[int] = None, pref_hours: Optional[tuple[float, float]] = None,
    ) -> FitAssessment:
        window, mode, buffer, intervals = self._event_schedule(event)
        conflict = self._conflict(window, mode, buffer, intervals)
        before, after = self._neighbors(window, intervals)
        inbound = self._inbound(event, mode, before, intervals, home_campus, window)
        travel, source = inbound.minutes, inbound.source
        flexible = self.flexible_overlaps(event)
        in_exam = bool(window and self.is_in_exam_week(window[0]))
        outbound = None
        if mode != "online" and after:
            first, _ = self._location(event, (), None, inherit=False)
            last, _ = self._location(after, (), None, inherit=False)
            outbound = self._leg(
                self._travel(
                    first,
                    last,
                    mode,
                    origin_coord=self._coord_for_endpoint(first, event),
                    target_coord=self._coord_for_endpoint(last, after),
                ),
                after.start - window[1],
            )
        provisional = any(leg and leg.source == "fallback" for leg in (inbound, outbound))
        if conflict.has_conflict:
            return FitAssessment(0.0, True, in_exam, travel, 0, conflict.overlap_detail,
                                 dict.fromkeys(("f_travel", "f_slack", "f_pref", "f_exam"), 0.0),
                                 [conflict.overlap_detail or "Trùng thời khóa biểu"],
                                 flexible, inbound, outbound, provisional=provisional)

        limit = max_travel_min if max_travel_min and max_travel_min > 0 else self._max_travel
        ratio = travel / limit
        offline = 1 - 0.8 * ratio if ratio <= 1 else max(0, 0.2 - 0.2 * (ratio - 1))
        f_travel = 1.0 if mode == "online" else (1 + offline) / 2 if mode == "hybrid" else offline
        slack = None if mode == "online" else self._slack(window, before, after)
        f_slack = 1.0 if slack is None else min(1.0, 0.4 + 0.6 * max(0, slack - buffer) / self._comfort)
        hours = pref_hours or self._pref
        if len(hours) != 2 or not all(isfinite(hour) for hour in hours) or not 0 <= hours[0] < hours[1] <= 24:
            hours = self._pref
        f_pref = self._preference(window, hours)
        f_exam = self._exam_scores[mode] if in_exam else 1.0
        score = (self._weights["travel"] * f_travel + self._weights["slack"] * f_slack
                 + self._weights["pref_hours"] * f_pref) * f_exam

        reasons = []
        if not window:
            reasons.append("Thiếu giờ bắt đầu, chưa thể kiểm tra lịch")
        if source in {"fallback", "unverified"}:
            reasons.append(f"Chưa xác minh thời gian di chuyển, dùng {travel} phút dự phòng")
        elif source == "online":
            reasons.append("Sự kiện trực tuyến, không cần di chuyển")
        else:
            reasons.append(f"Di chuyển ước tính {travel} phút")
            if inbound.shortage_min:
                reasons.append(f"Thiếu khoảng {inbound.shortage_min} phút để đến sự kiện")
        if outbound and outbound.shortage_min:
            reasons.append(f"Thiếu khoảng {outbound.shortage_min} phút để tới lịch kế tiếp")
        if inbound.location_inferred:
            reasons.append("Điểm xuất phát được ước tính từ lịch trong ngày")
        if in_exam:
            reasons.append("Sự kiện bắt đầu trong tuần thi")
        elif window and len(reasons) < 3:
            reasons.append("Không trùng lịch theo quy tắc đệm")
        return FitAssessment(
            round(max(0.0, min(1.0, score)), 6), False, in_exam, travel, slack,
            sub_scores={key: round(value, 6) for key, value in (
                ("f_travel", f_travel), ("f_slack", f_slack), ("f_pref", f_pref), ("f_exam", f_exam))},
            reasons=reasons[:3],
            flexible_overlaps=flexible, inbound=inbound, outbound=outbound, provisional=provisional,
        )
