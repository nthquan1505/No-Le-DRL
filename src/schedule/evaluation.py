"""Batch entry point for a single student's recommendation request."""

from dataclasses import asdict, replace
from datetime import datetime, time, timedelta
from hashlib import sha256
import json

from .calendar import CalendarReadError, read_calendar
from .engine import ScheduleEngine


class FitEvaluator:
    def __init__(self, *, cache=None, config=None, routing=None, routing_version="1"):
        self.cache = cache
        self.config = config
        self.routing = routing
        self.routing_version = routing_version

    def evaluate(
        self, user_id, events, context, *, weekly_slots=(), calendar_service=None, calendar_token=None,
        calendar_id="primary", exam_event_ids=(), exclude_event_ids=(), home_address=None, home_campus=None,
        max_travel_min=None, pref_hours=None, now=None,
    ):
        """Return event ID → FitAssessment for the supplied recommendation candidates."""
        slots = tuple(weekly_slots) or tuple(getattr(context, "weekly_slots", ()))
        schedule = ScheduleEngine(context, weekly_slots=slots, config=self.config)
        now_raw = now.isoformat() if hasattr(now, "isoformat") else str(now) if now else None
        now = (schedule._parse(now_raw) if now_raw else None) or datetime.now(schedule._tz)
        windows = [(event, schedule._window(event)) for event in events]
        windows = [(event, window) for event, window in windows if window is None or window[1] > now]
        if not windows:
            if self.cache:
                self.cache.clean()
            return {}

        calendar_status = "local"
        dated = [window for _, window in windows if window]
        if (calendar_service is not None or calendar_token is not None) and dated:
            padding = timedelta(minutes=max(0, context.default_travel_buffer_min))
            first = (min(start for start, _ in dated) - padding).date()
            last = (max(end for _, end in dated) + padding).date()
            begin = datetime.combine(first, time.min, schedule._tz)
            until = datetime.combine(last + timedelta(days=1), time.min, schedule._tz)
            try:
                service = calendar_service
                if service is None:
                    from .oauth import load_calendar_service

                    service = load_calendar_service(calendar_token)
                context = read_calendar(
                    service, context, time_min=begin, time_max=until, calendar_id=calendar_id,
                    exam_event_ids=exam_event_ids, exclude_event_ids=exclude_event_ids,
                )
                calendar_status = "synced"
            except CalendarReadError:
                calendar_status = "unavailable"

        routes = {}
        if self.routing is not None:
            resolve = getattr(self.routing, "estimate", self.routing)

        def route(origin, destination):
            key = (origin, destination)
            if key not in routes:
                routes[key] = resolve(origin, destination)
            return routes[key]

        engine = ScheduleEngine(context, weekly_slots=slots, config=self.config, home_address=home_address,
                                route_minutes=route if self.routing is not None else None)
        if self.cache:
            self.cache.clean()
        inputs = {"home_campus": home_campus, "max_travel_min": max_travel_min, "pref_hours": pref_hours}
        results = {}
        for event, window in windows:
            signature = self._signature(engine, event, window, inputs, home_address, calendar_status)
            available = calendar_status != "unavailable"
            cached = self.cache.get(user_id, event.event_id, signature) if self.cache and available else None
            if cached is None:
                result = engine.compute_fit(event, **inputs)
                result = replace(result, calendar_status=calendar_status,
                                 provisional=result.provisional or not available)
                if self.cache and available:
                    self.cache.put(user_id, event.event_id, signature, result, end=window[1] if window else None)
            else:
                detail = engine.check_conflict(event).overlap_detail if cached.has_conflict else None
                result = replace(cached, conflict_detail=detail, flexible_overlaps=engine.flexible_overlaps(event),
                                 reasons=[detail or "Trùng thời khóa biểu"] if cached.has_conflict else cached.reasons)
            results[event.event_id] = result
        return results

    def _signature(self, engine, event, window, inputs, home_address, calendar_status):
        # Include complete local days so before/after slots and inherited origins invalidate together.
        schedule = []
        in_exam = False
        if window:
            start, end = window
            buffer = max(0, engine.get_schedule_context().default_travel_buffer_min)
            day_start = datetime.combine((start - timedelta(minutes=buffer)).date(), time.min, engine._tz)
            day_end = datetime.combine((end + timedelta(minutes=buffer)).date() + timedelta(days=1),
                                       time.min, engine._tz)
            schedule = sorted(
                (slot.start.isoformat(), slot.end.isoformat(), slot.location or "", slot.is_exam, slot.kind)
                for slot in engine._intervals(day_start, day_end, 0)
                if slot.start < day_end and slot.end > day_start
            )
            in_exam = engine.is_in_exam_week(start)
        snapshot = {
            "version": self.routing_version, "route_enabled": self.routing is not None,
            "routing_config": getattr(self.routing, "cache_signature", None),
            "event": (event.start_datetime, event.end_datetime, event.online_mode, event.is_online, event.location),
            "inputs": inputs, "home": home_address, "calendar": calendar_status,
            "schedule": schedule, "in_exam": in_exam, "config": self.config,
            "buffer": engine.get_schedule_context().default_travel_buffer_min,
        }
        return sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
