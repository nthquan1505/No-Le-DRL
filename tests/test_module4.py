import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.schedule import BusyInterval, FitCache, FitEvaluator, ScheduleContext, ScheduleEngine, WeeklyBusySlot
from src.schedule.calendar import add_calendar_events
from src.schedule.routing import CarRouting, RouteEstimate


@dataclass
class Event:
    start_datetime: str | None = "2026-11-14T14:00:00"
    end_datetime: str | None = "2026-11-14T16:00:00+07:00"
    online_mode: str | None = "offline"
    is_online: bool | None = None
    location: str | None = "cs2_lt"
    event_id: str = "event-1"


class ScheduleTests(unittest.TestCase):
    def test_handoff_cases(self):
        exam_weeks = [("2026-11-09", "2026-11-15")]
        before = BusyInterval("2026-11-14T11:00", "2026-11-14T13:00")
        online = Event("2026-11-14T19:00", "2026-11-14T20:30", "online")
        cases = [
            (Event(), ScheduleContext([replace(before, end_datetime="2026-11-14T14:30")]), None, None,
             (0, 0, 0, 0), 0.0),
            (online, ScheduleContext(), None, None, (1, 1, 1, 1), 1.0),
            (Event(), ScheduleContext([before]), "cs2_lt", None, (.822222, 1, 1, 1), .92),
            (Event(), ScheduleContext([replace(before, end_datetime="2026-11-14T13:30")]), "cs1_nvc", None,
             (.2, .4, 1, 1), .49),
            (Event(), ScheduleContext(exam_weeks=exam_weeks), "cs1_nvc", None, (.2, 1, 1, .45), .288),
            (online, ScheduleContext(exam_weeks=exam_weeks), None, None, (1, 1, 1, .9), .9),
            (Event("2026-11-14T07:30", "2026-11-14T10:00", location="Outside"), ScheduleContext(),
             "cs1_nvc", lambda *_: 60, (.133333, 1, .7, 1), .52),
            (Event(None, location=None), ScheduleContext(), None, None, (.466667, 1, .5, 1), .61),
        ]
        for number, (event, context, home, route, components, score) in enumerate(cases, 1):
            with self.subTest(case=number):
                result = ScheduleEngine(context, route_minutes=route).compute_fit(event, home_campus=home)
                self.assertEqual(result.fit_score, score)
                self.assertEqual(tuple(result.sub_scores.values()), components)
                self.assertEqual(result.has_conflict, number == 1)
                self.assertTrue(1 <= len(result.reasons) <= 3)

    def test_weekly_overlap_across_midnight_and_validity(self):
        slot = WeeklyBusySlot(5, 23, 23.75, valid_from="2026-11-14", valid_until="2026-11-14")
        event = Event("2026-11-14T17:00Z", "2026-11-15T01:00")
        engine = ScheduleEngine(ScheduleContext(weekly_slots=[slot]))
        self.assertTrue(engine.check_conflict(event).has_conflict)
        self.assertFalse(engine.check_conflict(replace(
            event, start_datetime="2026-11-22T00:00", end_datetime="2026-11-22T01:00"
        )).has_conflict)
        overnight = WeeklyBusySlot(5, 22, 2)
        engine = ScheduleEngine(ScheduleContext(weekly_slots=[overnight]))
        self.assertTrue(engine.check_conflict(event).has_conflict)

    def test_online_modes_and_default_duration(self):
        busy = BusyInterval("2026-11-14T15:30", "2026-11-14T16:00")
        online = Event(end_datetime="2026-11-14T13:00", online_mode=None, is_online=True)
        self.assertFalse(ScheduleEngine(ScheduleContext([busy])).check_conflict(online).has_conflict)
        self.assertTrue(ScheduleEngine(ScheduleContext([replace(busy, is_exam=True)])).check_conflict(online).has_conflict)
        config = {"schedule": {"online_conflicts_with_regular_classes": True}}
        self.assertTrue(ScheduleEngine(ScheduleContext([busy]), config=config).check_conflict(online).has_conflict)

    def test_calendar_uses_previous_appointment_for_route(self):
        context = add_calendar_events(ScheduleContext(), [{
            "id": "appointment", "location": "Clinic Q3",
            "start": {"dateTime": "2026-11-14T12:20+07:00"},
            "end": {"dateTime": "2026-11-14T13:20+07:00"},
        }])
        calls = []

        def estimate(origin, target):
            calls.append((origin, target))
            return 55

        result = ScheduleEngine(context, route_minutes=estimate).compute_fit(Event(), home_campus="cs1_nvc")
        self.assertEqual(calls, [("Clinic Q3", "cs2_lt")])
        self.assertEqual(result.fit_score, .52)
        self.assertIn("Thiếu khoảng 15 phút để đến sự kiện", result.reasons)

    def test_hybrid_and_invalid_travel_limit(self):
        result = ScheduleEngine().compute_fit(Event(online_mode="hybrid"), home_campus="cs1_nvc", max_travel_min=0)
        self.assertEqual(result.sub_scores["f_travel"], .6)
        self.assertEqual(result.fit_score, .82)


class RoutingTests(unittest.TestCase):
    def test_osrm_precedes_google_and_budget_is_shared(self):
        requests = []
        osrm_available = False

        def request_json(request):
            requests.append(request.full_url)
            if "googleapis" in request.full_url:
                self.assertEqual(json.loads(request.data)["travelMode"], "DRIVE")
                return {"routes": [{"duration": "3300s"}]}
            if not osrm_available:
                return {"code": "NoRoute"}
            return {"routes": [{"duration": 2399.5}]}

        with TemporaryDirectory() as directory:
            settings = dict(
                google_key="test-key", daily_google_limit=1, quota_db=Path(directory) / "quota.db",
                osrm_url="https://osrm.invalid", geocode=lambda _: (106.7, 10.8), request_json=request_json,
            )
            self.assertEqual(CarRouting(**settings)("clinic", "cs2_lt"), 55)
            self.assertIsNone(CarRouting(**settings)("clinic", "cs2_lt"))
            osrm_available = True
            self.assertEqual(CarRouting(**settings)("clinic", "cs2_lt"), 40)
        self.assertEqual(sum("googleapis" in request for request in requests), 1)
        self.assertIn("/route/v1/driving/", requests[0])

    def test_catalog_aliases_bypass_geocoding(self):
        def unexpected(_):
            self.fail("Known coordinates must bypass geocoding")

        router = CarRouting(
            osrm_url="https://osrm.invalid", geocode=unexpected,
            catalog={"places": [{
                "id": "guest", "address": "Guest campus", "aliases": ["Guest"],
                "longitude": 106.8, "latitude": 10.8,
            }], "travel_minutes": [{"origin": "cs2_lt", "destination": "guest", "minutes": 7}]},
            request_json=lambda _: {"routes": [{"duration": 721}]},
        )
        self.assertEqual(router.estimate("cs2_lt", "Guest").source, "catalog")
        self.assertEqual(router("Guest", "cs2_lt"), 13)


class EvaluationTests(unittest.TestCase):
    now = datetime(2026, 11, 13, tzinfo=timezone.utc)

    @staticmethod
    def calendar_item(item_id, start, end, location=None, *, free=False):
        return {"id": item_id, "summary": item_id, "location": location,
                "start": {"dateTime": start}, "end": {"dateTime": end},
                "transparency": "transparent" if free else "opaque"}

    def test_both_legs_home_and_flexible(self):
        items = [
            self.calendar_item("Đọc sách", "2026-11-14T14:00", "2026-11-14T15:00", "Elsewhere", free=True),
            self.calendar_item("Clinic", "2026-11-14T16:30", "2026-11-14T17:20", "Clinic Q3"),
        ]
        context = add_calendar_events(ScheduleContext(), items)
        calls = []

        def route(origin, target):
            calls.append((origin, target))
            return 45

        result = ScheduleEngine(context, home_address="Home address", route_minutes=route).compute_fit(Event())
        self.assertEqual(calls, [("Home address", "cs2_lt"), ("cs2_lt", "Clinic Q3")])
        self.assertEqual(result.flexible_overlaps, ("Đọc sách",))
        self.assertFalse(result.has_conflict)
        self.assertEqual(result.nearest_slack_min, 30)
        self.assertEqual(result.outbound.shortage_min, 15)
        self.assertEqual(result.fit_score, .49)

    def test_inherited_and_explicit_home_location(self):
        before = BusyInterval("2026-11-14T10:00", "2026-11-14T11:00", location="Clinic Q3")
        reading = BusyInterval("2026-11-14T12:00", "2026-11-14T13:20")

        def route(origin, target):
            return {"Clinic Q3": 55, "Home address": 20}[origin]

        for location, minutes, inferred in [(None, 55, True), ("Ở nhà", 20, False)]:
            engine = ScheduleEngine(ScheduleContext([before, replace(reading, location=location)]),
                                    home_address="Home address", route_minutes=route)
            result = engine.compute_fit(Event())
            self.assertEqual(result.estimated_travel_min, minutes)
            self.assertEqual(result.inbound.location_inferred, inferred)

    def test_cache_refresh_by_day_and_dedup_across_events(self):
        with TemporaryDirectory() as directory:
            cache = FitCache(Path(directory) / "cache.sqlite3", clock=self.now.timestamp)
            calls = []

            def route(origin, target):
                calls.append((origin, target))
                return 10

            evaluator = FitEvaluator(cache=cache, routing=route)
            saturday = Event(event_id="sat")
            sunday = Event("2026-11-15T14:00", "2026-11-15T16:00", event_id="sun")
            clinic = BusyInterval("2026-11-14T13:00", "2026-11-14T14:30", "Confidential visit")
            options = dict(home_address="Home address", now=self.now)
            first = evaluator.evaluate("u1", [saturday, sunday], ScheduleContext([clinic]), **options)
            self.assertTrue(first["sat"].has_conflict)
            self.assertEqual(calls, [("Home address", "cs2_lt")])
            self.assertNotIn(b"Confidential visit", cache.path.read_bytes())
            calls.clear()
            second = evaluator.evaluate("u1", [saturday, sunday], ScheduleContext(), **options)
            self.assertFalse(second["sat"].has_conflict)
            self.assertEqual(second["sun"], first["sun"])
            self.assertEqual(calls, [("Home address", "cs2_lt")])
            self.assertEqual(cache.delete_user("u1"), 2)
            self.assertEqual(cache.delete_user("u1"), 0)

    def test_calendar_read_failure_preserves_cache(self):
        busy = self.calendar_item("test exam", "2026-11-14T13:30", "2026-11-14T15:00")

        class Calendar:
            count = 0
            error = False

            def events(self):
                return self

            def list(self, **_):
                self.count += 1
                return self

            def execute(self):
                if self.error:
                    raise OSError("offline")
                return {"items": [busy]}

        with TemporaryDirectory() as directory:
            calendar = Calendar()
            cache = FitCache(Path(directory) / "cache.sqlite3", clock=self.now.timestamp)
            evaluator = FitEvaluator(cache=cache)
            kwargs = dict(calendar_service=calendar, now=self.now, home_campus="cs2_lt")
            result = evaluator.evaluate("u", [Event(), Event(event_id="other")], ScheduleContext(), **kwargs)
            self.assertTrue(result["event-1"].has_conflict)
            self.assertEqual(calendar.count, 1)
            calendar.error = True
            temporary = evaluator.evaluate("u", [Event()], ScheduleContext(), **kwargs)["event-1"]
            self.assertTrue(temporary.provisional)
            self.assertEqual(temporary.calendar_status, "unavailable")
            calendar.error = False
            self.assertEqual(evaluator.evaluate("u", [Event()], ScheduleContext(), **kwargs)["event-1"], result["event-1"])

    def test_cache_cleanup_runs_daily(self):
        timestamp = self.now.timestamp()
        with TemporaryDirectory() as directory:
            cache = FitCache(Path(directory) / "cache.sqlite3", clock=lambda: timestamp)
            assessment = ScheduleEngine().compute_fit(Event(location="cs2_lt"), home_campus="cs2_lt")
            cache.put("u", "old", "x", assessment, end=self.now + timedelta(days=1))
            cache.put("u", "unused", "y", assessment)
            self.assertEqual(cache.clean(), 0)
            timestamp += 2 * 86400
            self.assertEqual(cache.clean(), 1)
            timestamp += 29 * 86400
            self.assertEqual(cache.clean(), 1)

    def test_cache_detects_overnight_change(self):
        event = Event("2026-11-14T23:45", "2026-11-14T23:55", location="cs2_lt")
        appointment = BusyInterval("2026-11-15T00:10", "2026-11-15T01:00", location="cs2_lt")
        with TemporaryDirectory() as directory:
            cache = FitCache(Path(directory) / "cache.sqlite3", clock=self.now.timestamp)
            evaluator = FitEvaluator(cache=cache)
            options = dict(home_campus="cs2_lt", now=self.now)
            self.assertFalse(evaluator.evaluate("u", [event], ScheduleContext(), **options)["event-1"].has_conflict)
            self.assertTrue(evaluator.evaluate(
                "u", [event], ScheduleContext([appointment]), **options
            )["event-1"].has_conflict)

    def test_google_route_result_is_not_persisted(self):
        with TemporaryDirectory() as directory:
            cache = FitCache(Path(directory) / "cache.sqlite3", clock=self.now.timestamp)
            evaluator = FitEvaluator(cache=cache, routing=lambda *_: RouteEstimate(25, "google"))
            result = evaluator.evaluate(
                "u", [Event()], ScheduleContext(), home_address="Test address", now=self.now,
            )["event-1"]
            self.assertEqual(result.estimated_travel_min, 25)
            self.assertEqual(cache.delete_user("u"), 0)

    def test_db_campus_variants_and_coordinates_integration(self):
        engine = ScheduleEngine()
        # Real HCMUS campus variants from drl_database.db
        cs1_ev = Event(location="Cơ sở phường Chợ Quán, Trường Đại học Khoa học tự nhiên, ĐHQG-HCM")
        cs2_ev = Event(location="Trường Đại học Khoa học Tự nhiên, ĐHQG - HCM cơ sở Đông Hòa")
        self.assertEqual(engine.compute_fit(cs1_ev, home_campus="cs1_nvc").estimated_travel_min, 10)
        self.assertEqual(engine.compute_fit(cs2_ev, home_campus="cs1_nvc").estimated_travel_min, 45)
        # User default_coordinate near CS1 + external event with pre-geocoded coordinate
        from src.schemas import EventNormalized

        ext_ev = EventNormalized(
            event_id="db_12",
            title="Hội thảo liên trường",
            start_datetime="2026-11-14T14:00:00+07:00",
            end_datetime="2026-11-14T16:00:00+07:00",
            online_mode="offline",
            is_online=False,
            location="Hội trường A, Trường Đại học Kinh tế - Luật, ĐHQG - HCM",
            metadata={"coordinate": "10.871343,106.778523"},
        )
        fit_from_cs2_coord = engine.compute_fit(ext_ev, home_campus="10.875809,106.799178")
        fit_from_cs1_coord = engine.compute_fit(ext_ev, home_campus="10.762622,106.682172")
        self.assertLess(fit_from_cs2_coord.estimated_travel_min, fit_from_cs1_coord.estimated_travel_min)
        self.assertGreater(fit_from_cs2_coord.fit_score, fit_from_cs1_coord.fit_score)
        self.assertFalse(fit_from_cs2_coord.provisional)


if __name__ == "__main__":
    unittest.main()
