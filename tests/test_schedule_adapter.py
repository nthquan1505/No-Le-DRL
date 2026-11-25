"""Unit tests for Module 4 Schedule Adapter (`src/recommendation/schedule_adapter.py`)."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.recommendation.schedule_adapter import (
    LocalScheduleAdapter,
    ScheduleProvider,
    detect_campus,
)
from src.schemas import EventNormalized


class TestScheduleAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.tz = timezone(timedelta(hours=7))
        self.json_schedule = {
            "default_travel_buffer_min": 30,
            "exam_weeks": [
                ["2026-11-02", "2026-11-08"],
            ],
            "busy_intervals": [
                {
                    "start_datetime": "2026-10-20T09:00:00+07:00",
                    "end_datetime": "2026-10-20T11:00:00+07:00",
                    "label": "Thi giữa kỳ Giải tích",
                    "location": "227 Nguyễn Văn Cừ, Q5",
                    "is_exam": True,
                }
            ],
            "weekly_slots": [
                {
                    "day_of_week": 0,  # Monday 2026-10-19
                    "start_hour": "07:30",
                    "end_hour": 11.0,
                    "label": "Cơ sở Trí tuệ Nhân tạo",
                    "location": "CS1 - Nguyễn Văn Cừ",
                }
            ],
        }
        self.adapter = LocalScheduleAdapter.from_json_dict(self.json_schedule)

    def test_implements_schedule_provider_protocol(self) -> None:
        self.assertIsInstance(self.adapter, ScheduleProvider)
        self.assertEqual(len(self.adapter.weekly_slots), 1)
        self.assertEqual(self.adapter.weekly_slots[0].start_hour, 7.5)

    def test_ics_parsing_unfolds_rrule_utc_tzid_allday_and_exams(self) -> None:
        ics = (
            "BEGIN:VCALENDAR\r\n"
            "VERSION:2.0\r\n"
            "BEGIN:VEVENT\r\n"
            "SUMMARY:Nhap mon Tu duy Tinh\r\n"
            "  toan (TKB)\r\n"
            "LOCATION:CS2 - Linh Trung\\, Thu Duc\r\n"
            "DTSTART;TZID=Asia/Ho_Chi_Minh:20261005T073000\r\n"
            "DTEND;TZID=Asia/Ho_Chi_Minh:20261005T110000\r\n"
            "RRULE:FREQ=WEEKLY;BYDAY=MO,WE;UNTIL=20261231T235959Z\r\n"
            "END:VEVENT\r\n"
            "BEGIN:VEVENT\r\n"
            "SUMMARY:Thi cuoi ky Mon Co so\r\n"
            "DTSTART:20261105T020000Z\r\n"
            "DTEND:20261105T040000Z\r\n"
            "END:VEVENT\r\n"
            "BEGIN:VEVENT\r\n"
            "SUMMARY:Ngay hoi Sinh vien (Ca ngay)\r\n"
            "DTSTART;VALUE=DATE:20261025\r\n"
            "DTEND;VALUE=DATE:20261026\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )
        ics_adapter = LocalScheduleAdapter.from_ics_text(ics)
        slots = ics_adapter.weekly_slots
        ctx = ics_adapter.get_schedule_context()

        self.assertEqual(len(slots), 2)
        self.assertEqual({s.day_of_week for s in slots}, {0, 2})
        self.assertEqual(slots[0].label, "Nhap mon Tu duy Tinh toan (TKB)")
        self.assertEqual(slots[0].location, "CS2 - Linh Trung, Thu Duc")
        self.assertEqual(slots[0].valid_from, "2026-10-05")
        self.assertEqual(slots[0].valid_until, "2027-01-01")  # 2026-12-31T23:59:59Z -> +07:00 next day

        self.assertEqual(len(ctx.busy_intervals), 2)
        exam_iv = ctx.busy_intervals[0]
        self.assertTrue(exam_iv.is_exam)
        # 02:00Z == 09:00+07:00
        self.assertIn("T09:00:00+07:00", exam_iv.start_datetime)

        # Wednesday inside recurring class window -> conflict
        wed_ev = EventNormalized(
            event_id="ev_wed",
            title="Workshop",
            start_datetime="2026-10-21T09:00:00+07:00",
            end_datetime="2026-10-21T10:30:00+07:00",
            online_mode="offline",
        )
        res = ics_adapter.check_conflict(wed_ev)
        self.assertTrue(res.has_conflict)
        self.assertEqual(res.conflict_type, "weekly_class")

    def test_conflict_boundaries_buffer_online_and_naive_timestamps(self) -> None:
        # Monday 2026-10-19 class ends at 11:00; buffer = 30m -> cutoff at 11:30
        inside_buffer = EventNormalized(
            event_id="e1",
            title="Seminar AI",
            start_datetime="2026-10-19T11:20:00+07:00",
            end_datetime="2026-10-19T12:30:00+07:00",
            online_mode="offline",
        )
        r1 = self.adapter.check_conflict(inside_buffer)
        self.assertTrue(r1.has_conflict)
        self.assertEqual(r1.conflict_type, "weekly_class")
        self.assertEqual(r1.travel_buffer_min, 30)

        outside_buffer = EventNormalized(
            event_id="e2",
            title="Seminar AI",
            start_datetime="2026-10-19T11:30:00+07:00",
            end_datetime="2026-10-19T12:30:00+07:00",
            online_mode="offline",
        )
        self.assertFalse(self.adapter.check_conflict(outside_buffer).has_conflict)

        # Online event overlaps regular class -> allowed (no conflict)
        online_regular = EventNormalized(
            event_id="e3",
            title="Webinar Online",
            start_datetime="2026-10-19T08:30:00+07:00",
            end_datetime="2026-10-19T10:00:00+07:00",
            online_mode="online",
        )
        self.assertFalse(self.adapter.check_conflict(online_regular).has_conflict)

        # Online event overlaps exam -> MUST conflict
        online_exam = EventNormalized(
            event_id="e4",
            title="Webinar Online",
            start_datetime="2026-10-20T09:30:00+07:00",
            end_datetime="2026-10-20T10:30:00+07:00",
            online_mode="online",
        )
        r4 = self.adapter.check_conflict(online_exam)
        self.assertTrue(r4.has_conflict)
        self.assertEqual(r4.conflict_type, "exam")

        # Naive timestamp -> handled cleanly without TypeError
        naive_ev = EventNormalized(
            event_id="e5",
            title="Naive Event",
            start_datetime="2026-10-19T08:30:00",
            end_datetime="2026-10-19T10:00:00",
            online_mode="offline",
        )
        self.assertTrue(self.adapter.check_conflict(naive_ev).has_conflict)

    def test_campus_detection_travel_and_fit_scoring(self) -> None:
        self.assertEqual(detect_campus("Hội trường I, 227 Nguyễn Văn Cừ, Q5"), "cs1_nvc")
        self.assertEqual(detect_campus("Cơ sở 2 Linh Trung, Thủ Đức"), "cs2_lt")
        self.assertIsNone(detect_campus("Nhà văn hóa Thanh Niên"))

        same_campus_free = EventNormalized(
            event_id="f1",
            title="Seminar CS1",
            start_datetime="2026-10-22T14:00:00+07:00",
            online_mode="offline",
            location="227 Nguyễn Văn Cừ",
            fee_vnd=0,
        )
        cross_campus_paid = EventNormalized(
            event_id="f2",
            title="Hội thao CS2",
            start_datetime="2026-10-22T14:00:00+07:00",
            online_mode="offline",
            location="Linh Trung, Thủ Đức",
            fee_vnd=20000,
        )
        exam_week_offline = EventNormalized(
            event_id="f3",
            title="Offline in Exam Week",
            start_datetime="2026-11-04T14:00:00+07:00",
            online_mode="offline",
            location="227 Nguyễn Văn Cừ",
            fee_vnd=0,
        )
        exam_week_online = EventNormalized(
            event_id="f4",
            title="Online in Exam Week",
            start_datetime="2026-11-04T14:00:00+07:00",
            online_mode="online",
            fee_vnd=0,
        )

        self.assertEqual(self.adapter.estimate_travel_minutes(same_campus_free, "cs1_nvc"), 10)
        self.assertEqual(self.adapter.estimate_travel_minutes(cross_campus_paid, "cs1_nvc"), 45)
        self.assertEqual(self.adapter.estimate_travel_minutes(exam_week_online, "cs1_nvc"), 0)

        fit_same = self.adapter.compute_fit(same_campus_free, home_campus="cs1_nvc")
        fit_cross = self.adapter.compute_fit(cross_campus_paid, home_campus="cs1_nvc", max_travel_min=30)
        fit_exam_off = self.adapter.compute_fit(exam_week_offline, home_campus="cs1_nvc")
        fit_exam_on = self.adapter.compute_fit(exam_week_online, home_campus="cs1_nvc")

        self.assertGreater(fit_same.fit_score, fit_cross.fit_score)
        self.assertTrue(fit_exam_off.in_exam_week)
        self.assertGreater(fit_exam_on.fit_score, fit_exam_off.fit_score)
        self.assertTrue(self.adapter.is_in_exam_week(datetime(2026, 11, 4, 14, 0, tzinfo=self.tz)))

        conflict_ev = EventNormalized(
            event_id="f5",
            title="Conflict Event",
            start_datetime="2026-10-19T08:00:00+07:00",
            online_mode="offline",
        )
        fit_conflict = self.adapter.compute_fit(conflict_ev)
        self.assertTrue(fit_conflict.has_conflict)
        self.assertEqual(fit_conflict.fit_score, 0.0)


if __name__ == "__main__":
    unittest.main()
