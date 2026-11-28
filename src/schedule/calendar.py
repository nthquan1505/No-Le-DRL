"""Convert Google Calendar Events.list items into a schedule snapshot."""

from dataclasses import replace

from .engine import BusyInterval, ScheduleContext


class CalendarReadError(RuntimeError):
    """Calendar snapshot could not be read."""


def read_calendar(service, context: ScheduleContext, *, time_min, time_max, calendar_id="primary",
                  exam_event_ids=(), exclude_event_ids=()):
    """Read expanded instances using a caller-authorized Google Calendar service."""
    items = []
    token = None
    while True:
        try:
            page = service.events().list(
                calendarId=calendar_id, timeMin=time_min.isoformat(), timeMax=time_max.isoformat(),
                singleEvents=True, showDeleted=False, pageToken=token,
                fields="nextPageToken,items(id,status,transparency,summary,location,start,end)",
            ).execute()
        except Exception as error:
            raise CalendarReadError("Không thể đồng bộ Google Calendar") from error
        items.extend(page.get("items", ()))
        token = page.get("nextPageToken")
        if not token:
            return add_calendar_events(
                context, items, exam_event_ids=exam_event_ids, exclude_event_ids=exclude_event_ids,
            )


def add_calendar_events(context: ScheduleContext, items, *, exam_event_ids=(), exclude_event_ids=()):
    """Merge expanded instances returned by an authorized calendar reader."""
    intervals = list(context.busy_intervals)
    flexible = list(getattr(context, "flexible_intervals", ()))
    exams, excluded = set(exam_event_ids), set(exclude_event_ids)
    seen = set()
    for item in items:
        event_id = item.get("id")
        if event_id in excluded or item.get("status") == "cancelled":
            continue
        first, last = item.get("start", {}), item.get("end", {})
        start = first.get("dateTime") or first.get("date")
        end = last.get("dateTime") or last.get("date")
        key = (event_id, start, end)
        if not start or not end or key in seen:
            continue
        seen.add(key)
        target = flexible if item.get("transparency") == "transparent" else intervals
        target.append(BusyInterval(
            start, end, item.get("summary") or "Lịch bận", item.get("location"), event_id in exams,
        ))
    if isinstance(context, ScheduleContext):
        return replace(context, busy_intervals=intervals, flexible_intervals=flexible)
    # Accept the existing upstream Module 3 dataclass without changing its schema.
    return ScheduleContext(intervals, list(context.exam_weeks), context.default_travel_buffer_min,
                           list(getattr(context, "weekly_slots", ())), flexible)
