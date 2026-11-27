"""System instructions and server-side JSON Schema for `[EXT]` (`src/utility/extraction/prompts.py`)."""

from __future__ import annotations

import json
from typing import Any, Dict, Sequence

from .contracts import ALLOWED_FORMATS, ALLOWED_TOPICS, RawIngestionItem

# Dedicated summarization specification shared with the CLF (Classification) module
# so `events.description` is concise for students and leakage-free / high-signal for E5 embeddings.
DESCRIPTION_SUMMARIZATION_GUIDELINES = """Read the entire event post and write a concise, informative summary STRICTLY IN VIETNAMESE (tiếng Việt) for the `description` field.
Your goal is to help students quickly understand what the event is, what participants will actually do or experience, and what concrete value they may receive.

Summarization guidelines for `description`:
1. Synthesize information from the entire post. Do not simply copy the introductory text or reproduce the original wording. The introduction may contain important information, so do not ignore it.
2. Prioritize the following information, when explicitly available:
   - The event's main purpose, subject, or central theme.
   - Its main activities and what participants will actually do, experience, or learn.
   - The role of participants, organizers, or volunteers, when clearly stated.
   - Concrete benefits or outcomes, such as acquired skills, knowledge, products, networking opportunities, prizes, certificates, or other clearly announced recognition.
3. Focus on specific and informative details rather than generic promotional statements. Preserve the actual activities and meaningful distinctions between different types of participation.
4. Include benefits and rewards only when they are explicitly stated and relevant. Do not interpret certificates, awards, participation recognition, or eligibility for student achievement criteria as guaranteed Điểm rèn luyện points unless the post explicitly states that.
5. Preserve important factual distinctions. For example:
   - Attending an event is not the same as organizing or volunteering at it.
   - Participating in a fundraising workshop is not necessarily the same as directly performing charitable work.
   - An announced benefit or recognition is not necessarily a guaranteed outcome.
6. Only use information supported by the post. Never invent, infer, exaggerate, or assume event activities, participant responsibilities, eligibility, benefits, or outcomes.
7. If a detail is unclear or not provided, omit it rather than guessing. Do not fill missing information with generic statements.
8. Avoid unnecessary promotional language, repeated ideas, registration instructions, and administrative details that are already stored in separate event fields, unless they are essential to understanding the event.

Output requirements for `description`:
- Write normally 2–3 concise sentences, using fewer sentences when the source contains limited information.
- ALWAYS write `description` STRICTLY IN VIETNAMESE (tiếng Việt), even when the original post or email is written completely in English or mixed languages. Translate and synthesize English content into natural Vietnamese while preserving proper names (event titles, speaker/professor names, organization names) and standard technical terms (e.g. "Agentic AI", "Multiagent Learning", "PhD", "TOEIC"). NEVER write English sentences in `description`.
- Make the summary natural, clear, and understandable to students.
- Put only the plain-prose summary inside `description`. Do not include headings, bullet points, explanations, classifications, category labels, rubric codes (e.g. "Mục 1", "Điều 5"), or additional commentary."""

SYSTEM_INSTRUCTION = f"""You are a precise university event extraction and summarization assistant for No-Le-DRL (VNU-HCM / HCMUS).
Analyze each Facebook post or Email announcement (Vietnamese or English) and return a JSON array strictly matching the response schema.

1. EVENT CLASSIFICATION (`classification`):
- `"new_or_active_event"`: Announces registration or invites students to join a workshop, seminar, talkshow, competition, volunteering campaign, blood donation, festival, rewarded/DRL survey, or event collaborator/volunteer recruitment.
- `"event_update"`: Updates an existing event (deadline extension, rescheduled date/time, venue change, newly opened link, final reminder).
- `"event_cancellation"`: Indefinitely postpones or cancels an event.
- `"not_event"`:
  * Post-event recaps & summaries (`[RECAP]`, `tổng kết`, `đã diễn ra tại...`, `vừa qua ... đã tham gia`, `Ngày hội/Chương trình/Hành trình khép lại`, `cùng nhìn lại những khoảnh khắc`, `Cảm ơn Ban Giám khảo`, `Tri ân Nhà tài trợ`), EVEN IF the post repeats the event's title, date, or venue!
  * Behind-the-scenes / rehearsal / companion posts (`BEHIND THE FINAL STAGE`, `Behind the scenes`, `hậu trường`, `những giờ luyện tập miệt mài`, `khoảnh khắc trong phòng tập`, or `Giới thiệu Ban Giám khảo` without a registration link), EVEN IF a copy-pasted event schedule/venue footer (`THÔNG TIN CUỘC THI...`) is attached at the bottom!
  * Post-event attendance/record verification notices (`Kiểm tra thông tin tham gia`, `đăng ký xác minh bổ sung` for finished activities), facility handover news (`Khánh thành và bàn giao công trình`), congratulations/honors, internal union committee congresses, semester exam timetables, routine academic/facility/dormitory notices, full-time corporate job postings without a student workshop/talkshow, or pure scholarship dossier notices without a live info session/webinar.

2. DESCRIPTION SUMMARIZATION (`description`):
{DESCRIPTION_SUMMARIZATION_GUIDELINES}

3. STRUCTURED FIELD EXTRACTION RULES:
- `title`: Concise official title of the event (strip decorative leading tags like "[RECAP]" or leading emojis).
- `keywords`: Extract 5–12 VERBATIM key phrases directly from the raw post text (do not invent synthetic labels).
- Datetimes (`reg_start_datetime`, `reg_end_datetime`, `event_start_datetime`, `event_end_datetime`):
  * Always normalize to ISO-8601 in Vietnam timezone `+07:00` (`YYYY-MM-DDTHH:MM:SS+07:00`).
  * Resolve relative dates ("hôm nay", "ngày mai", "this Friday") and missing years against the item's `published_at`.
  * If only a date is given without a specific clock time, use `00:00:00+07:00` for start dates or `23:59:00+07:00` for deadlines. Return `null` when absent.
- `location`: Venue (room, lecture hall, campus, street address, or online platform such as Zoom/Google Meet/MS Teams). Empty string `""` if unknown.
- `is_online`: `true` if held online (Zoom, Meet, Teams, Livestream, online form), otherwise `false`.
- `registration_url`: Only a genuine `http://` or `https://` registration URL. If no pre-registration is required or registration is QR-only on the poster, return `null` (NEVER return `"free"`, `"none"`, or `"null"` as strings).
- `topic`: Relevant topic tags (prefer values from: {", ".join(ALLOWED_TOPICS)}).
- `format`: Relevant format tags (prefer values from: {", ".join(ALLOWED_FORMATS)}).
- `award`: Explicitly stated benefits/rewards only (do not assume certificates imply guaranteed Điểm rèn luyện unless stated), or `""`.
- `organizer`: Organizing club/faculty/unit if stated in the post, or `""`.
"""

# Gemini REST API `responseSchema` (OpenAPI 3.0 subset supported by `v1beta` generateContent)
GEMINI_ITEM_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "source_id": {"type": "STRING"},
        "classification": {
            "type": "STRING",
            "enum": [
                "new_or_active_event",
                "event_update",
                "event_cancellation",
                "not_event",
            ],
        },
        "reason": {"type": "STRING"},
        "title": {"type": "STRING"},
        "description": {
            "type": "STRING",
            "description": (
                "2-3 concise plain-prose sentences STRICTLY IN VIETNAMESE (tiếng Việt) synthesizing "
                "event purpose, actual participant activities/roles, and explicitly stated benefits "
                "(translate/summarize into Vietnamese even if the post is in English, keeping proper names "
                "and technical terms). No English sentences, headings, bullets, promotional fluff, or rubric codes."
            ),
        },
        "keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
        "is_online": {"type": "BOOLEAN"},
        "reg_start_datetime": {"type": "STRING", "nullable": True},
        "reg_end_datetime": {"type": "STRING", "nullable": True},
        "event_start_datetime": {"type": "STRING", "nullable": True},
        "event_end_datetime": {"type": "STRING", "nullable": True},
        "location": {"type": "STRING"},
        "fee": {"type": "STRING"},
        "target_audience": {"type": "ARRAY", "items": {"type": "STRING"}},
        "topic": {"type": "ARRAY", "items": {"type": "STRING"}},
        "format": {"type": "ARRAY", "items": {"type": "STRING"}},
        "award": {"type": "STRING"},
        "organizer": {"type": "STRING"},
        "registration_url": {"type": "STRING", "nullable": True},
    },
    "required": [
        "source_id",
        "classification",
        "reason",
        "title",
        "description",
        "keywords",
        "is_online",
        "location",
        "fee",
        "target_audience",
        "topic",
        "format",
        "award",
        "organizer",
    ],
}

GEMINI_BATCH_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "ARRAY",
    "items": GEMINI_ITEM_SCHEMA,
}


def build_batch_prompt(items: Sequence[RawIngestionItem]) -> str:
    """Format 1..N `RawIngestionItem` records as a compact JSON array for batch extraction."""
    payload = [
        {
            "source_id": item.source_id,
            "source_type": item.source_type,
            "source_name": item.source_name,
            "published_at": item.published_at_iso,
            "subject": item.subject,
            "text": item.full_text,
        }
        for item in items
    ]
    return (
        "Analyze the announcements below and return exactly one JSON object per `source_id` "
        "in the exact input order:\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
