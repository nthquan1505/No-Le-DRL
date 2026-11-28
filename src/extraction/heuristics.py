"""Deterministic field sanitizer & offline heuristic extractor (`src/extraction/heuristics.py`).

Serves two roles:
1. Post-LLM Sanitizer (`sanitize_and_enrich_draft`): enforces `+07:00` ISO-8601 timestamps,
   chronological sanity, clean `registration_url` unwrapping (`None` instead of `'free'`/`'null'`),
   and missing field backfilling from raw text.
2. Offline Fallback (`heuristic_extract_item`): extracts structured event drafts with zero
   cloud API calls when LLM keys are absent or temporarily unreachable.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import unicodedata
from urllib.parse import parse_qs, unquote, urlencode, urlparse, urlunparse

from database import to_iso_utc7

from .contracts import (
    UTC7,
    EventClassification,
    ExtractedEventDraft,
    ExtractionMethod,
    RawIngestionItem,
    coerce_utc7_datetime,
)
from .prefilter import normalize_vi_ascii, reason_about_post

# ---------------------------------------------------------------------------
# URL Cleaning & Registration Link Extraction
# ---------------------------------------------------------------------------
PLACEHOLDER_URL_VALUES = frozenset(
    {
        "",
        "free",
        "null",
        "none",
        "n/a",
        "na",
        "tba",
        "tbd",
        "qr",
        "quét mã qr",
        "quet ma qr",
        "không cần đăng ký",
        "khong can dang ky",
    }
)

TRACKING_QUERY_KEYS = frozenset(
    {"fbclid", "__cft__", "__tn__", "utm_source", "utm_medium", "utm_campaign", "mibextid"}
)

RAW_URL_RE = re.compile(r"https?://[^\s<>()\"'\]\[]+", re.IGNORECASE)

PREFERRED_REG_HOST_HINTS: Tuple[str, ...] = (
    "forms.gle",
    "docs.google.com/forms",
    "forms.office.com",
    "bit.ly",
    "tinyurl",
    "linktr.ee",
    "myaloha.vn",
    "quickom",
    "eventbrite",
    "lu.ma",
)

NON_REGISTRATION_FB_PATH_RE = re.compile(
    r"^/(?:[^/]+/)?(?:posts|photos|videos|reel|permalink\.php|story\.php|hashtag)/?",
    re.IGNORECASE,
)


def clean_registration_url(raw_url: Any) -> Optional[str]:
    """Unwrap Facebook redirect links, strip tracking params, and reject non-URL placeholders."""
    if raw_url is None:
        return None
    text = str(raw_url).strip().rstrip(".,;:!?)>\"'")
    if text.lower() in PLACEHOLDER_URL_VALUES:
        return None
    if not text.startswith(("http://", "https://")):
        return None

    parsed = urlparse(text)
    host = (parsed.netloc or "").lower()

    # Unwrap https://l.facebook.com/l.php?u=...
    if host in ("l.facebook.com", "lm.facebook.com") and parsed.path.startswith("/l.php"):
        qs = parse_qs(parsed.query)
        inner = qs.get("u", [None])[0]
        if inner:
            return clean_registration_url(unquote(inner))

    # Strip tracking query parameters while keeping functional ones (e.g. Google Form viewform)
    if parsed.query:
        raw_pairs = parse_qs(parsed.query, keep_blank_values=True)
        filtered = {
            k: v
            for k, v in raw_pairs.items()
            if k.lower() not in TRACKING_QUERY_KEYS and not k.lower().startswith("__cft__")
        }
        new_query = urlencode(filtered, doseq=True)
        parsed = parsed._replace(query=new_query)

    cleaned = urlunparse(parsed).strip()
    return cleaned if cleaned.startswith(("http://", "https://")) else None


def find_registration_url_in_text(text: str, self_post_url: Optional[str] = None) -> Optional[str]:
    """Find the best candidate registration URL directly inside post/email text."""
    if not text:
        return None

    candidates: List[str] = []
    for match in RAW_URL_RE.finditer(text):
        cleaned = clean_registration_url(match.group(0))
        if not cleaned:
            continue
        if self_post_url and cleaned.rstrip("/") == clean_registration_url(self_post_url or ""):
            continue
        parsed = urlparse(cleaned)
        host = (parsed.netloc or "").lower()
        if "facebook.com" in host or "fb.com" in host:
            if NON_REGISTRATION_FB_PATH_RE.search(parsed.path or "") or parsed.path.strip("/") in ("",):
                continue
        candidates.append(cleaned)

    if not candidates:
        return None

    for url in candidates:
        low = url.lower()
        if any(hint in low for hint in PREFERRED_REG_HOST_HINTS):
            return url
    return candidates[0]


# ---------------------------------------------------------------------------
# Vietnamese Date/Time Parsing & Chronological Sanity
# ---------------------------------------------------------------------------
DATE_DMY_RE = re.compile(
    r"\b(\d{1,2})[/\-\.](\d{1,2})(?:[/\-\.](\d{2,4}))?\b"
)
TIME_HM_RE = re.compile(
    r"\b(\d{1,2})\s*(?:h|g|:)\s*(\d{2})?(?:\s*(am|pm|sáng|chiều|tối))?\b",
    re.IGNORECASE,
)
TIME_RANGE_RE = re.compile(
    r"\b(\d{1,2})\s*(?:h|g|:)\s*(\d{2})?\s*(?:-|–|đến|tới|to)\s*(\d{1,2})\s*(?:h|g|:)\s*(\d{2})?\b",
    re.IGNORECASE,
)

REG_LINE_CUE_RE = re.compile(
    r"(?:hạn\s*(?:chót\s*)?đăng\s*ký|đóng\s*đơn|thời\s*gian\s*đăng\s*ký|deadline\s*(?:đăng\s*ký)?)\s*[:\-–]?\s*([^\n]+)",
    re.IGNORECASE,
)
EVENT_TIME_LINE_RE = re.compile(
    r"(?:thời\s*gian(?:\s*tổ\s*chức|\s*diễn\s*ra)?|time|when)\s*[:\-–]\s*([^\n]+)",
    re.IGNORECASE,
)
LOCATION_LINE_RE = re.compile(
    r"(?:địa\s*điểm(?:\s*tổ\s*chức)?|hình\s*thức(?:\s*&\s*địa\s*điểm)?|nơi\s*diễn\s*ra|venue|location)\s*[:\-–]\s*([^\n]+)",
    re.IGNORECASE,
)
FEE_LINE_RE = re.compile(
    r"(?:lệ\s*phí|phí\s*tham\s*(?:gia|dự)|vé\s*tham\s*dự|chi\s*phí|fee)\s*[:\-–]\s*([^\n]+)",
    re.IGNORECASE,
)
AUDIENCE_LINE_RE = re.compile(
    r"(?:đối\s*tượng(?:\s*tham\s*(?:gia|dự))?)\s*[:\-–]\s*([^\n]+)",
    re.IGNORECASE,
)
AWARD_LINE_RE = re.compile(
    r"(?:quyền\s*lợi|giải\s*thưởng|cơ\s*cấu\s*giải\s*thưởng)\s*[:\-–]\s*([^\n]+)",
    re.IGNORECASE,
)


def _infer_year(day: int, month: int, ref_dt: datetime) -> int:
    year = ref_dt.year
    try:
        candidate = datetime(year, month, day, tzinfo=UTC7)
        # Handle December posts announcing January/February events next year
        if (ref_dt - candidate).days > 120:
            return year + 1
    except ValueError:
        pass
    return year


def _parse_date_time_from_snippet(
    snippet: str,
    ref_dt: datetime,
    default_hour: int = 0,
    default_minute: int = 0,
) -> Tuple[Optional[str], Optional[str]]:
    """Parse `(start_iso, end_iso)` from a Vietnamese schedule line."""
    m_date = DATE_DMY_RE.search(snippet)
    if not m_date:
        return None, None

    day = int(m_date.group(1))
    month = int(m_date.group(2))
    raw_year = m_date.group(3)
    if raw_year:
        year = int(raw_year)
        if year < 100:
            year += 2000
    else:
        year = _infer_year(day, month, ref_dt)

    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None, None

    # Remove the matched date substring before scanning for hours/minutes
    without_date = snippet[: m_date.start()] + " " + snippet[m_date.end() :]

    m_range = TIME_RANGE_RE.search(without_date)
    if m_range:
        sh, sm = int(m_range.group(1)), int(m_range.group(2) or 0)
        eh, em = int(m_range.group(3)), int(m_range.group(4) or 0)
        try:
            s_dt = datetime(year, month, day, min(sh, 23), min(sm, 59), tzinfo=UTC7)
            e_dt = datetime(year, month, day, min(eh, 23), min(em, 59), tzinfo=UTC7)
            if e_dt <= s_dt:
                return s_dt.isoformat(), None
            return s_dt.isoformat(), e_dt.isoformat()
        except ValueError:
            return None, None

    m_time = TIME_HM_RE.search(without_date)
    hour, minute = default_hour, default_minute
    if m_time:
        cand_h = int(m_time.group(1))
        cand_m = int(m_time.group(2) or 0)
        meridiem = (m_time.group(3) or "").lower()
        if meridiem in ("pm", "chiều", "tối") and cand_h < 12:
            cand_h += 12
        if 0 <= cand_h <= 23 and 0 <= cand_m <= 59:
            hour, minute = cand_h, cand_m

    try:
        dt = datetime(year, month, day, hour, minute, tzinfo=UTC7)
        return dt.isoformat(), None
    except ValueError:
        return None, None


def normalize_iso_datetime(val: Any) -> Optional[str]:
    """Normalize an ISO string or datetime to `+07:00` ISO-8601, or `None` if invalid."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("null", "none", "n/a", "tba", "tbd"):
        return None
    return to_iso_utc7(s)


# ---------------------------------------------------------------------------
# Rule-based Tag, Keyword & Lifecycle Inference
# ---------------------------------------------------------------------------
TOPIC_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("AI", re.compile(r"\b(?:ai|artificial intelligence|machine learning|deep learning|llm|genai)\b", re.I)),
    ("lập trình", re.compile(r"\b(?:lập trình|coding|hackathon|icpc|algorithm|thuật toán|software)\b", re.I)),
    ("học thuật", re.compile(r"\b(?:học thuật|academic|nghiên cứu khoa học|nckh|olympic|chuyên đề)\b", re.I)),
    ("khoa học", re.compile(r"\b(?:khoa học|stem|vật lý|hóa học|sinh học|toán học|công nghệ)\b", re.I)),
    ("hướng nghiệp", re.compile(r"\b(?:hướng nghiệp|career|tuyển dụng|cv|phỏng vấn|việc làm|job fair|thực tập)\b", re.I)),
    ("thiện nguyện", re.compile(r"\b(?:thiện nguyện|tình nguyện|mùa hè xanh|xuân tình nguyện|chủ nhật xanh|hiến máu)\b", re.I)),
    ("quyên góp", re.compile(r"\b(?:quyên góp|gây quỹ|ủng hộ|fundraising)\b", re.I)),
    ("thể thao", re.compile(r"\b(?:thể thao|hội thao|bóng đá|bóng chuyền|cầu lông|chạy bộ|marathon|cờ vua|esports)\b", re.I)),
    ("văn nghệ", re.compile(r"\b(?:văn nghệ|âm nhạc|ca nhạc|acoustic|concert|hội diễn)\b", re.I)),
    ("pháp luật", re.compile(r"\b(?:pháp luật|hiến pháp|luật|an toàn giao thông|phòng chống ma túy)\b", re.I)),
    ("chính trị - xã hội", re.compile(r"\b(?:mác\s*[\-–]?\s*lênin|tư tưởng hồ chí minh|lịch sử đảng|đoàn thanh niên)\b", re.I)),
    ("kỹ năng", re.compile(r"\b(?:kỹ năng mềm|kỹ năng|quản lý thời gian|giao tiếp|thuyết trình)\b", re.I)),
    ("khởi nghiệp", re.compile(r"\b(?:khởi nghiệp|startup|đổi mới sáng tạo|innostar)\b", re.I)),
    ("cộng đồng", re.compile(r"\b(?:sinh viên|cộng đồng|clb|câu lạc bộ|ngày hội)\b", re.I)),
)

FORMAT_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("workshop", re.compile(r"\b(?:workshop|thực hành|hands-on|bootcamp)\b", re.I)),
    ("seminar", re.compile(r"\b(?:seminar|webinar|hội thảo|chuyên đề)\b", re.I)),
    ("talkshow", re.compile(r"\b(?:talkshow|talk\s+show)\b", re.I)),
    ("tọa đàm", re.compile(r"\b(?:tọa đàm|toạ đàm|giao lưu|chia sẻ)\b", re.I)),
    ("cuộc thi", re.compile(r"\b(?:cuộc thi|hội thi|giải đấu|hackathon|olympic|contest)\b", re.I)),
    ("ngày hội", re.compile(r"\b(?:ngày hội|festival|fair|hội trại)\b", re.I)),
    ("hiến máu", re.compile(r"\b(?:hiến máu)\b", re.I)),
    ("tình nguyện", re.compile(r"\b(?:tình nguyện|thiện nguyện|ra quân|chiến dịch)\b", re.I)),
    ("khảo sát", re.compile(r"\b(?:khảo sát|survey)\b", re.I)),
    ("tuyển cộng tác viên", re.compile(r"\b(?:tuyển cộng tác viên|tuyển ctv|tuyển tình nguyện viên)\b", re.I)),
    ("online", re.compile(r"\b(?:online|trực tuyến|zoom|google meet|ms teams|microsoft teams|livestream)\b", re.I)),
    ("offline", re.compile(r"\b(?:offline|trực tiếp|hội trường|giảng đường|phòng|cơ sở)\b", re.I)),
    ("đồng đội", re.compile(r"\b(?:đồng đội|theo đội|nhóm\s+\d+|team)\b", re.I)),
    ("cá nhân", re.compile(r"\b(?:cá nhân|mỗi thí sinh)\b", re.I)),
)

LITERAL_KEYWORD_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:workshop|seminar|webinar|talkshow|tọa đàm|hội thảo|cuộc thi|hội thi|ngày hội|hackathon)\b", re.I),
    re.compile(r"\b(?:tình nguyện|thiện nguyện|hiến máu|mùa hè xanh|xuân tình nguyện|chủ nhật xanh|gây quỹ)\b", re.I),
    re.compile(r"\b(?:điểm rèn luyện|giấy chứng nhận|học thuật|nghiên cứu khoa học|hướng nghiệp|kỹ năng mềm)\b", re.I),
    re.compile(r"\b(?:trực tuyến|trực tiếp|cộng tác viên|ban tổ chức|sinh viên)\b", re.I),
)

CANCEL_ASCII_RE = re.compile(
    r"(?:\btam\s+hoan\b|\bhoan\s+to\s+chuc\b|\bhuy\s+to\s+chuc\b|\bhuy\s+su\s+kien\b|\bthong\s+bao\s+huy\b|\bthong\s+bao\s+hoan\b|\bcancelled\b|\bcanceled\b)",
    re.I,
)
UPDATE_ASCII_RE = re.compile(
    r"(?:\bgia\s+han\s+dang\s+ky\b|\bkeo\s+dai\s+thoi\s+gian\s+dang\s+ky\b|\bdoi\s+lich\b|\bdoi\s+thoi\s+gian\b|\bthay\s+doi\s+dia\s+diem\b|\bthay\s+doi\s+thoi\s+gian\b|\bcap\s+nhat\s+dia\s+diem\b|\brescheduled\b)",
    re.I,
)

LEADING_NOISE_TITLE_RE = re.compile(
    r"^(?:[\W_]+|\[\s*(?:thông\s*báo|update|cập\s*nhật|nhắc\s*nhở|reminder|hủy|tạm\s*hoãn)\s*\])+",
    re.IGNORECASE,
)

LOGISTICS_LINE_SKIP_RE = re.compile(
    r"^(?:[\W_]*)(?:thời\s*gian|địa\s*điểm|hạn\s*(?:chót\s*)?đăng\s*ký|đóng\s*đơn|"
    r"link\s*đăng\s*ký|đăng\s*ký\s*(?:ngay\s*)?tại|form\s*đăng\s*ký|quét\s*mã\s*qr|"
    r"mọi\s*thắc\s*mắc|liên\s*hệ|fanpage|hotline|email\s*:|time\s*:|venue\s*:|location\s*:|registration\s*:)",
    re.IGNORECASE,
)
GREETING_LINE_SKIP_RE = re.compile(
    r"^(?:thân\s*chào|kính\s*chào|chào\s*các\s*(?:bạn|em)|dear\s+|hello\s+)",
    re.IGNORECASE,
)


def clean_summary_description(raw_desc: str) -> str:
    """Ensure LLM/heuristic `description` is plain 2-3 sentence prose without headings or bullets."""
    text = unicodedata.normalize("NFC", raw_desc or "").strip()
    if not text:
        return ""
    lines: List[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"^(?:[\s#*•\-–]*)(?:tóm\s*tắt|summary|description)\s*:\s*", "", raw_line, flags=re.I).strip()
        line = re.sub(r"^\s*(?:[#*•\-–]+|\d+[\.\)])\s*", "", line).strip()
        if line:
            lines.append(line)
    joined = re.sub(r"\s+", " ", " ".join(lines)).strip()
    return joined


def build_fallback_description(full_text: str, title: str = "") -> str:
    """Build a concise 2-3 sentence offline summary omitting greetings, URLs, and duplicate logistics."""
    norm_title = clean_summary_description(title).lower()
    kept_lines: List[str] = []
    for raw_line in (full_text or "").splitlines():
        s = RAW_URL_RE.sub("", raw_line).strip()
        # Strip decorative symbols/emojis at line boundaries while keeping letters/numbers/punctuation
        s = re.sub(r"^[^\wÀ-ỹ\[\(]+", "", s).strip()
        if len(s) < 15 or s.startswith("#"):
            continue
        if norm_title and s.lower().strip(" !?.") == norm_title.strip(" !?."):
            continue
        if GREETING_LINE_SKIP_RE.search(s) or LOGISTICS_LINE_SKIP_RE.search(s):
            continue
        if not s.endswith((".", "!", "?")):
            s = s.rstrip(" ,;:-–") + "."
        kept_lines.append(s)
        if len(kept_lines) >= 3:
            break

    if not kept_lines:
        compact = re.sub(r"\s+", " ", RAW_URL_RE.sub("", full_text or "")).strip()
        return compact[:300].strip()

    summary = re.sub(r"\s+", " ", " ".join(kept_lines)).strip()
    return summary[:450].rsplit(" ", 1)[0] + ("..." if len(summary) > 450 else "")


def extract_verbatim_keywords(text: str, existing: Optional[Sequence[str]] = None) -> List[str]:
    """Keep verbatim keywords that appear in `text` and backfill from regex matches if sparse."""
    norm_text = unicodedata.normalize("NFC", text or "")
    low_text = norm_text.lower()

    results: List[str] = []
    seen: set[str] = set()

    for kw in existing or []:
        clean_kw = unicodedata.normalize("NFC", str(kw or "")).strip()
        if clean_kw and clean_kw.lower() in low_text and clean_kw.lower() not in seen:
            seen.add(clean_kw.lower())
            results.append(clean_kw)

    if len(results) < 5:
        for pat in LITERAL_KEYWORD_PATTERNS:
            for m in pat.finditer(norm_text):
                phrase = m.group(0).strip()
                if phrase.lower() not in seen:
                    seen.add(phrase.lower())
                    results.append(phrase)
                    if len(results) >= 10:
                        return results
    return results[:12]


def infer_topics_and_formats(text: str, topics: Sequence[str], formats: Sequence[str]) -> Tuple[List[str], List[str]]:
    out_topics = [str(x).strip() for x in topics if str(x).strip()]
    out_formats = [str(x).strip() for x in formats if str(x).strip()]
    seen_t = {x.lower() for x in out_topics}
    seen_f = {x.lower() for x in out_formats}

    for label, pat in TOPIC_PATTERNS:
        if pat.search(text) and label.lower() not in seen_t:
            seen_t.add(label.lower())
            out_topics.append(label)

    for label, pat in FORMAT_PATTERNS:
        if pat.search(text) and label.lower() not in seen_f:
            seen_f.add(label.lower())
            out_formats.append(label)

    return out_topics[:6], out_formats[:6]


def clean_title(raw_title: str, fallback_text: str) -> str:
    """Clean leading decorative emojis/tags or derive a clean title from the first meaningful line."""
    candidate = unicodedata.normalize("NFC", (raw_title or "")).strip()
    if not candidate:
        for line in fallback_text.splitlines():
            s = line.strip()
            if len(s) >= 8 and not s.lower().startswith(("chào ", "thân chào", "kính chào", "dear ")):
                candidate = s
                break
    cleaned = LEADING_NOISE_TITLE_RE.sub("", candidate).strip(" -–|:🔥📢📣🌟✨🎉")
    return (cleaned or candidate)[:300].strip()


def infer_lifecycle_classification(text: str, raw_cls: Optional[str]) -> EventClassification:
    """Validate or infer `EventClassification` from raw LLM output + explicit text cues."""
    ascii_text = normalize_vi_ascii(text)
    valid: set[str] = {
        "new_or_active_event",
        "event_update",
        "event_cancellation",
        "not_event",
    }
    if raw_cls == "not_event":
        # Guard against LLM misclassifying explicit cancellation/update notices as not_event
        if CANCEL_ASCII_RE.search(ascii_text):
            return "event_cancellation"
        if UPDATE_ASCII_RE.search(ascii_text):
            return "event_update"
        return "not_event"

    if CANCEL_ASCII_RE.search(ascii_text):
        return "event_cancellation"
    if raw_cls in valid:
        return raw_cls  # type: ignore[return-value]
    if UPDATE_ASCII_RE.search(ascii_text):
        return "event_update"
    return "new_or_active_event"


def _extract_regex_fields(text: str, ref_dt: datetime) -> Dict[str, Any]:
    """Extract fallback schedule, location, fee, audience, and award directly from raw text."""
    ev_start: Optional[str] = None
    ev_end: Optional[str] = None
    reg_end: Optional[str] = None
    location = ""
    fee = ""
    award = ""
    audiences: List[str] = []

    m_ev_line = EVENT_TIME_LINE_RE.search(text)
    if m_ev_line:
        ev_start, ev_end = _parse_date_time_from_snippet(m_ev_line.group(1), ref_dt, default_hour=8, default_minute=0)

    m_reg_line = REG_LINE_CUE_RE.search(text)
    if m_reg_line:
        reg_end, _ = _parse_date_time_from_snippet(m_reg_line.group(1), ref_dt, default_hour=23, default_minute=59)

    if not ev_start:
        ev_start, ev_end = _parse_date_time_from_snippet(text, ref_dt, default_hour=8, default_minute=0)

    m_loc = LOCATION_LINE_RE.search(text)
    if m_loc:
        location = m_loc.group(1).strip(" .;-–")

    m_fee = FEE_LINE_RE.search(text)
    if m_fee:
        fee = m_fee.group(1).strip(" .;-–")
    elif re.search(r"\bmiễn\s*phí\b|\bfree\s+entry\b", text, re.I):
        fee = "Miễn phí"

    m_aud = AUDIENCE_LINE_RE.search(text)
    if m_aud:
        audiences = [m_aud.group(1).strip(" .;-–")]
    elif re.search(r"\bsinh\s*viên\b", text, re.I):
        audiences = ["sinh viên"]

    m_award = AWARD_LINE_RE.search(text)
    if m_award:
        award = m_award.group(1).strip(" .;-–")
    else:
        perks: List[str] = []
        if re.search(r"\bđiểm\s*rèn\s*luyện\b|\bđrl\b", text, re.I):
            perks.append("Điểm rèn luyện")
        if re.search(r"\bgiấy\s*chứng\s*nhận\b|\bcertificate\b", text, re.I):
            perks.append("Giấy chứng nhận")
        award = ", ".join(perks)

    return {
        "event_start_datetime": ev_start,
        "event_end_datetime": ev_end,
        "reg_end_datetime": reg_end,
        "location": location,
        "fee": fee,
        "target_audience": audiences,
        "award": award,
    }


def sanitize_and_enrich_draft(
    raw_data: Mapping[str, Any],
    item: RawIngestionItem,
    method: ExtractionMethod,
) -> ExtractedEventDraft:
    """Validate, sanitize, and backfill an LLM or heuristic extraction dictionary."""
    full_text = item.full_text
    ref_dt = coerce_utc7_datetime(item.published_at)
    cls = infer_lifecycle_classification(full_text, str(raw_data.get("classification") or "").strip() or None)

    gate = reason_about_post(item.content, subject=item.subject)
    if (
        not gate.is_candidate
        and cls not in ("event_update", "event_cancellation")
        and gate.reason.startswith("disqualified:")
        and not gate.reason.startswith("disqualified:insufficient_event_signals")
    ):
        cls = "not_event"

    if cls == "not_event":
        return ExtractedEventDraft(
            source_id=item.source_id,
            classification="not_event",
            reason=str(raw_data.get("reason") or gate.reason or "classified_not_event").strip(),
            post_url=item.post_url,
            images_url=list(item.images_url),
            extraction_method=method,
        )

    fallback = _extract_regex_fields(full_text, ref_dt)

    title = clean_title(str(raw_data.get("title") or ""), item.subject or full_text)
    desc = clean_summary_description(str(raw_data.get("description") or ""))
    if not desc:
        desc = build_fallback_description(full_text, title=title)

    reg_start = normalize_iso_datetime(raw_data.get("reg_start_datetime"))
    reg_end = normalize_iso_datetime(raw_data.get("reg_end_datetime")) or fallback["reg_end_datetime"]
    ev_start = normalize_iso_datetime(raw_data.get("event_start_datetime")) or fallback["event_start_datetime"]
    ev_end = normalize_iso_datetime(raw_data.get("event_end_datetime")) or fallback["event_end_datetime"]

    # Chronological sanity checks
    if ev_start and ev_end and ev_end < ev_start:
        ev_end = None
    if reg_start and reg_end and reg_end < reg_start:
        reg_start = None

    location = str(raw_data.get("location") or fallback["location"] or "").strip()
    topics, formats = infer_topics_and_formats(
        full_text,
        raw_data.get("topic") or [],
        raw_data.get("format") or [],
    )

    is_online = bool(raw_data.get("is_online", False))
    if not is_online and "online" in [f.lower() for f in formats] and "offline" not in [f.lower() for f in formats]:
        is_online = True
    if re.search(r"^(?:online|trực\s*tuyến|zoom|google\s*meet|ms\s*teams)", location, re.I):
        is_online = True

    reg_url = clean_registration_url(raw_data.get("registration_url"))
    if reg_url is None:
        reg_url = find_registration_url_in_text(full_text, self_post_url=item.post_url)

    keywords = extract_verbatim_keywords(full_text, raw_data.get("keywords") or [])
    audiences = [str(x).strip() for x in (raw_data.get("target_audience") or fallback["target_audience"]) if str(x).strip()]
    fee = str(raw_data.get("fee") or fallback["fee"] or "").strip()
    award = str(raw_data.get("award") or fallback["award"] or "").strip()
    organizer = str(raw_data.get("organizer") or "").strip()

    return ExtractedEventDraft(
        source_id=item.source_id,
        classification=cls,
        reason=str(raw_data.get("reason") or "extracted").strip(),
        title=title,
        description=desc,
        keywords=keywords,
        is_online=is_online,
        reg_start_datetime=reg_start,
        reg_end_datetime=reg_end,
        event_start_datetime=ev_start,
        event_end_datetime=ev_end,
        location=location,
        fee=fee,
        target_audience=audiences,
        topic=topics,
        format=formats,
        award=award,
        organizer=organizer,
        registration_url=reg_url,
        post_url=item.post_url,
        images_url=list(item.images_url),
        extraction_method=method,
    )


def heuristic_extract_item(item: RawIngestionItem) -> ExtractedEventDraft:
    """Full deterministic offline extractor when no LLM API keys are available."""
    gate = reason_about_post(item.content, subject=item.subject)
    if not gate.is_candidate:
        return ExtractedEventDraft(
            source_id=item.source_id,
            classification="not_event",
            reason=gate.reason,
            post_url=item.post_url,
            images_url=list(item.images_url),
            extraction_method="heuristic",
        )

    cls = infer_lifecycle_classification(item.full_text, None)
    return sanitize_and_enrich_draft(
        {"classification": cls, "reason": f"heuristic_fallback ({gate.reason})"},
        item=item,
        method="heuristic",
    )
