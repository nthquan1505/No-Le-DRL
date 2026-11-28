"""Stage-1 deterministic Vietnamese/English pre-filter (`src/extraction/prefilter.py`).

Saves LLM API quota by rejecting unambiguous non-events (recaps, congratulations,
exam timetables, room maintenance, pure corporate job ads) while guaranteeing high
recall on new events, updates/deadline extensions, and cancellations.
"""

from __future__ import annotations

import re
from typing import List, Tuple
import unicodedata

from .contracts import PreFilterDecision


def normalize_vi_ascii(text: str) -> str:
    """Lowercase, map `đ -> d`, strip accents, and collapse whitespace."""
    raw = (text or "").lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", raw)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", stripped).strip()


# ---------------------------------------------------------------------------
# Lifecycle cues (ALWAYS pass Stage 1 so updates & cancellations are never missed)
# ---------------------------------------------------------------------------
LIFECYCLE_SIGNAL_RE = re.compile(
    r"(?:"
    r"\btam\s+hoan\b|\bhoan\s+to\s+chuc\b|\bhuy\s+to\s+chuc\b|\bhuy\s+su\s+kien\b"
    r"|\bthong\s+bao\s+hoan\b|\bthong\s+bao\s+huy\b|\bdoi\s+lich\b|\bdoi\s+thoi\s+gian\b"
    r"|\bthay\s+doi\s+dia\s+diem\b|\bthay\s+doi\s+thoi\s+gian\b|\bcap\s+nhat\s+dia\s+diem\b"
    r"|\bgia\s+han\s+dang\s+ky\b|\bkeo\s+dai\s+thoi\s+gian\s+dang\s+ky\b"
    r"|\bpostponed\b|\bcancelled\b|\bcanceled\b|\brescheduled\b"
    r")",
    re.IGNORECASE,
)

# Active registration CTA that overrides soft recap/congratulation false positives
ACTIVE_REGISTRATION_CTA_RE = re.compile(
    r"(?:"
    r"\bmo\s+don\s+dang\s+ky\b|\bmo\s+cong\s+dang\s+ky\b|\blink\s+dang\s+ky\s*:"
    r"|\bform\s+dang\s+ky\s*:|\bbieu\s+mau\s+dang\s+ky\s*:|\bdang\s+ky\s+tham\s+(?:gia|du)\b"
    r"|forms\.gle/|docs\.google\.com/forms"
    r")",
    re.IGNORECASE,
)

GENUINE_REG_URL_RE = re.compile(
    r"https?://(?!(?:www\.|m\.)?(?:facebook\.com|fb\.com)/(?:[^/\s]+/)?(?:posts|videos|reel|photos|permalink|story|\s|$))"
    r"(?!(?:www\.)?(?:tiktok\.com|instagram\.com|youtube\.com))[^\s<>()\"'\]]+",
    re.IGNORECASE,
)

FOOTER_SPLIT_RE = re.compile(
    r"\n\s*[_\-=]{5,}\s*\n[\s\S]*?(?:thong\s+tin\s+(?:cuoc\s+thi|chuong\s+trinh|vong|su\s+kien))[\s\S]*$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Step 1: Disqualification patterns (evaluated on accent-stripped text)
# ---------------------------------------------------------------------------
HARD_NON_EVENT_RE = re.compile(
    r"(?:"
    r"\[\s*recap\b|\brecap\s+(?:vong|workshop|tap\s+huan|chuong\s+trinh|ngay\s+hoi|su\s+kien)\b"
    r"|\bcam\s+on\s+(?:quy\s+)?ban\s+giam\s+khao\b|\btri\s+an\s+(?:nha\s+tai\s+tro|ban\s+giam\s+khao|ban\s+chap\s+hanh)\b"
    r"|\bkiem\s+tra\s+thong\s+tin\s+(?:va\s+minh\s+chung\s+)?tham\s+gia\b|\bdang\s+ky\s+xac\s+minh\s+bo\s+sung\b"
    r"|\bkhanh\s+thanh\s+va\s+ban\s+giao\s+cong\s+trinh\b|\ble\s+ban\s+giao\s+cong\s+trinh\b"
    r")",
    re.IGNORECASE,
)

BEHIND_SCENES_OR_COMPANION_RE = re.compile(
    r"(?:"
    r"\bbehind\s+the\s+(?:scenes?|final\s+stage|stage)\b|\bhau\s+truong\b"
    r"|\bphia\s+sau\s+(?:nhung\s+man\s+trinh\s+dien|anh\s+den|san\s+khau)\b"
    r"|\bnhung\s+gio\s+luyen\s+tap\s+miet\s+mai\b|\bkhoanh\s+khac\s+trong\s+phong\s+tap\b"
    r"|^[^\w\n]{0,12}[^\n]{0,75}\bgioi\s+thieu\s+(?:hoi\s+dong\s+)?ban\s+giam\s+khao\b"
    r")",
    re.IGNORECASE,
)

RECAP_OR_CONGRATS_RE = re.compile(
    r"(?:"
    r"\btong\s+ket\s+(?:chuong\s+trinh|tieu\s+ban|ket\s+qua|mot\s+hanh\s+trinh)\b"
    r"|\b(?:ngay\s+hoi|chuong\s+trinh|su\s+kien|cuoc\s+thi|hanh\s+trinh|mua\s+giai|dem\s+chung\s+ket)\s+(?:da\s+)?(?:chinh\s+thuc\s+)?khep\s+lai\b"
    r"|\bda\s+chinh\s+thuc\s+khep\s+lai\b|\bhanh\s+trinh\s+khep\s+lai\b|\bkhep\s+lai\s+nhung\s+man\s+tranh\s+tai\b"
    r"|\bda\s+dien\s+ra\s+thanh\s+cong\b|\bcac\s+dien\s+gia\s+da\s+mang\s+den\b"
    r"|\bda\s+chung\s+kien\s+nhung\s+ban\s+hoa\s+ca\s+cuoi\s+cung\b|\bthe\s+lasting\s+melody\b"
    r"|\b(?:vong|chuong\s+trinh|su\s+kien|ngay\s+hoi|buoi\s+le|cuoc\s+thi)\b[^\n]{0,110}?\bda\s+dien\s+ra\s+tai\b"
    r"|\bvua\s+qua\s*,\s*(?:dong\s+dao|hon\s+\d+|cac\s+ban|doan\s+vien|sinh\s+vien)\b"
    r"|\bcam\s+on\s+su\s+gop\s+mat\b|\bgui\s+loi\s+cam\s+on\s+chan\s+thanh\b"
    r"|\b(?:cung\s+)?nhin\s+lai\s+nhung\s+(?:khoanh\s+khac|hinh\s+anh)\b"
    r"|\bchuc\s+mung\s+(?:cac\s+)?(?:thi\s+sinh|doi\s+thi|ca\s+nhan|tap\s+the|thanh\s+cong)\b"
    r"|\bvinh\s+danh\b|\bket\s+qua\s+(?:chung\s+cuoc|tuyen\s+(?:dung|nhan\s+su|cong\s+tac\s+vien))\b"
    r"|\btin\s+tot\s+chuyen\s+dep\b"
    r")",
    re.IGNORECASE,
)

ROUTINE_ADMIN_NOTICE_RE = re.compile(
    r"(?:"
    r"\blich\s+thi\s+(?:chi\s+tiet\s+)?(?:cuoi\s+ky|giua\s+ky)\b"
    r"|\bphong\s+chong\s+gian\s+lan\b|\bsua\s+chua\s+phong\s+may\b|\bse\s+duoc\s+sua\s+chua\b"
    r"|\bdang\s+ky\s+o\s+ky\s+tuc\s+xa\b|\ble\s+trao\s+bang\s+tot\s+nghiep\b"
    r"|\bthong\s+bao\s+dong\s+hoc\s+phi\b|\bdang\s+ky\s+hoc\s+phan\b"
    r"|\bhoi\s+nghi\s+(?:mo\s+rong\s+)?(?:ban\s+chap\s+hanh|doan\s+khoa|hoi\s+sinh\s+vien)\b|\bhoi\s+nghi\s+dai\s+bieu\b"
    r")",
    re.IGNORECASE,
)

SCHOLARSHIP_DOSSIER_RE = re.compile(
    r"(?:\bhoc\s+bong\b|\bho\s+so\s+hoc\s+bong\b|\bxet\s+cap\s+hoc\s+bong\b)",
    re.IGNORECASE,
)

LIVE_SESSION_CUE_RE = re.compile(
    r"(?:\bhoi\s+thao\b|\bseminar\b|\bwebinar\b|\btalkshow\b|\btoa\s+dam\b|\bbuoi\s+gioi\s+thieu\b|\blivestream\b|\bworkshop\b)",
    re.IGNORECASE,
)

CORPORATE_JOB_AD_RE = re.compile(
    r"(?:"
    r"\bmanagement\s+trainee\b|\bengineering\s+development\s+program\b"
    r"|\btuyen\s+dung\s+(?:nhan\s+vien|ky\s+su|fresher|intern)\b"
    r"|\blam\s+viec\s+toan\s+thoi\s+gian\b"
    r")",
    re.IGNORECASE,
)

COMPANION_FAQ_ALBUM_RE = re.compile(
    r"(?:\bgiai\s+dap\s+tron\s+ven\s+trong\s+tung\s+buc\s+anh\b|\bbo\s+cau\s+hoi\s+thuong\s+gap\b)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Step 2: Positive qualification signals
# ---------------------------------------------------------------------------
STRONG_EVENT_SIGNAL_RE = re.compile(
    r"(?:"
    r"\bworkshop\b|\btalkshow\b|\btalk\s+show\b|\bseminar\b|\bwebinar\b"
    r"|\btoa\s+dam\b|\bhoi\s+thao\b|\bcuoc\s+thi\b|\bhoi\s+thi\b|\bhackathon\b"
    r"|\bngay\s+hoi\b|\bfestival\b|\bhien\s+mau\b|\bchien\s+dich\s+tinh\s+nguyen\b"
    r"|\bhoat\s+dong\s+tinh\s+nguyen\b|\bchu\s+nhat\s+xanh\b|\bxuan\s+tinh\s+nguyen\b"
    r"|\bmua\s+he\s+xanh\b|\btuyen\s+cong\s+tac\s+vien\b|\btuyen\s+ctv\b"
    r"|\bmo\s+don\s+dang\s+ky\b|\bdang\s+ky\s+tham\s+gia\b|\blink\s+dang\s+ky\b"
    r"|\bthoi\s+gian\s+to\s+chuc\b|\bdoi\s+tuong\s+tham\s+gia\b|\bmini[\s\-]?course\b"
    r")",
    re.IGNORECASE,
)

EVENT_KEYWORDS_ASCII: Tuple[str, ...] = (
    "dang ky",
    "tham gia",
    "tham du",
    "workshop",
    "talkshow",
    "seminar",
    "webinar",
    "toa dam",
    "hoi thao",
    "cuoc thi",
    "hoi thi",
    "ngay hoi",
    "tinh nguyen",
    "thien nguyen",
    "hien mau",
    "quyen gop",
    "gay quy",
    "giai dau",
    "hoi thao",
    "thoi gian",
    "dia diem",
    "han dang ky",
    "han chot",
    "doi tuong",
    "le phi",
    "mien phi",
    "diem ren luyen",
    "drl",
    "giay chung nhan",
    "forms.gle",
    "docs.google.com/forms",
    "bit.ly",
    "tinyurl",
    "linktr.ee",
)

SCHEDULE_OR_URL_CUE_RE = re.compile(
    r"(?:"
    r"\bthoi\s+gian\s*:|\bdia\s+diem\s*:|\bngay\s+\d{1,2}[/\-\.]\d{1,2}\b"
    r"|\bluc\s+\d{1,2}\s*(?:h|g|:)\s*\d{0,2}\b"
    r"|forms\.gle/|docs\.google\.com/forms|bit\.ly/|tinyurl\.|linktr\.ee/"
    r")",
    re.IGNORECASE,
)


def count_event_keywords(ascii_text: str) -> List[str]:
    """Return distinct matched event keywords found in accent-stripped lowercase text."""
    matched: List[str] = []
    for kw in EVENT_KEYWORDS_ASCII:
        if kw in ascii_text:
            matched.append(kw)
    return matched


def reason_about_post(content: str, subject: str = "") -> PreFilterDecision:
    """Evaluate raw post/email text before calling an LLM.

    Prioritizes recall for events, schedule/venue updates, and cancellations while
    filtering out clear non-events at zero API cost.
    """
    combined = f"{subject}\n{content}".strip()
    if len(combined) < 20:
        return PreFilterDecision(
            is_candidate=False,
            reason="disqualified:too_short",
        )

    ascii_text = normalize_vi_ascii(combined)

    # 0. Lifecycle announcements (updates / extensions / cancellations) always qualify
    m_lifecycle = LIFECYCLE_SIGNAL_RE.search(ascii_text)
    if m_lifecycle:
        return PreFilterDecision(
            is_candidate=True,
            reason=f"qualified:lifecycle_signal ({m_lifecycle.group(0)})",
            matched_signals=[m_lifecycle.group(0)],
            is_lifecycle_signal=True,
        )

    # Strip copy-pasted event info footers (`_____ ... THÔNG TIN CUỘC THI ...`) before checking active CTA
    raw_without_footer = FOOTER_SPLIT_RE.sub("", combined.lower().replace("đ", "d"))
    ascii_without_footer = normalize_vi_ascii(raw_without_footer)
    has_reg_url = bool(GENUINE_REG_URL_RE.search(combined))
    has_active_cta = bool(ACTIVE_REGISTRATION_CTA_RE.search(ascii_without_footer)) and (
        has_reg_url or "mo don" in ascii_without_footer or "mo cong" in ascii_without_footer
    )

    # 1. Hard disqualifications (never overridden by copy-pasted footers or verification sheets)
    m_hard = HARD_NON_EVENT_RE.search(ascii_text)
    if m_hard:
        return PreFilterDecision(
            is_candidate=False,
            reason=f"disqualified:hard_non_event ({m_hard.group(0)})",
        )

    m_admin = ROUTINE_ADMIN_NOTICE_RE.search(ascii_text)
    if m_admin:
        return PreFilterDecision(
            is_candidate=False,
            reason=f"disqualified:routine_admin_notice ({m_admin.group(0)})",
        )

    m_bts = BEHIND_SCENES_OR_COMPANION_RE.search(ascii_text)
    if m_bts and not has_reg_url:
        return PreFilterDecision(
            is_candidate=False,
            reason=f"disqualified:behind_the_scenes_or_companion ({m_bts.group(0)})",
        )

    m_faq = COMPANION_FAQ_ALBUM_RE.search(ascii_text)
    if m_faq and not has_active_cta:
        return PreFilterDecision(
            is_candidate=False,
            reason=f"disqualified:companion_faq_post ({m_faq.group(0)})",
        )

    m_recap = RECAP_OR_CONGRATS_RE.search(ascii_text)
    if m_recap and not has_active_cta:
        return PreFilterDecision(
            is_candidate=False,
            reason=f"disqualified:recap_or_congrats ({m_recap.group(0)})",
        )

    if SCHOLARSHIP_DOSSIER_RE.search(ascii_text) and not LIVE_SESSION_CUE_RE.search(ascii_text):
        return PreFilterDecision(
            is_candidate=False,
            reason="disqualified:scholarship_dossier_without_live_session",
        )

    if CORPORATE_JOB_AD_RE.search(ascii_text) and not LIVE_SESSION_CUE_RE.search(ascii_text):
        return PreFilterDecision(
            is_candidate=False,
            reason="disqualified:corporate_job_posting",
        )

    # 2. Positive qualification tiers
    matched_kws = count_event_keywords(ascii_text)
    m_strong = STRONG_EVENT_SIGNAL_RE.search(ascii_text)
    if m_strong:
        return PreFilterDecision(
            is_candidate=True,
            reason=f"qualified:strong_event_signal ({m_strong.group(0)})",
            matched_signals=matched_kws or [m_strong.group(0)],
        )

    if len(matched_kws) >= 3:
        return PreFilterDecision(
            is_candidate=True,
            reason=f"qualified:keyword_density (count={len(matched_kws)})",
            matched_signals=matched_kws,
        )

    if len(matched_kws) >= 2 and SCHEDULE_OR_URL_CUE_RE.search(ascii_text):
        return PreFilterDecision(
            is_candidate=True,
            reason=f"qualified:moderate_keywords_with_schedule_or_link (count={len(matched_kws)})",
            matched_signals=matched_kws,
        )

    return PreFilterDecision(
        is_candidate=False,
        reason=f"disqualified:insufficient_event_signals (count={len(matched_kws)})",
        matched_signals=matched_kws,
    )
