"""[CRD] Credit Calculation Module (`utility/credit_calculator.py`).

Implements the CRD responsibilities defined in `SYSTEM_ARCHITECTURE.md`:
1. `credit_calc(event_or_id, role='participant') -> int`:
   Deterministic event credit calculation from event metadata (`title`, `description`,
   `keywords`, `topic`, `format`, `award`, `sources`, start/end timestamps, `category`)
   and the user's participation `role` (`'participant'`, `'host'`, `'collaborator'`),
   following HCMUS Regulation 1028/QĐ-KHTN-CTSV.
2. `gain_calc(...) -> float`:
   Computes the Remaining-Rubric-Aware normalized `gain` in `[0.0, 1.0]` for `RCM`:
   - If `not credit.is_credit_eligible` or `credit.effective_points <= 0` -> `0.0`
   - Otherwise:
       P_use = min(P_eff, 0.7 * P_min + 0.3 * P_exp)  if source == "rule_fallback"
             = P_eff                                   otherwise
       Gain  = clip_[0, 1]( min(1.0, P_use / 10.0) * (0.5 + 0.5 * target_gap_c / M_c) * r_src )
     with r_src in {1.00 ("explicit"), 0.90 ("rule"), 0.80 ("rule_fallback")}.
3. Database integration helpers (`drl_database.db` via `DRLDatabaseManager`):
   - `update_missing_min_credits()`: CRD(S) batch populates `events.min_credit` where `NULL`.
   - `finalize_completed_subscriptions()`: CRD(U) finalizes `event_subscription.credit` where
     `status = 'done' AND credit IS NULL` and increments `users.credits[category - 1]` capped at
     `rubric.max_credit[category - 1]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import unicodedata

from database import DRLDatabaseManager, decode_json_list, encode_json_list, get_drl_db

# Default HCMUS 5-category caps (Muc 1..5 in Quy che 1028/QD-KHTN-CTSV): 30, 20, 25, 25, 30
DEFAULT_MAX_CREDIT: Tuple[int, int, int, int, int] = (30, 20, 25, 25, 30)
TOTAL_MAX_SCORE: int = 100

# Gain formula constants
GAIN_P_REF: float = 10.0
GAIN_GAP_FLOOR: float = 0.5
GAIN_FALLBACK_MIN_WEIGHT: float = 0.7

SOURCE_RELIABILITY: Dict[str, float] = {
    "explicit": 1.00,
    "rule": 0.90,
    "rule_fallback": 0.80,
}

ROLE_ALIASES: Dict[str, str] = {
    "participant": "participant",
    "attendee": "participant",
    "member": "participant",
    "volunteer": "participant",
    "tnv": "participant",
    "tinh nguyen vien": "participant",
    "tham gia": "participant",
    "du thi": "participant",
    "thi sinh": "participant",
    "huy dong luc luong": "participant",
    "gay quy": "participant",
    "host": "host",
    "organizer": "host",
    "btc": "host",
    "ban to chuc": "host",
    "thanh vien ban to chuc": "host",
    "collaborator": "collaborator",
    "ctv": "collaborator",
    "cong tac vien": "collaborator",
}


def normalize_vi(text: str) -> str:
    """Lowercase, strip Vietnamese diacritics (`đ -> d`), and collapse non-alphanumerics."""
    raw = (text or "").lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", raw)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    cleaned = re.sub(r"[^a-z0-9]+", " ", stripped)
    return re.sub(r"\s+", " ", cleaned).strip()


def _pad(norm_text: str) -> str:
    return f" {norm_text} " if norm_text else " "


def _has_phrase(padded_norm_text: str, phrase: str) -> bool:
    p = normalize_vi(phrase)
    return bool(p) and f" {p} " in padded_norm_text


def normalize_role(role: Optional[str]) -> str:
    """Normalize role input into canonical DB role (`'participant'`, `'host'`, `'collaborator'`)."""
    if not role:
        return "participant"
    norm = normalize_vi(str(role))
    if norm in ROLE_ALIASES:
        return ROLE_ALIASES[norm]
    if "ban to chuc" in norm or norm == "btc" or "host" in norm or "organizer" in norm:
        return "host"
    if "cong tac vien" in norm or norm == "ctv" or "collaborat" in norm:
        return "collaborator"
    return "participant"


def _coerce_str_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, tuple):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        if s.startswith("["):
            decoded = decode_json_list(s, default=[])
            return [str(x).strip() for x in decoded if str(x).strip()]
        return [s]
    return [str(val).strip()]


def _coerce_5_floats(values: Optional[Sequence[Union[int, float]]], default: Sequence[Union[int, float]]) -> List[float]:
    if values is None:
        return [float(x) for x in default]
    if isinstance(values, str):
        values = decode_json_list(values, default=list(default))
    items = list(values)
    if len(items) < 5:
        items.extend(default[len(items) : 5])
    return [float(items[i]) for i in range(5)]


@dataclass(frozen=True)
class CreditEstimate:
    """Structured credit evaluation result for an event and user role."""

    event_id: Optional[int]
    category: Optional[int]  # 1..5 (None if mandatory 6.4 or unmapped)
    rubric_item_id: Optional[str]  # e.g. "1.6", "1.7", "3.1", "3.14", "4.3", "4.4", "6.4"
    role: str  # 'participant' | 'host' | 'collaborator'
    is_credit_eligible: bool
    expected_points: float  # P_exp
    min_points: float  # P_min
    max_points: float  # P_max
    effective_points: float  # P_eff (remaining-rubric-aware; equals expected_points if no user ledger)
    source: str  # 'explicit' | 'rule' | 'rule_fallback'
    capped_by: Optional[str] = None
    inferred_level: Optional[str] = None
    inferred_duration: Optional[str] = None
    warnings: Tuple[str, ...] = ()

    @property
    def credit(self) -> int:
        """Integer credit granted for this event/role (conservative when `rule_fallback` for participant)."""
        if not self.is_credit_eligible or self.expected_points <= 0:
            return 0
        if self.source == "rule_fallback" and self.role == "participant":
            return max(0, int(round(self.min_points)))
        return max(0, int(round(self.expected_points)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "category": self.category,
            "rubric_item_id": self.rubric_item_id,
            "role": self.role,
            "is_credit_eligible": self.is_credit_eligible,
            "expected_points": self.expected_points,
            "min_points": self.min_points,
            "max_points": self.max_points,
            "effective_points": self.effective_points,
            "credit": self.credit,
            "source": self.source,
            "capped_by": self.capped_by,
            "inferred_level": self.inferred_level,
            "inferred_duration": self.inferred_duration,
            "warnings": list(self.warnings),
        }


# =====================================================================
# RUBRIC SPECIFICATION (Quy chế 1028/QĐ-KHTN-CTSV — ĐH KHTN, ĐHQG-HCM)
# =====================================================================
# Roles in `points`: 'participant', 'host' (Ban tổ chức), 'collaborator' (Cộng tác viên)
RUBRIC_ITEMS: Dict[str, Dict[str, Any]] = {
    # Category 1: Học tập & NCKH (max 30)
    "1.5": {
        "category": 1,
        "kind": "achievement",
        "variant_by": "none",
        "points": {"participant": {"any": 15}},
        "role_redirect": {"host": "1.7", "collaborator": "1.7"},
    },
    "1.6": {
        "category": 1,
        "kind": "event",
        "variant_by": "none",
        "points": {"participant": {"any": 5}},
        "role_redirect": {"host": "1.7", "collaborator": "1.7"},
    },
    "1.7": {
        "category": 1,
        "kind": "role_variant",
        "variant_by": "none",
        "points": {
            "host": {"any": 10},
            "collaborator": {"any": 5},
            "participant": {"any": 5},
        },
    },
    "1.8": {
        "category": 1,
        "kind": "membership",
        "variant_by": "none",
        "points": {
            "participant": {"any": 5},
            "host": {"any": 5},
            "collaborator": {"any": 5},
        },
    },
    "1.9": {
        "category": 1,
        "kind": "event",
        "variant_by": "none",
        "points": {
            "participant": {"any": 5},
            "host": {"any": 5},
            "collaborator": {"any": 5},
        },
    },
    # Category 2: Chấp hành nội quy, quy chế (max 20)
    "2.2": {
        "category": 2,
        "kind": "task",
        "variant_by": "none",
        "points": {"participant": {"any": 5}},
    },
    "2.3": {
        "category": 2,
        "kind": "event",
        "variant_by": "none",
        "points": {
            "participant": {"any": 5},
            "host": {"any": 5},
            "collaborator": {"any": 5},
        },
    },
    # Category 3: Chính trị - XH, VH-VN, TDTT (max 25)
    "3.1": {
        "category": 3,
        "kind": "event",
        "variant_by": "level",
        "points": {
            "participant": {"truong_tro_len": 10, "khoa": 5, "lop": 5},
        },
        "default_expected": {"participant": 5},
        "role_redirect": {"host": "3.14", "collaborator": "3.14"},
    },
    "3.2": {
        "category": 3,
        "kind": "event",
        "variant_by": "none",
        "points": {"participant": {"any": 5}},
        "role_redirect": {"host": "3.14", "collaborator": "3.14"},
    },
    "3.3": {
        "category": 3,
        "kind": "event",
        "variant_by": "none",
        "points": {"participant": {"any": 5}},
        "role_redirect": {"host": "3.14", "collaborator": "3.14"},
    },
    "3.4": {
        "category": 3,
        "kind": "membership",
        "variant_by": "none",
        "points": {"participant": {"any": 5}},
        "role_redirect": {"host": "3.14", "collaborator": "3.14"},
    },
    "3.5": {
        "category": 3,
        "kind": "event",
        "variant_by": "none",
        "points": {"participant": {"any": 5}},
        "role_redirect": {"host": "3.14", "collaborator": "3.14"},
    },
    "3.6": {
        "category": 3,
        "kind": "achievement",
        "variant_by": "none",
        "points": {"participant": {"any": 15}},
        "role_redirect": {"host": "3.14", "collaborator": "3.14"},
    },
    "3.8": {
        "category": 3,
        "kind": "event",
        "variant_by": "none",
        "points": {"participant": {"any": 5}},
        "role_redirect": {"host": "3.14", "collaborator": "3.14"},
    },
    "3.9": {
        "category": 3,
        "kind": "event",
        "variant_by": "none",
        "points": {"participant": {"any": 5}},
        "role_redirect": {"host": "3.14", "collaborator": "3.14"},
    },
    "3.10": {
        "category": 3,
        "kind": "event",
        "variant_by": "none",
        "points": {"participant": {"any": 5}},
        "role_redirect": {"host": "3.14", "collaborator": "3.14"},
    },
    "3.11": {
        "category": 3,
        "kind": "achievement",
        "variant_by": "none",
        "points": {"participant": {"any": 15}},
        "role_redirect": {"host": "3.14", "collaborator": "3.14"},
    },
    "3.12": {
        "category": 3,
        "kind": "event",
        "variant_by": "none",
        "points": {"participant": {"any": 15}},
        "role_redirect": {"host": "3.14", "collaborator": "3.14"},
    },
    "3.13": {
        "category": 3,
        "kind": "event",
        "variant_by": "none",
        "points": {"participant": {"any": 10}},
        "role_redirect": {"host": "3.14", "collaborator": "3.14"},
    },
    "3.14": {
        "category": 3,
        "kind": "role_variant",
        "variant_by": "level",
        "points": {
            "host": {"truong_tro_len": 10, "khoa": 5, "lop": 5},
            "collaborator": {"truong_tro_len": 5, "khoa": 3, "lop": 0},
        },
        "fallback_min": {"host": 5, "collaborator": 3},
        "default_expected": {"host": 5, "collaborator": 3},
    },
    # Category 4: Xây dựng và phát triển cộng đồng (max 25)
    "4.2": {
        "category": 4,
        "kind": "event",
        "variant_by": "none",
        "points": {"participant": {"any": 10}},
        "role_redirect": {"host": "4.4", "collaborator": "4.4"},
    },
    "4.3": {
        "category": 4,
        "kind": "event",
        "variant_by": "duration",
        "points": {
            "participant": {"lt_24h": 5, "24h_to_lt_1w": 10, "ge_1w": 15},
        },
        "fallback_min": {"participant": 5},
        "default_expected": {"participant": 5},
        "role_redirect": {"host": "4.4", "collaborator": "4.4"},
    },
    "4.4": {
        "category": 4,
        "kind": "role_variant",
        "variant_by": "level",
        "points": {
            "host": {"truong_tro_len": 10, "khoa": 5, "lop": 5},
            "collaborator": {"truong_tro_len": 5, "khoa": 3, "lop": 0},
        },
        "fallback_min": {"host": 5, "collaborator": 3},
        "default_expected": {"host": 5, "collaborator": 3},
    },
    "4.5": {
        "category": 4,
        "kind": "event",
        "variant_by": "none",
        "points": {"participant": {"any": 3}},
        "role_redirect": {"host": "4.4", "collaborator": "4.4"},
    },
    "4.6": {
        "category": 4,
        "kind": "event",
        "variant_by": "none",
        "points": {"participant": {"any": 10}},
        "role_redirect": {"host": "4.4", "collaborator": "4.4"},
    },
    # Category 5: Nội dung khuyến khích (max 30)
    "5.1": {
        "category": 5,
        "kind": "achievement",
        "variant_by": "award_tier",
        "points": {
            "participant": {"quoc_te": 20, "quoc_gia": 15, "truong": 10, "doan_the": 5},
        },
        "fallback_min": {"participant": 5},
        "default_expected": {"participant": 10},
    },
    "5.2": {
        "category": 5,
        "kind": "achievement",
        "variant_by": "award_tier",
        "points": {
            "participant": {"first": 5, "other": 3},
        },
        "fallback_min": {"participant": 3},
        "default_expected": {"participant": 5},
    },
    "5.3": {
        "category": 5,
        "kind": "achievement",
        "variant_by": "none",
        "points": {"participant": {"any": 15}},
    },
    "5.4": {
        "category": 5,
        "kind": "achievement",
        "variant_by": "none",
        "points": {"participant": {"any": 25}},
    },
    "5.5": {
        "category": 5,
        "kind": "achievement",
        "variant_by": "none",
        "points": {"participant": {"any": 15}},
    },
    "5.7": {
        "category": 5,
        "kind": "achievement",
        "variant_by": "none",
        "points": {"participant": {"any": 10}},
    },
    # Penalty / Mandatory event (0 bonus points; attend to avoid -20/-10 penalty)
    "6.4": {
        "category": None,
        "kind": "mandatory_event",
        "variant_by": "none",
        "points": {"participant": {"any": 0}},
    },
}

DEFAULT_CATEGORY_SUBITEM: Dict[int, str] = {
    1: "1.9",
    2: "2.3",
    3: "3.5",
    4: "4.3",
    5: "5.2",
}


# =====================================================================
# DETERMINISTIC ATTRIBUTE & ITEM INFERENCE HELPERS
# =====================================================================

_NEGATION_RE = re.compile(
    r"\bkhong\s+(?:duoc\s+)?(?:cong|tinh|co)\s+diem\s+(?:ren\s+luyen|rl|drl)\b"
    r"|\bkhong\s+(?:duoc\s+)?(?:cong|tinh)\s+drl\b"
)

_ROLE_POINT_PATTERNS: Dict[str, Tuple[re.Pattern[str], ...]] = {
    "host": (
        re.compile(r"\b(?:ban\s+to\s+chuc|btc)\s*[:\-–]?\s*\+?\s*(\d{1,2})\s*(?:diem\s*(?:ren\s+luyen|rl|drl)|drl|d\b)"),
    ),
    "collaborator": (
        re.compile(r"\b(?:cong\s+tac\s+vien|ctv)\s*[:\-–]?\s*\+?\s*(\d{1,2})\s*(?:diem\s*(?:ren\s+luyen|rl|drl)|drl|d\b)"),
    ),
    "participant": (
        re.compile(r"\b(?:tham\s+gia|nguoi\s+tham\s+du|thi\s+sinh|tinh\s+nguyen\s+vien|tnv)\s*[:\-–]?\s*\+?\s*(\d{1,2})\s*(?:diem\s*(?:ren\s+luyen|rl|drl)|drl|d\b)"),
    ),
}

_GENERAL_EXPLICIT_PTS_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:cong|nhan\s+duoc|nhan|duoc\s+cong|huong)\s*\+?\s*(\d{1,2})\s*(?:diem\s+(?:ren\s+luyen|rl|drl)|drl)\b"),
    re.compile(r"\+\s*(\d{1,2})\s*(?:diem\s+(?:ren\s+luyen|rl|drl)|drl)\b"),
    re.compile(r"\b(\d{1,2})\s*(?:diem\s+ren\s+luyen|diem\s+rl|drl)\b"),
)

_LEVEL_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "truong_tro_len",
        (
            "quoc te",
            "international",
            "asean",
            "toan quoc",
            "cap quoc gia",
            "trung uong",
            "cap thanh pho",
            "cap thanh",
            "thanh doan",
            "hoi sinh vien thanh pho",
            "nha van hoa sinh vien",
            "nvhsv",
            "cap dhqg",
            "cap dai hoc quoc gia",
            "doan dhqg",
            "hoi sinh vien dhqg",
            "lien truong",
            "cap truong",
            "toan truong",
            "doan truong",
            "hoi sinh vien truong",
            "toan the sinh vien truong",
            "phong cong tac sinh vien",
            "phong ctsv",
            "trung tam ho tro sinh vien",
            "sacus",
        ),
    ),
    (
        "khoa",
        (
            "cap khoa",
            "toan khoa",
            "lien chi doan",
            "lien chi hoi",
            "lch khoa",
            "doan khoa",
            "khoa to chuc",
            "sinh vien khoa",
            "khoa cong nghe thong tin",
            "khoa toan tin",
            "khoa vat ly",
            "khoa hoa hoc",
            "khoa sinh hoc",
            "khoa moi truong",
            "khoa dia chat",
            "khoa dien tu vien thong",
            "clb sinh vien 5 tot",
            "cau lac bo",
        ),
    ),
    (
        "lop",
        (
            "cap lop",
            "chi doan",
            "chi hoi",
            "ban can su lop",
            "sinh hoat lop",
            "tap the lop",
        ),
    ),
)

_DURATION_TEXT_RE = re.compile(r"\b(\d{1,2})\s*(tuan|thang|ngay|gio|tieng)\b")


def extract_explicit_points(
    event: Mapping[str, Any],
    role: str = "participant",
) -> Tuple[bool, Optional[int]]:
    """Extract `(is_negated, explicit_points)` from `award`, `title`, and `description`.

    Returns:
      - `(True, None)` if the announcement explicitly states no DRL points are granted.
      - `(False, pts)` if explicit DRL points (`1..30`) applicable to `role` are found.
      - `(False, None)` otherwise.
    """
    canon_role = normalize_role(role)
    award_norm = normalize_vi(str(event.get("award") or ""))
    body_norm = normalize_vi(f"{event.get('title') or ''} {event.get('description') or ''}")
    combined = f"{award_norm} {body_norm}".strip()

    if not combined:
        return False, None

    if _NEGATION_RE.search(combined):
        return True, None

    # 1. Check role-specific explicit point announcements first (works for participant/host/collaborator)
    for pat in _ROLE_POINT_PATTERNS.get(canon_role, ()):
        m = pat.search(combined)
        if m:
            pts = int(m.group(1))
            if 1 <= pts <= 30:
                return False, pts

    # 2. General announced DRL points apply directly to 'participant'
    #    (host/collaborator credits follow BTC/CTV rubric schedules unless explicitly specified above)
    if canon_role == "participant":
        for target_text in (award_norm, body_norm):
            if not target_text:
                continue
            for pat in _GENERAL_EXPLICIT_PTS_PATTERNS:
                m = pat.search(target_text)
                if m:
                    pts = int(m.group(1))
                    if 1 <= pts <= 30:
                        return False, pts

    return False, None


def extract_event_level(event: Mapping[str, Any]) -> Optional[str]:
    """Infer organization scope bucket (`'truong_tro_len'`, `'khoa'`, `'lop'`, or `None`)."""
    title = str(event.get("title") or "")
    desc = str(event.get("description") or "")
    sources = str(event.get("sources") or "")
    organizer = str(event.get("organizer") or "")
    fanpage = str(event.get("fanpage_name") or "")

    # Check organizer/sources first for strong structural unit cues (e.g., Chi hội vs Đoàn khoa vs Đoàn trường)
    org_padded = _pad(normalize_vi(f"{organizer} {fanpage} {sources}"))
    full_padded = _pad(normalize_vi(f"{title} {desc} {organizer} {fanpage} {sources}"))

    # Explicit scope keywords in full text ("cấp trường", "cấp khoa", "cấp lớp", "toàn quốc", "quốc tế"...)
    explicit_truong = (
        "quoc te",
        "cap quoc gia",
        "toan quoc",
        "trung uong",
        "cap thanh pho",
        "cap thanh",
        "thanh doan",
        "nha van hoa sinh vien",
        "cap dhqg",
        "lien truong",
        "cap truong",
        "toan truong",
    )
    if any(_has_phrase(full_padded, p) for p in explicit_truong):
        return "truong_tro_len"
    if any(_has_phrase(full_padded, p) for p in ("cap khoa", "toan khoa")):
        return "khoa"
    if any(_has_phrase(full_padded, p) for p in ("cap lop", "sinh hoat lop", "tap the lop")):
        return "lop"

    # Unit-based inference: check higher-level units (school -> faculty) before class units
    # so "Liên chi Hội" / "Đoàn Khoa" is never shadowed by the substring "chi hội".
    for bucket, phrases in _LEVEL_PATTERNS:
        if any(_has_phrase(org_padded, p) for p in phrases):
            return bucket

    for bucket, phrases in _LEVEL_PATTERNS:
        if any(_has_phrase(full_padded, p) for p in phrases):
            return bucket

    return None


def _parse_iso_dt(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def extract_duration_bucket(event: Mapping[str, Any]) -> Optional[str]:
    """Infer canonical volunteering duration bucket (`'lt_24h'`, `'24h_to_lt_1w'`, `'ge_1w'`)."""
    text_norm = normalize_vi(f"{event.get('title') or ''} {event.get('description') or ''}")
    padded = _pad(text_norm)

    # Known long-term campaign priors
    if any(_has_phrase(padded, p) for p in ("mua he xanh", "xuan tinh nguyen", "tiep suc mua thi")):
        return "ge_1w"
    if any(_has_phrase(padded, p) for p in ("chu nhat xanh", "ngay chu nhat xanh")):
        return "lt_24h"

    # Explicit duration expressions in text (e.g. "2 ngày 1 đêm", "4 giờ", "2 tuần")
    for m in _DURATION_TEXT_RE.finditer(text_norm):
        num = int(m.group(1))
        unit = m.group(2)
        if unit in ("tuan", "thang"):
            return "ge_1w"
        if unit == "ngay":
            if num >= 7:
                return "ge_1w"
            if num >= 2:
                return "24h_to_lt_1w"
            return "lt_24h"
        if unit in ("gio", "tieng"):
            if num >= 24 * 7:
                return "ge_1w"
            if num >= 24:
                return "24h_to_lt_1w"
            return "lt_24h"

    # Fallback to event start/end timestamps
    start_dt = _parse_iso_dt(event.get("event_start_datetime") or event.get("start_datetime"))
    end_dt = _parse_iso_dt(event.get("event_end_datetime") or event.get("end_datetime"))
    if start_dt is not None and end_dt is not None and end_dt > start_dt:
        hours = (end_dt - start_dt).total_seconds() / 3600.0
        if hours >= 24.0 * 7:
            return "ge_1w"
        if hours >= 24.0:
            return "24h_to_lt_1w"
        return "lt_24h"

    return None


def infer_rubric_item(event: Mapping[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    """Infer `(rubric_item_id, category)` from CLF output or event record fields.

    Supports three synchronization contracts with Classification (`CLF`):
    1. Fine-grained sub-item ref (`rubric_item_id`, `sub_code`, `predicted_criterion_ref`,
       or `event["classification"]["predicted_criterion_ref"]` from Module 3 kNN classifier).
    2. Database category (`events.category` in `1..5` populated by `CLF(S)` in `drl_database.db`),
       narrowing resolution to sub-items within that category.
    3. Standalone fallback inference from `title`, `description`, `topic`, `format`, `keywords`, `award`
       when `CLF` has not classified the event yet.
    """
    cls_obj = event.get("classification")
    if isinstance(cls_obj, Mapping):
        cls_status = str(cls_obj.get("status") or "").upper()
        if cls_status in ("NO_MATCH", "SKIPPED_NOT_EVENT", "REJECTED"):
            return None, None
        cls_ref = cls_obj.get("predicted_criterion_ref") or cls_obj.get("rubric_item_id")
        if isinstance(cls_ref, str) and cls_ref.strip() in RUBRIC_ITEMS:
            sub_id = cls_ref.strip()
            return sub_id, RUBRIC_ITEMS[sub_id]["category"]

    explicit_sub = (
        event.get("rubric_item_id")
        or event.get("sub_code")
        or event.get("predicted_criterion_ref")
    )
    if isinstance(explicit_sub, str) and explicit_sub.strip() in RUBRIC_ITEMS:
        sub_id = explicit_sub.strip()
        return sub_id, RUBRIC_ITEMS[sub_id]["category"]

    title_norm = normalize_vi(str(event.get("title") or ""))
    desc_norm = normalize_vi(str(event.get("description") or ""))
    award_norm = normalize_vi(str(event.get("award") or ""))

    topics = _coerce_str_list(event.get("topic"))
    if not topics and isinstance(event.get("category"), (list, tuple)):
        topics = _coerce_str_list(event.get("category"))
    formats = _coerce_str_list(event.get("format"))
    keywords = _coerce_str_list(event.get("keywords"))

    # Strip university-name boilerplate ("Đại học Khoa học Tự nhiên") before topical matching
    # so the substring "khoa hoc" does not falsely trigger Category 1 on cultural/ceremonial events.
    title_clean = re.sub(r"\bkhoa hoc tu nhien\b", "hcmus", title_norm)
    desc_clean = re.sub(r"\bkhoa hoc tu nhien\b", "hcmus", desc_norm)

    tags_norm = normalize_vi(" ".join(topics + formats + keywords))
    full_padded = _pad(f"{title_clean} {desc_clean} {tags_norm} {award_norm}".strip())
    title_padded = _pad(title_clean)
    tags_padded = _pad(tags_norm)

    # 0. Mandatory penalty-avoidance event (6.4: Tuần sinh hoạt công dân)
    if any(
        _has_phrase(full_padded, p)
        for p in ("tuan sinh hoat cong dan", "sinh hoat cong dan dau khoa", "sinh hoat cong dan", "shcd")
    ):
        return "6.4", None

    # Check if CLF already assigned category in 1..5
    raw_cat = event.get("category")
    assigned_cat: Optional[int] = raw_cat if isinstance(raw_cat, int) and 1 <= raw_cat <= 5 else None

    is_contest = any(
        _has_phrase(full_padded, p)
        for p in (
            "cuoc thi",
            "hoi thi",
            "olympic",
            "hackathon",
            "datathon",
            "icpc",
            "ctf",
            "giai dau",
            "tournament",
            "contest",
            "competition",
            "tranh tai",
        )
    ) or any(_has_phrase(tags_padded, p) for p in ("cuoc thi", "thi", "competition", "contest"))

    # If CLF assigned a category (1..5), pick the most specific sub-item inside that category
    if assigned_cat == 1:
        if any(
            _has_phrase(full_padded, p)
            for p in ("eureka", "de tai nghien cuu khoa hoc", "de tai nckh", "sinh vien nghien cuu khoa hoc", "khoa luan tot nghiep")
        ):
            return "1.5", 1
        if is_contest:
            return "1.6", 1
        if any(_has_phrase(full_padded, p) for p in ("sinh hoat thuong xuyen", "cau lac bo hoc thuat", "clb hoc thuat")):
            return "1.8", 1
        return "1.9", 1

    if assigned_cat == 2:
        if any(_has_phrase(full_padded, p) for p in ("danh gia mon hoc", "danh gia giang day", "danh gia khoa hoc")):
            return "2.2", 2
        return "2.3", 2

    if assigned_cat == 3:
        if any(_has_phrase(full_padded, p) for p in ("doi tuyen cap truong", "tuyen chon doi tuyen")):
            return "3.6", 3
        if any(_has_phrase(full_padded, p) for p in ("dai bieu chinh thuc", "tuyen chon dai bieu")):
            return "3.12", 3
        if any(_has_phrase(full_padded, p) for p in ("giao luu quoc te", "trao doi van hoa quoc te")):
            return "3.13", 3
        if any(_has_phrase(full_padded, p) for p in ("san pham truyen thong", "du an truyen thong")):
            return "3.11", 3
        if any(_has_phrase(full_padded, p) for p in ("chay bo", "viet da", "uprace", "marathon", "thanh nien khoe", "thu thach suc khoe")):
            return "3.2", 3
        if is_contest:
            return "3.1", 3
        if any(_has_phrase(full_padded, p) for p in ("ngay hoi viec lam", "career fair", "job fair", "tu van tuyen sinh")):
            return "3.8", 3
        if any(_has_phrase(full_padded, p) for p in ("lich su phat trien nha truong", "truyen thong nha truong")):
            return "3.9", 3
        if any(_has_phrase(full_padded, p) for p in ("doi thoai voi lanh dao", "gop y van ban")):
            return "3.10", 3
        if any(_has_phrase(full_padded, p) for p in ("huy dong luc luong", "kham suc khoe")):
            return "3.3", 3
        return "3.5", 3

    if assigned_cat == 4:
        if any(_has_phrase(full_padded, p) for p in ("hien mau", "giot hong", "giot mau")):
            return "4.2", 4
        if any(
            _has_phrase(full_padded, p)
            for p in ("hien phap", "phap luat", "luat cu tru", "phong chong ma tuy", "phong chong te nan", "lua dao mang", "an toan giao thong")
        ):
            return "4.6", 4
        if any(_has_phrase(full_padded, p) for p in ("quyen gop", "gay quy", "ung ho dong bao")) and not any(
            _has_phrase(full_padded, p) for p in ("tinh nguyen vien", "chien dich tinh nguyen", "mua he xanh", "xuan tinh nguyen")
        ):
            return "4.5", 4
        return "4.3", 4

    if assigned_cat == 5:
        if any(_has_phrase(full_padded, p) for p in ("bai bao khoa hoc quoc te", "tap chi quoc te")):
            return "5.4", 5
        if any(_has_phrase(full_padded, p) for p in ("bai bao khoa hoc", "tap chi trong nuoc", "hoi thao khoa hoc cap quoc gia")):
            return "5.5", 5
        if any(_has_phrase(full_padded, p) for p in ("hoc bong vuot kho",)):
            return "5.7", 5
        if any(_has_phrase(full_padded, p) for p in ("huy chuong", "bang khen", "giay khen hieu truong")):
            return "5.1", 5
        return "5.2", 5

    # Unclassified event (category is None): infer across all categories in priority order
    if any(_has_phrase(full_padded, p) for p in ("hien mau", "giot hong", "giot mau")):
        return "4.2", 4

    if any(
        _has_phrase(full_padded, p)
        for p in ("hien phap", "tuyen truyen phap luat", "ngay phap luat", "luat cu tru", "lua dao mang", "buon ban nguoi", "an toan giao thong")
    ):
        return "4.6", 4

    if any(_has_phrase(full_padded, p) for p in ("danh gia mon hoc", "danh gia giang day")):
        return "2.2", 2

    if any(_has_phrase(full_padded, p) for p in ("khao sat", "dieu tra xa hoi", "lay y kien sinh vien")) and not is_contest:
        return "2.3", 2

    if any(
        _has_phrase(full_padded, p)
        for p in ("mua he xanh", "xuan tinh nguyen", "tinh nguyen", "thien nguyen", "volunteering", "volunteer")
    ):
        return "4.3", 4

    if any(_has_phrase(full_padded, p) for p in ("quyen gop", "gay quy", "fundraising")):
        return "4.5", 4

    if any(
        _has_phrase(full_padded, p)
        for p in ("mac lenin", "tu tuong ho chi minh", "ly luan chinh tri", "tam nhin xuyen the ky", "theo dau chan nguoi")
    ):
        return "3.1", 3

    if any(
        _has_phrase(full_padded, p)
        for p in ("viet da", "chay bo", "uprace", "marathon", "thanh nien khoe", "thu thach suc khoe")
    ):
        return "3.2", 3

    if any(_has_phrase(full_padded, p) for p in ("eureka", "de tai nghien cuu khoa hoc", "sinh vien nghien cuu khoa hoc")):
        return "1.5", 1

    if is_contest:
        if any(
            _has_phrase(full_padded, p)
            for p in (
                "hoc thuat",
                "academic",
                "lap trinh",
                "coding",
                "hackathon",
                "datathon",
                "icpc",
                "ctf",
                "ai",
                "data",
                "toan",
                "vat ly",
                "hoa hoc",
                "sinh hoc",
                "tin hoc",
                "khoi nghiep",
                "khoa hoc",
            )
        ):
            return "1.6", 1
        return "3.1", 3

    if any(_has_phrase(full_padded, p) for p in ("ngay hoi viec lam", "career fair", "job fair", "tu van tuyen sinh")):
        return "3.8", 3

    if any(
        _has_phrase(full_padded, p)
        for p in (
            "hoi thao",
            "workshop",
            "seminar",
            "chuyen de",
            "hoc thuat",
            "academic",
            "huong nghiep",
            "viet va cham diem cv",
            "cv",
            "phong van",
            "thi thu toeic",
            "mini course",
            "summer school",
            "khoa hoc",
        )
    ):
        # Distinguish cultural/craft/tradition workshops (3.5/3.9) from academic/skill/career workshops (1.9)
        if any(_has_phrase(title_padded, p) for p in ("truyen thong", "ngay van hoa", "co phuc", "thu cong")) and not any(
            _has_phrase(full_padded, p) for p in ("hoc thuat", "khoa hoc", "ai", "lap trinh", "huong nghiep", "cv", "toeic")
        ):
            return "3.5", 3
        return "1.9", 1

    if any(_has_phrase(full_padded, p) for p in ("kham suc khoe", "huy dong luc luong")):
        return "3.3", 3

    if any(_has_phrase(full_padded, p) for p in ("lich su phat trien nha truong", "truyen thong nha truong", "ngay nha giao", "tri an thay co", "le ky niem")):
        return "3.9", 3

    if any(
        _has_phrase(full_padded, p)
        for p in ("ngay hoi", "festival", "van hoa", "van nghe", "ca nhac", "the thao", "trien lam", "cong dong", "tan sinh vien", "sinh hoat")
    ):
        return "3.5", 3

    return None, None


def _resolve_award_tier(event: Mapping[str, Any], item_id: str) -> Optional[str]:
    text = _pad(normalize_vi(f"{event.get('title') or ''} {event.get('description') or ''} {event.get('award') or ''}"))
    if item_id == "5.1":
        if _has_phrase(text, "quoc te"):
            return "quoc_te"
        if _has_phrase(text, "quoc gia") or _has_phrase(text, "giam doc dai hoc quoc gia"):
            return "quoc_gia"
        if _has_phrase(text, "hieu truong") or _has_phrase(text, "bang khen"):
            return "truong"
        if _has_phrase(text, "giay khen"):
            return "doan_the"
        return None
    if item_id == "5.2":
        if any(_has_phrase(text, p) for p in ("giai nhat", "giai dac biet", "vo dich", "quan quan")):
            return "first"
        if any(_has_phrase(text, p) for p in ("giai nhi", "giai ba", "khuyen khich", "giai phu", "a quan")):
            return "other"
        return None
    return None


# =====================================================================
# CORE CRD CALCULATOR CLASS & MODULE-LEVEL FUNCTIONS
# =====================================================================

class CreditCalculator:
    """Deterministic CRD engine for `credit_calc`, `gain_calc`, and `drl_database.db` workflows."""

    def __init__(self, db: Optional[DRLDatabaseManager] = None) -> None:
        self._db = db

    @property
    def db(self) -> DRLDatabaseManager:
        if self._db is None:
            self._db = get_drl_db()
        return self._db

    def _load_event(self, event_or_id: Union[int, Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if isinstance(event_or_id, Mapping):
            return dict(event_or_id)
        if isinstance(event_or_id, int):
            return self.db.fetch_one(
                "SELECT * FROM events WHERE event_id = ?;",
                (int(event_or_id),),
                table="events",
            )
        raise TypeError(f"Expected event_id (int) or event mapping, got {type(event_or_id).__name__}")

    def _load_user_and_rubric(
        self,
        user_or_id: Optional[Union[str, Mapping[str, Any]]],
    ) -> Tuple[Optional[List[float]], List[float]]:
        """Load `(user_credits_5, max_credit_5)` from user object/ID and `rubric` table."""
        if user_or_id is None:
            return None, [float(x) for x in DEFAULT_MAX_CREDIT]

        if isinstance(user_or_id, Mapping):
            user_row = dict(user_or_id)
        else:
            user_row = self.db.fetch_one(
                "SELECT * FROM users WHERE user_id = ?;",
                (str(user_or_id),),
                table="users",
            )

        if not user_row:
            return [0.0] * 5, [float(x) for x in DEFAULT_MAX_CREDIT]

        user_credits = _coerce_5_floats(user_row.get("credits"), (0, 0, 0, 0, 0))
        school_id = user_row.get("school_id")
        max_credits = [float(x) for x in DEFAULT_MAX_CREDIT]
        if school_id is not None:
            rubric_row = self.db.fetch_one(
                "SELECT * FROM rubric WHERE school_id = ?;",
                (int(school_id),),
                table="rubric",
            )
            if rubric_row and rubric_row.get("max_credit"):
                max_credits = _coerce_5_floats(rubric_row["max_credit"], DEFAULT_MAX_CREDIT)

        return user_credits, max_credits

    def estimate_event_credit(
        self,
        event_or_id: Union[int, Mapping[str, Any]],
        role: Optional[str] = "participant",
        *,
        user_credits: Optional[Sequence[Union[int, float]]] = None,
        max_credit: Optional[Sequence[Union[int, float]]] = None,
        use_cached_min_credit: bool = False,
    ) -> CreditEstimate:
        """Evaluate eligibility, raw `(P_exp, P_min, P_max)`, `P_eff`, and `source` for an event + role."""
        canon_role = normalize_role(role)
        event = self._load_event(event_or_id)
        if not event:
            eid = int(event_or_id) if isinstance(event_or_id, int) else None
            return CreditEstimate(
                event_id=eid,
                category=None,
                rubric_item_id=None,
                role=canon_role,
                is_credit_eligible=False,
                expected_points=0.0,
                min_points=0.0,
                max_points=0.0,
                effective_points=0.0,
                source="rule",
                capped_by="unclassified",
                warnings=("event_not_found",),
            )

        raw_eid = event.get("event_id")
        event_id = int(raw_eid) if isinstance(raw_eid, int) or (isinstance(raw_eid, str) and raw_eid.isdigit()) else None

        if bool(event.get("is_cancelled", False)):
            return CreditEstimate(
                event_id=event_id,
                category=event.get("category") if isinstance(event.get("category"), int) else None,
                rubric_item_id=None,
                role=canon_role,
                is_credit_eligible=False,
                expected_points=0.0,
                min_points=0.0,
                max_points=0.0,
                effective_points=0.0,
                source="rule",
                capped_by="cancelled",
                warnings=("event_cancelled",),
            )

        warnings: List[str] = []

        # 1. Infer rubric sub-item & category
        base_item_id, category = infer_rubric_item(event)
        if base_item_id == "6.4":
            return CreditEstimate(
                event_id=event_id,
                category=None,
                rubric_item_id="6.4",
                role=canon_role,
                is_credit_eligible=False,
                expected_points=0.0,
                min_points=0.0,
                max_points=0.0,
                effective_points=0.0,
                source="rule",
                capped_by="mandatory_event",
                warnings=("mandatory_event_avoid_penalty",),
            )

        # 2. Check explicit non-DRL negation or explicit DRL point announcement
        is_negated, explicit_pts = extract_explicit_points(event, role=canon_role)
        if is_negated:
            return CreditEstimate(
                event_id=event_id,
                category=category,
                rubric_item_id=base_item_id,
                role=canon_role,
                is_credit_eligible=False,
                expected_points=0.0,
                min_points=0.0,
                max_points=0.0,
                effective_points=0.0,
                source="explicit",
                capped_by="explicit_negation",
                warnings=("explicit_non_drl_negation",),
            )

        if base_item_id is None or base_item_id not in RUBRIC_ITEMS:
            if explicit_pts is not None:
                fallback_cat = category if category is not None else 3
                expected_pts = float(explicit_pts)
                min_pts = expected_pts
                max_pts = expected_pts
                source = "explicit"
                eff_item_id = None
            else:
                return CreditEstimate(
                    event_id=event_id,
                    category=category,
                    rubric_item_id=None,
                    role=canon_role,
                    is_credit_eligible=False,
                    expected_points=0.0,
                    min_points=0.0,
                    max_points=0.0,
                    effective_points=0.0,
                    source="rule",
                    capped_by="unclassified",
                    warnings=("unmapped_event",),
                )
        else:
            spec = RUBRIC_ITEMS[base_item_id]
            eff_item_id = base_item_id
            redirects = spec.get("role_redirect") or {}
            if canon_role in redirects:
                eff_item_id = redirects[canon_role]
                spec = RUBRIC_ITEMS[eff_item_id]
                warnings.append(f"role_redirected_to:{eff_item_id}")

            fallback_cat = spec["category"]
            inferred_level: Optional[str] = None
            inferred_duration: Optional[str] = None

            if explicit_pts is not None:
                expected_pts = float(explicit_pts)
                min_pts = expected_pts
                max_pts = expected_pts
                source = "explicit"
            else:
                points_table = spec["points"]
                role_variants = points_table.get(canon_role) or points_table.get("participant") or {}
                variant_by = spec.get("variant_by", "none")

                if not role_variants:
                    expected_pts, min_pts, max_pts, source = 0.0, 0.0, 0.0, "rule_fallback"
                    warnings.append("missing_role_points")
                elif variant_by == "none" or "any" in role_variants:
                    val = float(role_variants["any"])
                    expected_pts, min_pts, max_pts, source = val, val, val, "rule"
                elif variant_by == "level":
                    inferred_level = extract_event_level(event)
                    positive_vals = [float(v) for v in role_variants.values() if float(v) > 0]
                    all_vals = [float(v) for v in role_variants.values()]
                    max_pts = max(all_vals)
                    min_pts = float(
                        (spec.get("fallback_min") or {}).get(
                            canon_role,
                            min(positive_vals) if positive_vals else min(all_vals),
                        )
                    )
                    if inferred_level and inferred_level in role_variants:
                        val = float(role_variants[inferred_level])
                        expected_pts, min_pts, max_pts, source = val, val, val, "rule"
                    else:
                        expected_pts = float((spec.get("default_expected") or {}).get(canon_role, min_pts))
                        source = "rule_fallback"
                        warnings.append("level_missing_conservative_fallback")
                elif variant_by == "duration":
                    inferred_duration = extract_duration_bucket(event)
                    all_vals = [float(v) for v in role_variants.values()]
                    min_pts, max_pts = min(all_vals), max(all_vals)
                    if inferred_duration and inferred_duration in role_variants:
                        val = float(role_variants[inferred_duration])
                        expected_pts, min_pts, max_pts, source = val, val, val, "rule"
                    else:
                        expected_pts = float((spec.get("default_expected") or {}).get(canon_role, min_pts))
                        source = "rule_fallback"
                        warnings.append("duration_missing_conservative_fallback")
                elif variant_by == "award_tier":
                    tier = _resolve_award_tier(event, eff_item_id)
                    all_vals = [float(v) for v in role_variants.values()]
                    min_pts, max_pts = min(all_vals), max(all_vals)
                    if tier and tier in role_variants:
                        val = float(role_variants[tier])
                        expected_pts, min_pts, max_pts, source = val, val, val, "rule"
                    else:
                        expected_pts = float((spec.get("default_expected") or {}).get(canon_role, min_pts))
                        source = "rule_fallback"
                        warnings.append("award_tier_missing_conservative_fallback")
                else:
                    all_vals = [float(v) for v in role_variants.values()]
                    min_pts, max_pts = min(all_vals), max(all_vals)
                    expected_pts = min_pts
                    source = "rule_fallback"

        # If CRD(U) feed scoring passes an event that already has `events.min_credit` cached by CRD(S)
        if (
            use_cached_min_credit
            and canon_role == "participant"
            and isinstance(event.get("min_credit"), int)
            and event["min_credit"] >= 0
        ):
            cached = float(event["min_credit"])
            min_pts = min(min_pts, cached) if min_pts > 0 else cached
            expected_pts = cached

        inferred_lv = locals().get("inferred_level")
        inferred_dur = locals().get("inferred_duration")
        final_cat = category if category is not None else locals().get("fallback_cat")

        # 3. Apply Remaining-Rubric-Aware caps when `user_credits` is provided
        effective_pts = max(0.0, expected_pts)
        capped_by: Optional[str] = None

        if user_credits is not None and final_cat is not None and 1 <= final_cat <= 5:
            u_creds = _coerce_5_floats(user_credits, (0, 0, 0, 0, 0))
            m_creds = _coerce_5_floats(max_credit, DEFAULT_MAX_CREDIT)

            cat_idx = final_cat - 1
            rem_cat = max(0.0, m_creds[cat_idx] - max(0.0, u_creds[cat_idx]))
            current_total = sum(min(m_creds[i], max(0.0, u_creds[i])) for i in range(5))
            rem_total = max(0.0, float(TOTAL_MAX_SCORE) - current_total)

            if effective_pts > 0.0:
                if rem_cat <= 0.0:
                    effective_pts = 0.0
                    capped_by = "category"
                elif rem_total <= 0.0:
                    effective_pts = 0.0
                    capped_by = "total"
                else:
                    if effective_pts > rem_cat:
                        effective_pts = rem_cat
                        capped_by = "category"
                    if effective_pts > rem_total:
                        effective_pts = rem_total
                        capped_by = "total"

        is_eligible = expected_pts > 0.0 and (effective_pts > 0.0 or user_credits is None)

        return CreditEstimate(
            event_id=event_id,
            category=final_cat,
            rubric_item_id=locals().get("eff_item_id"),
            role=canon_role,
            is_credit_eligible=is_eligible,
            expected_points=expected_pts,
            min_points=min_pts,
            max_points=max_pts,
            effective_points=effective_pts,
            source=source,
            capped_by=capped_by,
            inferred_level=inferred_lv,
            inferred_duration=inferred_dur,
            warnings=tuple(warnings),
        )

    def credit_calc(
        self,
        event_or_id: Union[int, Mapping[str, Any]],
        role: Optional[str] = "participant",
    ) -> int:
        """Compute integer DRL credit for `(event_or_id, role)`."""
        est = self.estimate_event_credit(event_or_id, role=role)
        return est.credit

    def gain_calc_with_details(
        self,
        event_or_estimate: Union[int, Mapping[str, Any], CreditEstimate],
        user_or_id: Optional[Union[str, Mapping[str, Any]]] = None,
        *,
        role: Optional[str] = "participant",
        user_credits: Optional[Sequence[Union[int, float]]] = None,
        max_credit: Optional[Sequence[Union[int, float]]] = None,
        target_credits: Optional[Sequence[Union[int, float]]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute normalized `Gain` in `[0.0, 1.0]` and diagnostic breakdown.

        Formula:
          - If `not credit.is_credit_eligible` or `credit.effective_points <= 0`: Gain = 0.0
          - Else:
              P_use = min(P_eff, 0.7 * P_min + 0.3 * P_exp)  if source == "rule_fallback"
                    = P_eff                                   otherwise
              Gain  = clip_[0, 1]( min(1.0, P_use / 10.0) * (0.5 + 0.5 * target_gap_c / M_c) * r_src )
            with r_src in {1.00 (explicit), 0.90 (rule), 0.80 (rule_fallback)}.
        """
        loaded_u_creds, loaded_m_creds = self._load_user_and_rubric(user_or_id)
        u_creds = _coerce_5_floats(user_credits, loaded_u_creds or (0, 0, 0, 0, 0))
        m_creds = _coerce_5_floats(max_credit, loaded_m_creds)
        t_creds = _coerce_5_floats(target_credits, m_creds)

        if isinstance(event_or_estimate, CreditEstimate):
            credit = event_or_estimate
            # Re-apply user_credits cap if caller passed a raw CreditEstimate alongside user_credits
            if (user_credits is not None or user_or_id is not None) and credit.category and 1 <= credit.category <= 5:
                c_idx = credit.category - 1
                rem_cat = max(0.0, m_creds[c_idx] - max(0.0, u_creds[c_idx]))
                cur_tot = sum(min(m_creds[i], max(0.0, u_creds[i])) for i in range(5))
                rem_tot = max(0.0, float(TOTAL_MAX_SCORE) - cur_tot)
                p_eff = max(0.0, min(credit.effective_points, rem_cat, rem_tot))
                is_elig = credit.is_credit_eligible and p_eff > 0.0
            else:
                p_eff = credit.effective_points
                is_elig = credit.is_credit_eligible
        else:
            credit = self.estimate_event_credit(
                event_or_estimate,
                role=role,
                user_credits=u_creds,
                max_credit=m_creds,
                use_cached_min_credit=True,
            )
            p_eff = credit.effective_points
            is_elig = credit.is_credit_eligible

        if (not is_elig) or p_eff <= 0.0 or credit.category is None or not (1 <= credit.category <= 5):
            return 0.0, {
                "is_credit_eligible": False,
                "p_eff": 0.0,
                "p_use": 0.0,
                "g_pts": 0.0,
                "target_gap_c": 0.0,
                "m_c": 0.0,
                "m_gap": 0.0,
                "r_src": 0.0,
                "source": credit.source,
                "category": credit.category,
                "rubric_item_id": credit.rubric_item_id,
            }

        if credit.source == "rule_fallback":
            blended = GAIN_FALLBACK_MIN_WEIGHT * credit.min_points + (1.0 - GAIN_FALLBACK_MIN_WEIGHT) * credit.expected_points
            p_use = min(p_eff, blended)
        else:
            p_use = p_eff

        g_pts = min(1.0, max(0.0, p_use / GAIN_P_REF))

        cat_idx = credit.category - 1
        m_c = max(1.0, float(m_creds[cat_idx]))
        current_c = min(m_c, max(0.0, float(u_creds[cat_idx])))
        target_c = min(m_c, max(0.0, float(t_creds[cat_idx])))
        target_gap_c = max(0.0, target_c - current_c)

        gap_ratio = max(0.0, min(1.0, target_gap_c / m_c))
        m_gap = GAIN_GAP_FLOOR + (1.0 - GAIN_GAP_FLOOR) * gap_ratio

        r_src = float(SOURCE_RELIABILITY.get(credit.source, 0.90))
        gain = round(max(0.0, min(1.0, g_pts * m_gap * r_src)), 6)

        return gain, {
            "is_credit_eligible": True,
            "p_min": round(credit.min_points, 4),
            "p_exp": round(credit.expected_points, 4),
            "p_eff": round(p_eff, 4),
            "p_use": round(p_use, 4),
            "g_pts": round(g_pts, 6),
            "target_gap_c": round(target_gap_c, 4),
            "m_c": round(m_c, 4),
            "gap_ratio": round(gap_ratio, 6),
            "m_gap": round(m_gap, 6),
            "r_src": round(r_src, 6),
            "source": credit.source,
            "category": credit.category,
            "rubric_item_id": credit.rubric_item_id,
        }

    def gain_calc(
        self,
        event_or_estimate: Union[int, Mapping[str, Any], CreditEstimate],
        user_or_id: Optional[Union[str, Mapping[str, Any]]] = None,
        *,
        role: Optional[str] = "participant",
        user_credits: Optional[Sequence[Union[int, float]]] = None,
        max_credit: Optional[Sequence[Union[int, float]]] = None,
        target_credits: Optional[Sequence[Union[int, float]]] = None,
    ) -> float:
        """Return normalized `gain` in `[0.0, 1.0]` for recommendation scoring."""
        gain, _ = self.gain_calc_with_details(
            event_or_estimate,
            user_or_id=user_or_id,
            role=role,
            user_credits=user_credits,
            max_credit=max_credit,
            target_credits=target_credits,
        )
        return gain

    # ------------------------------------------------------------------
    # SYSTEM_ARCHITECTURE.md Database Workflows (`CRD(S)` & `CRD(U)`)
    # ------------------------------------------------------------------

    def update_missing_min_credits(self, *, backfill_null_category: bool = False) -> int:
        """CRD(S) workflow step 3: populate `events.min_credit` for active events where `min_credit IS NULL`."""
        rows = self.db.fetch_all(
            "SELECT * FROM events WHERE is_cancelled = FALSE AND min_credit IS NULL ORDER BY event_id ASC;",
            table="events",
        )
        updated = 0
        for ev in rows:
            est = self.estimate_event_credit(ev, role="participant")
            updates: Dict[str, Any] = {"min_credit": est.credit}
            if backfill_null_category and ev.get("category") is None and est.category is not None:
                updates["category"] = est.category
            changed = self.db.update_records("events", updates, {"event_id": ev["event_id"]})
            if changed > 0:
                updated += 1
        return updated

    def finalize_completed_subscriptions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """CRD(U) workflow step 7: finalize `event_subscription` rows where `status = 'done' AND credit IS NULL`.

        Atomically sets `event_subscription.credit = credit_calc(event_id, role)` and updates
        `users.credits[category - 1] = min(rubric.max_credit[category - 1], users.credits[category - 1] + credit)`
        so each completed event is counted at most once while preserving manual baseline credits.
        """
        if user_id is not None:
            pending = self.db.fetch_all(
                "SELECT * FROM event_subscription WHERE status = 'done' AND credit IS NULL AND user_id = ?;",
                (str(user_id),),
                table="event_subscription",
            )
        else:
            pending = self.db.fetch_all(
                "SELECT * FROM event_subscription WHERE status = 'done' AND credit IS NULL;",
                table="event_subscription",
            )

        finalized: List[Dict[str, Any]] = []
        for sub in pending:
            uid = str(sub["user_id"])
            eid = int(sub["event_id"])
            sub_role = normalize_role(sub.get("role"))

            ev = self._load_event(eid)
            if not ev:
                continue

            est = self.estimate_event_credit(ev, role=sub_role)
            awarded_credit = est.credit
            cat = ev.get("category") if isinstance(ev.get("category"), int) else est.category

            with self.db.transaction() as conn:
                # Guard `credit IS NULL` inside the transaction to prevent double-counting
                cur = conn.execute(
                    "UPDATE event_subscription SET credit = ?, role = ? "
                    "WHERE user_id = ? AND event_id = ? AND status = 'done' AND credit IS NULL;",
                    (awarded_credit, sub_role, uid, eid),
                )
                if cur.rowcount == 0:
                    continue

                new_user_credits: Optional[List[int]] = None
                if awarded_credit > 0 and cat is not None and 1 <= cat <= 5:
                    u_row = conn.execute(
                        "SELECT credits, school_id FROM users WHERE user_id = ?;",
                        (uid,),
                    ).fetchone()
                    if u_row is not None:
                        raw_u_credits = decode_json_list(u_row["credits"], default=[0, 0, 0, 0, 0])
                        u_ints = [int(raw_u_credits[i]) if i < len(raw_u_credits) else 0 for i in range(5)]

                        max_caps = list(DEFAULT_MAX_CREDIT)
                        if u_row["school_id"] is not None:
                            r_row = conn.execute(
                                "SELECT max_credit FROM rubric WHERE school_id = ?;",
                                (int(u_row["school_id"]),),
                            ).fetchone()
                            if r_row is not None and r_row["max_credit"]:
                                decoded_caps = decode_json_list(r_row["max_credit"], default=list(DEFAULT_MAX_CREDIT))
                                max_caps = [
                                    int(decoded_caps[i]) if i < len(decoded_caps) else DEFAULT_MAX_CREDIT[i]
                                    for i in range(5)
                                ]

                        idx = cat - 1
                        u_ints[idx] = min(max_caps[idx], max(0, u_ints[idx]) + awarded_credit)
                        new_user_credits = u_ints
                        conn.execute(
                            "UPDATE users SET credits = ? WHERE user_id = ?;",
                            (encode_json_list(u_ints), uid),
                        )

                finalized.append(
                    {
                        "user_id": uid,
                        "event_id": eid,
                        "role": sub_role,
                        "category": cat,
                        "rubric_item_id": est.rubric_item_id,
                        "credit": awarded_credit,
                        "updated_user_credits": new_user_credits,
                    }
                )

        return finalized


# =====================================================================
# CONVENIENCE MODULE-LEVEL API (per SYSTEM_ARCHITECTURE.md)
# =====================================================================

def estimate_event_credit(
    event_or_id: Union[int, Mapping[str, Any]],
    role: Optional[str] = "participant",
    *,
    user_credits: Optional[Sequence[Union[int, float]]] = None,
    max_credit: Optional[Sequence[Union[int, float]]] = None,
    use_cached_min_credit: bool = False,
    db: Optional[DRLDatabaseManager] = None,
) -> CreditEstimate:
    """Module-level wrapper returning a detailed `CreditEstimate`."""
    return CreditCalculator(db=db).estimate_event_credit(
        event_or_id,
        role=role,
        user_credits=user_credits,
        max_credit=max_credit,
        use_cached_min_credit=use_cached_min_credit,
    )


def credit_calc(
    event_or_id: Union[int, Mapping[str, Any]],
    role: Optional[str] = "participant",
    *,
    db: Optional[DRLDatabaseManager] = None,
) -> int:
    """Calculate integer DRL credit for `event_or_id` and `role` (`'participant'`, `'host'`, `'collaborator'`)."""
    return CreditCalculator(db=db).credit_calc(event_or_id, role=role)


def gain_calc(
    event_or_estimate: Union[int, Mapping[str, Any], CreditEstimate],
    user_or_id: Optional[Union[str, Mapping[str, Any]]] = None,
    *,
    role: Optional[str] = "participant",
    user_credits: Optional[Sequence[Union[int, float]]] = None,
    max_credit: Optional[Sequence[Union[int, float]]] = None,
    target_credits: Optional[Sequence[Union[int, float]]] = None,
    db: Optional[DRLDatabaseManager] = None,
) -> float:
    """Calculate normalized `gain` in `[0.0, 1.0]` for event recommendation."""
    return CreditCalculator(db=db).gain_calc(
        event_or_estimate,
        user_or_id=user_or_id,
        role=role,
        user_credits=user_credits,
        max_credit=max_credit,
        target_credits=target_credits,
    )


def gain_calc_with_details(
    event_or_estimate: Union[int, Mapping[str, Any], CreditEstimate],
    user_or_id: Optional[Union[str, Mapping[str, Any]]] = None,
    *,
    role: Optional[str] = "participant",
    user_credits: Optional[Sequence[Union[int, float]]] = None,
    max_credit: Optional[Sequence[Union[int, float]]] = None,
    target_credits: Optional[Sequence[Union[int, float]]] = None,
    db: Optional[DRLDatabaseManager] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Calculate `(gain, details_dict)` with full formula trace."""
    return CreditCalculator(db=db).gain_calc_with_details(
        event_or_estimate,
        user_or_id=user_or_id,
        role=role,
        user_credits=user_credits,
        max_credit=max_credit,
        target_credits=target_credits,
    )


def update_missing_min_credits(
    *,
    backfill_null_category: bool = False,
    db: Optional[DRLDatabaseManager] = None,
) -> int:
    """CRD(S): Populate `events.min_credit` where `min_credit IS NULL`."""
    return CreditCalculator(db=db).update_missing_min_credits(
        backfill_null_category=backfill_null_category,
    )


def finalize_completed_subscriptions(
    user_id: Optional[str] = None,
    *,
    db: Optional[DRLDatabaseManager] = None,
) -> List[Dict[str, Any]]:
    """CRD(U): Finalize `event_subscription.credit` where `status = 'done' AND credit IS NULL`."""
    return CreditCalculator(db=db).finalize_completed_subscriptions(user_id=user_id)
