"""Offline test-fixture adapter converting legacy data/events.json to EventNormalized (D-6).

Per approved Decision D-6, upstream M1 & M2 own raw scraping, normalization, and missing-field
handling in production. This module exists strictly for offline testing and evaluation against
the legacy static snapshot in data/events.json.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from configs.model_config import PROCESSED_DATA_DIR, RAW_EVENTS_JSON_PATH
from src.contract_validator import parse_iso_datetime, validate_event_normalized
from src.schemas import EventNormalized, TargetAudienceStructured

_FREE_PATTERNS = ("mien phi", "free", "khong mat phi", "khong thu phi")
_COHORT_RE = re.compile(r"\b(?:k|khoa)\s*(20\d{2}|\d{2})\b", re.IGNORECASE)
_K_FEE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*k\b", re.IGNORECASE)
_VND_FEE_RE = re.compile(
    r"(\d{1,3}(?:[.,]\d{3})+|\d+)\s*(?:vnd|vnđ|đ|d|dong|đồng)\b",
    re.IGNORECASE,
)


def _strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", (text or "").lower().replace("đ", "d"))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", s).strip()


def parse_vietnamese_fee(raw_fee: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    """Parse messy Vietnamese fee strings into integer VND.

    Returns (fee_vnd, flag_or_none).
    """
    if raw_fee is None:
        return None, "fee_missing"
    text = str(raw_fee).strip()
    if not text:
        return None, "fee_missing"

    norm = _strip_accents(text)
    if any(pat in norm for pat in _FREE_PATTERNS):
        return 0, None

    m_k = _K_FEE_RE.search(norm)
    if m_k:
        amount = float(m_k.group(1).replace(",", "."))
        return int(round(amount * 1000)), None

    m_vnd = _VND_FEE_RE.search(text)
    if m_vnd:
        digits = re.sub(r"[.,]", "", m_vnd.group(1))
        return int(digits), None

    return None, "fee_unparseable"


def parse_online_mode(raw: Dict[str, Any]) -> Tuple[Optional[str], Optional[bool], Optional[str]]:
    """Harmonize online_mode and is_online from legacy events.json fields."""
    raw_mode = (raw.get("online_mode") or "").strip().lower()
    raw_is_online = raw.get("is_online")
    formats = [str(f).strip().lower() for f in (raw.get("format") or []) if f]

    mode: Optional[str] = raw_mode if raw_mode in ("online", "offline", "hybrid") else None
    if mode is None:
        has_on = any("online" in f for f in formats)
        has_off = any("offline" in f for f in formats)
        if has_on and has_off:
            mode = "hybrid"
        elif has_on:
            mode = "online"
        elif has_off:
            mode = "offline"
        elif raw_is_online is True:
            mode = "online"
        elif raw_is_online is False:
            mode = "offline"

    is_online: Optional[bool]
    if mode == "online":
        is_online = True
    elif mode == "offline":
        is_online = False
    elif mode == "hybrid":
        is_online = True
    else:
        is_online = None

    flag = "online_mode_missing" if mode is None else None
    return mode, is_online, flag


def parse_target_audience(raw_audience: Any) -> TargetAudienceStructured:
    """Convert free-text target_audience list into TargetAudienceStructured."""
    if not raw_audience:
        return TargetAudienceStructured()
    raw_list = [str(x).strip() for x in raw_audience if x and str(x).strip()]
    joined = _strip_accents(" ".join(raw_list))
    cohorts = sorted({m.group(0).upper().replace(" ", "") for m in _COHORT_RE.finditer(joined)})
    all_students = any(
        cue in joined
        for cue in (
            "toan the sinh vien",
            "tat ca sinh vien",
            "moi sinh vien",
            "sinh vien hcmus",
            "sinh vien dang theo hoc",
        )
    ) or (joined == "sinh vien")
    return TargetAudienceStructured(
        raw_texts=raw_list,
        cohorts=cohorts,
        faculties=[],
        classes=[],
        all_students=all_students,
    )


def convert_legacy_event(raw: Dict[str, Any], index: int = 0) -> EventNormalized:
    """Convert one raw record from data/events.json into a validated EventNormalized contract."""
    flags: List[str] = []
    metadata: Dict[str, Any] = {
        "extraction_method": raw.get("extraction_method"),
        "scraped_at": raw.get("scraped_at"),
        "posted_at": raw.get("posted_at"),
        "fanpage_name": raw.get("fanpage_name"),
    }

    event_id = str(raw.get("post_id") or f"legacy_{index:04d}").strip()
    title = str(raw.get("title") or "").strip()
    description = str(raw.get("description") or "").strip()
    organizer = str(raw.get("organizer") or raw.get("fanpage_name") or "").strip()
    if not organizer:
        flags.append("organizer_missing")

    start_dt_str = raw.get("start_datetime")
    end_dt_str = raw.get("end_datetime")
    reg_start_str = raw.get("reg_start_datetime")
    reg_end_str = raw.get("reg_end_datetime")

    if not start_dt_str:
        flags.append("start_datetime_missing")
    if not end_dt_str:
        flags.append("end_datetime_missing")
    if not reg_end_str:
        flags.append("reg_end_datetime_missing")

    # Sanitize reg_end_datetime > start_datetime anomalies while preserving raw value
    start_dt = parse_iso_datetime(start_dt_str) if start_dt_str else None
    reg_end_dt = parse_iso_datetime(reg_end_str) if reg_end_str else None
    if start_dt and reg_end_dt and reg_end_dt > start_dt:
        flags.append("reg_end_after_start")
        metadata["raw_reg_end_datetime"] = reg_end_str
        reg_end_str = None

    online_mode, is_online, mode_flag = parse_online_mode(raw)
    if mode_flag:
        flags.append(mode_flag)

    fee_vnd, fee_flag = parse_vietnamese_fee(raw.get("fee"))
    if fee_flag:
        flags.append(fee_flag)

    # Legacy events.json never provides capacity / seats_left
    flags.extend(["capacity_missing", "seats_left_missing"])

    return EventNormalized(
        event_id=event_id,
        title=title,
        description=description,
        organizer=organizer,
        start_datetime=start_dt_str,
        end_datetime=end_dt_str,
        reg_start_datetime=reg_start_str,
        reg_end_datetime=reg_end_str,
        online_mode=online_mode,
        is_online=is_online,
        location=raw.get("location"),
        fee_vnd=fee_vnd,
        fee_raw=raw.get("fee"),
        target_audience=parse_target_audience(raw.get("target_audience")),
        capacity=None,
        seats_left=None,
        registration_url=raw.get("registration_url") or None,
        source_type=str(raw.get("source_type") or "unknown"),
        source_url=raw.get("post_url") or raw.get("source_link"),
        categories=[str(c).strip() for c in (raw.get("category") or []) if c],
        missing_field_flags=flags,
        metadata=metadata,
    )


def convert_events_file(
    input_path: Path = RAW_EVENTS_JSON_PATH,
    output_path: Optional[Path] = None,
) -> Tuple[List[EventNormalized], Dict[str, Any]]:
    """Convert all events in data/events.json into EventNormalized fixtures and validate them."""
    with open(input_path, encoding="utf-8") as handle:
        raw_events = json.load(handle)

    normalized: List[EventNormalized] = []
    invalid_count = 0
    warning_count = 0
    flag_counts: Dict[str, int] = {}

    for idx, raw in enumerate(raw_events):
        ev = convert_legacy_event(raw, index=idx)
        report = validate_event_normalized(ev)
        if not report.is_valid:
            invalid_count += 1
        warning_count += len(report.warnings)
        for fl in ev.missing_field_flags:
            flag_counts[fl] = flag_counts.get(fl, 0) + 1
        normalized.append(ev)

    summary = {
        "input_path": str(input_path),
        "total_events": len(normalized),
        "invalid_events": invalid_count,
        "total_warnings": warning_count,
        "missing_field_flag_counts": dict(sorted(flag_counts.items())),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "event-normalized-fixture-v1",
            "summary": summary,
            "events": [ev.to_dict() for ev in normalized],
        }
        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(payload, out, ensure_ascii=False, indent=2)

    return normalized, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert legacy data/events.json into EventNormalized offline test fixtures."
    )
    parser.add_argument("--input", type=Path, default=RAW_EVENTS_JSON_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "test_fixtures_events.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing output.")
    args = parser.parse_args()

    _, summary = convert_events_file(
        input_path=args.input,
        output_path=None if args.dry_run else args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
