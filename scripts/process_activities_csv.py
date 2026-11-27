#!/usr/bin/env python3
"""Deterministic ETL script to process raw HCMUS activity records into CSV datasets.

Reads either an `.xlsx` export (using pure stdlib `zipfile` + `xml.etree.ElementTree`,
fixing IEEE-754 float corruption on `Mục` codes like `1.1`, `5.10`, `5.11`) or a saved
raw CSV (`NH,Mục,HK,HĐ`), extracts and normalizes `Organizer`, peels training-program
suffixes (`ĐA`, `CTĐA`, `CTDA`, `ĐT`, `CQ`, `Chuẩn`), extracts `Vai_Tro` cues, looks up
`Tieu_Chi_Quy_Che` and `Diem_Toi_Da` directly from `data/raw/rubric.yaml`, deduplicates,
and writes:
  - `./res/events_full.csv`
  - `./data/raw/events_dataset.csv`
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUBRIC_PATH = PROJECT_ROOT / "data" / "raw" / "rubric.yaml"
DEFAULT_RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "activities_hk2_25_26_raw.csv"
DEFAULT_EVENTS_FULL_PATH = PROJECT_ROOT / "res" / "events_full.csv"
DEFAULT_EVENTS_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "events_dataset.csv"

XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

# Ordered canonical organizer variants (longest prefixes matched first).
# Every key is normalized via Unicode NFC + whitespace collapse before lookup.
ORGANIZER_ALIAS_MAP: Dict[str, str] = {
    "KTX ĐHQG-HCM": "KTX ĐHQG-HCM",
    "CDCS KHLN": "CDCS KHLN",
    "LCH SH-CNSH": "LCH SH-CNSH",
    "LCH VL-VLKT": "LCH VL-VLKT",
    "LCH VLVLKT": "LCH VL-VLKT",
    "LCH KHCNVL": "LCH KHCNVL",
    "ĐK SH-CNSH": "ĐK SH-CNSH",
    "ĐK VL-VLKT": "ĐK VL-VLKT",
    "ĐK VLVLKT": "ĐK VL-VLKT",
    "ĐK KHCNVL": "ĐK KHCNVL",
    "ĐKKHCNVL": "ĐK KHCNVL",
    "LCH CNTT": "LCH CNTT",
    "LCH ĐTVT": "LCH ĐTVT",
    "LCH DTVT": "LCH ĐTVT",
    "CLB TTĐT": "CLB TTĐT",
    "CLB SV5T": "CLB SV5T",
    "CLB MICUS": "CLB MICUS",
    "U-Sci Hub": "U-Sci Hub",
    "ĐK CNTT": "ĐK CNTT",
    "ĐK ĐTVT": "ĐK ĐTVT",
    "ĐK DTVT": "ĐK ĐTVT",
    "LCH TTH": "LCH TTH",
    "SH-CNSH": "SH-CNSH",
    "VL-VLKT": "VL-VLKT",
    "KHXH&NV": "KHXHNV",
    "CLB BEE": "CLB BEE",
    "CLB LLT": "CLB LLT",
    "CLB TTV": "CLB TTV",
    "CLB ONC": "CLB ONC",
    "CLB IHC": "CLB IHC",
    "CH KHLN": "CH KHLN",
    "Thuvien": "Thuvien",
    "KHCNVL": "KHCNVL",
    "KHXHNV": "KHXHNV",
    "LCH HH": "LCH HH",
    "LCH MT": "LCH MT",
    "LCH ĐC": "LCH ĐC",
    "LCH DC": "LCH ĐC",
    "CLB KN": "CLB KN",
    "ĐK TTH": "ĐK TTH",
    "ĐT-VT": "ĐT-VT",
    "NCKNK": "NCKNK",
    "NVHSV": "NVHSV",
    "MICUS": "CLB MICUS",
    "ĐK HH": "ĐK HH",
    "ĐK MT": "ĐK MT",
    "ĐKHH": "ĐK HH",
    "ĐKMT": "ĐK MT",
    "CLBS": "CLBS",
    "CTSV": "CTSV",
    "CNTT": "CNTT",
    "CTXH": "CTXH",
    "VNXK": "VNXK",
    "BCSL": "BCSL",
    "CBSV": "CBSV",
    "TTTT": "TTTT",
    "KHLN": "KHLN",
    "KHTN": "KHTN",
    "SV5T": "CLB SV5T",
    "ĐTN": "ĐTN",
    "HSV": "HSV",
    "TTH": "TTH",
    "IHC": "CLB IHC",
    "KMT": "MT",
    "HH": "HH",
    "MT": "MT",
    "DC": "ĐC",
}

SORTED_ORGANIZER_PREFIXES: List[str] = sorted(
    ORGANIZER_ALIAS_MAP.keys(),
    key=lambda s: (-len(s), s),
)

# Standalone prefixes allowed to match with whitespace separator (even without '-'/'_'/'_')
STANDALONE_SPACE_PREFIXES = {
    "ĐK SH-CNSH",
    "ĐK VL-VLKT",
    "ĐK VLVLKT",
    "ĐK KHCNVL",
    "ĐKKHCNVL",
    "ĐK CNTT",
    "ĐK ĐTVT",
    "ĐK DTVT",
    "ĐK TTH",
    "ĐK HH",
    "ĐKHH",
    "ĐK MT",
    "ĐKMT",
    "LCH SH-CNSH",
    "LCH VL-VLKT",
    "LCH VLVLKT",
    "LCH KHCNVL",
    "LCH CNTT",
    "LCH ĐTVT",
    "LCH DTVT",
    "LCH TTH",
    "LCH HH",
    "LCH MT",
    "LCH ĐC",
    "LCH DC",
    "SH-CNSH",
    "VL-VLKT",
    "KHXH&NV",
    "KHXHNV",
    "KHCNVL",
    "CDCS KHLN",
    "CH KHLN",
    "KTX ĐHQG-HCM",
    "CTSV",
    "ĐTN",
    "HSV",
    "TTTT",
    "CNTT",
}

# Explicit training-program suffixes to strip from the tail of Ten_Hoat_Dong
PROGRAM_SUFFIXES_EXACT = ("CTĐA", "CTDA", "ĐA", "ĐT", "CQ", "Chuẩn")

# Explicit role suffixes mapped to canonical Vai_Tro labels
ROLE_SUFFIX_SPECS: List[Tuple[str, str, bool]] = [
    # (pattern_token, canonical_role, case_sensitive)
    ("HĐLL", "Huy động lực lượng", True),
    ("HDLL", "Huy động lực lượng", True),
    ("BTC", "Ban tổ chức", True),
    ("CTV", "Cộng tác viên", True),
    ("TNV", "Tình nguyện viên", True),
    ("Tham gia", "Tham gia", False),
    ("TG", "Tham gia", True),
    ("TV", "Tham gia", True),
    ("Quyên góp", "Quyên góp", False),
    ("QG", "Quyên góp", True),
    ("Gây quỹ", "Gây quỹ", False),
    ("GQ", "Gây quỹ", True),
    ("Khen thưởng", "Khen thưởng", False),
    ("Giải thưởng", "Khen thưởng", False),
    ("Thành tích", "Khen thưởng", False),
    ("Đạt giải", "Khen thưởng", False),
]


def norm_text(text: Any) -> str:
    """Normalize Unicode NFC, convert NBSP to space, and collapse whitespace."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def format_xlsx_muc(raw_val: str, is_string_cell: bool) -> str:
    """Normalize `Mục` without IEEE-754 float corruption (preserving '5.10' etc.)."""
    s = norm_text(raw_val)
    if not s:
        return ""
    if is_string_cell:
        return s
    return f"{float(s):.10g}"


def read_xlsx_rows(xlsx_path: Path) -> List[Dict[str, str]]:
    """Read `NH, Mục, HK, HĐ` rows from the first sheet of an `.xlsx` file."""
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", XLSX_NS):
                shared_strings.append(
                    "".join(t.text or "" for t in si.findall(".//m:t", XLSX_NS))
                )

        sheet_names = sorted(
            n for n in zf.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
        )
        if not sheet_names:
            raise ValueError(f"No worksheet XML found in {xlsx_path}")
        sheet_root = ET.fromstring(zf.read(sheet_names[0]))

        raw_rows: List[Dict[str, Tuple[str, bool]]] = []
        for row_el in sheet_root.findall(".//m:sheetData/m:row", XLSX_NS):
            row_map: Dict[str, Tuple[str, bool]] = {}
            for cell_el in row_el.findall("m:c", XLSX_NS):
                ref = cell_el.attrib.get("r", "")
                col = "".join(ch for ch in ref if ch.isalpha())
                cell_type = cell_el.attrib.get("t")
                v_el = cell_el.find("m:v", XLSX_NS)
                is_el = cell_el.find("m:is", XLSX_NS)

                if cell_type == "s" and v_el is not None and v_el.text is not None:
                    row_map[col] = (shared_strings[int(v_el.text)], True)
                elif cell_type == "inlineStr" and is_el is not None:
                    val = "".join(x.text or "" for x in is_el.findall(".//m:t", XLSX_NS))
                    row_map[col] = (val, True)
                elif v_el is not None and v_el.text is not None:
                    row_map[col] = (v_el.text, cell_type == "str")
                else:
                    row_map[col] = ("", True)
            raw_rows.append(row_map)

    if not raw_rows:
        raise ValueError(f"Worksheet in {xlsx_path} is empty")

    headers = {col: norm_text(val[0]) for col, val in raw_rows[0].items()}
    expected = {"A": "NH", "B": "Mục", "C": "HK", "D": "HĐ"}
    for col, exp_name in expected.items():
        if headers.get(col) != exp_name:
            raise ValueError(
                f"Unexpected header at column {col}: got {headers.get(col)!r}, expected {exp_name!r}"
            )

    out: List[Dict[str, str]] = []
    for row_map in raw_rows[1:]:
        nh = norm_text(row_map.get("A", ("", True))[0])
        muc_raw, muc_is_str = row_map.get("B", ("", True))
        muc = format_xlsx_muc(muc_raw, muc_is_str)
        hk = norm_text(row_map.get("C", ("", True))[0])
        hd = norm_text(row_map.get("D", ("", True))[0])
        if not any((nh, muc, hk, hd)):
            continue
        out.append({"NH": nh, "Mục": muc, "HK": hk, "HĐ": hd})
    return out


def read_raw_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """Read `NH,Mục,HK,HĐ` rows from a raw CSV file."""
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        out: List[Dict[str, str]] = []
        for row in reader:
            nh = norm_text(row.get("NH"))
            muc = norm_text(row.get("Mục"))
            hk = norm_text(row.get("HK"))
            hd = norm_text(row.get("HĐ"))
            if not any((nh, muc, hk, hd)):
                continue
            out.append({"NH": nh, "Mục": muc, "HK": hk, "HĐ": hd})
    return out


def write_raw_csv(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    """Persist verbatim extracted raw rows (`NH,Mục,HK,HĐ`) as UTF-8 CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["NH", "Mục", "HK", "HĐ"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_points_value(val: Optional[float]) -> str:
    if val is None:
        return ""
    if float(val).is_integer():
        return str(int(val))
    return f"{val:g}"


def load_rubric_metadata(rubric_path: Path) -> Dict[str, Tuple[str, str]]:
    """Build `{ref: (Tieu_Chi_Quy_Che, Diem_Toi_Da)}` strictly from `data/raw/rubric.yaml`."""
    rubric = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
    ref_map: Dict[str, Tuple[str, str]] = {}

    for item in (rubric.get("items") or {}).values():
        if not isinstance(item, dict) or "ref" not in item:
            continue
        ref = str(item["ref"]).strip()
        name = norm_text(item.get("name", ""))

        values: List[float] = []
        points = item.get("points")
        if isinstance(points, dict):
            for role_val in points.values():
                if isinstance(role_val, dict):
                    for v in role_val.values():
                        if isinstance(v, (int, float)):
                            values.append(float(v))
                elif isinstance(role_val, (int, float)):
                    values.append(float(role_val))

        bands = item.get("bands")
        if not values and isinstance(bands, list):
            for b in bands:
                if isinstance(b, dict) and isinstance(b.get("points"), (int, float)):
                    values.append(float(b["points"]))

        best: Optional[float] = None
        if values:
            best = min(values) if all(v <= 0 for v in values) else max(values)
            cap = item.get("cap_points")
            if isinstance(cap, (int, float)) and best > 0:
                best = min(best, float(cap))

        ref_map[ref] = (name, _format_points_value(best))

    return ref_map


def extract_organizer(raw_hd: str) -> Tuple[str, str]:
    """Extract and normalize leading `Organizer` prefix, returning `(organizer, remainder)`."""
    text = norm_text(raw_hd)
    organizer = ""

    # Up to 2 passes to strip rare doubled prefixes (e.g. "LCH VL-VLKT - LCH VLVLKT - ...")
    for _ in range(2):
        matched_here = False
        for prefix in SORTED_ORGANIZER_PREFIXES:
            escaped = re.escape(prefix)
            # 1. Explicit separator after prefix: "-", "–", "_", ":" (not followed by word char inside compound)
            m_sep = re.match(
                rf"^{escaped}\s*[-–_:]\s*(.+)$",
                text,
                flags=re.IGNORECASE,
            )
            if m_sep:
                canonical = ORGANIZER_ALIAS_MAP[prefix]
                if not organizer:
                    organizer = canonical
                text = norm_text(m_sep.group(1))
                matched_here = True
                break

            # 2. Space-only separator restricted to unmistakable organization abbreviations
            if prefix in STANDALONE_SPACE_PREFIXES:
                m_sp = re.match(
                    rf"^{escaped}\s+(.+)$",
                    text,
                    flags=re.IGNORECASE,
                )
                if m_sp:
                    canonical = ORGANIZER_ALIAS_MAP[prefix]
                    if not organizer:
                        organizer = canonical
                    text = norm_text(m_sp.group(1))
                    matched_here = True
                    break
        if not matched_here:
            break

    return organizer, text


def clean_title_and_extract_role(text: str, ma_muc: str) -> Tuple[str, str]:
    """Strip program suffixes (`ĐA`/`CTĐA`/`ĐT`/`CQ`/`Chuẩn`) and extract role (`Vai_Tro`)."""
    cleaned = norm_text(text)
    extracted_role = ""

    # Strip trailing unmatched quote spacing only when needed after suffix removal
    for _ in range(6):
        peeled = False

        # A. Check training-program suffixes at the tail
        for prog_tok in PROGRAM_SUFFIXES_EXACT:
            esc = re.escape(prog_tok)
            m_prog = re.match(
                rf"^(.*?)(?:\s*[-–_]\s*|\s+){esc}\s*$",
                cleaned,
            )
            if m_prog and m_prog.group(1).strip():
                cleaned = norm_text(m_prog.group(1))
                peeled = True
                break
        if peeled:
            continue

        # B. Check role suffixes at the tail
        for role_tok, role_label, case_sensitive in ROLE_SUFFIX_SPECS:
            esc = re.escape(role_tok)
            flags = 0 if case_sensitive else re.IGNORECASE
            # Require explicit separator (-, –, _, or whitespace) before the role token
            m_role = re.match(
                rf"^(.*?)(?:\s*[-–_]\s*|\s+){esc}\s*$",
                cleaned,
                flags=flags,
            )
            if m_role and m_role.group(1).strip():
                cleaned = norm_text(m_role.group(1))
                if not extracted_role:
                    extracted_role = role_label
                peeled = True
                break
        if peeled:
            continue

        break

    # Clean dangling trailing separators left after stripping suffixes
    cleaned = re.sub(r"[\s\-–_]+$", "", cleaned).strip()

    # Check leading role cues (when suffix didn't already supply a role)
    m_hdll = re.match(r"^(?:HĐLL|HDLL)\s+(.+)$", cleaned)
    if m_hdll:
        cleaned = norm_text(m_hdll.group(1))
        if not extracted_role:
            extracted_role = "Huy động lực lượng"
    elif not extracted_role:
        if re.match(r"^(?:CTV|Cộng tác viên)\b", cleaned, flags=re.IGNORECASE):
            extracted_role = "Cộng tác viên"
        elif re.match(r"^Tình nguyện viên\b", cleaned, flags=re.IGNORECASE):
            extracted_role = "Tình nguyện viên"

    # For explicit role-variant items without extracted suffix role, infer unambiguous defaults
    if not extracted_role and ma_muc in ("1.7", "3.14", "4.4"):
        if re.search(r"\b(?:CTV|Cộng tác viên)\b", cleaned, flags=re.IGNORECASE):
            extracted_role = "Cộng tác viên"

    return norm_text(cleaned), extracted_role


def parse_muc_sort_key(ma_muc: str) -> Tuple[int, int]:
    parts = ma_muc.split(".")
    return int(parts[0]), int(parts[1])


def process_rows(
    raw_rows: Sequence[Dict[str, str]],
    rubric_map: Dict[str, Tuple[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any]]:
    """Process raw rows into deduplicated `events_full` and `events_dataset` records."""
    unknown_mucs: List[str] = []
    parsed_records: List[Dict[str, str]] = []

    for row in raw_rows:
        ma_muc = norm_text(row["Mục"])
        if ma_muc not in rubric_map:
            unknown_mucs.append(ma_muc)
            continue

        tieu_chi, diem_toi_da = rubric_map[ma_muc]
        organizer, remainder = extract_organizer(row["HĐ"])
        clean_title, vai_tro = clean_title_and_extract_role(remainder, ma_muc)

        if not clean_title:
            clean_title = norm_text(row["HĐ"])

        parsed_records.append(
            {
                "Ma_Muc": ma_muc,
                "Organizer": organizer,
                "Ten_Hoat_Dong": clean_title,
                "Vai_Tro": vai_tro,
                "Tieu_Chi_Quy_Che": tieu_chi,
                "Diem_Toi_Da": diem_toi_da,
            }
        )

    if unknown_mucs:
        raise ValueError(f"Unknown Ma_Muc codes not found in rubric.yaml: {sorted(set(unknown_mucs))}")

    # Deduplicate for events_full.csv:
    # Group by (Ma_Muc, organizer_lower, title_lower).
    # Keep distinct non-empty roles; if both empty and non-empty roles exist for the same key,
    # discard the empty-role duplicate.
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    for rec in parsed_records:
        key = (rec["Ma_Muc"], rec["Organizer"].lower(), rec["Ten_Hoat_Dong"].lower())
        grouped.setdefault(key, []).append(rec)

    dedup_full: List[Dict[str, str]] = []
    for _, recs in grouped.items():
        non_empty_roles: Dict[str, Dict[str, str]] = {}
        empty_role_rec: Optional[Dict[str, str]] = None
        for r in recs:
            role_key = r["Vai_Tro"].lower()
            if role_key:
                non_empty_roles.setdefault(role_key, r)
            elif empty_role_rec is None:
                empty_role_rec = r
        if non_empty_roles:
            dedup_full.extend(non_empty_roles.values())
        elif empty_role_rec is not None:
            dedup_full.append(empty_role_rec)

    dedup_full.sort(
        key=lambda r: (
            *parse_muc_sort_key(r["Ma_Muc"]),
            r["Organizer"].lower(),
            r["Ten_Hoat_Dong"].lower(),
            r["Vai_Tro"].lower(),
        )
    )

    events_full_rows: List[Dict[str, str]] = []
    for idx, rec in enumerate(dedup_full, start=1):
        events_full_rows.append({"STT": str(idx), **rec})

    # Deduplicate for events_dataset.csv on (title_lower, Ma_Muc)
    seen_flat: Dict[Tuple[str, str], Dict[str, str]] = {}
    for rec in dedup_full:
        fkey = (rec["Ten_Hoat_Dong"].lower(), rec["Ma_Muc"])
        if fkey not in seen_flat:
            seen_flat[fkey] = {
                "Ten_Hoat_Dong": rec["Ten_Hoat_Dong"],
                "Ma_Muc": rec["Ma_Muc"],
            }

    events_dataset_rows = sorted(
        seen_flat.values(),
        key=lambda r: (*parse_muc_sort_key(r["Ma_Muc"]), r["Ten_Hoat_Dong"].lower()),
    )

    stats = {
        "raw_count": len(raw_rows),
        "parsed_count": len(parsed_records),
        "events_full_count": len(events_full_rows),
        "events_dataset_count": len(events_dataset_rows),
        "merged_duplicates_full": len(parsed_records) - len(events_full_rows),
        "muc_counts": Counter(r["Ma_Muc"] for r in events_full_rows),
        "organizer_counts": Counter(r["Organizer"] or "<EMPTY>" for r in events_full_rows),
        "role_counts": Counter(r["Vai_Tro"] or "<EMPTY>" for r in events_full_rows),
    }
    return events_full_rows, events_dataset_rows, stats


def write_events_full(rows: Sequence[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "STT",
        "Ma_Muc",
        "Organizer",
        "Ten_Hoat_Dong",
        "Vai_Tro",
        "Tieu_Chi_Quy_Che",
        "Diem_Toi_Da",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_events_dataset(rows: Sequence[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Ten_Hoat_Dong", "Ma_Muc"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministically process raw HCMUS activity lists into events_full.csv & events_dataset.csv."
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=None,
        help="Optional path to input .xlsx file. Saves extracted raw CSV to --save-raw.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_CSV_PATH,
        help="Input raw CSV path (used when --xlsx is not supplied).",
    )
    parser.add_argument(
        "--save-raw",
        type=Path,
        default=DEFAULT_RAW_CSV_PATH,
        help="Where to write extracted raw CSV when reading from --xlsx.",
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=DEFAULT_RUBRIC_PATH,
        help="Path to data/raw/rubric.yaml.",
    )
    parser.add_argument(
        "--out-full",
        type=Path,
        default=DEFAULT_EVENTS_FULL_PATH,
        help="Output path for events_full.csv.",
    )
    parser.add_argument(
        "--out-dataset",
        type=Path,
        default=DEFAULT_EVENTS_DATASET_PATH,
        help="Output path for events_dataset.csv.",
    )
    args = parser.parse_args()

    if args.xlsx is not None:
        raw_rows = read_xlsx_rows(args.xlsx)
        write_raw_csv(raw_rows, args.save_raw)
        print(f"Extracted {len(raw_rows)} raw rows from {args.xlsx} -> {args.save_raw}")
    else:
        raw_rows = read_raw_csv_rows(args.input)
        print(f"Loaded {len(raw_rows)} raw rows from {args.input}")

    rubric_map = load_rubric_metadata(args.rubric)
    full_rows, dataset_rows, stats = process_rows(raw_rows, rubric_map)

    write_events_full(full_rows, args.out_full)
    write_events_dataset(dataset_rows, args.out_dataset)

    print(f"Wrote {stats['events_full_count']} rows -> {args.out_full}")
    print(f"Wrote {stats['events_dataset_count']} rows -> {args.out_dataset}")
    print(f"Duplicates merged (events_full): {stats['merged_duplicates_full']}")

    # Audit residual program suffixes
    residual_prog = [
        r["Ten_Hoat_Dong"]
        for r in full_rows
        if re.search(r"(?:[-–_\s])(?:CTĐA|CTDA|ĐA|ĐT|CQ|Chuẩn)\s*$", r["Ten_Hoat_Dong"])
    ]
    print(f"Residual program-suffix matches: {len(residual_prog)}")
    if residual_prog:
        print("  Samples:", residual_prog[:10])

    print("\nVai_Tro distribution:")
    for k, v in stats["role_counts"].most_common():
        print(f"  {k:22s}: {v}")

    print("\nTop 20 Organizer values:")
    for k, v in stats["organizer_counts"].most_common(20):
        print(f"  {k:20s}: {v}")

    print(f"\nDistinct Ma_Muc codes ({len(stats['muc_counts'])}):")
    for muc, cnt in sorted(stats["muc_counts"].items(), key=lambda kv: parse_muc_sort_key(kv[0])):
        print(f"  {muc:>4s}: {cnt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
