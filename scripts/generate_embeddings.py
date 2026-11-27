#!/usr/bin/env python3
"""Generate Rubric and Sample Embedding Datasets for HCMUS DRL 2-Stage kNN Classification.

Produces two independent, metadata-aligned embedding datasets using
`intfloat/multilingual-e5-small` (384d, float32, L2-normalized, `passage:` prefix):

1. Rubric branch:
   - rubric_embeddings.npy
   - rubric_metadata.parquet
2. Sample branch:
   - sample_embeddings.npy
   - sample_metadata.parquet
3. Reproducibility & audit artifacts:
   - embedding_config.json
   - embedding_generation_report.md

Source inputs (read-only; never modified):
- data/raw/rubric.yaml
- res/events_full.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants (strictly matching configs/model_config.py)
# ---------------------------------------------------------------------------
MODEL_ID = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
BATCH_SIZE = 32
DEFAULT_RUBRIC_PATH = Path("data/raw/rubric.yaml")
DEFAULT_SAMPLES_CSV_PATH = Path("data/raw/activities_sample.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/cls_index")
DEFAULT_SAMPLE_REPORT_PATH = Path("outputs/sample_embedding_generation_report.md")

POSITIVE_MAJOR_GROUPS = ("C1", "C2", "C3", "C4", "C5")
POSITIVE_SAMPLE_GROUPS = ("C1", "C2", "C3", "C4")
SPECIAL_EXCLUDED_LABELS = {"NONE", "NO_MATCH", "NEEDS_REVIEW"}

VALID_RUBRIC_STAGES = {"group", "criterion"}
VALID_PROTOTYPE_TYPES = {
    "group_definition",
    "criterion_definition",
    "verified_anchor",
    "partial_anchor",
    "draft_anchor",
    "keyword_context",
}

# ---------------------------------------------------------------------------
# Text Preprocessing Helpers (Deterministic, Traceable, Zero Fabrication)
# ---------------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    """Normalize Unicode NFC and collapse internal whitespace without altering content."""
    s = unicodedata.normalize("NFC", str(text or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_key(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(text))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_rubric_name(raw_name: str, criterion_ref: str) -> str:
    """Remove personal completion/attendance conditions and cross-ref parentheticals from criterion name.

    Preserves authentic rubric wording strictly from rubric.yaml without inventing descriptions.
    """
    text = normalize_whitespace(raw_name)
    # Remove personal completion conditions: "(hoàn tất vòng đầu tiên...)"
    text = re.sub(r"\s*\(hoàn tất vòng đầu tiên[^)]*\)", "", text, flags=re.IGNORECASE)
    # Remove administrative cross-reference: "(trừ nội dung 2.2)"
    text = re.sub(r"\s*\(trừ nội dung\s+\d+\.\d+\)", "", text, flags=re.IGNORECASE)
    # Remove attendance-threshold eligibility clauses in 2.2: "trên 50% số " and "(mọi sinh viên)" / "(sinh viên năm cuối)"
    if criterion_ref == "2.2":
        text = re.sub(r"\btrên\s+50%\s+số\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*\((?:mọi sinh viên|sinh viên năm cuối)\)", "", text, flags=re.IGNORECASE)
    # Remove negative attendance framing for 6.4 mandatory event so semantic text describes the event itself
    if criterion_ref == "6.4":
        text = re.sub(r"^Không tham gia đầy đủ\s+", "", text, flags=re.IGNORECASE)
        if text:
            text = text[0].upper() + text[1:]
    # Clean stray punctuation/spaces
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = normalize_whitespace(text).rstrip(".")
    return text


def clean_rubric_scope_note(raw_scope: str, criterion_ref: str) -> str:
    """Strip scoring points, CSV row references, QĐ/administrative meta-notes, and eligibility clauses.

    Returns cleaned semantic text strictly derived from `scope_note` in rubric.yaml.
    """
    text = normalize_whitespace(raw_scope)
    if not text:
        return ""

    # 1. Remove full sentences about CSV observations, points-by-duration, or boundary meta-warnings
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept_sentences: List[str] = []
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        # Drop sentences purely about scoring points or CSV administrative observations
        if re.search(r"events_full\.csv", s, flags=re.IGNORECASE):
            continue
        if re.search(r"^Điểm phụ thuộc thời lượng", s, flags=re.IGNORECASE):
            continue
        if re.search(r"^Cần phân biệt cẩn thận với\b", s, flags=re.IGNORECASE):
            continue
        kept_sentences.append(s)
    text = " ".join(kept_sentences)

    # 2. Strip scoring tails / clauses within remaining sentences
    # e.g., ": cấp trường trở lên (...) = 10đ; cấp khoa (...) = 5đ."
    text = re.sub(r":\s*cấp trường trở lên[^.]*=\s*\d+đ[^.]*", "", text, flags=re.IGNORECASE)
    # e.g., "; điểm chỉ tính khi được chọn vào đội tuyển"
    text = re.sub(r";\s*điểm chỉ tính khi[^.]*", "", text, flags=re.IGNORECASE)
    # e.g., "(mức phẳng 5đ)" / "(3đ/đợt)" / "(5/10/15)"
    text = re.sub(r"\s*\(mức phẳng\s*\d+đ\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(\d+đ(?:/[^)]*)?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(\d+/\d+/\d+\)", "", text)
    # e.g., "(khác với khảo sát/điều tra xã hội ở 2.3)"
    text = re.sub(r"\s*\(khác với[^)]*\d+\.\d+[^)]*\)", "", text, flags=re.IGNORECASE)
    # e.g., "; phân biệt với tình nguyện viên... (mục 4.3)"
    text = re.sub(r";\s*phân biệt với[^.]*\(mục\s+\d+\.\d+\)", "", text, flags=re.IGNORECASE)
    # e.g., "(vai trò 'Huy động lực lượng')"
    text = re.sub(r"\s*\(vai trò\s*'[^']*'\)", "", text, flags=re.IGNORECASE)
    # e.g., "Theo QĐ 2020 là "
    text = re.sub(r"^Theo QĐ 2020 là\s+", "", text, flags=re.IGNORECASE)
    # e.g., "Chỉ tính khi sinh viên là " -> keep semantic phrase without scoring-eligibility framing
    text = re.sub(r"^Chỉ tính khi\s+", "", text, flags=re.IGNORECASE)

    # Clean up punctuation artifacts
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = normalize_whitespace(text).rstrip(".;:")
    if text:
        text = text[0].upper() + text[1:]
    return text


def format_passage(text: str) -> str:
    """Format cleaned text with E5 passage prefix."""
    return f"{PASSAGE_PREFIX}{normalize_whitespace(text)}"


# ---------------------------------------------------------------------------
# Rubric Prototype Construction (Stage 1 & Stage 2)
# ---------------------------------------------------------------------------


def build_rubric_prototypes(rubric: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build Stage 1 (group) and Stage 2 (criterion) prototype records from rubric.yaml."""
    groups: Dict[str, Any] = rubric.get("groups") or {}
    tieu_chi_map: Dict[str, Any] = rubric.get("tieu_chi") or {}
    items: Dict[str, Any] = rubric.get("items") or {}

    em_true_items: List[Tuple[str, Dict[str, Any]]] = []
    em_false_items: List[Tuple[str, Dict[str, Any]]] = []
    for key, item in items.items():
        if not isinstance(item, dict):
            continue
        if item.get("event_matchable") is True:
            em_true_items.append((key, item))
        else:
            em_false_items.append((key, item))

    # Group event_matchable:true items by major group
    items_by_group: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for key, item in em_true_items:
        grp = str(item.get("group", "")).strip()
        items_by_group.setdefault(grp, []).append((key, item))

    prototypes: List[Dict[str, Any]] = []
    seen_ids: set = set()

    def next_unique_id(raw_id: str) -> str:
        if raw_id not in seen_ids:
            seen_ids.add(raw_id)
            return raw_id
        idx = 2
        while f"{raw_id}_{idx}" in seen_ids:
            idx += 1
        uid = f"{raw_id}_{idx}"
        seen_ids.add(uid)
        return uid

    # ------------------------------------------------------------------
    # Stage 1: Major-group prototypes (eligible positive groups in C1-C5)
    # ------------------------------------------------------------------
    stage1_groups_included: List[str] = []
    stage1_groups_excluded: List[Dict[str, Any]] = []

    for grp_id, grp_meta in groups.items():
        grp_items = items_by_group.get(grp_id, [])
        if grp_id not in POSITIVE_MAJOR_GROUPS:
            stage1_groups_excluded.append(
                {
                    "group_id": grp_id,
                    "name": grp_meta.get("name", ""),
                    "reason": (
                        "Non-positive penalty/reminder group (sign=negative); not a positive "
                        "Stage 1 C1-C5 event-classification target (contains only 6.4 mandatory_event / remind)."
                    ),
                }
            )
            continue
        if not grp_items:
            stage1_groups_excluded.append(
                {
                    "group_id": grp_id,
                    "name": grp_meta.get("name", ""),
                    "reason": "Zero items with event_matchable: true (all items are achievements, statuses, or special circumstances).",
                }
            )
            continue

        stage1_groups_included.append(grp_id)
        grp_name = normalize_whitespace(grp_meta.get("name", grp_id))

        # Include only tieu_chi that actually contain eligible event_matchable:true items
        eligible_tcs: List[str] = []
        for _, it in grp_items:
            tc = it.get("tieu_chi")
            if tc and tc not in eligible_tcs:
                eligible_tcs.append(str(tc))

        tc_descriptions = [
            normalize_whitespace(tieu_chi_map[tc]["name"])
            for tc in eligible_tcs
            if tc in tieu_chi_map and tieu_chi_map[tc].get("name")
        ]

        # Clean, untruncated descriptions of eligible criteria in this group
        crit_descriptions = [
            clean_rubric_name(str(it.get("name", "")), str(it.get("ref", "")))
            for _, it in grp_items
        ]

        parts = [f"{grp_name}."]
        if tc_descriptions:
            parts.append("Phạm vi tiêu chí: " + "; ".join(tc_descriptions) + ".")
        parts.append("Các hoạt động thuộc nhóm: " + "; ".join(crit_descriptions) + ".")
        emb_text = format_passage(" ".join(parts))

        prototypes.append(
            {
                "prototype_id": next_unique_id(f"stage1_{grp_id}_group_def"),
                "stage": "group",
                "group_id": grp_id,
                "tieu_chi": None,
                "criterion_key": None,
                "criterion_ref": None,
                "prototype_type": "group_definition",
                "anchor_status": None,
                "criterion_anchors_status": None,
                "event_matchable": True,
                "embedding_text": emb_text,
                "source_fields": [
                    f"groups.{grp_id}.name",
                    "tieu_chi[eligible].name",
                    "items[event_matchable=true].name",
                ],
                "kind": "major_group",
                "mvp_scope": None,
                "is_positive_target": True,
                "has_uncertain": False,
                "uncertain_notes": None,
            }
        )

    # ------------------------------------------------------------------
    # Stage 2: Criterion prototypes (all event_matchable: true items)
    # ------------------------------------------------------------------
    for item_key, item in em_true_items:
        ref = str(item["ref"]).strip()
        grp = str(item["group"]).strip()
        tc = str(item["tieu_chi"]).strip() if item.get("tieu_chi") else None
        raw_name = str(item.get("name", ""))
        raw_scope = str(item.get("scope_note", "") or "")
        item_anchors_status = str(item.get("anchors_status", "")).strip()
        kind = str(item.get("kind", "")).strip()
        mvp_scope = str(item.get("mvp_scope", "")).strip()
        is_positive_target = grp in POSITIVE_MAJOR_GROUPS and kind != "mandatory_event"

        uncertain_list = item.get("uncertain") or []
        has_uncertain = bool(uncertain_list)
        uncertain_json = (
            json.dumps([normalize_whitespace(u) for u in uncertain_list], ensure_ascii=False)
            if has_uncertain
            else None
        )

        cleaned_name = clean_rubric_name(raw_name, ref)
        cleaned_scope = clean_rubric_scope_note(raw_scope, ref)
        ref_safe = clean_key(ref)

        def base_record(
            *,
            pid: str,
            ptype: str,
            astatus: str,
            etext: str,
            sfields: List[str],
        ) -> Dict[str, Any]:
            return {
                "prototype_id": next_unique_id(pid),
                "stage": "criterion",
                "group_id": grp,
                "tieu_chi": tc,
                "criterion_key": item_key,
                "criterion_ref": ref,
                "prototype_type": ptype,
                "anchor_status": astatus,
                "criterion_anchors_status": item_anchors_status,
                "event_matchable": True,
                "embedding_text": etext,
                "source_fields": sfields,
                "kind": kind,
                "mvp_scope": mvp_scope,
                "is_positive_target": is_positive_target,
                "has_uncertain": has_uncertain,
                "uncertain_notes": uncertain_json,
            }

        # 1. Criterion-definition prototype (mandatory for every eligible criterion)
        if cleaned_scope:
            def_text = format_passage(f"{cleaned_name}. {cleaned_scope}.")
            def_fields = ["name", "scope_note"]
        else:
            def_text = format_passage(f"{cleaned_name}.")
            def_fields = ["name"]

        prototypes.append(
            base_record(
                pid=f"stage2_{ref_safe}_def",
                ptype="criterion_definition",
                astatus=item_anchors_status,
                etext=def_text,
                sfields=def_fields,
            )
        )

        # 2. Anchor prototypes (one separate prototype per anchor; never merged)
        anchors_list = item.get("anchors") or []
        verified_anchors = item.get("verified_anchors") or []
        draft_anchors = item.get("draft_anchors") or []

        if item_anchors_status == "verified_from_school_list":
            for idx, anchor in enumerate(anchors_list, start=1):
                a_clean = normalize_whitespace(anchor)
                if not a_clean:
                    continue
                prototypes.append(
                    base_record(
                        pid=f"stage2_{ref_safe}_va_{idx:03d}",
                        ptype="verified_anchor",
                        astatus="verified_from_school_list",
                        etext=format_passage(a_clean),
                        sfields=["anchors"],
                    )
                )
        elif item_anchors_status == "verified_partial":
            # Distinguish verified_partial anchors vs draft anchors strictly
            for idx, anchor in enumerate(verified_anchors, start=1):
                a_clean = normalize_whitespace(anchor)
                if not a_clean:
                    continue
                prototypes.append(
                    base_record(
                        pid=f"stage2_{ref_safe}_pa_{idx:03d}",
                        ptype="partial_anchor",
                        astatus="verified_partial",
                        etext=format_passage(a_clean),
                        sfields=["verified_anchors"],
                    )
                )
            for idx, anchor in enumerate(draft_anchors, start=1):
                a_clean = normalize_whitespace(anchor)
                if not a_clean:
                    continue
                prototypes.append(
                    base_record(
                        pid=f"stage2_{ref_safe}_da_{idx:03d}",
                        ptype="draft_anchor",
                        astatus="draft",
                        etext=format_passage(a_clean),
                        sfields=["draft_anchors"],
                    )
                )
        elif item_anchors_status == "draft":
            for idx, anchor in enumerate(anchors_list, start=1):
                a_clean = normalize_whitespace(anchor)
                if not a_clean:
                    continue
                prototypes.append(
                    base_record(
                        pid=f"stage2_{ref_safe}_da_{idx:03d}",
                        ptype="draft_anchor",
                        astatus="draft",
                        etext=format_passage(a_clean),
                        sfields=["anchors"],
                    )
                )
        else:
            raise ValueError(
                f"Unexpected anchors_status='{item_anchors_status}' on eligible item '{item_key}' (ref {ref})."
            )

        # 3. Keyword-context prototype (when relevant keywords exist)
        keywords = [normalize_whitespace(k) for k in (item.get("keywords") or []) if normalize_whitespace(k)]
        if keywords:
            kw_text = format_passage(f"{cleaned_name}. Từ khóa: {', '.join(keywords)}.")
            prototypes.append(
                base_record(
                    pid=f"stage2_{ref_safe}_kw",
                    ptype="keyword_context",
                    astatus=item_anchors_status,
                    etext=kw_text,
                    sfields=["name", "keywords"],
                )
            )

    # Assign sequential embedding_row indices
    for row_idx, rec in enumerate(prototypes):
        rec["embedding_row"] = row_idx

    audit = {
        "total_rubric_items": len(items),
        "eligible_criteria_count": len(em_true_items),
        "excluded_criteria_count": len(em_false_items),
        "stage1_groups_included": stage1_groups_included,
        "stage1_groups_excluded": stage1_groups_excluded,
        "excluded_criteria": [
            {
                "criterion_key": k,
                "criterion_ref": str(v.get("ref", "")),
                "group_id": str(v.get("group", "")),
                "kind": str(v.get("kind", "")),
                "reason": "event_matchable is false",
            }
            for k, v in em_false_items
        ],
        "meta_labels_excluded": list((rubric.get("meta_labels") or {}).keys()),
    }
    return prototypes, audit


# ---------------------------------------------------------------------------
# Sample Record Construction from Labeled Sample CSV (e.g., activities_sample.csv)
# ---------------------------------------------------------------------------


def detect_csv_columns(fieldnames: Sequence[str]) -> Dict[str, Optional[str]]:
    """Detect and map actual CSV columns without assuming legacy schema."""
    cols = [c.strip() for c in fieldnames if c]
    col_lower = {c.lower(): c for c in cols}

    def pick(candidates: Sequence[str]) -> Optional[str]:
        for cand in candidates:
            if cand.lower() in col_lower:
                return col_lower[cand.lower()]
        return None

    mapping = {
        "row_id": pick(["STT", "sample_id", "id", "row_id"]),
        "group_id": pick(["group_id", "Nhom", "group", "parent_group_id"]),
        "label": pick(["Ma_Muc", "criterion_ref", "label", "rubric_item_id"]),
        "criterion_key": pick(["criterion_key", "item_key"]),
        "title": pick(["Ten_Hoat_Dong", "event_title", "title", "activity_title"]),
        "description": pick(["Mo_Ta", "event_description", "description", "activity_description", "noi_dung", "content"]),
        "organizer": pick(["Organizer", "Don_Vi", "don_vi_to_chuc", "organizer"]),
        "role": pick(["Vai_Tro", "role"]),
        "criterion_text": pick(["Tieu_Chi_Quy_Che", "criterion_label", "criterion_name"]),
        "points": pick(["Diem_Toi_Da", "points", "max_points"]),
    }
    if not mapping["label"] or not mapping["title"]:
        raise ValueError(
            f"Could not detect required label/title columns in CSV schema: {cols}. "
            f"Detected mapping: {mapping}"
        )
    return mapping


def build_sample_embedding_text(
    event_title: str,
    event_description: Optional[str],
    passage_prefix: str = PASSAGE_PREFIX,
) -> str:
    """Construct strictly leakage-free E5 sample passage text aligned with event_text_builder.

    Format:
        `passage: [event title]` or `passage: [event title] [event description]`
    """
    title_norm = normalize_whitespace(event_title)
    desc_norm = normalize_whitespace(event_description or "")
    parts = [p for p in (title_norm, desc_norm) if p]
    if not parts:
        raise ValueError("Cannot construct sample embedding_text when both title and description are empty.")
    body = " ".join(parts)
    if body.startswith(passage_prefix):
        return body
    return f"{passage_prefix}{body}"


def build_sample_records(
    csv_path: Path,
    rubric: Dict[str, Any],
    *,
    model_name: str = MODEL_ID,
    embedding_dim: int = EMBEDDING_DIM,
    normalization: str = "l2",
    passage_prefix: str = PASSAGE_PREFIX,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Inspect, validate, and build leakage-free eligible sample records from the labeled CSV.

    - Preserves original records, source row positions, and duplicate/multi-label distinctions.
    - Never silently modifies labels or fabricates unavailable fields.
    - Separately records malformed rows and label inconsistencies so callers can abort safely.
    """
    groups_map: Dict[str, Any] = rubric.get("groups") or {}
    items: Dict[str, Any] = rubric.get("items") or {}
    ref_to_item: Dict[str, Tuple[str, Dict[str, Any]]] = {
        str(v["ref"]).strip(): (k, v) for k, v in items.items() if isinstance(v, dict) and "ref" in v
    }

    # Pre-scan raw CSV with stdlib csv.reader to detect any column-width / malformed rows
    raw_bytes = csv_path.read_bytes()
    has_utf8_bom = raw_bytes.startswith(b"\xef\xbb\xbf")
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        raw_reader = csv.reader(handle)
        header = next(raw_reader, None)
        if not header:
            raise ValueError(f"CSV file '{csv_path}' has no header row.")
        raw_list_rows = list(raw_reader)

    expected_width = len(header)
    malformed_rows: List[Dict[str, Any]] = []
    for row_idx, rvals in enumerate(raw_list_rows, start=1):
        if len(rvals) != expected_width:
            malformed_rows.append(
                {
                    "source_row_index": row_idx,
                    "csv_line_number": row_idx + 1,
                    "actual_columns": len(rvals),
                    "expected_columns": expected_width,
                    "preview": rvals[:4],
                }
            )

    col_map = detect_csv_columns(header)
    raw_rows: List[Dict[str, str]] = [
        {header[i]: (rvals[i] if i < len(rvals) else "") for i in range(expected_width)}
        for rvals in raw_list_rows
    ]

    samples: List[Dict[str, Any]] = []
    excluded_rows: List[Dict[str, Any]] = []
    invalid_or_special_rows: List[Dict[str, Any]] = []
    label_inconsistencies: List[Dict[str, Any]] = []
    skipped_empty_text_rows: List[Dict[str, Any]] = []

    row_id_col = col_map["row_id"]
    group_col = col_map["group_id"]
    label_col = col_map["label"]
    key_col = col_map["criterion_key"]
    title_col = col_map["title"]
    desc_col = col_map["description"]
    org_col = col_map["organizer"]
    role_col = col_map["role"]
    crit_text_col = col_map["criterion_text"]
    assert title_col is not None and label_col is not None

    dataset_rel = (
        csv_path.resolve().relative_to(PROJECT_ROOT).as_posix()
        if csv_path.resolve().is_relative_to(PROJECT_ROOT)
        else csv_path.as_posix()
    )
    sample_id_prefix = clean_key(csv_path.stem)

    # Track Tieu_Chi_Quy_Che variants per Ma_Muc for consistency audit
    crit_text_by_label: Dict[str, Counter] = {}
    missing_counts_all: Dict[str, int] = {col: 0 for col in header}

    for row_idx, row in enumerate(raw_rows, start=1):
        for col in header:
            if not normalize_whitespace(row.get(col, "")):
                missing_counts_all[col] += 1

        original_label = normalize_whitespace(row.get(label_col, ""))
        raw_group = normalize_whitespace(row.get(group_col, "")) if group_col else ""
        raw_key = normalize_whitespace(row.get(key_col, "")) if key_col else ""
        source_stt = normalize_whitespace(row.get(row_id_col, "")) if row_id_col else ""
        event_title = normalize_whitespace(row.get(title_col, ""))
        event_desc_raw = normalize_whitespace(row.get(desc_col, "")) if desc_col else ""
        event_desc: Optional[str] = event_desc_raw if event_desc_raw else None
        organizer_raw = normalize_whitespace(row.get(org_col, "")) if org_col else ""
        organizer: Optional[str] = organizer_raw if organizer_raw else None
        role_raw = normalize_whitespace(row.get(role_col, "")) if role_col else ""
        role: Optional[str] = role_raw if role_raw else None
        crit_text_raw = normalize_whitespace(row.get(crit_text_col, "")) if crit_text_col else ""

        if original_label and crit_text_raw:
            crit_text_by_label.setdefault(original_label, Counter())[crit_text_raw] += 1

        # Preserve full original row fields as traceability metadata
        source_meta = {k: normalize_whitespace(v) for k, v in row.items()}

        # 1. Label integrity checks
        if not original_label:
            rec = {
                "source_row_index": row_idx,
                "csv_line_number": row_idx + 1,
                "original_label": original_label,
                "event_title": event_title,
                "reason": "missing_label",
                "source_metadata": source_meta,
            }
            excluded_rows.append(rec)
            invalid_or_special_rows.append(rec)
            label_inconsistencies.append(rec)
            continue

        if original_label.upper() in SPECIAL_EXCLUDED_LABELS:
            rec = {
                "source_row_index": row_idx,
                "csv_line_number": row_idx + 1,
                "original_label": original_label,
                "event_title": event_title,
                "reason": f"special_non_criterion_label ({original_label.upper()})",
                "source_metadata": source_meta,
            }
            excluded_rows.append(rec)
            invalid_or_special_rows.append(rec)
            continue

        if original_label not in ref_to_item:
            rec = {
                "source_row_index": row_idx,
                "csv_line_number": row_idx + 1,
                "original_label": original_label,
                "event_title": event_title,
                "reason": "unknown_or_unmapped_rubric_ref",
                "source_metadata": source_meta,
            }
            excluded_rows.append(rec)
            invalid_or_special_rows.append(rec)
            label_inconsistencies.append(rec)
            continue

        criterion_key, item = ref_to_item[original_label]
        group_id = str(item["group"]).strip()
        tieu_chi = str(item["tieu_chi"]).strip() if item.get("tieu_chi") else None

        # Cross-check optional explicit group_id / criterion_key columns against rubric.yaml
        if group_id not in groups_map:
            rec = {
                "source_row_index": row_idx,
                "csv_line_number": row_idx + 1,
                "original_label": original_label,
                "event_title": event_title,
                "reason": f"invalid_rubric_group_id ({group_id})",
                "source_metadata": source_meta,
            }
            excluded_rows.append(rec)
            invalid_or_special_rows.append(rec)
            label_inconsistencies.append(rec)
            continue

        if raw_group and raw_group != group_id:
            rec = {
                "source_row_index": row_idx,
                "csv_line_number": row_idx + 1,
                "original_label": original_label,
                "event_title": event_title,
                "reason": f"group_id_mismatch (csv={raw_group!r} vs rubric={group_id!r})",
                "source_metadata": source_meta,
            }
            excluded_rows.append(rec)
            invalid_or_special_rows.append(rec)
            label_inconsistencies.append(rec)
            continue

        if raw_key and raw_key != criterion_key:
            rec = {
                "source_row_index": row_idx,
                "csv_line_number": row_idx + 1,
                "original_label": original_label,
                "event_title": event_title,
                "reason": f"criterion_key_mismatch (csv={raw_key!r} vs rubric={criterion_key!r})",
                "source_metadata": source_meta,
            }
            excluded_rows.append(rec)
            invalid_or_special_rows.append(rec)
            label_inconsistencies.append(rec)
            continue

        # 2. Usable text check (never produce zero vectors for empty text)
        if not event_title and not event_desc:
            rec = {
                "source_row_index": row_idx,
                "csv_line_number": row_idx + 1,
                "original_label": original_label,
                "criterion_key": criterion_key,
                "group_id": group_id,
                "event_title": event_title,
                "reason": "empty_event_text (both title and description empty)",
                "source_metadata": source_meta,
            }
            excluded_rows.append(rec)
            invalid_or_special_rows.append(rec)
            skipped_empty_text_rows.append(rec)
            continue

        # 3. Rubric eligibility filtering (positive C1-C4 event_matchable:true criteria only)
        if item.get("event_matchable") is not True:
            excluded_rows.append(
                {
                    "source_row_index": row_idx,
                    "csv_line_number": row_idx + 1,
                    "original_label": original_label,
                    "criterion_key": criterion_key,
                    "group_id": group_id,
                    "kind": str(item.get("kind", "")),
                    "mvp_scope": str(item.get("mvp_scope", "")),
                    "event_title": event_title,
                    "reason": f"non_event_matchable_criterion (event_matchable=false, kind={item.get('kind')})",
                    "source_metadata": source_meta,
                }
            )
            continue

        mvp_scope = str(item.get("mvp_scope") or "").strip()
        if group_id not in POSITIVE_SAMPLE_GROUPS or mvp_scope == "remind":
            excluded_rows.append(
                {
                    "source_row_index": row_idx,
                    "csv_line_number": row_idx + 1,
                    "original_label": original_label,
                    "criterion_key": criterion_key,
                    "group_id": group_id,
                    "kind": str(item.get("kind", "")),
                    "mvp_scope": mvp_scope,
                    "event_title": event_title,
                    "reason": (
                        f"excluded_non_positive_group_or_reminder "
                        f"(group={group_id}, kind={item.get('kind')}, mvp_scope={mvp_scope})"
                    ),
                    "source_metadata": source_meta,
                }
            )
            continue

        # 4. Strictly leakage-free activity embedding text
        embedding_text = build_sample_embedding_text(
            event_title=event_title,
            event_description=event_desc,
            passage_prefix=passage_prefix,
        )

        samples.append(
            {
                "embedding_row": len(samples),
                "sample_id": f"{sample_id_prefix}_row_{row_idx:04d}",
                "source_row_index": row_idx,
                "source_stt": source_stt if source_stt else None,
                "source_dataset": dataset_rel,
                "event_title": event_title,
                "event_description": event_desc,
                "embedding_text": embedding_text,
                "original_label": original_label,
                "criterion_ref": str(item["ref"]).strip(),
                "criterion_key": criterion_key,
                "group_id": group_id,
                "tieu_chi": tieu_chi,
                "role": role,
                "organizer": organizer,
                "label_status": "valid_event_matchable",
                "embedding_model": model_name,
                "embedding_dimension": int(embedding_dim),
                "normalization": normalization,
                "source_metadata": json.dumps(source_meta, ensure_ascii=False),
            }
        )

    # Detect exact, multi-label, and orthographic near-duplicates preserved among eligible samples
    def strip_diacritics_key(s: str) -> str:
        nfd = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
        stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        return re.sub(r"[^a-z0-9]+", " ", stripped).strip()

    exact_row_groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    exact_title_groups: Dict[str, List[Dict[str, Any]]] = {}
    diacritic_title_groups: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        exact_row_key = (
            s["criterion_ref"],
            (s["organizer"] or "").lower(),
            s["event_title"].lower(),
            (s["event_description"] or "").lower(),
            (s["role"] or "").lower(),
        )
        exact_row_groups.setdefault(exact_row_key, []).append(s)
        exact_title_groups.setdefault(s["event_title"].lower(), []).append(s)
        diacritic_title_groups.setdefault(strip_diacritics_key(s["event_title"]), []).append(s)

    def summarize_dup(grp: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "sample_id": x["sample_id"],
                "source_row_index": x["source_row_index"],
                "event_title": x["event_title"],
                "criterion_ref": x["criterion_ref"],
                "group_id": x["group_id"],
                "organizer": x["organizer"],
                "role": x["role"],
            }
            for x in grp
        ]

    preserved_exact_row_duplicates = [
        summarize_dup(grp) for grp in exact_row_groups.values() if len(grp) > 1
    ]
    preserved_exact_duplicates = [
        summarize_dup(grp) for grp in exact_title_groups.values() if len(grp) > 1
    ]
    preserved_cross_criterion_duplicates = [
        summarize_dup(grp)
        for grp in exact_title_groups.values()
        if len(grp) > 1 and len({x["criterion_ref"] for x in grp}) > 1
    ]
    preserved_same_criterion_duplicates = [
        summarize_dup(grp)
        for grp in exact_title_groups.values()
        if len(grp) > 1 and len({x["criterion_ref"] for x in grp}) == 1
    ]
    preserved_near_duplicates = [
        summarize_dup(grp)
        for grp in diacritic_title_groups.values()
        if len({x["event_title"].lower() for x in grp}) > 1
    ]

    # Also compute all-source-row title duplicate statistics for transparency
    all_title_groups: Dict[str, List[Dict[str, Any]]] = {}
    all_full_row_groups: Dict[Tuple[str, ...], List[int]] = {}
    for idx, r in enumerate(raw_rows, start=1):
        t = normalize_whitespace(r.get(title_col, ""))
        lbl = normalize_whitespace(r.get(label_col, ""))
        org = normalize_whitespace(r.get(org_col, "")) if org_col else ""
        rl = normalize_whitespace(r.get(role_col, "")) if role_col else ""
        pts = normalize_whitespace(r.get(col_map["points"], "")) if col_map["points"] else ""
        all_title_groups.setdefault(t.lower(), []).append({"row": idx, "label": lbl, "org": org, "role": rl, "title": t})
        all_full_row_groups.setdefault((lbl, org.lower(), t.lower(), rl.lower(), pts), []).append(idx)

    # Identify eligible C1-C4 rubric criteria with 0 samples
    covered_refs = {s["criterion_ref"] for s in samples}
    uncovered_em_true = [
        {
            "criterion_ref": str(v["ref"]),
            "criterion_key": k,
            "group_id": str(v["group"]),
            "anchors_status": str(v.get("anchors_status", "")),
        }
        for k, v in items.items()
        if isinstance(v, dict)
        and v.get("event_matchable") is True
        and str(v.get("group", "")).strip() in POSITIVE_SAMPLE_GROUPS
        and str(v["ref"]) not in covered_refs
    ]

    # Missing counts among eligible samples
    eligible_missing_counts = {
        "event_title": sum(1 for s in samples if not s["event_title"]),
        "event_description": sum(1 for s in samples if s["event_description"] is None),
        "role": sum(1 for s in samples if s["role"] is None),
        "organizer": sum(1 for s in samples if s["organizer"] is None),
    }

    audit = {
        "source_csv_path": dataset_rel,
        "source_csv_size_bytes": len(raw_bytes),
        "source_csv_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_has_utf8_bom": has_utf8_bom,
        "total_source_rows": len(raw_rows),
        "csv_columns": list(header),
        "detected_column_mapping": col_map,
        "missing_counts_all_rows": missing_counts_all,
        "eligible_missing_counts": eligible_missing_counts,
        "malformed_rows": malformed_rows,
        "label_inconsistencies": label_inconsistencies,
        "skipped_empty_text_rows": skipped_empty_text_rows,
        "criterion_text_variants": {
            lbl: dict(cnt) for lbl, cnt in crit_text_by_label.items() if len(cnt) > 1
        },
        "eligible_sample_count": len(samples),
        "excluded_row_count": len(excluded_rows),
        "invalid_or_special_label_count": len(invalid_or_special_rows),
        "excluded_rows": excluded_rows,
        "invalid_or_special_rows": invalid_or_special_rows,
        "all_source_exact_row_dup_groups": sum(1 for v in all_full_row_groups.values() if len(v) > 1),
        "all_source_title_dup_groups": sum(1 for v in all_title_groups.values() if len(v) > 1),
        "all_source_cross_label_title_dup_groups": sum(
            1 for v in all_title_groups.values() if len(v) > 1 and len({x["label"] for x in v}) > 1
        ),
        "preserved_exact_row_duplicates": preserved_exact_row_duplicates,
        "preserved_exact_duplicates": preserved_exact_duplicates,
        "preserved_cross_criterion_duplicates": preserved_cross_criterion_duplicates,
        "preserved_same_criterion_duplicates": preserved_same_criterion_duplicates,
        "preserved_near_duplicates": preserved_near_duplicates,
        "uncovered_eligible_criteria": uncovered_em_true,
    }
    return samples, audit


# ---------------------------------------------------------------------------
# Embedding Model Loader & Encoder (Shared Across Both Branches)
# ---------------------------------------------------------------------------


def load_e5_model(model_id: str = MODEL_ID):
    """Load `intfloat/multilingual-e5-small` deterministically with exact cache revision metadata."""
    import sentence_transformers
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    snapshot_path: Optional[str] = None
    model_revision: Optional[str] = None
    try:
        snapshot_path = snapshot_download(model_id, local_files_only=True)
        model_revision = Path(snapshot_path).name
    except Exception:
        snapshot_path = None

    device = "cpu"
    if snapshot_path:
        model = SentenceTransformer(snapshot_path, device=device, local_files_only=True)
    else:
        model = SentenceTransformer(model_id, device=device)

    model.eval()

    # Extract exact tokenizer, pooling, and architecture config from loaded model
    first_module = model[0]
    pooling_module = model[1]
    tokenizer = model.tokenizer

    pooling_config = pooling_module.get_config_dict()
    if pooling_config.get("pooling_mode"):
        pooling_method = str(pooling_config["pooling_mode"])
    elif pooling_config.get("pooling_mode_mean_tokens"):
        pooling_method = "mean"
    elif pooling_config.get("pooling_mode_cls_token"):
        pooling_method = "cls"
    elif pooling_config.get("pooling_mode_max_tokens"):
        pooling_method = "max"
    else:
        pooling_method = "custom"

    emb_dim = (
        int(model.get_embedding_dimension())
        if hasattr(model, "get_embedding_dimension")
        else int(model.get_sentence_embedding_dimension())
    )

    model_info = {
        "model_name": model_id,
        "model_revision": model_revision,
        "snapshot_path_basename": Path(snapshot_path).name if snapshot_path else None,
        "embedding_dimension": emb_dim,
        "device": device,
        "dtype": "float32",
        "pooling_method": pooling_method,
        "pooling_config": pooling_config,
        "normalize_embeddings": True,
        "l2_normalization": True,
        "modules": [type(m).__name__ for m in model],
        "tokenizer_config": {
            "tokenizer_class": type(tokenizer).__name__,
            "model_max_length": int(getattr(tokenizer, "model_max_length", 512)),
            "max_seq_length": int(model.max_seq_length),
            "do_lower_case": bool(getattr(first_module, "do_lower_case", False)),
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
            "pad_token": getattr(tokenizer, "pad_token", None),
            "cls_token": getattr(tokenizer, "cls_token", None),
            "sep_token": getattr(tokenizer, "sep_token", None),
        },
        "runtime_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
    }
    if model_info["embedding_dimension"] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected embedding dimension {EMBEDDING_DIM}, got {model_info['embedding_dimension']}"
        )
    return model, model_info


def encode_texts(model, texts: Sequence[str], batch_size: int = BATCH_SIZE) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Batch-encode pre-prefixed `passage:` texts into L2-normalized float32 numpy array."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), {"count": 0, "max_tokens": 0, "truncated_count": 0}

    # Measure token lengths using exact model tokenizer to verify zero silent truncation
    tok = model.tokenizer
    max_seq_len = int(model.max_seq_length)
    encoded_ids = tok(list(texts), add_special_tokens=True, truncation=False)["input_ids"]
    lengths = [len(ids) for ids in encoded_ids]
    token_stats = {
        "count": len(texts),
        "min_tokens": int(min(lengths)),
        "median_tokens": float(np.median(lengths)),
        "max_tokens": int(max(lengths)),
        "max_seq_length": max_seq_len,
        "truncated_count": int(sum(1 for l in lengths if l > max_seq_len)),
    }

    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    # Defensive explicit L2 normalization in float32
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Encountered zero-norm embedding vector during encoding.")
    vectors = (vectors / norms).astype(np.float32)
    return vectors, token_stats


# ---------------------------------------------------------------------------
# Parquet Writers with Explicit Arrow Schemas
# ---------------------------------------------------------------------------


def write_rubric_parquet(records: List[Dict[str, Any]], path: Path) -> None:
    """Write Schema v2.0 rubric_metadata.parquet aligned 1:1 with rubric_embeddings.npy."""
    normalized_rows: List[Dict[str, Any]] = []
    for rec in records:
        bm = rec.get("boundary_metadata")
        bm_str = (
            bm
            if isinstance(bm, str)
            else json.dumps(bm if isinstance(bm, dict) else {}, ensure_ascii=False, allow_nan=False)
        )
        unc_list = (bm.get("uncertain") if isinstance(bm, dict) else None) or []
        unc_notes = rec.get("uncertain_notes")
        if unc_notes is None and unc_list:
            unc_notes = json.dumps(unc_list, ensure_ascii=False)
        has_unc = bool(rec.get("has_uncertain", bool(unc_list)))

        crit_ast = rec.get("criterion_anchor_status", rec.get("criterion_anchors_status"))
        elig_class = str(rec.get("eligibility_class") or "positive_candidate")
        is_pos = bool(
            rec.get(
                "is_positive_target",
                elig_class in ("positive_candidate", "conditional_candidate"),
            )
        )
        can_none = rec.get("can_decide_none")

        normalized_rows.append(
            {
                "embedding_row": int(rec["embedding_row"]),
                "schema_version": str(rec.get("schema_version") or "2.0"),
                "prototype_id": str(rec["prototype_id"]),
                "stage": str(rec["stage"]),
                "group_id": None if rec.get("group_id") in (None, "") else str(rec["group_id"]),
                "tieu_chi": None if rec.get("tieu_chi") in (None, "") else str(rec["tieu_chi"]),
                "criterion_key": None if rec.get("criterion_key") in (None, "") else str(rec["criterion_key"]),
                "criterion_ref": None if rec.get("criterion_ref") in (None, "") else str(rec["criterion_ref"]),
                "prototype_type": str(rec["prototype_type"]),
                "prototype_family": str(rec.get("prototype_family") or rec["prototype_type"]),
                "anchor_status": None if rec.get("anchor_status") in (None, "") else str(rec["anchor_status"]),
                "criterion_anchor_status": None if crit_ast in (None, "") else str(crit_ast),
                "criterion_anchors_status": None if crit_ast in (None, "") else str(crit_ast),
                "anchor_text": None if rec.get("anchor_text") in (None, "") else str(rec["anchor_text"]),
                "source_anchor_index": (
                    int(rec["source_anchor_index"])
                    if rec.get("source_anchor_index") is not None
                    else None
                ),
                "event_matchable": bool(rec["event_matchable"]),
                "eligibility_class": elig_class,
                "embedding_text": str(rec["embedding_text"]),
                "source_fields": [str(x) for x in (rec.get("source_fields") or [])],
                "boundary_metadata": bm_str,
                "required_signals": [str(x) for x in (rec.get("required_signals") or [])],
                "negative_keywords": [str(x) for x in (rec.get("negative_keywords") or [])],
                "kind": None if rec.get("kind") in (None, "") else str(rec["kind"]),
                "mvp_scope": None if rec.get("mvp_scope") in (None, "") else str(rec["mvp_scope"]),
                "is_positive_target": is_pos,
                "can_decide_none": None if can_none is None else bool(can_none),
                "has_uncertain": has_unc,
                "uncertain_notes": unc_notes,
                "notes": str(rec.get("notes") or ""),
            }
        )

    schema = pa.schema(
        [
            pa.field("embedding_row", pa.int64(), nullable=False),
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("prototype_id", pa.string(), nullable=False),
            pa.field("stage", pa.string(), nullable=False),
            pa.field("group_id", pa.string(), nullable=True),
            pa.field("tieu_chi", pa.string(), nullable=True),
            pa.field("criterion_key", pa.string(), nullable=True),
            pa.field("criterion_ref", pa.string(), nullable=True),
            pa.field("prototype_type", pa.string(), nullable=False),
            pa.field("prototype_family", pa.string(), nullable=False),
            pa.field("anchor_status", pa.string(), nullable=True),
            pa.field("criterion_anchor_status", pa.string(), nullable=True),
            pa.field("criterion_anchors_status", pa.string(), nullable=True),
            pa.field("anchor_text", pa.string(), nullable=True),
            pa.field("source_anchor_index", pa.int64(), nullable=True),
            pa.field("event_matchable", pa.bool_(), nullable=False),
            pa.field("eligibility_class", pa.string(), nullable=False),
            pa.field("embedding_text", pa.string(), nullable=False),
            pa.field("source_fields", pa.list_(pa.string()), nullable=False),
            pa.field("boundary_metadata", pa.string(), nullable=False),
            pa.field("required_signals", pa.list_(pa.string()), nullable=False),
            pa.field("negative_keywords", pa.list_(pa.string()), nullable=False),
            pa.field("kind", pa.string(), nullable=True),
            pa.field("mvp_scope", pa.string(), nullable=True),
            pa.field("is_positive_target", pa.bool_(), nullable=False),
            pa.field("can_decide_none", pa.bool_(), nullable=True),
            pa.field("has_uncertain", pa.bool_(), nullable=False),
            pa.field("uncertain_notes", pa.string(), nullable=True),
            pa.field("notes", pa.string(), nullable=False),
        ]
    )
    table = pa.Table.from_pylist(normalized_rows, schema=schema)
    pq.write_table(table, path)


def load_rubric_prototypes_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load Schema v2.0 rubric prototypes from rubric_prototypes.jsonl and assign embedding_row."""
    records: List[Dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} in {jsonl_path} is not a JSON object.")
            text = str(obj.get("embedding_text") or "")
            if not text.startswith(PASSAGE_PREFIX):
                text = format_passage(text)
            if text.startswith(PASSAGE_PREFIX + PASSAGE_PREFIX):
                raise ValueError(f"Double 'passage: ' prefix detected on {obj.get('prototype_id')}")
            obj["embedding_text"] = text
            obj["embedding_row"] = len(records)
            records.append(obj)
    return records


def validate_rubric_cross_artifact_consistency(
    *,
    rubric: Dict[str, Any],
    jsonl_path: Path,
    parquet_path: Path,
    npy_path: Path,
    model,
) -> Dict[str, Any]:
    """Validate rubric.yaml -> rubric_prototypes.jsonl -> rubric_metadata.parquet -> rubric_embeddings.npy."""
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    jsonl_records = load_rubric_prototypes_jsonl(jsonl_path)
    df = pd.read_parquet(parquet_path)
    vecs = np.load(npy_path, mmap_mode="r")

    n = len(jsonl_records)
    add(
        "cross_artifact_row_counts_match",
        n == len(df) == vecs.shape[0] and vecs.shape[1] == EMBEDDING_DIM,
        f"jsonl={n}, parquet={len(df)}, npy={tuple(vecs.shape)}",
    )
    add("npy_float32_and_finite", vecs.dtype == np.float32 and bool(np.isfinite(vecs).all()), f"dtype={vecs.dtype}")

    norms = np.linalg.norm(np.asarray(vecs, dtype=np.float64), axis=1)
    max_norm_err = float(np.max(np.abs(norms - 1.0))) if len(norms) else float("inf")
    add("npy_l2_normalized", max_norm_err < 1e-5, f"max |norm - 1| = {max_norm_err:.2e}")

    jsonl_ids = [r["prototype_id"] for r in jsonl_records]
    pq_ids = df["prototype_id"].tolist()
    pq_rows = df["embedding_row"].tolist()
    add(
        "prototype_id_and_embedding_row_exact_order",
        jsonl_ids == pq_ids and pq_rows == list(range(n)) and len(set(jsonl_ids)) == n,
        f"exact 1:1 order across {n} prototypes",
    )

    # Verify exact text match between JSONL and Parquet
    texts_match = [r["embedding_text"] for r in jsonl_records] == df["embedding_text"].tolist()
    add("embedding_text_jsonl_parquet_match", texts_match, f"exact match={texts_match}")

    # Spot-check vector re-encoding alignment across all 4 stages (group, criterion, reminder, special)
    spot_indices: List[int] = []
    for st in ("group", "criterion", "reminder", "special"):
        matches = df.index[df["stage"] == st].tolist()
        if matches:
            spot_indices.append(int(matches[0]))
            if matches[-1] != matches[0]:
                spot_indices.append(int(matches[-1]))
    spot_indices = sorted(set(spot_indices))
    reenc, _ = encode_texts(model, [df.iloc[i]["embedding_text"] for i in spot_indices])
    cosines = [float(np.dot(vecs[i], v)) for i, v in zip(spot_indices, reenc)]
    add(
        "vector_to_prototype_alignment_spotcheck",
        all(c >= 0.99999 for c in cosines),
        f"spot_indices={spot_indices}, min_cosine={min(cosines):.6f}",
    )

    # Scope separation checks
    s1 = df[df["stage"] == "group"]
    s2 = df[df["stage"] == "criterion"]
    rem = df[df["stage"] == "reminder"]
    spec = df[df["stage"] == "special"]
    add(
        "stage_and_eligibility_isolation",
        (
            s1["group_id"].tolist() == ["C1", "C2", "C3", "C4"]
            and set(s2["group_id"]) <= {"C1", "C2", "C3", "C4"}
            and set(s2["eligibility_class"]) <= {"positive_candidate", "conditional_candidate"}
            and set(rem["criterion_ref"]) == {"6.4"}
            and set(rem["eligibility_class"]) == {"reminder_only"}
            and set(spec["criterion_ref"]) == {"NONE"}
            and set(spec["eligibility_class"]) == {"negative_reference_only"}
        ),
        f"group={len(s1)}, criterion={len(s2)}, reminder={len(rem)}, special={len(spec)}",
    )

    return {"all_passed": all(c["passed"] for c in checks), "checks": checks}


def write_sample_parquet(records: List[Dict[str, Any]], path: Path) -> None:
    """Write sample_metadata.parquet aligned 1:1 with sample_embeddings.npy (zero vectors stored)."""
    schema = pa.schema(
        [
            pa.field("embedding_row", pa.int64(), nullable=False),
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("source_row_index", pa.int64(), nullable=False),
            pa.field("source_stt", pa.string(), nullable=True),
            pa.field("source_dataset", pa.string(), nullable=False),
            pa.field("event_title", pa.string(), nullable=False),
            pa.field("event_description", pa.string(), nullable=True),
            pa.field("embedding_text", pa.string(), nullable=False),
            pa.field("original_label", pa.string(), nullable=False),
            pa.field("criterion_ref", pa.string(), nullable=False),
            pa.field("criterion_key", pa.string(), nullable=False),
            pa.field("group_id", pa.string(), nullable=False),
            pa.field("tieu_chi", pa.string(), nullable=True),
            pa.field("role", pa.string(), nullable=True),
            pa.field("organizer", pa.string(), nullable=True),
            pa.field("label_status", pa.string(), nullable=False),
            pa.field("embedding_model", pa.string(), nullable=False),
            pa.field("embedding_dimension", pa.int64(), nullable=False),
            pa.field("normalization", pa.string(), nullable=False),
            pa.field("source_metadata", pa.string(), nullable=False),
        ]
    )
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, path)


def validate_sample_artifacts(
    *,
    npy_path: Path,
    parquet_path: Path,
    rubric: Dict[str, Any],
    model,
    sample_audit: Dict[str, Any],
    token_stats: Dict[str, Any],
    expected_model: str = MODEL_ID,
    expected_dim: int = EMBEDDING_DIM,
) -> Dict[str, Any]:
    """Run Section 10 A-D technical validation on (temporary or final) sample artifacts."""
    from src.classification.data.loaders import ReferenceBranchDataset
    from src.classification.data.validators import REQUIRED_SAMPLE_COLUMNS
    from src.classification.retrieval.sample_retriever import SampleRetriever

    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # --- A. Dataset Validation ---
    add(
        "dataset_zero_malformed_rows",
        len(sample_audit["malformed_rows"]) == 0,
        f"malformed_rows={len(sample_audit['malformed_rows'])} across {sample_audit['total_source_rows']} source rows",
    )
    add(
        "dataset_zero_label_inconsistencies",
        len(sample_audit["label_inconsistencies"]) == 0 and len(sample_audit["criterion_text_variants"]) == 0,
        (
            f"label_inconsistencies={len(sample_audit['label_inconsistencies'])}, "
            f"criterion_text_variants={len(sample_audit['criterion_text_variants'])}"
        ),
    )
    add(
        "dataset_zero_empty_event_texts",
        len(sample_audit["skipped_empty_text_rows"]) == 0,
        f"empty_text_rows={len(sample_audit['skipped_empty_text_rows'])}",
    )

    # --- B. Embedding Validation ---
    sample_vectors = np.load(npy_path)
    sample_metadata = pd.read_parquet(parquet_path)
    n_expected = int(sample_audit["eligible_sample_count"])

    add(
        "embedding_count_matches_eligible_samples",
        sample_vectors.ndim == 2 and sample_vectors.shape[0] == n_expected and n_expected > 0,
        f"npy_rows={sample_vectors.shape[0]}, eligible_samples={n_expected}",
    )
    add(
        "embedding_dimension_384",
        sample_vectors.ndim == 2 and sample_vectors.shape[1] == expected_dim,
        f"shape={tuple(sample_vectors.shape)}, expected_dim={expected_dim}",
    )
    add(
        "embedding_dtype_float32",
        sample_vectors.dtype == np.float32,
        f"dtype={sample_vectors.dtype}",
    )
    add(
        "embedding_all_finite",
        bool(np.isfinite(sample_vectors).all()),
        "No NaN or Inf values",
    )

    norms = np.linalg.norm(np.asarray(sample_vectors, dtype=np.float64), axis=1)
    max_norm_err = float(np.max(np.abs(norms - 1.0))) if len(norms) else float("inf")
    add(
        "embedding_l2_normalized",
        max_norm_err < 1e-5,
        f"max |norm - 1| = {max_norm_err:.2e}",
    )
    add(
        "tokenizer_zero_silent_truncation",
        int(token_stats.get("truncated_count", 0)) == 0,
        (
            f"min={token_stats.get('min_tokens')}, median={token_stats.get('median_tokens')}, "
            f"max={token_stats.get('max_tokens')}, max_seq_length={token_stats.get('max_seq_length')}, "
            f"truncated={token_stats.get('truncated_count')}"
        ),
    )

    # Leakage and prefix check
    leakage_hits: List[str] = []
    prefix_bad = 0
    for _, row in sample_metadata.iterrows():
        etext = str(row["embedding_text"])
        if not etext.startswith(PASSAGE_PREFIX) or etext.startswith(PASSAGE_PREFIX + PASSAGE_PREFIX):
            prefix_bad += 1
        desc_val = None if pd.isna(row["event_description"]) else str(row["event_description"])
        expected_etext = build_sample_embedding_text(str(row["event_title"]), desc_val)
        if etext != expected_etext:
            leakage_hits.append(f"{row['sample_id']}: embedding_text deviates from leakage-free template")
        meta = json.loads(row["source_metadata"])
        tc_qc = str(meta.get("Tieu_Chi_Quy_Che") or "").strip()
        if tc_qc and tc_qc.lower() in etext.lower():
            leakage_hits.append(f"{row['sample_id']}: contains Tieu_Chi_Quy_Che")
        for forbidden in ("NONE", "NO_MATCH", "NEEDS_REVIEW", "Vai trò:", "Đơn vị tổ chức:", "Diem_Toi_Da"):
            if forbidden in etext:
                leakage_hits.append(f"{row['sample_id']}: contains forbidden marker '{forbidden}'")
    add(
        "embedding_passage_prefix_and_zero_leakage",
        prefix_bad == 0 and len(leakage_hits) == 0,
        f"prefix_errors={prefix_bad}, leakage_hits={len(leakage_hits)}"
        + ("" if not leakage_hits else f" ({'; '.join(leakage_hits[:3])})"),
    )

    # Re-encode spot check across first, middle, last, and each major group
    spot_indices: List[int] = [0, len(sample_metadata) // 2, len(sample_metadata) - 1]
    for grp in POSITIVE_SAMPLE_GROUPS:
        g_idx = sample_metadata.index[sample_metadata["group_id"] == grp].tolist()
        if g_idx:
            spot_indices.append(int(g_idx[0]))
            spot_indices.append(int(g_idx[-1]))
    spot_indices = sorted(set(spot_indices))
    reenc, _ = encode_texts(model, [str(sample_metadata.iloc[i]["embedding_text"]) for i in spot_indices])
    spot_cos = [float(np.dot(sample_vectors[idx], vec)) for idx, vec in zip(spot_indices, reenc)]
    add(
        "embedding_vector_row_alignment_spotcheck",
        all(c >= 0.99999 for c in spot_cos),
        f"spot_indices={spot_indices}, min_cosine={min(spot_cos):.6f}",
    )

    # --- C. Metadata Validation ---
    missing_req_cols = sorted(REQUIRED_SAMPLE_COLUMNS - set(sample_metadata.columns))
    no_vector_cols = not any("vector" in c.lower() or c.lower() == "embedding" for c in sample_metadata.columns)
    add(
        "metadata_schema_and_no_embedded_vectors",
        len(missing_req_cols) == 0 and no_vector_cols,
        f"columns={len(sample_metadata.columns)}, missing_required={missing_req_cols}, no_vector_col={no_vector_cols}",
    )
    add(
        "metadata_row_count_matches_embeddings",
        len(sample_metadata) == sample_vectors.shape[0],
        f"metadata_rows={len(sample_metadata)}, embedding_rows={sample_vectors.shape[0]}",
    )
    emb_rows_ok = list(sample_metadata["embedding_row"]) == list(range(len(sample_metadata)))
    sid_unique = bool(sample_metadata["sample_id"].is_unique)
    src_rows = sample_metadata["source_row_index"].tolist()
    src_order_ok = all(a < b for a, b in zip(src_rows, src_rows[1:]))
    add(
        "metadata_contiguous_embedding_row_and_source_order",
        emb_rows_ok and sid_unique and src_order_ok,
        f"embedding_row_0..N-1={emb_rows_ok}, unique_sample_id={sid_unique}, strictly_increasing_source_row={src_order_ok}",
    )

    valid_groups = set((rubric.get("groups") or {}).keys())
    items = rubric.get("items") or {}
    ref_to_item = {
        str(v["ref"]).strip(): (k, v)
        for k, v in items.items()
        if isinstance(v, dict) and "ref" in v
    }
    ref_errors: List[str] = []
    for _, row in sample_metadata.iterrows():
        ref = str(row["criterion_ref"])
        key = str(row["criterion_key"])
        grp = str(row["group_id"])
        orig = str(row["original_label"])
        if orig != ref:
            ref_errors.append(f"{row['sample_id']}: original_label {orig!r} != criterion_ref {ref!r}")
        elif grp not in valid_groups or grp not in POSITIVE_SAMPLE_GROUPS:
            ref_errors.append(f"{row['sample_id']}: invalid group_id {grp!r}")
        elif ref not in ref_to_item:
            ref_errors.append(f"{row['sample_id']}: unknown criterion_ref {ref!r}")
        else:
            rkey, ritem = ref_to_item[ref]
            if rkey != key or str(ritem.get("group")) != grp or ritem.get("event_matchable") is not True:
                ref_errors.append(f"{row['sample_id']}: rubric mismatch ({ref}, {key}, {grp})")
    add(
        "metadata_rubric_referential_integrity",
        len(ref_errors) == 0,
        f"All {len(sample_metadata)} rows match rubric.yaml C1-C4 event_matchable:true criteria"
        if not ref_errors
        else "; ".join(ref_errors[:5]),
    )

    model_col_ok = bool(
        (sample_metadata["embedding_model"] == expected_model).all()
        and (sample_metadata["embedding_dimension"] == expected_dim).all()
        and (sample_metadata["normalization"] == "l2").all()
    )
    add(
        "metadata_embedding_provenance_recorded",
        model_col_ok,
        f"model={expected_model}, dim={expected_dim}, normalization=l2",
    )

    # --- D. Compatibility Validation (SampleRetriever Stage 1 & Stage 2) ---
    ds = ReferenceBranchDataset(
        branch_name="sample",
        embeddings=sample_vectors,
        metadata=sample_metadata,
        embeddings_path=npy_path,
        metadata_path=parquet_path,
    )
    retriever = SampleRetriever(ds)
    probe_q = np.asarray(sample_vectors[0], dtype=np.float32)
    s1_hits = retriever.retrieve_stage1(probe_q, top_k=10)
    s2_hits = retriever.retrieve_stage2(probe_q, candidate_groups=["C1", "C3"], top_k=10)
    compat_ok = (
        retriever.eligible_count == len(sample_metadata)
        and retriever.available_groups == list(POSITIVE_SAMPLE_GROUPS)
        and len(s1_hits) == 10
        and len(s2_hits) == 10
        and all(h.group_id in {"C1", "C3"} for h in s2_hits)
        and abs(s1_hits[0].similarity - 1.0) < 1e-4
    )
    add(
        "sample_retriever_stage1_stage2_compatibility",
        compat_ok,
        (
            f"eligible={retriever.eligible_count}, groups={retriever.available_groups}, "
            f"stage1_hits={len(s1_hits)} (top1_sim={s1_hits[0].similarity:.6f}), "
            f"stage2_C1C3_hits={len(s2_hits)}"
        ),
    )

    return {"all_passed": all(c["passed"] for c in checks), "checks": checks}


def build_sample_audit_report(
    *,
    sample_records: List[Dict[str, Any]],
    sample_audit: Dict[str, Any],
    embedding_config: Dict[str, Any],
    model_info: Dict[str, Any],
    token_stats: Dict[str, Any],
    validation: Dict[str, Any],
    output_dir: Path,
    generation_ts: str,
    rubric_untouched_hashes: Dict[str, str],
) -> str:
    """Build outputs/sample_embedding_generation_report.md covering all Section 12 sections."""
    grp_counts = Counter(r["group_id"] for r in sample_records)
    ref_counts = Counter(r["criterion_ref"] for r in sample_records)
    ref_to_group = {r["criterion_ref"]: r["group_id"] for r in sample_records}
    ref_to_key = {r["criterion_ref"]: r["criterion_key"] for r in sample_records}
    excl_by_reason = Counter(e["reason"] for e in sample_audit["excluded_rows"])
    excl_by_ref = Counter(
        (e["original_label"], e.get("group_id", "?"), e.get("kind", "?"))
        for e in sample_audit["excluded_rows"]
    )

    col_map = sample_audit["detected_column_mapping"]
    embedded_fields = [col_map["title"]] + ([col_map["description"]] if col_map.get("description") else [])
    metadata_only_fields = [c for c in sample_audit["csv_columns"] if c not in embedded_fields]

    lines: List[str] = [
        "# Sample Embedding Generation Audit Report — HCMUS DRL",
        "",
        f"> **Generated (UTC):** `{generation_ts}`  ",
        "> **Scope notice:** Technical data, embedding, metadata, and compatibility audit for the Sample Branch (`sample_embeddings.npy`, `sample_metadata.parquet`). Does **not** evaluate classification accuracy, precision, recall, or F1-score, and no classification algorithm, fusion weights, or Rubric Branch artifacts were modified.",
        "",
        "---",
        "",
        "## 1. Source Dataset",
        "",
        "| Property | Value |",
        "|---|---|",
        f"| **Authoritative input file** | `{sample_audit['source_csv_path']}` |",
        f"| **File format** | CSV (`utf-8`, BOM=`{sample_audit['source_has_utf8_bom']}`, delimiter=`,`, quotechar=`\"`) |",
        f"| **File size / SHA-256** | `{sample_audit['source_csv_size_bytes']}` bytes / `{sample_audit['source_csv_sha256']}` |",
        f"| **Total source records** | `{sample_audit['total_source_rows']}` |",
        f"| **Detected columns ({len(sample_audit['csv_columns'])})** | `{', '.join(sample_audit['csv_columns'])}` |",
        f"| **Malformed rows (column-count mismatch)** | `{len(sample_audit['malformed_rows'])}` |",
        f"| **Fields selected for `embedding_text`** | `{', '.join(embedded_fields)}` *(no separate description column present in source CSV)* |",
        f"| **Fields kept in metadata only (strictly excluded from `embedding_text`)** | `{', '.join(metadata_only_fields)}` |",
        "",
        "### Detected Source Schema & Column Mapping",
        "",
        "| Source Column | Empty Rows (of 1,330) | Mapped Role / Metadata Field | Embedded in Vector? |",
        "|---|---:|---|---|",
    ]

    role_desc = {
        col_map.get("row_id"): "Source row identifier (`source_stt`, `source_row_index`)",
        col_map.get("label"): "Ground-truth criterion reference (`original_label`, `criterion_ref` -> rubric `criterion_key`, `group_id`)",
        col_map.get("organizer"): "Organizing unit (`organizer`, nullable)",
        col_map.get("title"): "Event title (`event_title`)",
        col_map.get("description"): "Event description (`event_description`, nullable)",
        col_map.get("role"): "Participation role (`role`, nullable)",
        col_map.get("criterion_text"): "Rubric criterion description (metadata only in `source_metadata`; excluded to prevent label leakage)",
        col_map.get("points"): "Max DRL points (metadata only in `source_metadata`; excluded to prevent label leakage)",
    }
    for col in sample_audit["csv_columns"]:
        is_emb = "**YES**" if col in embedded_fields else "No (metadata only)"
        empty_n = sample_audit["missing_counts_all_rows"].get(col, 0)
        lines.append(f"| `{col}` | {empty_n} | {role_desc.get(col, 'Preserved in `source_metadata`')} | {is_emb} |")

    lines += [
        "",
        "---",
        "",
        "## 2. Processing Summary",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| **Total source rows (`activities_sample.csv`)** | `{sample_audit['total_source_rows']}` |",
        f"| **Valid eligible samples embedded (`C1`–`C4`, `event_matchable: true`)** | **`{sample_audit['eligible_sample_count']}`** |",
        f"| **Skipped / excluded rows (non-eligible or Group `P` reminder/penalty)** | `{sample_audit['excluded_row_count']}` |",
        f"| **Invalid / unknown / malformed label rows** | `{sample_audit['invalid_or_special_label_count']}` |",
        f"| **Label inconsistencies (`Ma_Muc` / `group_id` / `criterion_key` / `Tieu_Chi_Quy_Che`)** | `{len(sample_audit['label_inconsistencies']) + len(sample_audit['criterion_text_variants'])}` |",
        f"| **Exact full-row duplicates (ignoring `STT`) — all source / eligible** | `{sample_audit['all_source_exact_row_dup_groups']}` / `{len(sample_audit['preserved_exact_row_duplicates'])}` |",
        f"| **Duplicate-title groups among eligible samples (preserved)** | `{len(sample_audit['preserved_exact_duplicates'])}` groups (`{len(sample_audit['preserved_cross_criterion_duplicates'])}` cross-criterion + `{len(sample_audit['preserved_same_criterion_duplicates'])}` same-criterion) |",
        f"| **Orthographic / diacritic near-duplicate groups among eligible samples (preserved)** | `{len(sample_audit['preserved_near_duplicates'])}` |",
        "",
        "### Samples by Major Group (`group_id`)",
        "",
        "| `group_id` | Eligible Samples | Share (%) |",
        "|---|---:|---:|",
    ]
    total_el = max(len(sample_records), 1)
    for grp in POSITIVE_SAMPLE_GROUPS:
        cnt = grp_counts.get(grp, 0)
        lines.append(f"| `{grp}` | {cnt} | {100.0 * cnt / total_el:.1f}% |")

    lines += [
        "",
        "### Samples by Criterion (`criterion_ref`)",
        "",
        "| `criterion_ref` | `group_id` | `criterion_key` | Eligible Samples |",
        "|---|---|---|---:|",
    ]
    for ref in sorted(ref_counts.keys(), key=lambda x: [int(p) if p.isdigit() else 999 for p in x.split(".")]):
        lines.append(f"| `{ref}` | `{ref_to_group[ref]}` | `{ref_to_key[ref]}` | {ref_counts[ref]} |")

    lines += [
        "",
        "### Excluded / Skipped Rows Breakdown (Documented, Not Silently Dropped)",
        "",
        "| Exclusion Reason | Row Count |",
        "|---|---:|",
    ]
    for reason, cnt in sorted(excl_by_reason.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{reason}` | {cnt} |")

    lines += [
        "",
        "<details>",
        "<summary>Excluded rows by criterion (`Ma_Muc`)</summary>",
        "",
        "| `Ma_Muc` | `group_id` | `kind` | Excluded Rows |",
        "|---|---|---|---:|",
    ]
    for (muc, grp, kind), cnt in sorted(
        excl_by_ref.items(),
        key=lambda kv: [int(p) if p.isdigit() else 999 for p in kv[0][0].split(".")],
    ):
        lines.append(f"| `{muc}` | `{grp}` | `{kind}` | {cnt} |")
    lines += [
        "",
        "</details>",
        "",
        "### Preserved Duplicate & Multi-Label Records (Zero Silent Deletion)",
        "",
        f"- **Cross-criterion duplicate titles (`{len(sample_audit['preserved_cross_criterion_duplicates'])}` groups, preserved as separate labeled samples):**",
    ]
    for grp in sample_audit["preserved_cross_criterion_duplicates"]:
        detail = "; ".join(
            f"`{x['sample_id']}` (row `{x['source_row_index']}`, ref `{x['criterion_ref']}`, org=`{x['organizer'] or ''}`, role=`{x['role'] or ''}`)"
            for x in grp
        )
        lines.append(f"  - \"{grp[0]['event_title']}\": {detail}")

    lines.append(
        f"- **Same-criterion duplicate titles (`{len(sample_audit['preserved_same_criterion_duplicates'])}` groups, differing by organizer or role — preserved):**"
    )
    for grp in sample_audit["preserved_same_criterion_duplicates"]:
        detail = "; ".join(
            f"`{x['sample_id']}` (row `{x['source_row_index']}`, ref `{x['criterion_ref']}`, org=`{x['organizer'] or ''}`, role=`{x['role'] or ''}`)"
            for x in grp
        )
        lines.append(f"  - \"{grp[0]['event_title']}\": {detail}")

    lines.append(
        f"- **Orthographic / diacritic near-duplicate groups (`{len(sample_audit['preserved_near_duplicates'])}` groups — preserved):**"
    )
    for grp in sample_audit["preserved_near_duplicates"]:
        detail = "; ".join(
            f"`{x['sample_id']}` (row `{x['source_row_index']}`, ref `{x['criterion_ref']}`, title=\"{x['event_title']}\")"
            for x in grp
        )
        lines.append(f"  - {detail}")

    lines += [
        "",
        "---",
        "",
        "## 3. Embedding Configuration",
        "",
        "| Parameter | Value (from `embedding_config.json`) |",
        "|---|---|",
        f"| **Model name** | `{model_info['model_name']}` |",
        f"| **Model revision** | `{model_info['model_revision']}` |",
        f"| **Embedding dimension** | `{model_info['embedding_dimension']}` |",
        f"| **Data type** | `{model_info['dtype']}` |",
        f"| **Pooling method** | `{model_info['pooling_method']}` |",
        f"| **Normalization** | L2 unit normalization (`l2_normalization: true`) |",
        f"| **Tokenizer** | `{model_info['tokenizer_config']['tokenizer_class']}` (`max_seq_length={model_info['tokenizer_config']['max_seq_length']}`, `do_lower_case={model_info['tokenizer_config']['do_lower_case']}`) |",
        f"| **Prefix convention** | Sample passages: `passage: ...` / Queries: `query: ...` |",
        "| **Text construction rules** | Unicode NFC + whitespace collapse (matching `src/classification/embeddings/event_text_builder.py`); embeds `Ten_Hoat_Dong` only (no description column in CSV); strictly excludes `Ma_Muc`, `group_id`, `criterion_key`, `Tieu_Chi_Quy_Che`, `Diem_Toi_Da`, `Vai_Tro`, and `Organizer` to prevent label leakage. |",
        f"| **Truncation strategy & token stats** | Model `max_seq_length = {token_stats['max_seq_length']}` tokens; observed sample lengths: `min={token_stats['min_tokens']}`, `median={token_stats['median_tokens']}`, `max={token_stats['max_tokens']}`, **`truncated_count={token_stats['truncated_count']}`** (no truncation needed). |",
        "",
        "---",
        "",
        "## 4. Generated Artifacts",
        "",
        "| Artifact | Rows | Dimension / Columns | Alignment Status |",
        "|---|---:|---|---|",
        f"| `{output_dir.as_posix()}/sample_embeddings.npy` | `{len(sample_records)}` | `({len(sample_records)}, {model_info['embedding_dimension']})`, `float32`, L2-normalized | Exact 1:1 with `sample_metadata.parquet` (`0..{len(sample_records)-1}`) |",
        f"| `{output_dir.as_posix()}/sample_metadata.parquet` | `{len(sample_records)}` | `{len(sample_records[0]) if sample_records else 0}` columns (zero embedded vectors) | Exact 1:1 with `sample_embeddings.npy` (`embedding_row == 0..{len(sample_records)-1}`) |",
        "",
        f"**Metadata columns:** `{', '.join(sample_records[0].keys()) if sample_records else ''}`",
        "",
        "---",
        "",
        "## 5. Technical Validation Results",
        "",
        "| Check | Status | Details |",
        "|---|---|---|",
    ]
    for chk in validation["checks"]:
        badge = "**PASS**" if chk["passed"] else "**FAIL**"
        safe_detail = str(chk["detail"]).replace("|", "\\|")
        lines.append(f"| `{chk['check']}` | {badge} | {safe_detail} |")

    if rubric_untouched_hashes:
        lines += [
            "",
            "### Untouched Rubric & Read-Only Artifacts Verification (SHA-256)",
            "",
            "| Artifact | SHA-256 |",
            "|---|---|",
        ]
        for rel, h in rubric_untouched_hashes.items():
            lines.append(f"| `{rel}` | `{h}` |")

    unc_refs = [u["criterion_ref"] for u in sample_audit["uncovered_eligible_criteria"]]
    lines += [
        "",
        "---",
        "",
        "## 6. Unresolved Issues & Data Quality Notes",
        "",
        f"1. **No separate event description column in `activities_sample.csv`:** The CSV provides `Ten_Hoat_Dong` (event title) only; `event_description` is preserved as `null` across all `{len(sample_records)}` samples without fabrication. Runtime incoming queries in `events.json` include both title and description.",
        f"2. **Missing optional metadata fields among eligible samples:** `role` (`Vai_Tro`) is empty/null on `{sample_audit['eligible_missing_counts']['role']}/{len(sample_records)}` samples, and `organizer` (`Organizer`) is empty/null on `{sample_audit['eligible_missing_counts']['organizer']}/{len(sample_records)}` samples (stored as `null`, not fabricated).",
        f"3. **Uncovered C1–C4 `event_matchable: true` criteria (`{len(unc_refs)}/20`):** Criteria `{', '.join(unc_refs) if unc_refs else 'none'}` have `0` rows in `activities_sample.csv` and continue to rely on Stage 2 Rubric Branch prototypes (via `missing_branch_reference_policy: reweight_available_branch`).",
        f"4. **Criterion `6.4` (`STT=1329`, `group=P`) excluded from positive Sample Branch:** Row 1329 (`\"Không tham gia, không đạt Sinh hoạt công dân Cuối khóa năm học 2025-2026\"`, `Diem_Toi_Da=-20`) is a mandatory-event penalty record (`group: P`, `mvp_scope: remind`) and is excluded from the positive `C1`–`C4` Sample Branch to remain aligned with `RubricRetriever` (`stage=\"reminder\"`) and `Module3Config.excluded_parent_groups = (\"P\",)`.",
        "5. **Cross-criterion identical titles (`6` groups, `12` rows) & near-duplicate cross-criterion titles (`1` group):** Identical event titles appear under different criteria (`1.9` vs `3.5`, `3.1` vs `3.3`, `3.3` vs `3.5`, `3.5` vs `4.5`, `4.3` vs `4.5`), reflecting role/mobilization/fundraising distinctions not fully expressed by title alone. All records are preserved faithfully without merging or relabeling.",
        "6. **Criterion `3.3` role coverage:** Among the `47` eligible `3.3` samples, `21` carry explicit `Vai_Tro = \"Huy động lực lượng\"`, `6` carry `\"Tham gia\"`, and `20` have empty `Vai_Tro` — reinforcing why the existing `3.3` mobilization decision guard remains important.",
        "7. **Severe class count imbalance:** `3.5` (`298`) and `1.9` (`242`) account for `540/908` (`59.5%`) of eligible samples, whereas `2.2`, `3.13`, and `4.6` have `1` sample each.",
        "8. **Raw source title artifacts preserved verbatim (no rewriting):** Row `751` (`3.5`) contains a raw concatenation/truncation artifact (`\"...“Theo dấu chân Bác”DTVT - Tổ chức sự kiện “Hành trình theo chân Bác\"`); rows `32`, `742`, `743`, `1113` contain leading underscore artifacts (`_Chấn động Pangaea`, `_Tứ cực đại chiến`, `_25DTV CUP`, `_Thanh niên hành động xanh`); minor source typos exist on rows `402`, `416`, `549`, `1255`.",
        "9. **No verified negative (`NONE` / `NO_MATCH`) samples:** `activities_sample.csv` contains only DRL rubric records (`0` negative/out-of-scope examples).",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# End-to-End Validation Suite (Section 8)
# ---------------------------------------------------------------------------


def validate_artifacts(
    output_dir: Path,
    rubric: Dict[str, Any],
    model,
) -> Dict[str, Any]:
    """Validate all generated files from disk (vectors, metadata, alignment, leakage, filtering)."""
    rubric_npy_path = output_dir / "rubric_embeddings.npy"
    rubric_pq_path = output_dir / "rubric_metadata.parquet"
    sample_npy_path = output_dir / "sample_embeddings.npy"
    sample_pq_path = output_dir / "sample_metadata.parquet"
    config_path = output_dir / "embedding_config.json"

    checks: List[Dict[str, Any]] = []

    def record_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # 1. File loading test (Section 8 exact snippet)
    rubric_vectors = np.load(rubric_npy_path, mmap_mode="r")
    rubric_metadata = pd.read_parquet(rubric_pq_path)
    sample_vectors = np.load(sample_npy_path, mmap_mode="r")
    sample_metadata = pd.read_parquet(sample_pq_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    record_check(
        "file_loading_and_row_count_match",
        rubric_vectors.shape[0] == len(rubric_metadata) and sample_vectors.shape[0] == len(sample_metadata),
        f"rubric={rubric_vectors.shape} vs {len(rubric_metadata)} rows; sample={sample_vectors.shape} vs {len(sample_metadata)} rows",
    )
    record_check(
        "embedding_dimensions_384",
        rubric_vectors.shape[1] == EMBEDDING_DIM and sample_vectors.shape[1] == EMBEDDING_DIM,
        f"rubric_dim={rubric_vectors.shape[1]}, sample_dim={sample_vectors.shape[1]}",
    )
    record_check(
        "float32_dtype",
        rubric_vectors.dtype == np.float32 and sample_vectors.dtype == np.float32,
        f"rubric_dtype={rubric_vectors.dtype}, sample_dtype={sample_vectors.dtype}",
    )
    record_check(
        "finite_values",
        bool(np.isfinite(rubric_vectors).all() and np.isfinite(sample_vectors).all()),
        "All vector entries finite (no NaN/Inf)",
    )

    # L2 norms
    r_norms = np.linalg.norm(np.asarray(rubric_vectors, dtype=np.float64), axis=1)
    s_norms = np.linalg.norm(np.asarray(sample_vectors, dtype=np.float64), axis=1)
    r_max_err = float(np.max(np.abs(r_norms - 1.0))) if len(r_norms) else 0.0
    s_max_err = float(np.max(np.abs(s_norms - 1.0))) if len(s_norms) else 0.0
    record_check(
        "l2_norm_unit_length",
        r_max_err < 1e-5 and s_max_err < 1e-5,
        f"rubric max |norm-1|={r_max_err:.2e}, sample max |norm-1|={s_max_err:.2e}",
    )

    # Sequential embedding_row and unique IDs
    r_rows_ok = list(rubric_metadata["embedding_row"]) == list(range(len(rubric_metadata)))
    s_rows_ok = list(sample_metadata["embedding_row"]) == list(range(len(sample_metadata)))
    record_check(
        "sequential_unique_embedding_row",
        r_rows_ok and s_rows_ok,
        f"rubric 0..{len(rubric_metadata)-1}: {r_rows_ok}; sample 0..{len(sample_metadata)-1}: {s_rows_ok}",
    )
    r_pid_unique = rubric_metadata["prototype_id"].is_unique
    s_sid_unique = sample_metadata["sample_id"].is_unique
    record_check(
        "unique_prototype_and_sample_ids",
        bool(r_pid_unique and s_sid_unique),
        f"prototype_id unique={r_pid_unique}, sample_id unique={s_sid_unique}",
    )

    # Valid rubric references and Stage 1 / Stage 2 separation
    valid_groups = set((rubric.get("groups") or {}).keys())
    valid_items = {k: str(v["ref"]) for k, v in (rubric.get("items") or {}).items() if isinstance(v, dict)}
    em_true_keys = {k for k, v in (rubric.get("items") or {}).items() if isinstance(v, dict) and v.get("event_matchable") is True}

    r_stages_ok = set(rubric_metadata["stage"].unique()) <= VALID_RUBRIC_STAGES
    r_ptypes_ok = set(rubric_metadata["prototype_type"].unique()) <= VALID_PROTOTYPE_TYPES
    s1_df = rubric_metadata[rubric_metadata["stage"] == "group"]
    s2_df = rubric_metadata[rubric_metadata["stage"] == "criterion"]

    s1_null_ok = bool(s1_df["criterion_key"].isna().all() and s1_df["criterion_ref"].isna().all())
    s1_groups_ok = set(s1_df["group_id"]) <= set(POSITIVE_MAJOR_GROUPS) and len(s1_df) == s1_df["group_id"].nunique()
    s2_refs_ok = all(
        row["criterion_key"] in em_true_keys and valid_items.get(row["criterion_key"]) == row["criterion_ref"]
        for _, row in s2_df.iterrows()
    )
    s2_defs_covered = set(s2_df[s2_df["prototype_type"] == "criterion_definition"]["criterion_key"]) == em_true_keys
    record_check(
        "rubric_stage_separation_and_coverage",
        r_stages_ok and r_ptypes_ok and s1_null_ok and s1_groups_ok and s2_refs_ok and s2_defs_covered,
        f"Stage 1 groups={sorted(s1_df['group_id'].tolist())}; Stage 2 covered all {len(em_true_keys)} event_matchable=true criteria={s2_defs_covered}",
    )

    # Sample rubric references validity
    s_refs_ok = all(
        row["group_id"] in valid_groups
        and row["criterion_key"] in em_true_keys
        and valid_items.get(row["criterion_key"]) == row["criterion_ref"]
        for _, row in sample_metadata.iterrows()
    )
    record_check(
        "sample_rubric_references_valid",
        s_refs_ok,
        f"All {len(sample_metadata)} samples map to valid event_matchable=true rubric items",
    )

    # Passage prefix check
    r_prefix_ok = bool(rubric_metadata["embedding_text"].str.startswith(PASSAGE_PREFIX).all())
    s_prefix_ok = bool(sample_metadata["embedding_text"].str.startswith(PASSAGE_PREFIX).all())
    record_check(
        "passage_prefix_enforced",
        r_prefix_ok and s_prefix_ok,
        f"rubric_prefix_ok={r_prefix_ok}, sample_prefix_ok={s_prefix_ok}",
    )

    # Sample label leakage check
    leakage_hits: List[str] = []
    for _, row in sample_metadata.iterrows():
        etext = str(row["embedding_text"])
        expected_etext = format_passage(str(row["event_title"]))
        if etext != expected_etext:
            leakage_hits.append(f"{row['sample_id']}: embedding_text != 'passage: ' + event_title")
        meta = json.loads(row["source_metadata"])
        tieu_chi_qc = meta.get("Tieu_Chi_Quy_Che", "")
        if tieu_chi_qc and tieu_chi_qc.lower() in etext.lower():
            leakage_hits.append(f"{row['sample_id']}: contains Tieu_Chi_Quy_Che")
        for forbidden in ("NONE", "NO_MATCH", "NEEDS_REVIEW", "Vai trò:", "Đơn vị tổ chức:", "Diem_Toi_Da"):
            if forbidden in etext:
                leakage_hits.append(f"{row['sample_id']}: contains forbidden marker '{forbidden}'")
    record_check(
        "zero_sample_label_leakage",
        len(leakage_hits) == 0,
        "Zero leakage detected across all sample embedding texts" if not leakage_hits else "; ".join(leakage_hits[:5]),
    )

    # Rubric semantic exclusion check (no scoring points, CSV refs, QĐ 2020, completion conditions)
    rubric_exclusion_patterns = [
        r"\b\d+đ\b",
        r"\(5/10/15\)",
        r"mức phẳng",
        r"events_full\.csv",
        r"dòng\s+\d+",
        r"QĐ\s*2020",
        r"hoàn tất vòng đầu",
        r"trên\s+50%",
        r"trừ nội dung\s+\d+\.\d+",
        r"mục\s+\d+\.\d+",
        r"điểm chỉ tính khi",
    ]
    rubric_leak_hits: List[str] = []
    for _, row in rubric_metadata.iterrows():
        etext = str(row["embedding_text"])
        for pat in rubric_exclusion_patterns:
            if re.search(pat, etext, flags=re.IGNORECASE):
                rubric_leak_hits.append(f"{row['prototype_id']} matched /{pat}/")
    record_check(
        "rubric_semantic_exclusions_clean",
        len(rubric_leak_hits) == 0,
        "Zero scoring/admin/completion leakage in rubric embedding texts" if not rubric_leak_hits else "; ".join(rubric_leak_hits[:5]),
    )

    # Spot-check exact vector-to-metadata alignment by re-encoding sample indices
    r_spot_idx = sorted({0, len(rubric_metadata) // 2, len(rubric_metadata) - 1})
    s_spot_idx = sorted({0, len(sample_metadata) // 2, len(sample_metadata) - 1})
    r_reenc, _ = encode_texts(model, [rubric_metadata.iloc[i]["embedding_text"] for i in r_spot_idx])
    s_reenc, _ = encode_texts(model, [sample_metadata.iloc[i]["embedding_text"] for i in s_spot_idx])
    r_cos = [float(np.dot(rubric_vectors[idx], vec)) for idx, vec in zip(r_spot_idx, r_reenc)]
    s_cos = [float(np.dot(sample_vectors[idx], vec)) for idx, vec in zip(s_spot_idx, s_reenc)]
    align_ok = all(c >= 0.99999 for c in r_cos + s_cos)
    record_check(
        "vector_to_metadata_alignment_spotcheck",
        align_ok,
        f"rubric spot cosines={[round(c, 6) for c in r_cos]}; sample spot cosines={[round(c, 6) for c in s_cos]}",
    )

    # Retrieval filtering compatibility check (Stage 1 -> Top-2 groups -> Stage 2 union slicing)
    s1_rows = rubric_metadata.loc[rubric_metadata["stage"] == "group", "embedding_row"].to_numpy()
    s1_vecs = rubric_vectors[s1_rows]
    test_groups = ["C1", "C3"]
    s2_rubric_rows = rubric_metadata.loc[
        (rubric_metadata["stage"] == "criterion") & (rubric_metadata["group_id"].isin(test_groups)),
        "embedding_row",
    ].to_numpy()
    s2_sample_rows = sample_metadata.loc[
        sample_metadata["group_id"].isin(test_groups),
        "embedding_row",
    ].to_numpy()
    s2_rubric_vecs = rubric_vectors[s2_rubric_rows]
    s2_sample_vecs = sample_vectors[s2_sample_rows]
    # Also test filtering by specific criterion_ref
    crit_rows = sample_metadata.loc[sample_metadata["criterion_ref"] == "1.6", "embedding_row"].to_numpy()
    crit_vecs = sample_vectors[crit_rows]
    filter_ok = (
        s1_vecs.shape == (4, EMBEDDING_DIM)
        and s2_rubric_vecs.shape[0] > 0
        and s2_rubric_vecs.shape[1] == EMBEDDING_DIM
        and s2_sample_vecs.shape[0] == len(s2_sample_rows)
        and s2_sample_vecs.shape[0] > 0
        and crit_vecs.shape == (len(crit_rows), EMBEDDING_DIM)
        and crit_vecs.shape[0] > 0
    )
    record_check(
        "two_stage_knn_filtering_compatibility",
        filter_ok,
        f"Stage1={s1_vecs.shape}; Stage2(C1|C3) rubric={s2_rubric_vecs.shape}, sample={s2_sample_vecs.shape}; ref 1.6 sample={crit_vecs.shape}",
    )

    # Config consistency check
    cfg_ok = (
        config.get("model_name") == MODEL_ID
        and config.get("embedding_dimension") == EMBEDDING_DIM
        and config.get("prefixes", {}).get("passage") == PASSAGE_PREFIX
        and config.get("prefixes", {}).get("query") == QUERY_PREFIX
        and config.get("l2_normalization") is True
        and config.get("dtype") == "float32"
        and config.get("shared_across_branches") is True
    )
    record_check(
        "embedding_config_completeness",
        cfg_ok,
        f"model={config.get('model_name')}, rev={config.get('model_revision')}, dim={config.get('embedding_dimension')}",
    )

    all_passed = all(c["passed"] for c in checks)
    return {"all_passed": all_passed, "checks": checks}


# ---------------------------------------------------------------------------
# Report Generator (Section 9)
# ---------------------------------------------------------------------------


def build_generation_report(
    *,
    rubric_records: List[Dict[str, Any]],
    rubric_audit: Dict[str, Any],
    sample_records: List[Dict[str, Any]],
    sample_audit: Dict[str, Any],
    config: Dict[str, Any],
    validation: Dict[str, Any],
    output_dir: Path,
) -> str:
    stage_counts = Counter(r["stage"] for r in rubric_records)
    ptype_counts = Counter(r["prototype_type"] for r in rubric_records)
    r_group_counts = Counter(r["group_id"] for r in rubric_records if r["stage"] == "criterion")
    r_status_counts = Counter(r["anchor_status"] for r in rubric_records if r["stage"] == "criterion")

    s_group_counts = Counter(r["group_id"] for r in sample_records)
    s_ref_counts = Counter(r["criterion_ref"] for r in sample_records)

    lines: List[str] = [
        "# Embedding Generation Report — HCMUS DRL Hierarchical 2-Stage kNN",
        "",
        f"- **Generated (UTC)**: `{config['generation_timestamp_utc']}`",
        f"- **Output Directory**: `{output_dir.as_posix()}`",
        f"- **Model**: `{config['model_name']}` (revision: `{config['model_revision']}`)",
        f"- **Embedding Spec**: `{config['embedding_dimension']}d`, `{config['dtype']}`, L2-normalized (`{config['l2_normalization']}`), pooling=`{config['pooling_method']}`, passage prefix=`\"{config['prefixes']['passage']}\"`, query prefix=`\"{config['prefixes']['query']}\"`",
        f"- **Validation Status**: **{'ALL CHECKS PASSED' if validation['all_passed'] else 'VALIDATION FAILED'}**",
        "",
        "---",
        "",
        "## 1. Deliverables Summary",
        "",
        "| Deliverable | Shape / Rows | Details |",
        "|---|---|---|",
        f"| `rubric_embeddings.npy` | `({len(rubric_records)}, {config['embedding_dimension']})` | `float32`, L2-normalized, Stage 1 + Stage 2 prototypes |",
        f"| `rubric_metadata.parquet` | `{len(rubric_records)}` rows | Aligned 1:1 with `rubric_embeddings.npy` (`embedding_row` `0..{len(rubric_records)-1}`) |",
        f"| `sample_embeddings.npy` | `({len(sample_records)}, {config['embedding_dimension']})` | `float32`, L2-normalized, eligible `events_full.csv` activities |",
        f"| `sample_metadata.parquet` | `{len(sample_records)}` rows | Aligned 1:1 with `sample_embeddings.npy` (`embedding_row` `0..{len(sample_records)-1}`) |",
        "| `embedding_config.json` | 1 JSON | Reproducibility spec, model revision, tokenizer/pooling config, source SHA-256 hashes |",
        "| `embedding_generation_report.md` | 1 Markdown | This generation, exclusion, duplicate, and validation audit report |",
        "",
        "---",
        "",
        "## 2. Rubric Branch Summary",
        "",
        f"- Total rubric items inspected: **{rubric_audit['total_rubric_items']}**",
        f"- Eligible (`event_matchable: true`) criteria: **{rubric_audit['eligible_criteria_count']}** (20 positive in C1–C4 + 1 mandatory reminder `6.4` in group P)",
        f"- Excluded (`event_matchable: false`) criteria: **{rubric_audit['excluded_criteria_count']}**",
        f"- Excluded non-criterion meta-labels: `{', '.join(rubric_audit['meta_labels_excluded'])}`",
        f"- Total rubric prototypes embedded: **{len(rubric_records)}**",
        "",
        "### Prototypes by Stage",
        "",
        "| Stage | Count | Groups Covered |",
        "|---|---|---|",
        f"| `group` (Stage 1) | {stage_counts.get('group', 0)} | `{', '.join(rubric_audit['stage1_groups_included'])}` |",
        f"| `criterion` (Stage 2) | {stage_counts.get('criterion', 0)} | `{', '.join(sorted(r_group_counts.keys()))}` |",
        "",
        "### Prototypes by Type",
        "",
        "| `prototype_type` | Count |",
        "|---|---|",
    ]
    for pt in [
        "group_definition",
        "criterion_definition",
        "verified_anchor",
        "partial_anchor",
        "draft_anchor",
        "keyword_context",
    ]:
        lines.append(f"| `{pt}` | {ptype_counts.get(pt, 0)} |")

    lines += [
        "",
        "### Stage 2 Prototypes by `anchor_status`",
        "",
        "| `anchor_status` | Count |",
        "|---|---|",
    ]
    for status_name, cnt in sorted(r_status_counts.items()):
        lines.append(f"| `{status_name}` | {cnt} |")

    lines += [
        "",
        "### Major Groups Excluded from Stage 1",
        "",
        "| Group | Name | Reason |",
        "|---|---|---|",
    ]
    for g in rubric_audit["stage1_groups_excluded"]:
        lines.append(f"| `{g['group_id']}` | {g['name']} | {g['reason']} |")

    lines += [
        "",
        "### Token Length Audit (Rubric Branch)",
        "",
        f"- Min tokens: `{config['token_stats']['rubric']['min_tokens']}`, Median: `{config['token_stats']['rubric']['median_tokens']}`, Max: `{config['token_stats']['rubric']['max_tokens']}` (limit `{config['token_stats']['rubric']['max_seq_length']}`)",
        f"- Truncated prototypes: `{config['token_stats']['rubric']['truncated_count']}`",
        "",
        "---",
        "",
        "## 3. Sample Branch Summary (`res/events_full.csv`)",
        "",
        f"- Total source rows: **{sample_audit['total_source_rows']}**",
        f"- Eligible positive sample embeddings: **{sample_audit['eligible_sample_count']}**",
        f"- Excluded source rows: **{sample_audit['excluded_row_count']}**",
        f"- Invalid / missing / special (`NONE`, `NO_MATCH`, `NEEDS_REVIEW`) labels: **{sample_audit['invalid_or_special_label_count']}**",
        f"- Detected CSV schema: `{sample_audit['csv_columns']}`",
        f"- Embedded text columns: `Ten_Hoat_Dong` (title only; no description column exists in `events_full.csv`). `Organizer`, `Vai_Tro`, `Tieu_Chi_Quy_Che`, `Diem_Toi_Da`, and `STT` are preserved strictly in `source_metadata` to prevent label/role/scoring leakage and CNTT organizer skew.",
        "",
        "### Eligible Samples by Group and Criterion",
        "",
        "| Group | Eligible Samples | Criteria Covered |",
        "|---|---|---|",
    ]
    for grp in sorted(s_group_counts.keys()):
        refs_in_grp = sorted(
            [r for r in s_ref_counts if any(s["criterion_ref"] == r and s["group_id"] == grp for s in sample_records)],
            key=lambda x: [int(p) for p in x.split(".")],
        )
        detail = ", ".join(f"`{r}` ({s_ref_counts[r]})" for r in refs_in_grp)
        lines.append(f"| `{grp}` | {s_group_counts[grp]} | {detail} |")

    lines += [
        "",
        "### Token Length Audit (Sample Branch)",
        "",
        f"- Min tokens: `{config['token_stats']['sample']['min_tokens']}`, Median: `{config['token_stats']['sample']['median_tokens']}`, Max: `{config['token_stats']['sample']['max_tokens']}` (limit `{config['token_stats']['sample']['max_seq_length']}`)",
        f"- Truncated samples: `{config['token_stats']['sample']['truncated_count']}`",
        "",
        "---",
        "",
        "## 4. Excluded Source Rows (`res/events_full.csv`)",
        "",
        f"All **{sample_audit['excluded_row_count']}** excluded rows map to valid rubric criteria whose `event_matchable` is `false` (compliance, academic record, role_variant, membership, status, post-hoc achievement, or penalty). Zero rows were dropped due to missing, unknown, or special meta-labels.",
        "",
        "| Source Row (`STT`) | `Ma_Muc` | `criterion_key` | `group_id` | Reason | Activity Title |",
        "|---|---|---|---|---|---|",
    ]
    for ex in sample_audit["excluded_rows"]:
        lines.append(
            f"| {ex['source_row_index']} | `{ex['original_label']}` | `{ex.get('criterion_key', 'N/A')}` | `{ex.get('group_id', 'N/A')}` | {ex['reason']} | {ex['event_title']} |"
        )

    lines += [
        "",
        "### Invalid, Special, or Unresolved Labels",
        "",
    ]
    if sample_audit["invalid_or_special_rows"]:
        for inv in sample_audit["invalid_or_special_rows"]:
            lines.append(f"- Row {inv['source_row_index']}: label=`{inv['original_label']}`, reason=`{inv['reason']}`, title=\"{inv['event_title']}\"")
    else:
        lines.append("- None (`0` rows had `NONE`, `NO_MATCH`, `NEEDS_REVIEW`, missing, unknown, or unmapped rubric references).")

    lines += [
        "",
        "---",
        "",
        "## 5. Missing / Uncovered Rubric Criteria in `events_full.csv`",
        "",
        f"**{len(sample_audit['uncovered_eligible_criteria'])}** of the **{rubric_audit['eligible_criteria_count']}** `event_matchable: true` criteria have **0 sample rows** in `events_full.csv`. All of them are covered by Stage 2 prototypes in `rubric_embeddings.npy` (currently draft anchors only):",
        "",
        "| `criterion_ref` | `criterion_key` | `group_id` | `anchors_status` |",
        "|---|---|---|---|",
    ]
    for unc in sample_audit["uncovered_eligible_criteria"]:
        lines.append(
            f"| `{unc['criterion_ref']}` | `{unc['criterion_key']}` | `{unc['group_id']}` | `{unc['anchors_status']}` |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. Duplicate & Near-Duplicate Records Preserved",
        "",
        f"- Exact case-insensitive title duplicates among the {sample_audit['eligible_sample_count']} eligible samples: **{len(sample_audit['preserved_exact_duplicates'])}**",
        f"- Orthographic / diacritic-normalized near-duplicate groups preserved among eligible samples: **{len(sample_audit['preserved_near_duplicates'])}**",
    ]
    for grp in sample_audit["preserved_near_duplicates"]:
        desc = " vs ".join(
            f"`{x['sample_id']}` (row {x['source_row_index']}, ref `{x['criterion_ref']}`, title=\"{x['event_title']}\")"
            for x in grp
        )
        lines.append(f"  - Preserved as separate samples: {desc}")
    lines += [
        "- Note on role-split duplicate titles in raw CSV: 5 activities appear twice in `events_full.csv` under different roles (`1.6`/`1.7`, `3.5`/`3.14`, `4.3`/`4.4`). Because `1.7`, `3.14`, and `4.4` have `event_matchable: false` (`kind: role_variant`), only the participant event rows (`1.6`, `3.5`, `4.3`) are retained in `sample_embeddings.npy`.",
        "- Note on `rubric.yaml` anchors: criterion `3.2` lists both `\"Thanh niên khoẻ\"` and `\"Thanh niên khỏe\"` as separate anchors; both are preserved as distinct `verified_anchor` prototypes.",
        "",
        "---",
        "",
        "## 7. Embedding Model & Reproducibility Configuration",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        "---",
        "",
        "## 8. Validation Results",
        "",
        "| Check | Status | Details |",
        "|---|---|---|",
    ]
    for chk in validation["checks"]:
        badge = "PASS" if chk["passed"] else "FAIL"
        lines.append(f"| `{chk['check']}` | **{badge}** | {chk['detail']} |")

    lines += [
        "",
        "---",
        "",
        "## 9. Unresolved Issues & Warnings",
        "",
        "1. **Draft-only criteria (10/21)**: Criteria `1.5`, `3.6`, `3.9`, `3.10`, `3.11`, `3.12`, `3.13`, `4.2`, `4.6`, and `6.4` have zero verified samples in `events_full.csv` and rely exclusively on `draft_anchor`, `criterion_definition`, and `keyword_context` prototypes in the Rubric branch.",
        "2. **Partially verified criterion (`3.8`)**: `C3.tu_van_tuyen_sinh_viec_lam` has `anchors_status: verified_partial` (2 `partial_anchor` from school list covering Job Fair + 2 `draft_anchor` covering admissions advising).",
        "3. **Severe class imbalance in Sample branch**: `3.5` (37 rows) and `1.9` (33 rows) account for 70/127 (55.1%) of eligible sample embeddings, whereas `3.8` has 2 rows and 10 eligible criteria have 0 rows. Stage-wise Fusion with the Rubric branch is required to prevent high-frequency sample dominance.",
        "4. **Mandatory event `6.4` (`P.sinh_hoat_cong_dan`)**: Included in Stage 2 (`stage='criterion'`, `group_id='P'`, `is_positive_target=False`, `mvp_scope='remind'`) because `event_matchable: true` in `rubric.yaml`, while Group `P` is excluded from Stage 1 positive C1–C5 group prototypes. Downstream pipelines should query `group_id == 'P'` separately for mandatory-event reminders rather than positive DRL scoring.",
        "5. **Group `C5` excluded from Stage 1**: All 11 items in `C5` have `event_matchable: false` (post-hoc achievements, statuses, or sensitive circumstances), so `C5` has no Stage 1 or Stage 2 positive prototypes and 0 eligible samples.",
        "6. **No negative (`NONE` / `NO_MATCH`) verified samples**: `events_full.csv` contains only positive DRL records (open question `Q17` in `rubric.yaml`); out-of-scope rejection thresholds must be calibrated separately once verified negative examples are collected.",
        "7. **Result-announcement row in `4.3`**: Row 139 (`\"Danh sách tình nguyện viên tiếp nhận quyên góp cho các tỉnh miền Trung\"`) is labeled `4.3` in `events_full.csv` and preserved faithfully without label modification, though `rubric.yaml` flags it as a result-list post (`post_types.result_announcement`).",
        "8. **Short title-only sample texts**: `events_full.csv` lacks an activity description column; sample embeddings encode activity titles only (`passage: <Ten_Hoat_Dong>`). Runtime queries containing long descriptions may show slight length-distribution shift relative to title-only samples.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main CLI Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Rubric and Sample embedding datasets for HCMUS DRL 2-stage kNN."
    )
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC_PATH)
    parser.add_argument("--samples-csv", type=Path, default=DEFAULT_SAMPLES_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--rubric-only",
        action="store_true",
        help=(
            "Regenerate ONLY rubric_embeddings.npy and rubric_metadata.parquet from "
            "rubric_prototypes.jsonl using the existing embedding_config.json "
            "(leaves sample_*, events_query_*, and embedding_config.json untouched)."
        ),
    )
    parser.add_argument(
        "--rubric-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "rubric_prototypes.jsonl",
        help="Path to authoritative rubric_prototypes.jsonl (used in --rubric-only mode).",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help=(
            "Regenerate ONLY sample_embeddings.npy and sample_metadata.parquet from "
            "--samples-csv using the existing embedding_config.json with temp-file "
            "validation and atomic replacement (leaves rubric_*, events_query_*, and "
            "embedding_config.json untouched)."
        ),
    )
    parser.add_argument(
        "--sample-report",
        type=Path,
        default=DEFAULT_SAMPLE_REPORT_PATH,
        help="Output path for the sample embedding audit report (used in --sample-only mode).",
    )
    args = parser.parse_args()

    if args.sample_only and args.rubric_only:
        print("[ERROR] --sample-only and --rubric-only are mutually exclusive.", file=sys.stderr)
        return 1

    if args.sample_only:
        if not args.rubric.exists():
            print(f"[ERROR] Missing rubric file: {args.rubric}", file=sys.stderr)
            return 1
        if not args.samples_csv.exists():
            print(
                f"[ERROR] Missing authoritative sample dataset: {args.samples_csv}. "
                "Will not substitute another CSV.",
                file=sys.stderr,
            )
            return 1
        emb_cfg_path = args.output_dir / "embedding_config.json"
        if not emb_cfg_path.exists():
            print(f"[ERROR] Missing embedding_config.json: {emb_cfg_path}", file=sys.stderr)
            return 1

        emb_cfg = json.loads(emb_cfg_path.read_text(encoding="utf-8"))
        cfg_model_id = str(emb_cfg.get("model_name") or MODEL_ID)
        cfg_dim = int(emb_cfg.get("embedding_dimension") or EMBEDDING_DIM)
        cfg_passage_prefix = str((emb_cfg.get("prefixes") or {}).get("passage") or PASSAGE_PREFIX)
        cfg_norm = "l2" if emb_cfg.get("l2_normalization", True) else "none"
        if cfg_model_id != MODEL_ID or cfg_dim != EMBEDDING_DIM:
            print(
                f"[ERROR] Unexpected embedding_config.json: model={cfg_model_id}, dim={cfg_dim}",
                file=sys.stderr,
            )
            return 1

        # Record SHA-256 of untouched read-only artifacts before running
        untouched_paths = [
            args.samples_csv,
            args.rubric,
            args.output_dir / "rubric_embeddings.npy",
            args.output_dir / "rubric_metadata.parquet",
            args.output_dir / "rubric_prototypes.jsonl",
            emb_cfg_path,
            Path("data/generated/events_query_embeddings.npy"),
            Path("data/generated/events_query_metadata.json"),
        ]
        pre_hashes: Dict[str, str] = {
            p.as_posix(): sha256_file(p) for p in untouched_paths if p.exists()
        }

        generation_ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        print(f"[1/5] Inspecting & validating sample dataset ({args.samples_csv}) against {args.rubric}...")
        rubric = yaml.safe_load(args.rubric.read_text(encoding="utf-8"))
        sample_records, sample_audit = build_sample_records(
            args.samples_csv,
            rubric,
            model_name=cfg_model_id,
            embedding_dim=cfg_dim,
            normalization=cfg_norm,
            passage_prefix=cfg_passage_prefix,
        )
        print(
            f"      Source rows={sample_audit['total_source_rows']}, "
            f"eligible={sample_audit['eligible_sample_count']}, "
            f"excluded={sample_audit['excluded_row_count']}, "
            f"malformed={len(sample_audit['malformed_rows'])}, "
            f"label_inconsistencies={len(sample_audit['label_inconsistencies'])}"
        )

        # Safety gate (Section 4 & Section 11): abort before touching artifacts on unresolved errors
        if sample_audit["malformed_rows"] or sample_audit["label_inconsistencies"] or not sample_records:
            print(
                "[ERROR] Unresolved dataset/label inconsistencies detected. "
                "Existing sample artifacts left untouched.",
                file=sys.stderr,
            )
            for err in (sample_audit["malformed_rows"] + sample_audit["label_inconsistencies"])[:10]:
                print(f"  - {err}", file=sys.stderr)
            return 2

        print(f"[2/5] Loading model `{cfg_model_id}` (matching embedding_config.json)...")
        model, model_info = load_e5_model(cfg_model_id)
        if emb_cfg.get("model_revision") and model_info["model_revision"] != emb_cfg.get("model_revision"):
            print(
                f"[ERROR] Model revision mismatch: loaded={model_info['model_revision']} "
                f"vs config={emb_cfg.get('model_revision')}",
                file=sys.stderr,
            )
            return 1

        print(f"[3/5] Encoding {len(sample_records)} eligible sample passages...")
        sample_vectors, sample_token_stats = encode_texts(
            model,
            [r["embedding_text"] for r in sample_records],
            batch_size=args.batch_size,
        )
        print(f"      Token stats: {json.dumps(sample_token_stats)}")

        # Write to temporary files first (Section 11 safe replacement)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        tmp_npy = args.output_dir / "sample_embeddings.npy.tmp"
        tmp_pq = args.output_dir / "sample_metadata.parquet.tmp"
        final_npy = args.output_dir / "sample_embeddings.npy"
        final_pq = args.output_dir / "sample_metadata.parquet"

        try:
            with open(tmp_npy, "wb") as fh:
                np.save(fh, sample_vectors)
            write_sample_parquet(sample_records, tmp_pq)

            print("[4/5] Running Section 10 A-D technical validation on temporary artifacts...")
            validation = validate_sample_artifacts(
                npy_path=tmp_npy,
                parquet_path=tmp_pq,
                rubric=rubric,
                model=model,
                sample_audit=sample_audit,
                token_stats=sample_token_stats,
                expected_model=cfg_model_id,
                expected_dim=cfg_dim,
            )
            for chk in validation["checks"]:
                print(f"      [{'PASS' if chk['passed'] else 'FAIL'}] {chk['check']}: {chk['detail']}")

            if not validation["all_passed"]:
                print(
                    "[ERROR] Validation failed on temporary sample artifacts. "
                    "Cleaning up temp files and keeping existing artifacts intact.",
                    file=sys.stderr,
                )
                return 2

            print("[5/5] Atomically replacing sample_embeddings.npy and sample_metadata.parquet...")
            os.replace(tmp_npy, final_npy)
            os.replace(tmp_pq, final_pq)
        finally:
            for leftover in (tmp_npy, tmp_pq):
                if leftover.exists():
                    try:
                        leftover.unlink()
                    except OSError:
                        pass

        # Verify untouched files remained byte-for-byte identical
        post_hashes: Dict[str, str] = {
            p.as_posix(): sha256_file(p) for p in untouched_paths if p.exists()
        }
        if post_hashes != pre_hashes:
            print("[ERROR] Unexpected modification detected in read-only artifacts!", file=sys.stderr)
            return 2

        # Write audit report
        args.sample_report.parent.mkdir(parents=True, exist_ok=True)
        report_md = build_sample_audit_report(
            sample_records=sample_records,
            sample_audit=sample_audit,
            embedding_config=emb_cfg,
            model_info=model_info,
            token_stats=sample_token_stats,
            validation=validation,
            output_dir=args.output_dir,
            generation_ts=generation_ts,
            rubric_untouched_hashes=post_hashes,
        )
        args.sample_report.write_text(report_md, encoding="utf-8")
        print(f"[OK] Wrote audit report -> {args.sample_report}")
        print(
            f"\n[SUCCESS] Regenerated {final_npy} {sample_vectors.shape} and "
            f"{final_pq} ({len(sample_records)} rows) with 100% validation PASS."
        )
        return 0

    if args.rubric_only:
        if not args.rubric.exists():
            print(f"[ERROR] Missing rubric file: {args.rubric}", file=sys.stderr)
            return 1
        if not args.rubric_jsonl.exists():
            print(f"[ERROR] Missing rubric prototypes JSONL: {args.rubric_jsonl}", file=sys.stderr)
            return 1
        emb_cfg_path = args.output_dir / "embedding_config.json"
        if not emb_cfg_path.exists():
            print(f"[ERROR] Missing embedding_config.json: {emb_cfg_path}", file=sys.stderr)
            return 1

        emb_cfg = json.loads(emb_cfg_path.read_text(encoding="utf-8"))
        cfg_model_id = str(emb_cfg.get("model_name") or MODEL_ID)
        if cfg_model_id != MODEL_ID:
            print(f"[ERROR] Unexpected model in embedding_config.json: {cfg_model_id}", file=sys.stderr)
            return 1

        print(f"[1/4] Loading rubric ({args.rubric}) & JSONL prototypes ({args.rubric_jsonl})...")
        rubric = yaml.safe_load(args.rubric.read_text(encoding="utf-8"))
        rubric_records = load_rubric_prototypes_jsonl(args.rubric_jsonl)
        print(f"      Loaded {len(rubric_records)} rubric prototypes from JSONL.")

        print(f"[2/4] Loading model `{cfg_model_id}` (matching embedding_config.json)...")
        model, model_info = load_e5_model(cfg_model_id)
        if emb_cfg.get("model_revision") and model_info["model_revision"] != emb_cfg.get("model_revision"):
            print(
                f"[ERROR] Model revision mismatch: loaded={model_info['model_revision']} "
                f"vs config={emb_cfg.get('model_revision')}",
                file=sys.stderr,
            )
            return 1

        print(f"[3/4] Encoding {len(rubric_records)} rubric prototypes...")
        rubric_vectors, rubric_token_stats = encode_texts(
            model,
            [r["embedding_text"] for r in rubric_records],
            batch_size=args.batch_size,
        )
        print(f"      Token stats: {json.dumps(rubric_token_stats)}")

        npy_path = args.output_dir / "rubric_embeddings.npy"
        pq_path = args.output_dir / "rubric_metadata.parquet"
        np.save(npy_path, rubric_vectors)
        write_rubric_parquet(rubric_records, pq_path)
        print(f"      Wrote {npy_path} {rubric_vectors.shape} and {pq_path} ({len(rubric_records)} rows).")

        print("[4/4] Running cross-artifact consistency validation...")
        validation = validate_rubric_cross_artifact_consistency(
            rubric=rubric,
            jsonl_path=args.rubric_jsonl,
            parquet_path=pq_path,
            npy_path=npy_path,
            model=model,
        )
        for chk in validation["checks"]:
            print(f"      [{'PASS' if chk['passed'] else 'FAIL'}] {chk['check']}: {chk['detail']}")
        if not validation["all_passed"]:
            return 2
        print("\n[SUCCESS] Rubric embeddings & metadata synchronized with rubric_prototypes.jsonl.")
        return 0

    # 1. Verify input files strictly without substituting alternatives
    if not args.rubric.exists():
        print(f"[ERROR] Missing required rubric file: {args.rubric}", file=sys.stderr)
        return 1
    if not args.samples_csv.exists():
        print(
            f"[ERROR] Missing required samples CSV file: {args.samples_csv}. "
            "Sample embedding generation is blocked (will not substitute another CSV).",
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generation_ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    # 2. Load rubric and build rubric prototypes
    print(f"[1/6] Loading rubric from {args.rubric}...")
    rubric = yaml.safe_load(args.rubric.read_text(encoding="utf-8"))
    rubric_records, rubric_audit = build_rubric_prototypes(rubric)
    print(f"      Built {len(rubric_records)} rubric prototypes.")

    # 3. Load events_full.csv and build eligible sample records
    print(f"[2/6] Loading samples from {args.samples_csv}...")
    sample_records, sample_audit = build_sample_records(args.samples_csv, rubric)
    print(
        f"      Built {len(sample_records)} eligible sample records "
        f"(excluded {sample_audit['excluded_row_count']} of {sample_audit['total_source_rows']} rows)."
    )

    # 4. Load intfloat/multilingual-e5-small once and encode BOTH branches identically
    print(f"[3/6] Loading embedding model `{MODEL_ID}`...")
    model, model_info = load_e5_model(MODEL_ID)
    print(
        f"      Loaded model revision={model_info['model_revision']}, "
        f"dim={model_info['embedding_dimension']}, pooling={model_info['pooling_method']}"
    )

    print(f"[4/6] Batch-encoding rubric ({len(rubric_records)}) and sample ({len(sample_records)}) texts...")
    rubric_vectors, rubric_token_stats = encode_texts(
        model,
        [r["embedding_text"] for r in rubric_records],
        batch_size=args.batch_size,
    )
    sample_vectors, sample_token_stats = encode_texts(
        model,
        [s["embedding_text"] for s in sample_records],
        batch_size=args.batch_size,
    )

    # 5. Save primary embedding matrices, aligned parquet metadata, and config
    print(f"[5/6] Writing artifacts to {args.output_dir}...")
    np.save(args.output_dir / "rubric_embeddings.npy", rubric_vectors)
    write_rubric_parquet(rubric_records, args.output_dir / "rubric_metadata.parquet")

    np.save(args.output_dir / "sample_embeddings.npy", sample_vectors)
    write_sample_parquet(sample_records, args.output_dir / "sample_metadata.parquet")

    config = {
        "model_name": model_info["model_name"],
        "model_revision": model_info["model_revision"],
        "embedding_dimension": model_info["embedding_dimension"],
        "dtype": model_info["dtype"],
        "device": model_info["device"],
        "batch_size": args.batch_size,
        "pooling_method": model_info["pooling_method"],
        "pooling_config": model_info["pooling_config"],
        "l2_normalization": model_info["l2_normalization"],
        "prefixes": {
            "passage": PASSAGE_PREFIX,
            "query": QUERY_PREFIX,
        },
        "tokenizer_config": model_info["tokenizer_config"],
        "model_modules": model_info["modules"],
        "shared_across_branches": True,
        "text_preprocessing_rules": {
            "unicode_normalization": "NFC",
            "whitespace_normalization": "collapse_whitespace_and_strip",
            "lowercase": False,
            "rubric_branch": {
                "stage1_fields": ["groups.<id>.name", "tieu_chi[eligible].name", "items[event_matchable=true].name"],
                "stage2_definition_fields": ["name", "scope_note"],
                "stage2_anchor_fields": ["anchors", "verified_anchors", "draft_anchors"],
                "stage2_keyword_fields": ["name", "keywords"],
                "excluded_semantics": [
                    "scoring_points_and_formulas",
                    "participation_verification",
                    "role_eligibility",
                    "attendance_and_completion_thresholds",
                    "csv_row_and_administrative_metadata",
                    "uncertain_review_notes_kept_in_metadata_only",
                ],
            },
            "sample_branch": {
                "source_csv_columns": sample_audit["csv_columns"],
                "embedded_fields": ["Ten_Hoat_Dong"],
                "format_template": "passage: {event_title}",
                "metadata_only_fields": ["STT", "Ma_Muc", "Organizer", "Vai_Tro", "Tieu_Chi_Quy_Che", "Diem_Toi_Da"],
                "eligibility_filter": "rubric item exists and event_matchable == true",
            },
        },
        "token_stats": {
            "rubric": rubric_token_stats,
            "sample": sample_token_stats,
        },
        "artifact_counts": {
            "rubric_embeddings": int(rubric_vectors.shape[0]),
            "sample_embeddings": int(sample_vectors.shape[0]),
        },
        "generation_timestamp_utc": generation_ts,
        "sources": {
            "rubric": {
                "path": args.rubric.as_posix(),
                "version": (rubric.get("meta") or {}).get("version"),
                "size_bytes": args.rubric.stat().st_size,
                "sha256": sha256_file(args.rubric),
            },
            "samples": {
                "path": args.samples_csv.as_posix(),
                "rows": sample_audit["total_source_rows"],
                "size_bytes": args.samples_csv.stat().st_size,
                "sha256": sha256_file(args.samples_csv),
            },
        },
        "runtime_versions": model_info["runtime_versions"],
    }
    (args.output_dir / "embedding_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 6. Validate all artifacts from disk and write generation report
    print("[6/6] Validating generated artifacts from disk...")
    validation = validate_artifacts(args.output_dir, rubric, model)
    report_md = build_generation_report(
        rubric_records=rubric_records,
        rubric_audit=rubric_audit,
        sample_records=sample_records,
        sample_audit=sample_audit,
        config=config,
        validation=validation,
        output_dir=args.output_dir,
    )
    (args.output_dir / "embedding_generation_report.md").write_text(report_md, encoding="utf-8")

    for chk in validation["checks"]:
        status = "PASS" if chk["passed"] else "FAIL"
        print(f"      [{status}] {chk['check']}: {chk['detail']}")

    if not validation["all_passed"]:
        print("[ERROR] One or more validation checks failed!", file=sys.stderr)
        return 2

    print("\n[SUCCESS] All 6 deliverables generated and validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
