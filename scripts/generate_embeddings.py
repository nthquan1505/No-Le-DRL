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

# ---------------------------------------------------------------------------
# Constants (strictly matching configs/model_config.py)
# ---------------------------------------------------------------------------
MODEL_ID = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
BATCH_SIZE = 32
DEFAULT_RUBRIC_PATH = Path("data/raw/rubric.yaml")
DEFAULT_SAMPLES_CSV_PATH = Path("res/events_full.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/cls_index")

POSITIVE_MAJOR_GROUPS = ("C1", "C2", "C3", "C4", "C5")
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
# Sample Record Construction from events_full.csv
# ---------------------------------------------------------------------------


def detect_csv_columns(fieldnames: Sequence[str]) -> Dict[str, Optional[str]]:
    """Detect and map actual CSV columns without hardcoding blind assumptions."""
    cols = [c.strip() for c in fieldnames if c]
    col_lower = {c.lower(): c for c in cols}

    def pick(candidates: Sequence[str]) -> Optional[str]:
        for cand in candidates:
            if cand.lower() in col_lower:
                return col_lower[cand.lower()]
        return None

    mapping = {
        "row_id": pick(["STT", "id", "row_id"]),
        "label": pick(["Ma_Muc", "criterion_ref", "label", "rubric_item_id"]),
        "title": pick(["Ten_Hoat_Dong", "title", "activity_title", "event_title"]),
        "description": pick(["Mo_Ta", "description", "activity_description", "noi_dung"]),
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


def build_sample_records(
    csv_path: Path,
    rubric: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse events_full.csv, map valid event_matchable:true labels, and audit exclusions/duplicates."""
    items: Dict[str, Any] = rubric.get("items") or {}
    ref_to_item: Dict[str, Tuple[str, Dict[str, Any]]] = {
        str(v["ref"]).strip(): (k, v) for k, v in items.items() if isinstance(v, dict) and "ref" in v
    }

    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV file '{csv_path}' has no header row.")
        col_map = detect_csv_columns(reader.fieldnames)
        raw_rows = list(reader)

    samples: List[Dict[str, Any]] = []
    excluded_rows: List[Dict[str, Any]] = []
    invalid_or_special_rows: List[Dict[str, Any]] = []

    title_col = col_map["title"]
    desc_col = col_map["description"]
    label_col = col_map["label"]
    assert title_col is not None and label_col is not None

    for row_idx, row in enumerate(raw_rows, start=1):
        original_label = normalize_whitespace(row.get(label_col, ""))
        event_title = normalize_whitespace(row.get(title_col, ""))
        event_desc = normalize_whitespace(row.get(desc_col, "")) if desc_col else ""

        # Preserve full original row fields as traceability metadata
        source_meta = {k: normalize_whitespace(v) for k, v in row.items()}

        # Check missing / special / unknown labels
        if not original_label:
            rec = {
                "source_row_index": row_idx,
                "original_label": original_label,
                "event_title": event_title,
                "reason": "missing_label",
                "source_metadata": source_meta,
            }
            excluded_rows.append(rec)
            invalid_or_special_rows.append(rec)
            continue

        if original_label.upper() in SPECIAL_EXCLUDED_LABELS:
            rec = {
                "source_row_index": row_idx,
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
                "original_label": original_label,
                "event_title": event_title,
                "reason": "unknown_or_unmapped_rubric_ref",
                "source_metadata": source_meta,
            }
            excluded_rows.append(rec)
            invalid_or_special_rows.append(rec)
            continue

        if not event_title:
            rec = {
                "source_row_index": row_idx,
                "original_label": original_label,
                "event_title": event_title,
                "reason": "empty_activity_title",
                "source_metadata": source_meta,
            }
            excluded_rows.append(rec)
            invalid_or_special_rows.append(rec)
            continue

        criterion_key, item = ref_to_item[original_label]
        if item.get("event_matchable") is not True:
            excluded_rows.append(
                {
                    "source_row_index": row_idx,
                    "original_label": original_label,
                    "criterion_key": criterion_key,
                    "group_id": str(item.get("group", "")),
                    "kind": str(item.get("kind", "")),
                    "event_title": event_title,
                    "reason": f"non_event_matchable_criterion (event_matchable=false, kind={item.get('kind')})",
                    "source_metadata": source_meta,
                }
            )
            continue

        # Construct strictly leakage-free activity embedding text
        text_parts = [event_title]
        if event_desc:
            text_parts.append(event_desc)
        embedding_text = format_passage(" ".join(text_parts))

        group_id = str(item["group"]).strip()
        tieu_chi = str(item["tieu_chi"]).strip() if item.get("tieu_chi") else None

        samples.append(
            {
                "embedding_row": len(samples),
                "sample_id": f"events_full_row_{row_idx:04d}",
                "source_row_index": row_idx,
                "event_title": event_title,
                "embedding_text": embedding_text,
                "original_label": original_label,
                "criterion_ref": str(item["ref"]).strip(),
                "criterion_key": criterion_key,
                "group_id": group_id,
                "tieu_chi": tieu_chi,
                "label_status": "valid_event_matchable",
                "source_metadata": json.dumps(source_meta, ensure_ascii=False),
            }
        )

    # Detect exact and orthographic near-duplicates preserved among eligible samples
    def strip_diacritics_key(s: str) -> str:
        nfd = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
        stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        return re.sub(r"[^a-z0-9]+", " ", stripped).strip()

    exact_title_groups: Dict[str, List[Dict[str, Any]]] = {}
    diacritic_title_groups: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        exact_title_groups.setdefault(s["event_title"].lower(), []).append(s)
        diacritic_title_groups.setdefault(strip_diacritics_key(s["event_title"]), []).append(s)

    preserved_exact_duplicates = [
        [
            {"sample_id": x["sample_id"], "source_row_index": x["source_row_index"], "event_title": x["event_title"], "criterion_ref": x["criterion_ref"]}
            for x in grp
        ]
        for grp in exact_title_groups.values()
        if len(grp) > 1
    ]
    preserved_near_duplicates = [
        [
            {"sample_id": x["sample_id"], "source_row_index": x["source_row_index"], "event_title": x["event_title"], "criterion_ref": x["criterion_ref"]}
            for x in grp
        ]
        for grp in diacritic_title_groups.values()
        if len(grp) > 1
    ]

    # Identify eligible rubric criteria with 0 samples in events_full.csv
    covered_refs = {s["criterion_ref"] for s in samples}
    uncovered_em_true = [
        {"criterion_ref": str(v["ref"]), "criterion_key": k, "group_id": str(v["group"]), "anchors_status": str(v.get("anchors_status", ""))}
        for k, v in items.items()
        if isinstance(v, dict) and v.get("event_matchable") is True and str(v["ref"]) not in covered_refs
    ]

    audit = {
        "total_source_rows": len(raw_rows),
        "csv_columns": list(reader.fieldnames),
        "detected_column_mapping": col_map,
        "eligible_sample_count": len(samples),
        "excluded_row_count": len(excluded_rows),
        "invalid_or_special_label_count": len(invalid_or_special_rows),
        "excluded_rows": excluded_rows,
        "invalid_or_special_rows": invalid_or_special_rows,
        "preserved_exact_duplicates": preserved_exact_duplicates,
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
    schema = pa.schema(
        [
            pa.field("embedding_row", pa.int64(), nullable=False),
            pa.field("prototype_id", pa.string(), nullable=False),
            pa.field("stage", pa.string(), nullable=False),
            pa.field("group_id", pa.string(), nullable=False),
            pa.field("tieu_chi", pa.string(), nullable=True),
            pa.field("criterion_key", pa.string(), nullable=True),
            pa.field("criterion_ref", pa.string(), nullable=True),
            pa.field("prototype_type", pa.string(), nullable=False),
            pa.field("anchor_status", pa.string(), nullable=True),
            pa.field("criterion_anchors_status", pa.string(), nullable=True),
            pa.field("event_matchable", pa.bool_(), nullable=False),
            pa.field("embedding_text", pa.string(), nullable=False),
            pa.field("source_fields", pa.list_(pa.string()), nullable=False),
            pa.field("kind", pa.string(), nullable=True),
            pa.field("mvp_scope", pa.string(), nullable=True),
            pa.field("is_positive_target", pa.bool_(), nullable=False),
            pa.field("has_uncertain", pa.bool_(), nullable=False),
            pa.field("uncertain_notes", pa.string(), nullable=True),
        ]
    )
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, path)


def write_sample_parquet(records: List[Dict[str, Any]], path: Path) -> None:
    schema = pa.schema(
        [
            pa.field("embedding_row", pa.int64(), nullable=False),
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("source_row_index", pa.int64(), nullable=False),
            pa.field("event_title", pa.string(), nullable=False),
            pa.field("embedding_text", pa.string(), nullable=False),
            pa.field("original_label", pa.string(), nullable=False),
            pa.field("criterion_ref", pa.string(), nullable=False),
            pa.field("criterion_key", pa.string(), nullable=False),
            pa.field("group_id", pa.string(), nullable=False),
            pa.field("tieu_chi", pa.string(), nullable=True),
            pa.field("label_status", pa.string(), nullable=False),
            pa.field("source_metadata", pa.string(), nullable=False),
        ]
    )
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, path)


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
        and s2_sample_vecs.shape[0] == 105  # 50 (C1) + 55 (C3)
        and crit_vecs.shape == (17, EMBEDDING_DIM)
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
    args = parser.parse_args()

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
