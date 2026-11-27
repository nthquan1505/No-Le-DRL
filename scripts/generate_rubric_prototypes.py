#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_rubric_prototypes.py (Schema v2.0)
===========================================
Authoritative Rubric Prototype Generator for the Rubric Branch of the HCMUS DRL
Hierarchical Two-Stage kNN Classification System.

Reads `data/raw/rubric.yaml` (single source of truth; NEVER modified) and outputs:
  - data/processed/cls_index/rubric_prototypes.jsonl  (Schema v2.0, E5 passage-ready)
  - data/processed/cls_index/rubric_prototypes.yaml   (human-readable synchronized mirror)

Key Schema v2.0 improvements:
  1. Stage 1 Group Prototypes (C1-C4): complete, structured, untruncated Vietnamese
     text derived strictly from `groups`, eligible `tieu_chi`, and `event_matchable: true`
     criteria (C5 excluded because it has 0 event_matchable:true criteria).
  2. Stage 2 Criterion Prototypes (20 criteria in C1-C4):
     - `criterion_definition`: structured positive scope selectively combining `name`,
       cleaned `scope_note`, cleaned `inclusion_rules`, positive `conditions`, and `min_level`.
     - `verified_anchor`: preserves every school-verified anchor with `anchor_text` &
       `source_anchor_index`; contextualizes 3.3 verified anchors confirmed with role
       `Huy động lực lượng` in `school_data.roles_seen`.
     - `draft_anchor`: clearly separated from verified anchors (`anchor_status="draft"`).
     - `keyword_context`: max 1 per criterion from `keywords` only (`negative_keywords` excluded).
  3. Separate `boundary_metadata` layer (`inclusion_rules`, `exclusion_rules`,
     `confusable_with`, `negative_keywords`, `conditions`, `min_level`, `role_context`,
     `requires_role_context`, `uncertain`) stored strictly in metadata, never mixed into
     positive voting embeddings.
  4. Special cases:
     - 3.3: mobilization-contextualized verified anchors + `required_signals`.
     - 3.8: distinguishes individual anchor `anchor_status` (`verified_from_school_list` /
       `draft`) from criterion-level `criterion_anchor_status` (`verified_partial`).
     - 6.4: separated into `stage="reminder"`, `eligibility_class="reminder_only"`,
       `prototype_type="reminder_definition"` (plus reminder anchors/keyword_context).
     - NONE: separated into `stage="special"`, `prototype_type="draft_negative_reference"`,
       `eligibility_class="negative_reference_only"`, `can_decide_none=False`.
     - NEEDS_REVIEW: routing status only; zero embedding prototypes generated.
  5. Explicit `eligibility_class` and `prototype_family` metadata for unbiased retrieval.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "2.0"
PASSAGE_PREFIX = "passage: "
POSITIVE_MAJOR_GROUPS = ("C1", "C2", "C3", "C4")

DEFAULT_RUBRIC_PATH = Path("data/raw/rubric.yaml")
DEFAULT_OUT_DIR = Path("data/processed/cls_index")

MIN_LEVEL_VI: Dict[str, str] = {
    "lop": "từ cấp Lớp trở lên",
    "khoa": "từ cấp Khoa trở lên",
    "truong": "từ cấp Trường trở lên",
    "dhqg": "từ cấp ĐHQG-HCM trở lên",
    "thanh_pho": "từ cấp Thành phố trở lên",
    "quoc_gia": "từ cấp Quốc gia trở lên",
    "quoc_te": "từ cấp Quốc tế",
}


# ---------------------------------------------------------------------------
# Deterministic Text Cleaning Helpers (Zero Fabrication)
# ---------------------------------------------------------------------------


def normalize_text(text: Any) -> str:
    """Normalize Unicode NFC and collapse internal horizontal whitespace."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_single_line(text: Any) -> str:
    """Normalize Unicode NFC and collapse all whitespace to single spaces."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", s).strip()


def clean_key(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(s))


def ensure_passage_prefix(text: str) -> str:
    """Add `passage: ` prefix exactly once (never double-prefix)."""
    t = normalize_text(text)
    if t.startswith(PASSAGE_PREFIX):
        return t
    return f"{PASSAGE_PREFIX}{t}"


def clean_criterion_name(raw_name: str, criterion_ref: str) -> str:
    """Clean personal completion/attendance thresholds & cross-ref parentheticals from name.

    Preserves authentic Vietnamese rubric terminology without inventing wording.
    """
    text = normalize_single_line(raw_name)
    text = re.sub(r"\s*\(hoàn tất vòng đầu tiên[^)]*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(trừ nội dung\s+\d+\.\d+\)", "", text, flags=re.IGNORECASE)
    if criterion_ref == "2.2":
        text = re.sub(r"\btrên\s+50%\s+số\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(
            r"\s*\((?:mọi sinh viên|sinh viên năm cuối)\)",
            "",
            text,
            flags=re.IGNORECASE,
        )
    if criterion_ref == "6.4":
        text = re.sub(r"^Không tham gia đầy đủ\s+", "", text, flags=re.IGNORECASE)
        if text:
            text = text[0].upper() + text[1:]
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return normalize_single_line(text).rstrip(".")


def clean_scope_or_rule_text(raw_text: str) -> str:
    """Remove scoring points, CSV row refs, QĐ 2020 meta-commentary, and cross-ref notes."""
    text = normalize_single_line(raw_text)
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept: List[str] = []
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        if re.search(r"events_full\.csv", s, flags=re.IGNORECASE):
            continue
        if re.search(r"^Điểm phụ thuộc thời lượng", s, flags=re.IGNORECASE):
            continue
        if re.search(r"^Cần phân biệt cẩn thận với\b", s, flags=re.IGNORECASE):
            continue
        kept.append(s)
    text = " ".join(kept)

    # Strip scoring points, CSV line numbers, and administrative references
    text = re.sub(r":\s*cấp trường trở lên[^.]*=\s*\d+đ[^.]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r";\s*điểm chỉ tính khi[^.]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(mức phẳng\s*\d+đ\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(\d+đ(?:/[^)]*)?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(\d+/\d+/\d+\)", "", text)
    text = re.sub(r",?\s*vd\.?\s*dòng\s+\d+(?:[–\-]\d+)?(?:\s*—[^)]*)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(dòng\s+\d+(?:[–\-]\d+)?[^)]*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r",?\s*dòng\s+\d+(?:[–\-]\d+)?(?:\s*—[^)]*)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(khác với[^)]*\d+\.\d+[^)]*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r";\s*phân biệt với[^.]*\(mục\s+\d+\.\d+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Theo QĐ 2020 là\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Thực tế 2025[–\-]26:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*được xếp vào\s+\d+\.\d+\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Chỉ tính khi\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r",?\s*\bvd\.?(?=\s*\))", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(\s*\)", "", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = normalize_single_line(text).rstrip(".;:")
    if text:
        text = text[0].upper() + text[1:]
    return text


# ---------------------------------------------------------------------------
# Eligibility Classification & Boundary Metadata Helpers (Sections 5 & 7)
# ---------------------------------------------------------------------------


def determine_eligibility_class(
    *,
    event_matchable: bool,
    group_id: Optional[str],
    mvp_scope: Optional[str],
    is_none_label: bool = False,
) -> str:
    """Determine prototype retrieval eligibility from event_matchable, group, kind, mvp_scope."""
    if is_none_label:
        return "negative_reference_only"
    if mvp_scope == "remind":
        return "reminder_only"
    if not event_matchable:
        return "not_event_matchable"
    if group_id in POSITIVE_MAJOR_GROUPS:
        if mvp_scope == "recommend_apply":
            return "conditional_candidate"
        if mvp_scope == "recommend":
            return "positive_candidate"
    return "not_event_matchable"


def extract_boundary_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """Copy structured classification boundary fields verbatim from rubric.yaml (zero inference)."""
    return {
        "inclusion_rules": [normalize_single_line(x) for x in (item.get("inclusion_rules") or [])],
        "exclusion_rules": [normalize_single_line(x) for x in (item.get("exclusion_rules") or [])],
        "confusable_with": list(item.get("confusable_with") or []),
        "negative_keywords": [normalize_single_line(x) for x in (item.get("negative_keywords") or [])],
        "conditions": [normalize_single_line(x) for x in (item.get("conditions") or [])],
        "min_level": item.get("min_level"),
        "role_context": list(item.get("role_context") or []),
        "requires_role_context": bool(item.get("requires_role_context", False)),
        "uncertain": [normalize_single_line(x) for x in (item.get("uncertain") or [])],
    }


def extract_required_signals(ref: str, item: Dict[str, Any]) -> List[str]:
    """Record explicit rubric-mandated signals (e.g., 3.3 mobilization, min_level, selection)."""
    signals: List[str] = []
    if ref == "3.3":
        signals.append("explicit_student_mobilization_signal")
    if item.get("min_level"):
        signals.append(f"min_level:{item['min_level']}")
    if item.get("mvp_scope") == "recommend_apply" or item.get("points_condition"):
        signals.append("selection_or_completion_required")
    if item.get("requires_role_context"):
        signals.append("requires_role_context")
    return signals


def match_evidence_example(
    anchor: str,
    school_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return exact, unambiguous matching record from school_data.evidence_examples if present."""
    examples = school_data.get("evidence_examples") or []
    norm_anchor = normalize_single_line(anchor).lower()
    matches = [
        ex
        for ex in examples
        if isinstance(ex, dict) and normalize_single_line(ex.get("title", "")).lower() == norm_anchor
    ]
    return matches[0] if len(matches) == 1 else None


def all_school_rows_have_mobilization_role(school_data: Dict[str, Any]) -> bool:
    """Confirm from rubric.yaml school_data that 100% of recorded rows carry 'Huy động lực lượng'."""
    rows = int(school_data.get("rows") or 0)
    roles_seen = school_data.get("roles_seen") or {}
    if rows <= 0 or not isinstance(roles_seen, dict):
        return False
    return len(roles_seen) == 1 and int(roles_seen.get("Huy động lực lượng", 0)) == rows


# ---------------------------------------------------------------------------
# Stage 1 & Stage 2 Text Builders (Section 4 & Section 6)
# ---------------------------------------------------------------------------


def build_stage1_group_text(
    grp_id: str,
    grp_meta: Dict[str, Any],
    eligible_items: List[Tuple[str, Dict[str, Any]]],
    tieu_chi_map: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """Build complete, structured, untruncated Stage 1 Group Prototype text strictly from rubric.yaml."""
    grp_name = normalize_single_line(grp_meta.get("name", grp_id))

    # Collect unique eligible tieu_chi descriptions belonging to event_matchable:true criteria
    tc_names: List[str] = []
    for _, it in eligible_items:
        tc = it.get("tieu_chi")
        if tc and tc in tieu_chi_map:
            tc_desc = normalize_single_line(tieu_chi_map[tc].get("name", ""))
            if tc_desc and tc_desc not in tc_names:
                tc_names.append(tc_desc)

    scope_line = "; ".join(tc_names) if tc_names else grp_name

    # Concise, complete, untruncated bullet per eligible criterion (no mid-word slicing!)
    # For large groups (e.g. C3 with 11 criteria), prefer shorter cleaned scope_note and strip
    # parenthetical example lists so the Stage 1 group prototype stays complete and < 512 tokens.
    large_group = len(eligible_items) >= 8
    topic_bullets: List[str] = []
    used_scope_notes = False
    for _, it in eligible_items:
        ref = str(it.get("ref", ""))
        cname = clean_criterion_name(str(it.get("name", "")), ref)
        bullet = cname
        if large_group and ref != "3.3":
            cscope = clean_scope_or_rule_text(str(it.get("scope_note") or ""))
            if cscope and len(cscope) < len(cname):
                bullet = cscope
                used_scope_notes = True
            bullet = normalize_single_line(re.sub(r"\s*\([^)]*\)", "", bullet)).rstrip(".,;:")
        if bullet:
            topic_bullets.append(f"- {bullet}")

    # Collect representative positive keywords from eligible criteria to distinguish group scope
    # Cap per-criterion sample to keep all 4 group prototypes balanced and strictly below 512 tokens
    if len(eligible_items) <= 4:
        per_item_kw_limit = 4
    elif len(eligible_items) >= 8:
        per_item_kw_limit = 1
    else:
        per_item_kw_limit = 2
    distinguishing_kws: List[str] = []
    seen_kw = set()
    for _, it in eligible_items:
        added = 0
        for kw in it.get("keywords") or []:
            kw_clean = normalize_single_line(kw)
            # Deduplicate Vietnamese tone-placement variants (e.g. "khỏe" vs "khoẻ")
            kw_low = kw_clean.lower().replace("oẻ", "ỏe").replace("oà", "òa").replace("uỳ", "ùy")
            if kw_clean and kw_low not in seen_kw:
                seen_kw.add(kw_low)
                distinguishing_kws.append(kw_clean)
                added += 1
                if added >= per_item_kw_limit:
                    break

    blocks: List[str] = [
        f"Nhóm {grp_id} - {grp_name}.",
        f"Phạm vi:\n{scope_line}.",
        "Các loại hoạt động tiêu biểu:\n" + "\n".join(topic_bullets),
    ]
    source_fields = [
        f"groups.{grp_id}.name",
        "tieu_chi[eligible].name",
        "items[event_matchable=true].name",
    ]
    if used_scope_notes:
        source_fields.append("items[event_matchable=true].scope_note")
    if distinguishing_kws:
        blocks.append(
            "Đặc điểm nhận diện:\n"
            + f"Từ khóa tiêu biểu: {', '.join(distinguishing_kws)}."
        )
        source_fields.append("items[event_matchable=true].keywords")

    text = ensure_passage_prefix("\n\n".join(blocks))
    return text, source_fields


def build_criterion_definition_text(
    ref: str,
    item: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """Build focused, structured positive `criterion_definition` embedding text from rubric.yaml."""
    cleaned_name = clean_criterion_name(str(item.get("name", "")), ref)
    cleaned_scope = clean_scope_or_rule_text(str(item.get("scope_note") or ""))
    raw_inclusions = item.get("inclusion_rules") or []
    cleaned_inclusions = [
        c for c in (clean_scope_or_rule_text(r) for r in raw_inclusions) if c
    ]
    raw_conditions = [
        normalize_single_line(c)
        for c in (item.get("conditions") or [])
        if normalize_single_line(c)
    ]
    min_level = item.get("min_level")

    blocks: List[str] = [f"Tiêu chí {ref} - {cleaned_name}."]
    source_fields: List[str] = ["name"]

    if cleaned_scope:
        blocks.append(f"Phạm vi:\n{cleaned_scope}.")
        source_fields.append("scope_note")

    if cleaned_inclusions:
        inc_lines = "\n".join(f"- {rule}." for rule in cleaned_inclusions)
        blocks.append(f"Các hoạt động thuộc tiêu chí:\n{inc_lines}")
        source_fields.append("inclusion_rules")

    # Positive identification conditions (level, positive conditions, explicit 3.3 signal)
    cond_lines: List[str] = []
    if min_level and str(min_level) in MIN_LEVEL_VI:
        cond_lines.append(f"- Quy mô tổ chức {MIN_LEVEL_VI[str(min_level)]}.")
        source_fields.append("min_level")
    for cond in raw_conditions:
        # Avoid mixing personal completion/grading formulas into pure event meaning unless positive scope
        if re.search(r"trên\s+50%|hoàn tất vòng đầu", cond, flags=re.IGNORECASE):
            continue
        cond_lines.append(f"- {cond.rstrip('.')}.")
        if "conditions" not in source_fields:
            source_fields.append("conditions")
    if ref == "3.3":
        cond_lines.append(
            "- Bài đăng hoặc thông báo phải có tín hiệu huy động sinh viên tham dự, hỗ trợ hoặc cổ vũ theo phân công."
        )

    if cond_lines:
        blocks.append("Điều kiện nhận diện:\n" + "\n".join(cond_lines))

    return ensure_passage_prefix("\n\n".join(blocks)), source_fields


def build_mobilization_anchor_text(anchor_title: str) -> str:
    """Contextualized embedding text for 3.3 verified anchors confirmed with role 'Huy động lực lượng'."""
    clean_title = normalize_single_line(anchor_title)
    body = (
        f"Sự kiện: {clean_title}.\n\n"
        "Ngữ cảnh phân loại:\n"
        "Sự kiện này được ghi nhận ở mục 3.3 vì sinh viên được huy động tham dự, cổ vũ hoặc hỗ trợ.\n\n"
        "Tín hiệu nhận diện:\n"
        "Thông báo có nội dung huy động sinh viên tham dự, hỗ trợ hoặc cổ vũ theo phân công."
    )
    return ensure_passage_prefix(body)


def build_keyword_context_text(cleaned_name: str, keywords: Sequence[str]) -> str:
    """Structured positive keyword context text (never includes negative_keywords)."""
    kws = [normalize_single_line(k) for k in keywords if normalize_single_line(k)]
    body = f"{cleaned_name}.\n\nTừ khóa nhận diện:\n{', '.join(kws)}."
    return ensure_passage_prefix(body)


# ---------------------------------------------------------------------------
# Prototype Record Factory & Main Generation Logic
# ---------------------------------------------------------------------------


def make_prototype_record(
    *,
    prototype_id: str,
    stage: str,
    group_id: Optional[str],
    tieu_chi: Optional[str],
    criterion_key: Optional[str],
    criterion_ref: Optional[str],
    prototype_type: str,
    prototype_family: str,
    embedding_text: str,
    anchor_status: Optional[str],
    criterion_anchor_status: Optional[str],
    anchor_text: Optional[str],
    source_anchor_index: Optional[int],
    source_fields: List[str],
    event_matchable: bool,
    eligibility_class: str,
    boundary_metadata: Dict[str, Any],
    required_signals: List[str],
    negative_keywords: List[str],
    mvp_scope: Optional[str],
    kind: Optional[str],
    can_decide_none: Optional[bool] = None,
    notes: str = "",
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "prototype_id": prototype_id,
        "stage": stage,
        "group_id": group_id,
        "tieu_chi": tieu_chi,
        "criterion_key": criterion_key,
        "criterion_ref": criterion_ref,
        "prototype_type": prototype_type,
        "prototype_family": prototype_family,
        "embedding_text": embedding_text,
        "anchor_status": anchor_status,
        "criterion_anchor_status": criterion_anchor_status,
        "anchor_text": anchor_text,
        "source_anchor_index": source_anchor_index,
        "source_fields": source_fields,
        "event_matchable": bool(event_matchable),
        "eligibility_class": eligibility_class,
        "boundary_metadata": boundary_metadata,
        "required_signals": required_signals,
        "negative_keywords": negative_keywords,
        "mvp_scope": mvp_scope,
        "kind": kind,
        "notes": notes,
    }
    if can_decide_none is not None:
        rec["can_decide_none"] = bool(can_decide_none)
    return rec


def generate_prototypes(rubric: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate complete Schema v2.0 rubric prototypes strictly from `rubric.yaml`."""
    groups: Dict[str, Any] = rubric.get("groups") or {}
    tieu_chi_map: Dict[str, Any] = rubric.get("tieu_chi") or {}
    items: Dict[str, Any] = rubric.get("items") or {}
    meta_labels: Dict[str, Any] = rubric.get("meta_labels") or {}

    prototypes: List[Dict[str, Any]] = []
    seen_ids: set = set()

    def unique_id(raw_id: str) -> str:
        if raw_id not in seen_ids:
            seen_ids.add(raw_id)
            return raw_id
        n = 2
        while f"{raw_id}_{n}" in seen_ids:
            n += 1
        uid = f"{raw_id}_{n}"
        seen_ids.add(uid)
        return uid

    # Partition rubric items
    group_eligible_items: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {
        g: [] for g in POSITIVE_MAJOR_GROUPS
    }
    em_true_c1_c4: List[Tuple[str, Dict[str, Any]]] = []
    reminder_items: List[Tuple[str, Dict[str, Any]]] = []
    excluded_items: List[Tuple[str, Dict[str, Any]]] = []

    for item_key, item in items.items():
        if not isinstance(item, dict):
            continue
        em = item.get("event_matchable") is True
        grp = str(item.get("group", "")).strip()
        mvp_scope = str(item.get("mvp_scope") or "").strip()

        if em and grp in POSITIVE_MAJOR_GROUPS and mvp_scope in ("recommend", "recommend_apply"):
            group_eligible_items[grp].append((item_key, item))
            em_true_c1_c4.append((item_key, item))
        elif em and mvp_scope == "remind":
            reminder_items.append((item_key, item))
        else:
            excluded_items.append((item_key, item))

    # ------------------------------------------------------------------
    # 1. Stage 1: Group Prototypes (exactly 1 per eligible group C1-C4)
    # ------------------------------------------------------------------
    for grp_id in POSITIVE_MAJOR_GROUPS:
        grp_items = group_eligible_items.get(grp_id) or []
        if not grp_items:
            continue
        grp_meta = groups.get(grp_id) or {}
        emb_text, sfields = build_stage1_group_text(grp_id, grp_meta, grp_items, tieu_chi_map)
        prototypes.append(
            make_prototype_record(
                prototype_id=unique_id(f"stage1_{grp_id}_group_def"),
                stage="group",
                group_id=grp_id,
                tieu_chi=None,
                criterion_key=None,
                criterion_ref=None,
                prototype_type="group_definition",
                prototype_family="group_definition",
                embedding_text=emb_text,
                anchor_status="derived",
                criterion_anchor_status=None,
                anchor_text=None,
                source_anchor_index=None,
                source_fields=sfields,
                event_matchable=True,
                eligibility_class="positive_candidate",
                boundary_metadata={},
                required_signals=[],
                negative_keywords=[],
                mvp_scope=None,
                kind="major_group",
                notes=f"Stage 1 structured group prototype for {grp_id} (covers {len(grp_items)} event_matchable:true criteria)",
            )
        )

    # ------------------------------------------------------------------
    # 2. Stage 2: Criterion Prototypes (20 positive/conditional items in C1-C4)
    # ------------------------------------------------------------------
    for item_key, item in em_true_c1_c4:
        ref = str(item["ref"]).strip()
        grp = str(item["group"]).strip()
        tc = str(item["tieu_chi"]).strip() if item.get("tieu_chi") else None
        kind = str(item.get("kind") or "").strip()
        mvp_scope = str(item.get("mvp_scope") or "").strip()
        crit_anchor_status = str(item.get("anchors_status") or "").strip()
        elig_class = determine_eligibility_class(
            event_matchable=True,
            group_id=grp,
            mvp_scope=mvp_scope,
        )
        boundary_meta = extract_boundary_metadata(item)
        req_signals = extract_required_signals(ref, item)
        neg_kws = list(boundary_meta["negative_keywords"])
        school_data = item.get("school_data") or {}
        ref_safe = clean_key(ref)
        cleaned_name = clean_criterion_name(str(item.get("name", "")), ref)

        # A. criterion_definition
        def_text, def_fields = build_criterion_definition_text(ref, item)
        prototypes.append(
            make_prototype_record(
                prototype_id=unique_id(f"stage2_{ref_safe}_def"),
                stage="criterion",
                group_id=grp,
                tieu_chi=tc,
                criterion_key=item_key,
                criterion_ref=ref,
                prototype_type="criterion_definition",
                prototype_family="definition",
                embedding_text=def_text,
                anchor_status="derived",
                criterion_anchor_status=crit_anchor_status,
                anchor_text=None,
                source_anchor_index=None,
                source_fields=def_fields,
                event_matchable=True,
                eligibility_class=elig_class,
                boundary_metadata=boundary_meta,
                required_signals=req_signals,
                negative_keywords=neg_kws,
                mvp_scope=mvp_scope,
                kind=kind,
                notes="Structured positive definition prototype derived from rubric.yaml",
            )
        )

        # B. verified_anchor & C. draft_anchor (strictly separated; never promote draft)
        anchors_list = item.get("anchors") or []
        verified_anchors = item.get("verified_anchors") or []
        draft_anchors = item.get("draft_anchors") or []

        is_3_3_mobilized = ref == "3.3" and all_school_rows_have_mobilization_role(school_data)

        if crit_anchor_status == "verified_from_school_list":
            for idx, raw_anchor in enumerate(anchors_list, start=1):
                a_clean = normalize_single_line(raw_anchor)
                if not a_clean:
                    continue
                if is_3_3_mobilized:
                    emb_text = build_mobilization_anchor_text(a_clean)
                    a_fields = ["anchors", "school_data.roles_seen", "scope_note", "inclusion_rules"]
                    a_note = "Contextualized 3.3 verified anchor confirmed with role 'Huy động lực lượng' in school_data"
                else:
                    ex = match_evidence_example(a_clean, school_data)
                    emb_text = ensure_passage_prefix(a_clean)
                    a_fields = ["anchors", "school_data.evidence_examples"] if ex else ["anchors"]
                    a_note = (
                        f"Matched school_data.evidence_examples csv_row={ex.get('csv_row')}"
                        if ex
                        else ""
                    )
                prototypes.append(
                    make_prototype_record(
                        prototype_id=unique_id(f"stage2_{ref_safe}_va_{idx:03d}"),
                        stage="criterion",
                        group_id=grp,
                        tieu_chi=tc,
                        criterion_key=item_key,
                        criterion_ref=ref,
                        prototype_type="verified_anchor",
                        prototype_family="verified_anchor",
                        embedding_text=emb_text,
                        anchor_status="verified_from_school_list",
                        criterion_anchor_status="verified_from_school_list",
                        anchor_text=a_clean,
                        source_anchor_index=idx,
                        source_fields=a_fields,
                        event_matchable=True,
                        eligibility_class=elig_class,
                        boundary_metadata=boundary_meta,
                        required_signals=req_signals,
                        negative_keywords=neg_kws,
                        mvp_scope=mvp_scope,
                        kind=kind,
                        notes=a_note,
                    )
                )

        elif crit_anchor_status == "verified_partial":
            # Section 6.2: individual verified anchor = verified_from_school_list,
            # overall criterion_anchor_status = verified_partial
            for idx, raw_anchor in enumerate(verified_anchors, start=1):
                a_clean = normalize_single_line(raw_anchor)
                if not a_clean:
                    continue
                ex = match_evidence_example(a_clean, school_data)
                a_fields = ["verified_anchors", "school_data.evidence_examples"] if ex else ["verified_anchors"]
                a_note = (
                    f"Individually verified anchor (csv_row={ex.get('csv_row')}) within verified_partial criterion"
                    if ex
                    else "Individually verified anchor within verified_partial criterion"
                )
                prototypes.append(
                    make_prototype_record(
                        prototype_id=unique_id(f"stage2_{ref_safe}_va_{idx:03d}"),
                        stage="criterion",
                        group_id=grp,
                        tieu_chi=tc,
                        criterion_key=item_key,
                        criterion_ref=ref,
                        prototype_type="verified_anchor",
                        prototype_family="verified_anchor",
                        embedding_text=ensure_passage_prefix(a_clean),
                        anchor_status="verified_from_school_list",
                        criterion_anchor_status="verified_partial",
                        anchor_text=a_clean,
                        source_anchor_index=idx,
                        source_fields=a_fields,
                        event_matchable=True,
                        eligibility_class=elig_class,
                        boundary_metadata=boundary_meta,
                        required_signals=req_signals,
                        negative_keywords=neg_kws,
                        mvp_scope=mvp_scope,
                        kind=kind,
                        notes=a_note,
                    )
                )
            for idx, raw_anchor in enumerate(draft_anchors, start=1):
                a_clean = normalize_single_line(raw_anchor)
                if not a_clean:
                    continue
                prototypes.append(
                    make_prototype_record(
                        prototype_id=unique_id(f"stage2_{ref_safe}_da_{idx:03d}"),
                        stage="criterion",
                        group_id=grp,
                        tieu_chi=tc,
                        criterion_key=item_key,
                        criterion_ref=ref,
                        prototype_type="draft_anchor",
                        prototype_family="draft_anchor",
                        embedding_text=ensure_passage_prefix(a_clean),
                        anchor_status="draft",
                        criterion_anchor_status="verified_partial",
                        anchor_text=a_clean,
                        source_anchor_index=idx,
                        source_fields=["draft_anchors"],
                        event_matchable=True,
                        eligibility_class=elig_class,
                        boundary_metadata=boundary_meta,
                        required_signals=req_signals,
                        negative_keywords=neg_kws,
                        mvp_scope=mvp_scope,
                        kind=kind,
                        notes="Unverified draft anchor within verified_partial criterion",
                    )
                )

        elif crit_anchor_status == "draft":
            for idx, raw_anchor in enumerate(anchors_list, start=1):
                a_clean = normalize_single_line(raw_anchor)
                if not a_clean:
                    continue
                prototypes.append(
                    make_prototype_record(
                        prototype_id=unique_id(f"stage2_{ref_safe}_da_{idx:03d}"),
                        stage="criterion",
                        group_id=grp,
                        tieu_chi=tc,
                        criterion_key=item_key,
                        criterion_ref=ref,
                        prototype_type="draft_anchor",
                        prototype_family="draft_anchor",
                        embedding_text=ensure_passage_prefix(a_clean),
                        anchor_status="draft",
                        criterion_anchor_status="draft",
                        anchor_text=a_clean,
                        source_anchor_index=idx,
                        source_fields=["anchors"],
                        event_matchable=True,
                        eligibility_class=elig_class,
                        boundary_metadata=boundary_meta,
                        required_signals=req_signals,
                        negative_keywords=neg_kws,
                        mvp_scope=mvp_scope,
                        kind=kind,
                        notes="Unverified draft anchor — illustrative rubric example only",
                    )
                )
        else:
            raise ValueError(f"Unsupported anchors_status='{crit_anchor_status}' on {item_key}")

        # D. keyword_context (at most 1 per criterion, positive keywords only)
        keywords = [normalize_single_line(k) for k in (item.get("keywords") or []) if normalize_single_line(k)]
        if keywords:
            prototypes.append(
                make_prototype_record(
                    prototype_id=unique_id(f"stage2_{ref_safe}_kw"),
                    stage="criterion",
                    group_id=grp,
                    tieu_chi=tc,
                    criterion_key=item_key,
                    criterion_ref=ref,
                    prototype_type="keyword_context",
                    prototype_family="keyword_context",
                    embedding_text=build_keyword_context_text(cleaned_name, keywords),
                    anchor_status="derived",
                    criterion_anchor_status=crit_anchor_status,
                    anchor_text=None,
                    source_anchor_index=None,
                    source_fields=["name", "keywords"],
                    event_matchable=True,
                    eligibility_class=elig_class,
                    boundary_metadata=boundary_meta,
                    required_signals=req_signals,
                    negative_keywords=neg_kws,
                    mvp_scope=mvp_scope,
                    kind=kind,
                    notes="Positive keyword retrieval context (negative_keywords excluded)",
                )
            )

    # ------------------------------------------------------------------
    # 3. Dedicated Reminder Stage (6.4 mandatory_event / mvp_scope=remind)
    # ------------------------------------------------------------------
    for item_key, item in reminder_items:
        ref = str(item["ref"]).strip()
        grp = str(item["group"]).strip()
        tc = str(item["tieu_chi"]).strip() if item.get("tieu_chi") else None
        kind = str(item.get("kind") or "").strip()
        mvp_scope = str(item.get("mvp_scope") or "").strip()
        crit_anchor_status = str(item.get("anchors_status") or "draft").strip()
        boundary_meta = extract_boundary_metadata(item)
        req_signals = extract_required_signals(ref, item)
        neg_kws = list(boundary_meta["negative_keywords"])
        ref_safe = clean_key(ref)
        cleaned_name = clean_criterion_name(str(item.get("name", "")), ref)

        def_text, def_fields = build_criterion_definition_text(ref, item)
        prototypes.append(
            make_prototype_record(
                prototype_id=unique_id(f"reminder_{ref_safe}_def"),
                stage="reminder",
                group_id=grp,
                tieu_chi=tc,
                criterion_key=item_key,
                criterion_ref=ref,
                prototype_type="reminder_definition",
                prototype_family="reminder_definition",
                embedding_text=def_text,
                anchor_status="derived",
                criterion_anchor_status=crit_anchor_status,
                anchor_text=None,
                source_anchor_index=None,
                source_fields=def_fields,
                event_matchable=True,
                eligibility_class="reminder_only",
                boundary_metadata=boundary_meta,
                required_signals=req_signals,
                negative_keywords=neg_kws,
                mvp_scope=mvp_scope,
                kind=kind,
                notes="Mandatory civic week reminder — excluded from positive C1-C4 Stage 2 ranking",
            )
        )

        for idx, raw_anchor in enumerate(item.get("anchors") or [], start=1):
            a_clean = normalize_single_line(raw_anchor)
            if not a_clean:
                continue
            prototypes.append(
                make_prototype_record(
                    prototype_id=unique_id(f"reminder_{ref_safe}_da_{idx:03d}"),
                    stage="reminder",
                    group_id=grp,
                    tieu_chi=tc,
                    criterion_key=item_key,
                    criterion_ref=ref,
                    prototype_type="draft_anchor",
                    prototype_family="draft_anchor",
                    embedding_text=ensure_passage_prefix(a_clean),
                    anchor_status="draft",
                    criterion_anchor_status=crit_anchor_status,
                    anchor_text=a_clean,
                    source_anchor_index=idx,
                    source_fields=["anchors"],
                    event_matchable=True,
                    eligibility_class="reminder_only",
                    boundary_metadata=boundary_meta,
                    required_signals=req_signals,
                    negative_keywords=neg_kws,
                    mvp_scope=mvp_scope,
                    kind=kind,
                    notes="Draft reminder anchor — mandatory_event / remind only",
                )
            )

        keywords = [normalize_single_line(k) for k in (item.get("keywords") or []) if normalize_single_line(k)]
        if keywords:
            prototypes.append(
                make_prototype_record(
                    prototype_id=unique_id(f"reminder_{ref_safe}_kw"),
                    stage="reminder",
                    group_id=grp,
                    tieu_chi=tc,
                    criterion_key=item_key,
                    criterion_ref=ref,
                    prototype_type="keyword_context",
                    prototype_family="keyword_context",
                    embedding_text=build_keyword_context_text(cleaned_name, keywords),
                    anchor_status="derived",
                    criterion_anchor_status=crit_anchor_status,
                    anchor_text=None,
                    source_anchor_index=None,
                    source_fields=["name", "keywords"],
                    event_matchable=True,
                    eligibility_class="reminder_only",
                    boundary_metadata=boundary_meta,
                    required_signals=req_signals,
                    negative_keywords=neg_kws,
                    mvp_scope=mvp_scope,
                    kind=kind,
                    notes="Reminder keyword context — mandatory_event / remind only",
                )
            )

    # ------------------------------------------------------------------
    # 4. Special Stage: NONE meta-label (negative_reference_only, draft)
    #    NEEDS_REVIEW intentionally gets ZERO prototypes.
    # ------------------------------------------------------------------
    none_label = meta_labels.get("NONE")
    if isinstance(none_label, dict):
        none_name = normalize_single_line(none_label.get("name", "Không tính điểm rèn luyện (NONE / NO_MATCH)"))
        none_desc = normalize_single_line(none_label.get("description", ""))
        none_inc = [normalize_single_line(x) for x in (none_label.get("inclusion_rules") or []) if normalize_single_line(x)]
        none_boundary = extract_boundary_metadata(none_label)
        none_neg_kws = list(none_boundary["negative_keywords"])

        def_parts = [f"{none_name}."]
        def_fields = ["meta_labels.NONE.name"]
        if none_desc:
            def_parts.append(f"Phạm vi:\n{none_desc}")
            def_fields.append("meta_labels.NONE.description")
        if none_inc:
            def_parts.append("Các nội dung ngoài khung ĐRL:\n" + "\n".join(f"- {r}" for r in none_inc))
            def_fields.append("meta_labels.NONE.inclusion_rules")

        prototypes.append(
            make_prototype_record(
                prototype_id=unique_id("special_NONE_def"),
                stage="special",
                group_id=None,
                tieu_chi=None,
                criterion_key="NONE",
                criterion_ref="NONE",
                prototype_type="draft_negative_reference",
                prototype_family="negative_reference",
                embedding_text=ensure_passage_prefix("\n\n".join(def_parts)),
                anchor_status="derived",
                criterion_anchor_status="draft",
                anchor_text=None,
                source_anchor_index=None,
                source_fields=def_fields,
                event_matchable=False,
                eligibility_class="negative_reference_only",
                boundary_metadata=none_boundary,
                required_signals=[],
                negative_keywords=none_neg_kws,
                mvp_scope="out_of_scope",
                kind="meta_label",
                can_decide_none=False,
                notes="Draft negative reference only — cannot independently decide NONE or enter positive voting",
            )
        )

        for idx, raw_anchor in enumerate(none_label.get("anchors") or [], start=1):
            a_clean = normalize_single_line(raw_anchor)
            if not a_clean:
                continue
            prototypes.append(
                make_prototype_record(
                    prototype_id=unique_id(f"special_NONE_da_{idx:03d}"),
                    stage="special",
                    group_id=None,
                    tieu_chi=None,
                    criterion_key="NONE",
                    criterion_ref="NONE",
                    prototype_type="draft_negative_reference",
                    prototype_family="negative_reference",
                    embedding_text=ensure_passage_prefix(a_clean),
                    anchor_status="draft",
                    criterion_anchor_status="draft",
                    anchor_text=a_clean,
                    source_anchor_index=idx,
                    source_fields=["meta_labels.NONE.anchors"],
                    event_matchable=False,
                    eligibility_class="negative_reference_only",
                    boundary_metadata=none_boundary,
                    required_signals=[],
                    negative_keywords=none_neg_kws,
                    mvp_scope="out_of_scope",
                    kind="meta_label",
                    can_decide_none=False,
                    notes="Draft negative anchor — not verified from school list (Q17)",
                )
            )

        none_kws = [normalize_single_line(k) for k in (none_label.get("keywords") or []) if normalize_single_line(k)]
        if none_kws:
            prototypes.append(
                make_prototype_record(
                    prototype_id=unique_id("special_NONE_kw"),
                    stage="special",
                    group_id=None,
                    tieu_chi=None,
                    criterion_key="NONE",
                    criterion_ref="NONE",
                    prototype_type="draft_negative_reference",
                    prototype_family="negative_reference",
                    embedding_text=build_keyword_context_text(none_name, none_kws),
                    anchor_status="derived",
                    criterion_anchor_status="draft",
                    anchor_text=None,
                    source_anchor_index=None,
                    source_fields=["meta_labels.NONE.name", "meta_labels.NONE.keywords"],
                    event_matchable=False,
                    eligibility_class="negative_reference_only",
                    boundary_metadata=none_boundary,
                    required_signals=[],
                    negative_keywords=none_neg_kws,
                    mvp_scope="out_of_scope",
                    kind="meta_label",
                    can_decide_none=False,
                    notes="Draft negative keyword reference only (negative_keywords excluded)",
                )
            )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "total_prototypes": len(prototypes),
        "by_stage": dict(Counter(p["stage"] for p in prototypes)),
        "by_prototype_type": dict(Counter(p["prototype_type"] for p in prototypes)),
        "by_prototype_family": dict(Counter(p["prototype_family"] for p in prototypes)),
        "by_eligibility_class": dict(Counter(p["eligibility_class"] for p in prototypes)),
        "by_anchor_status": dict(Counter(p["anchor_status"] for p in prototypes)),
        "positive_c1_c4_criteria_count": len(em_true_c1_c4),
        "reminder_criteria_count": len(reminder_items),
        "excluded_items_count": len(excluded_items),
    }
    return prototypes, summary


# ---------------------------------------------------------------------------
# Writers & Technical Validation
# ---------------------------------------------------------------------------


def write_jsonl(prototypes: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for rec in prototypes:
            handle.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")


def write_yaml_mirror(prototypes: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": "data/raw/rubric.yaml",
        "total_prototypes": len(prototypes),
        "prototypes": prototypes,
    }
    header = (
        "# rubric_prototypes.yaml (Schema v2.0)\n"
        "# Auto-generated by scripts/generate_rubric_prototypes.py — do not edit manually.\n"
    )
    body = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120)
    path.write_text(header + body, encoding="utf-8")


def validate_prototypes(prototypes: List[Dict[str, Any]], rubric: Dict[str, Any]) -> Dict[str, Any]:
    """Run Section 11.A (Data Validation) and 11.B (Scope Validation) checks."""
    checks: List[Dict[str, Any]] = []
    items = rubric.get("items") or {}
    ref_to_item = {str(v["ref"]).strip(): (k, v) for k, v in items.items() if isinstance(v, dict) and "ref" in v}
    em_false_keys = {k for k, v in items.items() if isinstance(v, dict) and v.get("event_matchable") is not True}
    em_true_c1_c4_keys = {
        k
        for k, v in items.items()
        if isinstance(v, dict)
        and v.get("event_matchable") is True
        and str(v.get("group")) in POSITIVE_MAJOR_GROUPS
    }

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # Unique prototype_ids
    pids = [p["prototype_id"] for p in prototypes]
    add("unique_prototype_ids", len(pids) == len(set(pids)), f"{len(pids)} prototypes, unique={len(set(pids))}")

    # Required fields & non-empty passage-prefixed embedding_text
    req_fields = {
        "schema_version",
        "prototype_id",
        "stage",
        "group_id",
        "criterion_key",
        "criterion_ref",
        "prototype_type",
        "prototype_family",
        "embedding_text",
        "anchor_status",
        "source_fields",
        "event_matchable",
        "eligibility_class",
        "boundary_metadata",
    }
    missing_fields = [p["prototype_id"] for p in prototypes if not req_fields.issubset(p.keys())]
    add("required_fields_present", len(missing_fields) == 0, f"missing={len(missing_fields)}")

    bad_texts = [
        p["prototype_id"]
        for p in prototypes
        if not isinstance(p.get("embedding_text"), str)
        or not p["embedding_text"].startswith(PASSAGE_PREFIX)
        or p["embedding_text"].startswith(PASSAGE_PREFIX + PASSAGE_PREFIX)
        or len(p["embedding_text"].strip()) <= len(PASSAGE_PREFIX)
    ]
    add("valid_single_passage_prefix_non_empty", len(bad_texts) == 0, f"bad_texts={len(bad_texts)}")

    # Truncation check: ensure no mid-word truncated group prototypes like old `[:60]` artifacts
    trunc_hits = [
        p["prototype_id"]
        for p in prototypes
        if p["stage"] == "group" and re.search(r"\b(?:tiểu l,|hoàn tất vò,|tình ngu,|Hiến phá\.)", p["embedding_text"])
    ]
    add("no_truncated_group_prototypes", len(trunc_hits) == 0, f"truncated_hits={trunc_hits}")

    # Stage 1 contains ONLY C1-C4 (1 per group)
    s1 = [p for p in prototypes if p["stage"] == "group"]
    s1_groups = [p["group_id"] for p in s1]
    add("stage1_only_c1_to_c4", s1_groups == ["C1", "C2", "C3", "C4"], f"stage1_groups={s1_groups}")

    # Stage 2 contains ONLY positive/conditional candidates from C1-C4 (zero P, zero NONE, zero event_matchable=false)
    s2 = [p for p in prototypes if p["stage"] == "criterion"]
    s2_bad_grp = [p["prototype_id"] for p in s2 if p["group_id"] not in POSITIVE_MAJOR_GROUPS]
    s2_bad_elig = [
        p["prototype_id"]
        for p in s2
        if p["eligibility_class"] not in ("positive_candidate", "conditional_candidate")
    ]
    s2_bad_em = [p["prototype_id"] for p in s2 if p["criterion_key"] in em_false_keys or not p["event_matchable"]]
    s2_def_keys = {p["criterion_key"] for p in s2 if p["prototype_type"] == "criterion_definition"}
    add(
        "stage2_only_eligible_c1_c4_candidates",
        not s2_bad_grp and not s2_bad_elig and not s2_bad_em and s2_def_keys == em_true_c1_c4_keys,
        f"s2={len(s2)}, covered_criteria={len(s2_def_keys)}/{len(em_true_c1_c4_keys)}, bad_grp={len(s2_bad_grp)}, bad_elig={len(s2_bad_elig)}, bad_em={len(s2_bad_em)}",
    )

    # 6.4 separated into stage="reminder"
    rem = [p for p in prototypes if p["stage"] == "reminder"]
    rem_ok = (
        len(rem) > 0
        and all(p["criterion_ref"] == "6.4" and p["group_id"] == "P" and p["eligibility_class"] == "reminder_only" for p in rem)
        and any(p["prototype_type"] == "reminder_definition" for p in rem)
    )
    add("criterion_6_4_separated_in_reminder_stage", rem_ok, f"reminder_count={len(rem)}")

    # NONE separated into stage="special", negative_reference_only, can_decide_none=False; zero NEEDS_REVIEW
    spec = [p for p in prototypes if p["stage"] == "special"]
    spec_ok = (
        len(spec) > 0
        and all(
            p["criterion_ref"] == "NONE"
            and p["prototype_type"] == "draft_negative_reference"
            and p["eligibility_class"] == "negative_reference_only"
            and p.get("can_decide_none") is False
            for p in spec
        )
    )
    nr_count = sum(1 for p in prototypes if p.get("criterion_ref") == "NEEDS_REVIEW" or p.get("criterion_key") == "NEEDS_REVIEW")
    add("none_isolated_and_zero_needs_review_prototypes", spec_ok and nr_count == 0, f"special_none={len(spec)}, needs_review={nr_count}")

    # 3.3 mobilization context & required_signals
    p33 = [p for p in prototypes if p["criterion_ref"] == "3.3"]
    p33_va = [p for p in p33 if p["prototype_type"] == "verified_anchor"]
    p33_ok = (
        len(p33_va) == 3
        and all("explicit_student_mobilization_signal" in p["required_signals"] for p in p33)
        and all("huy động" in p["embedding_text"].lower() and bool(p.get("anchor_text")) for p in p33_va)
    )
    add("criterion_3_3_mobilization_context_and_signals", p33_ok, f"3.3 verified_anchors={len(p33_va)}")

    # 3.8 individual verified vs criterion verified_partial
    p38_va = [p for p in prototypes if p["criterion_ref"] == "3.8" and p["prototype_type"] == "verified_anchor"]
    p38_da = [p for p in prototypes if p["criterion_ref"] == "3.8" and p["prototype_type"] == "draft_anchor"]
    p38_ok = (
        len(p38_va) == 2
        and len(p38_da) == 2
        and all(p["anchor_status"] == "verified_from_school_list" and p["criterion_anchor_status"] == "verified_partial" for p in p38_va)
        and all(p["anchor_status"] == "draft" and p["criterion_anchor_status"] == "verified_partial" for p in p38_da)
    )
    add("criterion_3_8_verified_partial_normalized", p38_ok, f"3.8 va={len(p38_va)}, da={len(p38_da)}")

    # Referential integrity against rubric.yaml + no draft promoted to verified
    ref_errors: List[str] = []
    draft_promoted: List[str] = []
    for p in prototypes:
        if p["stage"] in ("criterion", "reminder"):
            ref = str(p["criterion_ref"])
            if ref not in ref_to_item:
                ref_errors.append(f"{p['prototype_id']}: unknown ref {ref}")
            else:
                rkey, ritem = ref_to_item[ref]
                if rkey != p["criterion_key"] or str(ritem["group"]) != p["group_id"]:
                    ref_errors.append(f"{p['prototype_id']}: key/group mismatch")
                if ritem.get("anchors_status") == "draft" and p["prototype_type"] == "verified_anchor":
                    draft_promoted.append(p["prototype_id"])
    add("rubric_referential_integrity_and_no_draft_promotion", not ref_errors and not draft_promoted, f"ref_errors={len(ref_errors)}, draft_promoted={len(draft_promoted)}")

    return {"all_passed": all(c["passed"] for c in checks), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Schema v2.0 rubric prototypes from rubric.yaml.")
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rubric = yaml.safe_load(args.rubric.read_text(encoding="utf-8"))
    prototypes, summary = generate_prototypes(rubric)
    validation = validate_prototypes(prototypes, rubric)

    jsonl_path = args.out_dir / "rubric_prototypes.jsonl"
    yaml_path = args.out_dir / "rubric_prototypes.yaml"
    write_jsonl(prototypes, jsonl_path)
    write_yaml_mirror(prototypes, yaml_path)

    print(f"[OK] Wrote {jsonl_path} ({summary['total_prototypes']} records, schema v{SCHEMA_VERSION})")
    print(f"[OK] Wrote {yaml_path}")
    print("Summary:", json.dumps(summary, ensure_ascii=False))
    for chk in validation["checks"]:
        print(f"  [{'PASS' if chk['passed'] else 'FAIL'}] {chk['check']}: {chk['detail']}")

    return 0 if validation["all_passed"] else 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    raise SystemExit(main())
