"""Data-driven hierarchical ClsIndex dataset builder (Stage 1: C1..C5 -> Stage 2: <major>.<minor>).

Works identically on the current demo CSV (res/events_full.csv) and any future full dataset CSV
containing columns: Ma_Muc, Ten_Hoat_Dong, and optionally Organizer, Vai_Tro.

Aligned with:
- D-2: Partitioned parent-group (C1..C5) -> per-group fine-grained datasets for multi-group union KNN.
- T-1: Full rubric taxonomy (data-driven across all labels present in input CSV).
- T-2, T-3: Group P (6.*) and NONE/UNKNOWN are excluded by default (handled upstream by Module 1).
- T-4: Canonical item IDs use '<major>.<minor>' (e.g. '1.6', '1.7'); role/organizer context is
  preserved in classification text to avoid title-only collisions between role variants.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from configs.model_config import (
    CLS_INDEX_DIR,
    DEMO_EVENTS_FULL_CSV_PATH,
    RAW_EVENTS_JSON_PATH,
)
from src.contract_validator import validate_cls_sample
from src.schemas import (
    ACTIVE_CLASSIFIER_PARENT_GROUPS,
    VALID_PARENT_GROUPS,
    ClsSample,
    is_valid_item_id,
    parent_group_from_item_id,
)


def normalize_text_key(text: str) -> str:
    """Lowercase, strip Vietnamese diacritics, and collapse punctuation/whitespace."""
    s = unicodedata.normalize("NFD", (text or "").lower().replace("đ", "d"))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def token_jaccard(a: str, b: str) -> float:
    """Compute token-level Jaccard similarity between two normalized strings."""
    ta = set(normalize_text_key(a).split())
    tb = set(normalize_text_key(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def build_sample_text(title: str, organizer: str = "", role: str = "") -> str:
    """Construct rich text for P2.1 embedding so role variants (e.g. 1.6 vs 1.7) stay distinct."""
    parts = [title.strip()]
    if role.strip():
        parts.append(f"Vai trò: {role.strip()}")
    if organizer.strip():
        parts.append(f"Đơn vị tổ chức: {organizer.strip()}")
    return " | ".join(p for p in parts if p)


def load_eval_titles(events_json_path: Optional[Path]) -> List[str]:
    """Load event titles from data/events.json for leakage checks."""
    if events_json_path is None or not events_json_path.exists():
        return []
    with open(events_json_path, encoding="utf-8") as handle:
        data = json.load(handle)
    return [str(item.get("title") or "").strip() for item in data if item.get("title")]


def build_hierarchical_dataset(
    csv_path: Path = DEMO_EVENTS_FULL_CSV_PATH,
    events_json_path: Optional[Path] = RAW_EVENTS_JSON_PATH,
    excluded_parent_groups: Sequence[str] = ("P",),
    dataset_stage: str = "demo",
    leakage_jaccard_threshold: float = 0.5,
    exclude_overlapping_eval_titles: bool = True,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build partitioned parent-child ClsIndex dataset from a labeled CSV."""
    excluded_set: Set[str] = set(excluded_parent_groups)
    allowed_groups: Tuple[str, ...] = tuple(
        g for g in VALID_PARENT_GROUPS if g not in excluded_set
    )
    eval_titles = load_eval_titles(events_json_path)

    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    seen_exact: Set[Tuple[str, str]] = set()
    text_to_labels: Dict[str, Set[str]] = {}
    accepted_samples: List[ClsSample] = []
    excluded_by_group: List[Dict[str, Any]] = []
    skipped_duplicates: int = 0
    invalid_rows: List[Dict[str, Any]] = []
    leakage_excluded: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        item_id = (row.get("Ma_Muc") or "").strip()
        title = (row.get("Ten_Hoat_Dong") or "").strip()
        organizer = (row.get("Organizer") or "").strip()
        role = (row.get("Vai_Tro") or "").strip()

        if not title or not is_valid_item_id(item_id):
            invalid_rows.append({"row": idx, "Ma_Muc": item_id, "Ten_Hoat_Dong": title})
            continue

        parent_group = parent_group_from_item_id(item_id)
        if parent_group in excluded_set:
            excluded_by_group.append(
                {
                    "row": idx,
                    "rubric_item_id": item_id,
                    "parent_group_id": parent_group,
                    "title": title,
                }
            )
            continue

        text = build_sample_text(title=title, organizer=organizer, role=role)
        norm_key = normalize_text_key(text)

        # Deduplicate exact (normalized text, item_id) pairs
        if (norm_key, item_id) in seen_exact:
            skipped_duplicates += 1
            continue
        seen_exact.add((norm_key, item_id))

        sample = ClsSample(
            sample_id=f"{dataset_stage}_{idx:04d}",
            text=text,
            title=title,
            rubric_item_id=item_id,
            parent_group_id=parent_group,
            organizer=organizer,
            role=role,
            source=f"seed_{dataset_stage}",
            metadata={"csv_row": idx},
        )
        validation = validate_cls_sample(sample, allowed_parent_groups=allowed_groups)
        if not validation.is_valid:
            invalid_rows.append(
                {
                    "row": idx,
                    "errors": [e.message for e in validation.errors],
                }
            )
            continue

        # Check leakage against held-out events.json titles
        best_overlap = 0.0
        matched_eval_title = None
        for et in eval_titles:
            sim = token_jaccard(title, et)
            if sim > best_overlap:
                best_overlap = sim
                matched_eval_title = et

        if exclude_overlapping_eval_titles and best_overlap >= leakage_jaccard_threshold:
            leakage_excluded.append(
                {
                    "sample": sample.to_dict(),
                    "matched_eval_title": matched_eval_title,
                    "jaccard": round(best_overlap, 4),
                }
            )
            continue

        text_to_labels.setdefault(norm_key, set()).add(item_id)
        accepted_samples.append(sample)

    conflicting_texts = {
        k: sorted(v) for k, v in text_to_labels.items() if len(v) > 1
    }

    stage2_by_parent_group: Dict[str, List[Dict[str, Any]]] = {
        grp: [] for grp in allowed_groups if grp in ACTIVE_CLASSIFIER_PARENT_GROUPS or grp in allowed_groups
    }
    item_counts: Dict[str, int] = {}
    for sample in accepted_samples:
        stage2_by_parent_group.setdefault(sample.parent_group_id, []).append(sample.to_dict())
        item_counts[sample.rubric_item_id] = item_counts.get(sample.rubric_item_id, 0) + 1

    dataset = {
        "schema_version": "cls-hierarchical-dataset-v1",
        "dataset_stage": dataset_stage,
        "source_csv": str(csv_path),
        "excluded_parent_groups": list(excluded_parent_groups),
        "summary": {
            "total_csv_rows": len(rows),
            "accepted_samples": len(accepted_samples),
            "excluded_by_parent_group": len(excluded_by_group),
            "skipped_exact_duplicates": skipped_duplicates,
            "leakage_excluded_count": len(leakage_excluded),
            "invalid_rows": len(invalid_rows),
            "conflicting_normalized_texts": len(conflicting_texts),
            "counts_by_parent_group": {
                grp: len(items) for grp, items in stage2_by_parent_group.items()
            },
            "unique_rubric_items": len(item_counts),
            "counts_by_rubric_item": dict(
                sorted(item_counts.items(), key=lambda kv: [int(p) for p in kv[0].split(".")])
            ),
        },
        "conflicting_texts": conflicting_texts,
        "stage1_coarse_samples": [s.to_dict() for s in accepted_samples],
        "stage2_by_parent_group": stage2_by_parent_group,
        "excluded_by_group_records": excluded_by_group,
        "leakage_excluded_samples": leakage_excluded,
        "invalid_row_records": invalid_rows,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(dataset, out, ensure_ascii=False, indent=2)

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build partitioned hierarchical ClsIndex dataset from labeled CSV."
    )
    parser.add_argument("--input", type=Path, default=DEMO_EVENTS_FULL_CSV_PATH)
    parser.add_argument("--events-json", type=Path, default=RAW_EVENTS_JSON_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=CLS_INDEX_DIR / "hierarchical_dataset_demo.json",
    )
    parser.add_argument(
        "--stage",
        default="demo",
        help="Dataset stage tag (e.g. 'demo' or 'full').",
    )
    parser.add_argument(
        "--include-all-groups",
        action="store_true",
        help="Include group P (6.*) as well instead of excluding it per T-2.",
    )
    parser.add_argument(
        "--keep-eval-overlaps",
        action="store_true",
        help="Do not filter samples overlapping with data/events.json.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute summary without writing.")
    args = parser.parse_args()

    dataset = build_hierarchical_dataset(
        csv_path=args.input,
        events_json_path=args.events_json,
        excluded_parent_groups=() if args.include_all_groups else ("P",),
        dataset_stage=args.stage,
        exclude_overlapping_eval_titles=not args.keep_eval_overlaps,
        output_path=None if args.dry_run else args.output,
    )
    print(json.dumps(dataset["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
