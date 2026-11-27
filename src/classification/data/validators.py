"""Strict pre-flight and post-flight technical validators for the classification pipeline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from src.classification.data.loaders import LoadedAssets, ReferenceBranchDataset


class AssetValidationError(RuntimeError):
    """Raised when a critical asset inconsistency makes classification unreliable."""


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class ValidationReport:
    checks: List[ValidationCheck] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(ValidationCheck(name=name, passed=bool(passed), detail=detail))

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> List[ValidationCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [{"check": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks]


REQUIRED_SAMPLE_COLUMNS = {
    "embedding_row",
    "sample_id",
    "event_title",
    "criterion_ref",
    "criterion_key",
    "group_id",
    "label_status",
}

REQUIRED_RUBRIC_COLUMNS = {
    "embedding_row",
    "prototype_id",
    "prototype_type",
    "criterion_ref",
    "criterion_key",
    "group_id",
    "stage",
    "anchor_status",
    "event_matchable",
}


def _validate_matrix_and_alignment(
    branch: ReferenceBranchDataset,
    expected_dim: int,
    l2_tolerance: float,
    required_cols: Set[str],
    report: ValidationReport,
) -> None:
    prefix = f"{branch.branch_name}_branch"
    mat = branch.embeddings
    df = branch.metadata

    # 2D float32 shape check
    shape_ok = mat.ndim == 2 and mat.shape[1] == expected_dim and mat.shape[0] > 0
    report.add(
        f"{prefix}_embedding_shape",
        shape_ok,
        f"shape={tuple(mat.shape)}, expected=(N>0, {expected_dim})",
    )
    dtype_ok = mat.dtype == np.float32
    report.add(f"{prefix}_embedding_dtype", dtype_ok, f"dtype={mat.dtype}, expected=float32")

    finite_ok = bool(np.isfinite(mat).all())
    report.add(f"{prefix}_embedding_finite", finite_ok, "All reference vector entries finite")

    norms = np.linalg.norm(np.asarray(mat, dtype=np.float64), axis=1)
    max_norm_err = float(np.max(np.abs(norms - 1.0))) if len(norms) else float("inf")
    report.add(
        f"{prefix}_l2_normalized",
        max_norm_err <= l2_tolerance,
        f"max |norm - 1| = {max_norm_err:.2e} (tol={l2_tolerance:.1e})",
    )

    missing_cols = sorted(required_cols - set(df.columns))
    report.add(
        f"{prefix}_metadata_schema",
        len(missing_cols) == 0,
        "All required columns present" if not missing_cols else f"Missing columns: {missing_cols}",
    )

    row_count_ok = len(df) == mat.shape[0]
    report.add(
        f"{prefix}_row_count_alignment",
        row_count_ok,
        f"metadata rows={len(df)} vs embedding rows={mat.shape[0]}",
    )

    if "embedding_row" in df.columns:
        seq_ok = list(df["embedding_row"]) == list(range(len(df)))
        report.add(
            f"{prefix}_embedding_row_sequential",
            seq_ok,
            f"embedding_row matches exact 0..{len(df)-1}: {seq_ok}",
        )


def validate_loaded_assets(assets: LoadedAssets) -> ValidationReport:
    """Validate all input assets, schemas, vector matrices, and rubric references."""
    report = ValidationReport()
    emb_cfg = assets.embedding_config
    expected_dim = int(emb_cfg.get("embedding_dimension", 384))
    l2_tol = float((assets.config.get("embedding") or {}).get("l2_tolerance", 1e-4))

    # Embedding config consistency
    model_name = emb_cfg.get("model_name")
    prefixes = emb_cfg.get("prefixes") or {}
    cfg_ok = (
        bool(model_name)
        and expected_dim == 384
        and emb_cfg.get("dtype") == "float32"
        and emb_cfg.get("l2_normalization") is True
        and prefixes.get("query") == "query: "
        and prefixes.get("passage") == "passage: "
    )
    report.add(
        "embedding_config_valid",
        cfg_ok,
        f"model={model_name}, rev={emb_cfg.get('model_revision')}, dim={expected_dim}, prefixes={prefixes}",
    )

    # Validate Sample & Rubric reference matrices and metadata alignment
    _validate_matrix_and_alignment(
        assets.sample_branch, expected_dim, l2_tol, REQUIRED_SAMPLE_COLUMNS, report
    )
    _validate_matrix_and_alignment(
        assets.rubric_branch, expected_dim, l2_tol, REQUIRED_RUBRIC_COLUMNS, report
    )

    # Validate rubric.yaml reference integrity
    groups = assets.rubric.get("groups") or {}
    items = assets.rubric.get("items") or {}
    valid_groups = set(groups.keys())
    ref_to_item = {
        str(v["ref"]).strip(): (k, v)
        for k, v in items.items()
        if isinstance(v, dict) and "ref" in v
    }

    # Verify Sample branch refs/groups against rubric.yaml
    sm = assets.sample_branch.metadata
    sample_ref_errors: List[str] = []
    for _, row in sm.iterrows():
        ref = str(row["criterion_ref"])
        key = str(row["criterion_key"])
        grp = str(row["group_id"])
        if grp not in valid_groups:
            sample_ref_errors.append(f"sample {row['sample_id']}: invalid group_id '{grp}'")
        elif ref not in ref_to_item:
            sample_ref_errors.append(f"sample {row['sample_id']}: unknown criterion_ref '{ref}'")
        else:
            rkey, ritem = ref_to_item[ref]
            if rkey != key or str(ritem.get("group")) != grp or ritem.get("event_matchable") is not True:
                sample_ref_errors.append(
                    f"sample {row['sample_id']}: rubric mismatch ({ref}, {key}, {grp}, em={ritem.get('event_matchable')})"
                )
    report.add(
        "sample_rubric_referential_integrity",
        len(sample_ref_errors) == 0,
        f"All {len(sm)} samples valid against rubric.yaml"
        if not sample_ref_errors
        else "; ".join(sample_ref_errors[:5]),
    )

    # Verify Rubric branch refs/groups/stages against rubric.yaml (supports Schema v1 & v2)
    rm = assets.rubric_branch.metadata
    rubric_ref_errors: List[str] = []
    s1_count = 0
    s2_count = 0
    reminder_count = 0
    special_count = 0
    positive_groups = {"C1", "C2", "C3", "C4"}
    has_elig_col = "eligibility_class" in rm.columns

    for _, row in rm.iterrows():
        stage = str(row["stage"])
        grp_val = row["group_id"]
        grp = None if (grp_val is None or pd.isna(grp_val) or str(grp_val) == "") else str(grp_val)
        elig = str(row["eligibility_class"]) if (has_elig_col and not pd.isna(row["eligibility_class"])) else None

        if stage == "group":
            s1_count += 1
            if grp not in positive_groups:
                rubric_ref_errors.append(f"prototype {row['prototype_id']}: Stage 1 group_id '{grp}' not in C1-C4")
        elif stage == "criterion":
            s2_count += 1
            if grp not in valid_groups:
                rubric_ref_errors.append(f"prototype {row['prototype_id']}: invalid group_id '{grp}'")
                continue
            ref = str(row["criterion_ref"])
            key = str(row["criterion_key"])
            if ref not in ref_to_item:
                rubric_ref_errors.append(f"prototype {row['prototype_id']}: unknown criterion_ref '{ref}'")
            else:
                rkey, ritem = ref_to_item[ref]
                if rkey != key or str(ritem.get("group")) != grp:
                    rubric_ref_errors.append(
                        f"prototype {row['prototype_id']}: mismatch ({ref}, {key}, {grp})"
                    )
                if has_elig_col and elig not in ("positive_candidate", "conditional_candidate"):
                    rubric_ref_errors.append(
                        f"prototype {row['prototype_id']}: ineligible Stage 2 eligibility_class '{elig}'"
                    )
        elif stage == "reminder":
            reminder_count += 1
            ref = str(row["criterion_ref"])
            key = str(row["criterion_key"])
            if grp != "P" or ref not in ref_to_item or ref_to_item[ref][0] != key:
                rubric_ref_errors.append(f"prototype {row['prototype_id']}: invalid reminder ({ref}, {key}, {grp})")
            if has_elig_col and elig != "reminder_only":
                rubric_ref_errors.append(f"prototype {row['prototype_id']}: reminder eligibility_class='{elig}'")
        elif stage == "special":
            special_count += 1
            ref = str(row["criterion_ref"])
            if ref != "NONE" or bool(row["event_matchable"]):
                rubric_ref_errors.append(f"prototype {row['prototype_id']}: invalid special prototype ({ref})")
            if has_elig_col and elig != "negative_reference_only":
                rubric_ref_errors.append(f"prototype {row['prototype_id']}: special eligibility_class='{elig}'")
        else:
            rubric_ref_errors.append(f"prototype {row['prototype_id']}: unexpected stage '{stage}'")

    report.add(
        "rubric_prototypes_referential_integrity",
        len(rubric_ref_errors) == 0 and s1_count > 0 and s2_count > 0,
        f"Stage 1={s1_count}, Stage 2={s2_count}, reminder={reminder_count}, special={special_count}; errors={len(rubric_ref_errors)}",
    )

    # Verify input events integrity (database mode permits empty table gracefully)
    allow_empty = getattr(assets, "source_mode", "json") == "database"
    events_ok = True if allow_empty else len(assets.events) > 0
    post_ids = [ev.get("post_id") for ev in assets.events if ev.get("post_id") is not None]
    unique_pids = len(post_ids) == len(set(post_ids))
    report.add(
        "input_events_integrity",
        events_ok and unique_pids,
        f"source={getattr(assets, 'source_mode', 'json')}, total_events={len(assets.events)}, "
        f"non_null_post_ids={len(post_ids)}, unique_post_ids={unique_pids}",
    )

    if not report.all_passed:
        msgs = "; ".join(f"{f.name}: {f.detail}" for f in report.failures)
        raise AssetValidationError(f"Critical asset validation failure(s): {msgs}")

    return report


def validate_query_embeddings(
    query_embeddings: np.ndarray,
    query_metadata: Dict[str, Any],
    events: List[Dict[str, Any]],
    expected_dim: int = 384,
    l2_tolerance: float = 1e-4,
) -> ValidationReport:
    """Validate generated or reused shared query embeddings and metadata alignment."""
    report = ValidationReport()
    records = query_metadata.get("records") or []
    skipped = query_metadata.get("skipped_records") or []

    shape_ok = query_embeddings.ndim == 2 and query_embeddings.shape == (len(records), expected_dim)
    report.add(
        "query_embeddings_shape",
        shape_ok,
        f"shape={tuple(query_embeddings.shape)}, expected=({len(records)}, {expected_dim})",
    )
    dtype_ok = query_embeddings.dtype == np.float32
    report.add("query_embeddings_dtype", dtype_ok, f"dtype={query_embeddings.dtype}")

    finite_ok = bool(np.isfinite(query_embeddings).all())
    report.add("query_embeddings_finite", finite_ok, "All query vector values finite")

    if len(records) > 0:
        norms = np.linalg.norm(np.asarray(query_embeddings, dtype=np.float64), axis=1)
        max_err = float(np.max(np.abs(norms - 1.0)))
        norm_ok = max_err <= l2_tolerance
        norm_detail = f"max |norm - 1| = {max_err:.2e}"
    else:
        norm_ok = True
        norm_detail = "0 embedded events"
    report.add("query_embeddings_l2_normalized", norm_ok, norm_detail)

    # Verify exact 1:1 mapping and partition of input events into embedded + skipped
    emb_rows_ok = [r["embedding_row"] for r in records] == list(range(len(records)))
    report.add("query_metadata_sequential_rows", emb_rows_ok, f"0..{len(records)-1}: {emb_rows_ok}")

    emb_indices = [r["source_event_index"] for r in records]
    skip_indices = [r["source_event_index"] for r in skipped]
    all_indices = sorted(emb_indices + skip_indices)
    partition_ok = all_indices == list(range(len(events))) and len(set(emb_indices) & set(skip_indices)) == 0
    report.add(
        "query_event_partition_exact",
        partition_ok,
        f"embedded={len(records)} + skipped={len(skipped)} == input={len(events)}",
    )

    # Verify post_id and prefix consistency for every embedded row
    pid_match_ok = all(
        r.get("post_id") == events[r["source_event_index"]].get("post_id")
        and str(r.get("embedding_text", "")).startswith("query: ")
        for r in records
    )
    report.add(
        "query_post_id_and_prefix_alignment",
        pid_match_ok,
        f"All {len(records)} query rows match source post_id and 'query: ' prefix",
    )

    if not report.all_passed:
        msgs = "; ".join(f"{f.name}: {f.detail}" for f in report.failures)
        raise AssetValidationError(f"Query embedding validation failed: {msgs}")

    return report


def validate_output_records(
    input_events: List[Dict[str, Any]],
    output_events: List[Dict[str, Any]],
    rubric: Dict[str, Any],
    expected_algorithm: str,
) -> ValidationReport:
    """Technical integrity validation of classification_events.json output records."""
    report = ValidationReport()
    items = rubric.get("items") or {}
    valid_em_true_refs = {
        str(v["ref"]).strip(): (k, str(v["group"]).strip())
        for k, v in items.items()
        if isinstance(v, dict) and v.get("event_matchable") is True
    }
    valid_statuses = {"PREDICTED", "NEEDS_REVIEW", "NO_MATCH", "SKIPPED_NOT_EVENT"}

    count_ok = len(input_events) == len(output_events)
    report.add("output_event_count_exact", count_ok, f"input={len(input_events)}, output={len(output_events)}")

    order_and_fields_ok = True
    status_and_schema_ok = True
    predicted_rubric_ok = True
    stage2_shared_groups_ok = True
    error_notes: List[str] = []

    for idx, (inp, out) in enumerate(zip(input_events, output_events)):
        # Check every original field preserved unchanged
        for k, v in inp.items():
            if k not in out or out[k] != v:
                order_and_fields_ok = False
                error_notes.append(f"idx={idx}: original field '{k}' altered or missing")
                break
        if inp.get("post_id") != out.get("post_id"):
            order_and_fields_ok = False
            error_notes.append(f"idx={idx}: post_id mismatch")

        cls_obj = out.get("classification")
        if not isinstance(cls_obj, dict):
            status_and_schema_ok = False
            error_notes.append(f"idx={idx}: missing classification object")
            continue

        status = cls_obj.get("status")
        if status not in valid_statuses:
            status_and_schema_ok = False
            error_notes.append(f"idx={idx}: invalid status '{status}'")

        if status != "SKIPPED_NOT_EVENT" and cls_obj.get("algorithm") != expected_algorithm:
            status_and_schema_ok = False
            error_notes.append(f"idx={idx}: unexpected algorithm '{cls_obj.get('algorithm')}'")

        if status == "PREDICTED":
            ref = cls_obj.get("predicted_criterion_ref")
            key = cls_obj.get("predicted_criterion_key")
            grp = cls_obj.get("predicted_group_id")
            if ref not in valid_em_true_refs:
                predicted_rubric_ok = False
                error_notes.append(f"idx={idx}: predicted ref '{ref}' not event_matchable:true in rubric")
            else:
                exp_key, exp_grp = valid_em_true_refs[ref]
                if key != exp_key or grp != exp_grp:
                    predicted_rubric_ok = False
                    error_notes.append(f"idx={idx}: predicted lineage mismatch ({ref}, {key}, {grp})")
        else:
            # NEEDS_REVIEW, NO_MATCH, SKIPPED_NOT_EVENT must keep predicted criterion fields null
            if (
                cls_obj.get("predicted_criterion_ref") is not None
                or cls_obj.get("predicted_criterion_key") is not None
                or cls_obj.get("predicted_criterion_name") is not None
            ):
                status_and_schema_ok = False
                error_notes.append(f"idx={idx}: status={status} must have null predicted criterion fields")

        # Verify Stage 2 used the exact shared Top-2 groups from Stage 1
        if status != "SKIPPED_NOT_EVENT":
            scores = cls_obj.get("scores") or {}
            s1_fused = (scores.get("stage1") or {}).get("fused") or []
            s2_sample = (scores.get("stage2") or {}).get("sample") or []
            s2_rubric = (scores.get("stage2") or {}).get("rubric") or []
            selected_top_groups = {g["group_id"] for g in s1_fused if g.get("selected_for_stage2")}
            for entry in s2_sample + s2_rubric:
                if entry.get("group_id") not in selected_top_groups:
                    stage2_shared_groups_ok = False
                    error_notes.append(
                        f"idx={idx}: stage2 group '{entry.get('group_id')}' not in Stage 1 shared groups {selected_top_groups}"
                    )

        # Ensure no raw embedding vectors leaked into output JSON
        if "embedding" in cls_obj or "query_vector" in cls_obj:
            status_and_schema_ok = False
            error_notes.append(f"idx={idx}: raw embedding vector found in output JSON")

    report.add(
        "output_preserves_order_and_original_fields",
        order_and_fields_ok,
        "Original order, post_ids, and all event fields preserved"
        if order_and_fields_ok
        else "; ".join(error_notes[:5]),
    )
    report.add(
        "output_classification_schema_valid",
        status_and_schema_ok,
        "All classification objects conform to schema and status contracts"
        if status_and_schema_ok
        else "; ".join(error_notes[:5]),
    )
    report.add(
        "output_predicted_criteria_event_matchable",
        predicted_rubric_ok,
        "Every PREDICTED criterion exists in rubric.yaml with event_matchable: true"
        if predicted_rubric_ok
        else "; ".join(error_notes[:5]),
    )
    report.add(
        "output_stage2_uses_shared_stage1_groups",
        stage2_shared_groups_ok,
        "Both branches strictly restricted Stage 2 to shared Stage 1 candidate groups"
        if stage2_shared_groups_ok
        else "; ".join(error_notes[:5]),
    )

    # Verify strict JSON serializability with allow_nan=False (no NaN / Infinity)
    try:
        serialized = json.dumps(output_events, ensure_ascii=False, allow_nan=False)
        serialized.encode("utf-8")
        json_ok = True
        json_detail = f"Valid UTF-8 JSON ({len(serialized):,} chars), zero NaN/Infinity"
    except (ValueError, TypeError, UnicodeEncodeError) as exc:
        json_ok = False
        json_detail = f"Serialization failed: {exc}"
    report.add("output_valid_utf8_json_no_nan", json_ok, json_detail)

    return report
