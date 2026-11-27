"""Boundary contract validators for Module 3 inputs, classification outputs, and rubric.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.schemas import (
    VALID_CLASSIFICATION_STATUSES,
    VALID_FEEDBACK_ACTIONS,
    VALID_ONLINE_MODES,
    VALID_PARENT_GROUPS,
    ClassificationResult,
    ClsSample,
    EventNormalized,
    FeedbackEvent,
    ScoringWeights,
    is_valid_item_id,
    parent_group_from_item_id,
)


@dataclass
class ValidationIssue:
    """A single validation error or warning."""

    field: str
    code: str
    message: str
    severity: str = "error"  # "error" | "warning"


@dataclass
class ValidationReport:
    """Collection of validation issues for a contract object or dataset."""

    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, field_name: str, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(field_name, code, message, severity="error"))

    def add_warning(self, field_name: str, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(field_name, code, message, severity="warning"))


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 datetime string (supporting trailing 'Z' on Python 3.9+)."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def validate_event_normalized(event: EventNormalized) -> ValidationReport:
    """Validate an upstream EventNormalized contract payload (D-6)."""
    report = ValidationReport()

    if not event.event_id or not event.event_id.strip():
        report.add_error("event_id", "missing_required", "event_id must be a non-empty string.")
    if not event.title or not event.title.strip():
        report.add_error("title", "missing_required", "title must be a non-empty string.")

    parsed_dates: Dict[str, Optional[datetime]] = {}
    for dt_field in ("start_datetime", "end_datetime", "reg_start_datetime", "reg_end_datetime"):
        raw_val = getattr(event, dt_field)
        if raw_val is None:
            parsed_dates[dt_field] = None
            continue
        try:
            parsed_dates[dt_field] = parse_iso_datetime(raw_val)
        except ValueError as exc:
            parsed_dates[dt_field] = None
            report.add_error(dt_field, "invalid_datetime", f"Invalid ISO-8601 datetime: {exc}")

    start_dt = parsed_dates["start_datetime"]
    end_dt = parsed_dates["end_datetime"]
    reg_start_dt = parsed_dates["reg_start_datetime"]
    reg_end_dt = parsed_dates["reg_end_datetime"]

    if start_dt and end_dt and end_dt < start_dt:
        report.add_error("end_datetime", "end_before_start", "end_datetime is earlier than start_datetime.")
    if reg_start_dt and reg_end_dt and reg_end_dt < reg_start_dt:
        report.add_error(
            "reg_end_datetime",
            "reg_end_before_reg_start",
            "reg_end_datetime is earlier than reg_start_datetime.",
        )
    if reg_end_dt and start_dt and reg_end_dt > start_dt:
        report.add_warning(
            "reg_end_datetime",
            "reg_end_after_start",
            "reg_end_datetime is after start_datetime (possible late registration or extraction anomaly).",
        )

    if event.online_mode is not None and event.online_mode not in VALID_ONLINE_MODES:
        report.add_error(
            "online_mode",
            "invalid_enum",
            f"online_mode '{event.online_mode}' must be one of {VALID_ONLINE_MODES}.",
        )

    if event.online_mode == "online" and event.is_online is False:
        report.add_error("is_online", "mode_mismatch", "online_mode='online' conflicts with is_online=False.")
    elif event.online_mode == "offline" and event.is_online is True:
        report.add_error("is_online", "mode_mismatch", "online_mode='offline' conflicts with is_online=True.")

    if event.fee_vnd is not None and event.fee_vnd < 0:
        report.add_error("fee_vnd", "negative_value", "fee_vnd cannot be negative.")
    if event.capacity is not None and event.capacity < 0:
        report.add_error("capacity", "negative_value", "capacity cannot be negative.")
    if event.seats_left is not None and event.seats_left < 0:
        report.add_error("seats_left", "negative_value", "seats_left cannot be negative.")
    if (
        event.capacity is not None
        and event.seats_left is not None
        and event.seats_left > event.capacity
    ):
        report.add_error("seats_left", "exceeds_capacity", "seats_left cannot exceed capacity.")

    return report


def validate_cls_sample(
    sample: ClsSample,
    allowed_parent_groups: Sequence[str] = VALID_PARENT_GROUPS,
) -> ValidationReport:
    """Validate a single ClsIndex hierarchical sample (T-1, T-4)."""
    report = ValidationReport()

    if not sample.sample_id or not sample.sample_id.strip():
        report.add_error("sample_id", "missing_required", "sample_id must be non-empty.")
    if not sample.text or not sample.text.strip():
        report.add_error("text", "missing_required", "text must be non-empty.")
    if not is_valid_item_id(sample.rubric_item_id):
        report.add_error(
            "rubric_item_id",
            "invalid_item_id",
            f"rubric_item_id '{sample.rubric_item_id}' must match '<major>.<minor>' (e.g. '1.6').",
        )
    else:
        expected_parent = parent_group_from_item_id(sample.rubric_item_id)
        if sample.parent_group_id != expected_parent:
            report.add_error(
                "parent_group_id",
                "parent_child_mismatch",
                f"rubric_item_id '{sample.rubric_item_id}' belongs to '{expected_parent}', "
                f"not '{sample.parent_group_id}'.",
            )
    if sample.parent_group_id not in allowed_parent_groups:
        report.add_error(
            "parent_group_id",
            "disallowed_parent_group",
            f"parent_group_id '{sample.parent_group_id}' not in allowed groups {tuple(allowed_parent_groups)}.",
        )
    return report


def validate_classification_result(result: ClassificationResult) -> ValidationReport:
    """Validate D-2 hierarchical classification invariants on a ClassificationResult."""
    report = ValidationReport()

    if not result.event_id or not result.event_id.strip():
        report.add_error("event_id", "missing_required", "event_id must be non-empty.")
    if result.status not in VALID_CLASSIFICATION_STATUSES:
        report.add_error(
            "status",
            "invalid_status",
            f"status '{result.status}' must be one of {VALID_CLASSIFICATION_STATUSES}.",
        )

    for grp in result.selected_candidate_groups:
        if grp not in VALID_PARENT_GROUPS:
            report.add_error(
                "selected_candidate_groups",
                "invalid_parent_group",
                f"Invalid candidate parent group '{grp}'.",
            )
    for grp in result.stage1_scores:
        if grp not in VALID_PARENT_GROUPS:
            report.add_error(
                "stage1_scores",
                "invalid_parent_group",
                f"stage1_scores key '{grp}' must be a valid parent group ID.",
            )
    for item_id in result.stage2_scores:
        if not is_valid_item_id(item_id):
            report.add_error(
                "stage2_scores",
                "invalid_item_id",
                f"stage2_scores key '{item_id}' must be a canonical '<major>.<minor>' item ID.",
            )

    if result.rubric_item_id is not None:
        if not is_valid_item_id(result.rubric_item_id):
            report.add_error(
                "rubric_item_id",
                "invalid_item_id",
                f"rubric_item_id '{result.rubric_item_id}' must match '<major>.<minor>'.",
            )
        else:
            expected_parent = parent_group_from_item_id(result.rubric_item_id)
            if result.parent_group_id != expected_parent:
                report.add_error(
                    "parent_group_id",
                    "parent_child_mismatch",
                    f"rubric_item_id '{result.rubric_item_id}' requires parent_group_id '{expected_parent}', "
                    f"got '{result.parent_group_id}'.",
                )
            if (
                result.selected_candidate_groups
                and expected_parent not in result.selected_candidate_groups
            ):
                report.add_error(
                    "selected_candidate_groups",
                    "parent_not_in_stage1_candidates",
                    f"Predicted parent_group_id '{expected_parent}' was not in Stage 1 candidate groups "
                    f"{result.selected_candidate_groups}.",
                )

    if not result.is_calibrated and result.calibrated_confidence is not None:
        report.add_error(
            "calibrated_confidence",
            "uncalibrated_probability_leak",
            "calibrated_confidence must be None when is_calibrated is False (D-2 Requirement 5).",
        )
    if result.is_calibrated:
        if result.calibrated_confidence is None or not (0.0 <= result.calibrated_confidence <= 1.0):
            report.add_error(
                "calibrated_confidence",
                "invalid_probability_range",
                "calibrated_confidence must be in [0.0, 1.0] when is_calibrated is True.",
            )

    return report


def validate_scoring_weights(weights: ScoringWeights, tol: float = 1e-6) -> ValidationReport:
    """Validate D-5 four-weight scoring invariants."""
    report = ValidationReport()
    for name in ("w1_gain", "w2_interest", "w3_fit", "w4_urgency"):
        val = getattr(weights, name)
        if val < 0.0:
            report.add_error(name, "negative_weight", f"{name} must be >= 0.0, got {val}.")
    if abs(weights.total() - 1.0) > tol:
        report.add_error(
            "weights",
            "weights_not_normalized",
            f"Weights w1..w4 must sum to 1.0 (±{tol}), got {weights.total():.6f}.",
        )
    return report


def validate_feedback_event(event: FeedbackEvent) -> ValidationReport:
    """Validate Feedback Store event payloads including label_correction lineage."""
    report = ValidationReport()
    if not event.user_id.strip():
        report.add_error("user_id", "missing_required", "user_id must be non-empty.")
    if not event.event_id.strip():
        report.add_error("event_id", "missing_required", "event_id must be non-empty.")
    if event.action not in VALID_FEEDBACK_ACTIONS:
        report.add_error(
            "action",
            "invalid_action",
            f"action '{event.action}' must be one of {VALID_FEEDBACK_ACTIONS}.",
        )
    try:
        parse_iso_datetime(event.timestamp)
    except ValueError as exc:
        report.add_error("timestamp", "invalid_datetime", f"Invalid timestamp: {exc}")

    if event.action == "label_correction":
        if not event.corrected_item_id or not is_valid_item_id(event.corrected_item_id):
            report.add_error(
                "corrected_item_id",
                "invalid_item_id",
                "label_correction requires a valid canonical corrected_item_id (e.g. '1.6').",
            )
        else:
            expected_group = parent_group_from_item_id(event.corrected_item_id)
            if event.corrected_group_id and event.corrected_group_id != expected_group:
                report.add_error(
                    "corrected_group_id",
                    "parent_child_mismatch",
                    f"corrected_item_id '{event.corrected_item_id}' belongs to '{expected_group}', "
                    f"not '{event.corrected_group_id}'.",
                )
    return report


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load a YAML file using PyYAML."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load rubric.yaml. Install with `pip install PyYAML`."
        ) from exc
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level YAML mapping in {path}.")
    return data


def validate_rubric_dict(rubric: Dict[str, Any]) -> Tuple[ValidationReport, Dict[str, List[str]]]:
    """Validate rubric.yaml structural integrity and extract parent -> child ref mapping."""
    report = ValidationReport()
    hierarchy: Dict[str, List[str]] = {}

    groups = rubric.get("groups")
    items = rubric.get("items")
    scoring = rubric.get("scoring")

    if not isinstance(groups, dict) or not groups:
        report.add_error("groups", "missing_section", "rubric.yaml must define 'groups'.")
        return report, hierarchy
    if not isinstance(items, dict) or not items:
        report.add_error("items", "missing_section", "rubric.yaml must define 'items'.")
        return report, hierarchy
    if not isinstance(scoring, dict):
        report.add_error("scoring", "missing_section", "rubric.yaml must define 'scoring'.")
    else:
        if scoring.get("total_max") != 100:
            report.add_error("scoring.total_max", "unexpected_value", "Expected scoring.total_max == 100.")
        if scoring.get("floor") != 0:
            report.add_error("scoring.floor", "unexpected_value", "Expected scoring.floor == 0.")

    seen_refs: Dict[str, str] = {}
    for item_key, spec in items.items():
        if not isinstance(spec, dict):
            report.add_error(f"items.{item_key}", "invalid_item", "Item spec must be a mapping.")
            continue
        ref = spec.get("ref")
        group = spec.get("group")

        if not isinstance(ref, str) or not is_valid_item_id(ref):
            report.add_error(f"items.{item_key}.ref", "invalid_item_id", f"Invalid ref '{ref}'.")
            continue
        if ref in seen_refs:
            report.add_error(
                f"items.{item_key}.ref",
                "duplicate_ref",
                f"Duplicate ref '{ref}' (already used by '{seen_refs[ref]}').",
            )
        seen_refs[ref] = item_key

        if group not in groups:
            report.add_error(
                f"items.{item_key}.group",
                "unknown_group",
                f"Group '{group}' not found in rubric.groups.",
            )
        else:
            expected_group = parent_group_from_item_id(ref)
            if group != expected_group:
                report.add_error(
                    f"items.{item_key}.group",
                    "parent_child_mismatch",
                    f"Item ref '{ref}' expects group '{expected_group}', got '{group}'.",
                )
            hierarchy.setdefault(group, []).append(ref)

        for target_ref_key in spec.get("exclusive_with") or []:
            if target_ref_key not in items:
                report.add_error(
                    f"items.{item_key}.exclusive_with",
                    "broken_reference",
                    f"exclusive_with target '{target_ref_key}' not found in items.",
                )

        role_items = spec.get("role_items") or {}
        if isinstance(role_items, dict):
            for role_name, target_item in role_items.items():
                if target_item not in items:
                    report.add_error(
                        f"items.{item_key}.role_items.{role_name}",
                        "broken_reference",
                        f"role_items target '{target_item}' not found in items.",
                    )

    for grp in hierarchy:
        hierarchy[grp].sort(key=lambda r: tuple(int(p) for p in r.split(".")))

    return report, hierarchy


def validate_rubric_file(path: Path) -> Tuple[ValidationReport, Dict[str, List[str]]]:
    """Load and validate rubric.yaml from disk."""
    rubric = load_yaml_file(path)
    return validate_rubric_dict(rubric)
