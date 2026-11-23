"""Typed data contracts for No-Le-DRL Module 3.

Aligned with:
- D-1, D-2: Two-stage hierarchical KNN classification with distinct Stage 1 & Stage 2 scores.
- D-3, D-4: Strict boundary between P2 classification and Module 6 shared credit engine.
- D-5: Multi-criteria scoring using 4 weights (w1..w4) + additive unweighted Penalty.
- D-6: Module 3 consumes normalized EventNormalized contracts supplied by upstream M1/M2.
- T-1..T-4: Full rubric taxonomy using canonical '<major>.<minor>' item IDs (e.g. '1.6', '1.7')
  mapped to parent groups C1..C5 (and P when present in rubric).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

ITEM_ID_PATTERN = re.compile(r"^([1-6])\.([1-9]\d*)$")

MAJOR_TO_PARENT_GROUP: Dict[str, str] = {
    "1": "C1",
    "2": "C2",
    "3": "C3",
    "4": "C4",
    "5": "C5",
    "6": "P",
}

VALID_PARENT_GROUPS: Tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5", "P")
ACTIVE_CLASSIFIER_PARENT_GROUPS: Tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5")
VALID_ONLINE_MODES: Tuple[str, ...] = ("online", "offline", "hybrid")
VALID_CLASSIFICATION_STATUSES: Tuple[str, ...] = (
    "accepted",
    "pending_review",
    "rejected",
    "uncalibrated",
)
VALID_FEEDBACK_ACTIONS: Tuple[str, ...] = (
    "click",
    "save",
    "dismiss",
    "joined",
    "label_correction",
)


def is_valid_item_id(item_id: str) -> bool:
    """Return True if item_id follows canonical '<major>.<minor>' format (e.g. '1.6')."""
    if not isinstance(item_id, str):
        return False
    return bool(ITEM_ID_PATTERN.match(item_id.strip()))


def parent_group_from_item_id(item_id: str) -> str:
    """Map canonical '<major>.<minor>' item ID to its parent group ('C1'..'C5', 'P')."""
    if not isinstance(item_id, str):
        raise ValueError(f"Expected string item_id, got {type(item_id).__name__}")
    m = ITEM_ID_PATTERN.match(item_id.strip())
    if not m:
        raise ValueError(
            f"Invalid canonical rubric item_id '{item_id}'. Expected '<major>.<minor>' (e.g. '1.6')."
        )
    return MAJOR_TO_PARENT_GROUP[m.group(1)]


@dataclass
class TargetAudienceStructured:
    """Structured eligibility metadata supplied by upstream M1/M2 (D-6)."""

    raw_texts: List[str] = field(default_factory=list)
    cohorts: List[str] = field(default_factory=list)
    faculties: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    all_students: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TargetAudienceStructured":
        if not data:
            return cls()
        return cls(
            raw_texts=list(data.get("raw_texts", [])),
            cohorts=list(data.get("cohorts", [])),
            faculties=list(data.get("faculties", [])),
            classes=list(data.get("classes", [])),
            all_students=bool(data.get("all_students", False)),
        )


@dataclass
class EventNormalized:
    """Canonical upstream event contract produced by M1 & M2 and consumed by Module 3 (D-6)."""

    event_id: str
    title: str
    description: str = ""
    organizer: str = ""
    start_datetime: Optional[str] = None
    end_datetime: Optional[str] = None
    reg_start_datetime: Optional[str] = None
    reg_end_datetime: Optional[str] = None
    online_mode: Optional[str] = None
    is_online: Optional[bool] = None
    location: Optional[str] = None
    fee_vnd: Optional[int] = None
    fee_raw: Optional[str] = None
    target_audience: TargetAudienceStructured = field(default_factory=TargetAudienceStructured)
    capacity: Optional[int] = None
    seats_left: Optional[int] = None
    registration_url: Optional[str] = None
    source_type: str = "unknown"
    source_url: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    missing_field_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_classification_text(self, role: Optional[str] = None) -> str:
        """Format normalized event fields into a single text representation for P2.1 embedding."""
        parts: List[str] = []
        if self.title.strip():
            parts.append(self.title.strip())
        if role and role.strip():
            parts.append(f"Vai trò: {role.strip()}")
        if self.organizer.strip():
            parts.append(f"Đơn vị tổ chức: {self.organizer.strip()}")
        if self.description.strip():
            parts.append(self.description.strip())
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventNormalized":
        return cls(
            event_id=str(data["event_id"]),
            title=str(data["title"]),
            description=str(data.get("description") or ""),
            organizer=str(data.get("organizer") or ""),
            start_datetime=data.get("start_datetime"),
            end_datetime=data.get("end_datetime"),
            reg_start_datetime=data.get("reg_start_datetime"),
            reg_end_datetime=data.get("reg_end_datetime"),
            online_mode=data.get("online_mode"),
            is_online=data.get("is_online"),
            location=data.get("location"),
            fee_vnd=data.get("fee_vnd"),
            fee_raw=data.get("fee_raw"),
            target_audience=TargetAudienceStructured.from_dict(data.get("target_audience")),
            capacity=data.get("capacity"),
            seats_left=data.get("seats_left"),
            registration_url=data.get("registration_url"),
            source_type=str(data.get("source_type") or "unknown"),
            source_url=data.get("source_url"),
            categories=list(data.get("categories") or []),
            missing_field_flags=list(data.get("missing_field_flags") or []),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ClsSample:
    """A single labeled sample stored inside the hierarchical ClsIndex."""

    sample_id: str
    text: str
    title: str
    rubric_item_id: str
    parent_group_id: str
    organizer: str = ""
    role: str = ""
    source: str = "seed_demo"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClsSample":
        return cls(
            sample_id=str(data["sample_id"]),
            text=str(data["text"]),
            title=str(data["title"]),
            rubric_item_id=str(data["rubric_item_id"]),
            parent_group_id=str(data["parent_group_id"]),
            organizer=str(data.get("organizer") or ""),
            role=str(data.get("role") or ""),
            source=str(data.get("source") or "seed_demo"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ClassificationResult:
    """Output of the P2 two-stage hierarchical KNN classifier (D-2, T-4).

    Guarantees:
    - stage1_scores and stage2_scores are always kept distinct.
    - calibrated_confidence must remain None when is_calibrated is False.
    - rubric_item_id uses canonical '<major>.<minor>' format (e.g. '1.6').
    """

    event_id: str
    rubric_item_id: Optional[str]
    parent_group_id: Optional[str]
    selected_candidate_groups: List[str] = field(default_factory=list)
    stage1_scores: Dict[str, float] = field(default_factory=dict)
    stage2_scores: Dict[str, float] = field(default_factory=dict)
    aggregated_scores: Optional[Dict[str, float]] = None
    calibrated_confidence: Optional[float] = None
    is_calibrated: bool = False
    status: str = "uncalibrated"
    top_k_items: List[Dict[str, Any]] = field(default_factory=list)
    model_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassificationResult":
        return cls(
            event_id=str(data["event_id"]),
            rubric_item_id=data.get("rubric_item_id"),
            parent_group_id=data.get("parent_group_id"),
            selected_candidate_groups=list(data.get("selected_candidate_groups") or []),
            stage1_scores=dict(data.get("stage1_scores") or {}),
            stage2_scores=dict(data.get("stage2_scores") or {}),
            aggregated_scores=(
                dict(data["aggregated_scores"])
                if data.get("aggregated_scores") is not None
                else None
            ),
            calibrated_confidence=data.get("calibrated_confidence"),
            is_calibrated=bool(data.get("is_calibrated", False)),
            status=str(data.get("status") or "uncalibrated"),
            top_k_items=list(data.get("top_k_items") or []),
            model_version=str(data.get("model_version") or ""),
        )


@dataclass
class CreditEstimate:
    """Output of Module 6 Shared Python Library (D-3, D-4)."""

    event_id: str
    rubric_item_id: Optional[str]
    group_id: Optional[str]
    is_credit_eligible: bool
    expected_points: float
    min_points: float
    max_points: float
    effective_points: float
    capped_by: Optional[str] = None
    source: str = "rule"
    warning_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroupShortfall:
    """Per-group DRL progress and remaining capacity computed by Module 6."""

    current_by_group: Dict[str, float] = field(default_factory=dict)
    remaining_cap_by_group: Dict[str, float] = field(default_factory=dict)
    target_gap_by_group: Dict[str, float] = field(default_factory=dict)
    current_total: float = 0.0
    target_total: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoringWeights:
    """Four scoring weights for P6 (D-5): w1..w4 + unweighted Penalty."""

    w1_gain: float = 0.40
    w2_interest: float = 0.25
    w3_fit: float = 0.20
    w4_urgency: float = 0.15

    def total(self) -> float:
        return self.w1_gain + self.w2_interest + self.w3_fit + self.w4_urgency

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class BusyInterval:
    """Time interval when the student is unavailable (from Module 4)."""

    start_datetime: str
    end_datetime: str
    label: str = ""
    location: Optional[str] = None
    is_exam: bool = False


@dataclass
class ScheduleContext:
    """Timetable and exam context supplied by Module 4."""

    busy_intervals: List[BusyInterval] = field(default_factory=list)
    exam_weeks: List[Tuple[str, str]] = field(default_factory=list)
    default_travel_buffer_min: int = 30


@dataclass
class UserProfile:
    """Adaptive user profile state maintained by P4."""

    user_id: str
    cohort: Optional[str] = None
    faculty: Optional[str] = None
    class_id: Optional[str] = None
    interest_tags: List[str] = field(default_factory=list)
    user_vec: Optional[List[float]] = None
    target_total: float = 80.0
    target_by_group: Dict[str, float] = field(default_factory=dict)
    max_budget_vnd: Optional[int] = None
    max_travel_min: Optional[int] = None
    hidden_event_ids: List[str] = field(default_factory=list)
    participated_event_ids: List[str] = field(default_factory=list)
    saved_event_ids: List[str] = field(default_factory=list)
    weights: ScoringWeights = field(default_factory=ScoringWeights)


@dataclass
class DropReason:
    """Machine-readable hard-filter exclusion reason from P5."""

    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredEvent:
    """Multi-criteria scored event produced by P6."""

    event_id: str
    score_total: float
    gain: float
    interest: float
    fit: float
    urgency: float
    penalty: float
    weights: ScoringWeights
    rubric_item_id: Optional[str] = None
    parent_group_id: Optional[str] = None


@dataclass
class RecommendationCard:
    """User-facing explained recommendation output from P7 & P8."""

    event_id: str
    title: str
    score_total: float
    score_breakdown: Dict[str, float]
    top_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_exploration_slot: bool = False
    deadline_alert: Optional[str] = None
    needs_review: bool = False


@dataclass
class FeedbackEvent:
    """User interaction or label correction event written to Feedback Store."""

    user_id: str
    event_id: str
    action: str
    timestamp: str
    corrected_item_id: Optional[str] = None
    corrected_group_id: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
