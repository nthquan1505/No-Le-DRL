"""Stable boundary interface between Module 3 and Credit Calculation (`src/credit_calculation`).

Replace the implementation in `src/credit_calculation/credit_engine.py` with any object satisfying
`CreditCalculatorProtocol` without changing Module 3 code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

from src.schemas import CreditEstimate, EventNormalized, GroupShortfall


@dataclass
class LedgerEntry:
    """One activity or status already recorded in the student's semester DRL ledger."""

    rubric_item_id: str  # canonical '<major>.<minor>'
    points: float
    event_id: Optional[str] = None
    source: str = "system_detected"  # system_detected | self_reported | assumed_default | penalty


@dataclass
class UserLedger:
    """Student's current semester DRL state passed into Module 6."""

    entries: List[LedgerEntry] = field(default_factory=list)
    use_default_compliance_baseline: bool = True
    disabled_baseline_items: List[str] = field(default_factory=list)
    target_total: float = 80.0
    target_by_group: Dict[str, float] = field(default_factory=dict)


@dataclass
class LedgerEvaluation:
    """Deterministic semester DRL summary computed by Module 6."""

    raw_by_group: Dict[str, float]
    capped_by_group: Dict[str, float]
    total_penalties: float
    total_before_clamp: float
    total_score: float
    remaining_cap_by_group: Dict[str, float]


@runtime_checkable
class CreditCalculatorProtocol(Protocol):
    """Protocol implemented by Module 6 (local shared library or future remote adapter)."""

    def evaluate_ledger(self, ledger: UserLedger) -> LedgerEvaluation:
        """Compute current per-group capped totals, penalties, and clamped [0, 100] semester score."""

    def compute_group_shortfall(self, ledger: UserLedger) -> GroupShortfall:
        """Compute remaining group caps and target gaps used by P4 weights and P7 planner."""

    def estimate_event_credit(
        self,
        event: EventNormalized,
        rubric_item_id: Optional[str],
        ledger: Optional[UserLedger] = None,
        role: str = "participant",
        inferred_level: Optional[str] = None,
        inferred_duration: Optional[str] = None,
        explicit_points: Optional[float] = None,
    ) -> CreditEstimate:
        """Estimate eligibility, raw points, and cap-adjusted effective points for an event."""
