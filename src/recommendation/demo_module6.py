"""Demo Module 6 (`DemoCreditProvider`) — Replaceable Credit & Mandatory Evaluator.

Wraps the deterministic `RubricCreditEngine` (`data/raw/rubric.yaml`, HCMUS caps
C1=30, C2=20, C3=25, C4=25, C5=30) so recommendation demos can directly configure
current per-group DRL points (`current_by_group`) and `target_total` without building
full evidence ledgers, while preserving full `CreditCalculatorProtocol` compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.credit_calculation import (
    AttributeExtractor,
    CreditCalculatorProtocol,
    LedgerEntry,
    LedgerEvaluation,
    RubricCreditEngine,
    UserLedger,
)
from src.schemas import CreditEstimate, EventNormalized, GroupShortfall

POSITIVE_GROUPS = ("C1", "C2", "C3", "C4", "C5")

# Non-exclusive baseline item refs used purely to inject group-level totals into UserLedger
_SYNTHETIC_GROUP_REF: Dict[str, str] = {
    "C1": "1.1",
    "C2": "2.1",
    "C3": "3.7",
    "C4": "4.1",
    "C5": "5.7",
}

DEMO_CREDIT_PRESETS: Dict[str, Dict[str, Any]] = {
    "mid_semester": {
        "id": "mid_semester",
        "label": "Giữa kỳ — còn thiếu C3/C4 (57/85đ)",
        "description": "C1 còn dư 8đ, C3 và C4 còn thiếu nhiều so với mục tiêu 85 điểm.",
        "current_by_group": {"C1": 22.0, "C2": 15.0, "C3": 10.0, "C4": 10.0, "C5": 0.0},
        "target_total": 85.0,
        "target_by_group": {"C1": 25.0, "C2": 18.0, "C3": 22.0, "C4": 20.0, "C5": 0.0},
    },
    "c1_capped": {
        "id": "c1_capped",
        "label": "Chạm trần C1 (30/30đ) — cần C3/C4",
        "description": "C1 đã đạt trần tối đa 30/30đ; P5 sẽ lọc toàn bộ hoạt động thuộc nhóm C1.",
        "current_by_group": {"C1": 30.0, "C2": 15.0, "C3": 10.0, "C4": 10.0, "C5": 0.0},
        "target_total": 85.0,
        "target_by_group": {"C1": 30.0, "C2": 18.0, "C3": 22.0, "C4": 20.0, "C5": 0.0},
    },
    "near_target": {
        "id": "near_target",
        "label": "Gần đạt mục tiêu (83/85đ)",
        "description": "Tổng điểm đã gần đạt mục tiêu 85; phù hợp test chế độ ưu tiên sở thích.",
        "current_by_group": {"C1": 25.0, "C2": 18.0, "C3": 22.0, "C4": 18.0, "C5": 0.0},
        "target_total": 85.0,
        "target_by_group": {"C1": 26.0, "C2": 18.0, "C3": 23.0, "C4": 20.0, "C5": 0.0},
    },
}


@dataclass(frozen=True)
class MandatoryPenaltyInfo:
    """High-mandatory penalty-avoidance metadata resolved by Demo Module 6."""

    is_mandatory: bool
    penalty_if_missed: Optional[float] = None  # negative points if missed (e.g. -20.0)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_mandatory": self.is_mandatory,
            "penalty_if_missed": self.penalty_if_missed,
            "reason": self.reason,
        }


class DemoCreditProvider(CreditCalculatorProtocol):
    """Demo Module 6 wrapper backed by real `RubricCreditEngine` & `data/raw/rubric.yaml`."""

    def __init__(
        self,
        credit_engine: Optional[RubricCreditEngine] = None,
        *,
        enable_item_2_2_penalty_link: bool = False,
    ) -> None:
        if credit_engine is None:
            credit_engine = RubricCreditEngine(attribute_extractor=AttributeExtractor())
        self.engine = credit_engine
        self.enable_item_2_2_penalty_link = enable_item_2_2_penalty_link

    @property
    def group_caps(self) -> Dict[str, float]:
        return dict(self.engine.group_caps)

    @property
    def total_max(self) -> float:
        return float(self.engine.total_max)

    @staticmethod
    def available_presets() -> List[Dict[str, Any]]:
        return [dict(v) for v in DEMO_CREDIT_PRESETS.values()]

    def build_ledger_from_group_totals(
        self,
        current_by_group: Dict[str, float],
        *,
        target_total: float = 85.0,
        target_by_group: Optional[Dict[str, float]] = None,
    ) -> UserLedger:
        """Build a deterministic `UserLedger` directly from demo C1..C5 group totals."""
        entries: List[LedgerEntry] = []
        for grp in POSITIVE_GROUPS:
            pts = max(0.0, float(current_by_group.get(grp, 0.0)))
            if pts > 0.0:
                entries.append(
                    LedgerEntry(
                        rubric_item_id=_SYNTHETIC_GROUP_REF[grp],
                        points=pts,
                        source="demo_group_total",
                    )
                )

        resolved_targets: Dict[str, float] = {}
        if target_by_group:
            for grp in POSITIVE_GROUPS:
                if grp in target_by_group and target_by_group[grp] is not None:
                    resolved_targets[grp] = min(
                        self.engine.group_caps[grp],
                        max(0.0, float(target_by_group[grp])),
                    )

        return UserLedger(
            entries=entries,
            use_default_compliance_baseline=False,
            target_total=max(0.0, min(self.total_max, float(target_total))),
            target_by_group=resolved_targets,
        )

    def evaluate_ledger(self, ledger: UserLedger) -> LedgerEvaluation:
        return self.engine.evaluate_ledger(ledger)

    def compute_group_shortfall(self, ledger: UserLedger) -> GroupShortfall:
        return self.engine.compute_group_shortfall(ledger)

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
        # Fallback for DB-only rows that have `events.category (1..5)` + `events.min_credit`
        # without a fine-grained criterion_ref yet (per SYSTEM_ARCHITECTURE.md).
        meta = event.metadata or {}
        if not rubric_item_id and meta.get("db_category") is not None and meta.get("db_min_credit") is not None:
            cat_int = int(meta["db_category"])
            grp = f"C{cat_int}" if 1 <= cat_int <= 5 else None
            raw_pts = max(0.0, float(meta["db_min_credit"]))
            eff_pts = raw_pts
            capped_by: Optional[str] = None
            if ledger is not None and grp in self.engine.group_caps:
                eval_res = self.engine.evaluate_ledger(ledger)
                rem_group = eval_res.remaining_cap_by_group.get(grp, 0.0)
                rem_total = max(0.0, self.total_max - eval_res.total_score)
                if rem_group <= 0.0 and raw_pts > 0.0:
                    eff_pts, capped_by = 0.0, "group"
                elif rem_total <= 0.0 and raw_pts > 0.0:
                    eff_pts, capped_by = 0.0, "total"
                else:
                    if eff_pts > rem_group:
                        eff_pts, capped_by = rem_group, "group"
                    if eff_pts > rem_total:
                        eff_pts, capped_by = rem_total, "total"
            return CreditEstimate(
                event_id=event.event_id,
                rubric_item_id=None,
                group_id=grp,
                is_credit_eligible=raw_pts > 0.0 and (eff_pts > 0.0 or ledger is None),
                expected_points=raw_pts,
                min_points=raw_pts,
                max_points=raw_pts,
                effective_points=eff_pts,
                capped_by=capped_by,
                source="explicit",
                warning_flags=["db_min_credit_group_level"],
            )

        return self.engine.estimate_event_credit(
            event=event,
            rubric_item_id=rubric_item_id,
            ledger=ledger,
            role=role,
            inferred_level=inferred_level,
            inferred_duration=inferred_duration,
            explicit_points=explicit_points,
        )

    def get_mandatory_penalty_info(
        self,
        event: EventNormalized,
        credit: Optional[CreditEstimate] = None,
    ) -> MandatoryPenaltyInfo:
        """Identify high-mandatory events where skipping incurs DRL penalty (e.g. 6.4)."""
        meta = event.metadata or {}
        meta_mandatory = bool(meta.get("is_mandatory", False))
        meta_penalty = meta.get("penalty_if_missed_points")

        ref = (credit.rubric_item_id if credit is not None else None) or str(
            meta.get("rubric_item_id") or ""
        ).strip()
        warnings = list(credit.warning_flags) if credit is not None else []

        # 1. Rubric Group P mandatory_event (e.g. 6.4 Sinh hoat cong dan -> -20 pts)
        if "mandatory_event_avoid_penalty" in warnings or ref == "6.4":
            deduct = -20.0
            spec_lookup = self.engine.get_item_spec(ref or "6.4")
            if spec_lookup is not None:
                _, spec = spec_lookup
                pts_table = (spec.get("points") or {}).get("self") or {}
                neg_vals = [float(v) for v in pts_table.values() if float(v) < 0]
                if neg_vals:
                    deduct = min(neg_vals)  # worst-case penalty if missed (e.g. -20.0)
            if meta_penalty is not None:
                deduct = -abs(float(meta_penalty))
            return MandatoryPenaltyInfo(
                is_mandatory=True,
                penalty_if_missed=deduct,
                reason=f"Hoạt động bắt buộc theo quy chế (mục {ref or '6.4'}: bỏ lỡ bị trừ {abs(deduct):.0f}đ)",
            )

        # 2. Optional 2.2 -> 6.1 link (-5 pts if skipped)
        if self.enable_item_2_2_penalty_link and ref == "2.2":
            return MandatoryPenaltyInfo(
                is_mandatory=True,
                penalty_if_missed=-5.0,
                reason="Đánh giá môn học bắt buộc (mục 2.2 / 6.1: bỏ lỡ bị trừ 5đ)",
            )

        # 3. Explicit upstream/demo metadata flag `is_mandatory`
        if meta_mandatory:
            deduct_val = -abs(float(meta_penalty)) if meta_penalty is not None else None
            msg = (
                f"Hoạt động bắt buộc (bỏ lỡ bị trừ {abs(deduct_val):.0f}đ)"
                if deduct_val is not None
                else "Hoạt động được đánh dấu bắt buộc cao"
            )
            return MandatoryPenaltyInfo(
                is_mandatory=True,
                penalty_if_missed=deduct_val,
                reason=msg,
            )

        return MandatoryPenaltyInfo(is_mandatory=False, penalty_if_missed=None, reason="")
