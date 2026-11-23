"""Deterministic Rubric & Credit Eligibility Engine (Module 6 — Replaceable Local Library).

Strictly implements D-3 & D-4:
- Driven purely by data/raw/rubric.yaml (zero dependency on P2 classifier internals or confidence).
- Looks up items by canonical '<major>.<minor>' ID (T-4).
- Applies Q2/Q3 rubric rules: group caps (30/20/25/25/30) applied BEFORE unlimited negative
  penalties (P), then clamped to [0, 100].
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from configs.model_config import RUBRIC_YAML_PATH
from module6.attribute_extractor import AttributeExtractor, load_yaml_safe
from module6.interface import (
    CreditCalculatorProtocol,
    LedgerEntry,
    LedgerEvaluation,
    UserLedger,
)
from src.schemas import CreditEstimate, EventNormalized, GroupShortfall

POSITIVE_GROUPS: Tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5")


class RubricCreditEngine(CreditCalculatorProtocol):
    """Deterministic DRL credit evaluator backed by data/raw/rubric.yaml."""

    def __init__(
        self,
        rubric_path: Path = RUBRIC_YAML_PATH,
        attribute_extractor: Optional[AttributeExtractor] = None,
    ) -> None:
        self.rubric_path = rubric_path
        self.rubric = load_yaml_safe(rubric_path)
        self.attribute_extractor = attribute_extractor

        self._scoring = self.rubric["scoring"]
        self._groups = self.rubric["groups"]
        self._level_buckets: Dict[str, str] = dict(self.rubric.get("level_buckets") or {})
        self._items_by_key: Dict[str, Dict[str, Any]] = dict(self.rubric["items"])
        self._items_by_ref: Dict[str, Tuple[str, Dict[str, Any]]] = {
            str(spec["ref"]): (key, spec) for key, spec in self._items_by_key.items()
        }

        self.total_max: float = float(self._scoring.get("total_max", 100))
        self.floor: float = float(self._scoring.get("floor", 0))
        self.group_caps: Dict[str, float] = {
            grp: float(self._groups[grp]["max"])
            for grp in POSITIVE_GROUPS
            if self._groups.get(grp, {}).get("max") is not None
        }

        # Baseline assumed defaults (C1:15, C2:15, C3:10, C4:10 -> 50 pts total)
        self._baseline_items: Dict[str, Tuple[str, float]] = {}
        for key, spec in self._items_by_key.items():
            default_cfg = spec.get("default_assumption")
            if isinstance(default_cfg, dict) and default_cfg.get("enabled"):
                ref = str(spec["ref"])
                self._baseline_items[ref] = (
                    str(spec["group"]),
                    float(default_cfg["assumed_points"]),
                )

    def get_item_spec(self, rubric_item_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Look up (yaml_key, spec) by canonical '<major>.<minor>' ref."""
        return self._items_by_ref.get(str(rubric_item_id).strip())

    def _map_level_bucket(self, raw_level: Optional[str]) -> Optional[str]:
        if raw_level is None:
            return None
        lv = raw_level.strip().lower()
        if lv in ("lop", "khoa", "truong_tro_len"):
            return lv
        return self._level_buckets.get(lv)

    def _resolve_points_range(
        self,
        spec: Dict[str, Any],
        role: str,
        level_bucket: Optional[str],
        duration_bucket: Optional[str],
    ) -> Tuple[float, float, float, str, List[str]]:
        """Resolve (expected, min_pts, max_pts, source, warnings) for an item spec."""
        warnings: List[str] = []
        points_table = spec.get("points")
        if not isinstance(points_table, dict) or not points_table:
            warnings.append("no_point_table_for_item")
            return 0.0, 0.0, 0.0, "rule_fallback", warnings

        # Map role alias (e.g. C4.tinh_nguyen maps participant -> volunteer)
        role_alias = spec.get("role_alias") or {}
        effective_role = role_alias.get(role, role)
        if effective_role not in points_table:
            # Fallback to 'self' or single available role
            if "self" in points_table:
                effective_role = "self"
            elif "participant" in points_table:
                effective_role = "participant"
            else:
                effective_role = next(iter(points_table.keys()))
                warnings.append(f"role_fallback_to_{effective_role}")

        variant_map = points_table[effective_role]
        all_vals = [float(v) for v in variant_map.values()]
        min_pts, max_pts = min(all_vals), max(all_vals)
        variant_by = spec.get("variant_by", "none")

        if variant_by == "none" or "any" in variant_map:
            val = float(variant_map.get("any", min_pts))
            return val, val, val, "rule", warnings

        if variant_by == "level":
            if level_bucket and level_bucket in variant_map:
                val = float(variant_map[level_bucket])
                return val, val, val, "rule", warnings
            # Conservative fallback to minimum valid level when level cannot be inferred
            warnings.append("level_missing_conservative_fallback")
            return min_pts, min_pts, max_pts, "rule_fallback", warnings

        if variant_by == "duration":
            if duration_bucket and duration_bucket in variant_map:
                val = float(variant_map[duration_bucket])
                return val, val, val, "rule", warnings
            warnings.append("duration_missing_conservative_fallback")
            return min_pts, min_pts, max_pts, "rule_fallback", warnings

        # award_tier / outcome / other variants
        warnings.append(f"variant_{variant_by}_conservative_fallback")
        return min_pts, min_pts, max_pts, "rule_fallback", warnings

    def evaluate_ledger(self, ledger: UserLedger) -> LedgerEvaluation:
        """Apply rubric.yaml scoring formula: group caps -> subtract penalties -> clamp [0, 100]."""
        raw_by_group: Dict[str, float] = {grp: 0.0 for grp in POSITIVE_GROUPS}
        total_penalties = 0.0

        disabled_baseline = set(ledger.disabled_baseline_items)
        present_item_ids = {e.rubric_item_id for e in ledger.entries}

        if ledger.use_default_compliance_baseline:
            for ref, (grp, pts) in self._baseline_items.items():
                if ref not in disabled_baseline and ref not in present_item_ids:
                    raw_by_group[grp] += pts

        for entry in ledger.entries:
            lookup = self.get_item_spec(entry.rubric_item_id)
            if lookup is None:
                continue
            _, spec = lookup
            grp = str(spec["group"])
            if grp == "P" or entry.points < 0:
                total_penalties += float(entry.points)
            elif grp in raw_by_group:
                raw_by_group[grp] += float(entry.points)

        capped_by_group: Dict[str, float] = {}
        remaining_cap_by_group: Dict[str, float] = {}
        for grp in POSITIVE_GROUPS:
            cap = self.group_caps[grp]
            capped = min(raw_by_group[grp], cap)
            capped_by_group[grp] = capped
            remaining_cap_by_group[grp] = max(0.0, cap - capped)

        total_before_clamp = sum(capped_by_group.values()) + total_penalties
        total_score = min(self.total_max, max(self.floor, total_before_clamp))

        return LedgerEvaluation(
            raw_by_group=raw_by_group,
            capped_by_group=capped_by_group,
            total_penalties=total_penalties,
            total_before_clamp=total_before_clamp,
            total_score=total_score,
            remaining_cap_by_group=remaining_cap_by_group,
        )

    def compute_group_shortfall(self, ledger: UserLedger) -> GroupShortfall:
        """Compute per-group remaining cap and shortfall toward user's targets."""
        eval_res = self.evaluate_ledger(ledger)
        target_gap: Dict[str, float] = {}
        for grp in POSITIVE_GROUPS:
            target = float(ledger.target_by_group.get(grp, self.group_caps[grp]))
            target = min(target, self.group_caps[grp])
            target_gap[grp] = max(0.0, target - eval_res.capped_by_group[grp])

        return GroupShortfall(
            current_by_group=dict(eval_res.capped_by_group),
            remaining_cap_by_group=dict(eval_res.remaining_cap_by_group),
            target_gap_by_group=target_gap,
            current_total=eval_res.total_score,
            target_total=float(ledger.target_total),
        )

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
        """Deterministically evaluate credit eligibility and expected/effective points (D-3)."""
        if not rubric_item_id:
            return CreditEstimate(
                event_id=event.event_id,
                rubric_item_id=None,
                group_id=None,
                is_credit_eligible=False,
                expected_points=0.0,
                min_points=0.0,
                max_points=0.0,
                effective_points=0.0,
                capped_by="unclassified",
                source="rule",
                warning_flags=["missing_rubric_item_id"],
            )

        lookup = self.get_item_spec(rubric_item_id)
        if lookup is None:
            return CreditEstimate(
                event_id=event.event_id,
                rubric_item_id=rubric_item_id,
                group_id=None,
                is_credit_eligible=False,
                expected_points=0.0,
                min_points=0.0,
                max_points=0.0,
                effective_points=0.0,
                capped_by="unknown_item",
                source="rule",
                warning_flags=[f"unknown_rubric_item:{rubric_item_id}"],
            )

        item_key, spec = lookup
        group_id = str(spec["group"])
        kind = str(spec.get("kind") or "")
        warnings: List[str] = []

        # Penalty / mandatory event (Group P, e.g. 6.4 Sinh hoat cong dan): never awards +points
        if group_id == "P" or kind in ("penalty", "mandatory_event"):
            if kind == "mandatory_event":
                warnings.append("mandatory_event_avoid_penalty")
            else:
                warnings.append("penalty_item_not_positive_credit")
            return CreditEstimate(
                event_id=event.event_id,
                rubric_item_id=str(spec["ref"]),
                group_id=group_id,
                is_credit_eligible=False,
                expected_points=0.0,
                min_points=0.0,
                max_points=0.0,
                effective_points=0.0,
                capped_by="penalty_group",
                source="rule",
                warning_flags=warnings,
            )

        # Role redirection via role_items (e.g. participant 1.6 + role='organizer' -> 1.7)
        effective_spec = spec
        effective_key = item_key
        role_items = spec.get("role_items") or {}
        if role in role_items:
            redirect_key = role_items[role]
            if redirect_key in self._items_by_key:
                effective_key = redirect_key
                effective_spec = self._items_by_key[redirect_key]
                warnings.append(f"role_redirected_to:{effective_spec['ref']}")

        # Optional attribute extraction fallback if caller did not supply explicit attributes
        combined_text = f"{event.title} {event.description}"
        if self.attribute_extractor is not None:
            if explicit_points is None:
                ep = self.attribute_extractor.extract_explicit_points(combined_text)
                if ep["negated"]:
                    return CreditEstimate(
                        event_id=event.event_id,
                        rubric_item_id=str(effective_spec["ref"]),
                        group_id=str(effective_spec["group"]),
                        is_credit_eligible=False,
                        expected_points=0.0,
                        min_points=0.0,
                        max_points=0.0,
                        effective_points=0.0,
                        capped_by="explicit_negation",
                        source="explicit",
                        warning_flags=["explicit_non_drl_negation"],
                    )
                if ep["points"] is not None:
                    explicit_points = float(ep["points"])
            if inferred_level is None and effective_spec.get("variant_by") == "level":
                inferred_level, _ = self.attribute_extractor.extract_level(
                    combined_text, event.organizer
                )
            if inferred_duration is None and effective_spec.get("variant_by") == "duration":
                inferred_duration, is_prior = self.attribute_extractor.extract_duration_bucket(
                    combined_text
                )
                if is_prior:
                    warnings.append("duration_inferred_from_prior")

        # Check symmetric exclusive_with against existing ledger entries
        if ledger is not None:
            direct_excl = set(effective_spec.get("exclusive_with") or [])
            conflict = set()
            for entry in ledger.entries:
                found = self.get_item_spec(entry.rubric_item_id)
                if found is None:
                    continue
                existing_key, existing_spec = found
                existing_excl = set(existing_spec.get("exclusive_with") or [])
                if existing_key in direct_excl or effective_key in existing_excl:
                    conflict.add(existing_key)
            if conflict:
                warnings.append(f"exclusive_with_existing:{','.join(sorted(conflict))}")
                return CreditEstimate(
                    event_id=event.event_id,
                    rubric_item_id=str(effective_spec["ref"]),
                    group_id=str(effective_spec["group"]),
                    is_credit_eligible=False,
                    expected_points=0.0,
                    min_points=0.0,
                    max_points=0.0,
                    effective_points=0.0,
                    capped_by="exclusive_with",
                    source="rule",
                    warning_flags=warnings,
                )

        # Compute raw expected points
        if explicit_points is not None:
            expected_pts = max(0.0, float(explicit_points))
            min_pts = expected_pts
            max_pts = expected_pts
            source = "explicit"
        else:
            level_bucket = self._map_level_bucket(inferred_level)
            expected_pts, min_pts, max_pts, source, resolve_warnings = self._resolve_points_range(
                spec=effective_spec,
                role=role,
                level_bucket=level_bucket,
                duration_bucket=inferred_duration,
            )
            warnings.extend(resolve_warnings)

        # Apply remaining group cap & total cap when ledger is provided
        eff_group = str(effective_spec["group"])
        effective_pts = expected_pts
        capped_by: Optional[str] = None

        if ledger is not None:
            eval_res = self.evaluate_ledger(ledger)
            rem_group = eval_res.remaining_cap_by_group.get(eff_group, 0.0)
            rem_total = max(0.0, self.total_max - eval_res.total_score)

            if rem_group <= 0.0 and expected_pts > 0.0:
                effective_pts = 0.0
                capped_by = "group"
            elif rem_total <= 0.0 and expected_pts > 0.0:
                effective_pts = 0.0
                capped_by = "total"
            else:
                if effective_pts > rem_group:
                    effective_pts = rem_group
                    capped_by = "group"
                if effective_pts > rem_total:
                    effective_pts = rem_total
                    capped_by = "total"

        is_eligible = expected_pts > 0.0 and (effective_pts > 0.0 or ledger is None)
        return CreditEstimate(
            event_id=event.event_id,
            rubric_item_id=str(effective_spec["ref"]),
            group_id=eff_group,
            is_credit_eligible=is_eligible,
            expected_points=expected_pts,
            min_points=min_pts,
            max_points=max_pts,
            effective_points=effective_pts,
            capped_by=capped_by,
            source=source,
            warning_flags=warnings,
        )
