"""P4 User Profile, Cold-Start Onboarding & Behavior-Driven Adaptation.

Uses the unified `intfloat/multilingual-e5-small` (`384d` L2-normalized) embedder
for cold-start interest modeling, Rocchio-style vector adaptation (`u_declared` +
`u_learned`), and behavior-driven updates to the 4 scoring weights `w1..w4` (D-5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from configs.model_config import EMBEDDING_DIMENSION
from src.contract_validator import validate_scoring_weights
from src.embedder import BaseEmbedder, create_embedder
from src.recommendation.config import load_recommendation_config
from src.schemas import ScoringWeights, UserProfile
from src.vector_ops import l2_norm, l2_normalize

VALID_BEHAVIOR_ACTIONS: Tuple[str, ...] = ("click", "save", "joined", "dismiss")
WEIGHT_KEYS: Tuple[str, ...] = ("w1_gain", "w2_interest", "w3_fit", "w4_urgency")


@dataclass(frozen=True)
class OnboardingInput:
    """Cold-start onboarding selections provided by a student."""

    user_id: str
    interest_chips: Sequence[str] = ()
    free_text: Optional[str] = None
    weight_preset: str = "balanced"
    custom_weights: Optional[ScoringWeights] = None
    target_total: float = 80.0
    target_by_group: Dict[str, float] = field(default_factory=dict)
    cohort: Optional[str] = None
    faculty: Optional[str] = None
    class_id: Optional[str] = None
    home_campus: Optional[str] = None
    max_budget_vnd: Optional[int] = None
    max_travel_min: Optional[int] = None


@dataclass(frozen=True)
class BehaviorSignal:
    """One user interaction event driving P4 vector & weight adaptation."""

    event_id: str
    action: str  # "click" | "save" | "joined" | "dismiss"
    event_vector: Optional[List[float]] = None
    score_components: Optional[Dict[str, float]] = None
    dismiss_reason: Optional[str] = None


@dataclass
class UserProfileState:
    """Complete mutable/serializable state for a student's P4 profile."""

    profile: UserProfile
    home_campus: Optional[str] = None
    u_declared: Optional[List[float]] = None
    u_learned: Optional[List[float]] = None
    u_learned_accum: Optional[List[float]] = None
    weight_preset: str = "balanced"
    declared_weights: ScoringWeights = field(default_factory=ScoringWeights)
    behavior_count: int = 0
    weight_update_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        p = self.profile
        return {
            "profile": {
                "user_id": p.user_id,
                "cohort": p.cohort,
                "faculty": p.faculty,
                "class_id": p.class_id,
                "interest_tags": list(p.interest_tags),
                "user_vec": list(p.user_vec) if p.user_vec is not None else None,
                "target_total": float(p.target_total),
                "target_by_group": dict(p.target_by_group),
                "max_budget_vnd": p.max_budget_vnd,
                "max_travel_min": p.max_travel_min,
                "hidden_event_ids": list(p.hidden_event_ids),
                "participated_event_ids": list(p.participated_event_ids),
                "saved_event_ids": list(p.saved_event_ids),
                "weights": p.weights.to_dict(),
            },
            "home_campus": self.home_campus,
            "u_declared": list(self.u_declared) if self.u_declared is not None else None,
            "u_learned": list(self.u_learned) if self.u_learned is not None else None,
            "u_learned_accum": list(self.u_learned_accum) if self.u_learned_accum is not None else None,
            "weight_preset": self.weight_preset,
            "declared_weights": self.declared_weights.to_dict(),
            "behavior_count": int(self.behavior_count),
            "weight_update_count": int(self.weight_update_count),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfileState":
        p_raw = data["profile"]
        w_raw = p_raw.get("weights") or {}
        dw_raw = data.get("declared_weights") or w_raw
        profile = UserProfile(
            user_id=str(p_raw["user_id"]),
            cohort=p_raw.get("cohort"),
            faculty=p_raw.get("faculty"),
            class_id=p_raw.get("class_id"),
            interest_tags=list(p_raw.get("interest_tags") or []),
            user_vec=list(p_raw["user_vec"]) if p_raw.get("user_vec") is not None else None,
            target_total=float(p_raw.get("target_total", 80.0)),
            target_by_group=dict(p_raw.get("target_by_group") or {}),
            max_budget_vnd=p_raw.get("max_budget_vnd"),
            max_travel_min=p_raw.get("max_travel_min"),
            hidden_event_ids=list(p_raw.get("hidden_event_ids") or []),
            participated_event_ids=list(p_raw.get("participated_event_ids") or []),
            saved_event_ids=list(p_raw.get("saved_event_ids") or []),
            weights=ScoringWeights(
                w1_gain=float(w_raw.get("w1_gain", 0.40)),
                w2_interest=float(w_raw.get("w2_interest", 0.25)),
                w3_fit=float(w_raw.get("w3_fit", 0.20)),
                w4_urgency=float(w_raw.get("w4_urgency", 0.15)),
            ),
        )
        declared_weights = ScoringWeights(
            w1_gain=float(dw_raw.get("w1_gain", 0.40)),
            w2_interest=float(dw_raw.get("w2_interest", 0.25)),
            w3_fit=float(dw_raw.get("w3_fit", 0.20)),
            w4_urgency=float(dw_raw.get("w4_urgency", 0.15)),
        )
        return cls(
            profile=profile,
            home_campus=data.get("home_campus"),
            u_declared=list(data["u_declared"]) if data.get("u_declared") is not None else None,
            u_learned=list(data["u_learned"]) if data.get("u_learned") is not None else None,
            u_learned_accum=(
                list(data["u_learned_accum"]) if data.get("u_learned_accum") is not None else None
            ),
            weight_preset=str(data.get("weight_preset") or "balanced"),
            declared_weights=declared_weights,
            behavior_count=int(data.get("behavior_count", 0)),
            weight_update_count=int(data.get("weight_update_count", 0)),
        )


def _weights_to_list(w: ScoringWeights) -> List[float]:
    return [float(w.w1_gain), float(w.w2_interest), float(w.w3_fit), float(w.w4_urgency)]


def _list_to_weights(vals: Sequence[float]) -> ScoringWeights:
    return ScoringWeights(
        w1_gain=float(vals[0]),
        w2_interest=float(vals[1]),
        w3_fit=float(vals[2]),
        w4_urgency=float(vals[3]),
    )


def _project_bounded_simplex(values: Sequence[float], lo: float, hi: float) -> List[float]:
    """Project a 4-vector onto {w : sum(w) == 1.0, lo <= w_i <= hi}."""
    n = len(values)
    if n == 0:
        return []
    if n * lo > 1.0 + 1e-9 or n * hi < 1.0 - 1e-9:
        raise ValueError(f"Infeasible bounded simplex: n={n}, lo={lo}, hi={hi}")

    vals = [float(v) for v in values]
    tau_lo = min(vals) - hi - 1.0
    tau_hi = max(vals) - lo + 1.0

    for _ in range(80):
        tau = 0.5 * (tau_lo + tau_hi)
        s = sum(min(hi, max(lo, v - tau)) for v in vals)
        if s > 1.0:
            tau_lo = tau
        else:
            tau_hi = tau

    tau = 0.5 * (tau_lo + tau_hi)
    projected = [min(hi, max(lo, v - tau)) for v in vals]

    # Exact floating-point sum adjustment onto interior coordinate
    residual = 1.0 - sum(projected)
    if abs(residual) > 0.0:
        best_idx = max(
            range(n),
            key=lambda i: min(projected[i] - lo, hi - projected[i]),
        )
        projected[best_idx] = min(hi, max(lo, projected[best_idx] + residual))

    return [round(x, 8) for x in projected]


def _normalize_weights_strict(w: ScoringWeights) -> ScoringWeights:
    vals = _weights_to_list(w)
    if any(v < 0.0 or not math.isfinite(v) for v in vals):
        raise ValueError(f"All scoring weights must be finite and >= 0, got {vals}")
    total = sum(vals)
    if total <= 0.0:
        raise ValueError("Sum of scoring weights must be > 0")
    normalized = _list_to_weights([v / total for v in vals])
    rep = validate_scoring_weights(normalized)
    if not rep.is_valid:
        raise ValueError(f"Invalid scoring weights: {[i.message for i in rep.errors]}")
    return normalized


class UserProfileManager:
    """Manages P4 Cold-Start onboarding, vector blending, and weight adaptation."""

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._cfg = config if config is not None else load_recommendation_config()
        self._embedder = embedder
        self._chips: Dict[str, Dict[str, str]] = {
            str(k): {"id": str(k), "label": str(v["label"]), "text": str(v["text"])}
            for k, v in (self._cfg.get("interest_chips") or {}).items()
        }
        self._presets: Dict[str, Dict[str, Any]] = dict(self._cfg.get("weight_presets") or {})
        self._profile_cfg: Dict[str, Any] = dict(self._cfg.get("profile") or {})
        self._chip_vec_cache: Dict[str, List[float]] = {}

    def _get_embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            self._embedder, _ = create_embedder()
        return self._embedder

    def available_interest_chips(self) -> List[Dict[str, str]]:
        return [dict(c) for c in self._chips.values()]

    def available_weight_presets(self) -> Dict[str, Dict[str, Any]]:
        return {k: dict(v) for k, v in self._presets.items()}

    def _resolve_initial_weights(
        self,
        weight_preset: str,
        custom_weights: Optional[ScoringWeights],
    ) -> Tuple[str, ScoringWeights]:
        if custom_weights is not None:
            return "custom", _normalize_weights_strict(custom_weights)

        if weight_preset not in self._presets:
            raise ValueError(
                f"Unknown weight_preset '{weight_preset}'. Available: {sorted(self._presets.keys())}"
            )
        raw = self._presets[weight_preset]["weights"]
        weights = _normalize_weights_strict(
            ScoringWeights(
                w1_gain=float(raw["w1_gain"]),
                w2_interest=float(raw["w2_interest"]),
                w3_fit=float(raw["w3_fit"]),
                w4_urgency=float(raw["w4_urgency"]),
            )
        )
        return weight_preset, weights

    def _encode_cold_start_vector(
        self,
        chip_ids: Sequence[str],
        free_text: Optional[str],
    ) -> Optional[List[float]]:
        unique_chips = list(dict.fromkeys(str(c).strip() for c in chip_ids if str(c).strip()))
        for cid in unique_chips:
            if cid not in self._chips:
                raise ValueError(
                    f"Unknown interest chip '{cid}'. Available: {sorted(self._chips.keys())}"
                )

        clean_free = (free_text or "").strip()
        if not unique_chips and not clean_free:
            return None

        embedder = self._get_embedder()
        vectors_to_avg: List[List[float]] = []

        missing_chips = [cid for cid in unique_chips if cid not in self._chip_vec_cache]
        if missing_chips:
            texts = [self._chips[cid]["text"] for cid in missing_chips]
            encoded = embedder.encode_queries(texts)
            for cid, vec in zip(missing_chips, encoded):
                self._chip_vec_cache[cid] = l2_normalize(vec)

        for cid in unique_chips:
            vectors_to_avg.append(self._chip_vec_cache[cid])

        if clean_free:
            vectors_to_avg.append(l2_normalize(embedder.encode_query(clean_free)))

        dim = len(vectors_to_avg[0])
        summed = [0.0] * dim
        for vec in vectors_to_avg:
            for i in range(dim):
                summed[i] += vec[i]
        return l2_normalize(summed)

    def _blend_user_vector(
        self,
        u_declared: Optional[List[float]],
        u_learned: Optional[List[float]],
    ) -> Optional[List[float]]:
        if u_declared is None and u_learned is None:
            return None
        if u_learned is None:
            return list(u_declared)  # type: ignore[arg-type]
        if u_declared is None:
            return list(u_learned)

        beta = float(self._profile_cfg.get("declared_blend_weight", 0.70))
        beta = max(0.0, min(1.0, beta))
        dim = len(u_declared)
        blended = [beta * u_declared[i] + (1.0 - beta) * u_learned[i] for i in range(dim)]
        if l2_norm(blended) <= 1e-12:
            return list(u_declared)
        return l2_normalize(blended)

    @staticmethod
    def _extract_component_vector(score_components: Dict[str, float]) -> List[float]:
        gain = float(score_components.get("gain", score_components.get("w1_gain", 0.0)))
        interest = float(score_components.get("interest", score_components.get("w2_interest", 0.0)))
        fit = float(score_components.get("fit", score_components.get("w3_fit", 0.0)))
        urgency = float(score_components.get("urgency", score_components.get("w4_urgency", 0.0)))
        return [gain, interest, fit, urgency]

    def _adapt_weights(
        self,
        *,
        current_weights: ScoringWeights,
        declared_weights: ScoringWeights,
        action: str,
        score_components: Optional[Dict[str, float]],
        dismiss_reason: Optional[str],
    ) -> Tuple[ScoringWeights, bool]:
        adapt_cfg = self._profile_cfg.get("weight_adaptation") or {}
        if not bool(adapt_cfg.get("enabled", True)):
            return current_weights, False

        boosts_map = adapt_cfg.get("dismiss_reason_boosts") or {}
        reason_boosts = boosts_map.get(dismiss_reason) if (action == "dismiss" and dismiss_reason) else None
        if score_components is None and not reason_boosts:
            return current_weights, False

        lo = float(adapt_cfg.get("min_weight", 0.05))
        hi = float(adapt_cfg.get("max_weight", 0.70))
        pull = max(0.0, min(1.0, float(adapt_cfg.get("declared_anchor_pull", 0.10))))
        rates = adapt_cfg.get("action_learning_rates") or {}
        lr = float(rates.get(action, 0.0))

        w_curr = _weights_to_list(current_weights)
        w_decl = _weights_to_list(declared_weights)
        w_new = list(w_curr)

        if score_components is not None and lr != 0.0:
            comps = self._extract_component_vector(score_components)
            mean_c = sum(comps) / 4.0
            for i in range(4):
                w_new[i] += lr * (comps[i] - mean_c)

        if isinstance(reason_boosts, dict):
            for i, key in enumerate(WEIGHT_KEYS):
                if key in reason_boosts:
                    w_new[i] += float(reason_boosts[key])

        for i in range(4):
            w_new[i] = (1.0 - pull) * w_new[i] + pull * w_decl[i]

        projected = _project_bounded_simplex(w_new, lo=lo, hi=hi)
        updated = _list_to_weights(projected)
        rep = validate_scoring_weights(updated)
        if not rep.is_valid:
            raise ValueError(f"Weight adaptation violated invariants: {[i.message for i in rep.errors]}")
        return updated, True

    def create_cold_start_profile(self, onboarding: OnboardingInput) -> UserProfileState:
        if not onboarding.user_id or not str(onboarding.user_id).strip():
            raise ValueError("onboarding.user_id must be a non-empty string.")

        preset_name, init_weights = self._resolve_initial_weights(
            onboarding.weight_preset,
            onboarding.custom_weights,
        )
        chip_tags = list(dict.fromkeys(str(c).strip() for c in onboarding.interest_chips if str(c).strip()))
        u_declared = self._encode_cold_start_vector(chip_tags, onboarding.free_text)

        profile = UserProfile(
            user_id=str(onboarding.user_id).strip(),
            cohort=onboarding.cohort,
            faculty=onboarding.faculty,
            class_id=onboarding.class_id,
            interest_tags=chip_tags,
            user_vec=list(u_declared) if u_declared is not None else None,
            target_total=float(onboarding.target_total),
            target_by_group=dict(onboarding.target_by_group),
            max_budget_vnd=onboarding.max_budget_vnd,
            max_travel_min=onboarding.max_travel_min,
            weights=init_weights,
        )
        return UserProfileState(
            profile=profile,
            home_campus=onboarding.home_campus,
            u_declared=u_declared,
            u_learned=None,
            u_learned_accum=None,
            weight_preset=preset_name,
            declared_weights=init_weights,
            behavior_count=0,
            weight_update_count=0,
        )

    def record_behavior(
        self,
        state: UserProfileState,
        signal: BehaviorSignal,
    ) -> UserProfileState:
        action = str(signal.action).strip()
        if action not in VALID_BEHAVIOR_ACTIONS:
            raise ValueError(
                f"Unsupported behavior action '{action}'. Must be one of {VALID_BEHAVIOR_ACTIONS}"
            )
        eid = str(signal.event_id).strip()
        if not eid:
            raise ValueError("BehaviorSignal.event_id must be non-empty.")

        # 1. Update event state lists cleanly (deduplicated)
        p = state.profile
        if action == "save" and eid not in p.saved_event_ids:
            p.saved_event_ids.append(eid)
        elif action == "joined" and eid not in p.participated_event_ids:
            p.participated_event_ids.append(eid)
        elif action == "dismiss" and eid not in p.hidden_event_ids:
            p.hidden_event_ids.append(eid)

        # 2. Update Rocchio learned vector (u_learned)
        vec_weights = self._profile_cfg.get("behavior_vector_weights") or {}
        if signal.event_vector is not None:
            if len(signal.event_vector) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Expected {EMBEDDING_DIMENSION}d event_vector, got {len(signal.event_vector)}"
                )
            ev_vec = l2_normalize(signal.event_vector)
            alpha = float(vec_weights.get(action, 0.0))
            if state.u_learned_accum is None:
                state.u_learned_accum = [0.0] * EMBEDDING_DIMENSION
            for i in range(EMBEDDING_DIMENSION):
                state.u_learned_accum[i] += alpha * ev_vec[i]

            if l2_norm(state.u_learned_accum) > 1e-12:
                state.u_learned = l2_normalize(state.u_learned_accum)
            else:
                state.u_learned = None

            p.user_vec = self._blend_user_vector(state.u_declared, state.u_learned)

        # 3. Behavior-driven adaptation of w1..w4
        updated_weights, weights_changed = self._adapt_weights(
            current_weights=p.weights,
            declared_weights=state.declared_weights,
            action=action,
            score_components=signal.score_components,
            dismiss_reason=signal.dismiss_reason,
        )
        p.weights = updated_weights
        state.behavior_count += 1
        if weights_changed:
            state.weight_update_count += 1

        return state

    def set_custom_weights(
        self,
        state: UserProfileState,
        weights: ScoringWeights,
        *,
        update_declared: bool = True,
    ) -> UserProfileState:
        normalized = _normalize_weights_strict(weights)
        state.profile.weights = normalized
        if update_declared:
            state.declared_weights = normalized
            state.weight_preset = "custom"
        return state

    def reset_learned_behavior(self, state: UserProfileState) -> UserProfileState:
        """Reset learned behavior signals back to the student's cold-start baseline."""
        state.u_learned = None
        state.u_learned_accum = None
        state.profile.user_vec = list(state.u_declared) if state.u_declared is not None else None
        state.profile.weights = ScoringWeights(
            w1_gain=state.declared_weights.w1_gain,
            w2_interest=state.declared_weights.w2_interest,
            w3_fit=state.declared_weights.w3_fit,
            w4_urgency=state.declared_weights.w4_urgency,
        )
        state.profile.saved_event_ids.clear()
        state.profile.hidden_event_ids.clear()
        state.behavior_count = 0
        state.weight_update_count = 0
        return state
