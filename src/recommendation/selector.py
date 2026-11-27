"""P7 List Selection via Maximal Marginal Relevance (`src/recommendation/selector.py`).

Implements set-level Top-N list diversification using MMR + 1 optional Exploration Slot,
specifically tailored to No-Le-DRL's P6 scoring outputs and `intfloat/multilingual-e5-small`
384d embeddings:

1. Mandatory pinning (`pin_mandatory=True`): High-mandatory activities (e.g. item `6.4`
   Tuần Sinh hoạt Công dân) are pinned to the top (`slot_type="mandatory"`) so they are
   never suppressed by diversity penalties and do not distort `[0, 1]` relevance scaling.
2. Hybrid pairwise similarity countering E5 cosine anisotropy:
   - Pairwise E5 cosines are often compressed into `[0.72, 0.92]`. Rescaling via
     `pool_minmax` (with safe fallback to `[fixed_cos_min, fixed_cos_max]`) restores full
     `[0, 1]` discrimination so redundancy penalties actually take effect.
   - Combined with rubric taxonomy similarity (`same_item_similarity=1.0`,
     `same_group_similarity=0.5`, else `0.0`):
         Sim(i, j) = alpha_semantic * sim_sem(i, j) + (1 - alpha_semantic) * sim_rubric(i, j)
3. Incremental `O(N * M)` MMR loop maintaining `max_sim[e] = max_{s in R} Sim(e, s)`:
       MMR(e) = lambda * Rel(e) - (1 - lambda) * max_sim[e]
4. Optional Exploration Slot (`slot_type="exploration"`): When `top_n >= 3`,
   `rem_slots >= 2`, and `len(pool) >= rem_slots`, reserves the final slot for a useful,
   novel candidate (preferring unseen `parent_group_id`):
       ExploreScore(e) = (w_gain * Gain(e) + w_fit * Fit(e)) * (1 - max_sim[e])
5. Emits per-item `SelectionTrace` (compatible with P8 EXPLAIN v2) and side-by-side
   diversity metrics (`ILD`, semantic ILD, group/item coverage, average P6 score)
   comparing MMR Top-N against Pure P6 Top-N.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.recommendation.config import load_recommendation_config
from src.recommendation.scoring import ScoredCandidate

VALID_COSINE_RESCALE_MODES: Tuple[str, ...] = ("pool_minmax", "fixed", "none")


@dataclass(frozen=True)
class MMRConfig:
    """Validated runtime parameters controlling P7 MMR list selection."""

    enabled: bool = True
    top_n: int = 5
    lambda_param: float = 0.70
    alpha_semantic: float = 0.70
    same_item_similarity: float = 1.00
    same_group_similarity: float = 0.50
    cosine_rescale_mode: str = "pool_minmax"
    fixed_cos_min: float = 0.72
    fixed_cos_max: float = 0.92
    min_pool_cosine_span: float = 0.05
    enable_exploration_slot: bool = True
    exploration_gain_weight: float = 0.60
    exploration_fit_weight: float = 0.40
    pin_mandatory: bool = True

    @classmethod
    def from_config(
        cls,
        config: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "MMRConfig":
        cfg = config if config is not None else load_recommendation_config()
        base = dict((cfg.get("selection") or {}).get("mmr") or {})
        merged: Dict[str, Any] = {**base}
        if overrides:
            for k, v in overrides.items():
                if v is not None and v != "":
                    merged[k] = v

        def _bool(key: str, default: bool) -> bool:
            val = merged.get(key, default)
            if isinstance(val, str):
                return val.strip().lower() in ("1", "true", "yes", "on")
            return bool(val)

        def _float_clamped(key: str, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
            raw = float(merged.get(key, default))
            if not math.isfinite(raw):
                raw = default
            return max(lo, min(hi, raw))

        top_n = max(1, int(merged.get("top_n", 5)))
        mode = str(merged.get("cosine_rescale_mode", "pool_minmax")).strip().lower()
        if mode not in VALID_COSINE_RESCALE_MODES:
            mode = "pool_minmax"

        fixed_lo = _float_clamped("fixed_cos_min", 0.72, -1.0, 1.0)
        fixed_hi = _float_clamped("fixed_cos_max", 0.92, -1.0, 1.0)
        if fixed_hi <= fixed_lo:
            fixed_lo, fixed_hi = 0.72, 0.92

        exp_g = _float_clamped("exploration_gain_weight", 0.60, 0.0, 1.0)
        exp_f = _float_clamped("exploration_fit_weight", 0.40, 0.0, 1.0)
        exp_sum = exp_g + exp_f
        if exp_sum <= 1e-9:
            exp_g, exp_f = 0.60, 0.40
        else:
            exp_g, exp_f = round(exp_g / exp_sum, 6), round(exp_f / exp_sum, 6)

        return cls(
            enabled=_bool("enabled", True),
            top_n=top_n,
            lambda_param=_float_clamped("lambda_param", 0.70, 0.0, 1.0),
            alpha_semantic=_float_clamped("alpha_semantic", 0.70, 0.0, 1.0),
            same_item_similarity=_float_clamped("same_item_similarity", 1.00, 0.0, 1.0),
            same_group_similarity=_float_clamped("same_group_similarity", 0.50, 0.0, 1.0),
            cosine_rescale_mode=mode,
            fixed_cos_min=fixed_lo,
            fixed_cos_max=fixed_hi,
            min_pool_cosine_span=_float_clamped("min_pool_cosine_span", 0.05, 0.0, 1.0),
            enable_exploration_slot=_bool("enable_exploration_slot", True),
            exploration_gain_weight=exp_g,
            exploration_fit_weight=exp_f,
            pin_mandatory=_bool("pin_mandatory", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "top_n": self.top_n,
            "lambda_param": round(self.lambda_param, 6),
            "alpha_semantic": round(self.alpha_semantic, 6),
            "same_item_similarity": round(self.same_item_similarity, 6),
            "same_group_similarity": round(self.same_group_similarity, 6),
            "cosine_rescale_mode": self.cosine_rescale_mode,
            "fixed_cos_min": round(self.fixed_cos_min, 6),
            "fixed_cos_max": round(self.fixed_cos_max, 6),
            "min_pool_cosine_span": round(self.min_pool_cosine_span, 6),
            "enable_exploration_slot": self.enable_exploration_slot,
            "exploration_gain_weight": round(self.exploration_gain_weight, 6),
            "exploration_fit_weight": round(self.exploration_fit_weight, 6),
            "pin_mandatory": self.pin_mandatory,
        }


@dataclass(frozen=True)
class SimilarityCalibration:
    """Diagnostics for pairwise E5 cosine rescaling in P7 MMR."""

    mode_requested: str
    mode_applied: str
    raw_cos_min: Optional[float]
    raw_cos_max: Optional[float]
    scale_lo: Optional[float]
    scale_hi: Optional[float]
    valid_vector_count: int
    pair_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode_requested": self.mode_requested,
            "mode_applied": self.mode_applied,
            "raw_cos_min": round(self.raw_cos_min, 6) if self.raw_cos_min is not None else None,
            "raw_cos_max": round(self.raw_cos_max, 6) if self.raw_cos_max is not None else None,
            "scale_lo": round(self.scale_lo, 6) if self.scale_lo is not None else None,
            "scale_hi": round(self.scale_hi, 6) if self.scale_hi is not None else None,
            "valid_vector_count": int(self.valid_vector_count),
            "pair_count": int(self.pair_count),
        }


@dataclass(frozen=True)
class SelectionTrace:
    """Transparent P7 selection trace consumed by UI and P8 EXPLAIN v2."""

    slot_type: str  # "mandatory" | "relevance" | "exploration"
    is_exploration_slot: bool
    mmr_rank: int
    p6_rank: int
    rank_delta: int
    relevance_score: float
    mmr_score: Optional[float]
    max_redundancy: float
    most_similar_selected_id: Optional[str] = None
    most_similar_selected_title: Optional[str] = None
    raw_cosine_with_most_similar: Optional[float] = None
    semantic_sim_with_most_similar: Optional[float] = None
    rubric_sim_with_most_similar: Optional[float] = None
    explore_score: Optional[float] = None
    exploration_is_unseen_group: Optional[bool] = None
    selection_reason: str = ""
    plan_role: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_type": self.slot_type,
            "is_exploration_slot": self.is_exploration_slot,
            "mmr_rank": int(self.mmr_rank),
            "p6_rank": int(self.p6_rank),
            "rank_delta": int(self.rank_delta),
            "relevance_score": round(self.relevance_score, 6),
            "mmr_score": round(self.mmr_score, 6) if self.mmr_score is not None else None,
            "max_redundancy": round(self.max_redundancy, 6),
            "most_similar_selected_id": self.most_similar_selected_id,
            "most_similar_selected_title": self.most_similar_selected_title,
            "raw_cosine_with_most_similar": (
                round(self.raw_cosine_with_most_similar, 6)
                if self.raw_cosine_with_most_similar is not None
                else None
            ),
            "semantic_sim_with_most_similar": (
                round(self.semantic_sim_with_most_similar, 6)
                if self.semantic_sim_with_most_similar is not None
                else None
            ),
            "rubric_sim_with_most_similar": (
                round(self.rubric_sim_with_most_similar, 6)
                if self.rubric_sim_with_most_similar is not None
                else None
            ),
            "explore_score": round(self.explore_score, 6) if self.explore_score is not None else None,
            "exploration_is_unseen_group": self.exploration_is_unseen_group,
            "selection_reason": self.selection_reason,
            "plan_role": self.plan_role,
        }


@dataclass
class SelectedCandidate:
    """One item in the P7 MMR-selected list, pairing P6 `ScoredCandidate` with `SelectionTrace`."""

    scored: ScoredCandidate
    trace: SelectionTrace

    def to_dict(self) -> Dict[str, Any]:
        base = self.scored.to_dict()
        base["rank"] = int(self.trace.mmr_rank)
        base["p6_rank"] = int(self.trace.p6_rank)
        base["rank_delta"] = int(self.trace.rank_delta)
        base["is_exploration_slot"] = bool(self.trace.is_exploration_slot)
        base["selection_trace"] = self.trace.to_dict()
        return base


@dataclass(frozen=True)
class MMRMetrics:
    """Side-by-side quality & diversity metrics comparing MMR Top-N against Pure P6 Top-N."""

    mmr_ild: float
    p6_topn_ild: float
    ild_gain: float
    mmr_semantic_ild: float
    p6_topn_semantic_ild: float
    mmr_group_coverage: int
    mmr_groups: List[str]
    p6_topn_group_coverage: int
    p6_topn_groups: List[str]
    mmr_item_coverage: int
    p6_topn_item_coverage: int
    mmr_avg_score: float
    p6_topn_avg_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mmr_ild": round(self.mmr_ild, 6),
            "p6_topn_ild": round(self.p6_topn_ild, 6),
            "ild_gain": round(self.ild_gain, 6),
            "mmr_semantic_ild": round(self.mmr_semantic_ild, 6),
            "p6_topn_semantic_ild": round(self.p6_topn_semantic_ild, 6),
            "mmr_group_coverage": int(self.mmr_group_coverage),
            "mmr_groups": list(self.mmr_groups),
            "p6_topn_group_coverage": int(self.p6_topn_group_coverage),
            "p6_topn_groups": list(self.p6_topn_groups),
            "mmr_item_coverage": int(self.mmr_item_coverage),
            "p6_topn_item_coverage": int(self.p6_topn_item_coverage),
            "mmr_avg_score": round(self.mmr_avg_score, 6),
            "p6_topn_avg_score": round(self.p6_topn_avg_score, 6),
        }


@dataclass
class MMRSelectionResult:
    """Complete output of P7 MMR list selection."""

    selected: List[SelectedCandidate]
    config: MMRConfig
    similarity_calibration: SimilarityCalibration
    metrics: MMRMetrics
    total_candidates: int
    mandatory_count: int
    relevance_count: int
    exploration_count: int
    exploration_event_id: Optional[str]

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "config": self.config.to_dict(),
            "similarity_calibration": self.similarity_calibration.to_dict(),
            "metrics": self.metrics.to_dict(),
            "total_candidates": int(self.total_candidates),
            "selected_count": len(self.selected),
            "mandatory_count": int(self.mandatory_count),
            "relevance_count": int(self.relevance_count),
            "exploration_count": int(self.exploration_count),
            "exploration_event_id": self.exploration_event_id,
        }


def compute_rubric_similarity(
    a: ScoredCandidate,
    b: ScoredCandidate,
    *,
    same_item_sim: float = 1.00,
    same_group_sim: float = 0.50,
) -> float:
    """Compute structural taxonomy similarity between two events (`rubric_item_id` / `parent_group_id`)."""
    item_a = (a.scored_event.rubric_item_id or "").strip()
    item_b = (b.scored_event.rubric_item_id or "").strip()
    if item_a and item_b and item_a == item_b:
        return float(same_item_sim)

    grp_a = (a.scored_event.parent_group_id or "").strip()
    grp_b = (b.scored_event.parent_group_id or "").strip()
    if grp_a and grp_b and grp_a == grp_b:
        return float(same_group_sim)

    return 0.0


def build_similarity_matrices(
    candidates: Sequence[ScoredCandidate],
    config: MMRConfig,
    *,
    calibration_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SimilarityCalibration]:
    """Compute `(hybrid_sim, semantic_sim, rubric_sim, raw_cos, calibration)` matrices.

    If `calibration_indices` is supplied (e.g., non-mandatory candidate indices), the
    `pool_minmax` cosine range `[obs_min, obs_max]` is measured exclusively over those
    indices so pinning/unpinning a mandatory event does not perturb regular MMR scaling.
    """
    n = len(candidates)
    if n == 0:
        empty = np.zeros((0, 0), dtype=np.float64)
        cal = SimilarityCalibration(
            mode_requested=config.cosine_rescale_mode,
            mode_applied="empty",
            raw_cos_min=None,
            raw_cos_max=None,
            scale_lo=None,
            scale_hi=None,
            valid_vector_count=0,
            pair_count=0,
        )
        return empty, empty, empty, empty, cal

    rubric_sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        rubric_sim[i, i] = 1.0
        for j in range(i + 1, n):
            rs = compute_rubric_similarity(
                candidates[i],
                candidates[j],
                same_item_sim=config.same_item_similarity,
                same_group_sim=config.same_group_similarity,
            )
            rubric_sim[i, j] = rs
            rubric_sim[j, i] = rs

    # L2-normalize all available event vectors
    normed_vecs: List[Optional[np.ndarray]] = []
    valid_dim: Optional[int] = None
    valid_count = 0
    for sc in candidates:
        raw_v = sc.candidate.event_vec
        if raw_v is None or len(raw_v) == 0:
            normed_vecs.append(None)
            continue
        arr = np.asarray(raw_v, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        if not np.isfinite(norm) or norm <= 1e-12:
            normed_vecs.append(None)
            continue
        if valid_dim is None:
            valid_dim = int(arr.shape[0])
        if int(arr.shape[0]) != valid_dim:
            normed_vecs.append(None)
            continue
        normed_vecs.append(arr / norm)
        valid_count += 1

    raw_cos = np.full((n, n), np.nan, dtype=np.float64)
    valid_idxs = [i for i, v in enumerate(normed_vecs) if v is not None]
    if valid_idxs:
        mat = np.vstack([normed_vecs[i] for i in valid_idxs])  # type: ignore[misc]
        cos_block = np.clip(mat @ mat.T, -1.0, 1.0)
        for pos_i, orig_i in enumerate(valid_idxs):
            for pos_j, orig_j in enumerate(valid_idxs):
                raw_cos[orig_i, orig_j] = float(cos_block[pos_i, pos_j])

    # Determine calibration pairs (preferring `calibration_indices` when >= 2 valid vectors)
    cal_pool = list(calibration_indices) if calibration_indices is not None else list(range(n))
    cal_valid = [i for i in cal_pool if normed_vecs[i] is not None]
    if len(cal_valid) < 2:
        cal_valid = valid_idxs

    off_diag: List[float] = []
    for a_pos in range(len(cal_valid)):
        i = cal_valid[a_pos]
        for b_pos in range(a_pos + 1, len(cal_valid)):
            j = cal_valid[b_pos]
            val = raw_cos[i, j]
            if np.isfinite(val):
                off_diag.append(float(val))

    obs_min: Optional[float] = min(off_diag) if off_diag else None
    obs_max: Optional[float] = max(off_diag) if off_diag else None

    mode_req = config.cosine_rescale_mode
    scale_lo: Optional[float] = None
    scale_hi: Optional[float] = None

    if not off_diag:
        mode_applied = "raw"
    elif mode_req == "none":
        mode_applied = "raw"
    elif mode_req == "fixed":
        mode_applied = "fixed"
        scale_lo = config.fixed_cos_min
        scale_hi = config.fixed_cos_max
    else:
        # "pool_minmax": rescale observed pool range if wide enough and not an
        # all-near-duplicate cluster (obs_min >= 0.88). Otherwise fall back to fixed
        # E5 bounds so near-identical pairs are not stretched down to 0.0.
        assert obs_min is not None and obs_max is not None
        span = obs_max - obs_min
        if len(off_diag) >= 2 and span >= config.min_pool_cosine_span and obs_min < 0.88:
            mode_applied = "pool_minmax"
            scale_lo = obs_min
            scale_hi = obs_max
        elif obs_min >= (config.fixed_cos_min - 0.10):
            mode_applied = "fixed_fallback"
            scale_lo = config.fixed_cos_min
            scale_hi = config.fixed_cos_max
        else:
            mode_applied = "raw"

    semantic_sim = np.full((n, n), np.nan, dtype=np.float64)
    for i in valid_idxs:
        semantic_sim[i, i] = 1.0
    for a_pos in range(len(valid_idxs)):
        i = valid_idxs[a_pos]
        for b_pos in range(a_pos + 1, len(valid_idxs)):
            j = valid_idxs[b_pos]
            c = float(raw_cos[i, j])
            if scale_lo is not None and scale_hi is not None and (scale_hi - scale_lo) > 1e-9:
                s = max(0.0, min(1.0, (c - scale_lo) / (scale_hi - scale_lo)))
            else:
                s = max(0.0, min(1.0, c))
            semantic_sim[i, j] = s
            semantic_sim[j, i] = s

    alpha = float(config.alpha_semantic)
    hybrid_sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        hybrid_sim[i, i] = 1.0
        for j in range(i + 1, n):
            sem_ij = semantic_sim[i, j]
            if np.isfinite(sem_ij):
                val = alpha * float(sem_ij) + (1.0 - alpha) * float(rubric_sim[i, j])
            else:
                val = float(rubric_sim[i, j])
            val = max(0.0, min(1.0, val))
            hybrid_sim[i, j] = val
            hybrid_sim[j, i] = val

    calibration = SimilarityCalibration(
        mode_requested=mode_req,
        mode_applied=mode_applied,
        raw_cos_min=obs_min,
        raw_cos_max=obs_max,
        scale_lo=scale_lo,
        scale_hi=scale_hi,
        valid_vector_count=valid_count,
        pair_count=len(off_diag),
    )
    return hybrid_sim, semantic_sim, rubric_sim, raw_cos, calibration


def compute_intra_list_diversity(
    indices: Sequence[int],
    sim_matrix: np.ndarray,
) -> float:
    """Compute Intra-List Diversity (ILD) = mean_{i < j} (1 - Sim(i, j)) in `[0, 1]`."""
    k = len(indices)
    if k < 2:
        return 0.0
    dists: List[float] = []
    for a in range(k):
        idx_i = int(indices[a])
        for b in range(a + 1, k):
            idx_j = int(indices[b])
            sim_ij = sim_matrix[idx_i, idx_j]
            if np.isfinite(sim_ij):
                dists.append(max(0.0, min(1.0, 1.0 - float(sim_ij))))
    if not dists:
        return 0.0
    return float(sum(dists) / len(dists))


def _unique_nonempty(values: Sequence[Optional[str]]) -> List[str]:
    seen: List[str] = []
    for v in values:
        s = (v or "").strip()
        if s and s not in seen:
            seen.append(s)
    return seen


def _truncate_title(title: str, max_len: int = 48) -> str:
    clean = " ".join((title or "").split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1].rstrip() + "…"


class MMRSelector:
    """P7 List Selector implementing MMR + optional Exploration Slot."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._raw_config = config if config is not None else load_recommendation_config()
        self._default_mmr = MMRConfig.from_config(self._raw_config)

    @property
    def default_config(self) -> MMRConfig:
        return self._default_mmr

    def select(
        self,
        scored_candidates: Sequence[ScoredCandidate],
        *,
        mmr_config: Optional[MMRConfig] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> MMRSelectionResult:
        """Select Top-N diverse activities from P6-ranked `scored_candidates`."""
        cfg = (
            mmr_config
            if mmr_config is not None
            else MMRConfig.from_config(self._raw_config, overrides=overrides)
        )
        all_cands = list(scored_candidates)
        total = len(all_cands)

        if total == 0:
            _, _, _, _, cal = build_similarity_matrices([], cfg)
            return MMRSelectionResult(
                selected=[],
                config=cfg,
                similarity_calibration=cal,
                metrics=MMRMetrics(
                    mmr_ild=0.0,
                    p6_topn_ild=0.0,
                    ild_gain=0.0,
                    mmr_semantic_ild=0.0,
                    p6_topn_semantic_ild=0.0,
                    mmr_group_coverage=0,
                    mmr_groups=[],
                    p6_topn_group_coverage=0,
                    p6_topn_groups=[],
                    mmr_item_coverage=0,
                    p6_topn_item_coverage=0,
                    mmr_avg_score=0.0,
                    p6_topn_avg_score=0.0,
                ),
                total_candidates=0,
                mandatory_count=0,
                relevance_count=0,
                exploration_count=0,
                exploration_event_id=None,
            )

        # Separate mandatory vs regular pool indices while preserving P6 1-based rank
        if cfg.pin_mandatory:
            mandatory_idxs = [i for i, sc in enumerate(all_cands) if sc.candidate.mandatory_info.is_mandatory]
            pool_idxs = [i for i, sc in enumerate(all_cands) if not sc.candidate.mandatory_info.is_mandatory]
        else:
            mandatory_idxs = []
            pool_idxs = list(range(total))

        hybrid_sim, semantic_sim, rubric_sim, raw_cos, cal = build_similarity_matrices(
            all_cands,
            cfg,
            calibration_indices=pool_idxs if len(pool_idxs) >= 2 else None,
        )

        # Bypass mode: keep pure P6 Top-N order when MMR is explicitly disabled
        if not cfg.enabled:
            chosen_idxs = list(range(min(cfg.top_n, total)))
            selected_items: List[SelectedCandidate] = []
            mandatory_cnt = 0
            relevance_cnt = 0
            for rank_pos, idx in enumerate(chosen_idxs, start=1):
                sc = all_cands[idx]
                is_mand = bool(sc.candidate.mandatory_info.is_mandatory)
                slot_type = "mandatory" if is_mand else "relevance"
                if is_mand:
                    mandatory_cnt += 1
                else:
                    relevance_cnt += 1
                trace = SelectionTrace(
                    slot_type=slot_type,
                    is_exploration_slot=False,
                    mmr_rank=rank_pos,
                    p6_rank=idx + 1,
                    rank_delta=0,
                    relevance_score=float(sc.scored_event.score_total),
                    mmr_score=None,
                    max_redundancy=0.0,
                    selection_reason=(
                        "Hoạt động bắt buộc (MMR đang tắt — giữ nguyên thứ tự P6)."
                        if is_mand
                        else "MMR đang tắt — giữ nguyên thứ tự xếp hạng thuần từ P6."
                    ),
                )
                selected_items.append(SelectedCandidate(scored=sc, trace=trace))

            metrics = self._compute_metrics(
                all_cands=all_cands,
                selected_idxs=chosen_idxs,
                top_n=cfg.top_n,
                hybrid_sim=hybrid_sim,
                semantic_sim=semantic_sim,
            )
            return MMRSelectionResult(
                selected=selected_items,
                config=cfg,
                similarity_calibration=cal,
                metrics=metrics,
                total_candidates=total,
                mandatory_count=mandatory_cnt,
                relevance_count=relevance_cnt,
                exploration_count=0,
                exploration_event_id=None,
            )

        selected_items = []
        selected_global_idxs: List[int] = []

        # Step 0: Pin mandatory activities at the top
        for idx in mandatory_idxs[: cfg.top_n]:
            sc = all_cands[idx]
            rank_pos = len(selected_items) + 1
            p6_rank = idx + 1
            trace = SelectionTrace(
                slot_type="mandatory",
                is_exploration_slot=False,
                mmr_rank=rank_pos,
                p6_rank=p6_rank,
                rank_delta=p6_rank - rank_pos,
                relevance_score=float(sc.scored_event.score_total),
                mmr_score=None,
                max_redundancy=0.0,
                selection_reason=(
                    "Hoạt động bắt buộc được ghim cố định lên đầu danh sách "
                    "(không chịu phạt đa dạng MMR để tránh bị trừ điểm ĐRL)."
                ),
            )
            selected_items.append(SelectedCandidate(scored=sc, trace=trace))
            selected_global_idxs.append(idx)

        rem_slots = cfg.top_n - len(selected_items)
        mandatory_cnt = len(selected_items)
        relevance_cnt = 0
        exploration_cnt = 0
        exploration_eid: Optional[str] = None

        if rem_slots > 0 and pool_idxs:
            # Reserve final slot for Exploration when N >= 3, at least 2 regular slots remain,
            # and there are at least `rem_slots` regular candidates available.
            use_explore = (
                cfg.enable_exploration_slot
                and cfg.top_n >= 3
                and rem_slots >= 2
                and len(pool_idxs) >= rem_slots
            )
            mmr_slots = rem_slots - 1 if use_explore else rem_slots

            # Relevance in [0, 1] for non-mandatory pool
            rel_by_idx: Dict[int, float] = {
                i: max(0.0, min(1.0, float(all_cands[i].scored_event.score_total)))
                for i in pool_idxs
            }

            # Step 2-3: Incremental O(N * M) MMR loop
            remaining = list(pool_idxs)
            chosen_regular: List[int] = []
            max_sim: Dict[int, float] = {i: 0.0 for i in remaining}
            most_sim_partner: Dict[int, Optional[int]] = {i: None for i in remaining}

            lam = cfg.lambda_param
            for step in range(min(mmr_slots, len(remaining))):
                if step == 0:
                    best_i = max(
                        remaining,
                        key=lambda i: (
                            rel_by_idx[i],
                            all_cands[i].scored_event.gain,
                            -i,
                        ),
                    )
                    best_mmr = lam * rel_by_idx[best_i]
                else:
                    def _mmr_key(i: int) -> Tuple[float, float, float, int]:
                        mmr_val = lam * rel_by_idx[i] - (1.0 - lam) * max_sim[i]
                        return (
                            mmr_val,
                            rel_by_idx[i],
                            all_cands[i].scored_event.gain,
                            -i,
                        )

                    best_i = max(remaining, key=_mmr_key)
                    best_mmr = lam * rel_by_idx[best_i] - (1.0 - lam) * max_sim[best_i]

                sc = all_cands[best_i]
                rank_pos = len(selected_items) + 1
                p6_rank = best_i + 1
                delta = p6_rank - rank_pos
                partner_i = most_sim_partner[best_i]
                red_val = max_sim[best_i]

                partner_id: Optional[str] = None
                partner_title: Optional[str] = None
                raw_c: Optional[float] = None
                sem_s: Optional[float] = None
                rub_s: Optional[float] = None

                if partner_i is not None:
                    partner_ev = all_cands[partner_i].candidate.event
                    partner_id = partner_ev.event_id
                    partner_title = partner_ev.title
                    if np.isfinite(raw_cos[best_i, partner_i]):
                        raw_c = float(raw_cos[best_i, partner_i])
                    if np.isfinite(semantic_sim[best_i, partner_i]):
                        sem_s = float(semantic_sim[best_i, partner_i])
                    rub_s = float(rubric_sim[best_i, partner_i])

                if step == 0:
                    reason = (
                        f"Đứng đầu điểm phù hợp P6 (Rel={rel_by_idx[best_i]:.3f}) trong nhóm hoạt động thường "
                        "→ chọn làm mốc khởi đầu của vòng lặp MMR."
                    )
                else:
                    sem_txt = f"{sem_s:.3f}" if sem_s is not None else "N/A"
                    rub_txt = f"{rub_s:.2f}" if rub_s is not None else "0.00"
                    p_short = _truncate_title(partner_title or partner_id or "—")
                    shift_txt = ""
                    if delta > 0:
                        shift_txt = f" Được đẩy tăng +{delta} bậc so với P6 (#{p6_rank} → #{rank_pos}) nhờ ít trùng lặp."
                    elif delta < 0:
                        shift_txt = f" Lùi {abs(delta)} bậc so với P6 (#{p6_rank} → #{rank_pos}) do trùng lặp một phần với sự kiện đã chọn."
                    reason = (
                        f"MMR = {lam:.2f}×{rel_by_idx[best_i]:.3f} − {1.0 - lam:.2f}×{red_val:.3f} = {best_mmr:.3f} "
                        f"(gần nhất với '{p_short}': Sim={red_val:.3f}, Sem={sem_txt}, Rubric={rub_txt}).{shift_txt}"
                    )

                trace = SelectionTrace(
                    slot_type="relevance",
                    is_exploration_slot=False,
                    mmr_rank=rank_pos,
                    p6_rank=p6_rank,
                    rank_delta=delta,
                    relevance_score=rel_by_idx[best_i],
                    mmr_score=best_mmr,
                    max_redundancy=red_val,
                    most_similar_selected_id=partner_id,
                    most_similar_selected_title=partner_title,
                    raw_cosine_with_most_similar=raw_c,
                    semantic_sim_with_most_similar=sem_s,
                    rubric_sim_with_most_similar=rub_s,
                    selection_reason=reason,
                )
                selected_items.append(SelectedCandidate(scored=sc, trace=trace))
                selected_global_idxs.append(best_i)
                chosen_regular.append(best_i)
                remaining.remove(best_i)
                relevance_cnt += 1

                # Incremental max_sim update in O(M)
                for i in remaining:
                    s_val = float(hybrid_sim[i, best_i])
                    if most_sim_partner[i] is None or s_val > max_sim[i]:
                        max_sim[i] = s_val
                        most_sim_partner[i] = best_i

            # Step 4: Exploration Slot (final slot)
            if use_explore and remaining:
                chosen_groups = {
                    (all_cands[i].scored_event.parent_group_id or "").strip()
                    for i in chosen_regular
                    if (all_cands[i].scored_event.parent_group_id or "").strip()
                }
                unseen_pool = [
                    i
                    for i in remaining
                    if (all_cands[i].scored_event.parent_group_id or "").strip() not in chosen_groups
                ]
                exp_pool = unseen_pool if unseen_pool else remaining
                is_unseen_group = bool(unseen_pool)

                w_g = cfg.exploration_gain_weight
                w_f = cfg.exploration_fit_weight

                def _explore_val(i: int) -> float:
                    se = all_cands[i].scored_event
                    utility = w_g * float(se.gain) + w_f * float(se.fit)
                    novelty = max(0.0, 1.0 - max_sim[i])
                    return utility * novelty

                best_exp = max(
                    exp_pool,
                    key=lambda i: (
                        _explore_val(i),
                        1.0 - max_sim[i],
                        rel_by_idx[i],
                        -i,
                    ),
                )
                exp_score = _explore_val(best_exp)
                sc = all_cands[best_exp]
                rank_pos = len(selected_items) + 1
                p6_rank = best_exp + 1
                delta = p6_rank - rank_pos
                partner_i = most_sim_partner[best_exp]
                red_val = max_sim[best_exp]
                novelty = max(0.0, 1.0 - red_val)

                partner_id = None
                partner_title = None
                raw_c = None
                sem_s = None
                rub_s = None
                if partner_i is not None:
                    partner_ev = all_cands[partner_i].candidate.event
                    partner_id = partner_ev.event_id
                    partner_title = partner_ev.title
                    if np.isfinite(raw_cos[best_exp, partner_i]):
                        raw_c = float(raw_cos[best_exp, partner_i])
                    if np.isfinite(semantic_sim[best_exp, partner_i]):
                        sem_s = float(semantic_sim[best_exp, partner_i])
                    rub_s = float(rubric_sim[best_exp, partner_i])

                grp_label = sc.scored_event.parent_group_id or "khác"
                if is_unseen_group:
                    exp_reason = (
                        f"Slot khám phá (SEL_EXPLORATION): mở rộng sang nhóm {grp_label} chưa xuất hiện "
                        f"trong {len(chosen_regular)} gợi ý chính; hữu ích (Gain={sc.scored_event.gain:.2f}, "
                        f"Fit={sc.scored_event.fit:.2f}) × độ mới lạ (1 − {red_val:.2f} = {novelty:.2f}) = {exp_score:.3f}."
                    )
                else:
                    exp_reason = (
                        f"Slot khám phá (SEL_EXPLORATION): ứng viên mới lạ nhất ngoài cụm đã chọn "
                        f"(Gain={sc.scored_event.gain:.2f}, Fit={sc.scored_event.fit:.2f}, "
                        f"độ mới lạ 1 − {red_val:.2f} = {novelty:.2f}) → Explore={exp_score:.3f}."
                    )

                trace = SelectionTrace(
                    slot_type="exploration",
                    is_exploration_slot=True,
                    mmr_rank=rank_pos,
                    p6_rank=p6_rank,
                    rank_delta=delta,
                    relevance_score=rel_by_idx[best_exp],
                    mmr_score=None,
                    max_redundancy=red_val,
                    most_similar_selected_id=partner_id,
                    most_similar_selected_title=partner_title,
                    raw_cosine_with_most_similar=raw_c,
                    semantic_sim_with_most_similar=sem_s,
                    rubric_sim_with_most_similar=rub_s,
                    explore_score=exp_score,
                    exploration_is_unseen_group=is_unseen_group,
                    selection_reason=exp_reason,
                )
                selected_items.append(SelectedCandidate(scored=sc, trace=trace))
                selected_global_idxs.append(best_exp)
                exploration_cnt = 1
                exploration_eid = sc.scored_event.event_id

        metrics = self._compute_metrics(
            all_cands=all_cands,
            selected_idxs=selected_global_idxs,
            top_n=cfg.top_n,
            hybrid_sim=hybrid_sim,
            semantic_sim=semantic_sim,
        )
        return MMRSelectionResult(
            selected=selected_items,
            config=cfg,
            similarity_calibration=cal,
            metrics=metrics,
            total_candidates=total,
            mandatory_count=mandatory_cnt,
            relevance_count=relevance_cnt,
            exploration_count=exploration_cnt,
            exploration_event_id=exploration_eid,
        )

    @staticmethod
    def _compute_metrics(
        *,
        all_cands: Sequence[ScoredCandidate],
        selected_idxs: Sequence[int],
        top_n: int,
        hybrid_sim: np.ndarray,
        semantic_sim: np.ndarray,
    ) -> MMRMetrics:
        p6_topn_idxs = list(range(min(top_n, len(all_cands))))

        mmr_ild = compute_intra_list_diversity(selected_idxs, hybrid_sim)
        p6_ild = compute_intra_list_diversity(p6_topn_idxs, hybrid_sim)
        mmr_sem_ild = compute_intra_list_diversity(selected_idxs, semantic_sim)
        p6_sem_ild = compute_intra_list_diversity(p6_topn_idxs, semantic_sim)

        mmr_groups = _unique_nonempty([all_cands[i].scored_event.parent_group_id for i in selected_idxs])
        p6_groups = _unique_nonempty([all_cands[i].scored_event.parent_group_id for i in p6_topn_idxs])
        mmr_items = _unique_nonempty([all_cands[i].scored_event.rubric_item_id for i in selected_idxs])
        p6_items = _unique_nonempty([all_cands[i].scored_event.rubric_item_id for i in p6_topn_idxs])

        mmr_avg = (
            sum(all_cands[i].scored_event.score_total for i in selected_idxs) / len(selected_idxs)
            if selected_idxs
            else 0.0
        )
        p6_avg = (
            sum(all_cands[i].scored_event.score_total for i in p6_topn_idxs) / len(p6_topn_idxs)
            if p6_topn_idxs
            else 0.0
        )

        return MMRMetrics(
            mmr_ild=mmr_ild,
            p6_topn_ild=p6_ild,
            ild_gain=mmr_ild - p6_ild,
            mmr_semantic_ild=mmr_sem_ild,
            p6_topn_semantic_ild=p6_sem_ild,
            mmr_group_coverage=len(mmr_groups),
            mmr_groups=mmr_groups,
            p6_topn_group_coverage=len(p6_groups),
            p6_topn_groups=p6_groups,
            mmr_item_coverage=len(mmr_items),
            p6_topn_item_coverage=len(p6_items),
            mmr_avg_score=mmr_avg,
            p6_topn_avg_score=p6_avg,
        )


def select_top_n_mmr(
    scored_candidates: Sequence[ScoredCandidate],
    *,
    top_n: int = 5,
    lambda_param: float = 0.70,
    alpha_semantic: float = 0.70,
    cosine_rescale_mode: str = "pool_minmax",
    enable_exploration_slot: bool = True,
    pin_mandatory: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> MMRSelectionResult:
    """Convenience functional wrapper around `MMRSelector`."""
    selector = MMRSelector(config=config)
    return selector.select(
        scored_candidates,
        overrides={
            "enabled": True,
            "top_n": top_n,
            "lambda_param": lambda_param,
            "alpha_semantic": alpha_semantic,
            "cosine_rescale_mode": cosine_rescale_mode,
            "enable_exploration_slot": enable_exploration_slot,
            "pin_mandatory": pin_mandatory,
        },
    )
