"""Configurable similarity-to-weight transforms, normalization, and Stage-wise Fusion."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def similarity_to_weight(
    similarity: float,
    mode: str = "positive_clip",
    temperature: float = 0.1,
    power: float = 2.0,
) -> float:
    """Transform cosine similarity into a non-negative kNN voting weight."""
    sim = float(similarity)
    if mode == "positive_clip":
        return max(sim, 0.0)
    if mode == "shifted":
        return max((sim + 1.0) / 2.0, 0.0)
    if mode == "exp_scaled":
        temp = max(float(temperature), 1e-6)
        # Clamp exponent argument for numerical safety
        arg = max(min(sim / temp, 50.0), -50.0)
        return math.exp(arg)
    if mode == "power":
        return max(sim, 0.0) ** float(power)
    raise ValueError(f"Unsupported similarity weighting mode: '{mode}'")


def normalize_score_map(
    scores: Dict[str, float],
    method: str = "min_max",
    universe: Optional[Sequence[str]] = None,
    fill_missing_with: float = 0.0,
) -> Dict[str, float]:
    """Normalize a raw score mapping using a configurable method.

    Note: Normalized scores remain ranking scores, NOT calibrated probabilities.
    """
    keys = list(universe) if universe is not None else list(scores.keys())
    if not keys:
        return {}

    vals = {k: float(scores.get(k, fill_missing_with)) for k in keys}

    if method in ("identity", "none"):
        return vals

    if method == "sum_normalize":
        clipped = {k: max(v, 0.0) for k, v in vals.items()}
        total = sum(clipped.values())
        if total <= 0.0:
            return {k: 0.0 for k in keys}
        return {k: v / total for k, v in clipped.items()}

    if method == "min_max":
        v_list = list(vals.values())
        lo = min(v_list)
        hi = max(v_list)
        if hi - lo <= 1e-12:
            # All candidates tied: preserve 1.0 if strictly positive signal, else 0.0
            const = 1.0 if hi > 0.0 else 0.0
            return {k: const for k in keys}
        return {k: (v - lo) / (hi - lo) for k, v in vals.items()}

    raise ValueError(f"Unsupported score normalization method: '{method}'")


def fuse_stage1_groups(
    *,
    eligible_groups: Sequence[str],
    sample_raw_scores: Dict[str, float],
    rubric_raw_scores: Dict[str, float],
    beta: float,
    normalization_method: str,
    top_k_groups: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
    """Fuse Stage 1 Sample and Rubric major-group scores: S1(g) = beta*S_sample + (1-beta)*S_rubric."""
    groups = list(eligible_groups)
    beta_val = float(beta)
    if not (0.0 <= beta_val <= 1.0):
        raise ValueError(f"stage1_fusion_beta must be in [0, 1], got {beta_val}")

    sample_norm = normalize_score_map(
        sample_raw_scores,
        method=normalization_method,
        universe=groups,
        fill_missing_with=0.0,
    )
    rubric_norm = normalize_score_map(
        rubric_raw_scores,
        method=normalization_method,
        universe=groups,
        fill_missing_with=0.0,
    )

    fused_entries: List[Dict[str, Any]] = []
    for grp in groups:
        s_raw = sample_raw_scores.get(grp)
        r_raw = rubric_raw_scores.get(grp)
        s_n = sample_norm[grp]
        r_n = rubric_norm[grp]
        fused = beta_val * s_n + (1.0 - beta_val) * r_n
        fused_entries.append(
            {
                "group_id": grp,
                "fused_score": round(float(fused), 6),
                "sample_raw_score": round(float(s_raw), 6) if s_raw is not None else None,
                "sample_norm_score": round(float(s_n), 6),
                "rubric_raw_score": round(float(r_raw), 6) if r_raw is not None else None,
                "rubric_norm_score": round(float(r_n), 6),
                "beta": beta_val,
            }
        )

    fused_entries.sort(key=lambda e: (-e["fused_score"], -(e["rubric_raw_score"] or -1.0), e["group_id"]))
    for rank_idx, entry in enumerate(fused_entries, start=1):
        entry["rank"] = rank_idx
        entry["selected_for_stage2"] = rank_idx <= top_k_groups

    return fused_entries, sample_norm, rubric_norm


def fuse_stage2_criteria(
    *,
    candidate_refs: Iterable[str],
    sample_raw_scores: Dict[str, float],
    rubric_raw_scores: Dict[str, float],
    sample_available_refs: Set[str],
    rubric_available_refs: Set[str],
    ref_to_meta: Dict[str, Dict[str, Any]],
    alpha: float,
    normalization_method: str,
    missing_branch_policy: str = "reweight_available_branch",
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    """Fuse Stage 2 Sample and Rubric criterion scores aligned by criterion_ref.

    Explicitly represents missing branch evidence and avoids treating zero reference
    coverage in one branch as proof of irrelevance when `reweight_available_branch` is set.
    """
    alpha_val = float(alpha)
    if not (0.0 <= alpha_val <= 1.0):
        raise ValueError(f"stage2_fusion_alpha must be in [0, 1], got {alpha_val}")

    union_refs = sorted(set(str(r) for r in candidate_refs))
    if not union_refs:
        return [], {}, {}

    # Normalize each branch ONLY over criteria where that branch has reference coverage
    sample_covered_universe = [r for r in union_refs if r in sample_available_refs]
    rubric_covered_universe = [r for r in union_refs if r in rubric_available_refs]

    sample_norm_covered = normalize_score_map(
        sample_raw_scores,
        method=normalization_method,
        universe=sample_covered_universe,
        fill_missing_with=0.0,
    )
    rubric_norm_covered = normalize_score_map(
        rubric_raw_scores,
        method=normalization_method,
        universe=rubric_covered_universe,
        fill_missing_with=0.0,
    )

    sample_norm_out: Dict[str, Optional[float]] = {}
    rubric_norm_out: Dict[str, Optional[float]] = {}
    fused_entries: List[Dict[str, Any]] = []

    for ref in union_refs:
        has_sample_ref_coverage = ref in sample_available_refs
        has_rubric_ref_coverage = ref in rubric_available_refs

        s_raw = sample_raw_scores.get(ref)
        r_raw = rubric_raw_scores.get(ref)

        if s_raw is not None:
            s_status = "retrieved"
        elif has_sample_ref_coverage:
            s_status = "not_in_top_k"
        else:
            s_status = "no_reference_samples"

        if r_raw is not None:
            r_status = "retrieved"
        elif has_rubric_ref_coverage:
            r_status = "not_in_top_k"
        else:
            r_status = "no_reference_prototypes"

        s_norm: Optional[float] = sample_norm_covered.get(ref) if has_sample_ref_coverage else None
        r_norm: Optional[float] = rubric_norm_covered.get(ref) if has_rubric_ref_coverage else None
        sample_norm_out[ref] = s_norm
        rubric_norm_out[ref] = r_norm

        missing_branches: List[str] = []
        if not has_sample_ref_coverage:
            missing_branches.append("sample")
        if not has_rubric_ref_coverage:
            missing_branches.append("rubric")

        if missing_branch_policy == "reweight_available_branch":
            if has_sample_ref_coverage and has_rubric_ref_coverage:
                eff_alpha = alpha_val
                fused = eff_alpha * float(s_norm or 0.0) + (1.0 - eff_alpha) * float(r_norm or 0.0)
            elif has_rubric_ref_coverage:
                eff_alpha = 0.0
                fused = float(r_norm or 0.0)
            elif has_sample_ref_coverage:
                eff_alpha = 1.0
                fused = float(s_norm or 0.0)
            else:
                eff_alpha = alpha_val
                fused = 0.0
        elif missing_branch_policy == "zero_fill":
            eff_alpha = alpha_val
            fused = eff_alpha * float(s_norm or 0.0) + (1.0 - eff_alpha) * float(r_norm or 0.0)
        else:
            raise ValueError(f"Unsupported missing_branch_reference_policy: '{missing_branch_policy}'")

        meta = ref_to_meta.get(ref) or {}
        fused_entries.append(
            {
                "criterion_ref": ref,
                "criterion_key": meta.get("criterion_key"),
                "group_id": meta.get("group_id"),
                "criterion_name": meta.get("name"),
                "fused_score": round(float(fused), 6),
                "sample_raw_score": round(float(s_raw), 6) if s_raw is not None else None,
                "sample_norm_score": round(float(s_norm), 6) if s_norm is not None else None,
                "sample_evidence_status": s_status,
                "rubric_raw_score": round(float(r_raw), 6) if r_raw is not None else None,
                "rubric_norm_score": round(float(r_norm), 6) if r_norm is not None else None,
                "rubric_evidence_status": r_status,
                "effective_alpha": eff_alpha,
                "missing_branches": missing_branches,
            }
        )

    fused_entries.sort(
        key=lambda e: (
            -e["fused_score"],
            -(e["rubric_raw_score"] if e["rubric_raw_score"] is not None else -1.0),
            -(e["sample_raw_score"] if e["sample_raw_score"] is not None else -1.0),
            e["criterion_ref"],
        )
    )
    for rank_idx, entry in enumerate(fused_entries, start=1):
        entry["rank"] = rank_idx

    return fused_entries, sample_norm_out, rubric_norm_out
