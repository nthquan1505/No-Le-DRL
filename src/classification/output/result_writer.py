"""Writers for outputs/classification_events.json and outputs/classification_run_report.md."""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from src.classification.data.loaders import LoadedAssets
from src.classification.data.validators import ValidationReport
from src.classification.embeddings.event_embedder import SharedQueryEmbeddingSet


def write_classification_events(
    output_events: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Serialize classification_events.json as strict UTF-8 with allow_nan=False."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(output_events, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output_path.write_text(payload, encoding="utf-8")


def write_classification_run_report(
    *,
    report_path: Path,
    assets: LoadedAssets,
    query_set: SharedQueryEmbeddingSet,
    output_events: List[Dict[str, Any]],
    status_counts: Dict[str, int],
    algorithm_name: str,
    registered_algorithms: List[str],
    asset_validation: ValidationReport,
    query_validation: ValidationReport,
    output_validation: ValidationReport,
    output_json_path: Path,
) -> None:
    """Generate outputs/classification_run_report.md (technical summary; no accuracy claims)."""
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    alg_cfg = assets.config.get("algorithm") or {}
    dec_cfg = assets.config.get("decision") or {}
    emb_cfg = assets.embedding_config
    all_passed = (
        asset_validation.all_passed
        and query_validation.all_passed
        and output_validation.all_passed
    )

    predicted_ref_counts = Counter(
        ev["classification"]["predicted_criterion_ref"]
        for ev in output_events
        if ev["classification"]["status"] == "PREDICTED"
    )
    review_reason_counts = Counter()
    for ev in output_events:
        if ev["classification"]["status"] in ("NEEDS_REVIEW", "NO_MATCH"):
            for r in ev["classification"].get("review_reasons") or []:
                key = r.split(" (", 1)[0]
                review_reason_counts[key] += 1

    lines: List[str] = [
        "# Classification Run Report — HCMUS DRL Event Classification",
        "",
        f"- **Run Timestamp (UTC)**: `{now_utc}`",
        f"- **Selected Algorithm**: `{algorithm_name}` (via `AlgorithmRegistry`; registered: `{registered_algorithms}`)",
        f"- **Parameter & Threshold Status**: `provisional_not_evaluated` (`calibrated: false`)",
        f"- **Technical Validation Status**: **{'ALL CHECKS PASSED' if all_passed else 'FAILED CHECKS PRESENT'}**",
        "",
        "> **Scope Notice**: This report records technical execution, artifact alignment, and provisional status counts only. Model evaluation (accuracy, precision, recall, F1) and parameter tuning are explicitly out of scope for this phase (see `FUTURE_WORK.md`).",
        "",
        "---",
        "",
        "## 1. Run Counts & Deliverables",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Input events (`{assets.events_path.as_posix()}`) | **{len(assets.events)}** |",
        f"| Embedded classifiable events (`events_query_embeddings.npy`) | **{query_set.metadata['embedded_events_count']}** |",
        f"| Skipped non-event / empty-text records | **{query_set.metadata['skipped_events_count']}** |",
        f"| Shared query embeddings reused from cache | `{query_set.reused_from_cache}` |",
        f"| Output records (`{output_json_path.as_posix()}`) | **{len(output_events)}** |",
        f"| Sample reference vectors (`sample_embeddings.npy`) | `{tuple(assets.sample_branch.embeddings.shape)}` |",
        f"| Rubric reference vectors (`rubric_embeddings.npy`) | `{tuple(assets.rubric_branch.embeddings.shape)}` |",
        "",
        "### Classification Status Breakdown (Provisional Settings)",
        "",
        "| Status | Count | Share |",
        "|---|---|---|",
    ]

    total = max(len(output_events), 1)
    for st in ("PREDICTED", "NEEDS_REVIEW", "NO_MATCH", "SKIPPED_NOT_EVENT"):
        cnt = status_counts.get(st, 0)
        lines.append(f"| `{st}` | {cnt} | {100.0 * cnt / total:.1f}% |")

    lines += [
        "",
        "### Predicted Criterion Distribution (`status = PREDICTED` only)",
        "",
    ]
    if predicted_ref_counts:
        lines += [
            "| `criterion_ref` | Count |",
            "|---|---|",
        ]
        for ref, cnt in sorted(
            predicted_ref_counts.items(),
            key=lambda kv: (-kv[1], [int(p) for p in str(kv[0]).split(".")]),
        ):
            lines.append(f"| `{ref}` | {cnt} |")
    else:
        lines.append("- None under current provisional thresholds.")

    lines += [
        "",
        "### Review / Abstention Trigger Frequency",
        "",
    ]
    if review_reason_counts:
        lines += [
            "| Trigger | Occurrences |",
            "|---|---|",
        ]
        for trig, cnt in review_reason_counts.most_common():
            lines.append(f"| `{trig}` | {cnt} |")
    else:
        lines.append("- None.")

    lines += [
        "",
        "---",
        "",
        "## 2. Shared Query Embedding Configuration",
        "",
        f"- **Model**: `{emb_cfg.get('model_name')}` (revision: `{emb_cfg.get('model_revision')}`)",
        f"- **Dimension / Dtype / Normalization**: `{emb_cfg.get('embedding_dimension')}d`, `{emb_cfg.get('dtype')}`, `L2` (`pooling={emb_cfg.get('pooling_method')}`)",
        f"- **Query Prefix**: `\"{(emb_cfg.get('prefixes') or {}).get('query', 'query: ')}\"`",
        f"- **Query Format**: `query: [event title] [event description]` (title + description only; zero label/category/reason leakage)",
        f"- **Token Stats**: `{json.dumps(query_set.metadata.get('token_stats', {}), ensure_ascii=False)}`",
        f"- **Shared Across Branches & Stages**: `True` (single vector per eligible event passed identically to Sample Stage 1, Rubric Stage 1, Sample Stage 2, and Rubric Stage 2)",
        "",
        "---",
        "",
        "## 3. Selected Algorithm & Provisional Parameters",
        "",
        "```json",
        json.dumps(
            {
                "algorithm": alg_cfg,
                "decision": dec_cfg,
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "---",
        "",
        "## 4. Technical Validation Results",
        "",
        "### 4.1. Pre-flight Input & Reference Asset Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for c in asset_validation.checks:
        lines.append(f"| `{c.name}` | **{'PASS' if c.passed else 'FAIL'}** | {c.detail} |")

    lines += [
        "",
        "### 4.2. Shared Query Embedding Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for c in query_validation.checks:
        lines.append(f"| `{c.name}` | **{'PASS' if c.passed else 'FAIL'}** | {c.detail} |")

    lines += [
        "",
        "### 4.3. Output & Workflow Integrity Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for c in output_validation.checks:
        lines.append(f"| `{c.name}` | **{'PASS' if c.passed else 'FAIL'}** | {c.detail} |")

    lines += [
        "",
        "---",
        "",
        "## 5. Processing Notes & Unresolved Issues",
        "",
        "1. **Provisional Parameters**: `beta = 0.5`, `alpha = 0.5`, `K1 = 10`, `K2 = 10`, prototype type weights, and abstention/review thresholds are provisional (`provisional_not_evaluated`). No evaluation or hyperparameter tuning has been performed.",
        "2. **Uncalibrated Scores**: Cosine similarities, weighted kNN votes, and Stage-wise Fusion scores are ranking scores only (`calibrated: false`), not calibrated probabilities.",
        "3. **Sample Coverage Gap**: 10 of 21 `event_matchable: true` criteria have zero samples in `sample_embeddings.npy` (`1.5`, `3.6`, `3.9`–`3.13`, `4.2`, `4.6`, `6.4`). Stage 2 Fusion uses `reweight_available_branch` so missing Sample coverage is represented explicitly rather than treated as proof of irrelevance, while draft-only rubric matches are routed to `NEEDS_REVIEW`.",
        "4. **Unresolved Rubric Boundaries**: Open questions from `rubric.yaml` (`Q10` for `3.1` vs `3.5`, `3.3` explicit mobilization requirement, `Q23` for `1.9` vs `3.5`, `Q17` lack of verified `NONE` samples) remain preserved as metadata and review triggers rather than hardcoded definitive rules.",
        "5. **Classification Boundary**: Event classification classifies announcement text only; it does not verify student attendance, organizer/collaborator role eligibility, or DRL point calculation (handled downstream by Module 6 / Module 7).",
        "",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
