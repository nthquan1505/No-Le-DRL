#!/usr/bin/env python3
"""CLI entrypoint for the modular HCMUS DRL Event Classification System.

Delegates strictly to `ClassificationPipeline` (no kNN, retrieval, or fusion logic here).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.classification.pipeline.classification_pipeline import ClassificationPipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the modular HCMUS DRL event classification pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/classification_config.yaml"),
        help="Path to classification_config.yaml",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root directory for resolving relative config paths.",
    )
    args = parser.parse_args()

    pipeline = ClassificationPipeline(
        config_path=args.config,
        project_root=args.project_root,
    )
    summary = pipeline.run()

    print(f"Algorithm:           {summary.algorithm_name}")
    print(f"Input events:        {summary.total_input_events}")
    print(f"Embedded events:     {summary.embedded_events_count} (reused={summary.query_embeddings_reused})")
    print(f"Skipped events:      {summary.skipped_events_count}")
    print(f"Output events:       {summary.output_events_count}")
    print(f"Status breakdown:    {summary.status_counts}")
    print(f"Output JSON:         {summary.output_json_path}")
    print(f"Run report:          {summary.output_report_path}")

    all_ok = (
        summary.asset_validation.all_passed
        and summary.query_validation.all_passed
        and summary.output_validation.all_passed
    )
    print(f"Technical checks:    {'ALL PASSED' if all_ok else 'FAILED'}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
