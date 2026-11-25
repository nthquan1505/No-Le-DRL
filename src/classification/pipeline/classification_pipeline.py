"""Algorithm-agnostic orchestration pipeline for HCMUS DRL event classification.

Responsibilities (orchestration ONLY; zero kNN or fusion internals):
  1. Load and validate input events, rubric.yaml, embedding_config.json, and both
     independent reference branches (Sample and Rubric).
  2. Generate or reuse ONE shared query embedding per classifiable event.
  3. Instantiate independent SampleRetriever and RubricRetriever.
  4. Resolve the configured classification algorithm through AlgorithmRegistry.
  5. Pass the exact same shared query vector to the algorithm for every eligible event.
  6. Apply ConfidenceAndDecisionHandler to produce standardized classification records.
  7. Preserve all original event fields and order, attach `classification`, and write
     `classification_events.json` and `classification_run_report.md`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.classification.algorithms.algorithm_registry import (
    AlgorithmRegistry,
    build_default_registry,
)
from src.classification.data.loaders import LoadedAssets, load_all_assets
from src.classification.data.validators import (
    ValidationReport,
    validate_loaded_assets,
    validate_output_records,
    validate_query_embeddings,
)
from src.classification.decision.confidence_handler import ConfidenceAndDecisionHandler
from src.classification.embeddings.event_embedder import (
    SharedQueryEmbeddingSet,
    generate_or_load_shared_query_embeddings,
)
from src.classification.output.result_writer import (
    write_classification_events,
    write_classification_run_report,
)
from src.classification.retrieval.rubric_retriever import RubricRetriever
from src.classification.retrieval.sample_retriever import SampleRetriever


@dataclass
class PipelineRunSummary:
    """Structured technical execution summary returned by ClassificationPipeline.run()."""

    algorithm_name: str
    total_input_events: int
    embedded_events_count: int
    skipped_events_count: int
    output_events_count: int
    status_counts: Dict[str, int]
    query_embeddings_reused: bool
    output_json_path: Path
    output_report_path: Path
    query_embeddings_path: Path
    query_metadata_path: Path
    asset_validation: ValidationReport
    query_validation: ValidationReport
    output_validation: ValidationReport
    sample_stage1_calls: int
    sample_stage2_calls: int
    rubric_stage1_calls: int
    rubric_stage2_calls: int


class ClassificationPipeline:
    """Modular, algorithm-agnostic orchestrator for HCMUS DRL event classification."""

    def __init__(
        self,
        config_path: Path,
        *,
        project_root: Optional[Path] = None,
        registry: Optional[AlgorithmRegistry] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()
        self.registry = registry if registry is not None else build_default_registry()

    def _resolve_path(self, raw_path: str) -> Path:
        p = Path(raw_path)
        return p if p.is_absolute() else self.project_root / p

    def run(self) -> PipelineRunSummary:
        # 1. Load & validate all read-only assets
        assets: LoadedAssets = load_all_assets(self.config_path, project_root=self.project_root)
        asset_validation = validate_loaded_assets(assets)

        paths_cfg = assets.config["paths"]
        emb_cfg = assets.config.get("embedding") or {}
        alg_cfg = assets.config.get("algorithm") or {}
        dec_cfg = assets.config.get("decision") or {}

        algorithm_name = str(alg_cfg.get("name") or "")
        if not algorithm_name:
            raise ValueError("Missing 'algorithm.name' in classification_config.yaml")
        algorithm_params = dict(alg_cfg.get("params") or {})

        query_emb_path = self._resolve_path(paths_cfg["query_embeddings"])
        query_meta_path = self._resolve_path(paths_cfg["query_metadata"])
        output_json_path = self._resolve_path(paths_cfg["output_classification_json"])
        output_report_path = self._resolve_path(paths_cfg["output_run_report_md"])

        # 2. Generate or reuse ONE shared query embedding per classifiable event
        query_set: SharedQueryEmbeddingSet = generate_or_load_shared_query_embeddings(
            events=assets.events,
            events_path=assets.events_path,
            events_sha256=assets.events_sha256,
            embedding_config=assets.embedding_config,
            query_embeddings_path=query_emb_path,
            query_metadata_path=query_meta_path,
            reuse_existing=bool(emb_cfg.get("reuse_existing_query_embeddings", True)),
            batch_size=int(emb_cfg.get("batch_size", 32)),
            device=str(emb_cfg.get("device", "cpu")),
            l2_tolerance=float(emb_cfg.get("l2_tolerance", 1e-4)),
        )
        query_validation = validate_query_embeddings(
            query_embeddings=query_set.embeddings,
            query_metadata=query_set.metadata,
            events=assets.events,
            expected_dim=int(assets.embedding_config["embedding_dimension"]),
            l2_tolerance=float(emb_cfg.get("l2_tolerance", 1e-4)),
        )

        # 3. Instantiate independent Sample and Rubric retrieval branches
        sample_retriever = SampleRetriever(assets.sample_branch)
        rubric_retriever = RubricRetriever(assets.rubric_branch, assets.rubric)

        # 4. Resolve configured algorithm via AlgorithmRegistry
        algorithm = self.registry.create(
            algorithm_name,
            sample_retriever=sample_retriever,
            rubric_retriever=rubric_retriever,
            rubric_data=assets.rubric,
            params=algorithm_params,
        )

        # 5. Instantiate provisional confidence/decision handler
        decision_handler = ConfidenceAndDecisionHandler(
            rubric_retriever=rubric_retriever,
            decision_config=dec_cfg,
            max_suggested_candidates=int(algorithm_params.get("max_suggested_candidates", 5)),
        )

        # 6. Classify every input event in original order using the shared query vector
        output_events: List[Dict[str, Any]] = []
        status_counts: Dict[str, int] = {
            "PREDICTED": 0,
            "NEEDS_REVIEW": 0,
            "NO_MATCH": 0,
            "SKIPPED_NOT_EVENT": 0,
        }

        for idx, raw_event in enumerate(assets.events):
            out_record = copy.deepcopy(raw_event)
            if idx in query_set.skipped_by_event_index:
                skip_info = query_set.skipped_by_event_index[idx]
                cls_payload = decision_handler.build_skipped_result(
                    algorithm_name=algorithm.name,
                    skip_reason=str(skip_info.get("skip_reason", "skipped_not_event")),
                )
            else:
                emb_row = query_set.event_index_to_row[idx]
                shared_query_vec = query_set.embeddings[emb_row]
                prediction = algorithm.classify(shared_query_vec, event_context=raw_event)
                cls_payload = decision_handler.decide(prediction=prediction, event=raw_event)

            status = cls_payload["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            out_record["classification"] = cls_payload
            output_events.append(out_record)

        # 7. Technical output validation (Section 14)
        output_validation = validate_output_records(
            input_events=assets.events,
            output_events=output_events,
            rubric=assets.rubric,
            expected_algorithm=algorithm.name,
        )

        # Verify shared query call counts match embedded events 1:1
        embedded_n = int(query_set.metadata["embedded_events_count"])
        shared_calls_ok = (
            sample_retriever.stage1_query_calls == embedded_n
            and rubric_retriever.stage1_query_calls == embedded_n
            and sample_retriever.stage2_query_calls == embedded_n
            and rubric_retriever.stage2_query_calls == embedded_n
        )
        output_validation.add(
            "shared_query_reused_across_branches_and_stages",
            shared_calls_ok,
            (
                f"embedded={embedded_n}; calls: sample_s1={sample_retriever.stage1_query_calls}, "
                f"rubric_s1={rubric_retriever.stage1_query_calls}, "
                f"sample_s2={sample_retriever.stage2_query_calls}, "
                f"rubric_s2={rubric_retriever.stage2_query_calls}"
            ),
        )

        # 8. Serialize output JSON and Markdown run report
        write_classification_events(output_events, output_json_path)
        write_classification_run_report(
            report_path=output_report_path,
            assets=assets,
            query_set=query_set,
            output_events=output_events,
            status_counts=status_counts,
            algorithm_name=algorithm.name,
            registered_algorithms=self.registry.list_algorithms(),
            asset_validation=asset_validation,
            query_validation=query_validation,
            output_validation=output_validation,
            output_json_path=output_json_path,
        )

        return PipelineRunSummary(
            algorithm_name=algorithm.name,
            total_input_events=len(assets.events),
            embedded_events_count=embedded_n,
            skipped_events_count=int(query_set.metadata["skipped_events_count"]),
            output_events_count=len(output_events),
            status_counts=status_counts,
            query_embeddings_reused=query_set.reused_from_cache,
            output_json_path=output_json_path,
            output_report_path=output_report_path,
            query_embeddings_path=query_emb_path,
            query_metadata_path=query_meta_path,
            asset_validation=asset_validation,
            query_validation=query_validation,
            output_validation=output_validation,
            sample_stage1_calls=sample_retriever.stage1_query_calls,
            sample_stage2_calls=sample_retriever.stage2_query_calls,
            rubric_stage1_calls=rubric_retriever.stage1_query_calls,
            rubric_stage2_calls=rubric_retriever.stage2_query_calls,
        )
