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
    MultiProfileQueryEmbeddingSet,
    SharedQueryEmbeddingSet,
    generate_or_load_multi_profile_query_embeddings,
    generate_or_load_shared_query_embeddings,
)
from src.classification.embeddings.event_text_builder import CLASSIFICATION_VECTOR_PROFILES
from src.classification.output.result_writer import (
    write_classification_events,
    write_classification_run_report,
)
from src.classification.retrieval.rubric_retriever import RubricRetriever
from src.classification.retrieval.sample_retriever import SampleRetriever
from src.db_event_adapter import write_classification_category_to_db


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

    def _apply_branch_vector_params(
        self,
        algorithm_params: Dict[str, Any],
        emb_cfg: Dict[str, Any],
        branch_vector_override: Optional[Dict[str, Optional[str]]] = None,
    ) -> Dict[str, Any]:
        params = dict(algorithm_params)
        sel = dict(emb_cfg.get("branch_vector_selection") or {})
        if branch_vector_override:
            for k, v in branch_vector_override.items():
                if v is not None:
                    sel[k] = v

        sample_vec = str(sel.get("sample_branch") or "title_description")
        rubric_vec = str(sel.get("rubric_branch") or "title_description")
        for label, val in (("sample_branch", sample_vec), ("rubric_branch", rubric_vec)):
            if val not in CLASSIFICATION_VECTOR_PROFILES:
                raise ValueError(
                    f"Invalid {label} vector profile {val!r}. "
                    f"Expected one of {CLASSIFICATION_VECTOR_PROFILES}."
                )

        params["sample_branch_vector"] = sample_vec
        params["rubric_branch_vector"] = rubric_vec
        for stage_key in ("sample_stage1", "sample_stage2", "rubric_stage1", "rubric_stage2"):
            stage_val = sel.get(stage_key)
            if stage_val is not None:
                if stage_val not in CLASSIFICATION_VECTOR_PROFILES:
                    raise ValueError(
                        f"Invalid {stage_key} vector profile {stage_val!r}. "
                        f"Expected one of {CLASSIFICATION_VECTOR_PROFILES}."
                    )
                params[f"{stage_key}_vector"] = stage_val
        return params

    def run_from_database(
        self,
        *,
        db: Any = None,
        db_path: Optional[Path] = None,
        only_unclassified: Optional[bool] = None,
        exclude_cancelled: Optional[bool] = None,
        branch_vector_selection: Optional[Dict[str, Optional[str]]] = None,
        write_back_db: Optional[bool] = None,
        write_output_files: bool = False,
        embedder: Any = None,
    ) -> Dict[str, Any]:
        """Run dual-branch 3-vector Classification directly against `drl_database.db`.

        Generates all 3 classification query vectors (`title`, `title_description`,
        `title_description_format`) per classifiable event, routes the configured
        vector independently to Sample Branch vs Rubric Branch, handles empty DB
        (`0` events) cleanly, and optionally writes `events.category` (`1..5`) back.
        """
        assets: LoadedAssets = load_all_assets(
            self.config_path,
            project_root=self.project_root,
            input_source_override="database",
            db=db,
            db_path_override=db_path,
            only_unclassified=only_unclassified,
            exclude_cancelled=exclude_cancelled,
        )
        asset_validation = validate_loaded_assets(assets)

        paths_cfg = assets.config["paths"]
        source_cfg = assets.config.get("input_source") or {}
        emb_cfg = assets.config.get("embedding") or {}
        alg_cfg = assets.config.get("algorithm") or {}
        dec_cfg = assets.config.get("decision") or {}

        algorithm_name = str(alg_cfg.get("name") or "")
        if not algorithm_name:
            raise ValueError("Missing 'algorithm.name' in classification_config.yaml")

        profiles = tuple(emb_cfg.get("classification_profiles") or CLASSIFICATION_VECTOR_PROFILES)
        algorithm_params = self._apply_branch_vector_params(
            dict(alg_cfg.get("params") or {}),
            emb_cfg,
            branch_vector_override=branch_vector_selection,
        )

        mp_emb_path = (
            self._resolve_path(paths_cfg["multi_profile_query_embeddings"])
            if paths_cfg.get("multi_profile_query_embeddings") and write_output_files
            else None
        )
        mp_meta_path = (
            self._resolve_path(paths_cfg["multi_profile_query_metadata"])
            if paths_cfg.get("multi_profile_query_metadata") and write_output_files
            else None
        )

        multi_set: MultiProfileQueryEmbeddingSet = generate_or_load_multi_profile_query_embeddings(
            events=assets.events,
            events_path=assets.events_path,
            events_sha256=assets.events_sha256,
            embedding_config=assets.embedding_config,
            profiles=profiles,
            embeddings_path=mp_emb_path,
            metadata_path=mp_meta_path,
            reuse_existing=bool(emb_cfg.get("reuse_existing_query_embeddings", True)),
            batch_size=int(emb_cfg.get("batch_size", 32)),
            device=str(emb_cfg.get("device", "cpu")),
            l2_tolerance=float(emb_cfg.get("l2_tolerance", 1e-4)),
            embedder=embedder,
        )

        sample_retriever = SampleRetriever(assets.sample_branch)
        rubric_retriever = RubricRetriever(assets.rubric_branch, assets.rubric)
        algorithm = self.registry.create(
            algorithm_name,
            sample_retriever=sample_retriever,
            rubric_retriever=rubric_retriever,
            rubric_data=assets.rubric,
            params=algorithm_params,
        )
        decision_handler = ConfidenceAndDecisionHandler(
            rubric_retriever=rubric_retriever,
            decision_config=dec_cfg,
            max_suggested_candidates=int(algorithm_params.get("max_suggested_candidates", 5)),
        )

        should_write_db = (
            bool(write_back_db)
            if write_back_db is not None
            else bool(source_cfg.get("write_back_category", True))
        )

        output_events: List[Dict[str, Any]] = []
        status_counts: Dict[str, int] = {
            "PREDICTED": 0,
            "NEEDS_REVIEW": 0,
            "NO_MATCH": 0,
            "SKIPPED_NOT_EVENT": 0,
        }
        db_categories_updated = 0

        for idx, raw_event in enumerate(assets.events):
            out_record = copy.deepcopy(raw_event)
            if idx in multi_set.skipped_by_event_index:
                skip_info = multi_set.skipped_by_event_index[idx]
                cls_payload = decision_handler.build_skipped_result(
                    algorithm_name=algorithm.name,
                    skip_reason=str(skip_info.get("skip_reason", "skipped_not_event")),
                )
            else:
                emb_row = multi_set.event_index_to_row[idx]
                vecs_for_event = multi_set.get_vectors_for_row(emb_row)
                default_vec = vecs_for_event.get("title_description")
                if default_vec is None:
                    default_vec = next(iter(vecs_for_event.values()))
                prediction = algorithm.classify(
                    default_vec,
                    event_context=raw_event,
                    query_vectors_by_profile=vecs_for_event,
                )
                cls_payload = decision_handler.decide(prediction=prediction, event=raw_event)

                if (
                    should_write_db
                    and cls_payload.get("status") == "PREDICTED"
                    and raw_event.get("event_id") is not None
                ):
                    ok_upd = write_classification_category_to_db(
                        int(raw_event["event_id"]),
                        cls_payload.get("predicted_group_id"),
                        db=db,
                        db_path=assets.events_path,
                    )
                    if ok_upd:
                        db_categories_updated += 1

            status = cls_payload["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            out_record["classification"] = cls_payload
            output_events.append(out_record)

        if write_output_files:
            output_json_path = self._resolve_path(paths_cfg["output_classification_json"])
            write_classification_events(output_events, output_json_path)

        return {
            "source_mode": "database",
            "db_path": str(assets.events_path),
            "algorithm": algorithm.name,
            "profiles_computed": list(profiles),
            "branch_vector_selection": {
                "sample_branch": algorithm_params["sample_branch_vector"],
                "rubric_branch": algorithm_params["rubric_branch_vector"],
                "sample_stage1": algorithm_params.get("sample_stage1_vector", algorithm_params["sample_branch_vector"]),
                "sample_stage2": algorithm_params.get("sample_stage2_vector", algorithm_params["sample_branch_vector"]),
                "rubric_stage1": algorithm_params.get("rubric_stage1_vector", algorithm_params["rubric_branch_vector"]),
                "rubric_stage2": algorithm_params.get("rubric_stage2_vector", algorithm_params["rubric_branch_vector"]),
            },
            "total_input_events": len(assets.events),
            "embedded_events_count": int(multi_set.metadata["embedded_events_count"]),
            "skipped_events_count": int(multi_set.metadata["skipped_events_count"]),
            "status_counts": status_counts,
            "db_categories_updated": db_categories_updated,
            "asset_validation_passed": asset_validation.all_passed,
            "events": output_events,
            "multi_profile_embeddings": multi_set,
        }

    def run(
        self,
        *,
        branch_vector_selection: Optional[Dict[str, Optional[str]]] = None,
    ) -> PipelineRunSummary:
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
        algorithm_params = self._apply_branch_vector_params(
            dict(alg_cfg.get("params") or {}),
            emb_cfg,
            branch_vector_override=branch_vector_selection,
        )

        use_multi_profile = (
            algorithm_params["sample_branch_vector"] != "title_description"
            or algorithm_params["rubric_branch_vector"] != "title_description"
            or any(
                algorithm_params.get(f"{k}_vector") not in (None, "title_description")
                for k in ("sample_stage1", "sample_stage2", "rubric_stage1", "rubric_stage2")
            )
        )

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

        multi_set: Optional[MultiProfileQueryEmbeddingSet] = None
        if use_multi_profile:
            profiles = tuple(emb_cfg.get("classification_profiles") or CLASSIFICATION_VECTOR_PROFILES)
            mp_emb_path = (
                self._resolve_path(paths_cfg["multi_profile_query_embeddings"])
                if paths_cfg.get("multi_profile_query_embeddings")
                else None
            )
            mp_meta_path = (
                self._resolve_path(paths_cfg["multi_profile_query_metadata"])
                if paths_cfg.get("multi_profile_query_metadata")
                else None
            )
            multi_set = generate_or_load_multi_profile_query_embeddings(
                events=assets.events,
                events_path=assets.events_path,
                events_sha256=assets.events_sha256,
                embedding_config=assets.embedding_config,
                profiles=profiles,
                embeddings_path=mp_emb_path,
                metadata_path=mp_meta_path,
                reuse_existing=bool(emb_cfg.get("reuse_existing_query_embeddings", True)),
                batch_size=int(emb_cfg.get("batch_size", 32)),
                device=str(emb_cfg.get("device", "cpu")),
                l2_tolerance=float(emb_cfg.get("l2_tolerance", 1e-4)),
            )

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
                vecs_by_prof = (
                    multi_set.get_vectors_for_row(multi_set.event_index_to_row[idx])
                    if multi_set is not None and idx in multi_set.event_index_to_row
                    else None
                )
                prediction = algorithm.classify(
                    shared_query_vec,
                    event_context=raw_event,
                    query_vectors_by_profile=vecs_by_prof,
                )
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
