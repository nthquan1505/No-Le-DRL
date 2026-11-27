"""Loaders for classification inputs, reference embeddings, rubric, and config."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.db_event_adapter import (
    compute_events_fingerprint,
    db_row_to_classification_event,
    fetch_events_from_db,
)


@dataclass(frozen=True)
class ReferenceBranchDataset:
    """Immutable container for one independent reference branch (Sample or Rubric)."""

    branch_name: str
    embeddings: np.ndarray
    metadata: pd.DataFrame
    embeddings_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class LoadedAssets:
    """All validated read-only inputs required to run event classification."""

    config: Dict[str, Any]
    events: List[Dict[str, Any]]
    events_path: Path
    events_sha256: str
    rubric: Dict[str, Any]
    rubric_path: Path
    embedding_config: Dict[str, Any]
    embedding_config_path: Path
    sample_branch: ReferenceBranchDataset
    rubric_branch: ReferenceBranchDataset
    source_mode: str = "json"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required YAML file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    return data


def load_json_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_classification_config(config_path: Path) -> Dict[str, Any]:
    return load_yaml_file(config_path)


def load_events(events_path: Path) -> List[Dict[str, Any]]:
    data = load_json_file(events_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {events_path}, got {type(data).__name__}")
    for idx, rec in enumerate(data):
        if not isinstance(rec, dict):
            raise ValueError(f"Event record at index {idx} in {events_path} is not a JSON object.")
    return data


def load_reference_branch(
    branch_name: str,
    embeddings_path: Path,
    metadata_path: Path,
) -> ReferenceBranchDataset:
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing {branch_name} embeddings file: {embeddings_path}. "
            "Do not substitute another dataset."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {branch_name} metadata parquet file: {metadata_path}. "
            "Do not substitute another dataset."
        )
    # Load read-only via mmap and copy/view as float32 read-only array
    embeddings = np.load(embeddings_path, mmap_mode="r")
    metadata = pd.read_parquet(metadata_path)
    return ReferenceBranchDataset(
        branch_name=branch_name,
        embeddings=embeddings,
        metadata=metadata,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )


def load_events_from_database(
    *,
    db: Any = None,
    db_path: Optional[Path] = None,
    only_unclassified: bool = False,
    exclude_cancelled: bool = True,
) -> Tuple[List[Dict[str, Any]], str, Path]:
    """Load scraped events from `drl_database.db` (`events` table) via `DRLDatabaseManager`."""
    rows = fetch_events_from_db(
        db=db,
        db_path=db_path,
        only_unclassified=only_unclassified,
        exclude_cancelled=exclude_cancelled,
    )
    events = [db_row_to_classification_event(r) for r in rows]
    fingerprint = compute_events_fingerprint(events)
    resolved_db_path = (
        Path(db.db_path)
        if db is not None and hasattr(db, "db_path")
        else (Path(db_path) if db_path is not None else Path("drl_database.db"))
    )
    return events, fingerprint, resolved_db_path


def load_all_assets(
    config_path: Path,
    project_root: Path | None = None,
    *,
    input_source_override: Optional[str] = None,
    db: Any = None,
    db_path_override: Optional[Path] = None,
    only_unclassified: Optional[bool] = None,
    exclude_cancelled: Optional[bool] = None,
) -> LoadedAssets:
    """Load all required input files (or SQLite DB events) using classification_config.yaml."""
    root = project_root if project_root is not None else Path.cwd()
    config = load_classification_config(config_path)
    paths_cfg = config.get("paths") or {}
    source_cfg = config.get("input_source") or {}

    def resolve(key: str, default: Optional[str] = None) -> Path:
        raw = paths_cfg.get(key, default)
        if not raw:
            raise ValueError(f"Missing required path key 'paths.{key}' in {config_path}")
        p = Path(raw)
        return p if p.is_absolute() else root / p

    source_mode = str(input_source_override or source_cfg.get("mode", "json")).strip().lower()
    if source_mode not in ("json", "database"):
        raise ValueError(f"Unsupported input_source mode {source_mode!r}. Expected 'database' or 'json'.")

    rubric_path = resolve("rubric_yaml")
    emb_cfg_path = resolve("embedding_config")
    sample_emb_path = resolve("sample_embeddings")
    sample_meta_path = resolve("sample_metadata")
    rubric_emb_path = resolve("rubric_embeddings")
    rubric_meta_path = resolve("rubric_metadata")

    if source_mode == "database":
        resolved_db_path = (
            Path(db_path_override)
            if db_path_override is not None
            else resolve("db_path", default="drl_database.db")
        )
        only_uncls = (
            bool(only_unclassified)
            if only_unclassified is not None
            else bool(source_cfg.get("only_unclassified", False))
        )
        excl_cancel = (
            bool(exclude_cancelled)
            if exclude_cancelled is not None
            else bool(source_cfg.get("exclude_cancelled", True))
        )
        events, events_sha256, events_path = load_events_from_database(
            db=db,
            db_path=resolved_db_path,
            only_unclassified=only_uncls,
            exclude_cancelled=excl_cancel,
        )
    else:
        events_path = resolve("events_json")
        events = load_events(events_path)
        events_sha256 = sha256_file(events_path)

    rubric = load_yaml_file(rubric_path)
    embedding_config = load_json_file(emb_cfg_path)
    if not isinstance(embedding_config, dict):
        raise ValueError(f"Expected JSON object in {emb_cfg_path}")

    sample_branch = load_reference_branch("sample", sample_emb_path, sample_meta_path)
    rubric_branch = load_reference_branch("rubric", rubric_emb_path, rubric_meta_path)

    return LoadedAssets(
        config=config,
        events=events,
        events_path=events_path,
        events_sha256=events_sha256,
        rubric=rubric,
        rubric_path=rubric_path,
        embedding_config=embedding_config,
        embedding_config_path=emb_cfg_path,
        sample_branch=sample_branch,
        rubric_branch=rubric_branch,
        source_mode=source_mode,
    )
