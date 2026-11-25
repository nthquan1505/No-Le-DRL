"""Loaders for classification inputs, reference embeddings, rubric, and config."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml


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


def load_all_assets(config_path: Path, project_root: Path | None = None) -> LoadedAssets:
    """Load all required input files using paths configured in classification_config.yaml."""
    root = project_root if project_root is not None else Path.cwd()
    config = load_classification_config(config_path)
    paths_cfg = config.get("paths") or {}

    def resolve(key: str) -> Path:
        raw = paths_cfg.get(key)
        if not raw:
            raise ValueError(f"Missing required path key 'paths.{key}' in {config_path}")
        p = Path(raw)
        return p if p.is_absolute() else root / p

    events_path = resolve("events_json")
    rubric_path = resolve("rubric_yaml")
    emb_cfg_path = resolve("embedding_config")
    sample_emb_path = resolve("sample_embeddings")
    sample_meta_path = resolve("sample_metadata")
    rubric_emb_path = resolve("rubric_embeddings")
    rubric_meta_path = resolve("rubric_metadata")

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
    )
