"""Configuration loader for the No-Le-DRL Recommendation subsystem."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from configs.model_config import PROJECT_ROOT

DEFAULT_RECOMMENDATION_CONFIG_PATH: Path = (
    PROJECT_ROOT / "configs" / "recommendation_config.yaml"
)
DEFAULT_SCHEDULE_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "schedule.yaml"
REQUIRED_TOP_LEVEL_KEYS = ("interest_chips", "weight_presets", "profile", "schedule")


def _deep_merge_defaults(base: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in target.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge_defaults(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_recommendation_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and validate required sections of `configs/recommendation_config.yaml`."""
    path = Path(config_path) if config_path is not None else DEFAULT_RECOMMENDATION_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Recommendation config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(raw).__name__}")

    if config_path is None and DEFAULT_SCHEDULE_CONFIG_PATH.is_file():
        sched_raw = yaml.safe_load(DEFAULT_SCHEDULE_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(sched_raw, dict) and isinstance(sched_raw.get("schedule"), dict):
            raw["schedule"] = _deep_merge_defaults(sched_raw["schedule"], dict(raw.get("schedule") or {}))

    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in raw]
    if missing:
        raise ValueError(f"Missing required section(s) in {path}: {missing}")

    return raw
