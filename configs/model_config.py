"""Runtime configuration and open experimental parameter slots for Module 3.

Aligned with approved decisions D-1 through D-6 and taxonomy decisions T-1 through T-5
in docs/module3_decision_log.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

# -----------------------------------------------------------------------------
# Embedding Model Constants (P2.1)
# -----------------------------------------------------------------------------
MODEL_ID: str = os.environ.get("NO_LE_DRL_MODEL_ID", "intfloat/multilingual-e5-small")
MODEL_DEVICE: str = os.environ.get("NO_LE_DRL_MODEL_DEVICE", "cpu")
EMBEDDING_DIMENSION: int = 384

E5_QUERY_PREFIX: str = "query: "
E5_PASSAGE_PREFIX: str = "passage: "

# -----------------------------------------------------------------------------
# Canonical Repository Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
RES_DIR: Path = PROJECT_ROOT / "res"

RUBRIC_YAML_PATH: Path = RAW_DATA_DIR / "rubric.yaml"
RAW_EVENTS_JSON_PATH: Path = DATA_DIR / "events.json"
DEMO_EVENTS_FULL_CSV_PATH: Path = RES_DIR / "events_full.csv"
DEMO_EVENTS_DATASET_CSV_PATH: Path = RAW_DATA_DIR / "events_dataset.csv"

EVENT_EMBEDDINGS_PATH: Path = PROCESSED_DATA_DIR / "event_embeddings.npy"
EVENT_METADATA_PATH: Path = PROCESSED_DATA_DIR / "event_metadata.json"
CLS_INDEX_DIR: Path = PROCESSED_DATA_DIR / "cls_index"

# -----------------------------------------------------------------------------
# Scoring Weights Metadata (D-5: 4 weights w1..w4 + additive unweighted Penalty)
# -----------------------------------------------------------------------------
DEFAULT_SCORING_WEIGHT_NAMES: Tuple[str, str, str, str] = (
    "w1_gain",
    "w2_interest",
    "w3_fit",
    "w4_urgency",
)


# -----------------------------------------------------------------------------
# Deferred D-2 Hyperparameter Slots (Intentionally Unset Until Validation)
# -----------------------------------------------------------------------------
@dataclass
class HierarchicalKNNConfig:
    """Configuration for the two-stage hierarchical KNN classifier (D-2).

    All numerical hyperparameters below remain intentionally unset (None) until
    empirically selected via validation experiments on the full dataset.
    """

    # Stage 1 (coarse parent-group selection: C1..C5)
    stage1_top_k: Optional[int] = None
    stage1_min_score_threshold: Optional[float] = None
    stage1_neighbors: Optional[int] = None

    # Stage 2 (fine-grained KNN over the union of all selected Stage 1 groups)
    stage2_neighbors: Optional[int] = None
    stage2_aggregation_strategy: Optional[str] = None

    # Calibration & acceptance/rejection gating
    calibration_method: Optional[str] = None
    acceptance_threshold: Optional[float] = None
    rejection_policy: Optional[str] = None

    def open_parameters(self) -> Tuple[str, ...]:
        """Return names of D-2 parameters that have not yet been locked."""
        fields = (
            "stage1_top_k",
            "stage1_min_score_threshold",
            "stage1_neighbors",
            "stage2_neighbors",
            "stage2_aggregation_strategy",
            "calibration_method",
            "acceptance_threshold",
            "rejection_policy",
        )
        return tuple(name for name in fields if getattr(self, name) is None)


@dataclass
class Module3Config:
    """Top-level configuration bundle for Module 3."""

    model_id: str = MODEL_ID
    model_device: str = MODEL_DEVICE
    embedding_dimension: int = EMBEDDING_DIMENSION
    query_prefix: str = E5_QUERY_PREFIX
    passage_prefix: str = E5_PASSAGE_PREFIX
    rubric_yaml_path: Path = RUBRIC_YAML_PATH
    cls_index_dir: Path = CLS_INDEX_DIR
    event_embeddings_path: Path = EVENT_EMBEDDINGS_PATH
    event_metadata_path: Path = EVENT_METADATA_PATH
    # T-2 & T-3: Group P (6.4) and NONE/UNKNOWN are handled upstream by Module 1.
    excluded_parent_groups: Tuple[str, ...] = ("P",)
    knn: HierarchicalKNNConfig = field(default_factory=HierarchicalKNNConfig)
