"""Common abstract interface and standardized result contracts for classification algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.classification.retrieval.rubric_retriever import RubricRetriever
from src.classification.retrieval.sample_retriever import SampleRetriever


@dataclass
class AlgorithmPrediction:
    """Standardized output produced by any registered classification algorithm.

    Passed from the algorithm module to `ConfidenceAndDecisionHandler` so decision
    handling, output serialization, and future algorithm comparisons stay uniform.
    """

    algorithm_name: str
    selected_groups: List[str]
    stage1_sample_scores: List[Dict[str, Any]]
    stage1_rubric_scores: List[Dict[str, Any]]
    stage1_fused_scores: List[Dict[str, Any]]
    stage2_sample_scores: List[Dict[str, Any]]
    stage2_rubric_scores: List[Dict[str, Any]]
    stage2_fused_scores: List[Dict[str, Any]]
    sample_neighbors: List[Dict[str, Any]]
    rubric_prototypes: List[Dict[str, Any]]
    group_margin: float
    criterion_margin: float
    top_raw_sample_similarity: Optional[float]
    top_raw_rubric_similarity: Optional[float]
    branch_top1_sample_criterion: Optional[str]
    branch_top1_rubric_criterion: Optional[str]
    branch_agreement: Optional[bool]
    top_candidate: Optional[Dict[str, Any]] = None
    criterion_rubric_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    algorithm_metadata: Dict[str, Any] = field(default_factory=dict)


class BaseClassificationAlgorithm(ABC):
    """Abstract base class for pluggable HCMUS DRL classification algorithms.

    Adding a new algorithm (e.g., flat kNN, BM25+semantic hybrid, reranker) only
    requires subclassing `BaseClassificationAlgorithm` and registering it in
    `AlgorithmRegistry` without touching data loading, shared embedding generation,
    the pipeline orchestrator, or the output writer.
    """

    name: str = "base"

    def __init__(
        self,
        *,
        sample_retriever: SampleRetriever,
        rubric_retriever: RubricRetriever,
        rubric_data: Dict[str, Any],
        params: Dict[str, Any],
    ) -> None:
        self.sample_retriever = sample_retriever
        self.rubric_retriever = rubric_retriever
        self.rubric_data = rubric_data
        self.params = dict(params)

    @abstractmethod
    def classify(
        self,
        query_vector: np.ndarray,
        event_context: Optional[Dict[str, Any]] = None,
        query_vectors_by_profile: Optional[Dict[str, np.ndarray]] = None,
    ) -> AlgorithmPrediction:
        """Classify one event using its shared query vector or per-profile query vectors."""
