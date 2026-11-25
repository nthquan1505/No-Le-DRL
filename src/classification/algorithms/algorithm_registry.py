"""Extensible registry mapping algorithm names to classification implementations."""

from __future__ import annotations

from typing import Any, Dict, List, Type

from src.classification.algorithms.base_algorithm import BaseClassificationAlgorithm
from src.classification.algorithms.hierarchical_two_stage_knn import HierarchicalTwoStageKNN
from src.classification.retrieval.rubric_retriever import RubricRetriever
from src.classification.retrieval.sample_retriever import SampleRetriever


class AlgorithmRegistry:
    """Registry for pluggable event classification algorithms."""

    def __init__(self) -> None:
        self._registry: Dict[str, Type[BaseClassificationAlgorithm]] = {}

    def register(
        self,
        name: str,
        algorithm_cls: Type[BaseClassificationAlgorithm],
    ) -> None:
        if not issubclass(algorithm_cls, BaseClassificationAlgorithm):
            raise TypeError(
                f"{algorithm_cls!r} must inherit from BaseClassificationAlgorithm."
            )
        self._registry[str(name)] = algorithm_cls

    def get(self, name: str) -> Type[BaseClassificationAlgorithm]:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "(none)"
            raise KeyError(
                f"Unknown classification algorithm '{name}'. Registered algorithms: {available}"
            )
        return self._registry[name]

    def create(
        self,
        name: str,
        *,
        sample_retriever: SampleRetriever,
        rubric_retriever: RubricRetriever,
        rubric_data: Dict[str, Any],
        params: Dict[str, Any],
    ) -> BaseClassificationAlgorithm:
        cls = self.get(name)
        return cls(
            sample_retriever=sample_retriever,
            rubric_retriever=rubric_retriever,
            rubric_data=rubric_data,
            params=params,
        )

    def list_algorithms(self) -> List[str]:
        return sorted(self._registry.keys())


def build_default_registry() -> AlgorithmRegistry:
    """Create the default algorithm registry containing initial registered algorithms."""
    registry = AlgorithmRegistry()
    registry.register(HierarchicalTwoStageKNN.name, HierarchicalTwoStageKNN)
    return registry
