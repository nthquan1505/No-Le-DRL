"""Classification pipeline orchestration."""

from .classification_pipeline import (
    ClassificationPipeline,
    PipelineRunSummary,
    connect_extractor_to_classifier,
)

__all__ = [
    "ClassificationPipeline",
    "PipelineRunSummary",
    "connect_extractor_to_classifier",
]
