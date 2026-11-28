"""Run from the patched Module 3 checkout with its dependencies installed."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np

from src.recommendation.filter import RecommendationCandidateInput
from src.recommendation.pipeline import run_recommendation_demo
from src.schemas import EventNormalized
from tests.test_recommendation_p5_p6 import _DeterministicStubEmbedder


def main():
    event = EventNormalized(
        event_id="m4", title="Hội thảo AI", organizer="Khoa CNTT",
        start_datetime="2026-11-14T14:00+07:00", end_datetime="2026-11-14T16:00+07:00",
        online_mode="offline", location="cs2_lt", fee_vnd=0,
    )
    classification = {"status": "PREDICTED", "predicted_group_id": "C1", "predicted_criterion_ref": "1.6"}
    payload = {
        "include_mandatory_demo": False, "strict_max_travel": False,
        "reference_now": "2026-11-01T08:00+07:00",
        "schedule": {"calendar_items": [{
            "id": "clinic", "start": {"dateTime": "2026-11-14T12:00+07:00"},
            "end": {"dateTime": "2026-11-14T13:20+07:00"}, "location": "Clinic Q3",
        }]},
    }
    samples = np.zeros((1, 384), dtype=np.float32)
    samples[0, 0] = 1
    calls = []

    def route(origin, destination):
        calls.append((origin, destination))
        return 55

    def run(candidate, **overrides):
        with patch("src.recommendation.pipeline.CarRouting.from_environment", return_value=route):
            return run_recommendation_demo(
                {**payload, **overrides}, candidates=[RecommendationCandidateInput(candidate, classification)],
                sample_embeddings=samples, embedder=_DeterministicStubEmbedder(),
            )

    result = run(event)
    rows = [row for row in result["ranked_events"] if row["event_id"] == "m4"]
    assert len(rows) == 1, result["dropped_events"]
    row = rows[0]
    assert row["components"]["fit"] == 0.52, row
    assert row["fit_assessment"]["nearest_slack_min"] == 40, row
    assert row["fit_assessment"]["estimated_travel_min"] == 55, row
    assert calls == [("Clinic Q3", "cs2_lt")], calls

    strict = run(event, strict_max_travel=True, max_travel_min=45)
    assert "too_far" in strict["pipeline_summary"]["drop_reason_counts"], strict["dropped_events"]
    conflicted = run(replace(event, start_datetime="2026-11-14T13:45+07:00"))
    assert "schedule_conflict" in conflicted["pipeline_summary"]["drop_reason_counts"], conflicted["dropped_events"]
    print("Module 3 P5/P6 integration passed")


if __name__ == "__main__":
    main()
