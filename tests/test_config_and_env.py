"""Phase 0 smoke tests: packaging, configuration constants, and open D-2 parameter slots."""

from __future__ import annotations

import unittest

from configs import (
    DEFAULT_SCORING_WEIGHT_NAMES,
    DEMO_EVENTS_FULL_CSV_PATH,
    E5_PASSAGE_PREFIX,
    E5_QUERY_PREFIX,
    EMBEDDING_DIMENSION,
    MODEL_DEVICE,
    MODEL_ID,
    RAW_EVENTS_JSON_PATH,
    RUBRIC_YAML_PATH,
    HierarchicalKNNConfig,
    Module3Config,
)


class TestConfigAndEnv(unittest.TestCase):
    def test_model_config_constants(self) -> None:
        self.assertEqual(MODEL_ID, "intfloat/multilingual-e5-small")
        self.assertEqual(MODEL_DEVICE, "cpu")
        self.assertEqual(EMBEDDING_DIMENSION, 384)
        self.assertEqual(E5_QUERY_PREFIX, "query: ")
        self.assertEqual(E5_PASSAGE_PREFIX, "passage: ")

    def test_required_data_assets_exist(self) -> None:
        self.assertTrue(RUBRIC_YAML_PATH.is_file(), f"Missing {RUBRIC_YAML_PATH}")
        self.assertTrue(RAW_EVENTS_JSON_PATH.is_file(), f"Missing {RAW_EVENTS_JSON_PATH}")
        self.assertTrue(
            DEMO_EVENTS_FULL_CSV_PATH.is_file(), f"Missing {DEMO_EVENTS_FULL_CSV_PATH}"
        )

    def test_d2_open_parameters_not_hardcoded(self) -> None:
        knn_cfg = HierarchicalKNNConfig()
        expected_open = (
            "stage1_top_k",
            "stage1_min_score_threshold",
            "stage1_neighbors",
            "stage2_neighbors",
            "stage2_aggregation_strategy",
            "calibration_method",
            "acceptance_threshold",
            "rejection_policy",
        )
        self.assertEqual(knn_cfg.open_parameters(), expected_open)
        for name in expected_open:
            self.assertIsNone(getattr(knn_cfg, name), f"{name} must remain unset until validation")

    def test_d5_four_scoring_weights(self) -> None:
        self.assertEqual(len(DEFAULT_SCORING_WEIGHT_NAMES), 4)
        self.assertEqual(
            DEFAULT_SCORING_WEIGHT_NAMES,
            ("w1_gain", "w2_interest", "w3_fit", "w4_urgency"),
        )

    def test_module3_config_defaults(self) -> None:
        cfg = Module3Config()
        self.assertEqual(cfg.embedding_dimension, 384)
        self.assertEqual(cfg.excluded_parent_groups, ("P",))


if __name__ == "__main__":
    unittest.main()
