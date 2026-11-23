"""Phase 1 data validation tests: events.json fixtures, hierarchical CSV splits, and rubric.yaml."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from configs import DEMO_EVENTS_FULL_CSV_PATH, RAW_EVENTS_JSON_PATH, RUBRIC_YAML_PATH
from scripts.prepare_cls_dataset import build_hierarchical_dataset, load_eval_titles, token_jaccard
from scripts.prepare_test_fixtures import convert_events_file, parse_vietnamese_fee
from src.contract_validator import validate_event_normalized, validate_rubric_dict
from src.schemas import parent_group_from_item_id


def _load_rubric_dict(path: Path) -> dict:
    """Load rubric.yaml via PyYAML when installed, or Ruby's stdlib YAML in minimal macOS envs."""
    try:
        import yaml  # type: ignore

        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except ImportError:
        proc = subprocess.run(
            [
                "ruby",
                "-ryaml",
                "-rjson",
                "-e",
                f"puts JSON.generate(YAML.load_file({str(path)!r}))",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(proc.stdout)


class TestEventFixtureAdapter(unittest.TestCase):
    def test_vietnamese_fee_parser(self) -> None:
        self.assertEqual(parse_vietnamese_fee("miễn phí"), (0, None))
        self.assertEqual(parse_vietnamese_fee("Hoàn toàn miễn phí"), (0, None))
        self.assertEqual(parse_vietnamese_fee("25.000 VNĐ/thí sinh"), (25000, None))
        self.assertEqual(parse_vietnamese_fee("20.000đ/thí sinh"), (20000, None))
        self.assertEqual(parse_vietnamese_fee("140k /đội"), (140000, None))
        self.assertEqual(parse_vietnamese_fee(None), (None, "fee_missing"))
        self.assertEqual(
            parse_vietnamese_fee("Hiểu rõ “Chi phí cơ hội” trong lĩnh vực môi trường"),
            (None, "fee_unparseable"),
        )

    def test_all_108_legacy_events_convert_cleanly(self) -> None:
        events, summary = convert_events_file(input_path=RAW_EVENTS_JSON_PATH, output_path=None)
        self.assertEqual(len(events), 108)
        self.assertEqual(summary["invalid_events"], 0)
        self.assertEqual(summary["total_warnings"], 0)
        self.assertEqual(summary["missing_field_flag_counts"]["fee_unparseable"], 2)
        self.assertEqual(summary["missing_field_flag_counts"]["reg_end_after_start"], 16)
        for ev in events:
            report = validate_event_normalized(ev)
            self.assertTrue(report.is_valid, f"Event {ev.event_id} failed: {report.errors}")


class TestHierarchicalClsDataset(unittest.TestCase):
    def test_demo_csv_builds_clean_partitioned_dataset_with_zero_leakage(self) -> None:
        dataset = build_hierarchical_dataset(
            csv_path=DEMO_EVENTS_FULL_CSV_PATH,
            events_json_path=RAW_EVENTS_JSON_PATH,
            excluded_parent_groups=("P",),
            dataset_stage="demo",
            exclude_overlapping_eval_titles=True,
            output_path=None,
        )
        summary = dataset["summary"]
        self.assertEqual(summary["invalid_rows"], 0)
        self.assertEqual(summary["conflicting_normalized_texts"], 0)
        self.assertGreater(summary["accepted_samples"], 100)
        self.assertEqual(summary["excluded_by_parent_group"], 1)  # row 6.1 in group P

        stage2 = dataset["stage2_by_parent_group"]
        self.assertEqual(set(stage2.keys()), {"C1", "C2", "C3", "C4", "C5"})

        eval_titles = load_eval_titles(RAW_EVENTS_JSON_PATH)
        for parent_grp, samples in stage2.items():
            for sample in samples:
                self.assertEqual(sample["parent_group_id"], parent_grp)
                self.assertEqual(parent_group_from_item_id(sample["rubric_item_id"]), parent_grp)
                for et in eval_titles:
                    self.assertLess(
                        token_jaccard(sample["title"], et),
                        0.5,
                        f"Leakage detected: {sample['title']} vs {et}",
                    )

    def test_include_all_groups_includes_group_p(self) -> None:
        dataset = build_hierarchical_dataset(
            csv_path=DEMO_EVENTS_FULL_CSV_PATH,
            events_json_path=RAW_EVENTS_JSON_PATH,
            excluded_parent_groups=(),
            dataset_stage="demo",
            exclude_overlapping_eval_titles=True,
            output_path=None,
        )
        self.assertIn("P", dataset["stage2_by_parent_group"])
        self.assertEqual(len(dataset["stage2_by_parent_group"]["P"]), 1)


class TestRubricTaxonomyIntegrity(unittest.TestCase):
    def test_rubric_yaml_51_items_zero_errors(self) -> None:
        rubric = _load_rubric_dict(RUBRIC_YAML_PATH)
        report, hierarchy = validate_rubric_dict(rubric)
        self.assertTrue(report.is_valid, f"Rubric validation errors: {report.errors}")
        total_items = sum(len(refs) for refs in hierarchy.values())
        self.assertEqual(total_items, 51)
        self.assertEqual(len(hierarchy["C1"]), 9)
        self.assertEqual(len(hierarchy["C2"]), 3)
        self.assertEqual(len(hierarchy["C3"]), 14)
        self.assertEqual(len(hierarchy["C4"]), 8)
        self.assertEqual(len(hierarchy["C5"]), 11)
        self.assertEqual(len(hierarchy["P"]), 6)


if __name__ == "__main__":
    unittest.main()
