from __future__ import annotations

import csv
from pathlib import Path
import unittest

import yaml

from scripts.evaluate_dinov3_regions import RegionRecord, filter_regions, read_region_table
from scripts.train_dinov3_region_head import filter_training_regions


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REGION_TABLE = REPO_ROOT / "outputs/cvat_region_analysis/manual_all/region_annotations.csv"
LEGACY_REGION_TABLE = REPO_ROOT / "outputs/region_analysis/cvat_region_analysis/region_annotations.csv"
SPECIAL_LABEL = "Nicht_bewertbar"


def load_config(relative_path: str) -> dict:
    with (REPO_ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_record(
    region_id: str,
    *,
    split: str,
    label: str,
    matched: bool = True,
    is_global_class: bool = True,
) -> RegionRecord:
    return RegionRecord(
        region_id=region_id,
        source_image=f"{region_id}.jpg",
        split=split,
        original_label=label,
        mapped_label=label,
        is_global_class=is_global_class,
        x_min=0.0,
        y_min=0.0,
        x_max=10.0,
        y_max=10.0,
        matched_manifest=matched,
        raw={},
    )


class RegionSplitGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.eval_config = load_config("configs/experiments/dinov3_region_eval.yaml")
        self.four_class_config = load_config("configs/experiments/dinov3_region_head.yaml")
        self.five_class_config = load_config("configs/experiments/dinov3_region_head_5class.yaml")

    def test_configs_use_canonical_table_and_exclude_test(self) -> None:
        expected_table = "outputs/cvat_region_analysis/manual_all/region_annotations.csv"
        for config in (self.eval_config, self.four_class_config, self.five_class_config):
            self.assertEqual(config["inputs"]["region_annotations"], expected_table)

        evaluation = self.eval_config["evaluation"]
        self.assertEqual(evaluation["default_split"], "val")
        self.assertEqual(set(evaluation["allowed_splits"]), {"train", "val"})
        self.assertEqual(evaluation["test_policy"], "excluded")

        for config in (self.four_class_config, self.five_class_config):
            self.assertEqual(config["data"]["train_split"], "train")
            self.assertEqual(config["data"]["val_split"], "val")
            self.assertEqual(config["data"]["test_policy"], "excluded")

        self.assertFalse(self.four_class_config["data"]["include_nicht_bewertbar"])
        self.assertTrue(self.five_class_config["data"]["include_nicht_bewertbar"])

    def test_filter_functions_exclude_test_and_unmatched(self) -> None:
        global_label = self.four_class_config["class_order"][0]
        rows = [
            make_record("train_global", split="train", label=global_label),
            make_record(
                "train_special",
                split="train",
                label=SPECIAL_LABEL,
                is_global_class=False,
            ),
            make_record("train_unmatched", split="train", label=global_label, matched=False),
            make_record("val_global", split="val", label=global_label),
            make_record(
                "val_special",
                split="val",
                label=SPECIAL_LABEL,
                is_global_class=False,
            ),
            make_record("val_unmatched", split="val", label=global_label, matched=False),
            make_record("test_global", split="test", label=global_label),
        ]
        mapping4 = {label: index for index, label in enumerate(self.four_class_config["class_order"])}
        mapping5 = {label: index for index, label in enumerate(self.five_class_config["class_order"])}

        selected4, ignored4 = filter_training_regions(
            rows,
            "train",
            mapping4,
            SPECIAL_LABEL,
            include_special_label=False,
        )
        self.assertEqual([row.region_id for row in selected4], ["train_global"])
        self.assertEqual([row.region_id for row in ignored4], ["train_special"])

        selected5, ignored5 = filter_training_regions(
            rows,
            "train",
            mapping5,
            SPECIAL_LABEL,
            include_special_label=True,
        )
        self.assertEqual(
            [row.region_id for row in selected5],
            ["train_global", "train_special"],
        )
        self.assertEqual(ignored5, [])
        self.assertTrue(all(row.split == "train" and row.matched_manifest for row in selected5))

        direct_val, summary = filter_regions(
            rows,
            split="val",
            special_label=SPECIAL_LABEL,
            include_special=False,
            max_regions=None,
        )
        self.assertEqual([row.region_id for row in direct_val], ["val_global"])
        self.assertEqual(summary["nicht_bewertbar_regions_in_split"], 1)
        self.assertTrue(summary["test_excluded"])
        self.assertTrue(summary["matched_required"])

    def test_local_canonical_table_has_expected_selections(self) -> None:
        if not CANONICAL_REGION_TABLE.exists():
            self.skipTest("Proprietary canonical region table is not available locally")

        rows = read_region_table(CANONICAL_REGION_TABLE)
        self.assertEqual(len(rows), 281)

        selections: dict[str, tuple[list[RegionRecord], list[RegionRecord]]] = {}
        for name, config in (
            ("four", self.four_class_config),
            ("five", self.five_class_config),
        ):
            mapping = {label: index for index, label in enumerate(config["class_order"])}
            include_special = bool(config["data"]["include_nicht_bewertbar"])
            train, _ = filter_training_regions(
                rows,
                "train",
                mapping,
                SPECIAL_LABEL,
                include_special_label=include_special,
            )
            val, _ = filter_training_regions(
                rows,
                "val",
                mapping,
                SPECIAL_LABEL,
                include_special_label=include_special,
            )
            selections[name] = (train, val)

        four_train, four_val = selections["four"]
        five_train, five_val = selections["five"]
        self.assertEqual((len(four_train), len(four_val)), (147, 35))
        self.assertEqual((len(five_train), len(five_val)), (181, 42))
        self.assertFalse(any(row.mapped_label == SPECIAL_LABEL for row in four_train + four_val))
        self.assertEqual(
            sum(row.mapped_label == SPECIAL_LABEL for row in five_train + five_val),
            41,
        )
        for selected in (four_train, four_val, five_train, five_val):
            self.assertTrue(all(row.split in {"train", "val"} for row in selected))
            self.assertTrue(all(row.matched_manifest for row in selected))

        direct_val, summary = filter_regions(
            rows,
            split="val",
            special_label=SPECIAL_LABEL,
            include_special=False,
            max_regions=None,
        )
        self.assertEqual(len(direct_val), 35)
        self.assertEqual(summary["nicht_bewertbar_regions_in_split"], 7)
        self.assertEqual(summary["test_rows_in_region_table"], 58)
        self.assertTrue(summary["test_excluded"])

    def test_canonical_train_val_rows_match_legacy_snapshot(self) -> None:
        if not CANONICAL_REGION_TABLE.exists() or not LEGACY_REGION_TABLE.exists():
            self.skipTest("Local canonical and legacy region tables are both required")

        def read_rows(path: Path) -> list[dict[str, str]]:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))

        def stable_key(row: dict[str, str]) -> tuple[str, ...]:
            return tuple(
                row[field]
                for field in (
                    "original_image_name",
                    "region_id",
                    "split",
                    "mapped_label",
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                )
            )

        legacy_rows = read_rows(LEGACY_REGION_TABLE)
        canonical_rows = [
            row
            for row in read_rows(CANONICAL_REGION_TABLE)
            if row["split"] in {"train", "val"} and row["matched_manifest"].lower() == "true"
        ]
        legacy_by_key = {stable_key(row): row for row in legacy_rows}
        canonical_by_key = {stable_key(row): row for row in canonical_rows}

        self.assertEqual(len(legacy_rows), 223)
        self.assertEqual(len(canonical_rows), 223)
        self.assertEqual(len(legacy_by_key), len(legacy_rows), "Duplicate legacy region key")
        self.assertEqual(len(canonical_by_key), len(canonical_rows), "Duplicate canonical region key")
        self.assertEqual(legacy_by_key, canonical_by_key)


if __name__ == "__main__":
    unittest.main()
