from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from PIL import Image

from bachelorarbeit.training import global_training_setup
from bachelorarbeit.training.global_training_setup import DEVELOPMENT_SPLITS
from scripts.train_yolov11_cls import assert_no_materialized_test_split, selected_yolo_splits


class TestsetGuardTest(unittest.TestCase):
    def test_training_and_yolo_materialization_exclude_test_by_default(self) -> None:
        self.assertEqual(DEVELOPMENT_SPLITS, ("train", "val"))
        self.assertEqual(selected_yolo_splits(include_final_test=False), ("train", "val"))
        self.assertEqual(selected_yolo_splits(include_final_test=True), ("train", "val", "test"))

    def test_existing_materialized_test_split_blocks_training_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            test_file = dataset_dir / "test" / "class_a" / "sample.jpg"
            test_file.parent.mkdir(parents=True)
            test_file.write_bytes(b"test-placeholder")
            with self.assertRaises(RuntimeError):
                assert_no_materialized_test_split(dataset_dir)

    def test_prepare_run_does_not_check_or_load_test_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "dataset"
            for name in ("train.png", "val.png"):
                path = dataset_root / "class_a" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color="white").save(path)

            manifest_path = root / "manifest.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["image_id", "relative_path", "label", "split", "group_id"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"image_id": "train", "relative_path": "class_a/train.png", "label": "class_a", "split": "train", "group_id": "train"},
                        {"image_id": "val", "relative_path": "class_a/val.png", "label": "class_a", "split": "val", "group_id": "val"},
                        {"image_id": "test", "relative_path": "class_a/missing-test.png", "label": "class_a", "split": "test", "group_id": "test"},
                    ]
                )

            config_path = root / "config.yaml"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "guard-test",
                        "model_family": "guard_family",
                        "model_variant": "none",
                        "split_manifest": "manifest.csv",
                        "output_dir": "outputs/guard-test",
                        "image_size": [8, 8],
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(global_training_setup, "REPO_ROOT", root):
                prepared = global_training_setup.prepare_run(
                    config_path=config_path,
                    dataset_root=dataset_root,
                    expected_model_family="guard_family",
                    dry_run=True,
                    smoke_test=True,
                    max_smoke_samples=1,
                )

            self.assertEqual(prepared.metadata["local_file_check_train_val_only"]["checked"], 2)
            self.assertEqual(set(prepared.metadata["smoke_records"]), {"train", "val"})
