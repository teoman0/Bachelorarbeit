from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from scripts.run_final_test_evaluation import evaluate_predictions, labels_from_mapping, write_confusion


class LabelOrderTest(unittest.TestCase):
    def test_prediction_metrics_and_confusion_use_checkpoint_order(self) -> None:
        class_to_index = {"second": 1, "first": 0, "third": 2}
        self.assertEqual(labels_from_mapping(class_to_index), ["first", "second", "third"])

        labels, _metrics, per_class, confusion = evaluate_predictions(
            y_true=[0, 1, 2, 2],
            y_pred=[0, 2, 2, 1],
            class_to_index=class_to_index,
        )
        self.assertEqual(labels, ["first", "second", "third"])
        self.assertEqual([row["class_name"] for row in per_class], labels)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "confusion.csv"
            write_confusion(output_path, labels, confusion)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["true_label", *labels])
        self.assertEqual([row[0] for row in rows[1:]], labels)
