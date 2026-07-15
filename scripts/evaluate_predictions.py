"""Evaluate classification predictions from a CSV file.

This script loads prediction tables only. It does not load images, initialize
models, download weights, create checkpoints, or start training.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {"true_label", "predicted_label"}


def parse_csv_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
    except Exception:
        return None
    return result.stdout.strip()


def package_versions() -> dict[str, str | None]:
    packages = ["pandas"]
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def load_predictions(path: Path, split_name: str | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Prediction path is not a file: {path}")

    predictions = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {missing}")

    if split_name is not None:
        if "split" not in predictions.columns:
            raise ValueError("--split was provided, but the CSV has no split column")
        predictions = predictions[predictions["split"].astype(str) == split_name].copy()

    if predictions.empty:
        raise ValueError("No prediction rows remain after loading and optional split filtering")

    predictions["true_label"] = predictions["true_label"].astype(str)
    predictions["predicted_label"] = predictions["predicted_label"].astype(str)
    return predictions


def infer_labels(
    predictions: pd.DataFrame,
    class_names: list[str] | None,
    ordinal_class_order: list[str] | None,
) -> list[str]:
    observed = set(predictions["true_label"]) | set(predictions["predicted_label"])
    if class_names is not None:
        labels = list(class_names)
    elif ordinal_class_order is not None:
        labels = list(ordinal_class_order)
    else:
        labels = sorted(observed)

    missing_observed = sorted(observed - set(labels))
    if missing_observed:
        labels.extend(missing_observed)
    return labels


def compute_metrics(predictions: pd.DataFrame, labels: list[str]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y_true = predictions["true_label"].tolist()
    y_pred = predictions["predicted_label"].tolist()
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true_label, predicted_label in zip(y_true, y_pred, strict=True):
        matrix[label_to_index[true_label]][label_to_index[predicted_label]] += 1

    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    support: list[int] = []
    true_positives: list[int] = []
    for index, _label in enumerate(labels):
        tp = matrix[index][index]
        predicted_count = sum(row[index] for row in matrix)
        true_count = sum(matrix[index])
        class_precision = tp / predicted_count if predicted_count else 0.0
        class_recall = tp / true_count if true_count else 0.0
        class_f1 = (
            2.0 * class_precision * class_recall / (class_precision + class_recall)
            if class_precision + class_recall
            else 0.0
        )
        precision.append(class_precision)
        recall.append(class_recall)
        f1.append(class_f1)
        support.append(true_count)
        true_positives.append(tp)

    per_class = pd.DataFrame(
        {
            "class_name": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    confusion = pd.DataFrame(matrix, index=labels, columns=labels)
    confusion.index.name = "true_label"

    total = len(predictions)
    recalls_with_support = [
        class_recall for class_recall, true_count in zip(recall, support, strict=True) if true_count > 0
    ]
    overall = {
        "num_rows": int(total),
        "accuracy": float(sum(true_positives) / total),
        "balanced_accuracy": float(sum(recalls_with_support) / len(recalls_with_support)),
        "macro_f1": float(sum(f1) / len(f1)),
    }
    return overall, per_class, confusion


def compute_ordinal_errors(predictions: pd.DataFrame, class_order: list[str] | None) -> dict[str, Any] | None:
    if class_order is None:
        return None

    observed = set(predictions["true_label"]) | set(predictions["predicted_label"])
    missing = sorted(observed - set(class_order))
    if missing:
        raise ValueError(f"Ordinal class order does not include observed labels: {missing}")

    positions = {label: index for index, label in enumerate(class_order)}
    distances = [
        abs(positions[true_label] - positions[predicted_label])
        for true_label, predicted_label in zip(
            predictions["true_label"],
            predictions["predicted_label"],
            strict=True,
        )
    ]
    return {
        "enabled": True,
        "class_order": class_order,
        "mean_absolute_class_distance": float(sum(distances) / len(distances)),
        "max_absolute_class_distance": int(max(distances)),
    }


def metrics_summary_rows(overall: dict[str, Any], ordinal: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = [
        {"metric": "num_rows", "value": overall["num_rows"]},
        {"metric": "accuracy", "value": overall["accuracy"]},
        {"metric": "balanced_accuracy", "value": overall["balanced_accuracy"]},
        {"metric": "macro_f1", "value": overall["macro_f1"]},
    ]
    if ordinal is not None:
        rows.extend(
            [
                {
                    "metric": "mean_absolute_class_distance",
                    "value": ordinal["mean_absolute_class_distance"],
                },
                {
                    "metric": "max_absolute_class_distance",
                    "value": ordinal["max_absolute_class_distance"],
                },
            ]
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Path]:
    prediction_path = Path(args.predictions).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = parse_csv_list(args.class_names)
    ordinal_class_order = parse_csv_list(args.ordinal_class_order)
    predictions = load_predictions(prediction_path, split_name=args.split)
    labels = infer_labels(
        predictions=predictions,
        class_names=class_names,
        ordinal_class_order=ordinal_class_order,
    )
    overall, per_class, confusion = compute_metrics(predictions, labels)
    ordinal = compute_ordinal_errors(predictions, ordinal_class_order)

    payload = {
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "script": "scripts/evaluate_predictions.py",
        "method_note": "Prediction-table evaluation only; no images, models, weights, checkpoints, or training are loaded.",
        "prediction_file": str(prediction_path),
        "split_filter": args.split,
        "labels": labels,
        "metadata": {
            "model_name": args.model_name,
            "config_id": args.config_id,
            "split_version": args.split_version,
            "seed": args.seed,
            "git_commit": git_commit(),
            "package_versions": package_versions(),
        },
        "overall_metrics": overall,
        "per_class_metrics": per_class.to_dict(orient="records"),
        "confusion_matrix": {
            "labels": labels,
            "rows_are_true_labels": True,
            "values": confusion.reset_index().to_dict(orient="records"),
        },
        "ordinal_evaluation": ordinal or {"enabled": False},
    }

    metrics_json = output_dir / "evaluation_metrics.json"
    metrics_csv = output_dir / "evaluation_metrics.csv"
    per_class_csv = output_dir / "per_class_metrics.csv"
    confusion_csv = output_dir / "confusion_matrix.csv"

    metrics_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(metrics_summary_rows(overall, ordinal)).to_csv(metrics_csv, index=False)
    per_class.to_csv(per_class_csv, index=False)
    confusion.to_csv(confusion_csv)

    return {
        "metrics_json": metrics_json,
        "metrics_csv": metrics_csv,
        "per_class_csv": per_class_csv,
        "confusion_csv": confusion_csv,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        help="CSV file with at least true_label and predicted_label columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
        help="Directory for JSON and CSV evaluation outputs.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split value to evaluate, for example test or val.",
    )
    parser.add_argument(
        "--class-names",
        default=None,
        help="Comma-separated class names in the desired metric/confusion-matrix order.",
    )
    parser.add_argument(
        "--ordinal-class-order",
        default=None,
        help="Comma-separated class order for optional ordinal error metrics.",
    )
    parser.add_argument("--model-name", default=None, help="Optional model name stored in the JSON metadata.")
    parser.add_argument("--config-id", default=None, help="Optional config ID stored in the JSON metadata.")
    parser.add_argument("--split-version", default=None, help="Optional split version stored in the JSON metadata.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed stored in the JSON metadata.")

    args = parser.parse_args()
    try:
        outputs = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
