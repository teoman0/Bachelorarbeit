"""Run the locked final test evaluation for the selected BMW-25 models.

The script performs inference only. It never trains, creates splits, or selects
models. Test inference is blocked unless --allow-final-test is supplied. The
separate --check-preprocessing mode compares input tensors without loading a
model, producing predictions, or writing artifacts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from bachelorarbeit.data.split_dataset import (  # noqa: E402
    ManifestRecord,
    filter_split,
    prepare_dinov3_image,
    read_split_manifest,
    resolve_record_path,
)
from bachelorarbeit.training.global_training_setup import (  # noqa: E402
    git_commit,
    load_yaml_config,
    package_versions,
)
from scripts.evaluate_dinov3_regions import (  # noqa: E402
    RegionRecord,
    load_region_crop,
    read_region_table,
    top_confusions,
)
from scripts.train_deit_tiny import ManifestTensorDataset  # noqa: E402
from scripts.train_dinov3_head import extract_features, load_dinov3, metric_payload  # noqa: E402


FINAL_SPLIT = "test"
SPECIAL_REGION_LABEL = "Nicht_bewertbar"
GLOBAL_MODEL_KEYS = (
    "yolo11n_cls",
    "dinov3_frozen_linear_head",
    "deit_tiny_scratch",
    "dinov3_partial_finetune_last2",
)
REGION_MODEL_KEYS = (
    "dinov3_region_head_4class",
    "dinov3_region_head_5class",
)
ALL_MODEL_KEYS = GLOBAL_MODEL_KEYS + REGION_MODEL_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/final_test_evaluation.yaml")
    parser.add_argument("--dataset-root", help="Local global-image dataset root; never recorded in outputs.")
    parser.add_argument("--manual-root", help="Local CVAT manual_all root; never recorded in outputs.")
    parser.add_argument(
        "--allow-final-test",
        action="store_true",
        help="Explicitly authorize the locked final test inference workflow.",
    )
    parser.add_argument(
        "--check-preprocessing",
        action="store_true",
        help="Compare legacy and training-time DINOv3 input tensors without model inference or output files.",
    )
    parser.add_argument("--max-preprocessing-images", type=int, default=None)
    parser.add_argument("--global-only", action="store_true")
    parser.add_argument("--regions-only", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODEL_KEYS,
        help="Evaluate only the listed locked model keys from the config.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    if args.allow_final_test and args.check_preprocessing:
        parser.error("--allow-final-test and --check-preprocessing are mutually exclusive")
    if args.global_only and args.regions_only:
        parser.error("--global-only and --regions-only are mutually exclusive")
    if args.max_preprocessing_images is not None and args.max_preprocessing_images < 1:
        parser.error("--max-preprocessing-images must be at least 1")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def require_final_test_access(*, allow_final_test: bool, check_preprocessing: bool) -> None:
    """Reject every mode that could touch test images without explicit intent."""

    if allow_final_test or check_preprocessing:
        return
    raise PermissionError(
        "Test access denied. Use --allow-final-test for the locked final inference "
        "or --check-preprocessing for the tensor-only technical audit."
    )


def repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def relative_to_repo(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Versioned metadata path must stay inside the repository: {path}") from exc


def ensure_local_directory(raw: str | None, description: str) -> Path:
    if not raw:
        raise ValueError(f"{description} is required for this mode")
    path = Path(raw).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")
    return path


def ensure_output_root(path: Path) -> Path:
    output_root = path.resolve()
    allowed_root = (REPO_ROOT / "outputs").resolve()
    if output_root != allowed_root and allowed_root not in output_root.parents:
        raise ValueError(f"Final evaluation artifacts must stay below outputs/: {path}")
    return output_root


def labels_from_mapping(class_to_index: dict[str, int]) -> list[str]:
    expected = list(range(len(class_to_index)))
    actual = sorted(class_to_index.values())
    if actual != expected:
        raise ValueError(f"Class indices must be contiguous from zero: {class_to_index}")
    return [label for label, _index in sorted(class_to_index.items(), key=lambda item: item[1])]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_confusion(path: Path, labels: list[str], matrix: list[list[int]]) -> None:
    if len(matrix) != len(labels) or any(len(row) != len(labels) for row in matrix):
        raise ValueError("Confusion-matrix dimensions do not match label order")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label"] + labels)
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label] + row)


def resolve_device(raw_device: str) -> Any:
    import torch

    if torch.cuda.is_available() and raw_device != "cpu":
        return torch.device(f"cuda:{raw_device}")
    return torch.device("cpu")


def hardware_metadata() -> dict[str, Any]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def configured_checkpoints(
    config: dict[str, Any],
    selected_models: set[str] | None = None,
) -> dict[str, str]:
    checkpoints: dict[str, str] = {}
    for section in ("global_models", "region_models"):
        for key, model_config in config.get(section, {}).items():
            if bool(model_config.get("enabled", True)) and (
                selected_models is None or key in selected_models
            ):
                checkpoints[key] = str(model_config["checkpoint"])
    return checkpoints


def provenance_metadata(
    config: dict[str, Any],
    config_path: Path,
    *,
    mode: str,
    device: str,
    batch_size: int,
    selected_models: set[str],
) -> dict[str, Any]:
    return {
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "script": "scripts/run_final_test_evaluation.py",
        "mode": mode,
        "experiment_name": config.get("experiment_name"),
        "config_path": relative_to_repo(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "git_commit": git_commit(),
        "seed": config.get("seed"),
        "split": FINAL_SPLIT,
        "split_manifest": str(config.get("split_manifest")),
        "device_requested": device,
        "batch_size": batch_size,
        "hardware": hardware_metadata(),
        "package_versions": package_versions(["ultralytics", "timm", "transformers"]),
        "selected_models": sorted(selected_models),
        "checkpoints": configured_checkpoints(config, selected_models),
        "dataset_root_recorded": False,
        "manual_root_recorded": False,
        "training_performed": False,
        "new_split_created": False,
        "model_selection_performed": False,
        "hyperparameter_tuning_performed": False,
        "corrected_evaluation_note": "corrected evaluation due to preprocessing inconsistency",
        "decision_note": "no model selection or hyperparameter tuning performed",
        "test_usage_note": (
            "The test split is used only for the locked final evaluation after model "
            "selection; it is not used for training or further decisions."
        ),
    }


def final_test_records(manifest: Path) -> list[ManifestRecord]:
    records = filter_split(read_split_manifest(manifest), FINAL_SPLIT)
    if not records:
        raise ValueError("No test records found in the grouped split manifest")
    return records


def select_model_keys(config: dict[str, Any], args: argparse.Namespace) -> set[str]:
    enabled = {
        key
        for section in ("global_models", "region_models")
        for key, model_config in config.get(section, {}).items()
        if bool(model_config.get("enabled", True))
    }
    requested = set(args.models) if args.models else enabled
    disabled = requested - enabled
    if disabled:
        raise ValueError(f"Requested models are disabled or missing from config: {sorted(disabled)}")
    if args.global_only:
        requested &= set(GLOBAL_MODEL_KEYS)
    if args.regions_only:
        requested &= set(REGION_MODEL_KEYS)
    if not requested:
        raise ValueError("Model selection is empty after applying scope filters")
    return requested


def evaluate_predictions(
    y_true: list[int],
    y_pred: list[int],
    class_to_index: dict[str, int],
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]], list[list[int]]]:
    labels = labels_from_mapping(class_to_index)
    index_to_class = {index: label for label, index in class_to_index.items()}
    metrics, per_class, confusion = metric_payload(y_true, y_pred, index_to_class)
    if [row["class_name"] for row in per_class] != labels:
        raise ValueError("Per-class metric order does not match checkpoint label order")
    return labels, metrics, per_class, confusion


def write_metrics_files(
    *,
    output_dir: Path,
    labels: list[str],
    metrics: dict[str, Any],
    per_class: list[dict[str, Any]],
    confusion: list[list[int]],
    prediction_file: Path,
    checkpoint: Path,
    model_name: str,
    provenance: dict[str, Any],
    preprocessing_description: str,
) -> dict[str, str]:
    payload = {
        "evaluation_provenance": provenance,
        "model_name": model_name,
        "checkpoint": relative_to_repo(checkpoint),
        "prediction_file": relative_to_repo(prediction_file),
        "labels": labels,
        "overall_metrics": metrics,
        "per_class_metrics": per_class,
        "confusion_matrix": {
            "labels": labels,
            "rows_are_true_labels": True,
            "values": confusion,
        },
        "top_confusions": top_confusions(labels, confusion),
    }
    metrics_json = output_dir / "test_metrics.json"
    metrics_csv = output_dir / "test_metrics.csv"
    per_class_csv = output_dir / "per_class_metrics_test.csv"
    confusion_csv = output_dir / "confusion_matrix_test.csv"
    run_metadata_json = output_dir / "run_metadata.json"
    write_json(metrics_json, payload)
    rows = [
        {"scope": "overall", "class_name": "", "metric": "num_rows", "value": metrics["num_rows"]},
        {"scope": "overall", "class_name": "", "metric": "accuracy", "value": metrics["accuracy"]},
        {
            "scope": "overall",
            "class_name": "",
            "metric": "balanced_accuracy",
            "value": metrics["balanced_accuracy"],
        },
        {"scope": "overall", "class_name": "", "metric": "macro_f1", "value": metrics["macro_f1"]},
    ]
    for row in per_class:
        for key in ("precision", "recall", "f1", "support"):
            rows.append({"scope": "per_class", "class_name": row["class_name"], "metric": key, "value": row[key]})
    write_csv(metrics_csv, rows, ["scope", "class_name", "metric", "value"])
    write_csv(
        per_class_csv,
        per_class,
        ["class_name", "precision", "recall", "f1", "support"],
    )
    write_confusion(confusion_csv, labels, confusion)
    write_json(
        run_metadata_json,
        {
            "evaluation_provenance": provenance,
            "model_name": model_name,
            "checkpoint": {
                "path": relative_to_repo(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
            },
            "seed": provenance["seed"],
            "class_order": labels,
            "num_test_items": metrics["num_rows"],
            "preprocessing": preprocessing_description,
            "started_at_utc": provenance["generated_at_utc"],
            "ended_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "note": "corrected evaluation due to preprocessing inconsistency",
            "decision_note": "no model selection or hyperparameter tuning performed",
        },
    )
    return {
        "metrics_json": relative_to_repo(metrics_json),
        "metrics_csv": relative_to_repo(metrics_csv),
        "per_class_metrics_csv": relative_to_repo(per_class_csv),
        "confusion_matrix": relative_to_repo(confusion_csv),
        "run_metadata": relative_to_repo(run_metadata_json),
    }


def result_summary(
    *,
    scope: str,
    model_name: str,
    output_dir: Path,
    checkpoint: Path,
    predictions: Path,
    labels: list[str],
    metrics: dict[str, Any],
    per_class: list[dict[str, Any]],
    confusion: list[list[int]],
    files: dict[str, str],
) -> dict[str, Any]:
    return {
        "scope": scope,
        "model_name": model_name,
        "output_dir": relative_to_repo(output_dir),
        "checkpoint": relative_to_repo(checkpoint),
        "predictions": relative_to_repo(predictions),
        "num_items": metrics["num_rows"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "labels": labels,
        "per_class_metrics": per_class,
        "confusion_matrix": confusion,
        "top_confusions": top_confusions(labels, confusion),
        "files": files,
    }


def evaluate_yolo(
    *,
    records: list[ManifestRecord],
    dataset_root: Path,
    checkpoint: Path,
    output_dir: Path,
    model_config: dict[str, Any],
    device: str,
    batch_size: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    names = model.names
    labels = (
        [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
        if isinstance(names, dict)
        else [str(name) for name in names]
    )
    class_to_index = {label: index for index, label in enumerate(labels)}
    manifest_labels = {record.label for record in records}
    if set(labels) != manifest_labels:
        raise ValueError(f"YOLO checkpoint labels do not match test manifest: {labels} vs {sorted(manifest_labels)}")

    prediction_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        paths = [resolve_record_path(dataset_root, record) for record in batch]
        results = model.predict(
            source=[str(path) for path in paths],
            imgsz=int(model_config.get("image_size", 320)),
            batch=batch_size,
            device=device,
            save=False,
            verbose=False,
        )
        for record, result in zip(batch, results, strict=True):
            probabilities = result.probs.data.detach().cpu().tolist()
            pred_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            true_index = class_to_index[record.label]
            y_true.append(true_index)
            y_pred.append(pred_index)
            row = {
                "image_id": record.image_id,
                "relative_path": record.relative_path,
                "split": record.split,
                "true_label": record.label,
                "predicted_label": labels[pred_index],
                "model_name": model_config["model_name"],
                "config_id": model_config["config_id"],
                "seed": provenance["seed"],
                "confidence": float(probabilities[pred_index]),
            }
            for label, value in zip(labels, probabilities, strict=True):
                row[f"prob_{label}"] = float(value)
            prediction_rows.append(row)

    prediction_path = output_dir / "predictions_test.csv"
    write_csv(
        prediction_path,
        prediction_rows,
        [
            "image_id",
            "relative_path",
            "split",
            "true_label",
            "predicted_label",
            "model_name",
            "config_id",
            "seed",
            "confidence",
        ]
        + [f"prob_{label}" for label in labels],
    )
    labels, metrics, per_class, confusion = evaluate_predictions(y_true, y_pred, class_to_index)
    files = write_metrics_files(
        output_dir=output_dir,
        labels=labels,
        metrics=metrics,
        per_class=per_class,
        confusion=confusion,
        prediction_file=prediction_path,
        checkpoint=checkpoint,
        model_name=str(model_config["model_name"]),
        provenance=provenance,
        preprocessing_description="Ultralytics classification preprocessing at the configured image size",
    )
    return result_summary(
        scope="global",
        model_name=str(model_config["model_name"]),
        output_dir=output_dir,
        checkpoint=checkpoint,
        predictions=prediction_path,
        labels=labels,
        metrics=metrics,
        per_class=per_class,
        confusion=confusion,
        files=files,
    )


def load_dino_head_checkpoint(
    checkpoint: Path,
    device: Any,
) -> tuple[Any, Any, Any, dict[str, Any], dict[str, int]]:
    import torch
    import torch.nn as nn

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_to_index = {str(label): int(index) for label, index in payload["class_to_index"].items()}
    feature_dim = int(payload.get("feature_info", {}).get("feature_dim", payload.get("feature_dim", 768)))
    processor, model = load_dinov3(str(payload["model_id"]), allow_model_download=False, device=device)
    if "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    head = nn.Linear(feature_dim, len(class_to_index)).to(device)
    head.load_state_dict(payload["head_state_dict"], strict=True)
    model.eval()
    head.eval()
    return processor, model, head, payload, class_to_index


def prepare_dino_global_images(
    records: list[ManifestRecord],
    dataset_root: Path,
    image_size: tuple[int, int],
) -> list[Any]:
    from PIL import Image

    images = []
    for record in records:
        with Image.open(resolve_record_path(dataset_root, record)) as image:
            images.append(prepare_dinov3_image(image, image_size))
    return images


def evaluate_dino_global(
    *,
    records: list[ManifestRecord],
    dataset_root: Path,
    checkpoint: Path,
    output_dir: Path,
    model_config: dict[str, Any],
    image_size: tuple[int, int],
    device: Any,
    batch_size: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    import torch

    processor, model, head, checkpoint_payload, class_to_index = load_dino_head_checkpoint(checkpoint, device)
    labels = labels_from_mapping(class_to_index)
    prediction_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    representation = str(
        checkpoint_payload.get("feature_info", {}).get(
            "feature_representation_used",
            checkpoint_payload.get("config", {}).get("feature_representation", "pooler_output_or_cls_token"),
        )
    )

    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            images = prepare_dino_global_images(batch, dataset_root, image_size)
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            features, representation = extract_features(outputs, representation)
            probabilities = torch.softmax(head(features), dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            for index, record in enumerate(batch):
                probs = probabilities[index].detach().cpu().tolist()
                pred_index = int(predictions[index].detach().cpu().item())
                true_index = class_to_index[record.label]
                y_true.append(true_index)
                y_pred.append(pred_index)
                row = {
                    "image_id": record.image_id,
                    "relative_path": record.relative_path,
                    "split": record.split,
                    "true_label": record.label,
                    "predicted_label": labels[pred_index],
                    "model_name": model_config["model_name"],
                    "config_id": model_config["config_id"],
                    "seed": provenance["seed"],
                    "feature_representation": representation,
                    "confidence": float(probs[pred_index]),
                }
                for label, value in zip(labels, probs, strict=True):
                    row[f"prob_{label}"] = float(value)
                prediction_rows.append(row)

    prediction_path = output_dir / "predictions_test.csv"
    write_csv(
        prediction_path,
        prediction_rows,
        [
            "image_id",
            "relative_path",
            "split",
            "true_label",
            "predicted_label",
            "model_name",
            "config_id",
            "seed",
            "feature_representation",
            "confidence",
        ]
        + [f"prob_{label}" for label in labels],
    )
    labels, metrics, per_class, confusion = evaluate_predictions(y_true, y_pred, class_to_index)
    files = write_metrics_files(
        output_dir=output_dir,
        labels=labels,
        metrics=metrics,
        per_class=per_class,
        confusion=confusion,
        prediction_file=prediction_path,
        checkpoint=checkpoint,
        model_name=str(model_config["model_name"]),
        provenance=provenance,
        preprocessing_description=(
            "EXIF transpose -> RGB -> aspect-preserving BICUBIC resize -> "
            "black padding to 224 x 224 -> DINOv3 processor"
        ),
    )
    return result_summary(
        scope="global",
        model_name=str(model_config["model_name"]),
        output_dir=output_dir,
        checkpoint=checkpoint,
        predictions=prediction_path,
        labels=labels,
        metrics=metrics,
        per_class=per_class,
        confusion=confusion,
        files=files,
    )


def evaluate_deit(
    *,
    records: list[ManifestRecord],
    dataset_root: Path,
    checkpoint: Path,
    output_dir: Path,
    model_config: dict[str, Any],
    image_size: tuple[int, int],
    device: Any,
    batch_size: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    import timm
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_to_index = {str(label): int(index) for label, index in payload["class_to_index"].items()}
    labels = labels_from_mapping(class_to_index)
    variant = str(payload.get("model_variant", "deit_tiny_patch16_224"))
    model = timm.create_model(variant, pretrained=False, num_classes=len(class_to_index)).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    dataset = ManifestTensorDataset(records, dataset_root, class_to_index, image_size)
    prediction_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            items = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            images = torch.stack([item[0] for item in items], dim=0).to(device)
            probabilities = torch.softmax(model(images), dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            for index, (_tensor, true_index, record) in enumerate(items):
                probs = probabilities[index].detach().cpu().tolist()
                pred_index = int(predictions[index].detach().cpu().item())
                y_true.append(int(true_index))
                y_pred.append(pred_index)
                row = {
                    "image_id": record.image_id,
                    "relative_path": record.relative_path,
                    "split": record.split,
                    "true_label": record.label,
                    "predicted_label": labels[pred_index],
                    "model_name": model_config["model_name"],
                    "config_id": model_config["config_id"],
                    "seed": provenance["seed"],
                    "confidence": float(probs[pred_index]),
                }
                for label, value in zip(labels, probs, strict=True):
                    row[f"prob_{label}"] = float(value)
                prediction_rows.append(row)

    prediction_path = output_dir / "predictions_test.csv"
    write_csv(
        prediction_path,
        prediction_rows,
        [
            "image_id",
            "relative_path",
            "split",
            "true_label",
            "predicted_label",
            "model_name",
            "config_id",
            "seed",
            "confidence",
        ]
        + [f"prob_{label}" for label in labels],
    )
    labels, metrics, per_class, confusion = evaluate_predictions(y_true, y_pred, class_to_index)
    files = write_metrics_files(
        output_dir=output_dir,
        labels=labels,
        metrics=metrics,
        per_class=per_class,
        confusion=confusion,
        prediction_file=prediction_path,
        checkpoint=checkpoint,
        model_name=str(model_config["model_name"]),
        provenance=provenance,
        preprocessing_description=(
            "EXIF transpose -> RGB -> aspect-preserving BICUBIC resize with black padding -> "
            "ImageNet normalization"
        ),
    )
    return result_summary(
        scope="global",
        model_name=str(model_config["model_name"]),
        output_dir=output_dir,
        checkpoint=checkpoint,
        predictions=prediction_path,
        labels=labels,
        metrics=metrics,
        per_class=per_class,
        confusion=confusion,
        files=files,
    )


def build_region_head(checkpoint_payload: dict[str, Any], device: Any) -> Any:
    import torch.nn as nn

    head_config = checkpoint_payload.get("config", {}).get("head", {})
    feature_dim = int(checkpoint_payload.get("feature_dim", head_config.get("input_dim", 768)))
    head = nn.Sequential(
        nn.Linear(feature_dim, int(head_config.get("hidden_dim", 128))),
        nn.ReLU(),
        nn.Dropout(float(head_config.get("dropout", 0.2))),
        nn.Linear(int(head_config.get("hidden_dim", 128)), len(checkpoint_payload["class_to_index"])),
    ).to(device)
    head.load_state_dict(checkpoint_payload["head_state_dict"], strict=True)
    head.eval()
    return head


def load_region_backbone(checkpoint_payload: dict[str, Any], device: Any) -> tuple[Any, Any]:
    import torch

    source_checkpoint = repo_path(str(checkpoint_payload["source_checkpoint"]))
    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    model_id = str(checkpoint_payload.get("source_model_id") or source_payload.get("model_id"))
    processor, model = load_dinov3(model_id, allow_model_download=False, device=device)
    model.load_state_dict(source_payload["model_state_dict"], strict=True)
    model.eval()
    return processor, model


def select_region_records(
    region_table: Path,
    class_to_index: dict[str, int],
    include_special: bool,
) -> tuple[list[RegionRecord], int]:
    selected: list[RegionRecord] = []
    excluded = 0
    for record in read_region_table(region_table):
        if record.split != FINAL_SPLIT or not record.matched_manifest:
            continue
        if record.mapped_label == SPECIAL_REGION_LABEL:
            if include_special and record.mapped_label in class_to_index:
                selected.append(record)
            else:
                excluded += 1
        elif record.is_global_class and record.mapped_label in class_to_index:
            selected.append(record)
        else:
            excluded += 1
    return selected, excluded


def evaluate_region_head(
    *,
    region_table: Path,
    manual_root: Path,
    checkpoint: Path,
    output_dir: Path,
    model_config: dict[str, Any],
    images_subdirectory: str,
    device: Any,
    batch_size: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_to_index = {str(label): int(index) for label, index in payload["class_to_index"].items()}
    labels = labels_from_mapping(class_to_index)
    records, excluded = select_region_records(
        region_table,
        class_to_index,
        bool(model_config.get("include_nicht_bewertbar", False)),
    )
    if not records:
        raise ValueError(f"No test regions selected for {model_config['model_name']}")
    processor, model = load_region_backbone(payload, device)
    head = build_region_head(payload, device)
    crop_context = SimpleNamespace(images_dir=manual_root / images_subdirectory, image_size=(224, 224))
    crop_mode = str(payload.get("crop_mode", "stretch_resize"))
    context_margin = float(payload.get("context_margin", 0.0))
    representation = str(payload.get("config", {}).get("feature_representation", "pooler_output_or_cls_token"))
    prediction_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            crop_items = [load_region_crop(record, crop_context, crop_mode, context_margin) for record in batch]
            inputs = processor(images=[item[0] for item in crop_items], return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            features, representation = extract_features(outputs, representation)
            probabilities = torch.softmax(head(features), dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            for index, record in enumerate(batch):
                probs = probabilities[index].detach().cpu().tolist()
                pred_index = int(predictions[index].detach().cpu().item())
                true_index = class_to_index[record.mapped_label]
                y_true.append(true_index)
                y_pred.append(pred_index)
                row = {
                    "region_id": record.region_id,
                    "source_image": record.source_image,
                    "split": record.split,
                    "original_label": record.original_label,
                    "mapped_label": record.mapped_label,
                    "is_global_class": record.is_global_class,
                    "x_min": record.raw["x_min"],
                    "y_min": record.raw["y_min"],
                    "x_max": record.raw["x_max"],
                    "y_max": record.raw["y_max"],
                    "true_label": record.mapped_label,
                    "predicted_label": labels[pred_index],
                    "confidence": float(probs[pred_index]),
                    "crop_mode": crop_mode,
                    "context_margin": context_margin,
                    "feature_representation": representation,
                    "correct": pred_index == true_index,
                    "model_name": model_config["model_name"],
                    "config_id": model_config["config_id"],
                    "seed": provenance["seed"],
                }
                for label, value in zip(labels, probs, strict=True):
                    row[f"prob_{label}"] = float(value)
                prediction_rows.append(row)

    prediction_path = output_dir / "predictions_regions_test.csv"
    write_csv(
        prediction_path,
        prediction_rows,
        [
            "region_id",
            "source_image",
            "split",
            "original_label",
            "mapped_label",
            "is_global_class",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "true_label",
            "predicted_label",
            "confidence",
            "crop_mode",
            "context_margin",
            "feature_representation",
            "correct",
            "model_name",
            "config_id",
            "seed",
        ]
        + [f"prob_{label}" for label in labels],
    )
    labels, metrics, per_class, confusion = evaluate_predictions(y_true, y_pred, class_to_index)
    files = write_metrics_files(
        output_dir=output_dir,
        labels=labels,
        metrics=metrics,
        per_class=per_class,
        confusion=confusion,
        prediction_file=prediction_path,
        checkpoint=checkpoint,
        model_name=str(model_config["model_name"]),
        provenance=provenance,
        preprocessing_description="Checkpoint crop mode and context margin -> DINOv3 processor",
    )
    summary = result_summary(
        scope="region",
        model_name=str(model_config["model_name"]),
        output_dir=output_dir,
        checkpoint=checkpoint,
        predictions=prediction_path,
        labels=labels,
        metrics=metrics,
        per_class=per_class,
        confusion=confusion,
        files=files,
    )
    summary["num_regions_excluded"] = excluded
    return summary


def tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def run_preprocessing_check(
    config: dict[str, Any],
    *,
    dataset_root: Path,
    max_images: int | None,
) -> dict[str, Any]:
    from PIL import Image, ImageOps
    from transformers import AutoImageProcessor

    manifest = repo_path(str(config["split_manifest"]))
    all_records = final_test_records(manifest)
    records = all_records[:max_images] if max_images is not None else all_records
    preprocessing_config = config["preprocessing"]["dino_global"]
    image_size_values = preprocessing_config.get("image_size", [224, 224])
    image_size = (int(image_size_values[0]), int(image_size_values[1]))
    model_config = config["global_models"]["dinov3_frozen_linear_head"]
    processor = AutoImageProcessor.from_pretrained(str(model_config["model_id"]), local_files_only=True)

    identical = 0
    square_count = 0
    square_changed = 0
    non_square_count = 0
    non_square_changed = 0
    shapes_equal = 0
    mean_differences: list[float] = []
    max_differences: list[float] = []
    square_mean_differences: list[float] = []
    square_max_differences: list[float] = []
    non_square_mean_differences: list[float] = []
    non_square_max_differences: list[float] = []
    non_square_details: list[dict[str, Any]] = []

    for record in records:
        image_path = resolve_record_path(dataset_root, record)
        with Image.open(image_path) as image:
            legacy_image = ImageOps.exif_transpose(image).convert("RGB")
            original_size = legacy_image.size
        with Image.open(image_path) as image:
            training_image = prepare_dinov3_image(image, image_size)
        legacy_tensor = processor(images=[legacy_image], return_tensors="pt")["pixel_values"][0]
        training_tensor = processor(images=[training_image], return_tensors="pt")["pixel_values"][0]
        same_shape = tuple(legacy_tensor.shape) == tuple(training_tensor.shape)
        shapes_equal += int(same_shape)
        is_identical = bool(same_shape and legacy_tensor.equal(training_tensor))
        identical += int(is_identical)
        difference = (legacy_tensor - training_tensor).abs()
        mean_difference = float(difference.mean().item())
        max_difference = float(difference.max().item())
        mean_differences.append(mean_difference)
        max_differences.append(max_difference)
        is_square = original_size[0] == original_size[1]
        if is_square:
            square_count += 1
            square_changed += int(not is_identical)
            square_mean_differences.append(mean_difference)
            square_max_differences.append(max_difference)
        else:
            non_square_count += 1
            non_square_changed += int(not is_identical)
            non_square_mean_differences.append(mean_difference)
            non_square_max_differences.append(max_difference)
            non_square_details.append(
                {
                    "image_id": record.image_id,
                    "relative_path": record.relative_path,
                    "original_size": list(original_size),
                    "legacy_tensor_shape": list(legacy_tensor.shape),
                    "training_tensor_shape": list(training_tensor.shape),
                    "mean_absolute_difference": mean_difference,
                    "max_absolute_difference": max_difference,
                    "legacy_tensor_sha256": tensor_sha256(legacy_tensor),
                    "training_tensor_sha256": tensor_sha256(training_tensor),
                    "geometry_difference": "stretched_to_square_vs_aspect_preserving_padding",
                }
            )

    changed = len(records) - identical
    conclusion = "no_problem" if changed == 0 else "reproducible_preprocessing_error"
    return {
        "mode": "check_preprocessing",
        "writes_files": False,
        "loads_models": False,
        "produces_predictions_or_metrics": False,
        "manifest": relative_to_repo(manifest),
        "test_records_total": len(all_records),
        "records_compared": len(records),
        "processor_class": type(processor).__name__,
        "processor_size": dict(processor.size),
        "processor_resample": int(processor.resample),
        "training_pipeline": "EXIF transpose -> RGB -> aspect-preserving bicubic resize with black padding -> DINOv3 processor",
        "legacy_final_pipeline": "EXIF transpose -> RGB -> direct DINOv3 processor square resize",
        "tensor_shapes_equal_count": shapes_equal,
        "identical_tensor_count": identical,
        "changed_tensor_count": changed,
        "square_image_count": square_count,
        "square_changed_tensor_count": square_changed,
        "square_mean_absolute_difference": (
            sum(square_mean_differences) / len(square_mean_differences) if square_mean_differences else 0.0
        ),
        "square_maximum_absolute_difference": max(square_max_differences, default=0.0),
        "non_square_image_count": non_square_count,
        "non_square_changed_tensor_count": non_square_changed,
        "non_square_mean_absolute_difference": (
            sum(non_square_mean_differences) / len(non_square_mean_differences) if non_square_mean_differences else 0.0
        ),
        "non_square_maximum_absolute_difference": max(non_square_max_differences, default=0.0),
        "mean_of_mean_absolute_differences": sum(mean_differences) / len(mean_differences),
        "maximum_absolute_difference": max(max_differences),
        "non_square_images": non_square_details,
        "conclusion": conclusion,
        "corrected_final_dino_evaluation_required": changed > 0,
    }


def validate_config(config: dict[str, Any]) -> None:
    if str(config.get("split")) != FINAL_SPLIT:
        raise ValueError("Final evaluation config must use split=test")
    safety = config.get("safety", {})
    if not bool(safety.get("model_selection_complete")):
        raise ValueError("Final test evaluation requires completed model selection")
    if bool(safety.get("training_allowed")):
        raise ValueError("Final evaluation config must keep training_allowed=false")


def run_final_evaluation(
    config: dict[str, Any],
    config_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_models = select_model_keys(config, args)
    run_global = bool(selected_models & set(GLOBAL_MODEL_KEYS))
    run_regions = bool(selected_models & set(REGION_MODEL_KEYS))
    dataset_root = ensure_local_directory(args.dataset_root, "--dataset-root") if run_global else None
    manual_root = ensure_local_directory(args.manual_root, "--manual-root") if run_regions else None
    manifest = repo_path(str(config["split_manifest"]))
    records = final_test_records(manifest)
    output_root = ensure_output_root(repo_path(str(config["output_root"])))
    device_raw = str(args.device if args.device is not None else config.get("runtime", {}).get("device", 0))
    batch_size = int(args.batch_size or config.get("runtime", {}).get("batch_size", 16))
    device = resolve_device(device_raw)
    provenance = provenance_metadata(
        config,
        config_path,
        mode="final_test",
        device=device_raw,
        batch_size=batch_size,
        selected_models=selected_models,
    )

    selected_checkpoint_configs: dict[str, str] = {}
    if run_global:
        selected_checkpoint_configs.update(
            {
                key: str(model_config["checkpoint"])
                for key, model_config in config.get("global_models", {}).items()
                if key in selected_models
            }
        )
    if run_regions:
        selected_checkpoint_configs.update(
            {
                key: str(model_config["checkpoint"])
                for key, model_config in config.get("region_models", {}).items()
                if key in selected_models
            }
        )
    checkpoints = {key: repo_path(path) for key, path in selected_checkpoint_configs.items()}
    missing_checkpoints = [relative_to_repo(path) for path in checkpoints.values() if not path.is_file()]
    if missing_checkpoints:
        raise FileNotFoundError(f"Missing selected checkpoints: {missing_checkpoints}")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "evaluation_metadata.json", provenance | {"status": "started"})

    summaries: list[dict[str, Any]] = []
    global_configs = config.get("global_models", {})
    if run_global:
        assert dataset_root is not None
        yolo_config = global_configs["yolo11n_cls"]
        if "yolo11n_cls" in selected_models:
            summaries.append(
                evaluate_yolo(
                    records=records,
                    dataset_root=dataset_root,
                    checkpoint=repo_path(yolo_config["checkpoint"]),
                    output_dir=output_root / "global" / "yolov11n_cls",
                    model_config=yolo_config,
                    device=device_raw,
                    batch_size=batch_size,
                    provenance=provenance,
                )
            )
        dino_preprocessing = config["preprocessing"]["dino_global"]
        size_values = dino_preprocessing.get("image_size", [224, 224])
        dino_image_size = (int(size_values[0]), int(size_values[1]))
        for key, folder in (
            ("dinov3_frozen_linear_head", "dinov3_linear_head"),
            ("dinov3_partial_finetune_last2", "dinov3_partial_finetune_last2"),
        ):
            model_config = global_configs[key]
            if key in selected_models:
                summaries.append(
                    evaluate_dino_global(
                        records=records,
                        dataset_root=dataset_root,
                        checkpoint=repo_path(model_config["checkpoint"]),
                        output_dir=output_root / "global" / folder,
                        model_config=model_config,
                        image_size=dino_image_size,
                        device=device,
                        batch_size=batch_size,
                        provenance=provenance,
                    )
                )
        deit_config = global_configs["deit_tiny_scratch"]
        if "deit_tiny_scratch" in selected_models:
            deit_size_values = config["preprocessing"]["deit"].get("image_size", [224, 224])
            summaries.append(
                evaluate_deit(
                    records=records,
                    dataset_root=dataset_root,
                    checkpoint=repo_path(deit_config["checkpoint"]),
                    output_dir=output_root / "global" / "deit_tiny_scratch",
                    model_config=deit_config,
                    image_size=(int(deit_size_values[0]), int(deit_size_values[1])),
                    device=device,
                    batch_size=batch_size,
                    provenance=provenance,
                )
            )

    if run_regions:
        assert manual_root is not None
        region_data = config["region_data"]
        region_table = repo_path(region_data["annotation_table"])
        if not region_table.is_file():
            raise FileNotFoundError(f"Region annotation table is missing: {region_table}")
        for key, folder in (
            ("dinov3_region_head_4class", "dinov3_region_head_4class"),
            ("dinov3_region_head_5class", "dinov3_region_head_5class"),
        ):
            model_config = config["region_models"][key]
            if key in selected_models:
                summaries.append(
                    evaluate_region_head(
                        region_table=region_table,
                        manual_root=manual_root,
                        checkpoint=repo_path(model_config["checkpoint"]),
                        output_dir=output_root / "regions" / folder,
                        model_config=model_config,
                        images_subdirectory=str(region_data.get("images_subdirectory", "images")),
                        device=device,
                        batch_size=batch_size,
                        provenance=provenance,
                    )
                )

    summary_rows = [
        {
            "scope": item["scope"],
            "model_name": item["model_name"],
            "num_items": item["num_items"],
            "accuracy": item["accuracy"],
            "balanced_accuracy": item["balanced_accuracy"],
            "macro_f1": item["macro_f1"],
            "checkpoint": item["checkpoint"],
            "output_dir": item["output_dir"],
        }
        for item in summaries
    ]
    write_csv(
        output_root / "summary_test_results.csv",
        summary_rows,
        ["scope", "model_name", "num_items", "accuracy", "balanced_accuracy", "macro_f1", "checkpoint", "output_dir"],
    )
    write_json(output_root / "summary_test_results.json", {"evaluation_provenance": provenance, "results": summaries})
    write_json(output_root / "evaluation_metadata.json", provenance | {"status": "completed", "result_count": len(summaries)})
    return {"output_root": relative_to_repo(output_root), "results": summaries}


def main() -> int:
    args = parse_args()
    try:
        require_final_test_access(
            allow_final_test=bool(args.allow_final_test),
            check_preprocessing=bool(args.check_preprocessing),
        )
        config_path = repo_path(args.config)
        config = load_yaml_config(config_path)
        validate_config(config)
        if args.check_preprocessing:
            dataset_root = ensure_local_directory(args.dataset_root, "--dataset-root")
            result = run_preprocessing_check(
                config,
                dataset_root=dataset_root,
                max_images=args.max_preprocessing_images,
            )
        else:
            result = run_final_evaluation(config, config_path, args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
