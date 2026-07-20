"""Evaluate a DINOv3 partial fine-tuning checkpoint on CVAT rectangle regions.

This script performs inference only. It reads a local region annotation table,
loads an existing DINOv3 partial fine-tuning checkpoint, crops annotated
rectangles from the source images, and computes four-class validation metrics.
It never trains a model, creates splits, or uses the test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.train_dinov3_head import (  # noqa: E402
    extract_features,
    load_dinov3,
    metric_payload,
    write_confusion,
)


DEFAULT_CONFIG = "configs/experiments/dinov3_region_eval.yaml"
REGION_REQUIRED_COLUMNS = {
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
    "matched_manifest",
}
VALID_CROP_MODES = {"pad_square", "stretch_resize"}


@dataclass(frozen=True)
class RegionRecord:
    region_id: str
    source_image: str
    split: str
    original_label: str
    mapped_label: str
    is_global_class: bool
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    matched_manifest: bool
    raw: dict[str, str]


@dataclass(frozen=True)
class EvalContext:
    config: dict[str, Any]
    config_path: Path
    region_table: Path
    region_summary: Path | None
    checkpoint_path: Path
    manual_root: Path | None
    images_dir: Path | None
    output_dir: Path
    image_size: tuple[int, int]
    class_to_index: dict[str, int]
    index_to_class: dict[int, str]
    special_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config.")
    parser.add_argument("--manual-root", default=None, help="Local manual_all root; required for --allow-evaluate.")
    parser.add_argument("--region-table", default=None, help="Override region annotation CSV.")
    parser.add_argument("--checkpoint", default=None, help="Override DINOv3 checkpoint path.")
    parser.add_argument("--dry-run", action="store_true", help="Report region counts only; write no files.")
    parser.add_argument("--check-model", action="store_true", help="Load DINOv3 checkpoint and report model readiness.")
    parser.add_argument("--allow-evaluate", action="store_true", help="Run local inference and write ignored output artifacts.")
    parser.add_argument("--allow-download", action="store_true", help="Allow Transformers to download DINOv3 weights.")
    parser.add_argument("--split", default=None, choices=["train", "val"], help="Split to evaluate. Defaults to val.")
    parser.add_argument("--max-regions", type=int, default=None, help="Optional local cap for debugging inference.")
    parser.add_argument("--crop-mode", default=None, choices=sorted(VALID_CROP_MODES), help="Region crop preprocessing mode.")
    parser.add_argument(
        "--include-nicht-bewertbar",
        action="store_true",
        help="Infer Nicht_bewertbar rows separately; they remain excluded from four-class metrics.",
    )
    args = parser.parse_args()

    active_modes = [args.dry_run, args.check_model, args.allow_evaluate]
    if sum(bool(value) for value in active_modes) > 1:
        parser.error("Choose only one mode: --dry-run, --check-model, or --allow-evaluate")
    if not any(active_modes):
        args.dry_run = True
    if args.max_regions is not None and args.max_regions < 1:
        parser.error("--max-regions must be at least 1")
    return args


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            cwd=REPO_ROOT,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def package_versions() -> dict[str, str | None]:
    packages = ["torch", "transformers", "Pillow", "PyYAML"]
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def bool_from_csv(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "ja"}


def read_region_table(path: Path) -> list[RegionRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Region annotation CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REGION_REQUIRED_COLUMNS - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Region annotation CSV is missing columns: {missing}")
        records = [
            RegionRecord(
                region_id=row["region_id"],
                source_image=row["source_image"],
                split=row["split"],
                original_label=row["original_label"],
                mapped_label=row["mapped_label"],
                is_global_class=bool_from_csv(row["is_global_class"]),
                x_min=float(row["x_min"]),
                y_min=float(row["y_min"]),
                x_max=float(row["x_max"]),
                y_max=float(row["y_max"]),
                matched_manifest=bool_from_csv(row["matched_manifest"]),
                raw=row,
            )
            for row in reader
        ]
    if not records:
        raise ValueError(f"Region annotation CSV contains no rows: {path}")
    return records


def read_optional_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required to read the DINOv3 checkpoint.") from exc
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "head_state_dict", "class_to_index", "model_id", "feature_info"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {missing}")
    return checkpoint


def class_mapping_from_config_and_checkpoint(config: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, int]:
    checkpoint_mapping = {str(label): int(index) for label, index in checkpoint["class_to_index"].items()}
    configured_order = [str(label) for label in config.get("class_order", [])]
    if configured_order:
        configured_mapping = {label: index for index, label in enumerate(configured_order)}
        if configured_mapping != checkpoint_mapping:
            raise ValueError(
                "Config class_order does not match checkpoint class_to_index. "
                f"config={configured_mapping}, checkpoint={checkpoint_mapping}"
            )
    return checkpoint_mapping


def resolve_manual_root(config: dict[str, Any], override: str | None, *, required: bool) -> Path | None:
    if override:
        manual_root = Path(override).expanduser()
    else:
        env_name = str(config.get("inputs", {}).get("manual_root_env", "BMW25_MANUAL_ALL_ROOT"))
        env_value = os.environ.get(env_name)
        manual_root = Path(env_value).expanduser() if env_value else None
    if manual_root is None:
        if required:
            raise ValueError("Manual root is required for --allow-evaluate. Pass --manual-root or set the configured env var.")
        return None
    if not manual_root.exists():
        raise FileNotFoundError(f"Manual root does not exist: {manual_root}")
    if not manual_root.is_dir():
        raise NotADirectoryError(f"Manual root is not a directory: {manual_root}")
    return manual_root


def prepare_context(args: argparse.Namespace) -> EvalContext:
    config_path = resolve_repo_path(args.config)
    config = load_yaml(config_path)
    if str(config.get("model_family")) != "dinov3_region_eval":
        raise ValueError(f"Expected model_family='dinov3_region_eval', got {config.get('model_family')!r}")

    inputs = config.get("inputs", {})
    output_cfg = config.get("output", {})
    eval_cfg = config.get("evaluation", {})
    region_table = resolve_repo_path(args.region_table or str(inputs["region_annotations"]))
    region_summary_raw = inputs.get("region_summary")
    region_summary = resolve_repo_path(str(region_summary_raw)) if region_summary_raw else None
    checkpoint_path = resolve_repo_path(args.checkpoint or str(inputs["checkpoint"]))
    checkpoint = load_checkpoint_metadata(checkpoint_path)
    class_to_index = class_mapping_from_config_and_checkpoint(config, checkpoint)
    index_to_class = {index: label for label, index in class_to_index.items()}
    image_size_raw = eval_cfg.get("image_size", [224, 224])
    output_dir = resolve_repo_path(str(output_cfg.get("output_dir", "outputs/region_analysis/dinov3_region_eval_bmw25_seed42")))
    manual_root = resolve_manual_root(config, args.manual_root, required=bool(args.allow_evaluate))
    images_dir = None
    if manual_root is not None:
        images_dir = manual_root / str(inputs.get("images_dir", "images"))
    return EvalContext(
        config=config,
        config_path=config_path,
        region_table=region_table,
        region_summary=region_summary,
        checkpoint_path=checkpoint_path,
        manual_root=manual_root,
        images_dir=images_dir,
        output_dir=output_dir,
        image_size=(int(image_size_raw[0]), int(image_size_raw[1])),
        class_to_index=class_to_index,
        index_to_class=index_to_class,
        special_label=str(eval_cfg.get("special_label", "Nicht_bewertbar")),
    )


def selected_split(context: EvalContext, args: argparse.Namespace) -> str:
    return str(args.split or context.config.get("evaluation", {}).get("default_split", "val"))


def selected_crop_mode(context: EvalContext, args: argparse.Namespace) -> str:
    mode = str(args.crop_mode or context.config.get("evaluation", {}).get("crop_mode", "pad_square"))
    if mode not in VALID_CROP_MODES:
        raise ValueError(f"Unsupported crop_mode: {mode}")
    return mode


def filter_regions(
    rows: list[RegionRecord],
    *,
    split: str,
    special_label: str,
    include_special: bool,
    max_regions: int | None,
) -> tuple[list[RegionRecord], dict[str, Any]]:
    source_split_counts: dict[str, int] = {}
    for row in rows:
        source_split_counts[row.split] = source_split_counts.get(row.split, 0) + 1

    candidates = [
        row
        for row in rows
        if row.split == split and row.matched_manifest and row.split != "test"
    ]
    global_rows = [
        row
        for row in candidates
        if row.is_global_class and row.mapped_label != special_label
    ]
    special_rows = [
        row
        for row in candidates
        if row.mapped_label == special_label or not row.is_global_class
    ]
    selected = global_rows + special_rows if include_special else global_rows
    if max_regions is not None:
        selected = selected[:max_regions]
    summary = {
        "source_rows_total": len(rows),
        "source_split_counts": source_split_counts,
        "selected_split": split,
        "test_rows_in_region_table": source_split_counts.get("test", 0),
        "test_excluded": True,
        "matched_required": True,
        "candidate_rows_in_split": len(candidates),
        "global_class_regions_in_split": len(global_rows),
        "nicht_bewertbar_regions_in_split": len(special_rows),
        "regions_selected_for_inference": len(selected),
        "regions_selected_for_four_class_metrics": sum(
            1 for row in selected if row.is_global_class and row.mapped_label != special_label
        ),
        "max_regions": max_regions,
        "regions_per_class_in_split": dict(sorted(count_by_label(global_rows).items())),
        "special_regions_included_in_predictions": include_special,
    }
    return selected, summary


def count_by_label(rows: list[RegionRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.mapped_label] = counts.get(row.mapped_label, 0) + 1
    return counts


def clip_box(record: RegionRecord, image: Image.Image) -> tuple[int, int, int, int]:
    width, height = image.size
    x_min = min(max(record.x_min, 0.0), float(width))
    y_min = min(max(record.y_min, 0.0), float(height))
    x_max = min(max(record.x_max, 0.0), float(width))
    y_max = min(max(record.y_max, 0.0), float(height))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Region has empty clipped box: {record.region_id}")
    return (
        int(round(x_min)),
        int(round(y_min)),
        int(round(x_max)),
        int(round(y_max)),
    )


def pad_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def crop_region_image(record: RegionRecord, context: EvalContext, crop_mode: str) -> Image.Image:
    if context.images_dir is None:
        raise ValueError("manual_root/images_dir is required for inference")
    image_path = context.images_dir / record.source_image
    if not image_path.exists():
        raise FileNotFoundError(f"Source image for region does not exist: {image_path}")
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        box = clip_box(record, image)
        crop = image.crop(box)
    if crop_mode == "pad_square":
        crop = pad_square(crop)
    elif crop_mode != "stretch_resize":
        raise ValueError(f"Unsupported crop_mode: {crop_mode}")
    return crop.resize(context.image_size, Image.Resampling.BICUBIC)


def device_from_config(context: EvalContext) -> Any:
    import torch

    raw_device = context.config.get("evaluation", {}).get("device", 0)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{raw_device}")
    return torch.device("cpu")


def cuda_info() -> dict[str, Any]:
    import torch

    cuda_available = torch.cuda.is_available()
    return {
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def model_id_for(context: EvalContext, checkpoint: dict[str, Any]) -> str:
    return str(context.config.get("model_id") or checkpoint.get("model_id"))


def load_model_and_head(context: EvalContext, args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any], Any]:
    import torch
    import torch.nn as nn

    checkpoint = load_checkpoint_metadata(context.checkpoint_path)
    device = device_from_config(context)
    allow_download = bool(args.allow_download or context.config.get("evaluation", {}).get("allow_download", False))
    processor, model = load_dinov3(
        model_id_for(context, checkpoint),
        allow_model_download=allow_download,
        device=device,
    )
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Backbone checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    feature_dim = int(checkpoint.get("feature_info", {}).get("feature_dim", 768))
    head = nn.Linear(feature_dim, len(context.class_to_index)).to(device)
    missing_head, unexpected_head = head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    if missing_head or unexpected_head:
        raise RuntimeError(f"Head checkpoint mismatch: missing={missing_head}, unexpected={unexpected_head}")
    model.eval()
    head.eval()
    return processor, model, head, checkpoint, device


def check_model(context: EvalContext, args: argparse.Namespace) -> dict[str, Any]:
    processor, model, head, checkpoint, device = load_model_and_head(context, args)
    crop_mode = selected_crop_mode(context, args)
    return {
        "model_name": model_id_for(context, checkpoint),
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "checkpoint_path": relative_to_repo(context.checkpoint_path),
        "checkpoint_loaded": True,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics"),
        "classes": [context.index_to_class[index] for index in sorted(context.index_to_class)],
        "head_class": type(head).__name__,
        "feature_dim": int(checkpoint.get("feature_info", {}).get("feature_dim", 768)),
        "device": str(device),
        **cuda_info(),
        "crop_mode": crop_mode,
        "input_size": list(context.image_size),
        "outputs_written": False,
    }


def dry_run(context: EvalContext, args: argparse.Namespace) -> dict[str, Any]:
    rows = read_region_table(context.region_table)
    split = selected_split(context, args)
    selected, summary = filter_regions(
        rows,
        split=split,
        special_label=context.special_label,
        include_special=bool(args.include_nicht_bewertbar),
        max_regions=args.max_regions,
    )
    region_summary = read_optional_summary(context.region_summary)
    if region_summary is not None:
        summary["upstream_region_summary"] = {
            "num_unmatched_regions_excluded": region_summary.get("num_unmatched_regions_excluded"),
            "num_test_regions_excluded": region_summary.get("num_test_regions_excluded"),
            "source_summary": relative_to_repo(context.region_summary) if context.region_summary else None,
        }
    summary.update(
        {
            "mode": "dry_run",
            "region_table": relative_to_repo(context.region_table),
            "crop_mode": selected_crop_mode(context, args),
            "input_size": list(context.image_size),
            "outputs_written": False,
            "test_usage_note": "Test regions are excluded and are not used for this validation workflow.",
            "selected_region_examples": [row.region_id for row in selected[:5]],
        }
    )
    return summary


def collate_images(images: list[Image.Image], processor: Any, device: Any) -> dict[str, Any]:
    inputs = processor(images=images, return_tensors="pt")
    return {key: value.to(device) for key, value in inputs.items()}


def predict_regions(context: EvalContext, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    rows = read_region_table(context.region_table)
    split = selected_split(context, args)
    crop_mode = selected_crop_mode(context, args)
    selected, filter_summary = filter_regions(
        rows,
        split=split,
        special_label=context.special_label,
        include_special=bool(args.include_nicht_bewertbar),
        max_regions=args.max_regions,
    )
    if not selected:
        raise ValueError("No regions selected for inference")
    processor, model, head, checkpoint, device = load_model_and_head(context, args)
    batch_size = int(context.config.get("evaluation", {}).get("batch_size", 16))
    labels = [context.index_to_class[index] for index in sorted(context.index_to_class)]
    prediction_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    representation_used = str(context.config.get("feature_representation", "pooler_output_or_cls_token"))

    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch_records = selected[start : start + batch_size]
            images = [crop_region_image(record, context, crop_mode) for record in batch_records]
            inputs = collate_images(images, processor, device)
            outputs = model(**inputs)
            features, representation_used = extract_features(
                outputs,
                str(context.config.get("feature_representation", "pooler_output_or_cls_token")),
            )
            logits = head(features)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1).values

            for row_index, record in enumerate(batch_records):
                pred_index = int(predictions[row_index].detach().cpu().item())
                pred_label = context.index_to_class[pred_index]
                is_metric_row = record.is_global_class and record.mapped_label != context.special_label
                correct: str | bool = ""
                if is_metric_row:
                    true_index = context.class_to_index[record.mapped_label]
                    y_true.append(true_index)
                    y_pred.append(pred_index)
                    correct = pred_index == true_index
                prediction = {
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
                    "pred_label": pred_label,
                    "confidence": float(confidences[row_index].detach().cpu().item()),
                    "correct": correct,
                }
                probs = probabilities[row_index].detach().cpu().tolist()
                for label, value in zip(labels, probs, strict=True):
                    prediction[f"prob_{label}"] = float(value)
                prediction_rows.append(prediction)

    metrics, per_class, confusion = metric_payload(y_true, y_pred, context.index_to_class)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    output_cfg = context.config.get("output", {})
    predictions_path = context.output_dir / str(output_cfg.get("predictions", f"predictions_regions_{split}.csv"))
    metrics_json_path = context.output_dir / str(output_cfg.get("metrics_json", f"{split}_region_metrics.json"))
    metrics_csv_path = context.output_dir / str(output_cfg.get("metrics_csv", f"{split}_region_metrics.csv"))
    confusion_path = context.output_dir / str(output_cfg.get("confusion_matrix", f"confusion_matrix_{split}_regions.csv"))

    write_prediction_csv(predictions_path, prediction_rows, labels)
    metrics_payload = {
        "created_by": "scripts/evaluate_dinov3_regions.py",
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "model_name": model_id_for(context, checkpoint),
        "checkpoint_path": relative_to_repo(context.checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "region_table": relative_to_repo(context.region_table),
        "split": split,
        "crop_mode": crop_mode,
        "input_size": list(context.image_size),
        "feature_representation_used": representation_used,
        "overall_metrics": metrics,
        "per_class_metrics": per_class,
        "confusion_matrix": {
            "labels": labels,
            "rows_are_true_labels": True,
            "values": confusion,
        },
        "num_four_class_regions": len(y_true),
        "num_nicht_bewertbar_inferred": sum(
            1 for row in prediction_rows if row["mapped_label"] == context.special_label
        ),
        "filter_summary": filter_summary,
        "test_used": False,
    }
    write_json(metrics_json_path, metrics_payload)
    write_metrics_csv(metrics_csv_path, metrics, per_class, len(y_true), metrics_payload["num_nicht_bewertbar_inferred"])
    write_confusion(confusion_path, labels, confusion)
    return {
        "mode": "allow_evaluate",
        "predictions": relative_to_repo(predictions_path),
        "metrics_json": relative_to_repo(metrics_json_path),
        "metrics_csv": relative_to_repo(metrics_csv_path),
        "confusion_matrix": relative_to_repo(confusion_path),
        "overall_metrics": metrics,
        "num_four_class_regions": len(y_true),
        "num_nicht_bewertbar_inferred": metrics_payload["num_nicht_bewertbar_inferred"],
        "test_used": False,
    }


def write_prediction_csv(path: Path, rows: list[dict[str, Any]], labels: list[str]) -> None:
    fieldnames = [
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
        "pred_label",
        "confidence",
    ] + [f"prob_{label}" for label in labels] + ["correct"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_metrics_csv(
    path: Path,
    metrics: dict[str, Any],
    per_class: list[dict[str, Any]],
    num_four_class_regions: int,
    num_special_inferred: int,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "class_name", "metric", "value"])
        writer.writeheader()
        writer.writerow({"scope": "overall", "class_name": "", "metric": "num_four_class_regions", "value": num_four_class_regions})
        writer.writerow({"scope": "overall", "class_name": "", "metric": "num_nicht_bewertbar_inferred", "value": num_special_inferred})
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            writer.writerow({"scope": "overall", "class_name": "", "metric": key, "value": metrics[key]})
        for row in per_class:
            for key in ("precision", "recall", "f1", "support"):
                writer.writerow({"scope": "per_class", "class_name": row["class_name"], "metric": key, "value": row[key]})


def main() -> int:
    args = parse_args()
    try:
        context = prepare_context(args)
        if args.check_model:
            result = check_model(context, args)
        elif args.allow_evaluate:
            result = predict_regions(context, args)
        else:
            result = dry_run(context, args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
