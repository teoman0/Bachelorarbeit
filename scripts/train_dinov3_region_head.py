"""Train a small frozen-DINOv3 region head on CVAT rectangle crops.

This workflow uses the existing CVAT region table and the existing DINOv3
partial fine-tuning checkpoint. The DINOv3 backbone remains frozen; only a
small local region head is trainable. The test split is never used.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image, ImageEnhance, ImageOps
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluate_dinov3_regions import (  # noqa: E402
    RegionRecord,
    crop_region_image,
    draw_overlay,
    load_checkpoint_metadata,
    read_region_table,
    relative_to_repo,
    resolve_repo_path,
    safe_slug,
    top_confusions,
)
from scripts.train_dinov3_head import (  # noqa: E402
    extract_features,
    load_dinov3,
    metric_payload,
    write_confusion,
)


DEFAULT_CONFIG = "configs/experiments/dinov3_region_head.yaml"
VALID_CROP_MODES = {"stretch_resize", "pad_square"}


@dataclass(frozen=True)
class RegionHeadContext:
    config: dict[str, Any]
    config_path: Path
    region_table: Path
    checkpoint_path: Path
    manual_root: Path | None
    images_dir: Path | None
    output_dir: Path
    image_size: tuple[int, int]
    class_to_index: dict[str, int]
    index_to_class: dict[int, str]
    train_records: list[RegionRecord]
    val_records: list[RegionRecord]
    ignored_train_records: list[RegionRecord]
    ignored_val_records: list[RegionRecord]
    source_split_counts: dict[str, int]
    crop_mode: str
    context_margin: float


class RegionCropDataset:
    def __init__(
        self,
        records: list[RegionRecord],
        *,
        images_dir: Path,
        image_size: tuple[int, int],
        class_to_index: dict[str, int],
        crop_mode: str,
        context_margin: float,
        augment: bool,
        augmentation_config: dict[str, Any],
    ) -> None:
        self.records = records
        self.crop_context = SimpleNamespace(images_dir=images_dir, image_size=image_size)
        self.class_to_index = class_to_index
        self.crop_mode = crop_mode
        self.context_margin = context_margin
        self.augment = augment
        self.augmentation_config = augmentation_config

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int, RegionRecord]:
        record = self.records[index]
        image = crop_region_image(record, self.crop_context, self.crop_mode, self.context_margin)
        if self.augment:
            image = augment_train_image(image, self.augmentation_config)
        return image, self.class_to_index[record.mapped_label], record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config.")
    parser.add_argument("--manual-root", default=None, help="Local manual_all root for smoke-test/training.")
    parser.add_argument("--region-table", default=None, help="Override region annotation CSV.")
    parser.add_argument("--checkpoint", default=None, help="Override DINOv3 checkpoint path.")
    parser.add_argument("--dry-run", action="store_true", help="Report region counts only; write no files.")
    parser.add_argument("--check-model", action="store_true", help="Load frozen backbone and instantiate region head; no training.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny train/val forward-backward check; no checkpoints.")
    parser.add_argument("--allow-training", action="store_true", help="Start the real local region-head training run.")
    parser.add_argument("--allow-download", action="store_true", help="Allow Transformers to download DINOv3 weights.")
    parser.add_argument("--crop-mode", choices=sorted(VALID_CROP_MODES), default=None)
    parser.add_argument("--context-margin", type=float, default=None)
    parser.add_argument("--include-nicht-bewertbar", action="store_true", help="Report special-label rows as ignored; never part of 4-class loss.")
    parser.add_argument("--max-smoke-samples", type=int, default=None)
    parser.add_argument("--batch-override", type=int, default=None)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument(
        "--export-region-images",
        action="store_true",
        help="After --allow-training, export validation region crops locally.",
    )
    parser.add_argument(
        "--export-overlays",
        action="store_true",
        help="After --allow-training, export validation ground-truth/prediction overlays locally.",
    )
    parser.add_argument(
        "--max-visualization-images",
        type=int,
        default=None,
        help="Limit source images used for validation visualization exports.",
    )
    args = parser.parse_args()

    active_modes = [args.dry_run, args.check_model, args.smoke_test, args.allow_training]
    if sum(bool(value) for value in active_modes) > 1:
        parser.error("Choose only one mode: --dry-run, --check-model, --smoke-test, or --allow-training")
    if not any(active_modes):
        args.dry_run = True
    if args.context_margin is not None and args.context_margin < 0.0:
        parser.error("--context-margin must be non-negative")
    if args.max_smoke_samples is not None and args.max_smoke_samples < 1:
        parser.error("--max-smoke-samples must be at least 1")
    if args.batch_override is not None and args.batch_override < 1:
        parser.error("--batch-override must be at least 1")
    if args.epochs_override is not None and args.epochs_override < 1:
        parser.error("--epochs-override must be at least 1")
    if args.max_visualization_images is not None and args.max_visualization_images < 1:
        parser.error("--max-visualization-images must be at least 1")
    if (args.export_region_images or args.export_overlays) and not args.allow_training:
        parser.error("Visualization exports require --allow-training")
    return args


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def resolve_manual_root(config: dict[str, Any], override: str | None, *, required: bool) -> Path | None:
    if override:
        root = Path(override).expanduser()
    else:
        env_name = str(config.get("inputs", {}).get("manual_root_env", "BMW25_MANUAL_ALL_ROOT"))
        env_value = os.environ.get(env_name)
        root = Path(env_value).expanduser() if env_value else None
    if root is None:
        if required:
            raise ValueError("Manual root is required for --smoke-test or --allow-training.")
        return None
    if not root.exists():
        raise FileNotFoundError(f"Manual root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Manual root is not a directory: {root}")
    return root


def prepare_context(args: argparse.Namespace) -> RegionHeadContext:
    config_path = resolve_repo_path(args.config)
    config = load_yaml(config_path)
    if str(config.get("model_family")) != "dinov3_region_head":
        raise ValueError(f"Expected model_family='dinov3_region_head', got {config.get('model_family')!r}")

    inputs = config.get("inputs", {})
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})
    region_table = resolve_repo_path(args.region_table or str(inputs["region_annotations"]))
    checkpoint_path = resolve_repo_path(args.checkpoint or str(inputs["checkpoint"]))
    checkpoint = load_checkpoint_metadata(checkpoint_path)
    class_to_index = class_mapping_from_config_and_checkpoint(config, checkpoint)
    index_to_class = {index: label for label, index in class_to_index.items()}
    image_size_raw = data_cfg.get("image_size", [224, 224])
    crop_mode = str(args.crop_mode or data_cfg.get("crop_mode", "stretch_resize"))
    if crop_mode not in VALID_CROP_MODES:
        raise ValueError(f"Unsupported crop_mode: {crop_mode}")
    context_margin = float(args.context_margin if args.context_margin is not None else data_cfg.get("context_margin", 0.0))
    if context_margin < 0.0:
        raise ValueError("--context-margin must be non-negative")

    rows = read_region_table(region_table)
    train_split = str(data_cfg.get("train_split", "train"))
    val_split = str(data_cfg.get("val_split", "val"))
    special_label = str(data_cfg.get("special_label", "Nicht_bewertbar"))
    train_records, ignored_train = filter_training_regions(rows, train_split, class_to_index, special_label)
    val_records, ignored_val = filter_training_regions(rows, val_split, class_to_index, special_label)
    if not train_records:
        raise ValueError("No train regions remain after filtering")
    if not val_records:
        raise ValueError("No val regions remain after filtering")

    required_images = bool(args.smoke_test or args.allow_training)
    manual_root = resolve_manual_root(config, args.manual_root, required=required_images)
    images_dir = manual_root / str(inputs.get("images_dir", "images")) if manual_root is not None else None
    output_dir = resolve_repo_path(str(output_cfg.get("output_dir", "outputs/region_analysis/dinov3_region_head_bmw25_seed42")))

    return RegionHeadContext(
        config=config,
        config_path=config_path,
        region_table=region_table,
        checkpoint_path=checkpoint_path,
        manual_root=manual_root,
        images_dir=images_dir,
        output_dir=output_dir,
        image_size=(int(image_size_raw[0]), int(image_size_raw[1])),
        class_to_index=class_to_index,
        index_to_class=index_to_class,
        train_records=train_records,
        val_records=val_records,
        ignored_train_records=ignored_train,
        ignored_val_records=ignored_val,
        source_split_counts=count_splits(rows),
        crop_mode=crop_mode,
        context_margin=context_margin,
    )


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


def filter_training_regions(
    rows: list[RegionRecord],
    split: str,
    class_to_index: dict[str, int],
    special_label: str,
) -> tuple[list[RegionRecord], list[RegionRecord]]:
    selected: list[RegionRecord] = []
    ignored: list[RegionRecord] = []
    for row in rows:
        if row.split != split or not row.matched_manifest or row.split == "test":
            continue
        if row.is_global_class and row.mapped_label in class_to_index and row.mapped_label != special_label:
            selected.append(row)
        else:
            ignored.append(row)
    return selected, ignored


def count_splits(rows: list[RegionRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.split] = counts.get(row.split, 0) + 1
    return counts


def class_distribution(rows: list[RegionRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.mapped_label] = counts.get(row.mapped_label, 0) + 1
    return dict(sorted(counts.items()))


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        return


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


def cuda_info() -> dict[str, Any]:
    import torch

    cuda_available = torch.cuda.is_available()
    return {
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def device_from_config(context: RegionHeadContext) -> Any:
    import torch

    raw_device = context.config.get("training", {}).get("device", 0)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{raw_device}")
    return torch.device("cpu")


def allow_download(context: RegionHeadContext, args: argparse.Namespace) -> bool:
    return bool(args.allow_download or context.config.get("training", {}).get("allow_download", False))


def selected_model_id(context: RegionHeadContext, checkpoint: dict[str, Any]) -> str:
    return str(context.config.get("model_id") or checkpoint.get("model_id"))


def load_frozen_backbone(context: RegionHeadContext, args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any], Any]:
    checkpoint = load_checkpoint_metadata(context.checkpoint_path)
    device = device_from_config(context)
    processor, backbone = load_dinov3(
        selected_model_id(context, checkpoint),
        allow_model_download=allow_download(context, args),
        device=device,
    )
    missing, unexpected = backbone.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Backbone checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return processor, backbone, checkpoint, device


def build_region_head(context: RegionHeadContext, feature_dim: int, device: Any) -> Any:
    import torch.nn as nn

    head_cfg = context.config.get("head", {})
    hidden_dim = int(head_cfg.get("hidden_dim", 128))
    dropout = float(head_cfg.get("dropout", 0.2))
    return nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, len(context.class_to_index)),
    ).to(device)


def parameter_count(module: Any, *, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return sum(parameter.numel() for parameter in module.parameters())


def effective_batch_size(context: RegionHeadContext, args: argparse.Namespace) -> int:
    return int(args.batch_override or context.config.get("training", {}).get("batch_size", 16))


def effective_epochs(context: RegionHeadContext, args: argparse.Namespace) -> int:
    return int(args.epochs_override or context.config.get("training", {}).get("epochs", 100))


def effective_patience(context: RegionHeadContext) -> int:
    return int(context.config.get("training", {}).get("patience", 15))


def smoke_sample_count(context: RegionHeadContext, args: argparse.Namespace) -> int:
    return int(args.max_smoke_samples or context.config.get("training", {}).get("smoke_max_samples_per_split", 2))


def augment_train_image(image: Any, augmentation_config: dict[str, Any]) -> Any:
    if random.random() < float(augmentation_config.get("horizontal_flip_probability", 0.0)):
        image = ImageOps.mirror(image)
    if random.random() < float(augmentation_config.get("vertical_flip_probability", 0.0)):
        image = ImageOps.flip(image)
    brightness = float(augmentation_config.get("brightness", 0.0))
    if brightness:
        factor = 1.0 + random.uniform(-brightness, brightness)
        image = ImageEnhance.Brightness(image).enhance(factor)
    contrast = float(augmentation_config.get("contrast", 0.0))
    if contrast:
        factor = 1.0 + random.uniform(-contrast, contrast)
        image = ImageEnhance.Contrast(image).enhance(factor)
    return image


def make_loader(context: RegionHeadContext, records: list[RegionRecord], *, processor: Any, batch_size: int, augment: bool, shuffle: bool) -> Any:
    import torch
    from torch.utils.data import DataLoader

    if context.images_dir is None:
        raise ValueError("manual_root/images_dir is required to load region crops")
    dataset = RegionCropDataset(
        records=records,
        images_dir=context.images_dir,
        image_size=context.image_size,
        class_to_index=context.class_to_index,
        crop_mode=context.crop_mode,
        context_margin=context.context_margin,
        augment=augment,
        augmentation_config=context.config.get("augmentation", {}),
    )

    def collate(batch: list[tuple[Any, int, RegionRecord]]) -> tuple[dict[str, Any], Any, list[RegionRecord]]:
        images = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        model_inputs = processor(images=images, return_tensors="pt")
        records_batch = [item[2] for item in batch]
        return model_inputs, labels, records_batch

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(context.config.get("training", {}).get("num_workers", 0)),
        collate_fn=collate,
    )


def move_inputs(inputs: dict[str, Any], labels: Any, device: Any) -> tuple[dict[str, Any], Any]:
    return {key: value.to(device) for key, value in inputs.items()}, labels.to(device)


def amp_autocast(device: Any, enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    import torch

    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def create_grad_scaler(device: Any, enabled: bool) -> Any:
    import torch

    if hasattr(torch, "amp"):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def class_weight_tensor(context: RegionHeadContext, device: Any) -> Any | None:
    import torch

    if not bool(context.config.get("training", {}).get("class_weights", True)):
        return None
    counts = class_distribution(context.train_records)
    total = sum(counts.values())
    num_classes = len(context.class_to_index)
    weights = []
    for index in sorted(context.index_to_class):
        label = context.index_to_class[index]
        count = counts.get(label, 0)
        if count <= 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate_head(
    *,
    backbone: Any,
    head: Any,
    loader: Any,
    context: RegionHeadContext,
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[int]], list[dict[str, Any]]]:
    import torch

    labels = [context.index_to_class[index] for index in sorted(context.index_to_class)]
    backbone.eval()
    head.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for inputs, labels_tensor, records in loader:
            inputs, labels_tensor = move_inputs(inputs, labels_tensor, device)
            outputs = backbone(**inputs)
            features, representation = extract_features(
                outputs,
                str(context.config.get("feature_representation", "pooler_output_or_cls_token")),
            )
            logits = head(features)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1).values
            for row_index, record in enumerate(records):
                true_index = int(labels_tensor[row_index].detach().cpu().item())
                pred_index = int(predictions[row_index].detach().cpu().item())
                y_true.append(true_index)
                y_pred.append(pred_index)
                row = {
                    "region_id": record.region_id,
                    "source_image": record.source_image,
                    "split": record.split,
                    "original_label": record.original_label,
                    "mapped_label": record.mapped_label,
                    "is_global_class": record.is_global_class,
                    "x_min": record.raw.get("x_min", record.x_min),
                    "y_min": record.raw.get("y_min", record.y_min),
                    "x_max": record.raw.get("x_max", record.x_max),
                    "y_max": record.raw.get("y_max", record.y_max),
                    "true_label": context.index_to_class[true_index],
                    "pred_label": context.index_to_class[pred_index],
                    "predicted_label": context.index_to_class[pred_index],
                    "confidence": float(confidences[row_index].detach().cpu().item()),
                    "crop_mode": context.crop_mode,
                    "context_margin": context.context_margin,
                    "feature_representation": representation,
                    "correct": pred_index == true_index,
                }
                probs = probabilities[row_index].detach().cpu().tolist()
                for label, value in zip(labels, probs, strict=True):
                    row[f"prob_{label}"] = float(value)
                prediction_rows.append(row)
    metrics, per_class, confusion = metric_payload(y_true, y_pred, context.index_to_class)
    return metrics, per_class, confusion, prediction_rows


def base_metadata(context: RegionHeadContext, mode: str) -> dict[str, Any]:
    return {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "script": "scripts/train_dinov3_region_head.py",
        "mode": mode,
        "experiment_name": context.config.get("experiment_name"),
        "model_family": context.config.get("model_family"),
        "model_id": context.config.get("model_id"),
        "checkpoint": relative_to_repo(context.checkpoint_path),
        "region_table": relative_to_repo(context.region_table),
        "dataset_splits_used": ["train", "val"],
        "test_usage_note": "Test regions are excluded and are never loaded by this workflow.",
        "test_rows_in_region_table": context.source_split_counts.get("test", 0),
        "nicht_bewertbar_excluded": True,
        "crop_mode": context.crop_mode,
        "context_margin": context.context_margin,
        "image_size": list(context.image_size),
        "class_to_index": context.class_to_index,
        "train_regions_4class": len(context.train_records),
        "val_regions_4class": len(context.val_records),
        "ignored_train_regions": len(context.ignored_train_records),
        "ignored_val_regions": len(context.ignored_val_records),
        "train_class_distribution": class_distribution(context.train_records),
        "val_class_distribution": class_distribution(context.val_records),
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "artifact_policy": context.config.get("artifact_policy", {}),
    }


def run_dry_run(context: RegionHeadContext) -> dict[str, Any]:
    metadata = base_metadata(context, "dry_run")
    metadata.update(
        {
            "outputs_written": False,
            "test_regions_excluded": True,
            "nicht_bewertbar_excluded_from_4class": True,
        }
    )
    return metadata


def run_check_model(context: RegionHeadContext, args: argparse.Namespace) -> dict[str, Any]:
    processor, backbone, checkpoint, device = load_frozen_backbone(context, args)
    feature_dim = int(checkpoint.get("feature_info", {}).get("feature_dim", context.config.get("head", {}).get("input_dim", 768)))
    head = build_region_head(context, feature_dim, device)
    backbone_trainable = parameter_count(backbone, trainable_only=True)
    head_trainable = parameter_count(head, trainable_only=True)
    return {
        "checkpoint_loaded": True,
        "checkpoint_path": relative_to_repo(context.checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_name": selected_model_id(context, checkpoint),
        "processor_class": type(processor).__name__,
        "backbone_class": type(backbone).__name__,
        "backbone_frozen": backbone_trainable == 0,
        "trainable_parameters_only_in_head": backbone_trainable == 0 and head_trainable > 0,
        "feature_dim": feature_dim,
        "head": str(context.config.get("head", {}).get("type", "mlp")),
        "backbone_parameters_total": parameter_count(backbone),
        "backbone_parameters_trainable": backbone_trainable,
        "head_parameters_total": parameter_count(head),
        "head_parameters_trainable": head_trainable,
        "device": str(device),
        **cuda_info(),
        "crop_mode": context.crop_mode,
        "context_margin": context.context_margin,
        "outputs_written": False,
    }


def run_smoke_test(context: RegionHeadContext, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn

    set_seed(int(context.config.get("training", {}).get("seed", 42)))
    processor, backbone, checkpoint, device = load_frozen_backbone(context, args)
    feature_dim = int(checkpoint.get("feature_info", {}).get("feature_dim", 768))
    head = build_region_head(context, feature_dim, device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(context, device))
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(context.config.get("training", {}).get("learning_rate", 0.001)),
        weight_decay=float(context.config.get("training", {}).get("weight_decay", 0.01)),
    )
    amp_enabled = bool(context.config.get("training", {}).get("amp", True) and device.type == "cuda")
    scaler = create_grad_scaler(device, amp_enabled)
    sample_count = smoke_sample_count(context, args)
    batch_size = min(sample_count, effective_batch_size(context, args))
    train_loader = make_loader(
        context,
        context.train_records[:sample_count],
        processor=processor,
        batch_size=batch_size,
        augment=True,
        shuffle=False,
    )
    inputs, labels, records = next(iter(train_loader))
    inputs, labels = move_inputs(inputs, labels, device)
    optimizer.zero_grad(set_to_none=True)
    backbone.eval()
    head.train()
    with amp_autocast(device, amp_enabled):
        with torch.no_grad():
            outputs = backbone(**inputs)
            features, representation = extract_features(
                outputs,
                str(context.config.get("feature_representation", "pooler_output_or_cls_token")),
            )
        logits = head(features)
        loss = criterion(logits, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    val_loader = make_loader(
        context,
        context.val_records[:sample_count],
        processor=processor,
        batch_size=batch_size,
        augment=False,
        shuffle=False,
    )
    metrics, per_class, confusion, _predictions = evaluate_head(
        backbone=backbone,
        head=head,
        loader=val_loader,
        context=context,
        device=device,
    )
    return {
        "checkpoint_loaded": True,
        "backbone_frozen": parameter_count(backbone, trainable_only=True) == 0,
        "trainable_parameters_only_in_head": parameter_count(backbone, trainable_only=True) == 0,
        "feature_dim": feature_dim,
        "feature_representation": representation,
        "device": str(device),
        **cuda_info(),
        "train_samples_loaded": len(records),
        "val_samples_loaded": min(sample_count, len(context.val_records)),
        "loss": float(loss.detach().cpu().item()),
        "val_metrics_on_tiny_subset_not_interpretable": metrics,
        "per_class_on_tiny_subset_not_interpretable": per_class,
        "confusion_on_tiny_subset_not_interpretable": confusion,
        "outputs_written": False,
        "checkpoint_written": False,
        "test_used": False,
    }


def run_training(context: RegionHeadContext, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn

    set_seed(int(context.config.get("training", {}).get("seed", 42)))
    processor, backbone, checkpoint, device = load_frozen_backbone(context, args)
    feature_dim = int(checkpoint.get("feature_info", {}).get("feature_dim", 768))
    head = build_region_head(context, feature_dim, device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(context, device))
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(context.config.get("training", {}).get("learning_rate", 0.001)),
        weight_decay=float(context.config.get("training", {}).get("weight_decay", 0.01)),
    )
    amp_enabled = bool(context.config.get("training", {}).get("amp", True) and device.type == "cuda")
    scaler = create_grad_scaler(device, amp_enabled)
    train_loader = make_loader(
        context,
        context.train_records,
        processor=processor,
        batch_size=effective_batch_size(context, args),
        augment=True,
        shuffle=True,
    )
    val_loader = make_loader(
        context,
        context.val_records,
        processor=processor,
        batch_size=effective_batch_size(context, args),
        augment=False,
        shuffle=False,
    )
    output_cfg = context.config.get("output", {})
    checkpoints_dir = context.output_dir / str(output_cfg.get("checkpoints_dir", "checkpoints"))
    context.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    labels = [context.index_to_class[index] for index in sorted(context.index_to_class)]
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_predictions: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, effective_epochs(context, args) + 1):
        head.train()
        train_loss_sum = 0.0
        train_count = 0
        for inputs, labels_tensor, _records in train_loader:
            inputs, labels_tensor = move_inputs(inputs, labels_tensor, device)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device, amp_enabled):
                with torch.no_grad():
                    outputs = backbone(**inputs)
                    features, _representation = extract_features(
                        outputs,
                        str(context.config.get("feature_representation", "pooler_output_or_cls_token")),
                    )
                logits = head(features)
                loss = criterion(logits, labels_tensor)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(labels_tensor.shape[0])
            train_loss_sum += float(loss.detach().cpu().item()) * batch_size
            train_count += batch_size

        val_metrics, per_class, confusion, predictions = evaluate_head(
            backbone=backbone,
            head=head,
            loader=val_loader,
            context=context,
            device=device,
        )
        train_loss = train_loss_sum / train_count if train_count else 0.0
        improved = float(val_metrics["macro_f1"]) > best_macro_f1
        if improved:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
            save_head_checkpoint(
                checkpoints_dir / str(output_cfg.get("best_checkpoint", "best_region_head.pt")),
                head=head,
                context=context,
                checkpoint=checkpoint,
                epoch=epoch,
                metrics=val_metrics,
                feature_dim=feature_dim,
            )
            best_payload = write_validation_outputs(context, labels, val_metrics, per_class, confusion, predictions, best_epoch)
            best_predictions = predictions
        else:
            epochs_without_improvement += 1

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "best_epoch": best_epoch,
            "best_macro_f1": best_macro_f1,
            "epochs_without_improvement": epochs_without_improvement,
        }
        epoch_rows.append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False), flush=True)
        if epochs_without_improvement >= effective_patience(context):
            break

    save_head_checkpoint(
        checkpoints_dir / str(output_cfg.get("last_checkpoint", "last_region_head.pt")),
        head=head,
        context=context,
        checkpoint=checkpoint,
        epoch=int(epoch_rows[-1]["epoch"]),
        metrics=epoch_rows[-1],
        feature_dim=feature_dim,
    )
    write_training_log(context.output_dir / str(output_cfg.get("training_log", "training_log.csv")), epoch_rows)
    write_training_log(context.output_dir / str(output_cfg.get("training_metrics", "training_metrics.csv")), epoch_rows)
    visualization_result: dict[str, Any] | None = None
    if args.export_region_images or args.export_overlays:
        visualization_result = export_training_visualizations(
            context=context,
            prediction_rows=best_predictions,
            crop_mode=context.crop_mode,
            context_margin=context.context_margin,
            export_region_images=bool(args.export_region_images),
            export_overlays=bool(args.export_overlays),
            max_visualization_images=args.max_visualization_images,
        )
    result = {
        "training_completed": True,
        "epochs_completed": int(epoch_rows[-1]["epoch"]),
        "early_stopped": epoch_rows[-1]["epochs_without_improvement"] >= effective_patience(context),
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "best_metrics": best_payload["overall_metrics"] if best_payload else None,
        "test_used": False,
        "local_outputs": {
            "output_dir": relative_to_repo(context.output_dir),
            "best_checkpoint": relative_to_repo(checkpoints_dir / str(output_cfg.get("best_checkpoint", "best_region_head.pt"))),
            "last_checkpoint": relative_to_repo(checkpoints_dir / str(output_cfg.get("last_checkpoint", "last_region_head.pt"))),
            "predictions_val": relative_to_repo(context.output_dir / str(output_cfg.get("predictions_val", "predictions_val.csv"))),
            "metrics_json": relative_to_repo(context.output_dir / str(output_cfg.get("metrics_json", "val_metrics.json"))),
            "metrics_csv": relative_to_repo(context.output_dir / str(output_cfg.get("metrics_csv", "val_metrics.csv"))),
            "confusion_matrix": relative_to_repo(context.output_dir / str(output_cfg.get("confusion_matrix", "confusion_matrix_val.csv"))),
            "training_log": relative_to_repo(context.output_dir / str(output_cfg.get("training_log", "training_log.csv"))),
            "training_metrics": relative_to_repo(context.output_dir / str(output_cfg.get("training_metrics", "training_metrics.csv"))),
        },
        "visualizations": visualization_result,
    }
    write_json(context.output_dir / str(output_cfg.get("run_metadata", "run_metadata.json")), base_metadata(context, "training") | {"training_result": result})
    return result


def write_validation_outputs(
    context: RegionHeadContext,
    labels: list[str],
    metrics: dict[str, Any],
    per_class: list[dict[str, Any]],
    confusion: list[list[int]],
    predictions: list[dict[str, Any]],
    best_epoch: int,
) -> dict[str, Any]:
    output_cfg = context.config.get("output", {})
    predictions_path = context.output_dir / str(output_cfg.get("predictions_val", "predictions_val.csv"))
    metrics_json_path = context.output_dir / str(output_cfg.get("metrics_json", "val_metrics.json"))
    metrics_csv_path = context.output_dir / str(output_cfg.get("metrics_csv", "val_metrics.csv"))
    confusion_path = context.output_dir / str(output_cfg.get("confusion_matrix", "confusion_matrix_val.csv"))
    write_predictions(predictions_path, predictions, labels)
    payload = {
        "overall_metrics": metrics,
        "per_class_metrics": per_class,
        "confusion_matrix": {
            "labels": labels,
            "rows_are_true_labels": True,
            "values": confusion,
        },
        "top_confusions": top_confusions(labels, confusion),
        "best_epoch": best_epoch,
        "checkpoint_metric": "macro_f1",
        "num_val_regions": len(context.val_records),
        "num_ignored_val_regions": len(context.ignored_val_records),
        "test_used": False,
    }
    write_json(metrics_json_path, payload)
    write_metrics_csv(metrics_csv_path, metrics, per_class, len(context.val_records), len(context.ignored_val_records), best_epoch)
    write_confusion(confusion_path, labels, confusion)
    return payload


def save_head_checkpoint(
    path: Path,
    *,
    head: Any,
    context: RegionHeadContext,
    checkpoint: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
    feature_dim: int,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "class_to_index": context.class_to_index,
            "source_checkpoint": relative_to_repo(context.checkpoint_path),
            "source_model_id": checkpoint.get("model_id"),
            "feature_dim": feature_dim,
            "crop_mode": context.crop_mode,
            "context_margin": context.context_margin,
            "epoch": epoch,
            "metrics": metrics,
            "config": context.config,
        },
        path,
    )


def prediction_to_region(row: dict[str, Any]) -> RegionRecord:
    return RegionRecord(
        region_id=str(row["region_id"]),
        source_image=str(row["source_image"]),
        split=str(row["split"]),
        original_label=str(row["original_label"]),
        mapped_label=str(row["mapped_label"]),
        is_global_class=str(row["is_global_class"]).lower() == "true",
        x_min=float(row["x_min"]),
        y_min=float(row["y_min"]),
        x_max=float(row["x_max"]),
        y_max=float(row["y_max"]),
        matched_manifest=True,
        raw={key: str(value) for key, value in row.items()},
    )


def selected_visualization_rows(
    context: RegionHeadContext,
    rows: list[dict[str, Any]],
    max_visualization_images: int | None,
) -> list[dict[str, Any]]:
    source_images = sorted({str(row["source_image"]) for row in rows})
    if max_visualization_images is None:
        configured_max = context.config.get("visualization", {}).get("max_visualization_images")
        max_visualization_images = int(configured_max) if configured_max is not None else len(source_images)
    selected_sources = set(source_images[:max_visualization_images])
    return [row for row in rows if str(row["source_image"]) in selected_sources]


def export_training_visualizations(
    *,
    context: RegionHeadContext,
    prediction_rows: list[dict[str, Any]],
    crop_mode: str,
    context_margin: float,
    export_region_images: bool,
    export_overlays: bool,
    max_visualization_images: int | None,
) -> dict[str, Any]:
    if context.images_dir is None:
        raise ValueError("manual_root/images_dir is required for visualization exports")

    output_cfg = context.config.get("output", {})
    base_dir = context.output_dir / str(output_cfg.get("visualizations_dir", "visualizations"))
    selected_rows = selected_visualization_rows(context, prediction_rows, max_visualization_images)
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        rows_by_source.setdefault(str(row["source_image"]), []).append(row)

    outputs: dict[str, Any] = {
        "visualizations_dir": relative_to_repo(base_dir),
        "source_images_selected": len(rows_by_source),
        "region_images_written": 0,
        "ground_truth_overlays_written": 0,
        "prediction_overlays_written": 0,
        "comparison_images_written": 0,
        "visualization_index_rows": len(selected_rows),
    }
    comparison_paths: dict[str, str] = {}
    crop_paths: dict[str, str] = {}

    if export_overlays:
        gt_dir = base_dir / "ground_truth"
        pred_dir = base_dir / "predictions"
        comparison_dir = base_dir / "comparison"
        gt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        comparison_dir.mkdir(parents=True, exist_ok=True)
        for source_image, rows in rows_by_source.items():
            image_path = context.images_dir / source_image
            with Image.open(image_path) as image:
                original = ImageOps.exif_transpose(image).convert("RGB")
            ground_truth = draw_overlay(original, rows, mode="ground_truth")
            prediction = draw_overlay(original, rows, mode="prediction")
            stem = safe_slug(Path(source_image).stem)
            gt_path = gt_dir / f"{stem}.jpg"
            pred_path = pred_dir / f"{stem}.jpg"
            comparison_path = comparison_dir / f"{stem}.jpg"
            ground_truth.save(gt_path, quality=90)
            prediction.save(pred_path, quality=90)
            comparison = Image.new(
                "RGB",
                (ground_truth.width + prediction.width, max(ground_truth.height, prediction.height)),
            )
            comparison.paste(ground_truth, (0, 0))
            comparison.paste(prediction, (ground_truth.width, 0))
            comparison.save(comparison_path, quality=90)
            comparison_paths[source_image] = relative_to_repo(comparison_path)
            outputs["ground_truth_overlays_written"] += 1
            outputs["prediction_overlays_written"] += 1
            outputs["comparison_images_written"] += 1

    if export_region_images:
        crops_dir = base_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for row in selected_rows:
            record = prediction_to_region(row)
            crop = crop_region_image(record, context, crop_mode, context_margin)
            correct = "na" if str(row.get("correct", "")) == "" else str(row["correct"]).lower()
            filename = "__".join(
                [
                    safe_slug(str(row["region_id"])),
                    f"true-{safe_slug(str(row['mapped_label']))}",
                    f"pred-{safe_slug(str(row['pred_label']))}",
                    f"conf-{float(row['confidence']):.3f}",
                    f"correct-{safe_slug(correct)}",
                ]
            )
            crop_path = crops_dir / f"{filename}.jpg"
            crop.save(crop_path, quality=90)
            crop_paths[str(row["region_id"])] = relative_to_repo(crop_path)
            outputs["region_images_written"] += 1

    index_path = base_dir / str(output_cfg.get("visualization_index", "region_visualization_index.csv"))
    write_visualization_index(index_path, selected_rows, comparison_paths, crop_paths)
    outputs["visualization_index"] = relative_to_repo(index_path)
    return outputs


def write_visualization_index(
    path: Path,
    rows: list[dict[str, Any]],
    comparison_paths: dict[str, str],
    crop_paths: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "region_id",
        "true_label",
        "pred_label",
        "confidence",
        "correct",
        "visualization_path",
        "crop_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            source_image = str(row["source_image"])
            writer.writerow(
                {
                    "image_id": safe_slug(Path(source_image).stem),
                    "region_id": row["region_id"],
                    "true_label": row["mapped_label"],
                    "pred_label": row["pred_label"],
                    "confidence": row["confidence"],
                    "correct": row["correct"],
                    "visualization_path": comparison_paths.get(source_image, ""),
                    "crop_path": crop_paths.get(str(row["region_id"]), ""),
                }
            )


def write_predictions(path: Path, rows: list[dict[str, Any]], labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "true_label",
        "pred_label",
        "predicted_label",
        "confidence",
        "crop_mode",
        "context_margin",
        "feature_representation",
        "correct",
    ] + [f"prob_{label}" for label in labels]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_csv(
    path: Path,
    metrics: dict[str, Any],
    per_class: list[dict[str, Any]],
    num_val_regions: int,
    num_ignored_val_regions: int,
    best_epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "class_name", "metric", "value"])
        writer.writeheader()
        writer.writerow({"scope": "overall", "class_name": "", "metric": "best_epoch", "value": best_epoch})
        writer.writerow({"scope": "overall", "class_name": "", "metric": "num_val_regions", "value": num_val_regions})
        writer.writerow({"scope": "overall", "class_name": "", "metric": "num_ignored_val_regions", "value": num_ignored_val_regions})
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            writer.writerow({"scope": "overall", "class_name": "", "metric": key, "value": metrics[key]})
        for row in per_class:
            for key in ("precision", "recall", "f1", "support"):
                writer.writerow({"scope": "per_class", "class_name": row["class_name"], "metric": key, "value": row[key]})


def write_training_log(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    try:
        context = prepare_context(args)
        if args.check_model:
            result = run_check_model(context, args)
        elif args.smoke_test:
            result = run_smoke_test(context, args)
        elif args.allow_training:
            result = run_training(context, args)
        else:
            result = run_dry_run(context)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
