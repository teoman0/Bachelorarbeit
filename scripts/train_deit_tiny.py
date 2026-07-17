"""Run the DeiT-Tiny from-scratch global classification workflow.

The default mode is a dry-run that validates the config, split manifest, class
mapping, and local train/val file availability. No pretrained weights are used.
Training starts only with --allow-training.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bachelorarbeit.data.split_dataset import (  # noqa: E402
    ManifestRecord,
    build_class_mapping,
    check_local_files,
    filter_split,
    read_split_manifest,
    resize_with_padding,
    resolve_record_path,
    split_distribution,
)
from bachelorarbeit.training.global_training_setup import load_yaml_config, resolve_repo_path  # noqa: E402


@dataclass(frozen=True)
class DeiTRunContext:
    config: dict[str, Any]
    config_path: Path
    dataset_root: Path
    manifest_path: Path
    records: list[ManifestRecord]
    train_records: list[ManifestRecord]
    val_records: list[ManifestRecord]
    class_to_index: dict[str, int]
    index_to_class: dict[int, str]
    output_dir: Path
    image_size: tuple[int, int]


class ManifestTensorDataset:
    def __init__(
        self,
        records: list[ManifestRecord],
        dataset_root: Path,
        class_to_index: dict[str, int],
        image_size: tuple[int, int],
    ) -> None:
        self.records = records
        self.dataset_root = dataset_root
        self.class_to_index = class_to_index
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int, ManifestRecord]:
        import torch

        record = self.records[index]
        path = resolve_record_path(self.dataset_root, record)
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = resize_with_padding(image, self.image_size)
            array = np.array(image, dtype=np.float32, copy=True) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, self.class_to_index[record.label], record


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
    packages = ["torch", "torchvision", "timm", "numpy", "Pillow", "PyYAML"]
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/deit_tiny_scratch.yaml")
    parser.add_argument("--dataset-root", required=True, help="Local dataset root; never written to versioned files.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config, manifest, train/val files, and metadata only.")
    parser.add_argument("--check-model", action="store_true", help="Create the timm DeiT-Tiny model without pretrained weights.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny train/val forward/backward check.")
    parser.add_argument("--allow-training", action="store_true", help="Start the real from-scratch training run.")
    parser.add_argument("--batch-override", type=int, default=None)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--max-smoke-samples", type=int, default=None)
    args = parser.parse_args()

    active_modes = [args.dry_run, args.check_model, args.smoke_test, args.allow_training]
    if sum(bool(value) for value in active_modes) > 1:
        parser.error("Choose only one mode: --dry-run, --check-model, --smoke-test, or --allow-training")
    if not any(active_modes):
        args.dry_run = True
    if args.batch_override is not None and args.batch_override < 1:
        parser.error("--batch-override must be at least 1")
    if args.epochs_override is not None and args.epochs_override < 1:
        parser.error("--epochs-override must be at least 1")
    if args.max_smoke_samples is not None and args.max_smoke_samples < 1:
        parser.error("--max-smoke-samples must be at least 1")
    return args


def prepare_context(args: argparse.Namespace) -> DeiTRunContext:
    config_path = resolve_repo_path(args.config)
    config = load_yaml_config(config_path)
    model_family = str(config.get("model_family", ""))
    if model_family != "deit_tiny_scratch":
        raise ValueError(f"Expected model_family='deit_tiny_scratch', got {model_family!r}")
    if bool(config.get("pretrained", False)):
        raise ValueError("DeiT-Tiny scratch config must keep pretrained=false")

    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    manifest_path = resolve_repo_path(str(config["split_manifest"]))
    records = read_split_manifest(manifest_path)
    train_records = filter_split(records, "train")
    val_records = filter_split(records, "val")
    if not train_records or not val_records:
        raise ValueError("Both train and val records are required for DeiT-Tiny workflow")

    class_to_index = build_class_mapping(records)
    index_to_class = {index: label for label, index in class_to_index.items()}
    image_size = tuple(int(value) for value in config.get("image_size", [224, 224]))
    output_dir = resolve_repo_path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    return DeiTRunContext(
        config=config,
        config_path=config_path,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        records=records,
        train_records=train_records,
        val_records=val_records,
        class_to_index=class_to_index,
        index_to_class=index_to_class,
        output_dir=output_dir,
        image_size=(image_size[0], image_size[1]),
    )


def effective_batch_size(context: DeiTRunContext, args: argparse.Namespace) -> int:
    return int(args.batch_override or context.config.get("batch_size", 16))


def effective_epochs(context: DeiTRunContext, args: argparse.Namespace) -> int:
    return int(args.epochs_override or context.config.get("epochs", 150))


def max_smoke_samples(context: DeiTRunContext, args: argparse.Namespace) -> int:
    training_cfg = context.config.get("training", {})
    return int(args.max_smoke_samples or training_cfg.get("smoke_max_samples_per_split", 2))


def device_from_config(context: DeiTRunContext) -> Any:
    import torch

    raw_device = context.config.get("training", {}).get("device", 0)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{raw_device}")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_variant(context: DeiTRunContext) -> str:
    return str(context.config.get("model_variant") or context.config.get("deit", {}).get("model_variant"))


def create_deit_model(context: DeiTRunContext, device: Any) -> Any:
    try:
        import timm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "timm is not installed in the active Python environment."
        ) from exc
    model = timm.create_model(
        model_variant(context),
        pretrained=False,
        num_classes=len(context.class_to_index),
    )
    model.to(device)
    return model


def count_parameters(model: Any) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"parameters_total": int(total), "parameters_trainable": int(trainable)}


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def split_file_checks(context: DeiTRunContext) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for split, records in (("train", context.train_records), ("val", context.val_records)):
        file_check = check_local_files(records, context.dataset_root)
        checks[split] = {
            "checked": file_check.checked,
            "existing": file_check.existing,
            "missing": file_check.missing,
            "missing_examples": list(file_check.missing_examples),
        }
        if file_check.missing:
            raise FileNotFoundError(
                f"Missing local {split} images: {file_check.missing} of {file_check.checked}; "
                f"examples={list(file_check.missing_examples)}"
            )
    return checks


def base_metadata(context: DeiTRunContext, args: argparse.Namespace, mode: str) -> dict[str, Any]:
    checks = split_file_checks(context)
    config = context.config
    return {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "script": "scripts/train_deit_tiny.py",
        "mode": mode,
        "experiment_name": config.get("experiment_name"),
        "model_family": config.get("model_family"),
        "model_variant": model_variant(context),
        "model_source": config.get("model_source") or config.get("deit", {}).get("source"),
        "pretrained": bool(config.get("pretrained", False)),
        "config_path": relative_to_repo(context.config_path),
        "split_manifest": relative_to_repo(context.manifest_path),
        "dataset_root_recorded": False,
        "dataset_splits_used": ["train", "val"],
        "test_usage_note": "The test split is reserved for final evaluation and is not loaded by this workflow.",
        "seed": config.get("seed"),
        "image_size": list(context.image_size),
        "batch_size": effective_batch_size(context, args),
        "batch_size_fallback": config.get("batch_size_fallback"),
        "epochs": effective_epochs(context, args),
        "optimizer": config.get("optimizer"),
        "learning_rate": config.get("learning_rate"),
        "weight_decay": config.get("weight_decay"),
        "checkpoint_metric": config.get("checkpoint_metric"),
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "class_to_index": context.class_to_index,
        "split_distribution": split_distribution(context.records),
        "local_file_check_train_val_only": checks,
        "artifact_policy": config.get("artifact_policy", {}),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_metadata(context: DeiTRunContext, args: argparse.Namespace, mode: str, extra: dict[str, Any] | None = None) -> Path:
    payload = base_metadata(context, args, mode)
    if extra:
        payload.update(extra)
    path = context.output_dir / f"{mode}_metadata.json"
    write_json(path, payload)
    return path


def collate_batch(batch: list[tuple[Any, int, ManifestRecord]], device: Any) -> tuple[Any, Any, list[ManifestRecord]]:
    import torch

    images = torch.stack([item[0] for item in batch], dim=0).to(device)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
    records = [item[2] for item in batch]
    return images, labels, records


def make_data_loader(
    records: list[ManifestRecord],
    context: DeiTRunContext,
    *,
    batch_size: int,
    shuffle: bool,
) -> Any:
    from torch.utils.data import DataLoader

    dataset = ManifestTensorDataset(
        records=records,
        dataset_root=context.dataset_root,
        class_to_index=context.class_to_index,
        image_size=context.image_size,
    )
    workers = int(context.config.get("training", {}).get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=lambda batch: batch,
    )


def metric_payload(y_true: list[int], y_pred: list[int], index_to_class: dict[int, str]) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[int]]]:
    labels = [index_to_class[index] for index in sorted(index_to_class)]
    matrix = [[0 for _ in labels] for _ in labels]
    for true_index, pred_index in zip(y_true, y_pred, strict=True):
        matrix[true_index][pred_index] += 1

    per_class = []
    true_positive_total = 0
    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        predicted_count = sum(row[index] for row in matrix)
        support = sum(matrix[index])
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        true_positive_total += true_positive
        per_class.append(
            {
                "class_name": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    recalls = [row["recall"] for row in per_class if row["support"] > 0]
    overall = {
        "num_rows": len(y_true),
        "accuracy": true_positive_total / len(y_true) if y_true else 0.0,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(row["f1"] for row in per_class) / len(per_class) if per_class else 0.0,
    }
    return overall, per_class, matrix


def evaluate(
    *,
    model: Any,
    loader: Any,
    context: DeiTRunContext,
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[int]], list[dict[str, Any]]]:
    import torch

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    labels = [context.index_to_class[index] for index in sorted(context.index_to_class)]

    with torch.no_grad():
        for batch in loader:
            images, labels_tensor, records = collate_batch(batch, device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            for row_index, record in enumerate(records):
                true_index = int(labels_tensor[row_index].detach().cpu().item())
                pred_index = int(predictions[row_index].detach().cpu().item())
                y_true.append(true_index)
                y_pred.append(pred_index)
                row = {
                    "image_id": record.image_id,
                    "relative_path": record.relative_path,
                    "split": record.split,
                    "true_label": context.index_to_class[true_index],
                    "predicted_label": context.index_to_class[pred_index],
                    "model_name": model_variant(context),
                    "config_id": str(context.config.get("experiment_name")),
                    "seed": str(context.config.get("seed")),
                }
                probs = probabilities[row_index].detach().cpu().tolist()
                for label, value in zip(labels, probs, strict=True):
                    row[f"prob_{label}"] = float(value)
                prediction_rows.append(row)

    overall, per_class, confusion = metric_payload(y_true, y_pred, context.index_to_class)
    return overall, per_class, confusion, prediction_rows


def write_predictions(path: Path, rows: list[dict[str, Any]], context: DeiTRunContext) -> None:
    labels = [context.index_to_class[index] for index in sorted(context.index_to_class)]
    fieldnames = [
        "image_id",
        "relative_path",
        "split",
        "true_label",
        "predicted_label",
        "model_name",
        "config_id",
        "seed",
    ] + [f"prob_{label}" for label in labels]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_confusion(path: Path, labels: list[str], matrix: list[list[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label"] + labels)
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label] + row)


def write_epoch_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_dry_run(context: DeiTRunContext, args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = write_metadata(
        context,
        args,
        "dry_run",
        {
            "dry_run_note": "No model is created, no images are loaded, and no training is started.",
            "test_file_availability_checked": False,
        },
    )
    return {"metadata": str(metadata_path), "output_dir": str(context.output_dir)}


def run_check_model(context: DeiTRunContext, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    device = device_from_config(context)
    model = create_deit_model(context, device)
    model.eval()
    parameter_counts = count_parameters(model)
    with torch.no_grad():
        dummy = torch.zeros((1, 3, context.image_size[1], context.image_size[0]), device=device)
        logits = model(dummy)
    extra = {
        "model_check": {
            "model_variant": model_variant(context),
            "model_class": type(model).__name__,
            "device": str(device),
            "pretrained": False,
            "num_classes": len(context.class_to_index),
            "logits_shape": list(logits.shape),
            **parameter_counts,
        }
    }
    metadata_path = write_metadata(context, args, "check_model", extra)
    return {"metadata": str(metadata_path), **extra["model_check"]}


def run_smoke_test(context: DeiTRunContext, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn

    set_seed(int(context.config.get("seed", 42)))
    device = device_from_config(context)
    model = create_deit_model(context, device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(context.config.get("learning_rate", 0.0005)),
        weight_decay=float(context.config.get("weight_decay", 0.05)),
    )
    criterion = nn.CrossEntropyLoss()
    sample_count = max_smoke_samples(context, args)
    train_loader = make_data_loader(
        context.train_records[:sample_count],
        context,
        batch_size=min(sample_count, effective_batch_size(context, args)),
        shuffle=False,
    )
    batch = next(iter(train_loader))
    images, labels, records = collate_batch(batch, device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    val_loader = make_data_loader(
        context.val_records[:sample_count],
        context,
        batch_size=min(sample_count, effective_batch_size(context, args)),
        shuffle=False,
    )
    metrics, per_class, confusion, _predictions = evaluate(
        model=model,
        loader=val_loader,
        context=context,
        device=device,
    )
    extra = {
        "smoke_test_result": {
            "model_variant": model_variant(context),
            "device": str(device),
            "num_classes": len(context.class_to_index),
            "train_samples_loaded": len(records),
            "val_samples_loaded": len(context.val_records[:sample_count]),
            "loss": float(loss.detach().cpu().item()),
            "val_metrics_on_tiny_subset_not_interpretable": metrics,
            "per_class_on_tiny_subset_not_interpretable": per_class,
            "confusion_on_tiny_subset_not_interpretable": confusion,
            "checkpoint_written": False,
        }
    }
    metadata_path = write_metadata(context, args, "smoke_test", extra)
    return {"metadata": str(metadata_path), **extra["smoke_test_result"]}


def run_training(context: DeiTRunContext, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn

    set_seed(int(context.config.get("seed", 42)))
    device = device_from_config(context)
    model = create_deit_model(context, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(context.config.get("learning_rate", 0.0005)),
        weight_decay=float(context.config.get("weight_decay", 0.05)),
    )
    amp_enabled = bool(context.config.get("training", {}).get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=amp_enabled)
    train_loader = make_data_loader(context.train_records, context, batch_size=effective_batch_size(context, args), shuffle=True)
    val_loader = make_data_loader(context.val_records, context, batch_size=effective_batch_size(context, args), shuffle=False)
    epochs = effective_epochs(context, args)
    best_macro_f1 = -1.0
    best_epoch = 0
    epoch_rows: list[dict[str, Any]] = []
    labels = [context.index_to_class[index] for index in sorted(context.index_to_class)]
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            images, labels_tensor, _records = collate_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if hasattr(torch, "amp"):
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = model(images)
                    loss = criterion(logits, labels_tensor)
            else:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits = model(images)
                    loss = criterion(logits, labels_tensor)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(labels_tensor.shape[0])
            train_loss_sum += float(loss.detach().cpu().item()) * batch_size
            train_count += batch_size

        val_metrics, per_class, confusion, prediction_rows = evaluate(
            model=model,
            loader=val_loader,
            context=context,
            device=device,
        )
        train_loss = train_loss_sum / train_count if train_count else 0.0
        epoch_row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        epoch_rows.append(epoch_row)
        if float(val_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            save_checkpoint(
                context.output_dir / "checkpoints" / "best_model.pt",
                model,
                context,
                epoch,
                val_metrics,
            )
            write_predictions(context.output_dir / "predictions_val.csv", prediction_rows, context)
            write_json(
                context.output_dir / "metrics_val.json",
                {
                    "overall_metrics": val_metrics,
                    "per_class_metrics": per_class,
                    "confusion_matrix": {
                        "labels": labels,
                        "rows_are_true_labels": True,
                        "values": confusion,
                    },
                    "best_epoch": best_epoch,
                    "checkpoint_metric": "macro_f1",
                },
            )
            write_confusion(context.output_dir / "confusion_matrix_val.csv", labels, confusion)

    save_checkpoint(
        context.output_dir / "checkpoints" / "last_model.pt",
        model,
        context,
        epochs,
        epoch_rows[-1],
    )
    write_epoch_metrics(context.output_dir / "training_metrics.csv", epoch_rows)
    extra = {
        "training_result": {
            "model_variant": model_variant(context),
            "device": str(device),
            "epochs_completed": epochs,
            "best_epoch": best_epoch,
            "best_macro_f1": best_macro_f1,
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "local_outputs": {
                "best_checkpoint": relative_to_repo(context.output_dir / "checkpoints" / "best_model.pt"),
                "last_checkpoint": relative_to_repo(context.output_dir / "checkpoints" / "last_model.pt"),
                "val_predictions": relative_to_repo(context.output_dir / "predictions_val.csv"),
                "val_metrics": relative_to_repo(context.output_dir / "metrics_val.json"),
            },
            "test_used": False,
        }
    }
    metadata_path = write_metadata(context, args, "training", extra)
    return {"metadata": str(metadata_path), **extra["training_result"]}


def save_checkpoint(
    path: Path,
    model: Any,
    context: DeiTRunContext,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_index": context.class_to_index,
            "model_variant": model_variant(context),
            "pretrained": False,
            "epoch": epoch,
            "metrics": metrics,
            "config": context.config,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    try:
        context = prepare_context(args)
        if args.dry_run:
            result = run_dry_run(context, args)
        elif args.check_model:
            result = run_check_model(context, args)
        elif args.smoke_test:
            result = run_smoke_test(context, args)
        elif args.allow_training:
            result = run_training(context, args)
        else:  # pragma: no cover - parse_args normalizes to dry-run.
            result = run_dry_run(context, args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
