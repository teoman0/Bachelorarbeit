"""Run DINOv3 partial fine-tuning for global classification.

This workflow is a controlled validation variant next to the frozen DINOv3
linear-head baseline. It keeps patch embedding and earlier transformer blocks
frozen, then trains the linear head, the final LayerNorm when present, and the
last N transformer blocks. The default mode is a dry-run. Long training starts
only with --allow-training.
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
from contextlib import nullcontext
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from bachelorarbeit.data.split_dataset import (  # noqa: E402
    ManifestRecord,
    build_class_mapping,
    check_local_files,
    filter_split,
    read_split_manifest,
    split_distribution,
)
from bachelorarbeit.training.global_training_setup import load_yaml_config, resolve_repo_path  # noqa: E402
from scripts.train_dinov3_head import (  # noqa: E402
    collate_images,
    evaluate,
    extract_features,
    infer_feature_info,
    load_dinov3,
    make_data_loader,
    write_confusion,
    write_json,
    write_predictions,
)


@dataclass(frozen=True)
class DinoPartialContext:
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


@dataclass(frozen=True)
class TrainableSelection:
    block_container_name: str
    block_count_total: int
    train_last_n_blocks: int
    trainable_block_names: list[str]
    trainable_final_layernorm_names: list[str]
    trainable_backbone_parameter_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/dinov3_partial_finetune_last2.yaml")
    parser.add_argument("--dataset-root", required=True, help="Local dataset root; never written to versioned files.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config, manifest, train/val files, and metadata only.")
    parser.add_argument("--check-model", action="store_true", help="Load DINOv3 and report the partial fine-tuning parameter plan.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny train/val forward/backward check; no checkpoints.")
    parser.add_argument("--allow-training", action="store_true", help="Start the real partial fine-tuning run.")
    parser.add_argument("--allow-download", action="store_true", help="Allow Transformers to download DINOv3 weights.")
    parser.add_argument("--model-variant", default=None, help="Override model_id/model_variant from config.")
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


def prepare_context(args: argparse.Namespace) -> DinoPartialContext:
    config_path = resolve_repo_path(args.config)
    config = load_yaml_config(config_path)
    model_family = str(config.get("model_family", ""))
    if model_family != "dinov3_partial_finetune":
        raise ValueError(f"Expected model_family='dinov3_partial_finetune', got {model_family!r}")

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
        raise ValueError("Both train and val records are required for partial fine-tuning")

    class_to_index = build_class_mapping(records)
    index_to_class = {index: label for label, index in class_to_index.items()}
    image_size = tuple(int(value) for value in config.get("image_size", [224, 224]))
    output_dir = resolve_repo_path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    return DinoPartialContext(
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


def selected_model_id(context: DinoPartialContext, args: argparse.Namespace) -> str:
    if args.model_variant:
        return args.model_variant
    dinov3_cfg = context.config.get("dinov3", {})
    return str(dinov3_cfg.get("model_id") or context.config.get("model_variant"))


def allow_download(context: DinoPartialContext, args: argparse.Namespace) -> bool:
    return bool(args.allow_download or context.config.get("dinov3", {}).get("allow_download", False))


def effective_batch_size(context: DinoPartialContext, args: argparse.Namespace) -> int:
    return int(args.batch_override or context.config.get("batch_size", 8))


def effective_epochs(context: DinoPartialContext, args: argparse.Namespace) -> int:
    return int(args.epochs_override or context.config.get("epochs", 30))


def effective_patience(context: DinoPartialContext) -> int:
    return int(context.config.get("patience", 10))


def max_smoke_samples(context: DinoPartialContext, args: argparse.Namespace) -> int:
    training_cfg = context.config.get("training", {})
    return int(args.max_smoke_samples or training_cfg.get("smoke_max_samples_per_split", 2))


def selected_feature_representation(context: DinoPartialContext) -> str:
    return str(context.config.get("feature_representation", "pooler_output_or_cls_token"))


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
    packages = ["torch", "torchvision", "transformers", "accelerate", "Pillow", "PyYAML"]
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def split_file_checks(context: DinoPartialContext) -> dict[str, Any]:
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


def base_metadata(context: DinoPartialContext, args: argparse.Namespace, mode: str) -> dict[str, Any]:
    checks = split_file_checks(context)
    config = context.config
    return {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "script": "scripts/train_dinov3_partial_finetune.py",
        "mode": mode,
        "experiment_name": config.get("experiment_name"),
        "model_family": config.get("model_family"),
        "model_variant": selected_model_id(context, args),
        "fallback_model_variant": config.get("fallback_model_variant"),
        "weights_source": config.get("dinov3", {}).get("weights_source"),
        "allow_download": allow_download(context, args),
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
        "patience": effective_patience(context),
        "optimizer": config.get("optimizer"),
        "backbone_lr": config.get("backbone_lr"),
        "head_lr": config.get("head_lr"),
        "weight_decay": config.get("weight_decay"),
        "freeze_backbone": config.get("freeze_backbone"),
        "partial_finetune": config.get("partial_finetune", {}),
        "head": config.get("head"),
        "feature_representation": config.get("feature_representation"),
        "checkpoint_metric": config.get("checkpoint_metric"),
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "class_to_index": context.class_to_index,
        "split_distribution": split_distribution(context.records),
        "local_file_check_train_val_only": checks,
        "artifact_policy": config.get("artifact_policy", {}),
    }


def write_metadata(
    context: DinoPartialContext,
    args: argparse.Namespace,
    mode: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = base_metadata(context, args, mode)
    if extra:
        payload.update(extra)
    path = context.output_dir / f"{mode}_metadata.json"
    write_json(path, payload)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        return


def device_from_config(context: DinoPartialContext) -> Any:
    import torch

    raw_device = context.config.get("training", {}).get("device", 0)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{raw_device}")
    return torch.device("cpu")


def find_transformer_block_sequence(model: Any, last_n_blocks: int) -> tuple[str, Any]:
    import torch.nn as nn

    candidates: list[tuple[int, int, str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList):
            continue
        if len(module) < last_n_blocks:
            continue
        lowered = name.lower()
        score = 0
        if "encoder" in lowered:
            score += 4
        if any(token in lowered for token in ("layer", "layers", "block", "blocks")):
            score += 4
        if len(module) >= 4:
            score += 2
        candidates.append((score, len(module), name, module))

    if not candidates:
        raise RuntimeError("Could not find a transformer block ModuleList in the DINOv3 model")

    candidates.sort(key=lambda item: (item[0], item[1], -len(item[2])), reverse=True)
    _score, _length, name, module = candidates[0]
    return name, module


def block_name(container_name: str, index: int) -> str:
    return f"{container_name}.{index}" if container_name else str(index)


def is_inside_any(name: str, prefixes: list[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def find_final_layernorms(model: Any, all_block_names: list[str]) -> list[tuple[str, Any]]:
    import torch.nn as nn

    candidates: list[tuple[int, str, Any]] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.LayerNorm):
            continue
        lowered = name.lower()
        leaf = lowered.rsplit(".", 1)[-1]
        if is_inside_any(name, all_block_names):
            continue
        if "embedding" in lowered or "patch" in lowered:
            continue
        if leaf in {"norm", "layernorm", "post_layernorm", "final_layer_norm"} or lowered.endswith("layernorm"):
            rank = 2 if "." not in name else 1
            candidates.append((rank, name, module))
    candidates.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return [(name, module) for _rank, name, module in candidates]


def apply_partial_finetune_plan(model: Any, context: DinoPartialContext) -> TrainableSelection:
    partial_cfg = context.config.get("partial_finetune", {})
    last_n_blocks = int(partial_cfg.get("train_last_n_blocks", 2))
    if last_n_blocks < 1:
        raise ValueError("partial_finetune.train_last_n_blocks must be at least 1")

    for parameter in model.parameters():
        parameter.requires_grad = False

    container_name, block_sequence = find_transformer_block_sequence(model, last_n_blocks)
    block_count = len(block_sequence)
    if last_n_blocks > block_count:
        raise ValueError(f"Requested last {last_n_blocks} blocks, but only {block_count} blocks were found")

    all_block_names = [block_name(container_name, index) for index in range(block_count)]
    trainable_block_names: list[str] = []
    for index in range(block_count - last_n_blocks, block_count):
        module = block_sequence[index]
        for parameter in module.parameters():
            parameter.requires_grad = True
        trainable_block_names.append(block_name(container_name, index))

    trainable_final_layernorm_names: list[str] = []
    if bool(partial_cfg.get("train_final_layernorm", True)):
        for name, module in find_final_layernorms(model, all_block_names):
            for parameter in module.parameters():
                parameter.requires_grad = True
            trainable_final_layernorm_names.append(name)

    trainable_backbone_parameter_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    return TrainableSelection(
        block_container_name=container_name,
        block_count_total=block_count,
        train_last_n_blocks=last_n_blocks,
        trainable_block_names=trainable_block_names,
        trainable_final_layernorm_names=trainable_final_layernorm_names,
        trainable_backbone_parameter_names=trainable_backbone_parameter_names,
    )


def parameter_count(module: Any, *, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return sum(parameter.numel() for parameter in module.parameters())


def cuda_info() -> dict[str, Any]:
    import torch

    cuda_available = torch.cuda.is_available()
    return {
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def create_head(feature_dim: int, num_classes: int, device: Any) -> Any:
    import torch.nn as nn

    return nn.Linear(feature_dim, num_classes).to(device)


def build_model_and_head(context: DinoPartialContext, args: argparse.Namespace) -> tuple[Any, Any, Any, TrainableSelection, dict[str, Any], Any]:
    device = device_from_config(context)
    model_id = selected_model_id(context, args)
    processor, model = load_dinov3(
        model_id,
        allow_model_download=allow_download(context, args),
        device=device,
    )
    selection = apply_partial_finetune_plan(model, context)
    feature_info = infer_feature_info(
        model,
        context.image_size,
        device,
        selected_feature_representation(context),
    )
    head = create_head(int(feature_info["feature_dim"]), len(context.class_to_index), device)
    return processor, model, head, selection, feature_info, device


def model_check_payload(
    *,
    context: DinoPartialContext,
    args: argparse.Namespace,
    model: Any,
    head: Any,
    processor: Any,
    selection: TrainableSelection,
    feature_info: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    backbone_total = parameter_count(model)
    backbone_trainable = parameter_count(model, trainable_only=True)
    head_total = parameter_count(head)
    head_trainable = parameter_count(head, trainable_only=True)
    return {
        "model_id": selected_model_id(context, args),
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "device": str(device),
        **cuda_info(),
        "feature_dim": feature_info["feature_dim"],
        "feature_shape": feature_info["feature_shape"],
        "feature_representation_used": feature_info["feature_representation_used"],
        "parameters_total": backbone_total + head_total,
        "parameters_trainable": backbone_trainable + head_trainable,
        "backbone_parameters_total": backbone_total,
        "backbone_parameters_trainable": backbone_trainable,
        "head_parameters_total": head_total,
        "head_parameters_trainable": head_trainable,
        "trainable_block_container": selection.block_container_name,
        "transformer_blocks_total": selection.block_count_total,
        "train_last_n_blocks": selection.train_last_n_blocks,
        "trainable_blocks": selection.trainable_block_names,
        "trainable_final_layernorms": selection.trainable_final_layernorm_names,
        "trainable_backbone_parameter_examples": selection.trainable_backbone_parameter_names[:30],
    }


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


def optimizer_for(context: DinoPartialContext, model: Any, head: Any) -> Any:
    import torch

    backbone_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    head_params = list(head.parameters())
    if not backbone_params:
        raise ValueError("No trainable backbone parameters found for partial fine-tuning")
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": float(context.config.get("backbone_lr", 1e-5)), "name": "backbone_last_blocks"},
            {"params": head_params, "lr": float(context.config.get("head_lr", 5e-4)), "name": "linear_head"},
        ],
        weight_decay=float(context.config.get("weight_decay", 0.05)),
    )


def run_dry_run(context: DinoPartialContext, args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = write_metadata(
        context,
        args,
        "dry_run",
        {
            "dry_run_note": "No model is loaded, no images are loaded, and no training is started.",
            "test_file_availability_checked": False,
        },
    )
    return {"metadata": relative_to_repo(metadata_path), "output_dir": relative_to_repo(context.output_dir)}


def run_check_model(context: DinoPartialContext, args: argparse.Namespace) -> dict[str, Any]:
    processor, model, head, selection, feature_info, device = build_model_and_head(context, args)
    check = model_check_payload(
        context=context,
        args=args,
        model=model,
        head=head,
        processor=processor,
        selection=selection,
        feature_info=feature_info,
        device=device,
    )
    metadata_path = write_metadata(context, args, "check_model", {"model_check": check})
    return {"metadata": relative_to_repo(metadata_path), **check}


def run_smoke_test(context: DinoPartialContext, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn

    set_seed(int(context.config.get("seed", 42)))
    processor, model, head, selection, feature_info, device = build_model_and_head(context, args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_for(context, model, head)
    amp_enabled = bool(context.config.get("training", {}).get("amp", True) and device.type == "cuda")
    scaler = create_grad_scaler(device, amp_enabled)
    sample_count = max_smoke_samples(context, args)
    batch_size = min(sample_count, effective_batch_size(context, args))

    loader = make_data_loader(
        context.train_records[:sample_count],
        context,
        batch_size=batch_size,
        shuffle=False,
    )
    batch = next(iter(loader))
    inputs, labels, records = collate_images(batch, processor, device)
    model.train()
    head.train()
    optimizer.zero_grad(set_to_none=True)
    with amp_autocast(device, amp_enabled):
        outputs = model(**inputs)
        features, representation = extract_features(outputs, selected_feature_representation(context))
        logits = head(features)
        loss = criterion(logits, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    val_loader = make_data_loader(
        context.val_records[:sample_count],
        context,
        batch_size=batch_size,
        shuffle=False,
    )
    metrics, per_class, confusion, _predictions = evaluate(
        model=model,
        head=head,
        processor=processor,
        loader=val_loader,
        context=context,
        device=device,
        preferred_representation=selected_feature_representation(context),
    )
    smoke = {
        "model_id": selected_model_id(context, args),
        "device": str(device),
        **cuda_info(),
        "feature_dim": feature_info["feature_dim"],
        "feature_representation_used": representation,
        "trainable_blocks": selection.trainable_block_names,
        "trainable_final_layernorms": selection.trainable_final_layernorm_names,
        "train_samples_loaded": len(records),
        "val_samples_loaded": len(context.val_records[:sample_count]),
        "loss": float(loss.detach().cpu().item()),
        "val_metrics_on_tiny_subset_not_interpretable": metrics,
        "per_class_on_tiny_subset_not_interpretable": per_class,
        "confusion_on_tiny_subset_not_interpretable": confusion,
        "checkpoint_written": False,
        "test_used": False,
    }
    metadata_path = write_metadata(context, args, "smoke_test", {"smoke_test_result": smoke})
    return {"metadata": relative_to_repo(metadata_path), **smoke}


def write_metrics_csv(path: Path, metrics: dict[str, Any], best_epoch: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["best_epoch", "num_rows", "accuracy", "balanced_accuracy", "macro_f1"],
        )
        writer.writeheader()
        writer.writerow({"best_epoch": best_epoch, **metrics})


def write_epoch_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    *,
    model: Any,
    head: Any,
    context: DinoPartialContext,
    args: argparse.Namespace,
    selection: TrainableSelection,
    feature_info: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "head_state_dict": head.state_dict(),
            "class_to_index": context.class_to_index,
            "model_id": selected_model_id(context, args),
            "feature_info": feature_info,
            "trainable_selection": {
                "block_container_name": selection.block_container_name,
                "block_count_total": selection.block_count_total,
                "train_last_n_blocks": selection.train_last_n_blocks,
                "trainable_block_names": selection.trainable_block_names,
                "trainable_final_layernorm_names": selection.trainable_final_layernorm_names,
            },
            "epoch": epoch,
            "metrics": metrics,
            "config": context.config,
        },
        path,
    )


def run_training(context: DinoPartialContext, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn

    set_seed(int(context.config.get("seed", 42)))
    processor, model, head, selection, feature_info, device = build_model_and_head(context, args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_for(context, model, head)
    amp_enabled = bool(context.config.get("training", {}).get("amp", True) and device.type == "cuda")
    scaler = create_grad_scaler(device, amp_enabled)

    train_loader = make_data_loader(
        context.train_records,
        context,
        batch_size=effective_batch_size(context, args),
        shuffle=True,
    )
    val_loader = make_data_loader(
        context.val_records,
        context,
        batch_size=effective_batch_size(context, args),
        shuffle=False,
    )
    epochs = effective_epochs(context, args)
    patience = effective_patience(context)
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, Any]] = []
    labels = [context.index_to_class[index] for index in sorted(context.index_to_class)]
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            inputs, labels_tensor, _records = collate_images(batch, processor, device)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device, amp_enabled):
                outputs = model(**inputs)
                features, _representation = extract_features(outputs, selected_feature_representation(context))
                logits = head(features)
                loss = criterion(logits, labels_tensor)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(labels_tensor.shape[0])
            train_loss_sum += float(loss.detach().cpu().item()) * batch_size
            train_count += batch_size

        val_metrics, per_class, confusion, prediction_rows = evaluate(
            model=model,
            head=head,
            processor=processor,
            loader=val_loader,
            context=context,
            device=device,
            preferred_representation=selected_feature_representation(context),
        )
        train_loss = train_loss_sum / train_count if train_count else 0.0
        improved = float(val_metrics["macro_f1"]) > best_macro_f1
        if improved:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                context.output_dir / "checkpoints" / "best_model.pt",
                model=model,
                head=head,
                context=context,
                args=args,
                selection=selection,
                feature_info=feature_info,
                epoch=epoch,
                metrics=val_metrics,
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
                    "patience": patience,
                },
            )
            write_metrics_csv(context.output_dir / "val_metrics.csv", val_metrics, best_epoch)
            write_confusion(context.output_dir / "confusion_matrix_val.csv", labels, confusion)
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
        if epochs_without_improvement >= patience:
            break

    save_checkpoint(
        context.output_dir / "checkpoints" / "last_model.pt",
        model=model,
        head=head,
        context=context,
        args=args,
        selection=selection,
        feature_info=feature_info,
        epoch=epoch_rows[-1]["epoch"],
        metrics=epoch_rows[-1],
    )
    write_epoch_metrics(context.output_dir / "training_log.csv", epoch_rows)
    result = {
        "model_id": selected_model_id(context, args),
        "device": str(device),
        **cuda_info(),
        "feature_info": feature_info,
        "epochs_completed": int(epoch_rows[-1]["epoch"]),
        "early_stopped": epoch_rows[-1]["epochs_without_improvement"] >= patience,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "trainable_blocks": selection.trainable_block_names,
        "trainable_final_layernorms": selection.trainable_final_layernorm_names,
        "local_outputs": {
            "best_checkpoint": relative_to_repo(context.output_dir / "checkpoints" / "best_model.pt"),
            "last_checkpoint": relative_to_repo(context.output_dir / "checkpoints" / "last_model.pt"),
            "val_predictions": relative_to_repo(context.output_dir / "predictions_val.csv"),
            "val_metrics": relative_to_repo(context.output_dir / "metrics_val.json"),
            "training_log": relative_to_repo(context.output_dir / "training_log.csv"),
        },
        "test_used": False,
    }
    metadata_path = write_metadata(context, args, "training", {"training_result": result})
    write_json(context.output_dir / "run_metadata.json", base_metadata(context, args, "training") | {"training_result": result})
    return {"metadata": relative_to_repo(metadata_path), **result}


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
