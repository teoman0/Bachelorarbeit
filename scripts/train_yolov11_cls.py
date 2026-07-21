"""Prepare and run the YOLOv11-cls global classification workflow.

By default the script performs a dry-run only. It can also create a local
YOLO-compatible classification folder structure from the versioned split
manifest. Real training starts only with --allow-training.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bachelorarbeit.data.split_dataset import (
    ManifestRecord,
    read_split_manifest,
    resolve_record_path,
    split_distribution,
)
from bachelorarbeit.training.global_training_setup import (
    PreparedRun,
    prepare_run,
    resolve_repo_path,
)


DEVELOPMENT_SPLITS = ("train", "val")
FINAL_TEST_SPLIT = "test"


def yolo_dataset_dir(config: dict[str, Any]) -> Path:
    yolo_cfg = config.get("yolo", {})
    raw_path = yolo_cfg.get(
        "temp_dataset_dir",
        "outputs/global_classification/yolov11_cls/yolo_dataset",
    )
    return resolve_repo_path(str(raw_path))


def local_summary_path(dataset_dir: Path) -> Path:
    return dataset_dir.parent / "yolo_dataset_summary.json"


def relative_to_repo_or_name(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def target_name(record: ManifestRecord) -> str:
    suffix = Path(record.relative_path).suffix.lower()
    if not suffix:
        suffix = ".jpg"
    return f"{record.image_id}{suffix}"


def ensure_link(source: Path, target: Path, method: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        try:
            if target.samefile(source):
                return "existing"
        except OSError:
            pass
        raise FileExistsError(f"Refusing to overwrite existing non-matching file: {target}")

    if method == "symlink":
        create_symlink(source, target)
        return "symlink"
    if method == "hardlink":
        create_hardlink(source, target)
        return "hardlink"
    if method != "auto":
        raise ValueError(f"Unsupported link method: {method}")

    try:
        create_symlink(source, target)
        return "symlink"
    except OSError:
        create_hardlink(source, target)
        return "hardlink"


def create_symlink(source: Path, target: Path) -> None:
    os.symlink(source, target)


def create_hardlink(source: Path, target: Path) -> None:
    os.link(source, target)


def selected_yolo_splits(include_final_test: bool) -> tuple[str, ...]:
    return DEVELOPMENT_SPLITS + ((FINAL_TEST_SPLIT,) if include_final_test else ())


def assert_no_materialized_test_split(dataset_dir: Path) -> None:
    test_dir = dataset_dir / FINAL_TEST_SPLIT
    if test_dir.is_dir() and any(path.is_file() for path in test_dir.rglob("*")):
        raise RuntimeError(
            "The prepared YOLO dataset contains a materialized test split. "
            "Training and smoke-test workflows require a train/val-only dataset view."
        )


def prepare_yolo_dataset(
    run: PreparedRun,
    link_method: str,
    *,
    include_final_test: bool = False,
) -> dict[str, Any]:
    if link_method not in {"auto", "symlink", "hardlink"}:
        raise ValueError("--link-method must be auto, symlink, or hardlink")

    records = read_split_manifest(run.manifest_path)
    selected_splits = selected_yolo_splits(include_final_test)
    dataset_dir = yolo_dataset_dir(run.config)
    if not include_final_test:
        assert_no_materialized_test_split(dataset_dir)
    class_to_index = run.class_to_index
    method_counts: Counter[str] = Counter()
    split_class_counts: dict[str, Counter[str]] = {
        split: Counter() for split in selected_splits
    }
    split_file_counts: Counter[str] = Counter()

    for label in class_to_index:
        for split in selected_splits:
            (dataset_dir / split / label).mkdir(parents=True, exist_ok=True)

    for record in records:
        if record.split not in selected_splits:
            continue
        source = resolve_record_path(run.dataset_root, record)
        target = dataset_dir / record.split / record.label / target_name(record)
        used_method = ensure_link(source, target, link_method)
        method_counts[used_method] += 1
        split_file_counts[record.split] += 1
        split_class_counts[record.split][record.label] += 1

    summary = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "script": "scripts/train_yolov11_cls.py",
        "mode": "prepare_yolo_dataset",
        "dataset_root_recorded": False,
        "source_manifest": str(run.manifest_path.relative_to(REPO_ROOT)),
        "yolo_dataset_dir": relative_to_repo_or_name(dataset_dir),
        "link_method_requested": link_method,
        "link_method_counts": dict(sorted(method_counts.items())),
        "copy_images": False,
        "class_to_index": class_to_index,
        "split_distribution": split_distribution(records),
        "prepared_file_counts": dict(sorted(split_file_counts.items())),
        "prepared_class_counts": {
            split: dict(sorted(counts.items())) for split, counts in split_class_counts.items()
        },
        "prepared_splits": list(selected_splits),
        "final_test_materialized": include_final_test,
        "test_usage_note": (
            "The test split is materialized only when --allow-final-test is supplied. "
            "It must never be used for training, validation, checkpoint selection, or decisions."
            if include_final_test
            else "Only train and val are materialized; test remains untouched."
        ),
    }
    path = local_summary_path(dataset_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def verify_prepared_yolo_dataset(config: dict[str, Any], class_to_index: dict[str, int]) -> Path:
    dataset_dir = yolo_dataset_dir(config)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            "Prepared YOLO dataset folder is missing. Run with --prepare-yolo-dataset first: "
            f"{dataset_dir}"
        )
    assert_no_materialized_test_split(dataset_dir)
    missing_dirs: list[str] = []
    for split in ("train", "val"):
        for label in class_to_index:
            path = dataset_dir / split / label
            if not path.exists() or not path.is_dir():
                missing_dirs.append(str(path))
    if missing_dirs:
        raise FileNotFoundError(
            "Prepared YOLO dataset is incomplete. Missing train/val class folders: "
            f"{missing_dirs[:10]}"
        )
    return dataset_dir


def int_override(value: int | None, fallback: int) -> int:
    return fallback if value is None else value


def image_size_from_config(config: dict[str, Any], override: int | None) -> int:
    if override is not None:
        return override
    raw_size = config.get("image_size", [320, 320])
    if isinstance(raw_size, list):
        return int(raw_size[0])
    return int(raw_size)


def build_training_plan(
    run: PreparedRun,
    *,
    epochs_override: int | None,
    batch_override: int | None,
    imgsz_override: int | None,
) -> dict[str, Any]:
    epochs = int_override(epochs_override, int(run.config.get("epochs", 75)))
    batch_size = int_override(batch_override, int(run.config.get("batch_size", 16)))
    imgsz = image_size_from_config(run.config, imgsz_override)
    yolo_cfg = run.config.get("yolo", {})
    plan = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "script": "scripts/train_yolov11_cls.py",
        "mode": "training_plan",
        "technical_test": epochs <= 1,
        "technical_test_note": (
            "A 1-epoch run is only a technical pipeline check and must not be "
            "interpreted as model performance."
        ),
        "model_variant": run.config.get("model_variant"),
        "pretrained": run.config.get("pretrained"),
        "data": relative_to_repo_or_name(yolo_dataset_dir(run.config)),
        "imgsz": imgsz,
        "batch": batch_size,
        "epochs": epochs,
        "patience": run.config.get("patience"),
        "amp": run.config.get("amp"),
        "device": run.config.get("device"),
        "workers": run.config.get("workers"),
        "optimizer": run.config.get("optimizer"),
        "learning_rate": run.config.get("learning_rate"),
        "weight_decay": run.config.get("weight_decay"),
        "project": yolo_cfg.get("project"),
        "name": yolo_cfg.get("name"),
        "checkpoint_metric_note": (
            "Ultralytics classification checkpointing may use built-in metrics such "
            "as top1 or validation loss. Macro-F1 is computed later from predictions "
            "with the project evaluation workflow."
        ),
        "test_usage_note": (
            "The test split is not used for training, validation, early stopping, "
            "checkpoint selection, or model decisions."
        ),
    }
    plan_path = run.output_dir / "yolo_training_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan


def run_ultralytics_training(run: PreparedRun, plan: dict[str, Any]) -> None:
    dataset_dir = verify_prepared_yolo_dataset(run.config, run.class_to_index)
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ultralytics is not installed in the active Python environment. "
            "Install the project requirements in a local environment before running "
            "YOLO training."
        ) from exc

    yolo_cfg = run.config.get("yolo", {})
    model = YOLO(str(run.config["model_variant"]))
    train_kwargs: dict[str, Any] = {
        "data": str(dataset_dir),
        "imgsz": int(plan["imgsz"]),
        "epochs": int(plan["epochs"]),
        "batch": int(plan["batch"]),
        "patience": int(run.config.get("patience", 15)),
        "amp": bool(run.config.get("amp", True)),
        "device": run.config.get("device", 0),
        "workers": int(run.config.get("workers", 4)),
        "optimizer": run.config.get("optimizer", "auto"),
        "project": yolo_cfg.get("project", "runs/global_classification"),
        "name": yolo_cfg.get("name", "yolov11_cls_bmw25_global"),
        "seed": int(run.config.get("seed", 42)),
    }
    if run.config.get("learning_rate") is not None:
        train_kwargs["lr0"] = float(run.config["learning_rate"])
    if run.config.get("weight_decay") is not None:
        train_kwargs["weight_decay"] = float(run.config["weight_decay"])

    model.train(**train_kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/yolov11_cls.yaml")
    parser.add_argument("--dataset-root", required=True, help="Local dataset root; not written to versioned files.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config, manifest, local files, and metadata only.")
    parser.add_argument("--prepare-yolo-dataset", action="store_true", help="Create local YOLO class-folder dataset with links.")
    parser.add_argument("--smoke-test", action="store_true", help="Load a tiny subset from train and val only.")
    parser.add_argument(
        "--allow-final-test",
        action="store_true",
        help="Allow test links only with --prepare-yolo-dataset; never valid during training or smoke tests.",
    )
    parser.add_argument("--max-smoke-samples", type=int, default=2)
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="Explicitly start YOLO training from the prepared local YOLO dataset.",
    )
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--batch-override", type=int, default=None)
    parser.add_argument("--imgsz-override", type=int, default=None)
    parser.add_argument(
        "--link-method",
        choices=["auto", "symlink", "hardlink"],
        default="auto",
        help="Local dataset materialization method. auto tries symlink first, then hardlink. Copying is not supported.",
    )
    args = parser.parse_args()

    if args.max_smoke_samples < 1:
        parser.error("--max-smoke-samples must be at least 1")
    if args.epochs_override is not None and args.epochs_override < 1:
        parser.error("--epochs-override must be at least 1")
    if args.batch_override is not None and args.batch_override < 1:
        parser.error("--batch-override must be at least 1")
    if args.imgsz_override is not None and args.imgsz_override < 1:
        parser.error("--imgsz-override must be at least 1")
    if args.allow_training and args.dry_run:
        parser.error("--dry-run and --allow-training are mutually exclusive")
    if args.allow_final_test and not args.prepare_yolo_dataset:
        parser.error("--allow-final-test is only valid with --prepare-yolo-dataset")
    if args.allow_final_test and (args.allow_training or args.smoke_test):
        parser.error("Final-test materialization cannot be combined with training or smoke tests")

    dry_run = args.dry_run or not args.allow_training

    run = prepare_run(
        config_path=Path(args.config),
        dataset_root=Path(args.dataset_root),
        expected_model_family="yolov11_cls",
        dry_run=dry_run,
        smoke_test=args.smoke_test,
        max_smoke_samples=args.max_smoke_samples,
        extra_metadata={
            "classification_workflow": "Ultralytics YOLO cls",
            "local_yolo_dataset_note": (
                "A YOLO-compatible train/val class-folder structure may be created "
                "locally from the manifest with symlinks or hardlinks. Test requires "
                "the separate --allow-final-test guard. "
                "Do not commit prepared image folders. Copy fallback is disabled."
            ),
            "test_usage_note": (
                "The test split is reserved for final evaluation and is not used for "
                "training, validation, early stopping, checkpoint selection, or model decisions."
            ),
            "real_training_status": "requested" if args.allow_training else "not_started",
        },
        extra_packages=["ultralytics"],
    )

    result: dict[str, Any] = {
        "metadata": str(run.metadata_path),
        "output_dir": str(run.output_dir),
        "mode": "training" if args.allow_training else "dry_run",
        "test_split_note": "Test is held out for final evaluation only.",
    }

    if args.prepare_yolo_dataset:
        summary = prepare_yolo_dataset(
            run,
            args.link_method,
            include_final_test=bool(args.allow_final_test),
        )
        result["prepared_yolo_dataset"] = summary["yolo_dataset_dir"]
        result["link_method_counts"] = summary["link_method_counts"]

    if args.allow_training:
        plan = build_training_plan(
            run,
            epochs_override=args.epochs_override,
            batch_override=args.batch_override,
            imgsz_override=args.imgsz_override,
        )
        result["training_plan"] = {
            "epochs": plan["epochs"],
            "batch": plan["batch"],
            "imgsz": plan["imgsz"],
            "technical_test": plan["technical_test"],
        }
        print(
            "Starting YOLO training. The test split is not used for training, validation, "
            "early stopping, checkpoint selection, or model decisions.",
            file=sys.stderr,
        )
        run_ultralytics_training(run, plan)
        result["training_started"] = True
    else:
        result["training_started"] = False

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
