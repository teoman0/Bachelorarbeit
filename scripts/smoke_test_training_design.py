"""Smoke-test the training design templates without loading data or models.

This script validates only the placeholder YAML structure in configs/templates.
It does not read images, download weights, create checkpoints, or start
training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "configs" / "templates"

EXPECTED_TEMPLATES = [
    "yolo11_cls_template.yaml",
    "dinov3_frozen_head_template.yaml",
    "deit_tiny_from_scratch_template.yaml",
    "dinov3_patch_analysis_template.yaml",
]

REQUIRED_PATHS = [
    ("template",),
    ("experiment", "name"),
    ("experiment", "seed"),
    ("dataset", "dataset_id"),
    ("dataset", "split_manifest"),
    ("dataset", "class_count"),
    ("split", "strategy"),
    ("split", "ratio"),
    ("split", "group_regex"),
    ("input", "size"),
    ("input", "channels"),
    ("input", "grayscale_handling"),
    ("preprocessing", "resize_strategy"),
    ("preprocessing", "normalization"),
    ("augmentations", "train"),
    ("augmentations", "val_test"),
    ("model", "group"),
    ("model", "variant"),
    ("model", "pretrained"),
    ("model", "weights_source"),
    ("training", "enabled"),
    ("training", "optimizer"),
    ("training", "scheduler"),
    ("training", "batch_size"),
    ("training", "epochs"),
    ("metrics", "global"),
    ("reproducibility", "package_versions"),
    ("reproducibility", "git_commit"),
    ("reproducibility", "output_dir"),
]


def get_nested(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            joined = ".".join(path)
            raise KeyError(f"Missing required field: {joined}")
        current = current[key]
    return current


def validate_template(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"{path.name}: expected a YAML mapping")

    for required_path in REQUIRED_PATHS:
        get_nested(config, required_path)

    if config["template"] is not True:
        raise ValueError(f"{path.name}: template must be true")

    ratio = get_nested(config, ("split", "ratio"))
    if not isinstance(ratio, list) or len(ratio) != 3:
        raise ValueError(f"{path.name}: split.ratio must contain train/val/test")
    if abs(sum(float(value) for value in ratio) - 1.0) > 1e-6:
        raise ValueError(f"{path.name}: split.ratio must sum to 1.0")

    input_size = get_nested(config, ("input", "size"))
    if not isinstance(input_size, list) or len(input_size) != 2:
        raise ValueError(f"{path.name}: input.size must be [height, width]")

    metrics = get_nested(config, ("metrics", "global"))
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(f"{path.name}: metrics.global must be a non-empty list")


def build_dummy_batch() -> list[dict[str, Any]]:
    return [
        {"image_id": "dummy_q1", "group_id": "dummy", "label": "class_a"},
        {"image_id": "dummy_q2", "group_id": "dummy", "label": "class_a"},
    ]


def main() -> None:
    dummy_batch = build_dummy_batch()
    if len({item["group_id"] for item in dummy_batch}) != 1:
        raise AssertionError("Dummy q-suffix grouping sanity check failed")

    for filename in EXPECTED_TEMPLATES:
        path = TEMPLATE_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing template: {path}")
        validate_template(path)

    print(f"Validated {len(EXPECTED_TEMPLATES)} templates with dummy data only.")


if __name__ == "__main__":
    main()
