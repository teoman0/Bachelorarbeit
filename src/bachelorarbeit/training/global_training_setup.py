"""Shared setup code for global classification training skeletons."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal runtimes.
    yaml = None

from bachelorarbeit.data.split_dataset import (
    SimpleImageTransform,
    SplitImageDataset,
    build_class_mapping,
    check_local_files,
    filter_split,
    read_split_manifest,
    split_distribution,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEVELOPMENT_SPLITS = ("train", "val")


@dataclass(frozen=True)
class PreparedRun:
    config: dict[str, Any]
    config_path: Path
    manifest_path: Path
    dataset_root: Path
    output_dir: Path
    class_to_index: dict[str, int]
    metadata_path: Path
    metadata: dict[str, Any]


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        if yaml is not None:
            config = yaml.safe_load(handle)
        else:
            config = parse_simple_yaml(handle.read())
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by the experiment configs.

    This fallback exists only so dry-runs work in minimal runtimes without
    PyYAML. It supports nested mappings, scalar lists, inline lists, comments,
    strings, booleans, integers, and floats.
    """

    raw_lines = []
    for line in text.splitlines():
        stripped = strip_yaml_comment(line).rstrip()
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        raw_lines.append((indent, stripped.lstrip(" ")))
    parsed, next_index = parse_yaml_block(raw_lines, 0, 0)
    if next_index != len(raw_lines):
        raise ValueError("Could not parse full YAML config")
    if not isinstance(parsed, dict):
        raise ValueError("Top-level YAML value must be a mapping")
    return parsed


def parse_yaml_block(
    lines: list[tuple[int, str]],
    start_index: int,
    indent: int,
) -> tuple[Any, int]:
    if start_index >= len(lines):
        return {}, start_index
    if lines[start_index][1].startswith("- "):
        values: list[Any] = []
        index = start_index
        while index < len(lines):
            line_indent, content = lines[index]
            if line_indent != indent or not content.startswith("- "):
                break
            item = content[2:].strip()
            if item:
                values.append(parse_yaml_scalar(item))
                index += 1
            else:
                nested, index = parse_yaml_block(lines, index + 1, next_indent(lines, index + 1, indent + 2))
                values.append(nested)
        return values, index

    mapping: dict[str, Any] = {}
    index = start_index
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent != indent or content.startswith("- "):
            break
        if ":" not in content:
            raise ValueError(f"Unsupported YAML line: {content}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            mapping[key] = parse_yaml_scalar(raw_value)
            index += 1
            continue
        child_indent = next_indent(lines, index + 1, indent + 2)
        if index + 1 >= len(lines) or child_indent <= indent:
            mapping[key] = {}
            index += 1
            continue
        value, index = parse_yaml_block(lines, index + 1, child_indent)
        mapping[key] = value
    return mapping, index


def next_indent(lines: list[tuple[int, str]], index: int, default: int) -> int:
    if index >= len(lines):
        return default
    return lines[index][0]


def strip_yaml_comment(line: str) -> str:
    quote: str | None = None
    for index, char in enumerate(line):
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
        if char == "#" and quote is None:
            if index == 0 or line[index - 1].isspace():
                return line[:index]
    return line


def parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_yaml_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


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


def package_versions(extra_packages: list[str] | None = None) -> dict[str, str | None]:
    packages = ["Pillow", "PyYAML"]
    if extra_packages:
        packages.extend(extra_packages)
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def prepare_run(
    *,
    config_path: Path,
    dataset_root: Path,
    expected_model_family: str,
    dry_run: bool,
    smoke_test: bool,
    max_smoke_samples: int,
    extra_metadata: dict[str, Any] | None = None,
    extra_packages: list[str] | None = None,
) -> PreparedRun:
    config_path = resolve_repo_path(str(config_path))
    config = load_yaml_config(config_path)
    model_family = str(config.get("model_family", ""))
    if model_family != expected_model_family:
        raise ValueError(
            f"Config model_family={model_family!r} does not match expected {expected_model_family!r}"
        )

    dataset_root = Path(dataset_root).expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    manifest_path = resolve_repo_path(str(config["split_manifest"]))
    records = read_split_manifest(manifest_path)
    class_to_index = build_class_mapping(records)
    development_records = [record for record in records if record.split in DEVELOPMENT_SPLITS]
    file_check = check_local_files(development_records, dataset_root)
    if file_check.missing:
        raise FileNotFoundError(
            "Dataset root does not contain all required train/val manifest files. "
            f"Missing {file_check.missing} of {file_check.checked}; "
            f"examples={list(file_check.missing_examples)}"
        )

    image_size = tuple(int(value) for value in config.get("image_size", [224, 224]))
    smoke_records: dict[str, list[dict[str, Any]]] = {}
    if smoke_test:
        transform = SimpleImageTransform(image_size=image_size, resize_mode="resize_pad", convert_rgb=True)
        for split in DEVELOPMENT_SPLITS:
            split_records = filter_split(records, split)[:max_smoke_samples]
            dataset = SplitImageDataset(
                records=split_records,
                dataset_root=dataset_root,
                class_to_index=class_to_index,
                transform=transform,
            )
            smoke_records[split] = []
            for index in range(len(dataset)):
                image, label_index, record = dataset[index]
                smoke_records[split].append(
                    {
                        "image_id": record.image_id,
                        "label": record.label,
                        "label_index": label_index,
                        "transformed_size": list(image.size),
                        "mode": image.mode,
                    }
                )

    output_dir = resolve_repo_path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "script_mode": "dry_run" if dry_run else "training_requested",
        "smoke_test": smoke_test,
        "method_note": (
            "Setup metadata only. No long training is started by default, no weights "
            "are downloaded, and no checkpoints are created."
        ),
        "experiment_name": config.get("experiment_name"),
        "model_family": model_family,
        "model_variant": config.get("model_variant"),
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "split_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "dataset_root_recorded": False,
        "seed": config.get("seed"),
        "image_size": list(image_size),
        "batch_size": config.get("batch_size"),
        "epochs": config.get("epochs"),
        "checkpoint_metric": config.get("checkpoint_metric"),
        "git_commit": git_commit(),
        "package_versions": package_versions(extra_packages),
        "class_to_index": class_to_index,
        "split_distribution": split_distribution(records),
        "local_file_check_train_val_only": {
            "checked": file_check.checked,
            "existing": file_check.existing,
            "missing": file_check.missing,
            "missing_examples": list(file_check.missing_examples),
        },
        "smoke_records": smoke_records,
        "artifact_policy": config.get("artifact_policy", {}),
    }
    if extra_metadata:
        metadata["model_specific"] = extra_metadata

    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return PreparedRun(
        config=config,
        config_path=config_path,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
        class_to_index=class_to_index,
        metadata_path=metadata_path,
        metadata=metadata,
    )
