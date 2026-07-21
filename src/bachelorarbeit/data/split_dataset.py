"""Dataset utilities for the grouped split manifest.

The helpers in this module only resolve local files from a versioned manifest.
They do not write absolute paths, create new splits, copy image data, or start
training.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps


REQUIRED_COLUMNS = {
    "image_id",
    "relative_path",
    "label",
    "split",
    "group_id",
}
VALID_SPLITS = {"train", "val", "test"}
SPLIT_ORDER = ("train", "val", "test")


@dataclass(frozen=True)
class ManifestRecord:
    image_id: str
    relative_path: str
    label: str
    split: str
    group_id: str
    file_extension: str | None = None
    width: int | None = None
    height: int | None = None
    channels: int | None = None


@dataclass(frozen=True)
class LocalFileCheck:
    checked: int
    existing: int
    missing: int
    missing_examples: tuple[str, ...]


class SimpleImageTransform:
    """Small PIL-only transform for smoke tests and skeleton pipelines."""

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        resize_mode: str = "resize_pad",
        convert_rgb: bool = True,
    ) -> None:
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.convert_rgb = convert_rgb

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image)
        if self.convert_rgb:
            image = image.convert("RGB")
        if self.resize_mode == "resize":
            return image.resize(self.image_size, Image.Resampling.BICUBIC)
        if self.resize_mode != "resize_pad":
            raise ValueError(f"Unsupported resize_mode: {self.resize_mode}")
        return resize_with_padding(image, self.image_size)


class SplitImageDataset:
    """Minimal PIL dataset backed by the split manifest."""

    def __init__(
        self,
        records: list[ManifestRecord],
        dataset_root: Path,
        class_to_index: dict[str, int],
        transform: SimpleImageTransform | None = None,
    ) -> None:
        self.records = records
        self.dataset_root = Path(dataset_root)
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Image.Image, int, ManifestRecord]:
        record = self.records[index]
        path = resolve_record_path(self.dataset_root, record)
        with Image.open(path) as image:
            image = image.copy()
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_index[record.label], record


def read_split_manifest(path: Path) -> list[ManifestRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Split manifest is not a file: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"Split manifest is missing columns: {missing_columns}")
        records = [record_from_row(row) for row in reader]

    if not records:
        raise ValueError(f"Split manifest contains no rows: {path}")
    return records


def record_from_row(row: dict[str, str]) -> ManifestRecord:
    split = row["split"].strip()
    if split not in VALID_SPLITS:
        raise ValueError(f"Unexpected split value: {split}")
    relative_path = row["relative_path"].strip()
    if Path(relative_path).is_absolute() or relative_path.startswith("../"):
        raise ValueError(f"Manifest relative_path is not repository-safe: {relative_path}")
    return ManifestRecord(
        image_id=row["image_id"].strip(),
        relative_path=relative_path,
        label=row["label"].strip(),
        split=split,
        group_id=row["group_id"].strip(),
        file_extension=optional_text(row.get("file_extension")),
        width=optional_int(row.get("width")),
        height=optional_int(row.get("height")),
        channels=optional_int(row.get("channels")),
    )


def optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


def filter_split(records: Iterable[ManifestRecord], split: str) -> list[ManifestRecord]:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported split: {split}")
    return [record for record in records if record.split == split]


def build_class_mapping(records: Iterable[ManifestRecord]) -> dict[str, int]:
    labels = sorted({record.label for record in records})
    if not labels:
        raise ValueError("Cannot build class mapping from empty records")
    return {label: index for index, label in enumerate(labels)}


def split_distribution(records: Iterable[ManifestRecord]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    all_records = list(records)
    for split in SPLIT_ORDER:
        split_records = [record for record in all_records if record.split == split]
        result[split] = {
            "images": len(split_records),
            "groups": len({record.group_id for record in split_records}),
            "class_distribution": dict(sorted(Counter(record.label for record in split_records).items())),
        }
    return result


def resolve_record_path(dataset_root: Path, record: ManifestRecord) -> Path:
    root = Path(dataset_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")
    relative = Path(record.relative_path)
    if relative.is_absolute() or str(relative).startswith(".."):
        raise ValueError(f"Refusing unsafe manifest path: {record.relative_path}")
    return root / relative


def check_local_files(
    records: Iterable[ManifestRecord],
    dataset_root: Path,
    max_examples: int = 10,
) -> LocalFileCheck:
    checked = 0
    existing = 0
    missing_examples: list[str] = []
    for record in records:
        checked += 1
        path = resolve_record_path(dataset_root, record)
        if path.exists() and path.is_file():
            existing += 1
            continue
        if len(missing_examples) < max_examples:
            missing_examples.append(record.relative_path)
    return LocalFileCheck(
        checked=checked,
        existing=existing,
        missing=checked - existing,
        missing_examples=tuple(missing_examples),
    )


def resize_with_padding(image: Image.Image, image_size: tuple[int, int]) -> Image.Image:
    target_width, target_height = image_size
    source_width, source_height = image.size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Cannot resize image with non-positive dimensions")
    scale = min(target_width / source_width, target_height / source_height)
    resized_size = (
        max(1, int(round(source_width * scale))),
        max(1, int(round(source_height * scale))),
    )
    resized = image.resize(resized_size, Image.Resampling.BICUBIC)
    canvas = Image.new(resized.mode, image_size)
    offset = (
        (target_width - resized_size[0]) // 2,
        (target_height - resized_size[1]) // 2,
    )
    canvas.paste(resized, offset)
    return canvas


def prepare_dinov3_image(
    image: Image.Image,
    image_size: tuple[int, int] = (224, 224),
) -> Image.Image:
    """Apply the canonical global DINOv3 image preparation used for training."""

    image = ImageOps.exif_transpose(image).convert("RGB")
    return resize_with_padding(image, image_size)
