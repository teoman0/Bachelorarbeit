"""Audit a local class-folder image dataset before training.

The script performs no training and no model evaluation. It inventories images
below class folders, checks basic integrity, writes small report tables, and
creates diagnostic figures for early dataset-suitability assessment.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib import metadata
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
INVENTORY_COLUMNS = [
    "image_id",
    "relative_path",
    "label",
    "width",
    "height",
    "channels",
    "file_extension",
    "file_size_bytes",
    "md5_hash",
    "perceptual_hash",
    "possible_group_id",
    "notes",
]
SUMMARY_COLUMNS = [
    "class_name",
    "num_images",
    "min_width",
    "max_width",
    "min_height",
    "max_height",
    "file_extensions",
    "num_corrupt_images",
]
DUPLICATE_COLUMNS = [
    "duplicate_type",
    "image_id_a",
    "image_id_b",
    "relative_path_a",
    "relative_path_b",
    "label_a",
    "label_b",
    "md5_hash",
    "perceptual_hash_a",
    "perceptual_hash_b",
    "hamming_distance",
    "notes",
]


@dataclass
class ImageRecord:
    image_id: str
    path: Path
    relative_path: str
    label: str
    width: int | None
    height: int | None
    channels: int | None
    file_extension: str
    file_size_bytes: int
    md5_hash: str
    perceptual_hash: str | None
    perceptual_hash_int: int | None
    possible_group_id: str | None
    notes: str
    corrupt: bool


class BKNode:
    def __init__(self, index: int, value: int) -> None:
        self.index = index
        self.value = value
        self.children: dict[int, BKNode] = {}


class BKTree:
    def __init__(self) -> None:
        self.root: BKNode | None = None

    def insert(self, index: int, value: int) -> None:
        if self.root is None:
            self.root = BKNode(index=index, value=value)
            return
        node = self.root
        while True:
            distance = hamming_distance_int(value, node.value)
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(index=index, value=value)
                return
            node = child

    def query(self, value: int, max_distance: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        matches: list[tuple[int, int]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming_distance_int(value, node.value)
            if distance <= max_distance:
                matches.append((node.index, distance))
            lower = distance - max_distance
            upper = distance + max_distance
            for child_distance, child in node.children.items():
                if lower <= child_distance <= upper:
                    stack.append(child)
        return matches


def parse_extensions(raw: str) -> set[str]:
    values = []
    for part in raw.split(","):
        item = part.strip().lower()
        if not item:
            continue
        values.append(item if item.startswith(".") else f".{item}")
    return set(values)


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def package_versions() -> dict[str, str | None]:
    packages = ["Pillow", "matplotlib", "numpy", "pandas"]
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def image_id_from_relative_path(relative_path: str) -> str:
    digest = hashlib.md5(relative_path.replace("\\", "/").lower().encode("utf-8"))
    return f"img_{digest.hexdigest()[:16]}"


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def channel_count(image: Image.Image) -> int:
    try:
        return len(image.getbands())
    except Exception:
        return 0


def perceptual_dhash(image: Image.Image, hash_size: int) -> tuple[str, int]:
    grayscale = image.convert("L").resize(
        (hash_size + 1, hash_size),
        Image.Resampling.LANCZOS,
    )
    pixels = grayscale.tobytes()
    bits: list[int] = []
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            bits.append(1 if pixels[offset + col] > pixels[offset + col + 1] else 0)

    value = 0
    for bit in bits:
        value = (value << 1) | bit
    hex_width = math.ceil(len(bits) / 4)
    return f"dhash{len(bits)}:{value:0{hex_width}x}", value


def hamming_distance_int(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def infer_group_id(stem: str, group_regex: str | None, use_heuristic: bool) -> tuple[str | None, str | None]:
    if group_regex:
        match = re.search(group_regex, stem)
        if match:
            if match.groupdict():
                group = next(iter(match.groupdict().values()))
            elif match.groups():
                group = match.group(1)
            else:
                group = match.group(0)
            group = group.strip("_-. ")
            if group:
                return group, "group_id_from_regex"

    if not use_heuristic:
        return None, None

    patterns = [
        r"(?i)(.*?)(?:[_-](?:patch|tile|crop)[_-]?\d+(?:[_-]\d+)*)$",
        r"(?i)(.*?)(?:[_-](?:img|image|bild|aufnahme|photo|foto|frame|view|sample)[_-]?\d+)$",
        r"(?i)(.*?)(?:[_-](?:q|quarter|viertel)[_-]?[1-4])$",
        r"(.*?)(?:[_-]\d{1,4})$",
    ]
    for pattern in patterns:
        match = re.match(pattern, stem)
        if not match:
            continue
        group = match.group(1).strip("_-. ")
        if group and group != stem and len(group) >= 3:
            return group, "group_id_from_filename_heuristic"
    return None, None


def discover_image_paths(data_root: Path, extensions: set[str]) -> list[tuple[Path, str]]:
    class_dirs = sorted(
        path
        for path in data_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not class_dirs:
        raise ValueError(f"No class folders found below data root: {data_root}")

    discovered: list[tuple[Path, str]] = []
    for class_dir in class_dirs:
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                discovered.append((path, class_dir.name))
    return discovered


def inspect_image(
    path: Path,
    label: str,
    data_root: Path,
    phash_size: int,
    group_regex: str | None,
    use_group_heuristic: bool,
) -> ImageRecord:
    relative_path = path.relative_to(data_root).as_posix()
    image_id = image_id_from_relative_path(relative_path)
    file_extension = path.suffix.lower()
    file_size_bytes = path.stat().st_size
    md5_hash = md5_file(path)
    possible_group_id, group_note = infer_group_id(
        path.stem,
        group_regex=group_regex,
        use_heuristic=use_group_heuristic,
    )
    notes = [group_note] if group_note else []

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            channels = channel_count(image)
            perceptual_hash, perceptual_hash_int = perceptual_dhash(
                image,
                hash_size=phash_size,
            )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        notes.append(f"corrupt_or_unreadable:{type(exc).__name__}")
        return ImageRecord(
            image_id=image_id,
            path=path,
            relative_path=relative_path,
            label=label,
            width=None,
            height=None,
            channels=None,
            file_extension=file_extension,
            file_size_bytes=file_size_bytes,
            md5_hash=md5_hash,
            perceptual_hash=None,
            perceptual_hash_int=None,
            possible_group_id=possible_group_id,
            notes=";".join(notes),
            corrupt=True,
        )

    return ImageRecord(
        image_id=image_id,
        path=path,
        relative_path=relative_path,
        label=label,
        width=width,
        height=height,
        channels=channels,
        file_extension=file_extension,
        file_size_bytes=file_size_bytes,
        md5_hash=md5_hash,
        perceptual_hash=perceptual_hash,
        perceptual_hash_int=perceptual_hash_int,
        possible_group_id=possible_group_id,
        notes=";".join(notes),
        corrupt=False,
    )


def inventory_row(record: ImageRecord) -> dict[str, Any]:
    return {
        "image_id": record.image_id,
        "relative_path": record.relative_path,
        "label": record.label,
        "width": record.width,
        "height": record.height,
        "channels": record.channels,
        "file_extension": record.file_extension,
        "file_size_bytes": record.file_size_bytes,
        "md5_hash": record.md5_hash,
        "perceptual_hash": record.perceptual_hash,
        "possible_group_id": record.possible_group_id,
        "notes": record.notes,
    }


def build_dataset_summary(records: list[ImageRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = sorted({record.label for record in records})
    for label in labels:
        class_records = [record for record in records if record.label == label]
        valid = [record for record in class_records if not record.corrupt]
        widths = [record.width for record in valid if record.width is not None]
        heights = [record.height for record in valid if record.height is not None]
        rows.append(
            {
                "class_name": label,
                "num_images": len(class_records),
                "min_width": min(widths) if widths else None,
                "max_width": max(widths) if widths else None,
                "min_height": min(heights) if heights else None,
                "max_height": max(heights) if heights else None,
                "file_extensions": ";".join(
                    sorted({record.file_extension for record in class_records})
                ),
                "num_corrupt_images": sum(record.corrupt for record in class_records),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})


def build_duplicate_rows(
    records: list[ImageRecord],
    phash_threshold: int,
    max_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exact_pairs: set[tuple[str, str]] = set()
    truncated = False

    by_md5: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_md5[record.md5_hash].append(record)

    for md5_hash, group in sorted(by_md5.items()):
        if len(group) < 2:
            continue
        for left, right in combinations(sorted(group, key=lambda r: r.relative_path), 2):
            pair = tuple(sorted((left.image_id, right.image_id)))
            exact_pairs.add(pair)
            rows.append(
                {
                    "duplicate_type": "identical_md5",
                    "image_id_a": left.image_id,
                    "image_id_b": right.image_id,
                    "relative_path_a": left.relative_path,
                    "relative_path_b": right.relative_path,
                    "label_a": left.label,
                    "label_b": right.label,
                    "md5_hash": md5_hash,
                    "perceptual_hash_a": left.perceptual_hash,
                    "perceptual_hash_b": right.perceptual_hash,
                    "hamming_distance": (
                        hamming_distance_int(left.perceptual_hash_int, right.perceptual_hash_int)
                        if left.perceptual_hash_int is not None
                        and right.perceptual_hash_int is not None
                        else None
                    ),
                    "notes": "same_file_content",
                }
            )
            if len(rows) >= max_rows:
                truncated = True
                return rows, {"duplicate_rows_truncated": truncated, "max_duplicate_rows": max_rows}

    tree = BKTree()
    phash_records = [
        record for record in records if not record.corrupt and record.perceptual_hash_int is not None
    ]
    for index, record in enumerate(phash_records):
        assert record.perceptual_hash_int is not None
        for previous_index, distance in tree.query(record.perceptual_hash_int, phash_threshold):
            previous = phash_records[previous_index]
            pair = tuple(sorted((previous.image_id, record.image_id)))
            if pair in exact_pairs:
                continue
            rows.append(
                {
                    "duplicate_type": "similar_perceptual_hash",
                    "image_id_a": previous.image_id,
                    "image_id_b": record.image_id,
                    "relative_path_a": previous.relative_path,
                    "relative_path_b": record.relative_path,
                    "label_a": previous.label,
                    "label_b": record.label,
                    "md5_hash": "",
                    "perceptual_hash_a": previous.perceptual_hash,
                    "perceptual_hash_b": record.perceptual_hash,
                    "hamming_distance": distance,
                    "notes": f"dhash_hamming_distance_le_{phash_threshold}",
                }
            )
            if len(rows) >= max_rows:
                truncated = True
                return rows, {"duplicate_rows_truncated": truncated, "max_duplicate_rows": max_rows}
        tree.insert(index=index, value=record.perceptual_hash_int)

    return rows, {"duplicate_rows_truncated": truncated, "max_duplicate_rows": max_rows}


def save_class_distribution(summary_rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["class_name"]) for row in summary_rows]
    counts = [int(row["num_images"]) for row in summary_rows]
    corrupt_counts = [int(row["num_corrupt_images"]) for row in summary_rows]
    valid_counts = [count - corrupt for count, corrupt in zip(counts, corrupt_counts, strict=True)]

    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(labels)), 4.0))
    ax.bar(labels, valid_counts, color="#4C78A8", label="readable")
    if any(corrupt_counts):
        ax.bar(labels, corrupt_counts, bottom=valid_counts, color="#E45756", label="corrupt")
        ax.legend(frameon=False)
    ax.set_title("Class distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of image files")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_image_size_distribution(records: list[ImageRecord], path: Path) -> None:
    valid = [record for record in records if not record.corrupt and record.width and record.height]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    if not valid:
        for ax in axes:
            ax.axis("off")
        axes[0].text(0.5, 0.5, "No readable images", ha="center", va="center")
    else:
        labels = sorted({record.label for record in valid})
        colors = plt.get_cmap("tab10")
        for index, label in enumerate(labels):
            subset = [record for record in valid if record.label == label]
            axes[0].scatter(
                [record.width for record in subset],
                [record.height for record in subset],
                s=24,
                alpha=0.75,
                label=label,
                color=colors(index % 10),
            )
        axes[0].set_title("Width vs. height")
        axes[0].set_xlabel("Width [px]")
        axes[0].set_ylabel("Height [px]")
        axes[0].grid(alpha=0.25)
        axes[0].legend(frameon=False, fontsize=8)

        megapixels = [
            (float(record.width) * float(record.height)) / 1_000_000.0 for record in valid
        ]
        bins = min(20, max(4, int(math.sqrt(len(megapixels)))))
        axes[1].hist(megapixels, bins=bins, color="#59A14F", edgecolor="white")
        axes[1].set_title("Image area")
        axes[1].set_xlabel("Megapixels")
        axes[1].set_ylabel("Number of images")
        axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def load_thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, (245, 245, 245))
        x = (size[0] - image.width) // 2
        y = (size[1] - image.height) // 2
        canvas.paste(image, (x, y))
        return canvas


def save_sample_grid(
    records: list[ImageRecord],
    path: Path,
    samples_per_class: int,
    seed: int,
) -> None:
    valid = [record for record in records if not record.corrupt]
    labels = sorted({record.label for record in valid})
    thumb_size = (180, 140)
    label_width = 190
    header_height = 24
    padding = 10
    row_height = thumb_size[1] + header_height + padding
    width = label_width + samples_per_class * (thumb_size[0] + padding) + padding
    height = max(1, len(labels)) * row_height + padding
    canvas = Image.new("RGB", (max(width, 640), max(height, 180)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    if not valid:
        draw.text((24, 76), "No readable images found.", fill=(30, 30, 30), font=font)
        canvas.save(path)
        return

    rng = random.Random(seed)
    for row_index, label in enumerate(labels):
        y0 = padding + row_index * row_height
        draw.text((padding, y0 + 8), label, fill=(20, 20, 20), font=font)
        subset = sorted([record for record in valid if record.label == label], key=lambda r: r.relative_path)
        selected = subset if len(subset) <= samples_per_class else rng.sample(subset, samples_per_class)
        selected = sorted(selected, key=lambda r: r.relative_path)
        for col_index, record in enumerate(selected):
            x = label_width + col_index * (thumb_size[0] + padding)
            thumbnail = load_thumbnail(record.path, thumb_size)
            canvas.paste(thumbnail, (x, y0))
            short_name = Path(record.relative_path).name
            if len(short_name) > 24:
                short_name = f"{short_name[:21]}..."
            draw.text((x, y0 + thumb_size[1] + 3), short_name, fill=(70, 70, 70), font=small_font)
    canvas.save(path)


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    records: list[ImageRecord],
    duplicate_metadata: dict[str, Any],
) -> None:
    label_counts = Counter(record.label for record in records)
    group_counts = Counter(
        record.possible_group_id for record in records if record.possible_group_id
    )
    metadata_payload = {
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "script": "scripts/audit_dataset.py",
        "purpose": "Early structural dataset audit before training.",
        "method_note": (
            "No training, no segmentation, no patch generation, no model testing, "
            "and no test-set based model decision are performed."
        ),
        "data_root": str(Path(args.data_root)),
        "output_dir": str(Path(args.output_dir)),
        "seed": args.seed,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "package_versions": package_versions(),
        "effective_arguments": vars(args),
        "num_image_files": len(records),
        "num_readable_images": sum(not record.corrupt for record in records),
        "num_corrupt_images": sum(record.corrupt for record in records),
        "class_counts": dict(sorted(label_counts.items())),
        "possible_group_id_count": len(group_counts),
        "possible_group_id_examples": dict(group_counts.most_common(20)),
        "perceptual_hash": {
            "algorithm": "simple horizontal difference hash",
            "hash_size": args.phash_size,
            "similarity_threshold_hamming_distance": args.phash_threshold,
        },
        "duplicates": duplicate_metadata,
    }
    path.write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).expanduser()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    output_dir = Path(args.output_dir).expanduser()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    extensions = parse_extensions(args.extensions)
    discovered = discover_image_paths(data_root=data_root, extensions=extensions)
    records = [
        inspect_image(
            path=path,
            label=label,
            data_root=data_root,
            phash_size=args.phash_size,
            group_regex=args.group_regex,
            use_group_heuristic=not args.no_group_heuristic,
        )
        for path, label in discovered
    ]

    summary_rows = build_dataset_summary(records)
    duplicate_rows, duplicate_metadata = build_duplicate_rows(
        records=records,
        phash_threshold=args.phash_threshold,
        max_rows=args.max_duplicate_rows,
    )

    write_csv(tables_dir / "dataset_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(
        tables_dir / "image_inventory.csv",
        [inventory_row(record) for record in records],
        INVENTORY_COLUMNS,
    )
    write_csv(tables_dir / "potential_duplicates.csv", duplicate_rows, DUPLICATE_COLUMNS)

    save_class_distribution(summary_rows, figures_dir / "class_distribution.png")
    save_image_size_distribution(records, figures_dir / "image_size_distribution.png")
    save_sample_grid(
        records=records,
        path=figures_dir / "sample_grid_per_class.png",
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    write_metadata(
        tables_dir / "dataset_audit_metadata.json",
        args=args,
        records=records,
        duplicate_metadata=duplicate_metadata,
    )

    result = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "image_files": len(records),
        "readable_images": sum(not record.corrupt for record in records),
        "corrupt_images": sum(record.corrupt for record in records),
        "classes": len(summary_rows),
        "potential_duplicate_rows": len(duplicate_rows),
        "outputs": [
            str(tables_dir / "dataset_summary.csv"),
            str(tables_dir / "image_inventory.csv"),
            str(tables_dir / "potential_duplicates.csv"),
            str(tables_dir / "dataset_audit_metadata.json"),
            str(figures_dir / "class_distribution.png"),
            str(figures_dir / "image_size_distribution.png"),
            str(figures_dir / "sample_grid_per_class.png"),
        ],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/raw",
        help="Local dataset root containing one subfolder per class.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory where tables/ and figures/ are written.",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated image extensions to include.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic sample image selection.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=4,
        help="Maximum number of thumbnails per class in sample_grid_per_class.png.",
    )
    parser.add_argument(
        "--phash-size",
        type=int,
        default=8,
        help="Difference-hash size; 8 creates a 64-bit perceptual hash.",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=6,
        help="Maximum Hamming distance for potential perceptual duplicates.",
    )
    parser.add_argument(
        "--max-duplicate-rows",
        type=int,
        default=20000,
        help="Cap potential_duplicates.csv rows so reports stay small.",
    )
    parser.add_argument(
        "--group-regex",
        default=None,
        help=(
            "Optional regex applied to filename stems. The first named or numbered "
            "capture group is used as possible_group_id."
        ),
    )
    parser.add_argument(
        "--no-group-heuristic",
        action="store_true",
        help="Disable conservative filename suffix stripping for possible_group_id.",
    )
    args = parser.parse_args()
    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be at least 1.")
    if args.phash_size < 4:
        parser.error("--phash-size must be at least 4.")
    if args.phash_threshold < 0:
        parser.error("--phash-threshold must be non-negative.")
    if args.max_duplicate_rows < 1:
        parser.error("--max-duplicate-rows must be at least 1.")

    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
