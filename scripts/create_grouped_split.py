"""Create a grouped train/val/test split manifest for class-folder images.

The script reads local image metadata only. It does not copy images, generate
patches, load models, download weights, create checkpoints, or start training.
All output paths in the manifest are relative to the provided dataset root.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_GROUP_REGEX = r"^(?P<group>.+?)[_-](?:q|quarter|viertel)[_-]?[1-4]$"
SPLIT_NAMES = ("train", "val", "test")
MANIFEST_COLUMNS = [
    "image_id",
    "relative_path",
    "label",
    "split",
    "group_id",
    "group_source",
    "file_extension",
    "width",
    "height",
    "channels",
]
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    relative_path: str
    label: str
    group_id: str
    group_source: str
    file_extension: str
    width: int
    height: int
    channels: int


@dataclass(frozen=True)
class GroupRecord:
    group_id: str
    records: tuple[ImageRecord, ...]
    label_counts: dict[str, int]

    @property
    def image_count(self) -> int:
        return len(self.records)


def parse_extensions(raw: str) -> set[str]:
    values = []
    for part in raw.split(","):
        item = part.strip().lower()
        if not item:
            continue
        values.append(item if item.startswith(".") else f".{item}")
    return set(values)


def parse_ratios(raw: str) -> dict[str, float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != 3:
        raise ValueError("--ratios must contain exactly three values: train,val,test")
    if any(value <= 0 for value in values):
        raise ValueError("--ratios values must be positive")
    total = sum(values)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("--ratios must sum to 1.0")
    return dict(zip(SPLIT_NAMES, values, strict=True))


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
    packages = ["Pillow"]
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def stable_image_id(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").lower()
    digest = hashlib.md5(normalized.encode("utf-8"))
    return f"img_{digest.hexdigest()[:16]}"


def infer_group_id(stem: str, group_regex: re.Pattern[str]) -> tuple[str, str]:
    match = group_regex.match(stem)
    if match:
        if match.groupdict():
            group = next(iter(match.groupdict().values()))
        elif match.groups():
            group = match.group(1)
        else:
            group = match.group(0)
        group = group.strip("_-. ")
        if group:
            return group, "quarter_suffix_removed"
    return stem, "filename_stem_no_quarter_suffix"


def channel_count(image: Image.Image) -> int:
    try:
        return len(image.getbands())
    except Exception:
        return 0


def discover_image_paths(data_root: Path, extensions: set[str]) -> list[tuple[Path, str]]:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {data_root}")

    class_dirs = sorted(
        path
        for path in data_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not class_dirs:
        raise ValueError(f"No class folders found below dataset root: {data_root}")

    discovered: list[tuple[Path, str]] = []
    for class_dir in class_dirs:
        label = class_dir.name
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                discovered.append((path, label))
    if not discovered:
        raise ValueError(f"No image files with configured extensions found below: {data_root}")
    return discovered


def inspect_image(path: Path, label: str, data_root: Path, group_regex: re.Pattern[str]) -> ImageRecord:
    relative_path = path.relative_to(data_root).as_posix()
    if Path(relative_path).is_absolute() or relative_path.startswith("../"):
        raise ValueError(f"Relative path escaped dataset root: {relative_path}")

    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            channels = channel_count(image)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Unreadable image file: {relative_path} ({type(exc).__name__})") from exc

    group_id, group_source = infer_group_id(path.stem, group_regex)
    return ImageRecord(
        image_id=stable_image_id(relative_path),
        relative_path=relative_path,
        label=label,
        group_id=group_id,
        group_source=group_source,
        file_extension=path.suffix.lower(),
        width=width,
        height=height,
        channels=channels,
    )


def load_records(args: argparse.Namespace) -> list[ImageRecord]:
    data_root = Path(args.data_root).expanduser().resolve()
    extensions = parse_extensions(args.extensions)
    group_regex = re.compile(args.group_regex, re.IGNORECASE)
    discovered = discover_image_paths(data_root=data_root, extensions=extensions)
    return [
        inspect_image(path=path, label=label, data_root=data_root, group_regex=group_regex)
        for path, label in discovered
    ]


def build_groups(records: list[ImageRecord]) -> list[GroupRecord]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)

    groups = []
    for group_id, group_records in grouped.items():
        label_counts = Counter(record.label for record in group_records)
        groups.append(
            GroupRecord(
                group_id=group_id,
                records=tuple(sorted(group_records, key=lambda record: record.relative_path)),
                label_counts=dict(sorted(label_counts.items())),
            )
        )
    return sorted(groups, key=lambda group: group.group_id)


def empty_split_stats(labels: list[str]) -> dict[str, dict[str, Any]]:
    return {
        split: {
            "images": 0,
            "groups": 0,
            "labels": {label: 0 for label in labels},
        }
        for split in SPLIT_NAMES
    }


def add_group_to_stats(stats: dict[str, dict[str, Any]], split: str, group: GroupRecord, multiplier: int) -> None:
    stats[split]["images"] += multiplier * group.image_count
    stats[split]["groups"] += multiplier
    for label, count in group.label_counts.items():
        stats[split]["labels"][label] += multiplier * count


def score_stats(
    stats: dict[str, dict[str, Any]],
    ratios: dict[str, float],
    total_images: int,
    total_groups: int,
    total_labels: dict[str, int],
) -> float:
    score = 0.0
    for split, ratio in ratios.items():
        target_images = total_images * ratio
        target_groups = total_groups * ratio
        image_error = stats[split]["images"] - target_images
        group_error = stats[split]["groups"] - target_groups
        score += 2.0 * (image_error * image_error) / max(target_images, 1.0)
        score += 0.5 * (group_error * group_error) / max(target_groups, 1.0)

        for label, total_label_count in total_labels.items():
            target = total_label_count * ratio
            label_error = stats[split]["labels"][label] - target
            score += (label_error * label_error) / max(target, 1.0)
    return score


def assign_grouped_split(
    groups: list[GroupRecord],
    ratios: dict[str, float],
    seed: int,
    local_search_iterations: int,
) -> dict[str, str]:
    labels = sorted({label for group in groups for label in group.label_counts})
    total_images = sum(group.image_count for group in groups)
    total_groups = len(groups)
    total_labels = Counter()
    for group in groups:
        total_labels.update(group.label_counts)

    rng = random.Random(seed)
    stats = empty_split_stats(labels)
    assignments: dict[str, str] = {}
    shuffled_groups = groups[:]
    rng.shuffle(shuffled_groups)
    ordered_groups = sorted(
        shuffled_groups,
        key=lambda group: (-len(group.label_counts), -group.image_count, group.group_id),
    )

    for group in ordered_groups:
        best_split = None
        best_score = None
        for split in SPLIT_NAMES:
            add_group_to_stats(stats, split, group, 1)
            score = score_stats(stats, ratios, total_images, total_groups, dict(total_labels))
            add_group_to_stats(stats, split, group, -1)
            if best_score is None or score < best_score:
                best_score = score
                best_split = split
        assert best_split is not None
        assignments[group.group_id] = best_split
        add_group_to_stats(stats, best_split, group, 1)

    current_score = score_stats(stats, ratios, total_images, total_groups, dict(total_labels))
    for _iteration in range(local_search_iterations):
        improved = False
        search_groups = groups[:]
        rng.shuffle(search_groups)
        for group in search_groups:
            current_split = assignments[group.group_id]
            best_split = current_split
            best_score = current_score
            add_group_to_stats(stats, current_split, group, -1)
            for candidate_split in SPLIT_NAMES:
                if candidate_split == current_split:
                    continue
                add_group_to_stats(stats, candidate_split, group, 1)
                candidate_score = score_stats(stats, ratios, total_images, total_groups, dict(total_labels))
                add_group_to_stats(stats, candidate_split, group, -1)
                if candidate_score + 1e-12 < best_score:
                    best_score = candidate_score
                    best_split = candidate_split
            add_group_to_stats(stats, best_split, group, 1)
            assignments[group.group_id] = best_split
            if best_split != current_split:
                current_score = best_score
                improved = True
        if not improved:
            break

    return assignments


def build_manifest_rows(records: list[ImageRecord], assignments: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for record in sorted(records, key=lambda item: item.relative_path):
        rows.append(
            {
                "image_id": record.image_id,
                "relative_path": record.relative_path,
                "label": record.label,
                "split": assignments[record.group_id],
                "group_id": record.group_id,
                "group_source": record.group_source,
                "file_extension": record.file_extension,
                "width": record.width,
                "height": record.height,
                "channels": record.channels,
            }
        )
    return rows


def split_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {
        split: {"images": 0, "groups": set(), "class_distribution": Counter()}
        for split in SPLIT_NAMES
    }
    for row in rows:
        split = str(row["split"])
        by_split[split]["images"] += 1
        by_split[split]["groups"].add(str(row["group_id"]))
        by_split[split]["class_distribution"][str(row["label"])] += 1

    total_images = len(rows)
    result: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        result[split] = {
            "images": by_split[split]["images"],
            "groups": len(by_split[split]["groups"]),
            "image_share": by_split[split]["images"] / total_images if total_images else 0.0,
            "class_distribution": dict(sorted(by_split[split]["class_distribution"].items())),
        }
    return result


def validate_manifest(rows: list[dict[str, Any]], records: list[ImageRecord]) -> dict[str, Any]:
    group_to_splits: dict[str, set[str]] = defaultdict(set)
    image_ids = []
    relative_paths = []
    for row in rows:
        group_to_splits[str(row["group_id"])].add(str(row["split"]))
        image_ids.append(str(row["image_id"]))
        relative_paths.append(str(row["relative_path"]))
        if Path(str(row["relative_path"])).is_absolute():
            raise ValueError(f"Manifest contains absolute path: {row['relative_path']}")

    leaking_groups = {
        group_id: sorted(splits)
        for group_id, splits in group_to_splits.items()
        if len(splits) > 1
    }
    duplicate_image_ids = sorted(
        image_id for image_id, count in Counter(image_ids).items() if count > 1
    )
    duplicate_relative_paths = sorted(
        relative_path for relative_path, count in Counter(relative_paths).items() if count > 1
    )
    return {
        "no_group_id_in_multiple_splits": not leaking_groups,
        "leaking_group_count": len(leaking_groups),
        "all_images_assigned_once": len(rows) == len(records)
        and not duplicate_image_ids
        and not duplicate_relative_paths,
        "manifest_rows": len(rows),
        "input_records": len(records),
        "duplicate_image_id_count": len(duplicate_image_ids),
        "duplicate_relative_path_count": len(duplicate_relative_paths),
    }


def build_summary(
    rows: list[dict[str, Any]],
    records: list[ImageRecord],
    groups: list[GroupRecord],
    ratios: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    labels = sorted({record.label for record in records})
    class_distribution = Counter(record.label for record in records)
    group_source_counts = Counter(record.group_source for record in records)
    multi_label_groups = [
        group for group in groups if len(group.label_counts) > 1
    ]
    summary = {
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "script": "scripts/create_grouped_split.py",
        "method_note": (
            "Grouped split manifest from local image metadata only. No image data is "
            "copied, no patches are generated, and no model training is started."
        ),
        "dataset_root": {
            "local_path_recorded": False,
            "note": "Absolute local dataset paths are intentionally not stored in this summary.",
        },
        "seed": args.seed,
        "ratios": ratios,
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "grouping": {
            "regex": args.group_regex,
            "logic": "Remove q1-q4, quarter, or viertel suffix from filename stem; fall back to full stem.",
            "images_with_quarter_suffix": group_source_counts.get("quarter_suffix_removed", 0),
            "images_without_detected_quarter_suffix": group_source_counts.get(
                "filename_stem_no_quarter_suffix",
                0,
            ),
            "multi_label_group_count": len(multi_label_groups),
        },
        "dataset": {
            "total_images": len(records),
            "num_classes": len(labels),
            "class_names": labels,
            "class_distribution": dict(sorted(class_distribution.items())),
            "num_groups": len(groups),
        },
        "splits": split_counts(rows),
        "checks": validate_manifest(rows, records),
    }
    return summary


def markdown_table(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    separator = ["---" for _ in header]
    lines = [
        "| " + " | ".join(str(value) for value in header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def format_share(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def summary_to_markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    grouping = summary["grouping"]
    checks = summary["checks"]
    split_rows = [["Split", "Bilder", "Gruppen", "Anteil"]]
    for split in SPLIT_NAMES:
        split_info = summary["splits"][split]
        split_rows.append(
            [
                split,
                split_info["images"],
                split_info["groups"],
                format_share(split_info["image_share"]),
            ]
        )

    class_rows = [["Klasse", "Gesamt"]]
    for label, count in dataset["class_distribution"].items():
        class_rows.append([label, count])

    per_split_class_rows = [["Split", "Klasse", "Bilder"]]
    for split in SPLIT_NAMES:
        for label, count in summary["splits"][split]["class_distribution"].items():
            per_split_class_rows.append([split, label, count])

    check_rows = [
        ["Pruefung", "Ergebnis"],
        ["Keine group_id in mehreren Splits", checks["no_group_id_in_multiple_splits"]],
        ["Alle Bilder genau einmal zugeordnet", checks["all_images_assigned_once"]],
        ["Leaking group count", checks["leaking_group_count"]],
        ["Duplicate image_id count", checks["duplicate_image_id_count"]],
        ["Duplicate relative_path count", checks["duplicate_relative_path_count"]],
    ]

    return "\n".join(
        [
            "# Datensatz-Split-Summary",
            "",
            f"Erzeugt am: `{summary['generated_at_utc']}`",
            "",
            "Diese Summary beschreibt den gruppierten Train/Val/Test-Split. Es wurden keine "
            "Bilddaten kopiert, keine Patches erzeugt und keine Modelle trainiert.",
            "",
            "## Datensatz",
            "",
            f"- Gesamtzahl Bilder: {dataset['total_images']}",
            f"- Anzahl Klassen: {dataset['num_classes']}",
            f"- Anzahl Gruppen: {dataset['num_groups']}",
            f"- Seed: {summary['seed']}",
            f"- Split-Verhaeltnis: train={summary['ratios']['train']}, "
            f"val={summary['ratios']['val']}, test={summary['ratios']['test']}",
            "",
            markdown_table(class_rows),
            "",
            "## Gruppierungslogik",
            "",
            f"- Regex: `{grouping['regex']}`",
            f"- Logik: {grouping['logic']}",
            f"- Bilder mit erkanntem q1-q4-/Viertel-Suffix: {grouping['images_with_quarter_suffix']}",
            f"- Bilder ohne erkanntes q1-q4-/Viertel-Suffix: {grouping['images_without_detected_quarter_suffix']}",
            f"- Gruppen mit mehr als einem Label: {grouping['multi_label_group_count']}",
            "",
            "## Split-Verteilung",
            "",
            markdown_table(split_rows),
            "",
            "## Klassenverteilung pro Split",
            "",
            markdown_table(per_split_class_rows),
            "",
            "## Pruefungen",
            "",
            markdown_table(check_rows),
            "",
            "## Versionierung",
            "",
            "Das Manifest enthaelt relative Pfade zum lokalen Dataset-Root. Vor einem Commit "
            "muss geprueft werden, ob diese relativen Dateinamen datenschutzrechtlich "
            "versionierbar sind.",
            "",
        ]
    )


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_output_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def run(args: argparse.Namespace) -> dict[str, Any]:
    ratios = parse_ratios(args.ratios)
    records = load_records(args)
    groups = build_groups(records)
    assignments = assign_grouped_split(
        groups=groups,
        ratios=ratios,
        seed=args.seed,
        local_search_iterations=args.local_search_iterations,
    )
    rows = build_manifest_rows(records, assignments)
    summary = build_summary(rows=rows, records=records, groups=groups, ratios=ratios, args=args)

    checks = summary["checks"]
    if not checks["no_group_id_in_multiple_splits"] or not checks["all_images_assigned_once"]:
        raise RuntimeError(f"Split validation failed: {checks}")

    manifest_path = resolve_output_path(args.output_manifest)
    summary_json_path = resolve_output_path(args.summary_json)
    summary_md_path = resolve_output_path(args.summary_md)
    write_manifest(manifest_path, rows)
    write_json(summary_json_path, summary)
    summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    summary_md_path.write_text(summary_to_markdown(summary), encoding="utf-8")

    return {
        "manifest": str(manifest_path),
        "summary_json": str(summary_json_path),
        "summary_md": str(summary_md_path),
        "total_images": summary["dataset"]["total_images"],
        "num_classes": summary["dataset"]["num_classes"],
        "num_groups": summary["dataset"]["num_groups"],
        "splits": summary["splits"],
        "checks": summary["checks"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/raw",
        help="Local dataset root containing one folder per class.",
    )
    parser.add_argument(
        "--output-manifest",
        default="data/splits/bmw25_grouped_split_manifest.csv",
        help="Output CSV split manifest path.",
    )
    parser.add_argument(
        "--summary-json",
        default="data/splits/bmw25_grouped_split_summary.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--summary-md",
        default="docs/dataset_split_summary.md",
        help="Output Markdown summary path.",
    )
    parser.add_argument(
        "--ratios",
        default="0.70,0.15,0.15",
        help="Comma-separated train,val,test ratios.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic tie-breaking and local search order.",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated image extensions to include.",
    )
    parser.add_argument(
        "--group-regex",
        default=DEFAULT_GROUP_REGEX,
        help="Regex for stripping q1-q4 or quarter suffixes from filename stems.",
    )
    parser.add_argument(
        "--local-search-iterations",
        type=int,
        default=25,
        help="Number of deterministic local-search passes after greedy assignment.",
    )
    args = parser.parse_args()

    if args.local_search_iterations < 0:
        parser.error("--local-search-iterations must be non-negative")

    try:
        result = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
