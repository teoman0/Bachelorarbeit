"""Prepare local CVAT rectangle annotations for region-based analysis.

The script converts CVAT rectangle annotations into a compact region table
that can later be used for local image-region scoring. It does not train
models, create splits, evaluate the test set, or copy image data unless crops
are explicitly requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = "configs/experiments/cvat_region_analysis.yaml"
REGION_COLUMNS = [
    "region_id",
    "source_image",
    "original_image_name",
    "group_id",
    "split",
    "original_label",
    "mapped_label",
    "is_global_class",
    "image_width",
    "image_height",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_area_ratio",
    "clipped",
    "matched_manifest",
    "exclude_reason",
]


@dataclass(frozen=True)
class FrameInfo:
    frame_index: int
    name: str
    width: int
    height: int


@dataclass(frozen=True)
class ManifestMatch:
    split: str
    group_id: str
    matched: bool
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config.")
    parser.add_argument(
        "--manual-root",
        default=None,
        help="Local manual_all root. If omitted, the config environment variable is used.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Read and report only; write no files.")
    parser.add_argument("--allow-export", action="store_true", help="Write region CSV and summary JSON locally.")
    parser.add_argument(
        "--allow-export-crops",
        action="store_true",
        help="Also write cropped regions locally. Implies --allow-export.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "all"],
        default=None,
        help="Filter regions by manifest split. 'all' inventories train/val/test/unmatched.",
    )
    parser.add_argument(
        "--include-nicht-bewertbar",
        action="store_true",
        help="Include the special Nicht_bewertbar label in exported rows.",
    )
    parser.add_argument(
        "--exclude-unmatched",
        action="store_true",
        help="Exclude regions whose source image cannot be matched to the split manifest.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return loaded


def resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_manual_root(config: dict[str, Any], override: str | None) -> Path:
    if override:
        root = Path(override).expanduser()
    else:
        env_name = str(config.get("inputs", {}).get("manual_root_env", "BMW25_MANUAL_ALL_ROOT"))
        env_value = os.environ.get(env_name)
        if not env_value:
            raise ValueError(
                f"Manual root is required. Set {env_name} or pass --manual-root."
            )
        root = Path(env_value).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Manual root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Manual root is not a directory: {root}")
    return root


def config_path(config: dict[str, Any], key: str, base: Path, section: str = "inputs") -> Path:
    raw = config.get(section, {}).get(key)
    if raw is None:
        raise KeyError(f"Config is missing {section}.{key}")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return base / path


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_frame_meta(path: Path) -> dict[int, FrameInfo]:
    payload = load_json(path)
    frames = payload.get("frames") if isinstance(payload, dict) else payload
    if not isinstance(frames, list):
        raise ValueError(f"frame_meta must contain a frames list: {path}")

    frame_infos: dict[int, FrameInfo] = {}
    for index, frame in enumerate(frames):
        try:
            frame_infos[index] = FrameInfo(
                frame_index=index,
                name=str(frame["name"]),
                width=int(frame["width"]),
                height=int(frame["height"]),
            )
        except KeyError as exc:
            raise ValueError(f"Frame {index} is missing key {exc!s}") from exc
    return frame_infos


def read_labels(path: Path) -> dict[int, str]:
    labels = load_json(path)
    if not isinstance(labels, list):
        raise ValueError(f"labels.json must contain a list: {path}")
    label_by_id: dict[int, str] = {}
    for item in labels:
        label_by_id[int(item["id"])] = str(item["name"])
    return label_by_id


def read_manual_manifest(path: Path) -> dict[str, bool]:
    if not path.exists():
        raise FileNotFoundError(f"Manual manifest does not exist: {path}")
    image_exists: dict[str, bool] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image", "local_image_exists"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manual manifest is missing columns: {sorted(missing)}")
        for row in reader:
            image_name = Path(str(row["image"])).name
            image_exists[image_name] = str(row["local_image_exists"]).strip().lower() == "true"
    return image_exists


def canonical_original_name(source_image: str, config: dict[str, Any]) -> str:
    pattern = str(config.get("matching", {}).get("manual_prefix_regex", r"(?i)^manual[_-]v\d+[_-]+"))
    return re.sub(pattern, "", Path(source_image).name)


def derive_group_id(filename: str, config: dict[str, Any]) -> str:
    name = canonical_original_name(filename, config)
    stem = Path(name).stem
    stem = stem.lstrip("_")
    quarter_pattern = str(
        config.get("matching", {}).get(
            "quarter_suffix_regex",
            r"(?i)(?:[_-]q[1-4]|[_-]quarter[1-4]|[_-]viertel[1-4])$",
        )
    )
    stem = re.sub(quarter_pattern, "", stem)
    return stem


def read_split_manifest(path: Path, config: dict[str, Any]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")
    split_by_group: dict[str, set[str]] = defaultdict(set)
    split_by_name: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"relative_path", "split", "group_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
        for row in reader:
            split = str(row["split"])
            group_id = str(row["group_id"])
            basename = Path(str(row["relative_path"])).name
            split_by_group[group_id].add(split)
            split_by_name[basename].add(split)
            split_by_name[canonical_original_name(basename, config)].add(split)
    return split_by_group, split_by_name


def match_frame_to_manifest(
    frame_name: str,
    config: dict[str, Any],
    split_by_group: dict[str, set[str]],
    split_by_name: dict[str, set[str]],
) -> ManifestMatch:
    original_name = canonical_original_name(frame_name, config)
    group_id = derive_group_id(frame_name, config)
    candidate_splits = set(split_by_group.get(group_id, set()))
    candidate_splits.update(split_by_name.get(Path(frame_name).name, set()))
    candidate_splits.update(split_by_name.get(original_name, set()))

    if len(candidate_splits) == 1:
        return ManifestMatch(
            split=next(iter(candidate_splits)),
            group_id=group_id,
            matched=True,
            reason="",
        )
    if not candidate_splits:
        return ManifestMatch(
            split="unmatched",
            group_id=group_id,
            matched=False,
            reason="not_in_manifest",
        )
    return ManifestMatch(
        split="ambiguous",
        group_id=group_id,
        matched=False,
        reason="ambiguous_manifest_split",
    )


def clip_box(points: list[Any], width: int, height: int) -> tuple[float, float, float, float, bool]:
    if len(points) != 4:
        raise ValueError(f"Rectangle points must contain exactly 4 numbers: {points}")
    x0, y0, x1, y1 = [float(value) for value in points]
    raw_min_x, raw_max_x = sorted([x0, x1])
    raw_min_y, raw_max_y = sorted([y0, y1])
    x_min = min(max(raw_min_x, 0.0), float(width))
    y_min = min(max(raw_min_y, 0.0), float(height))
    x_max = min(max(raw_max_x, 0.0), float(width))
    y_max = min(max(raw_max_y, 0.0), float(height))
    clipped = (
        x_min != raw_min_x
        or y_min != raw_min_y
        or x_max != raw_max_x
        or y_max != raw_max_y
    )
    return x_min, y_min, x_max, y_max, clipped


def build_region_rows(
    config: dict[str, Any],
    cvat_payload: dict[str, Any],
    frames: dict[int, FrameInfo],
    label_by_id: dict[int, str],
    manual_image_exists: dict[str, bool],
    split_by_group: dict[str, set[str]],
    split_by_name: dict[str, set[str]],
) -> list[dict[str, Any]]:
    label_mapping = dict(config.get("label_mapping", {}))
    special_labels = set(config.get("special_labels", []))
    rows: list[dict[str, Any]] = []

    shapes = cvat_payload.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("CVAT payload must contain a shapes list.")

    for shape_index, shape in enumerate(shapes):
        if str(shape.get("type")) != "rectangle":
            continue
        frame_index = int(shape["frame"])
        frame = frames.get(frame_index)
        if frame is None:
            raise ValueError(f"Shape {shape.get('id', shape_index)} refers to unknown frame {frame_index}")

        label_id = int(shape["label_id"])
        original_label = label_by_id.get(label_id)
        if original_label is None:
            raise ValueError(f"Shape {shape.get('id', shape_index)} uses unknown label_id {label_id}")
        mapped_label = str(label_mapping.get(original_label, original_label))
        is_global_class = original_label not in special_labels
        original_name = canonical_original_name(frame.name, config)
        manifest_match = match_frame_to_manifest(frame.name, config, split_by_group, split_by_name)
        x_min, y_min, x_max, y_max, clipped = clip_box(shape["points"], frame.width, frame.height)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        bbox_area = bbox_width * bbox_height
        bbox_area_ratio = bbox_area / float(frame.width * frame.height) if frame.width and frame.height else 0.0
        reasons: list[str] = []
        if manifest_match.reason:
            reasons.append(manifest_match.reason)
        if frame.name not in manual_image_exists:
            reasons.append("not_in_manual_manifest")
        elif not manual_image_exists[frame.name]:
            reasons.append("source_image_missing")
        if bbox_width <= 0.0 or bbox_height <= 0.0:
            reasons.append("invalid_or_empty_box")
        if original_label in special_labels:
            reasons.append("special_label")

        rows.append(
            {
                "region_id": f"region_{frame_index:04d}_{shape_index:04d}",
                "source_image": frame.name,
                "original_image_name": original_name,
                "group_id": manifest_match.group_id,
                "split": manifest_match.split,
                "original_label": original_label,
                "mapped_label": mapped_label,
                "is_global_class": bool(is_global_class),
                "image_width": frame.width,
                "image_height": frame.height,
                "x_min": round(x_min, 6),
                "y_min": round(y_min, 6),
                "x_max": round(x_max, 6),
                "y_max": round(y_max, 6),
                "bbox_width": round(bbox_width, 6),
                "bbox_height": round(bbox_height, 6),
                "bbox_area": round(bbox_area, 6),
                "bbox_area_ratio": round(bbox_area_ratio, 10),
                "clipped": bool(clipped),
                "matched_manifest": bool(manifest_match.matched),
                "exclude_reason": ";".join(reasons),
            }
        )
    return rows


def filter_rows(
    rows: list[dict[str, Any]],
    split: str,
    include_nicht_bewertbar: bool,
    exclude_unmatched: bool,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if split != "all" and row["split"] != split:
            continue
        if exclude_unmatched and not bool(row["matched_manifest"]):
            continue
        if not include_nicht_bewertbar and not bool(row["is_global_class"]):
            continue
        filtered.append(row)
    return filtered


def nested_counter(rows: list[dict[str, Any]], first_key: str, second_key: str) -> dict[str, dict[str, int]]:
    values: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        values[str(row[first_key])][str(row[second_key])] += 1
    return {key: dict(counter) for key, counter in sorted(values.items())}


def summarize(
    all_rows: list[dict[str, Any]],
    filtered_rows: list[dict[str, Any]],
    frames: dict[int, FrameInfo],
    manual_image_exists: dict[str, bool],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    unique_images = sorted({row["source_image"] for row in all_rows})
    filtered_images = sorted({row["source_image"] for row in filtered_rows})
    split_counter = Counter(str(row["split"]) for row in filtered_rows)
    summary = {
        "mode": "dry_run" if args.dry_run or not args.allow_export else "export",
        "split_filter": args.split,
        "include_nicht_bewertbar": bool(args.include_nicht_bewertbar),
        "exclude_unmatched": bool(args.exclude_unmatched),
        "num_frames_in_meta": len(frames),
        "num_manual_manifest_rows": len(manual_image_exists),
        "num_annotated_images_total": len(unique_images),
        "num_regions_total": len(all_rows),
        "num_annotated_images_after_filter": len(filtered_images),
        "num_regions_after_filter": len(filtered_rows),
        "regions_per_label_total": dict(Counter(str(row["original_label"]) for row in all_rows)),
        "regions_per_label_after_filter": dict(Counter(str(row["original_label"]) for row in filtered_rows)),
        "regions_per_mapped_label_after_filter": dict(
            Counter(str(row["mapped_label"]) for row in filtered_rows)
        ),
        "regions_per_split_after_filter": dict(split_counter),
        "regions_per_split_and_class_after_filter": nested_counter(
            filtered_rows, "split", "mapped_label"
        ),
        "num_nicht_bewertbar_total": sum(
            1 for row in all_rows if row["original_label"] == "Nicht_bewertbar"
        ),
        "num_nicht_bewertbar_after_filter": sum(
            1 for row in filtered_rows if row["original_label"] == "Nicht_bewertbar"
        ),
        "num_unmatched_regions_total": sum(1 for row in all_rows if not bool(row["matched_manifest"])),
        "num_unmatched_regions_after_filter": sum(
            1 for row in filtered_rows if not bool(row["matched_manifest"])
        ),
        "num_clipped_boxes_total": sum(1 for row in all_rows if bool(row["clipped"])),
        "num_clipped_boxes_after_filter": sum(1 for row in filtered_rows if bool(row["clipped"])),
        "num_missing_source_image_regions_total": sum(
            1 for row in all_rows if "source_image_missing" in str(row["exclude_reason"])
        ),
        "num_missing_source_image_regions_after_filter": sum(
            1 for row in filtered_rows if "source_image_missing" in str(row["exclude_reason"])
        ),
        "num_regions_not_in_manual_manifest_total": sum(
            1 for row in all_rows if "not_in_manual_manifest" in str(row["exclude_reason"])
        ),
        "num_regions_not_in_manual_manifest_after_filter": sum(
            1 for row in filtered_rows if "not_in_manual_manifest" in str(row["exclude_reason"])
        ),
        "num_test_regions_after_filter": int(split_counter.get("test", 0)),
        "test_usage_note": (
            "Test regions are inventoried only when split=all; they must not be used for "
            "model selection, validation reporting, or threshold decisions."
        ),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)) if output_dir.is_relative_to(REPO_ROOT) else None,
        "outputs_written": bool(args.allow_export and not args.dry_run),
    }
    return summary


def write_region_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def safe_folder_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unlabeled"


def export_crops(rows: list[dict[str, Any]], manual_root: Path, config: dict[str, Any], crops_dir: Path) -> int:
    from PIL import Image

    images_dir = config_path(config, "images_dir", manual_root)
    crops_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in rows:
        image_path = images_dir / str(row["source_image"])
        if not image_path.exists():
            raise FileNotFoundError(f"Cannot export crop, source image missing: {image_path}")
        with Image.open(image_path) as image:
            crop = image.crop(
                (
                    int(float(row["x_min"])),
                    int(float(row["y_min"])),
                    int(float(row["x_max"])),
                    int(float(row["y_max"])),
                )
            )
            label_dir = safe_folder_name(str(row["mapped_label"]))
            split_dir = safe_folder_name(str(row["split"]))
            target = crops_dir / split_dir / label_dir / f"{row['region_id']}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            crop.save(target)
            count += 1
    return count


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_file = resolve_repo_path(args.config)
    config = load_yaml(config_file)
    defaults = config.get("defaults", {})
    if args.split is None:
        args.split = str(defaults.get("split", "all"))
    if not args.include_nicht_bewertbar:
        args.include_nicht_bewertbar = bool(defaults.get("include_nicht_bewertbar", False))
    if not args.exclude_unmatched:
        args.exclude_unmatched = bool(defaults.get("exclude_unmatched", False))
    if args.allow_export_crops:
        args.allow_export = True

    manual_root = resolve_manual_root(config, args.manual_root)
    cvat_path = config_path(config, "cvat_annotations", manual_root)
    frame_meta_path = config_path(config, "frame_meta", manual_root)
    labels_path = config_path(config, "labels", manual_root)
    manual_manifest_path = config_path(config, "manual_manifest", manual_root)
    split_manifest_path = resolve_repo_path(config["inputs"]["split_manifest"])
    output_dir = resolve_repo_path(config.get("export", {}).get("output_dir", "outputs/cvat_region_analysis/manual_all"))

    frames = read_frame_meta(frame_meta_path)
    label_by_id = read_labels(labels_path)
    manual_image_exists = read_manual_manifest(manual_manifest_path)
    cvat_payload = load_json(cvat_path)
    split_by_group, split_by_name = read_split_manifest(split_manifest_path, config)
    all_rows = build_region_rows(
        config=config,
        cvat_payload=cvat_payload,
        frames=frames,
        label_by_id=label_by_id,
        manual_image_exists=manual_image_exists,
        split_by_group=split_by_group,
        split_by_name=split_by_name,
    )
    filtered_rows = filter_rows(
        rows=all_rows,
        split=str(args.split),
        include_nicht_bewertbar=bool(args.include_nicht_bewertbar),
        exclude_unmatched=bool(args.exclude_unmatched),
    )
    summary = summarize(all_rows, filtered_rows, frames, manual_image_exists, args, output_dir)

    if args.allow_export and not args.dry_run:
        region_name = str(config.get("export", {}).get("region_table", "region_annotations.csv"))
        summary_name = str(config.get("export", {}).get("summary_json", "region_annotations_summary.json"))
        write_region_csv(filtered_rows, output_dir / region_name)
        write_json(summary, output_dir / summary_name)
        if args.allow_export_crops:
            crops_name = str(config.get("export", {}).get("crops_dir", "crops"))
            summary["num_crops_written"] = export_crops(filtered_rows, manual_root, config, output_dir / crops_name)
            write_json(summary, output_dir / summary_name)

    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary.get("num_test_regions_after_filter", 0):
        print(
            "Hinweis: Testregionen wurden nur inventarisiert. Sie duerfen nicht "
            "fuer Modellwahl oder Validierungsentscheidungen genutzt werden.",
            file=sys.stderr,
        )
    if args.dry_run or not args.allow_export:
        print("Dry-run/No-export: Es wurden keine Dateien geschrieben.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
