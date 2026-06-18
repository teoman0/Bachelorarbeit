"""Prototype-based DINOv3 patch heatmaps for class-folder datasets.

The script performs no model training. It builds one frozen-feature prototype
per class from a small, group-safe reference subset, then scores preview image
patches against those prototypes.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from dinov3_patch_smoke_test import (
    extract_patches,
    get_nested,
    git_commit,
    git_dirty,
    load_config,
    load_dinov3,
    package_versions,
    select_embedding,
)


@dataclass(frozen=True)
class ClassInfo:
    folder: str
    label: str
    ordinal: float
    color_rgb: tuple[int, int, int]


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    class_index: int
    label: str
    ordinal: float
    group_id: str


def class_infos(config: dict[str, Any]) -> list[ClassInfo]:
    raw_classes = get_nested(config, ["dataset", "classes"], [])
    if not raw_classes:
        raise ValueError("dataset.classes must define the four class folders.")
    infos: list[ClassInfo] = []
    for item in raw_classes:
        color = tuple(int(v) for v in item["color_rgb"])
        if len(color) != 3:
            raise ValueError(f"color_rgb must have three values: {item}")
        infos.append(
            ClassInfo(
                folder=str(item["folder"]),
                label=str(item["label"]),
                ordinal=float(item["ordinal"]),
                color_rgb=color,  # type: ignore[arg-type]
            )
        )
    return infos


def discover_images(config: dict[str, Any], classes: list[ClassInfo]) -> list[ImageRecord]:
    root_raw = str(get_nested(config, ["dataset", "root"]))
    root_expanded = os.path.expandvars(root_raw)
    if root_expanded == root_raw and ("$" in root_raw or "%" in root_raw):
        raise ValueError(
            "Dataset root contains an unresolved environment variable: "
            f"{root_raw}. Set BMW25_DATA_ROOT or adjust dataset.root in the config."
        )
    root = Path(root_expanded).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    extensions = {
        str(ext).lower()
        for ext in get_nested(config, ["dataset", "image_extensions"], [".jpg"])
    }
    group_regex = re.compile(str(get_nested(config, ["split", "group_regex"])), re.I)

    records: list[ImageRecord] = []
    for class_index, info in enumerate(classes):
        class_dir = root / info.folder
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder does not exist: {class_dir}")
        files = sorted(
            {
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in extensions
            }
        )
        for path in files:
            match = group_regex.match(path.stem)
            group_id = match.group(1) if match else path.stem
            records.append(
                ImageRecord(
                    path=path,
                    class_index=class_index,
                    label=info.label,
                    ordinal=info.ordinal,
                    group_id=group_id,
                )
            )
    if not records:
        raise ValueError(f"No images found below {root}")
    return records


def select_reference_and_preview(
    records: list[ImageRecord],
    classes: list[ClassInfo],
    config: dict[str, Any],
    rng: random.Random,
) -> tuple[list[ImageRecord], list[ImageRecord], dict[str, Any]]:
    reference_groups_per_class = int(
        get_nested(config, ["split", "reference_groups_per_class"], 4)
    )
    max_reference_images_per_class = int(
        get_nested(config, ["split", "max_reference_images_per_class"], 8)
    )
    preview_groups_per_class = int(
        get_nested(config, ["split", "preview_groups_per_class"], 1)
    )
    preview_images_per_class = int(
        get_nested(config, ["split", "preview_images_per_class"], 1)
    )
    exclude_multilabel_reference = bool(
        get_nested(config, ["split", "exclude_multilabel_groups_from_reference"], True)
    )
    prefer_multilabel_preview = bool(
        get_nested(config, ["split", "prefer_multilabel_preview_groups"], True)
    )

    by_class_group: dict[int, dict[str, list[ImageRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    group_labels: dict[str, set[int]] = defaultdict(set)
    for record in records:
        by_class_group[record.class_index][record.group_id].append(record)
        group_labels[record.group_id].add(record.class_index)

    reference: list[ImageRecord] = []
    preview: list[ImageRecord] = []
    reference_groups_used: set[str] = set()
    selection_summary: dict[str, Any] = {}

    for class_index, info in enumerate(classes):
        groups = list(by_class_group[class_index].keys())
        rng.shuffle(groups)

        ref_groups: list[str] = []
        ref_records: list[ImageRecord] = []
        for group_id in groups:
            if group_id in reference_groups_used:
                continue
            if exclude_multilabel_reference and len(group_labels[group_id]) > 1:
                continue
            candidates = sorted(by_class_group[class_index][group_id], key=lambda r: r.path.name)
            if len(ref_groups) < reference_groups_per_class:
                ref_groups.append(group_id)
                reference_groups_used.add(group_id)
                for candidate in candidates:
                    if len(ref_records) < max_reference_images_per_class:
                        ref_records.append(candidate)
                if len(ref_groups) >= reference_groups_per_class:
                    break

        if len(ref_groups) < reference_groups_per_class:
            raise ValueError(
                f"Not enough reference groups for {info.label}: "
                f"{len(ref_groups)} < {reference_groups_per_class}"
            )

        preview_groups: list[str] = []
        preview_records: list[ImageRecord] = []
        preview_candidates = [group_id for group_id in groups if group_id not in reference_groups_used]
        if prefer_multilabel_preview:
            preview_candidates.sort(key=lambda group_id: len(group_labels[group_id]) <= 1)
        for group_id in preview_candidates:
            if len(preview_groups) >= preview_groups_per_class:
                continue
            candidates = sorted(by_class_group[class_index][group_id], key=lambda r: r.path.name)
            preview_groups.append(group_id)
            for candidate in candidates:
                if len(preview_records) < preview_images_per_class:
                    preview_records.append(candidate)
            if (
                len(preview_groups) >= preview_groups_per_class
                and len(preview_records) >= preview_images_per_class
            ):
                break

        if len(preview_records) < preview_images_per_class:
            raise ValueError(
                f"Not enough preview images for {info.label}: "
                f"{len(preview_records)} < {preview_images_per_class}"
            )

        reference.extend(ref_records)
        preview.extend(preview_records)
        selection_summary[info.label] = {
            "reference_groups": ref_groups,
            "reference_images": [str(record.path) for record in ref_records],
            "preview_groups": preview_groups,
            "preview_groups_are_multilabel": {
                group_id: len(group_labels[group_id]) > 1 for group_id in preview_groups
            },
            "preview_images": [str(record.path) for record in preview_records],
        }

    return reference, preview, selection_summary


def patch_records_for_image(
    record: ImageRecord,
    config: dict[str, Any],
) -> tuple[Image.Image, list[Image.Image], list[dict[str, int]]]:
    image = Image.open(record.path).convert("RGB")
    patches, coords = extract_patches(
        image=image,
        patch_size=int(get_nested(config, ["patches", "patch_size"], 448)),
        stride=int(get_nested(config, ["patches", "stride"], 448)),
        drop_incomplete=bool(get_nested(config, ["patches", "drop_incomplete"], False)),
        max_patches=None,
    )
    return image, patches, coords


def embed_pil_batches(
    patches: list[Image.Image],
    processor: Any,
    model: Any,
    torch: Any,
    device: str,
    batch_size: int,
    pooling: str,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(patches), batch_size):
            batch = patches[start : start + batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            selected = select_embedding(outputs, pooling=pooling)
            embeddings.append(selected.detach().cpu().float().numpy())
    return np.concatenate(embeddings, axis=0)


def l2_normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def build_prototypes(
    reference: list[ImageRecord],
    classes: list[ClassInfo],
    config: dict[str, Any],
    processor: Any,
    model: Any,
    torch: Any,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    batch_size = int(get_nested(config, ["model", "batch_size"], 8))
    pooling = str(get_nested(config, ["scoring", "pooling"], "pooler_or_cls"))
    by_class_embeddings: dict[int, list[np.ndarray]] = defaultdict(list)
    image_patch_counts: dict[str, int] = {}

    for record in reference:
        _, patches, _ = patch_records_for_image(record, config)
        embeddings = embed_pil_batches(
            patches=patches,
            processor=processor,
            model=model,
            torch=torch,
            device=device,
            batch_size=batch_size,
            pooling=pooling,
        )
        by_class_embeddings[record.class_index].append(l2_normalize(embeddings))
        image_patch_counts[str(record.path)] = len(patches)

    prototypes: list[np.ndarray] = []
    prototype_summary: dict[str, Any] = {}
    for class_index, info in enumerate(classes):
        class_embeddings = by_class_embeddings.get(class_index)
        if not class_embeddings:
            raise ValueError(f"No reference embeddings for {info.label}")
        stacked = np.concatenate(class_embeddings, axis=0)
        prototype = stacked.mean(axis=0, keepdims=True)
        prototype = l2_normalize(prototype)[0]
        prototypes.append(prototype)
        prototype_summary[info.label] = {
            "reference_patch_count": int(stacked.shape[0]),
            "embedding_dim": int(stacked.shape[1]),
        }

    return np.stack(prototypes, axis=0), {
        "by_class": prototype_summary,
        "image_patch_counts": image_patch_counts,
    }


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / max(temperature, 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def safe_stem(record: ImageRecord) -> str:
    clean_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", record.label).strip("_")
    clean_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", record.path.stem).strip("_")
    return f"{clean_label}__{clean_name}"


def blend_original_with_overlay(
    image: Image.Image,
    coords: list[dict[str, int]],
    colors: list[tuple[int, int, int]],
    output_path: Path,
    classes: list[ClassInfo],
    alpha: int,
    title: str,
) -> None:
    base = image.convert("RGBA")
    draw = ImageDraw.Draw(base, "RGBA")
    for coord, color in zip(coords, colors, strict=True):
        box = (
            coord["x"],
            coord["y"],
            coord["x"] + coord["width"],
            coord["y"] + coord["height"],
        )
        draw.rectangle(box, fill=(*color, alpha), outline=(255, 255, 255, 160), width=2)

    legend_width = 330
    canvas = Image.new("RGBA", (base.width + legend_width, base.height), (245, 245, 245, 255))
    canvas.alpha_composite(base, (0, 0))
    legend = ImageDraw.Draw(canvas, "RGBA")
    x0 = base.width + 24
    y = 24
    legend.text((x0, y), title, fill=(20, 20, 20, 255))
    y += 34
    for info in classes:
        legend.rectangle((x0, y, x0 + 28, y + 28), fill=(*info.color_rgb, 255))
        legend.text((x0 + 40, y + 6), f"{info.ordinal:g}: {info.label}", fill=(20, 20, 20, 255))
        y += 38
    canvas.convert("RGB").save(output_path)


def save_direct_overlay(
    image: Image.Image,
    coords: list[dict[str, int]],
    colors: list[tuple[int, int, int]],
    output_path: Path,
    alpha_values: np.ndarray,
    draw_grid: bool,
) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for coord, color, alpha in zip(coords, colors, alpha_values, strict=True):
        box = (
            coord["x"],
            coord["y"],
            coord["x"] + coord["width"],
            coord["y"] + coord["height"],
        )
        draw.rectangle(box, fill=(*color, int(alpha)))
        if draw_grid:
            draw.rectangle(box, outline=(255, 255, 255, 120), width=2)
    base.alpha_composite(overlay)
    base.convert("RGB").save(output_path)


def save_direct_comparison(
    image: Image.Image,
    direct_overlay_path: Path,
    output_path: Path,
) -> None:
    original = image.convert("RGB")
    overlay = Image.open(direct_overlay_path).convert("RGB")
    gap = 24
    canvas = Image.new(
        "RGB",
        (original.width * 2 + gap, original.height),
        (245, 245, 245),
    )
    canvas.paste(original, (0, 0))
    canvas.paste(overlay, (original.width + gap, 0))
    canvas.save(output_path)


def save_scores_csv(
    path: Path,
    coords: list[dict[str, int]],
    sims: np.ndarray,
    probs: np.ndarray,
    pred_indices: np.ndarray,
    classes: list[ClassInfo],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "patch_index",
            "x",
            "y",
            "width",
            "height",
            "predicted_label",
            "confidence",
        ]
        header.extend([f"similarity_{info.label}" for info in classes])
        header.extend([f"probability_{info.label}" for info in classes])
        writer.writerow(header)
        for patch_index, coord in enumerate(coords):
            pred_idx = int(pred_indices[patch_index])
            row: list[Any] = [
                patch_index,
                coord["x"],
                coord["y"],
                coord["width"],
                coord["height"],
                classes[pred_idx].label,
                float(probs[patch_index, pred_idx]),
            ]
            row.extend(float(value) for value in sims[patch_index])
            row.extend(float(value) for value in probs[patch_index])
            writer.writerow(row)


def score_preview_image(
    record: ImageRecord,
    classes: list[ClassInfo],
    prototypes: np.ndarray,
    config: dict[str, Any],
    processor: Any,
    model: Any,
    torch: Any,
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    image, patches, coords = patch_records_for_image(record, config)
    embeddings = embed_pil_batches(
        patches=patches,
        processor=processor,
        model=model,
        torch=torch,
        device=device,
        batch_size=int(get_nested(config, ["model", "batch_size"], 8)),
        pooling=str(get_nested(config, ["scoring", "pooling"], "pooler_or_cls")),
    )
    normalized = l2_normalize(embeddings)
    sims = normalized @ prototypes.T
    probs = softmax(
        sims,
        temperature=float(get_nested(config, ["scoring", "softmax_temperature"], 0.03)),
    )
    pred_indices = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    ordinals = np.array([info.ordinal for info in classes], dtype=np.float32)
    ordinal_scores = probs @ ordinals

    predicted_colors = [classes[int(index)].color_rgb for index in pred_indices]
    ordinal_min = float(ordinals.min())
    ordinal_max = float(ordinals.max())
    ordinal_colors: list[tuple[int, int, int]] = []
    for score in ordinal_scores:
        if ordinal_max > ordinal_min:
            weights = probs[len(ordinal_colors)]
            color = np.sum(
                np.array([info.color_rgb for info in classes], dtype=np.float32)
                * weights[:, None],
                axis=0,
            )
        else:
            color = np.array(classes[0].color_rgb, dtype=np.float32)
        ordinal_colors.append(tuple(int(np.clip(channel, 0, 255)) for channel in color))

    stem = safe_stem(record)
    class_overlay = output_dir / f"{stem}__class_overlay.png"
    ordinal_overlay = output_dir / f"{stem}__ordinal_overlay.png"
    class_direct = output_dir / f"{stem}__class_direct.png"
    ordinal_direct = output_dir / f"{stem}__ordinal_direct.png"
    class_compare = output_dir / f"{stem}__class_compare.png"
    scores_csv = output_dir / f"{stem}__patch_scores.csv"
    alpha = int(get_nested(config, ["output", "overlay_alpha"], 112))
    direct_alpha = int(get_nested(config, ["output", "direct_overlay_alpha"], 92))
    direct_min_alpha = int(get_nested(config, ["output", "direct_overlay_min_alpha"], 28))
    scale_by_confidence = bool(
        get_nested(config, ["output", "direct_overlay_scale_by_confidence"], True)
    )
    draw_direct_grid = bool(get_nested(config, ["output", "direct_overlay_grid"], True))
    if scale_by_confidence:
        alpha_values = direct_min_alpha + confidence * (direct_alpha - direct_min_alpha)
    else:
        alpha_values = np.full_like(confidence, direct_alpha, dtype=np.float32)

    blend_original_with_overlay(
        image=image,
        coords=coords,
        colors=predicted_colors,
        output_path=class_overlay,
        classes=classes,
        alpha=alpha,
        title="Nearest class",
    )
    blend_original_with_overlay(
        image=image,
        coords=coords,
        colors=ordinal_colors,
        output_path=ordinal_overlay,
        classes=classes,
        alpha=alpha,
        title="Ordinal mix",
    )
    save_direct_overlay(
        image=image,
        coords=coords,
        colors=predicted_colors,
        output_path=class_direct,
        alpha_values=alpha_values,
        draw_grid=draw_direct_grid,
    )
    save_direct_overlay(
        image=image,
        coords=coords,
        colors=ordinal_colors,
        output_path=ordinal_direct,
        alpha_values=alpha_values,
        draw_grid=draw_direct_grid,
    )
    save_direct_comparison(image, class_direct, class_compare)
    save_scores_csv(scores_csv, coords, sims, probs, pred_indices, classes)

    predicted_counts = Counter(classes[int(index)].label for index in pred_indices)
    return {
        "image": str(record.path),
        "true_folder_label": record.label,
        "group_id": record.group_id,
        "patch_count": len(coords),
        "mean_confidence": float(confidence.mean()),
        "min_confidence": float(confidence.min()),
        "max_confidence": float(confidence.max()),
        "mean_ordinal_score": float(ordinal_scores.mean()),
        "predicted_patch_counts": dict(predicted_counts),
        "outputs": [
            str(class_overlay),
            str(ordinal_overlay),
            str(class_direct),
            str(ordinal_direct),
            str(class_compare),
            str(scores_csv),
        ],
    }


def dataset_summary(records: list[ImageRecord], classes: list[ClassInfo]) -> dict[str, Any]:
    by_class = Counter(record.label for record in records)
    by_group: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_group[record.group_id].add(record.label)
    multi_label_groups = {
        group_id: sorted(labels)
        for group_id, labels in by_group.items()
        if len(labels) > 1
    }
    return {
        "image_count": len(records),
        "class_counts": {info.label: by_class[info.label] for info in classes},
        "group_count": len(by_group),
        "multi_label_group_count": len(multi_label_groups),
        "multi_label_group_examples": dict(list(multi_label_groups.items())[:20]),
    }


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    rng = random.Random(seed)
    np.random.seed(seed)

    output_dir = Path(str(get_nested(config, ["output", "directory"])))
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = class_infos(config)
    records = discover_images(config, classes)
    reference, preview, selection_summary = select_reference_and_preview(
        records=records,
        classes=classes,
        config=config,
        rng=rng,
    )

    processor, model, torch, device = load_dinov3(config)
    prototypes, prototype_summary = build_prototypes(
        reference=reference,
        classes=classes,
        config=config,
        processor=processor,
        model=model,
        torch=torch,
        device=device,
    )

    preview_results = []
    for record in preview:
        preview_results.append(
            score_preview_image(
                record=record,
                classes=classes,
                prototypes=prototypes,
                config=config,
                processor=processor,
                model=model,
                torch=torch,
                device=device,
                output_dir=output_dir,
            )
        )

    summary = {
        "experiment_name": config.get("experiment_name"),
        "purpose": config.get("purpose"),
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "config_path": str(config_path),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "seed": seed,
        "package_versions": package_versions(model_loaded=True),
        "dataset": dataset_summary(records, classes),
        "selection": selection_summary,
        "model": {
            "name": get_nested(config, ["model", "name"]),
            "source": get_nested(config, ["model", "source"]),
            "device": device,
        },
        "patches": {
            "patch_size": get_nested(config, ["patches", "patch_size"]),
            "stride": get_nested(config, ["patches", "stride"]),
            "drop_incomplete": get_nested(config, ["patches", "drop_incomplete"]),
        },
        "prototype_summary": prototype_summary,
        "preview_results": preview_results,
        "method_note": (
            "Qualitative nearest-prototype heatmaps only. DINOv3 is frozen; "
            "no segmentation model is trained and no pixel-level ground truth is used."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build qualitative DINOv3 prototype heatmaps for a class-folder dataset."
    )
    parser.add_argument(
        "--config",
        default="configs/dinov3_bmw25_prototype_heatmaps.yaml",
        help="Path to the heatmap config.",
    )
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
