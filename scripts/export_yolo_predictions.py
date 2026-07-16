"""Export YOLO classification predictions for a manifest split.

The script performs inference only. It does not train models, create
checkpoints, copy images, or evaluate the test split unless explicitly
requested with --allow-test.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MANIFEST_COLUMNS = {"image_id", "relative_path", "label", "split"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="YOLO classification model or local checkpoint, for example runs/.../weights/best.pt.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Split manifest CSV with image_id, relative_path, label, and split columns.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--yolo-dataset-dir",
        help="Local YOLO classification folder with split/class/image files.",
    )
    input_group.add_argument(
        "--dataset-root",
        help="Local original dataset root; manifest relative_path values are resolved below it.",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Manifest split to export. Only val is allowed by default.",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Allow --split test. Use only for the final test evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Local output directory for predictions_<split>.csv. Keep this under ignored outputs/.",
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--config-id", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--include-probabilities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one prob_<class> column per YOLO class when probabilities are available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing predictions_<split>.csv file.",
    )
    args = parser.parse_args()

    split = str(args.split)
    if split == "test" and not args.allow_test:
        parser.error("--split test is blocked unless --allow-test is set")
    if split != "val" and split != "test":
        parser.error("Only --split val is supported by default; test requires --allow-test")
    if args.imgsz < 1:
        parser.error("--imgsz must be at least 1")
    if args.batch < 1:
        parser.error("--batch must be at least 1")
    return args


def read_manifest(path: Path, split: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")
        rows = [row for row in reader if row["split"] == split]
    if not rows:
        raise ValueError(f"No rows found for split '{split}' in manifest: {path}")
    return rows


def unique_or_error(rows: list[dict[str, str]], column: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        value = row[column]
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:5])
        raise ValueError(f"Duplicate {column} values in selected split: {preview}")


def suffix_from_manifest(row: dict[str, str]) -> str:
    suffix = Path(row["relative_path"]).suffix.lower()
    return suffix or ".jpg"


def yolo_relative_image_path(row: dict[str, str]) -> Path:
    suffix = suffix_from_manifest(row)
    return Path(row["split"]) / row["label"] / f"{row['image_id']}{suffix}"


def resolved_image_path(
    row: dict[str, str],
    *,
    yolo_dataset_dir: Path | None,
    dataset_root: Path | None,
) -> tuple[Path, str]:
    if yolo_dataset_dir is not None:
        relative_path = yolo_relative_image_path(row)
        return yolo_dataset_dir / relative_path, relative_path.as_posix()

    assert dataset_root is not None
    relative_path = Path(row["relative_path"])
    return dataset_root / relative_path, relative_path.as_posix()


def validate_image_paths(
    rows: list[dict[str, str]],
    *,
    yolo_dataset_dir: Path | None,
    dataset_root: Path | None,
) -> list[tuple[dict[str, str], Path, str]]:
    resolved: list[tuple[dict[str, str], Path, str]] = []
    missing: list[str] = []
    non_images: list[str] = []
    seen_paths: set[Path] = set()
    duplicate_paths: list[str] = []

    for row in rows:
        path, relative_output_path = resolved_image_path(
            row,
            yolo_dataset_dir=yolo_dataset_dir,
            dataset_root=dataset_root,
        )
        if path in seen_paths:
            duplicate_paths.append(relative_output_path)
        seen_paths.add(path)
        if not path.exists():
            missing.append(relative_output_path)
        elif path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            non_images.append(relative_output_path)
        resolved.append((row, path, relative_output_path))

    if missing:
        raise FileNotFoundError(f"Missing images for selected split: {missing[:10]}")
    if non_images:
        raise ValueError(f"Unsupported image extensions: {non_images[:10]}")
    if duplicate_paths:
        raise ValueError(f"Duplicate resolved image paths: {duplicate_paths[:10]}")
    return resolved


def normalized_model_names(model: Any) -> list[str]:
    names = model.names
    if isinstance(names, dict):
        ordered = [names[key] for key in sorted(names, key=lambda value: int(value))]
    else:
        ordered = list(names)
    labels = [str(name) for name in ordered]
    if len(labels) != len(set(labels)):
        raise ValueError(f"YOLO class names are not unique: {labels}")
    return labels


def validate_class_mapping(labels: list[str], rows: list[dict[str, str]]) -> None:
    manifest_labels = {row["label"] for row in rows}
    model_labels = set(labels)
    if manifest_labels != model_labels:
        missing_in_model = sorted(manifest_labels - model_labels)
        missing_in_manifest = sorted(model_labels - manifest_labels)
        raise ValueError(
            "Manifest labels and YOLO model class names do not match. "
            f"Missing in model: {missing_in_model}; missing in manifest split: {missing_in_manifest}"
        )


def probability_column(label: str) -> str:
    return f"prob_{label}"


def predict_rows(
    *,
    model: Any,
    rows_with_paths: list[tuple[dict[str, str], Path, str]],
    labels: list[str],
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    probability_columns = [probability_column(label) for label in labels]

    for start in range(0, len(rows_with_paths), args.batch):
        batch = rows_with_paths[start : start + args.batch]
        source_paths = [str(path) for _row, path, _relative_path in batch]
        results = model.predict(
            source=source_paths,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            save=False,
            verbose=False,
        )
        if len(results) != len(batch):
            raise RuntimeError(f"Expected {len(batch)} predictions, got {len(results)}")

        for (manifest_row, _image_path, output_path), result in zip(batch, results, strict=True):
            if result.probs is None:
                raise RuntimeError("YOLO result did not contain classification probabilities")

            probabilities = result.probs.data.detach().cpu().tolist()
            if len(probabilities) != len(labels):
                raise RuntimeError(
                    f"Expected {len(labels)} class probabilities, got {len(probabilities)}"
                )
            predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            output_row = {
                "image_id": manifest_row["image_id"],
                "relative_path": manifest_row["relative_path"],
                "yolo_image_path": output_path,
                "split": manifest_row["split"],
                "true_label": manifest_row["label"],
                "predicted_label": labels[predicted_index],
                "model_name": args.model_name or Path(str(args.model)).stem,
                "config_id": args.config_id or "",
                "run_name": args.run_name or "",
                "seed": "" if args.seed is None else str(args.seed),
            }
            if args.include_probabilities:
                for column, value in zip(probability_columns, probabilities, strict=True):
                    output_row[column] = f"{float(value):.10g}"
            output_rows.append(output_row)
    return output_rows


def write_predictions(path: Path, rows: list[dict[str, str]], labels: list[str], include_probabilities: bool) -> None:
    fieldnames = [
        "image_id",
        "relative_path",
        "yolo_image_path",
        "split",
        "true_label",
        "predicted_label",
        "model_name",
        "config_id",
        "run_name",
        "seed",
    ]
    if include_probabilities:
        fieldnames.extend(probability_column(label) for label in labels)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ultralytics is not installed in the active Python environment."
        ) from exc

    manifest_path = repo_path(args.manifest)
    model_path = repo_path(args.model) if Path(args.model).suffix else Path(args.model)
    output_dir = repo_path(args.output_dir)
    output_path = output_dir / f"predictions_{args.split}.csv"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Prediction file already exists: {output_path}")

    yolo_dataset_dir = repo_path(args.yolo_dataset_dir) if args.yolo_dataset_dir else None
    dataset_root = repo_path(args.dataset_root) if args.dataset_root else None

    rows = read_manifest(manifest_path, args.split)
    unique_or_error(rows, "image_id")
    rows_with_paths = validate_image_paths(
        rows,
        yolo_dataset_dir=yolo_dataset_dir,
        dataset_root=dataset_root,
    )

    model = YOLO(str(model_path))
    labels = normalized_model_names(model)
    validate_class_mapping(labels, rows)
    prediction_rows = predict_rows(
        model=model,
        rows_with_paths=rows_with_paths,
        labels=labels,
        args=args,
    )
    if len(prediction_rows) != len(rows):
        raise RuntimeError(
            f"Prediction row count mismatch: expected {len(rows)}, got {len(prediction_rows)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_path, prediction_rows, labels, args.include_probabilities)
    return output_path


def main() -> int:
    args = parse_args()
    if args.split == "test":
        print(
            "WARNING: exporting test predictions. Use this only for the final evaluation.",
            file=sys.stderr,
        )
    try:
        output_path = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "predictions": str(output_path),
                "split": args.split,
                "test_used": args.split == "test",
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
