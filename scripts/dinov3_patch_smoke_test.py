"""Patch-based DINOv3 feature extraction smoke test.

This script intentionally performs no training and uses no labels. It only
checks whether an image can be split into patches, passed through a frozen
DINOv3 backbone, and projected back into a small diagnostic overlay.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_config(path: Path) -> dict[str, Any]:
    """Load a simple YAML config, using PyYAML when available."""
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f"Config root must be a mapping: {path}")
        return loaded
    except ModuleNotFoundError:
        pass

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key_value = line.strip().split(":", 1)
        if len(key_value) != 2:
            raise ValueError(f"Unsupported config line: {raw_line}")
        key, value = key_value[0].strip(), key_value[1].strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)
    return root


def get_nested(config: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def package_versions(model_loaded: bool) -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pillow": Image.__version__,
    }
    if model_loaded:
        import torch
        import transformers

        versions["torch"] = torch.__version__
        versions["transformers"] = transformers.__version__
    return versions


def make_synthetic_surface(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    base = 0.50 + 0.20 * x + 0.10 * y
    brush = 0.10 * np.sin(2.0 * math.pi * (x * 38.0 + y * 2.5))
    fine = rng.normal(0.0, 0.035, size=(height, width)).astype(np.float32)
    img = np.clip(base + brush + fine, 0.0, 1.0)
    rgb = np.stack(
        [
            img * 230.0,
            img * 225.0,
            img * 215.0,
        ],
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def resolve_input_image(
    config: dict[str, Any],
    image_arg: str | None,
    output_dir: Path,
    seed: int,
) -> tuple[Image.Image, str, bool]:
    image_path = image_arg or get_nested(config, ["input", "image_path"])
    if image_path:
        path = Path(image_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input image does not exist: {path}")
        return Image.open(path).convert("RGB"), str(path), False

    if not get_nested(config, ["input", "synthetic_fallback"], False):
        raise ValueError("No image_path configured and synthetic_fallback is false.")

    width = int(get_nested(config, ["input", "synthetic_width"], 512))
    height = int(get_nested(config, ["input", "synthetic_height"], 512))
    image = make_synthetic_surface(width=width, height=height, seed=seed)
    synthetic_path = output_dir / "synthetic_brushed_surface.png"
    image.save(synthetic_path)
    return image, str(synthetic_path), True


def extract_patches(
    image: Image.Image,
    patch_size: int,
    stride: int,
    drop_incomplete: bool,
    max_patches: int | None,
) -> tuple[list[Image.Image], list[dict[str, int]]]:
    width, height = image.size
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive.")
    if width < patch_size or height < patch_size:
        raise ValueError(
            f"Image {width}x{height} is smaller than patch_size={patch_size}."
        )

    x_positions = list(range(0, width - patch_size + 1, stride))
    y_positions = list(range(0, height - patch_size + 1, stride))
    if not drop_incomplete:
        if x_positions[-1] != width - patch_size:
            x_positions.append(width - patch_size)
        if y_positions[-1] != height - patch_size:
            y_positions.append(height - patch_size)

    patches: list[Image.Image] = []
    coordinates: list[dict[str, int]] = []
    for y in y_positions:
        for x in x_positions:
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            coordinates.append(
                {"x": x, "y": y, "width": patch_size, "height": patch_size}
            )
            if max_patches is not None and len(patches) >= max_patches:
                return patches, coordinates
    return patches, coordinates


def load_dinov3(config: dict[str, Any]):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise RuntimeError(
            "Missing deep-learning dependency "
            f"'{missing}'. Install the project requirements before the real DINOv3 run."
        ) from exc

    model_name = str(get_nested(config, ["model", "name"]))
    local_files_only = bool(get_nested(config, ["model", "local_files_only"], False))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    device_setting = str(get_nested(config, ["model", "device"], "auto"))
    if device_setting == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_setting

    common_kwargs = {
        "local_files_only": local_files_only,
    }
    if token:
        common_kwargs["token"] = token

    processor = AutoImageProcessor.from_pretrained(model_name, **common_kwargs)
    model = AutoModel.from_pretrained(model_name, **common_kwargs)
    model.eval()
    model.to(device)
    return processor, model, torch, device


def select_embedding(outputs: Any, pooling: str):
    if pooling == "pooler_or_cls" and getattr(outputs, "pooler_output", None) is not None:
        return outputs.pooler_output
    if pooling in {"pooler", "pooler_or_cls"}:
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        if pooling == "pooler":
            raise ValueError("Model output has no pooler_output.")
    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is None:
        raise ValueError("Model output has no last_hidden_state.")
    if pooling in {"cls", "pooler_or_cls"}:
        return last_hidden[:, 0]
    if pooling == "mean_tokens":
        return last_hidden[:, 1:].mean(dim=1)
    raise ValueError(f"Unsupported pooling mode: {pooling}")


def embed_patches(
    patches: list[Image.Image],
    config: dict[str, Any],
) -> tuple[np.ndarray, str]:
    processor, model, torch, device = load_dinov3(config)
    batch_size = int(get_nested(config, ["model", "batch_size"], 4))
    pooling = str(get_nested(config, ["features", "pooling"], "pooler_or_cls"))
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(patches), batch_size):
            batch = patches[start : start + batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            batch_embeddings = select_embedding(outputs, pooling=pooling)
            embeddings.append(batch_embeddings.detach().cpu().float().numpy())
    return np.concatenate(embeddings, axis=0), device


def cosine_to_mean(embeddings: np.ndarray) -> np.ndarray:
    mean_embedding = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
    )
    mean_norm = mean_embedding / np.maximum(
        np.linalg.norm(mean_embedding, axis=1, keepdims=True), 1e-12
    )
    return centered @ mean_norm.T[:, 0]


def save_overlay(
    image: Image.Image,
    coordinates: list[dict[str, int]],
    similarities: np.ndarray,
    output_path: Path,
) -> None:
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    lo = float(similarities.min())
    hi = float(similarities.max())
    span = max(hi - lo, 1e-12)

    for coord, similarity in zip(coordinates, similarities, strict=True):
        value = (float(similarity) - lo) / span
        red = int(255 * value)
        blue = int(255 * (1.0 - value))
        color = (red, 70, blue, 90)
        box = (
            coord["x"],
            coord["y"],
            coord["x"] + coord["width"],
            coord["y"] + coord["height"],
        )
        draw.rectangle(box, fill=color, outline=(255, 255, 255, 180), width=2)
    overlay.save(output_path)


def build_summary(
    *,
    config: dict[str, Any],
    config_path: Path,
    image_source: str,
    synthetic_image: bool,
    image: Image.Image,
    coordinates: list[dict[str, int]],
    dry_run: bool,
    embeddings: np.ndarray | None,
    similarities: np.ndarray | None,
    model_device: str | None,
    output_files: list[str],
) -> dict[str, Any]:
    model_loaded = embeddings is not None and not dry_run
    patch_entries = []
    norms = None
    if embeddings is not None:
        norms_array = np.linalg.norm(embeddings, axis=1)
        norms = {
            "min": float(norms_array.min()),
            "mean": float(norms_array.mean()),
            "max": float(norms_array.max()),
        }
    for index, coord in enumerate(coordinates):
        entry: dict[str, Any] = {"patch_index": index, **coord}
        if similarities is not None:
            entry["cosine_to_patch_mean"] = float(similarities[index])
        patch_entries.append(entry)

    return {
        "experiment_name": config.get("experiment_name"),
        "purpose": config.get("purpose"),
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "config_path": str(config_path),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "seed": config.get("seed"),
        "package_versions": package_versions(model_loaded=model_loaded),
        "model": {
            "name": get_nested(config, ["model", "name"]),
            "source": get_nested(config, ["model", "source"]),
            "loaded": model_loaded,
            "device": model_device,
        },
        "dry_run": dry_run,
        "input_image": {
            "source": image_source,
            "synthetic": synthetic_image,
            "width": image.size[0],
            "height": image.size[1],
        },
        "patches": {
            "count": len(coordinates),
            "patch_size": get_nested(config, ["patches", "patch_size"]),
            "stride": get_nested(config, ["patches", "stride"]),
            "entries": patch_entries,
        },
        "features": {
            "embedding_shape": list(embeddings.shape) if embeddings is not None else None,
            "embedding_norms": norms,
        },
        "system": {
            "platform": platform.platform(),
        },
        "outputs": output_files,
        "method_note": (
            "Technical smoke test only: no labels, no training, no split decision, "
            "no test-set evaluation."
        ),
    }


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(args.output_dir or get_nested(config, ["output", "directory"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    image, image_source, synthetic_image = resolve_input_image(
        config=config,
        image_arg=args.image,
        output_dir=output_dir,
        seed=seed,
    )

    patch_size = int(get_nested(config, ["patches", "patch_size"], 224))
    stride = int(get_nested(config, ["patches", "stride"], patch_size))
    max_patches_raw = get_nested(config, ["patches", "max_patches"], None)
    max_patches = None if max_patches_raw is None else int(max_patches_raw)
    drop_incomplete = bool(get_nested(config, ["patches", "drop_incomplete"], True))
    patches, coordinates = extract_patches(
        image=image,
        patch_size=patch_size,
        stride=stride,
        drop_incomplete=drop_incomplete,
        max_patches=max_patches,
    )

    embeddings: np.ndarray | None = None
    similarities: np.ndarray | None = None
    model_device: str | None = None
    output_files: list[str] = []
    if synthetic_image:
        output_files.append(image_source)

    if args.dry_run:
        rng = np.random.default_rng(seed)
        embeddings = rng.normal(0.0, 1.0, size=(len(patches), 8)).astype(np.float32)
        similarities = cosine_to_mean(embeddings)
    else:
        embeddings, model_device = embed_patches(patches, config)
        similarities = cosine_to_mean(embeddings)

    if bool(get_nested(config, ["features", "save_embeddings"], False)):
        embedding_path = output_dir / "patch_embeddings.npy"
        np.save(embedding_path, embeddings)
        output_files.append(str(embedding_path))

    if (
        not args.dry_run
        and similarities is not None
        and bool(get_nested(config, ["output", "save_patch_overlay"], True))
    ):
        overlay_path = output_dir / "patch_similarity_overlay.png"
        save_overlay(image, coordinates, similarities, overlay_path)
        output_files.append(str(overlay_path))

    summary_path = output_dir / "summary.json"
    output_files.append(str(summary_path))
    summary = build_summary(
        config=config,
        config_path=config_path,
        image_source=image_source,
        synthetic_image=synthetic_image,
        image=image,
        coordinates=coordinates,
        dry_run=args.dry_run,
        embeddings=embeddings,
        similarities=similarities,
        model_device=model_device,
        output_files=output_files,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch-based DINOv3 feature extraction smoke test."
    )
    parser.add_argument(
        "--config",
        default="configs/dinov3_patch_smoke_test.yaml",
        help="Path to the smoke-test config.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional single image path. Uses synthetic fallback if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from the config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test patch extraction and output writing without loading DINOv3.",
    )
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
