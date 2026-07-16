"""Prepare the YOLOv11-cls global classification training workflow.

Default execution is a dry-run/smoke-test setup. It reads the configured split
manifest, checks local image availability, writes run metadata, and does not
start a real Ultralytics training run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bachelorarbeit.training.global_training_setup import prepare_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/yolov11_cls.yaml")
    parser.add_argument("--dataset-root", required=True, help="Local dataset root; not written to versioned files.")
    parser.add_argument("--smoke-test", action="store_true", help="Load a tiny subset from train/val/test.")
    parser.add_argument("--max-smoke-samples", type=int, default=2)
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="Reserved for a future real YOLO training run; currently refuses to start training.",
    )
    args = parser.parse_args()

    if args.max_smoke_samples < 1:
        parser.error("--max-smoke-samples must be at least 1")

    run = prepare_run(
        config_path=Path(args.config),
        dataset_root=Path(args.dataset_root),
        expected_model_family="yolov11_cls",
        dry_run=not args.allow_training,
        smoke_test=args.smoke_test,
        max_smoke_samples=args.max_smoke_samples,
        extra_metadata={
            "classification_workflow": "Ultralytics YOLO cls",
            "local_yolo_dataset_note": (
                "A YOLO-compatible train/val/test class-folder structure may be "
                "created locally from the manifest, preferably with symlinks or links. "
                "Do not commit prepared image folders."
            ),
            "real_training_status": "not_started",
        },
        extra_packages=["ultralytics"],
    )

    if args.allow_training:
        raise SystemExit(
            "Refusing to start YOLO training from the skeleton. Finalize the local "
            "YOLO dataset preparation and training parameters first."
        )

    print(json.dumps({"metadata": str(run.metadata_path), "output_dir": str(run.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
