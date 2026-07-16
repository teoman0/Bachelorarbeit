"""Prepare the DeiT-Tiny from-scratch global classification workflow.

Default execution is a dry-run/smoke-test setup. It validates manifest-based
data access and writes run metadata without creating weights or checkpoints.
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
    parser.add_argument("--config", default="configs/experiments/deit_tiny_scratch.yaml")
    parser.add_argument("--dataset-root", required=True, help="Local dataset root; not written to versioned files.")
    parser.add_argument("--smoke-test", action="store_true", help="Load a tiny subset from train/val/test.")
    parser.add_argument("--max-smoke-samples", type=int, default=2)
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="Reserved for future from-scratch training; currently refuses to train.",
    )
    args = parser.parse_args()

    if args.max_smoke_samples < 1:
        parser.error("--max-smoke-samples must be at least 1")

    run = prepare_run(
        config_path=Path(args.config),
        dataset_root=Path(args.dataset_root),
        expected_model_family="deit_tiny_scratch",
        dry_run=not args.allow_training,
        smoke_test=args.smoke_test,
        max_smoke_samples=args.max_smoke_samples,
        extra_metadata={
            "architecture": "deit_tiny_patch16_224",
            "pretrained": False,
            "comparison_note": (
                "This is a from-scratch ViT control and does not use the same "
                "pretraining information as DINOv3."
            ),
            "real_training_status": "not_started",
        },
        extra_packages=["torch", "timm"],
    )

    if args.allow_training:
        raise SystemExit(
            "Refusing to start DeiT-Tiny training from the skeleton. Finalize "
            "training duration, regularization, and validation protocol first."
        )

    print(json.dumps({"metadata": str(run.metadata_path), "output_dir": str(run.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
