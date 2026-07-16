"""Prepare the DINOv3 frozen-backbone plus linear-head workflow.

Default execution is a dry-run/smoke-test setup. It validates manifest-based
data access and writes run metadata without loading DINOv3 weights or starting
training.
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
    parser.add_argument("--config", default="configs/experiments/dinov3_linear_head.yaml")
    parser.add_argument("--dataset-root", required=True, help="Local dataset root; not written to versioned files.")
    parser.add_argument("--smoke-test", action="store_true", help="Load a tiny subset from train/val/test.")
    parser.add_argument("--max-smoke-samples", type=int, default=2)
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="Reserved for future head training; currently refuses to load weights or train.",
    )
    args = parser.parse_args()

    if args.max_smoke_samples < 1:
        parser.error("--max-smoke-samples must be at least 1")

    run = prepare_run(
        config_path=Path(args.config),
        dataset_root=Path(args.dataset_root),
        expected_model_family="dinov3_linear_head",
        dry_run=not args.allow_training,
        smoke_test=args.smoke_test,
        max_smoke_samples=args.max_smoke_samples,
        extra_metadata={
            "backbone": "DINOv3 frozen",
            "head": "linear classification head",
            "weights_note": (
                "DINOv3 weights must be provided locally or through an explicitly "
                "approved source. The dry-run does not load or download weights."
            ),
            "feature_cache_policy": "optional local cache only; do not commit",
            "real_training_status": "not_started",
        },
        extra_packages=["torch", "transformers"],
    )

    if args.allow_training:
        raise SystemExit(
            "Refusing to start DINOv3 head training from the skeleton. Provide and "
            "document the local DINOv3 weights before enabling real training."
        )

    print(json.dumps({"metadata": str(run.metadata_path), "output_dir": str(run.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
