from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.run_final_test_evaluation import require_final_test_access, select_model_keys


class FinalTestGuardTest(unittest.TestCase):
    def test_final_evaluation_requires_explicit_permission(self) -> None:
        with self.assertRaises(PermissionError):
            require_final_test_access(allow_final_test=False, check_preprocessing=False)

        require_final_test_access(allow_final_test=True, check_preprocessing=False)
        require_final_test_access(allow_final_test=False, check_preprocessing=True)

    def test_explicit_model_selection_limits_final_evaluation(self) -> None:
        config = {
            "global_models": {
                "yolo11n_cls": {"enabled": True},
                "dinov3_frozen_linear_head": {"enabled": True},
                "dinov3_partial_finetune_last2": {"enabled": True},
            },
            "region_models": {
                "dinov3_region_head_4class": {"enabled": True},
            },
        }
        args = SimpleNamespace(
            models=["dinov3_frozen_linear_head", "dinov3_partial_finetune_last2"],
            global_only=False,
            regions_only=False,
        )

        self.assertEqual(
            select_model_keys(config, args),
            {"dinov3_frozen_linear_head", "dinov3_partial_finetune_last2"},
        )
