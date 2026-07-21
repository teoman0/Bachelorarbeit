from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from bachelorarbeit.data.split_dataset import ManifestRecord
from scripts.run_final_test_evaluation import prepare_dino_global_images
from scripts.train_dinov3_head import ManifestImageDataset


class PreprocessingConsistencyTest(unittest.TestCase):
    def test_dino_training_and_final_evaluation_share_image_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_root = Path(temporary_directory)
            relative_path = Path("class_a") / "sample.png"
            image_path = dataset_root / relative_path
            image_path.parent.mkdir(parents=True)
            array = np.zeros((60, 120, 3), dtype=np.uint8)
            array[:, :60, 0] = 255
            array[:, 60:, 1] = 255
            Image.fromarray(array, mode="RGB").save(image_path)

            record = ManifestRecord(
                image_id="sample",
                relative_path=relative_path.as_posix(),
                label="class_a",
                split="val",
                group_id="sample",
            )
            training_dataset = ManifestImageDataset([record], dataset_root, {"class_a": 0}, (224, 224))
            training_image, _label_index, _record = training_dataset[0]
            evaluation_image = prepare_dino_global_images([record], dataset_root, (224, 224))[0]

            self.assertEqual(training_image.size, (224, 224))
            self.assertEqual(evaluation_image.size, (224, 224))
            self.assertEqual(training_image.mode, "RGB")
            self.assertEqual(evaluation_image.mode, "RGB")
            self.assertEqual(training_image.tobytes(), evaluation_image.tobytes())
