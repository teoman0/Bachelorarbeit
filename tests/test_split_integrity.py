from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import unittest

from bachelorarbeit.data.split_dataset import read_split_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


class SplitIntegrityTest(unittest.TestCase):
    def test_grouped_split_is_complete_and_disjoint(self) -> None:
        records = read_split_manifest(REPO_ROOT / "data/splits/bmw25_grouped_split_manifest.csv")

        self.assertEqual(len(records), 4607)
        self.assertEqual(len({record.image_id for record in records}), len(records))
        self.assertEqual({record.split for record in records}, {"train", "val", "test"})

        splits_by_group: dict[str, set[str]] = defaultdict(set)
        for record in records:
            splits_by_group[record.group_id].add(record.split)

        self.assertTrue(all(len(splits) == 1 for splits in splits_by_group.values()))
        self.assertEqual(
            {split: sum(record.split == split for record in records) for split in ("train", "val", "test")},
            {"train": 3225, "val": 691, "test": 691},
        )
