from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_susceptibility import (  # noqa: E402
    diagnostic_views,
    enumerate_runs,
    expected_patch_intensity,
    locate_decoy_patch,
    stratified_train_holdout,
)


class DecoySusceptibilityTest(unittest.TestCase):
    def test_original_train_and_test_encodings(self) -> None:
        self.assertEqual(expected_patch_intensity(0, "train"), 255)
        self.assertEqual(expected_patch_intensity(9, "train"), 30)
        self.assertEqual(expected_patch_intensity(0, "test"), 0)
        self.assertEqual(expected_patch_intensity(9, "test"), 225)

    def test_locates_and_separates_patch(self) -> None:
        image = np.zeros((28, 28), dtype=np.uint8)
        image[8:20, 10:18] = 120
        image[-5:, -5:] = expected_patch_intensity(3, "train")
        rows, cols = locate_decoy_patch(image, 3, "train")
        self.assertEqual((rows.start, cols.start), (23, 23))
        views = diagnostic_views(image, 3, "train")
        self.assertTrue(np.array_equal(views["original"], image))
        self.assertTrue(np.all(views["digit_only"][-5:, -5:] == 0))
        self.assertEqual(int(views["digit_only"][10, 12]), 120)
        self.assertEqual(int(np.count_nonzero(views["patch_only"])), 25)
        self.assertTrue(np.all(views["patch_only"][-5:, -5:] == 180))

    def test_rejects_nonstandard_patch_encoding(self) -> None:
        image = np.zeros((28, 28), dtype=np.uint8)
        image[:5, :5] = 99
        with self.assertRaisesRegex(ValueError, "unmodified DecoyMNIST encoding"):
            locate_decoy_patch(image, 3, "train")

    def test_holdout_is_stratified_and_reproducible(self) -> None:
        by_label = {
            label: [Path(f"/{label}/{index:03d}.png") for index in range(100)]
            for label in range(10)
        }
        train_a, val_a = stratified_train_holdout(by_label)
        train_b, val_b = stratified_train_holdout(by_label)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(len(train_a), 900)
        self.assertEqual(len(val_a), 100)
        self.assertEqual(
            {label: sum(item_label == label for _, item_label in val_a) for label in range(10)},
            {label: 10 for label in range(10)},
        )

    def test_grid_has_nine_runs_in_stable_order(self) -> None:
        runs = enumerate_runs()
        self.assertEqual(len(runs), 9)
        self.assertEqual([run.run_index for run in runs], list(range(9)))
        self.assertEqual([run.seed for run in runs[:3]], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()

