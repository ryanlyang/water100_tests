#!/usr/bin/env python3
"""Tests for ImageNet-9 systematic corruption selection and manifests."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from imagenet9_systematic_corruption import (  # noqa: E402
    CLASS_COUNT,
    CLASS_NAMES,
    RANDOM_CONDITION,
    TRAIN_COUNT,
    build_manifest,
    prepare_manifest,
    select_indices,
)
from run_imagenet9_r4rr_systematic_corruption import (  # noqa: E402
    build_selection_args,
)


def labels() -> np.ndarray:
    return np.repeat(np.arange(len(CLASS_NAMES), dtype=np.int64), CLASS_COUNT)


def rows():
    result = []
    for index, label in enumerate(labels()):
        result.append(
            {
                "split_index": index,
                "sample_id": f"sample_{index:05d}",
                "label": int(label),
                "class_name": CLASS_NAMES[int(label)],
                "source_path": f"/data/sample_{index:05d}.JPEG",
            }
        )
    return result


class SelectionTests(unittest.TestCase):
    def test_final_selection_adapter_includes_training_shape(self) -> None:
        source = argparse.Namespace(
            sweep_summary=Path("summary.json"),
            teacher_map_root=Path("teacher_maps"),
            kl_increment=0.0,
            epochs=20,
            batch_size=96,
        )
        adapted = build_selection_args(source)
        self.assertEqual(adapted.alignment_loss, "forward_kl")
        self.assertIsNone(adapted.trial_number)
        self.assertEqual(adapted.epochs, 20)
        self.assertEqual(adapted.batch_size, 96)

    def test_each_systematic_selection_is_class_pure(self) -> None:
        values = labels()
        for label, class_name in enumerate(CLASS_NAMES):
            selected = select_indices(f"class_{class_name}", values, corruption_seed=0)
            self.assertEqual(selected.shape, (CLASS_COUNT,))
            self.assertTrue(np.all(values[selected] == label))

    def test_random_control_is_exact_and_deterministic(self) -> None:
        values = labels()
        first = select_indices(RANDOM_CONDITION, values, corruption_seed=0)
        second = select_indices(RANDOM_CONDITION, values, corruption_seed=0)
        different = select_indices(RANDOM_CONDITION, values, corruption_seed=1)
        self.assertEqual(first.shape, (CLASS_COUNT,))
        self.assertEqual(np.unique(first).size, CLASS_COUNT)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, different))

    def test_manifest_records_balanced_contract(self) -> None:
        manifest, selected, selected_rows = build_manifest(
            "class_fish", rows(), corruption_seed=0
        )
        self.assertEqual(len(selected), CLASS_COUNT)
        self.assertEqual(len(selected_rows), CLASS_COUNT)
        self.assertEqual(manifest["training_example_count"], TRAIN_COUNT)
        self.assertEqual(manifest["corrupted_example_count"], CLASS_COUNT)
        self.assertAlmostEqual(manifest["corrupted_fraction_of_training"], 1.0 / 9.0)
        self.assertEqual(manifest["corrupted_class_counts"]["fish"], CLASS_COUNT)
        self.assertEqual(sum(manifest["corrupted_class_counts"].values()), CLASS_COUNT)

    def test_persisted_manifest_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "manifest.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("split", "sample_id", "label", "class_name", "source_path"),
                )
                writer.writeheader()
                for row in rows():
                    writer.writerow({"split": "train", **{key: row[key] for key in writer.fieldnames[1:]}})
            output = root / "selection"
            first, first_indices, first_hash = prepare_manifest(
                RANDOM_CONDITION, source, output, corruption_seed=0
            )
            second, second_indices, second_hash = prepare_manifest(
                RANDOM_CONDITION, source, output, corruption_seed=0
            )
            self.assertEqual(first, second)
            self.assertEqual(first_indices, second_indices)
            self.assertEqual(first_hash, second_hash)


if __name__ == "__main__":
    unittest.main()
