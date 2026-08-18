#!/usr/bin/env python3
"""Focused tests for ImageNet-9 foreground-mask joins and aggregation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for path in (THIS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imagenet9_pointing_game_utils import parse_progress_jsonl, resolve_foreground_mask
from summarize_imagenet9_wb95_transfer_rise import summarize_method_variant


class ForegroundJoinTests(unittest.TestCase):
    def test_first_source_id_selects_composite_foreground(self) -> None:
        foreground = Path("/masks/00_dog/n00000001_1.npy")
        background = Path("/masks/00_dog/n00000002_2.npy")
        index = {
            ("00_dog", "n00000001_1"): foreground,
            ("00_dog", "n00000002_2"): background,
        }
        row = {
            "variant": "mixed_rand",
            "class_dir": "00_dog",
            "source_ids": "n00000001_1;n00000002_2",
            "relative_path": "mixed_rand/val/00_dog/example.JPEG",
        }
        path, source_id = resolve_foreground_mask(row, index)
        self.assertEqual(path, foreground)
        self.assertEqual(source_id, "n00000001_1")

    def test_missing_foreground_fails(self) -> None:
        row = {
            "variant": "original",
            "class_dir": "00_dog",
            "source_ids": "n00000001_1",
            "relative_path": "original/val/00_dog/example.JPEG",
        }
        with self.assertRaises(RuntimeError):
            resolve_foreground_mask(row, {})


class SummaryTests(unittest.TestCase):
    def test_population_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [0.50, 0.60, 0.70, 0.80, 0.90]
            for seed, value in enumerate(values):
                target = root / "erm" / "original" / f"seed_{seed}"
                target.mkdir(parents=True)
                payload = {
                    "method": "erm",
                    "seed": seed,
                    "variant": "original",
                    "target_mode": "label",
                    "explainer": "rise",
                    "mask_source": "backgrounds_challenge_fg_mask",
                    "errors": 0,
                    "pg_total": 4050,
                    "pg_acc": value,
                    "pg_macro_class_acc": value,
                    "pg_worst_class_acc": value - 0.1,
                    "pg_random_acc": 0.25,
                    "classification_acc": 0.9,
                    "classification_macro_class_acc": 0.9,
                    "saliency_mass_in_foreground": value,
                    "zero_saliency_maps": 0,
                    "rise_num_masks": 2000,
                    "rise_grid_size": 8,
                    "rise_p1": 0.1,
                    "rise_seed": 0,
                    "rise_masks_sha256": "fixed",
                }
                (target / "pointing_game_summary.json").write_text(json.dumps(payload))
            summary = summarize_method_variant(root, "erm", "original", range(5))
            self.assertAlmostEqual(summary["pg_acc_mean_pct"], 70.0)
            self.assertAlmostEqual(summary["pg_acc_std_pct"], 14.1421356237)


class ResumeTests(unittest.TestCase):
    def test_interrupted_trailing_progress_write_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "progress.jsonl"
            complete = {"sample_key": "original/example.JPEG", "pointing_hit": 1}
            encoded = json.dumps(complete) + "\n"
            path.write_bytes(encoded.encode("utf-8") + b'{"sample_key":"partial')

            rows = parse_progress_jsonl(path)

            self.assertEqual(rows, [complete])
            self.assertEqual(path.read_text(), encoded)


if __name__ == "__main__":
    unittest.main()
