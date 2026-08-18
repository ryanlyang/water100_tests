#!/usr/bin/env python3

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from run_imagenet9_wb95_alignment_transfer_5seed import (
    TARGET_EPOCHS,
    exposure_scaled_epoch,
    load_selection,
    result_method,
)


class Waterbirds95AlignmentTransferTests(unittest.TestCase):
    def test_exposure_scaling(self):
        self.assertEqual(TARGET_EPOCHS, 21)
        self.assertEqual(exposure_scaled_epoch(109), 12)
        self.assertEqual(exposure_scaled_epoch(31), 3)
        self.assertEqual(exposure_scaled_epoch(199), 20)

    def test_result_label_is_r4rr_compatible(self):
        label = result_method("squared_l2")
        self.assertTrue(label.startswith("r4rr"))
        self.assertTrue(label.endswith("klincr0"))

    def test_completed_sweep_winner_is_transferred_without_increment(self):
        fields = [
            "trial", "alignment_loss", "attention_epoch", "kl_lambda", "kl_incr",
            "base_lr", "classifier_lr", "lr2_mult", "best_balanced_val_acc",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep = root / "wb95_r4rr_squared_l2_sweep_1.csv"
            with sweep.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for trial in range(50):
                    writer.writerow(
                        {
                            "trial": trial,
                            "alignment_loss": "squared_l2",
                            "attention_epoch": 31 if trial == 45 else 100,
                            "kl_lambda": 2.203343413102074 if trial == 45 else 1.0,
                            "kl_incr": 7.0,
                            "base_lr": 0.009994468547012192,
                            "classifier_lr": 0.004722884783976869,
                            "lr2_mult": 0.28836026209735827,
                            "best_balanced_val_acc": 0.99 if trial == 45 else 0.5,
                        }
                    )
            args = argparse.Namespace(
                alignment_loss="squared_l2",
                sweep_csv=sweep,
                sweep_log_dir=root,
                min_sweep_trials=50,
                run_root=root / "run",
                teacher_map_root=root,
            )
            selection = load_selection(args)
            self.assertEqual(selection["source_best_trial"], 45)
            self.assertEqual(selection["source_hparams"]["attention_epoch"], 31)
            self.assertEqual(selection["target_hparams"]["attention_epoch"], 3)
            self.assertEqual(selection["source_hparams"]["kl_increment"], 7.0)
            self.assertEqual(selection["target_hparams"]["kl_increment"], 0.0)


if __name__ == "__main__":
    unittest.main()
