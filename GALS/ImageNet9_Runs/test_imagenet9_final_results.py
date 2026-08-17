#!/usr/bin/env python3

import csv
import argparse
import json
import tempfile
import unittest
from pathlib import Path

from imagenet9_final_utils import ALL_VARIANTS, evaluation_to_row, write_method_tables
from run_imagenet9_final_r4rr import load_selection as load_r4rr_selection


def evaluation(seed, mixed_same, mixed_rand):
    variants = {}
    for name in ALL_VARIANTS:
        value = 0.5
        if name == "mixed_same":
            value = mixed_same
        elif name == "mixed_rand":
            value = mixed_rand
        variants[name] = {
            "accuracy": value,
            "macro_class_accuracy": value,
        }
    return {
        "method": "erm",
        "seed": seed,
        "checkpoint": f"seed_{seed}.pt",
        "selection_value": 0.8,
        "variant_results": variants,
    }


class ImageNet9FinalResultTests(unittest.TestCase):
    def test_bg_gap_is_computed_within_seed(self):
        row = evaluation_to_row(evaluation(0, 0.81, 0.63))
        self.assertAlmostEqual(row["mixed_same"], 81.0)
        self.assertAlmostEqual(row["mixed_rand"], 63.0)
        self.assertAlmostEqual(row["bg_gap"], 18.0)

    def test_method_tables_use_population_std_and_partial_seed_sets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for seed, same, rand in ((0, 0.8, 0.6), (1, 0.9, 0.5)):
                path = root / f"evaluation_{seed}.json"
                path.write_text(json.dumps(evaluation(seed, same, rand)))
                paths.append(path)
            write_method_tables("erm", root, paths)
            with (root / "summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            gap = next(row for row in rows if row["metric"] == "bg_gap")
            self.assertAlmostEqual(float(gap["mean"]), 30.0)
            self.assertAlmostEqual(float(gap["std"]), 10.0)
            self.assertEqual(int(gap["n"]), 2)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["seeds"], [0, 1])
            self.assertEqual(summary["standard_deviation"], "population")

    def test_r4rr_final_selection_locks_loss_and_teacher_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher_root = root / "teacher"
            teacher_root.mkdir()
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "method": "r4rr",
                        "complete_trials": 50,
                        "target_complete_trials": 50,
                        "objective": "val_macro_class_accuracy",
                        "official_variants_used_for_selection": False,
                        "best_trial": 7,
                        "best_value": 0.9,
                        "best_params": {
                            "attention_epoch": 10,
                            "kl_lambda": 2.0,
                            "base_lr": 1e-3,
                            "classifier_lr": 2e-3,
                            "lr2_mult": 0.5,
                        },
                        "contract": {
                            "alignment_loss": "cosine",
                            "r4rr_teacher_maps": {"root": str(teacher_root)},
                            "weight_decay": 1e-5,
                            "fixed_momentum": 0.9,
                        },
                    }
                )
            )
            args = argparse.Namespace(
                sweep_summary=summary_path,
                alignment_loss="cosine",
                teacher_map_root=teacher_root,
                epochs=20,
                batch_size=96,
            )
            selection = load_r4rr_selection(args)
            self.assertEqual(selection["result_method"], "r4rr_cosine")
            self.assertEqual(selection["selection_mode"], "validation_best")
            args.alignment_loss = "forward_kl"
            with self.assertRaisesRegex(RuntimeError, "Alignment mismatch"):
                load_r4rr_selection(args)

    def test_r4rr_can_lock_a_completed_nonbest_trial(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher_root = root / "teacher"
            teacher_root.mkdir()
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "method": "r4rr",
                        "complete_trials": 50,
                        "target_complete_trials": 50,
                        "objective": "val_macro_class_accuracy",
                        "official_variants_used_for_selection": False,
                        "best_trial": 45,
                        "best_value": 0.995,
                        "best_params": {"attention_epoch": 11},
                        "contract": {
                            "alignment_loss": "forward_kl",
                            "r4rr_teacher_maps": {"root": str(teacher_root)},
                            "weight_decay": 1e-5,
                            "fixed_momentum": 0.9,
                        },
                    }
                )
            )
            (root / "trials.csv").write_text(
                "trial,state,objective,alignment_loss,attention_epoch,kl_lambda,"
                "base_lr,classifier_lr,lr2_mult\n"
                "13,COMPLETE,0.9940740740740741,forward_kl,14,"
                "4.972232449416291,9.650173804362784e-05,"
                "0.006215096340440499,0.21539158435419944\n"
            )
            args = argparse.Namespace(
                sweep_summary=summary_path,
                alignment_loss="forward_kl",
                teacher_map_root=teacher_root,
                epochs=20,
                batch_size=96,
                trial_number=13,
            )
            selection = load_r4rr_selection(args)
            self.assertEqual(selection["selection_mode"], "fixed_completed_trial")
            self.assertEqual(selection["selected_trial"], 13)
            self.assertEqual(selection["result_method"], "r4rr_forward_kl_trial13")
            self.assertAlmostEqual(selection["selected_value"], 0.9940740740740741)
            self.assertEqual(selection["best_params"]["attention_epoch"], 14)
            self.assertAlmostEqual(
                selection["best_params"]["classifier_lr"],
                0.006215096340440499,
            )
            args.kl_increment = 0.0
            zero_increment = load_r4rr_selection(args)
            self.assertEqual(
                zero_increment["result_method"],
                "r4rr_forward_kl_trial13_klincr0",
            )
            self.assertEqual(zero_increment["final_kl_increment_override"], 0.0)
            self.assertEqual(zero_increment["fixed"]["kl_increment"], 0.0)


if __name__ == "__main__":
    unittest.main()
