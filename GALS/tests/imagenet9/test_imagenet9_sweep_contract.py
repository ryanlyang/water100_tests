from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_NET9_RUNS = REPO_ROOT / "ImageNet9_Runs"
if str(IMAGE_NET9_RUNS) not in sys.path:
    sys.path.insert(0, str(IMAGE_NET9_RUNS))

import sweep_imagenet9_baseline as sweep


def _args(**overrides):
    values = {
        "method": "erm",
        "manifest": Path("/data/reconstructed/manifest.csv"),
        "epochs": 20,
        "batch_size": 96,
        "weight_decay": 1e-5,
        "fixed_momentum": 0.9,
        "train_seed": 0,
        "python": "/env/bin/python",
        "trainer": Path("/repo/train_imagenet9_baseline.py"),
        "num_workers": 8,
        "device": "cuda:0",
        "abn_checkpoint": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class ImageNet9SweepContractTests(unittest.TestCase):
    def test_all_expected_non_teacher_methods_have_search_spaces(self):
        self.assertEqual(set(sweep.METHODS), set(sweep.SEARCH_SPACES))
        self.assertEqual(
            set(sweep.NON_TEACHER_METHODS),
            {"erm", "upweight", "abn", "elrep"},
        )
        self.assertIn("gals", sweep.METHODS)

    def test_contract_excludes_official_variants(self):
        contract = sweep._contract(_args())
        self.assertFalse(contract["official_variants_used"])
        self.assertEqual(contract["objective"], "val_macro_class_accuracy")
        self.assertNotIn("official_test", str(contract))

    def test_contract_hash_changes_when_epochs_change(self):
        first = sweep._contract_hash(sweep._contract(_args(epochs=20)))
        second = sweep._contract_hash(sweep._contract(_args(epochs=30)))
        self.assertNotEqual(first, second)

    def test_abn_command_requires_pretrained_checkpoint(self):
        args = _args(method="abn", abn_checkpoint=None)
        params = {
            "base_lr": 1e-3,
            "classifier_lr": 1e-3,
            "momentum": 0.9,
            "abn_cls_weight": 1.0,
            "theta1": 1e-4,
            "theta2": 1e-5,
        }
        with self.assertRaisesRegex(ValueError, "ABN sweep requires"):
            sweep._trainer_command(args, params)

    def test_trainer_command_contains_only_original_manifest(self):
        args = _args()
        params = {
            "base_lr": 1e-3,
            "classifier_lr": 1e-3,
            "momentum": 0.9,
            "abn_cls_weight": 1.0,
            "theta1": 1e-4,
            "theta2": 1e-5,
        }
        command = sweep._trainer_command(args, params)
        self.assertIn(str(args.manifest), command)
        self.assertFalse(any("official" in token for token in command))


if __name__ == "__main__":
    unittest.main()
