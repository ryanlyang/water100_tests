#!/usr/bin/env python3

import argparse
import tempfile
import unittest
from pathlib import Path

import yaml

from run_imagenet9_wb95_transfer_5seed import load_config, result_method


CONFIG = Path(__file__).resolve().parent / "configs/waterbirds95_hparam_transfer.yaml"


class Waterbirds95TransferTests(unittest.TestCase):
    def args(self, method: str, config: Path = CONFIG):
        return argparse.Namespace(method=method, config=config)

    def test_all_registered_methods_have_valid_contracts(self):
        methods = ("erm", "upweight", "abn", "elrep", "gals", "afr", "clip_lr", "r4rr")
        for method in methods:
            with self.subTest(method=method):
                loaded = load_config(self.args(method))
                self.assertEqual(loaded["experiment"]["target_standard_epochs"], 21)
                self.assertEqual(loaded["params"]["trainer"], method)

    def test_r4rr_exposure_scaling_and_zero_increment(self):
        loaded = load_config(self.args("r4rr"))
        params = loaded["params"]
        self.assertEqual(params["source_attention_epoch"], 109)
        self.assertEqual(params["attention_epoch"], 12)
        self.assertEqual(float(params["kl_increment"]), 0.0)
        self.assertEqual(result_method("r4rr"), "r4rr_wb95_transfer_klincr0")

    def test_afr_is_one_prespecified_configuration(self):
        loaded = load_config(self.args("afr"))
        params = loaded["params"]
        self.assertEqual(params["target_stage1_epochs"], 7)
        self.assertEqual(float(params["gamma"]), 11.0)
        self.assertEqual(float(params["reg_coeff"]), 0.0)

    def test_inconsistent_attention_epoch_is_rejected(self):
        payload = yaml.safe_load(CONFIG.read_text())
        payload["methods"]["r4rr"]["attention_epoch"] = 11
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(yaml.safe_dump(payload))
            with self.assertRaisesRegex(RuntimeError, "attention epoch"):
                load_config(self.args("r4rr", path))


if __name__ == "__main__":
    unittest.main()
