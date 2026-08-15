#!/usr/bin/env python3

import argparse
import tempfile
import unittest
from pathlib import Path

from evaluate_imagenet9_openclip_siglip2_zeroshot import (
    MODEL_SPECS,
    build_contract,
)


class ImageNet9OpenClipZeroShotTests(unittest.TestCase):
    def test_fixed_model_specs_match_existing_experiments(self):
        self.assertEqual(MODEL_SPECS["openclip_laion"].model_name, "ViT-B-32")
        self.assertEqual(
            MODEL_SPECS["openclip_laion"].pretrained, "laion2b_s34b_b79k"
        )
        self.assertEqual(
            MODEL_SPECS["siglip2"].model_name, "ViT-B-16-SigLIP2-256"
        )
        self.assertEqual(MODEL_SPECS["siglip2"].pretrained, "webli")

    def test_contract_is_official_test_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            metadata = data_root / "metadata" / "protocol"
            metadata.mkdir(parents=True)
            (metadata / "official_test_manifest.csv").write_text(
                "variant,label,class_name,relative_path\n"
            )
            args = argparse.Namespace(
                data_root=data_root,
                protocol_name="protocol",
                models=["openclip_laion", "siglip2"],
                seed=0,
            )
            contract = build_contract(args)

        self.assertEqual(contract["evaluation_split"], "official_test_only")
        self.assertFalse(contract["validation_or_tuning_data_used"])
        self.assertFalse(contract["prompt_selection_on_official_test"])
        self.assertEqual(
            [model["pretrained"] for model in contract["models"]],
            ["laion2b_s34b_b79k", "webli"],
        )


if __name__ == "__main__":
    unittest.main()
