#!/usr/bin/env python3

import argparse
import tempfile
import unittest
from pathlib import Path

from evaluate_imagenet9_clip_vit_zeroshot import (
    MODEL_NAMES,
    PROMPT_CONCEPTS,
    build_contract,
    prompts_for_class,
    summarize_model,
)


class ImageNet9ClipZeroShotTests(unittest.TestCase):
    def test_fixed_models(self):
        self.assertEqual(MODEL_NAMES, ("RN50", "ViT-B/16", "ViT-B/32"))

    def test_prompt_contract_matches_teacher_maps(self):
        self.assertEqual(PROMPT_CONCEPTS["instrument"], "musical instrument")
        self.assertEqual(
            prompts_for_class("instrument"),
            [
                "an image of a musical instrument",
                "a photo of a musical instrument",
            ],
        )
        self.assertEqual(
            prompts_for_class("insect"),
            ["an image of an insect", "a photo of an insect"],
        )

    def test_robustness_summary(self):
        values = {
            "original": 0.80,
            "mixed_same": 0.75,
            "mixed_rand": 0.60,
            "mixed_next": 0.65,
            "only_fg": 0.70,
            "only_bg_b": 0.20,
            "only_bg_t": 0.10,
            "no_fg": 0.05,
        }
        results = {
            variant: {"macro_class_accuracy": value}
            for variant, value in values.items()
        }
        summary = summarize_model(results)
        self.assertAlmostEqual(summary["mixed_average_macro_class_accuracy"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["original_minus_mixed_rand"], 0.20)
        self.assertAlmostEqual(summary["only_bg_average_macro_class_accuracy"], 0.15)
        self.assertEqual(summary["worst_variant"], "no_fg")

    def test_contract_marks_official_data_as_test_only(self):
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
                models=list(MODEL_NAMES),
                seed=0,
            )
            contract = build_contract(args)
        self.assertEqual(contract["evaluation_split"], "official_test_only")
        self.assertFalse(contract["validation_or_tuning_data_used"])
        self.assertFalse(contract["prompt_selection_on_official_test"])
        self.assertEqual(contract["weights"], "openai")
        self.assertEqual(contract["implementation"], "open_clip")


if __name__ == "__main__":
    unittest.main()
