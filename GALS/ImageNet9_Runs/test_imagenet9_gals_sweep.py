#!/usr/bin/env python3
"""Focused unit checks for ImageNet-9 GALS aggregation and sweep parameters."""

from __future__ import annotations

import unittest
import tempfile
from argparse import Namespace
from pathlib import Path

import torch

import sweep_imagenet9_baseline as sweep
import generate_imagenet9_gals_vit_maps as map_generator
from train_imagenet9_baseline import combine_gals_attention, ground_truth_gradcam


class FakeTrial:
    def suggest_float(self, name, low, high, log=False):
        return {
            "base_lr": 1e-3,
            "classifier_lr": 2e-4,
            "grad_weight": 3e3,
            "cam_weight": 2.5,
            "abn_att_weight": 0.75,
        }[name]

    def suggest_categorical(self, name, choices):
        self.categorical = (name, tuple(choices))
        return choices[1]


class ImageNet9GALSTests(unittest.TestCase):
    def test_vit_contract_stays_compatible_and_rn50_is_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.csv"
            manifest.write_text("sample_id\nexample\n")
            common = {
                "manifest": manifest,
                "target_layer": "layer4.2.relu",
            }
            vit = map_generator._contract(
                Namespace(
                    **common,
                    map_type="transformer",
                    clip_checkpoint="ViT-B/32",
                )
            )
            rn50 = map_generator._contract(
                Namespace(
                    **common,
                    map_type="rn50_gradcam",
                    clip_checkpoint="RN50",
                )
            )

        self.assertEqual(vit["model"], "ViT-B/32")
        self.assertEqual(vit["method"], "clip_transformer_relevance")
        self.assertNotIn("target_layer", vit)
        self.assertEqual(rn50["model"], "RN50")
        self.assertEqual(rn50["method"], "clip_gradcam")
        self.assertEqual(rn50["target_layer"], "layer4.2.relu")
        self.assertEqual(rn50["expected_map_shape"], [2, 1, 7, 7])

    def test_average_nonzero_ignores_zero_prompt_and_normalizes(self):
        attention = torch.zeros(2, 2, 1, 2, 2)
        attention[0, 0, 0] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        attention[1, 0, 0] = torch.tensor([[0.0, 2.0], [4.0, 6.0]])
        attention[1, 1, 0] = torch.tensor([[0.0, 4.0], [8.0, 12.0]])

        combined, valid = combine_gals_attention(attention)

        self.assertEqual(valid.tolist(), [True, True])
        self.assertTrue(torch.allclose(combined[0, 0], torch.tensor([[0.0, 0.25], [0.5, 0.75]])))
        self.assertTrue(torch.allclose(combined[1, 0], torch.tensor([[0.0, 1 / 3], [2 / 3, 1.0]])))

    def test_all_zero_sample_is_marked_invalid(self):
        combined, valid = combine_gals_attention(torch.zeros(1, 2, 1, 7, 7))
        self.assertEqual(valid.tolist(), [False])
        self.assertEqual(int(torch.count_nonzero(combined)), 0)

    def test_gals_search_includes_categorical_criterion(self):
        trial = FakeTrial()
        params = sweep._suggest(trial, "gals", fixed_momentum=0.9)
        self.assertEqual(params["grad_criterion"], "L2")
        self.assertEqual(trial.categorical, ("grad_criterion", ("L1", "L2")))
        self.assertEqual(params["momentum"], 0.9)

    def test_gradcam_search_uses_established_three_ranges(self):
        params = sweep._suggest(FakeTrial(), "gals_gradcam", fixed_momentum=0.9)
        self.assertEqual(params["base_lr"], 1e-3)
        self.assertEqual(params["classifier_lr"], 2e-4)
        self.assertEqual(params["cam_weight"], 2.5)
        self.assertNotIn("grad_weight", sweep.SEARCH_SPACES["gals_gradcam"])

    def test_ground_truth_gradcam_is_normalized_and_differentiable(self):
        torch.manual_seed(0)
        features = torch.randn(2, 4, 3, 3, requires_grad=True)
        classifier = torch.nn.Linear(4, 3)
        logits = classifier(features.mean(dim=(2, 3)))
        targets = torch.tensor([0, 2])

        gradcam = ground_truth_gradcam(features, logits, targets)
        self.assertEqual(tuple(gradcam.shape), (2, 1, 3, 3))
        self.assertGreaterEqual(float(gradcam.min()), 0.0)
        self.assertLessEqual(float(gradcam.max()), 1.0 + 1e-6)

        gradcam.mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(classifier.weight.grad)

    def test_abn_attention_search_uses_established_three_ranges(self):
        params = sweep._suggest(FakeTrial(), "gals_abn", fixed_momentum=0.9)
        self.assertEqual(params["base_lr"], 1e-3)
        self.assertEqual(params["classifier_lr"], 2e-4)
        self.assertEqual(params["abn_att_weight"], 0.75)
        self.assertNotIn("abn_cls_weight", sweep.SEARCH_SPACES["gals_abn"])


if __name__ == "__main__":
    unittest.main()
