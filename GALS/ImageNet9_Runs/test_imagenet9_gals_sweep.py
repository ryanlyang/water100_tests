#!/usr/bin/env python3
"""Focused unit checks for ImageNet-9 GALS aggregation and sweep parameters."""

from __future__ import annotations

import unittest

import torch

import sweep_imagenet9_baseline as sweep
from train_imagenet9_baseline import combine_gals_attention


class FakeTrial:
    def suggest_float(self, name, low, high, log=False):
        return {"base_lr": 1e-3, "classifier_lr": 2e-3, "grad_weight": 3e3}[name]

    def suggest_categorical(self, name, choices):
        self.categorical = (name, tuple(choices))
        return choices[1]


class ImageNet9GALSTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
