#!/usr/bin/env python3
"""Focused checks for ImageNet-9 R4RR training and sweep contracts."""

from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sweep_imagenet9_baseline as sweep
from audit_imagenet9_r4rr_weclip_maps import voc_colormap
from train_imagenet9_r4rr import (
    ALIGNMENT_LOSSES,
    decode_target_mask,
    joint_train_transform,
    r4rr_alignment_loss,
)


class FakeTrial:
    def suggest_float(self, name, low, high, log=False):
        return {
            "kl_lambda": 25.0,
            "base_lr": 2e-3,
            "classifier_lr": 3e-4,
            "lr2_mult": 0.75,
        }[name]

    def suggest_int(self, name, low, high):
        self.int_call = (name, low, high)
        return 7


class ImageNet9R4RRSweepTests(unittest.TestCase):
    def test_map_count_ignores_dot_prefixed_atomic_pngs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sample_a.png").touch()
            (root / "sample_b.png").touch()
            (root / ".sample_c.deadbeef.png").touch()
            self.assertEqual(sweep._r4rr_map_counts(root), (2, 1))

    def test_target_mask_decodes_expected_voc_class_only(self):
        labels = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
        encoded = voc_colormap(10)[labels]
        mask = decode_target_mask(encoded, target_label=0)
        np.testing.assert_array_equal(
            mask,
            np.array([[0, 1, 0], [0, 1, 0]], dtype=np.uint8),
        )

    def test_joint_transform_keeps_image_and_mask_aligned(self):
        image_array = np.zeros((8, 8, 3), dtype=np.uint8)
        image_array[2:6, 1:4, 0] = 255
        mask_array = np.zeros((8, 8), dtype=np.uint8)
        mask_array[2:6, 1:4] = 1
        image, mask = joint_train_transform(
            Image.fromarray(image_array, mode="RGB"),
            Image.fromarray(mask_array, mode="L"),
            image_size=8,
            crop_params=(0, 0, 8, 8),
            flip=True,
        )
        red = image[0] * 0.229 + 0.485
        self.assertTrue(torch.equal(red.gt(0.5), mask.bool()))

    def test_alignment_skips_empty_teacher_maps(self):
        cams = torch.tensor(
            [
                [[0.0, 0.2], [0.5, 1.0]],
                [[1.0, 0.5], [0.2, 0.0]],
            ],
            requires_grad=True,
        )
        masks = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )
        loss, valid = r4rr_alignment_loss(cams, masks)
        expected_prob = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
        expected = F.kl_div(
            F.log_softmax(cams[:1].flatten(1), dim=1),
            expected_prob,
            reduction="batchmean",
        )
        self.assertEqual(valid.tolist(), [True, False])
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertEqual(float(cams.grad[1].abs().sum()), 0.0)

    def test_all_empty_batch_returns_differentiable_zero(self):
        cams = torch.randn(2, 3, 3, requires_grad=True)
        loss, valid = r4rr_alignment_loss(cams, torch.zeros(2, 12, 12))
        self.assertEqual(valid.tolist(), [False, False])
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertEqual(float(cams.grad.abs().sum()), 0.0)

    def test_all_registered_alignment_losses_are_finite_with_invalid_map_filtering(self):
        for loss_name in ALIGNMENT_LOSSES:
            cams = torch.randn(3, 4, 4, requires_grad=True)
            masks = torch.rand(3, 12, 12)
            masks[2].zero_()
            loss, valid = r4rr_alignment_loss(cams, masks, loss_name)
            self.assertEqual(valid.tolist(), [True, True, False])
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertEqual(float(cams.grad[2].abs().sum()), 0.0)

    def test_registered_search_space_matches_r4rr_protocol(self):
        self.assertEqual(
            sweep.SEARCH_SPACES["r4rr"],
            {
                "attention_epoch": (0, 19, "int"),
                "kl_lambda": (1.0, 500.0, "log"),
                "base_lr": (1e-5, 5e-2, "log"),
                "classifier_lr": (1e-5, 5e-2, "log"),
                "lr2_mult": (1e-1, 3.0, "log"),
            },
        )
        trial = FakeTrial()
        params = sweep._suggest(trial, "r4rr", fixed_momentum=0.9)
        self.assertEqual(trial.int_call, ("attention_epoch", 0, 19))
        self.assertEqual(params["attention_epoch"], 7)
        self.assertEqual(params["kl_lambda"], 25.0)
        self.assertEqual(params["lr2_mult"], 0.75)
        self.assertEqual(params["momentum"], 0.9)

    def test_r4rr_command_uses_dedicated_trainer_and_teacher_maps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = Namespace(
                method="r4rr",
                python="python",
                trainer=root / "train_imagenet9_r4rr.py",
                manifest=root / "manifest.csv",
                teacher_map_root=root / "maps",
                alignment_loss="jensen_shannon",
                train_seed=0,
                epochs=20,
                batch_size=96,
                num_workers=8,
                weight_decay=1e-5,
                device="cuda:0",
            )
            params = sweep._suggest(FakeTrial(), "r4rr", fixed_momentum=0.9)
            command = sweep._trainer_command(args, params)
        self.assertIn("--teacher-map-root", command)
        self.assertIn("--attention-epoch", command)
        self.assertIn("--kl-lambda", command)
        self.assertIn("--lr2-mult", command)
        self.assertEqual(command[command.index("--alignment-loss") + 1], "jensen_shannon")
        self.assertNotIn("--teacher-map-audit", command)


if __name__ == "__main__":
    unittest.main()
