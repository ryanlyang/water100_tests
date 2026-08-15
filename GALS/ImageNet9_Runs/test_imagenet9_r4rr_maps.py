#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from audit_imagenet9_r4rr_weclip_maps import decode_voc_colors, voc_colormap
from generate_imagenet9_r4rr_weclip_maps import (
    _save_prediction,
    _writable_rgb_array,
    shard_bounds,
)


class ImageNet9R4RRMapTests(unittest.TestCase):
    def test_shard_bounds_cover_final_partial_shard(self):
        self.assertEqual(shard_bounds(0, 1000, 45405), (0, 1000))
        self.assertEqual(shard_bounds(45, 1000, 45405), (45000, 45405))

    def test_voc_color_round_trip_for_ten_labels(self):
        labels = np.arange(10, dtype=np.uint8).reshape(2, 5)
        encoded = voc_colormap(10)[labels]
        decoded = decode_voc_colors(encoded, num_labels=10)
        np.testing.assert_array_equal(decoded, labels)

    def test_unknown_color_is_rejected(self):
        encoded = np.full((2, 2, 3), 17, dtype=np.uint8)
        with self.assertRaises(ValueError):
            decode_voc_colors(encoded, num_labels=10)

    def test_crf_image_buffer_is_writable_and_contiguous(self):
        source = Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8), mode="RGB")
        image = _writable_rgb_array(source)
        self.assertTrue(image.flags.writeable)
        self.assertTrue(image.flags.c_contiguous)
        image[0, 0] = (1, 2, 3)
        np.testing.assert_array_equal(image[0, 0], (1, 2, 3))

    def test_save_prediction_passes_writable_buffers_to_postprocessor(self):
        class BufferCheckingPostProcessor:
            def __call__(self, image, probabilities):
                if not image.flags.writeable or not image.flags.c_contiguous:
                    raise AssertionError("image buffer is not writable C-order")
                if not probabilities.flags.writeable or not probabilities.flags.c_contiguous:
                    raise AssertionError("probability buffer is not writable C-order")
                return probabilities

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jpg"
            destination = root / "prediction.png"
            Image.fromarray(np.zeros((6, 7, 3), dtype=np.uint8), mode="RGB").save(source)
            logits = torch.zeros((1, 10, 3, 4), dtype=torch.float32)
            stats = _save_prediction(
                destination, source, logits, BufferCheckingPostProcessor()
            )
            self.assertTrue(destination.is_file())
            self.assertEqual((stats["width"], stats["height"]), (7, 6))


if __name__ == "__main__":
    unittest.main()
