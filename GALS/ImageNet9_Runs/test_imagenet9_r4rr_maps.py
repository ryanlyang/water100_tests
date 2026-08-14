#!/usr/bin/env python3

import unittest

import numpy as np

from audit_imagenet9_r4rr_weclip_maps import decode_voc_colors, voc_colormap
from generate_imagenet9_r4rr_weclip_maps import shard_bounds


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


if __name__ == "__main__":
    unittest.main()
