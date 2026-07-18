from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_data import (  # noqa: E402
    corner_slices,
    expected_patch_intensity,
    locate_decoy_patch,
)
from decoy_oracle_view import load_oracle_view, reverse_training_patch  # noqa: E402


class DecoyOracleViewTest(unittest.TestCase):
    def test_every_label_and_corner_reverses_only_the_patch(self) -> None:
        for label in range(10):
            for rows, columns in corner_slices(28, 28):
                source = np.zeros((28, 28), dtype=np.uint8)
                source[9:19, 11:17] = 120
                source[rows, columns] = expected_patch_intensity(label, "train")
                frozen = source.copy()
                reversed_view = reverse_training_patch(source, label)
                self.assertTrue(np.array_equal(source, frozen))
                expected = frozen.copy()
                expected[rows, columns] = expected_patch_intensity(label, "test")
                self.assertTrue(np.array_equal(reversed_view, expected))
                located_rows, located_columns = locate_decoy_patch(
                    reversed_view, label, "test"
                )
                if label != 0:
                    self.assertEqual(
                        (located_rows.start, located_columns.start),
                        (rows.start, columns.start),
                    )

    def test_file_loader_does_not_mutate_source_png(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.png"
            source = np.zeros((28, 28), dtype=np.uint8)
            source[10:18, 10:18] = 140
            source[-5:, :5] = expected_patch_intensity(4, "train")
            Image.fromarray(source, mode="L").save(path)
            before = path.read_bytes()
            digest = hashlib.sha256(before).hexdigest()
            oracle = load_oracle_view(path, 4, expected_sha256=digest)
            after = path.read_bytes()
            self.assertEqual(before, after)
            self.assertEqual(oracle.mode, "RGB")
            oracle_gray = np.asarray(oracle.convert("L"), dtype=np.uint8)
            self.assertTrue(
                np.all(oracle_gray[-5:, :5] == expected_patch_intensity(4, "test"))
            )


if __name__ == "__main__":
    unittest.main()

