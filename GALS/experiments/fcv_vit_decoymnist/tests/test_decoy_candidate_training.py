from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_candidate_training import (  # noqa: E402
    evaluation_transform_spec,
    training_transform_spec,
    warmup_cosine_factor,
)
from decoy_full_config import enumerate_runs, load_and_validate_config  # noqa: E402


class CandidateTrainingProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_and_validate_config(
            EXPERIMENT_ROOT
            / "configs"
            / "decoymnist_vit_s16_fcv_full_online.yaml"
        )

    def test_grid_changes_only_locked_crop_minimum_in_geometry(self) -> None:
        runs = enumerate_runs(self.config)
        by_crop = {}
        for run in runs:
            by_crop.setdefault(run.crop_scale_min, training_transform_spec(self.config, run))
        self.assertEqual(set(by_crop), {1.0, 0.8, 0.6, 0.4})
        for crop_min, spec in by_crop.items():
            self.assertEqual(spec["scale"], [crop_min, 1.0])
            self.assertEqual(spec["ratio"], [1.0, 1.0])
            self.assertEqual(spec["interpolation"], "bicubic")
            self.assertEqual(spec["horizontal_flip_probability"], 0.0)

    def test_evaluation_geometry_is_crop_invariant_direct_resize(self) -> None:
        spec = evaluation_transform_spec(self.config)
        self.assertEqual(spec["operation"], "Resize")
        self.assertEqual(spec["size"], [224, 224])
        self.assertIsNone(spec["crop"])
        self.assertEqual(spec["horizontal_flip_probability"], 0.0)

    def test_warmup_cosine_schedule_has_locked_shape(self) -> None:
        factors = [
            warmup_cosine_factor(step, total_steps=100, warmup_steps=10)
            for step in range(100)
        ]
        self.assertAlmostEqual(factors[0], 0.1)
        self.assertAlmostEqual(factors[9], 1.0)
        self.assertAlmostEqual(factors[10], 1.0)
        self.assertGreater(factors[50], factors[99])
        self.assertGreaterEqual(min(factors), 0.0)


if __name__ == "__main__":
    unittest.main()
