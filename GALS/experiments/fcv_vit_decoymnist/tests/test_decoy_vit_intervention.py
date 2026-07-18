from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_teacher_masks import (  # noqa: E402
    CATEGORY_AMBIGUOUS,
    CATEGORY_BACKGROUND,
    CATEGORY_EVIDENCE,
)
from decoy_vit_intervention import (  # noqa: E402
    ViTInterventionError,
    mutually_safe_positions,
    replace_spatially_aligned_tokens,
    require_positions_present,
)


class SpatialInterventionTest(unittest.TestCase):
    def test_mutual_background_preserves_evidence_and_ambiguous_positions(self) -> None:
        target = np.asarray(
            [CATEGORY_BACKGROUND, CATEGORY_EVIDENCE, CATEGORY_AMBIGUOUS, CATEGORY_BACKGROUND]
        )
        donor = np.asarray(
            [CATEGORY_BACKGROUND, CATEGORY_BACKGROUND, CATEGORY_AMBIGUOUS, CATEGORY_EVIDENCE]
        )
        positions = mutually_safe_positions(target, donor)
        np.testing.assert_array_equal(positions, np.asarray([0]))

    def test_mutual_evidence_is_separate_control(self) -> None:
        target = np.asarray(
            [CATEGORY_EVIDENCE, CATEGORY_EVIDENCE, CATEGORY_BACKGROUND]
        )
        donor = np.asarray(
            [CATEGORY_EVIDENCE, CATEGORY_BACKGROUND, CATEGORY_EVIDENCE]
        )
        positions = mutually_safe_positions(
            target, donor, category=CATEGORY_EVIDENCE
        )
        np.testing.assert_array_equal(positions, np.asarray([0]))

    def test_required_decoy_cells_fail_closed(self) -> None:
        require_positions_present([0, 1, 7], [1, 7])
        with self.assertRaisesRegex(ViTInterventionError, "required decoy"):
            require_positions_present([0, 1], [1, 7])

    def test_tensor_replacement_integrity_when_torch_is_available(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("Local lightweight environment has no importable PyTorch")
        target = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        donor = target + 100.0
        replaced, audit = replace_spatially_aligned_tokens(target, donor, [1, 3])
        torch.testing.assert_close(replaced[[1, 3]], donor[[1, 3]])
        torch.testing.assert_close(replaced[[0, 2, 4]], target[[0, 2, 4]])
        self.assertEqual(audit.replaced_patch_count, 2)
        self.assertEqual(audit.preserved_token_max_abs_error, 0.0)
        self.assertEqual(audit.donor_token_max_abs_error, 0.0)


if __name__ == "__main__":
    unittest.main()
