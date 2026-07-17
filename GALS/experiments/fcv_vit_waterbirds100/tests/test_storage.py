from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.storage import StorageBudgetError, assert_storage_budget  # noqa: E402


class StorageBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "storage": {
                "hard_budget_gib": 40.0,
                "launch_guard_gib": 35.0,
                "worst_case_concurrent_growth_gib": 5.0,
            }
        }

    def test_projected_concurrent_growth_is_enforced_against_hard_cap(self) -> None:
        gib = 1024 ** 3
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "fcv.storage.allocated_bytes", return_value=int(34.5 * gib)
        ):
            receipt = assert_storage_budget(
                self.config, temporary, stage="before_epoch"
            )
        self.assertAlmostEqual(receipt["projected_peak_gib"], 39.5)

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "fcv.storage.allocated_bytes", return_value=int(35.5 * gib)
        ):
            with self.assertRaisesRegex(
                StorageBudgetError, "Projected hard storage budget exceeded"
            ):
                assert_storage_budget(self.config, temporary, stage="before_epoch")


if __name__ == "__main__":
    unittest.main()
