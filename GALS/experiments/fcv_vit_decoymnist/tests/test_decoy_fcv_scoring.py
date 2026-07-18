from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_fcv_scoring import (  # noqa: E402
    FCVScoringError,
    build_control_assignments,
    harmonic_fcv_score,
    majority_vote,
)


def _control_fixture():
    records = []
    for label in range(2):
        for index in range(6):
            target_id = f"label{label}_sample{index}"
            records.append(
                {
                    "target_sample_id": target_id,
                    "target_label": label,
                    "corner": "top_left",
                    "donors": [],
                }
            )
    return {"records": records}


class HarmonicSelectorTest(unittest.TestCase):
    def test_harmonic_score_is_symmetric_and_penalizes_imbalance(self) -> None:
        score = harmonic_fcv_score(0.9, 0.5)
        self.assertAlmostEqual(score, harmonic_fcv_score(0.5, 0.9))
        self.assertAlmostEqual(score, 2 * 0.9 * 0.5 / (0.9 + 0.5 + 1.0e-12))
        self.assertLess(score, (0.9 + 0.5) / 2.0)

    def test_harmonic_rule_rejects_posthoc_reweighting(self) -> None:
        with self.assertRaisesRegex(FCVScoringError, "locked"):
            harmonic_fcv_score(0.8, 0.7, epsilon=1.0e-6)

    def test_majority_vote_is_ten_class(self) -> None:
        self.assertEqual(majority_vote([7, 7, 2, 7, 2]), 7)
        with self.assertRaises(FCVScoringError):
            majority_vote([10])


class ControlAssignmentTest(unittest.TestCase):
    def test_assignments_are_deterministic_nonself_and_same_context(self) -> None:
        plan = _control_fixture()
        first = build_control_assignments(plan, seed=0, donors_per_target=5)
        second = build_control_assignments(plan, seed=0, donors_per_target=5)
        self.assertEqual(first, second)
        labels = {
            record["target_sample_id"]: record["target_label"]
            for record in plan["records"]
        }
        for target_id, assignment in first.items():
            donors = assignment["same_context_donor_ids"]
            self.assertEqual(len(donors), 5)
            self.assertNotIn(target_id, donors)
            self.assertTrue(all(labels[donor] == labels[target_id] for donor in donors))
            self.assertNotEqual(
                assignment["shuffled_mask_source_sample_id"], target_id
            )


if __name__ == "__main__":
    unittest.main()
