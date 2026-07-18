from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import decoy_online_evaluation as evaluation  # noqa: E402


class _FakeMetrics:
    def as_dict(self):
        return {
            "count": 10,
            "loss": 1.0,
            "accuracy": 0.5,
            "balanced_class_accuracy": 0.5,
            "worst_class_accuracy": 0.0,
            "per_class_accuracy": [0.5] * 10,
        }


class OnlineEvaluationBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_evaluator = evaluation.evaluate_classifier
        evaluation.evaluate_classifier = lambda *_args, **_kwargs: _FakeMetrics()

    def tearDown(self) -> None:
        evaluation.evaluate_classifier = self.original_evaluator

    def test_oracle_result_is_privileged_and_aggregate_only(self) -> None:
        result = evaluation.evaluate_oracle_online(
            object(), object(), object(), precision="amp_bfloat16", num_classes=10
        )
        self.assertEqual(result["visibility"], "oracle_analysis_only")
        self.assertEqual(result["selector_authorization"], "oracle_only")
        self.assertFalse(result["per_image_predictions_persisted"])
        self.assertNotIn("predictions", result)

    def test_test_result_is_posthoc_only_and_cannot_control_training(self) -> None:
        result = evaluation.evaluate_test_analysis_only(
            object(), object(), object(), precision="amp_bfloat16", num_classes=10
        )
        self.assertEqual(result["visibility"], "test_analysis_only")
        self.assertEqual(result["selector_authorization"], "posthoc_only")
        self.assertFalse(result["training_or_stopping_authorized"])
        self.assertFalse(result["per_image_predictions_persisted"])
        self.assertNotIn("predictions", result)


if __name__ == "__main__":
    unittest.main()
