from __future__ import annotations

import ast
import copy
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_full_config import enumerate_runs, load_and_validate_config  # noqa: E402
from decoy_online_study import (  # noqa: E402
    OnlineStudyError,
    _analysis_row,
    _biased_row,
    _controls_row,
    _fcv_row,
    assert_no_forbidden_persistence,
)


def _metrics(value: float = 0.5):
    return {
        "count": 100,
        "loss": 1.0,
        "accuracy": value,
        "balanced_class_accuracy": value,
        "worst_class_accuracy": value,
        "per_class_accuracy": [value] * 10,
    }


def _fcv_aggregate():
    primary = {
        "eligible_target_count": 80,
        "eligible_target_fraction": 0.8,
        "donor_draw_count": 400,
        "counterfactual_accuracy": 0.6,
        "counterfactual_majority_accuracy": 0.61,
        "mean_true_class_counterfactual_probability": 0.55,
        "mean_original_to_counterfactual_confidence_drop": 0.1,
        "mean_replaced_patch_count": 40.0,
        "changed_replacement_fraction": 1.0,
    }
    control = {
        "status": "complete",
        "eligible_target_count": 80,
        "eligible_target_fraction": 0.8,
        "donor_draw_count": 400,
        "counterfactual_accuracy": 0.7,
        "counterfactual_majority_accuracy": 0.71,
        "mean_true_class_counterfactual_probability": 0.65,
        "mean_original_to_counterfactual_confidence_drop": 0.05,
        "mean_replaced_patch_count": 40.0,
    }
    controls = {
        name: copy.deepcopy(control)
        for name in (
            "same_context",
            "random_mask",
            "shuffled_teacher_mask",
            "evidence_swap",
            "exact_synthetic_mask_analysis_only",
        )
    }
    return {
        "original_biased_validation_accuracy": 0.9,
        "harmonic_fcv_score": 0.72,
        "primary_fcv": primary,
        "controls": controls,
        "control_diagnostics_warning_only": True,
        "control_warning_count": 0,
        "control_warning_reason_counts": {},
        "identity_forward": {"max_abs_error": 0.0},
    }


class OnlineStudyContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_and_validate_config(
            EXPERIMENT_ROOT
            / "configs"
            / "decoymnist_vit_s16_fcv_full_online.yaml"
        )
        cls.campaign_run = enumerate_runs(cls.config)[0]

    def test_row_builders_keep_values_in_their_namespaces(self) -> None:
        aggregate = _fcv_aggregate()
        biased = _biased_row(
            self.campaign_run,
            1,
            {"loss": 1.0, "accuracy": 0.8},
            _metrics(0.9),
            lr_start=1.0e-6,
            lr_end=1.0e-5,
            train_seconds=1.0,
        )
        fcv = _fcv_row(self.campaign_run, 1, aggregate, seconds=2.0)
        controls = _controls_row(self.campaign_run, 1, aggregate)
        oracle = _analysis_row(
            "oracle_analysis_only",
            "oracle_validation",
            self.campaign_run,
            1,
            {"metrics": _metrics(0.7)},
            seconds=3.0,
        )
        test = _analysis_row(
            "test_analysis_only",
            "test",
            self.campaign_run,
            1,
            {"metrics": _metrics(0.6)},
            seconds=4.0,
        )
        self.assertNotIn("test_accuracy", biased)
        self.assertNotIn("test_accuracy", fcv)
        self.assertNotIn("test_accuracy", controls)
        self.assertNotIn("test_accuracy", oracle)
        self.assertEqual(test["test_accuracy"], 0.6)

    def test_storage_guard_rejects_checkpoint_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "online_metrics").mkdir()
            (root / "online_metrics" / "safe.csv").write_text("a\n1\n")
            assert_no_forbidden_persistence(self.config, root)
            (root / "candidate.pt").write_bytes(b"forbidden")
            with self.assertRaisesRegex(OnlineStudyError, "Forbidden"):
                assert_no_forbidden_persistence(self.config, root)

    def test_orchestrator_contains_no_training_state_serialization(self) -> None:
        source_path = SRC / "decoy_online_study.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "save"
        ]
        self.assertEqual(calls, [])
        self.assertNotIn("model_state_dict", source)
        self.assertNotIn("optimizer_state_dict", source)
        # The source intentionally names forbidden resume/checkpoint filename
        # fragments in its fail-closed storage guard.  What must remain absent
        # is any serialization call or training-state extraction.


if __name__ == "__main__":
    unittest.main()
