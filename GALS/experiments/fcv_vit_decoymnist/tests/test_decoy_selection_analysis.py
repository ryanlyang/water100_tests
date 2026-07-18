from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
SCRIPTS = EXPERIMENT_ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from decoy_selection_analysis import (  # noqa: E402
    _load_selector_pool,
    compute_gap_closure,
    freeze_selector_matrix,
    rank_quality_rows,
    select_best_candidate,
    analyze_posthoc_results,
)
from decoy_full_config import (  # noqa: E402
    canonical_config_sha256,
    enumerate_runs,
    load_and_validate_config,
    sha256_file,
)
from decoy_manifest_provenance import atomic_json  # noqa: E402
from decoy_online_schema import (  # noqa: E402
    atomic_write_namespace_rows,
    namespace_columns,
    namespace_output_path,
)
from finalize_full_campaign_smoke import smoke_projections  # noqa: E402


class SelectionAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_and_validate_config(
            EXPERIMENT_ROOT
            / "configs"
            / "decoymnist_vit_s16_fcv_full_online.yaml"
        )

    def test_exact_ties_use_ascending_candidate_id(self) -> None:
        frame = pd.DataFrame(
            {
                "candidate_id": ["candidate_b", "candidate_a", "candidate_c"],
                "score": [0.8, 0.8, 0.7],
            }
        )
        selected = select_best_candidate(frame, "score")
        self.assertEqual(selected["candidate_id"], "candidate_a")

    def test_gap_closure_is_unclipped_and_undefined_without_oracle_headroom(self) -> None:
        defined = compute_gap_closure(0.20, 0.50, 0.40)
        self.assertTrue(defined["defined"])
        self.assertAlmostEqual(defined["unclipped_gap_closed"], 1.5)
        undefined = compute_gap_closure(0.40, 0.50, 0.40)
        self.assertFalse(undefined["defined"])
        self.assertIsNone(undefined["unclipped_gap_closed"])

    def test_rank_report_uses_locked_top_k_overlap(self) -> None:
        frame = pd.DataFrame(
            {
                "candidate_id": [f"c{index}" for index in range(5)],
                "biased_validation_accuracy": [0.1, 0.2, 0.3, 0.4, 0.5],
                "harmonic_fcv_score": [0.5, 0.4, 0.3, 0.2, 0.1],
                "oracle_validation_accuracy": [0.1, 0.3, 0.2, 0.5, 0.4],
                "test_accuracy": [0.5, 0.4, 0.3, 0.2, 0.1],
            }
        )
        report = rank_quality_rows(frame, ("vanilla", "fcv", "oracle"), (1, 5))
        fcv_top1 = report.loc[
            (report["selector"] == "fcv") & (report["top_k"] == 1)
        ].iloc[0]
        self.assertEqual(fcv_top1["top_k_recall"], 1.0)
        self.assertAlmostEqual(fcv_top1["spearman_vs_test_accuracy"], 1.0)

    def test_freeze_path_has_no_posthoc_namespace_loader(self) -> None:
        source = inspect.getsource(freeze_selector_matrix) + inspect.getsource(
            _load_selector_pool
        )
        self.assertNotIn("test_analysis_only", source)
        self.assertNotIn("posthoc", source)

    def test_smoke_projection_scales_one_epoch_and_fails_closed(self) -> None:
        runtime, storage = smoke_projections(
            observed_seconds=100.0,
            observed_workspace_bytes=10_000,
            current_campaign_bytes=20_000,
            expected_candidates=1080,
            production_epochs=10,
            storage_budget_bytes=1024**3,
        )
        self.assertEqual(runtime["projected_task_seconds"], 1500.0)
        self.assertTrue(runtime["within_task_limit"])
        self.assertTrue(storage["within_budget"])

    def test_tigris_dependency_chain_and_resources_are_locked(self) -> None:
        array = (EXPERIMENT_ROOT / "slurm" / "full_campaign_online_array.sbatch").read_text()
        submit = (EXPERIMENT_ROOT / "scripts" / "submit_full_campaign.sh").read_text()
        self.assertIn("#SBATCH --account=reu-aisocial", array)
        self.assertIn("#SBATCH --partition=tigris", array)
        self.assertIn("#SBATCH --gres=gpu:gh200:1", array)
        self.assertIn("#SBATCH --array=0-107%8", array)
        self.assertIn("#SBATCH --time=1-00:00:00", array)
        self.assertIn('"afterok:${preflight_job}"', submit)
        self.assertIn('"afterok:${smoke_job}"', submit)
        self.assertIn('"afterok:${array_job}"', submit)
        self.assertIn('"afterok:${freeze_job}"', submit)

    @staticmethod
    def _namespace_row(namespace, campaign_run, epoch):
        row = {column: 0.0 for column in namespace_columns(namespace)}
        row.update(
            {
                "run_index": campaign_run.run_index,
                "run_id": campaign_run.run_id,
                "candidate_id": campaign_run.candidate_id(epoch),
                "epoch": epoch,
                "seed": campaign_run.seed,
                "learning_rate": campaign_run.learning_rate,
                "weight_decay": campaign_run.weight_decay,
                "crop_scale_min": campaign_run.crop_scale_min,
            }
        )
        normalized_index = (campaign_run.run_index * 10 + epoch) / 1081.0
        if namespace == "biased_validation":
            row["biased_validation_accuracy"] = normalized_index
        elif namespace == "fcv":
            row["harmonic_fcv_score"] = 1.0 - normalized_index
            row["fcv_counterfactual_accuracy"] = 0.25 + 0.5 * normalized_index
        elif namespace == "oracle_analysis_only":
            row["oracle_validation_accuracy"] = 0.5 + 0.25 * normalized_index
        elif namespace == "test_analysis_only":
            row["test_accuracy"] = 0.2 + 0.6 * normalized_index
            row["test_balanced_class_accuracy"] = row["test_accuracy"]
            row["test_worst_class_accuracy"] = max(0.0, row["test_accuracy"] - 0.1)
            for label in range(10):
                row[f"test_class_{label}_accuracy"] = row["test_accuracy"]
        elif namespace == "controls":
            row["control_diagnostics_warning_only"] = True
            row["control_warning_count"] = 0
            row["control_warning_reason_counts_json"] = "{}"
            for column in row:
                if column.endswith("_status"):
                    row[column] = "complete"
                elif column.endswith("_counterfactual_accuracy"):
                    row[column] = 0.3 + 0.2 * normalized_index
        return row

    def test_complete_1080_candidate_freeze_and_posthoc_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            namespaces = (
                "biased_validation",
                "fcv",
                "controls",
                "oracle_analysis_only",
                "test_analysis_only",
            )
            for campaign_run in enumerate_runs(self.config):
                records = {}
                for namespace in namespaces:
                    path = namespace_output_path(root, namespace, campaign_run.run_id)
                    rows = [
                        self._namespace_row(namespace, campaign_run, epoch)
                        for epoch in range(1, 11)
                    ]
                    atomic_write_namespace_rows(namespace, rows, path)
                    records[namespace] = {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "row_count": 10,
                    }
                summary_path = root / "run_summaries" / f"{campaign_run.run_id}.json"
                atomic_json(
                    {
                        "artifact_type": "fcv_vit_decoymnist_online_run_summary",
                        "artifact_version": 1,
                        "status": "complete",
                        "execution_mode": "production",
                        "config_sha256": canonical_config_sha256(self.config),
                        "run": asdict(campaign_run),
                        "completed_candidate_count": 10,
                        "namespace_artifacts": records,
                        "preflight_receipt_sha256": "preflight",
                        "pretrained_backbone_sha256": "backbone",
                        "test_metrics_used_for_training_or_selection": False,
                    },
                    summary_path,
                )
            fake_gate = {
                "artifact_path": str(root / "preflight" / "launch_gate.json"),
                "artifact_sha256": "gate",
                "preflight_receipt_sha256": "preflight",
                "pretrained_backbone_sha256": "backbone",
            }
            with mock.patch(
                "decoy_selection_analysis.validate_launch_gate",
                return_value=fake_gate,
            ):
                freeze = freeze_selector_matrix(self.config, root)
            self.assertEqual(freeze["candidate_count"], 1080)
            final = analyze_posthoc_results(self.config, root)
            self.assertEqual(final["candidate_count"], 1080)
            self.assertTrue(final["test_metrics_attached_after_selection_freeze"])
            self.assertFalse(final["test_metrics_affected_selection"])
            frozen_columns = pd.read_csv(
                root / "selection_results" / "frozen_selector_scores.csv",
                nrows=1,
            ).columns
            self.assertFalse(any(column.startswith("test_") for column in frozen_columns))
            outcomes = pd.read_csv(
                root / "selection_results" / "selector_test_outcomes.csv"
            )
            self.assertEqual(set(outcomes["selector"]), {"vanilla", "fcv", "oracle", "posthoc"})


if __name__ == "__main__":
    unittest.main()
