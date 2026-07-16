from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.candidate_training import (  # noqa: E402
    candidate_training_fingerprint,
    enumerate_sweep_runs,
)
from fcv.config import load_and_validate_config  # noqa: E402
from fcv.gap_analysis import (  # noqa: E402
    POOL_TEST_COLUMNS,
    GapAnalysisError,
    aggregate_pool_test_summaries,
    compute_gap_closure_summary,
    gap_analysis_fingerprint,
)
from fcv.test_evaluation import (  # noqa: E402
    FrozenSelection,
    SelectedCheckpoint,
    final_test_evaluation_fingerprint,
    recompute_test_metrics_from_frame,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class GapAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        config_path = (
            EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["paths"]["output_root"] = str(self.root / "outputs")
        local_config = self.root / "config.yaml"
        with local_config.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(raw, handle)
        self.config = load_and_validate_config(local_config)
        self.source_manifest = self.root / "test_manifest.csv"
        self.source_manifest.write_text("synthetic\n", encoding="utf-8")
        manifest_frame = pd.DataFrame(
            [
                {
                    "sample_id": f"test_{group}_{index}",
                    "label": 0 if group < 2 else 1,
                    "group": group,
                }
                for group in range(4)
                for index in range(10)
            ]
        )
        self.source = SimpleNamespace(
            manifest_path=self.source_manifest.resolve(),
            manifest_sha256=sha256_file(self.source_manifest),
            manifest_bundle_path=self.root / "SYNTHETIC_TEST_MANIFEST_BUNDLE",
            manifest_bundle_sha256="SYNTHETIC_TEST_MODE",
            dataset=SimpleNamespace(frame=manifest_frame),
            sample_count=40,
            batch_size=128,
            num_workers=8,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _metrics(worst: float) -> dict:
        correct = int(round(worst * 10))
        group_correct = [correct, correct, correct, correct]
        return {
            "loss": 0.5,
            "accuracy": worst,
            "balanced_group_accuracy": worst,
            "worst_group_accuracy": worst,
            "sample_count": 40,
            **{
                key: value
                for group in range(4)
                for key, value in {
                    f"group_{group}_accuracy": worst,
                    f"group_{group}_correct": group_correct[group],
                    f"group_{group}_count": 10,
                }.items()
            },
        }

    def _per_image(self, candidate_id: str, accuracy: float, path: Path) -> dict:
        correct_per_group = int(round(accuracy * 10))
        rows = []
        for row in self.source.dataset.frame.itertuples(index=False):
            within_group = int(str(row.sample_id).rsplit("_", 1)[1])
            prediction = int(row.label) if within_group < correct_per_group else 1 - int(row.label)
            probabilities = [0.2, 0.2]
            probabilities[prediction] = 0.8
            logits = [float(np.log(value)) for value in probabilities]
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "sample_id": row.sample_id,
                    "label": int(row.label),
                    "group": int(row.group),
                    "prediction": prediction,
                    "correct": int(prediction == int(row.label)),
                    "true_class_probability": probabilities[int(row.label)],
                    "loss": -float(np.log(probabilities[int(row.label)])),
                    "logits": json.dumps(logits, separators=(",", ":")),
                    "probabilities": json.dumps(probabilities, separators=(",", ":")),
                }
            )
        frame = pd.DataFrame(rows)
        frame.to_csv(path, index=False)
        return recompute_test_metrics_from_frame(
            frame, self.source, candidate_id=candidate_id
        )

    def _make_frozen_and_final(self, values: tuple[float, float, float]):
        selector_names = [
            "biased_validation_accuracy",
            "equal_weight_original_and_opposite_fcv_accuracy",
            "oracle_validation_balanced_group_accuracy",
        ]
        candidate_ids = ["candidate_z", "candidate_b", "candidate_a"]
        checkpoints = []
        for candidate_id in candidate_ids:
            checkpoint = self.root / f"{candidate_id}.pt"
            checkpoint.write_bytes(candidate_id.encode("utf-8"))
            checkpoints.append(checkpoint)
        table = pd.DataFrame(
            {
                "selector_name": selector_names,
                "selector_family": ["biased_validation", "fcv_primary", "oracle"],
                "availability": ["public", "public", "privileged"],
                "direction": ["maximize", "maximize", "maximize"],
                "selector_formula": ["biased", "fcv", "oracle"],
                "selector_score": [0.9, 0.8, 0.7],
                "selected_checkpoint_id": candidate_ids,
                "selected_checkpoint_path": [str(path) for path in checkpoints],
                "selected_checkpoint_sha256": [
                    sha256_file(path) for path in checkpoints
                ],
                "selected_hparams": ["{}", "{}", "{}"],
            }
        )
        selection_table = self.root / "selection_table.csv"
        table.to_csv(selection_table, index=False)
        matrix = self.root / "candidate_selector_scores.csv"
        pd.DataFrame(
            {
                "candidate_id": candidate_ids,
                "checkpoint_path": [str(path) for path in checkpoints],
                "checkpoint_sha256": [sha256_file(path) for path in checkpoints],
            }
        ).to_csv(matrix, index=False)
        selection_summary = self.root / "selection_summary.json"
        selection_summary.write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
        frozen = FrozenSelection(
            selection_table_path=selection_table.resolve(),
            selection_table_sha256=sha256_file(selection_table),
            selection_summary_path=selection_summary.resolve(),
            selector_matrix_path=matrix.resolve(),
            selector_matrix_sha256=sha256_file(matrix),
            table=table,
            unique_checkpoints=tuple(),
            pool_checkpoints=tuple(
                SelectedCheckpoint(
                    candidate_id=candidate_id,
                    checkpoint_path=path.resolve(),
                    checkpoint_sha256=sha256_file(path),
                    selectors=(),
                )
                for candidate_id, path in zip(candidate_ids, checkpoints)
            ),
        )
        final = table.copy()
        final["test_average_accuracy"] = list(values)
        final["test_balanced_group_accuracy"] = list(values)
        final["test_worst_group_accuracy"] = list(values)
        final["test_sample_count"] = 40
        per_image_paths = {}
        per_image_hashes = {}
        for candidate_id, value in zip(candidate_ids, values):
            path = self.root / f"{candidate_id}_test_per_image.csv"
            self._per_image(candidate_id, value, path)
            per_image_paths[candidate_id] = str(path.resolve())
            per_image_hashes[candidate_id] = sha256_file(path)
        final["test_per_image_path"] = [
            per_image_paths[candidate_id] for candidate_id in candidate_ids
        ]
        final_csv = self.root / "final_test_results.csv"
        final.to_csv(final_csv, index=False)
        final_summary = self.root / "final_test_results_summary.json"
        final_summary.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "artifact_type": "fcv_vit_final_test_results_summary",
                    "status": "complete",
                    "selection_table_path": str(frozen.selection_table_path),
                    "selection_table_sha256": frozen.selection_table_sha256,
                    "candidate_selector_matrix_path": str(frozen.selector_matrix_path),
                    "candidate_selector_matrix_sha256": frozen.selector_matrix_sha256,
                    "final_test_evaluation_fingerprint": final_test_evaluation_fingerprint(
                        self.config
                    ),
                    "test_manifest_path": str(self.source.manifest_path),
                    "test_manifest_sha256": self.source.manifest_sha256,
                    "manifest_bundle_path": str(self.source.manifest_bundle_path),
                    "manifest_bundle_sha256": self.source.manifest_bundle_sha256,
                    "test_sample_count": 40,
                    "execution": {"batch_size": 128, "num_workers": 8},
                    "final_test_results_path": str(final_csv.resolve()),
                    "final_test_results_sha256": sha256_file(final_csv),
                    "candidate_test_per_image_paths": per_image_paths,
                    "candidate_test_per_image_sha256": per_image_hashes,
                    "selection_frozen_before_test": True,
                    "test_metrics_affected_selection": False,
                    "selector_order_preserved": True,
                }
            ),
            encoding="utf-8",
        )
        return frozen, final_csv, final_summary

    def _make_pool_index(self, frozen: FrozenSelection):
        rows = []
        for candidate_id, worst, run_index in (
            ("candidate_z", 0.9, 0),
            ("candidate_b", 0.7, 1),
            ("candidate_a", 0.9, 2),
        ):
            row = {
                "run_index": run_index,
                "candidate_id": candidate_id,
                "epoch": 1,
                "seed": run_index,
                "learning_rate": 1.0e-5,
                "weight_decay": 0.01,
                "checkpoint_path": str(self.root / f"{candidate_id}.pt"),
                "checkpoint_sha256": sha256_file(
                    self.root / f"{candidate_id}.pt"
                ),
                "test_loss": 0.5,
                "test_accuracy": worst,
                "test_balanced_group_accuracy": worst,
                "test_worst_group_accuracy": worst,
                "summary_path": str(self.root / f"{candidate_id}_summary.json"),
            }
            per_image_path = self.root / f"{candidate_id}_pool_per_image.csv"
            metrics = self._per_image(candidate_id, worst, per_image_path)
            row["test_loss"] = metrics["loss"]
            row["per_image_path"] = str(per_image_path.resolve())
            row["per_image_sha256"] = sha256_file(per_image_path)
            for group in range(4):
                row[f"test_group_{group}_accuracy"] = worst
                row[f"test_group_{group}_count"] = 10
            rows.append(row)
        pool_csv = self.root / "candidate_pool_test_scores.csv"
        pd.DataFrame(rows, columns=POOL_TEST_COLUMNS).to_csv(pool_csv, index=False)
        pool_summary = self.root / "candidate_pool_test_scores_summary.json"
        pool_summary.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "artifact_type": "fcv_vit_posthoc_pool_test_index_summary",
                    "status": "complete",
                    "candidate_count": 3,
                    "expected_candidate_count": 3,
                    "gap_analysis_fingerprint": gap_analysis_fingerprint(self.config),
                    "test_manifest_path": str(self.source.manifest_path),
                    "test_manifest_sha256": self.source.manifest_sha256,
                    "manifest_bundle_path": str(self.source.manifest_bundle_path),
                    "manifest_bundle_sha256": self.source.manifest_bundle_sha256,
                    "test_sample_count": 40,
                    "execution": {"batch_size": 128, "num_workers": 8},
                    "output_csv": str(pool_csv.resolve()),
                    "output_csv_sha256": sha256_file(pool_csv),
                    "test_data_accessed": True,
                    "posthoc_pool_analysis_only": True,
                    "eligible_for_model_selection": False,
                    "test_metrics_affected_selection": False,
                    "selection_frozen_before_test": True,
                    "selection_table_path": str(frozen.selection_table_path),
                    "selection_table_sha256": frozen.selection_table_sha256,
                    "selection_summary_path": str(frozen.selection_summary_path),
                    "selection_summary_sha256": sha256_file(
                        frozen.selection_summary_path
                    ),
                    "candidate_selector_matrix_path": str(
                        frozen.selector_matrix_path
                    ),
                    "candidate_selector_matrix_sha256": frozen.selector_matrix_sha256,
                }
            ),
            encoding="utf-8",
        )
        return pool_csv, pool_summary

    def test_gap_formula_and_pool_tie_break(self) -> None:
        self.config["candidate_pool"]["expected_candidate_checkpoints"] = 3
        frozen, final_csv, final_summary = self._make_frozen_and_final(
            (0.4, 0.6, 0.8)
        )
        pool_csv, pool_summary = self._make_pool_index(frozen)
        output_csv = self.root / "gap_closure_summary.csv"
        output_summary = self.root / "gap_closure_summary.json"
        result = compute_gap_closure_summary(
            self.config,
            frozen,
            self.source,
            final_csv,
            final_summary,
            pool_csv,
            pool_summary,
            output_csv,
            output_summary,
        )
        row = pd.read_csv(output_csv).iloc[0]
        self.assertAlmostEqual(row["gap_closed_fraction"], 0.5)
        self.assertAlmostEqual(row["gap_closed_percent"], 50.0)
        self.assertEqual(row["gap_closure_status"], "defined_positive_oracle_gap")
        self.assertEqual(row["pool_upper_bound_candidate_id"], "candidate_a")
        self.assertTrue(result["posthoc_pool_upper_bound_is_not_a_selector"])
        self.assertFalse(result["test_metrics_affected_selection"])

    def test_zero_oracle_gap_is_explicitly_undefined(self) -> None:
        self.config["candidate_pool"]["expected_candidate_checkpoints"] = 3
        frozen, final_csv, final_summary = self._make_frozen_and_final(
            (0.4, 0.6, 0.4)
        )
        pool_csv, pool_summary = self._make_pool_index(frozen)
        output_csv = self.root / "zero_gap.csv"
        compute_gap_closure_summary(
            self.config,
            frozen,
            self.source,
            final_csv,
            final_summary,
            pool_csv,
            pool_summary,
            output_csv,
            self.root / "zero_gap.json",
        )
        row = pd.read_csv(output_csv).iloc[0]
        self.assertEqual(row["gap_closure_status"], "undefined_zero_oracle_gap")
        self.assertTrue(pd.isna(row["gap_closed_fraction"]))

    def test_pool_index_is_cryptographically_bound_to_frozen_selection(self) -> None:
        self.config["candidate_pool"]["expected_candidate_checkpoints"] = 3
        frozen, final_csv, final_summary = self._make_frozen_and_final(
            (0.4, 0.6, 0.8)
        )
        pool_csv, pool_summary = self._make_pool_index(frozen)
        frozen.selection_summary_path.write_text(
            json.dumps({"status": "changed_after_pool_scoring"}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(GapAnalysisError, "stale or incomplete"):
            compute_gap_closure_summary(
                self.config,
                frozen,
                self.source,
                final_csv,
                final_summary,
                pool_csv,
                pool_summary,
                self.root / "unsafe_gap.csv",
                self.root / "unsafe_gap.json",
            )

    def test_pool_aggregation_requires_every_posthoc_summary(self) -> None:
        config = deepcopy(self.config)
        config["training"]["learning_rates"] = [1.0e-5]
        config["training"]["weight_decays"] = [0.01]
        config["training"]["seeds"] = [0]
        config["training"]["epochs"] = 2
        config["candidate_pool"]["expected_training_runs"] = 1
        config["candidate_pool"]["expected_candidate_checkpoints"] = 2
        frozen, _, _ = self._make_frozen_and_final((0.4, 0.6, 0.8))
        run = enumerate_sweep_runs(config)[0]
        run_dir = self.root / "candidate_models" / run.run_id
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        metric_rows = []
        pool_dir = self.root / "pool_scores"
        pool_dir.mkdir()
        for epoch in (1, 2):
            candidate_id = run.candidate_id(epoch)
            checkpoint = checkpoints_dir / f"epoch_{epoch:03d}.pt"
            checkpoint.write_bytes(candidate_id.encode("utf-8"))
            metric_rows.append(
                {
                    "epoch": epoch,
                    "candidate_id": candidate_id,
                    "checkpoint_path": str(checkpoint.resolve()),
                }
            )
            per_image_path = pool_dir / f"{candidate_id}_pool_test_per_image.csv"
            metrics = self._per_image(candidate_id, 0.5 + epoch * 0.1, per_image_path)
            summary = {
                "schema_version": 2,
                "artifact_type": "fcv_vit_posthoc_pool_test_summary",
                "status": "complete",
                "candidate_id": candidate_id,
                "run": {
                    "run_index": run.run_index,
                    "learning_rate": run.learning_rate,
                    "weight_decay": run.weight_decay,
                    "seed": run.seed,
                },
                "epoch": epoch,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "training_fingerprint": candidate_training_fingerprint(config),
                "final_test_evaluation_fingerprint": final_test_evaluation_fingerprint(
                    config
                ),
                "gap_analysis_fingerprint": gap_analysis_fingerprint(config),
                "test_manifest_path": str(self.source.manifest_path),
                "test_manifest_sha256": self.source.manifest_sha256,
                "manifest_bundle_path": str(self.source.manifest_bundle_path),
                "manifest_bundle_sha256": self.source.manifest_bundle_sha256,
                "test_sample_count": 40,
                "per_image_csv_path": str(per_image_path.resolve()),
                "per_image_csv_sha256": sha256_file(per_image_path),
                "precision": "float32",
                "execution": {"batch_size": 128, "num_workers": 8},
                "metrics": metrics,
                "test_data_accessed": True,
                "posthoc_pool_analysis_only": True,
                "eligible_for_model_selection": False,
                "test_metrics_affected_selection": False,
                "selection_frozen_before_test": True,
                "selection_table_path": str(frozen.selection_table_path),
                "selection_table_sha256": frozen.selection_table_sha256,
                "selection_summary_path": str(frozen.selection_summary_path),
                "selection_summary_sha256": sha256_file(
                    frozen.selection_summary_path
                ),
                "candidate_selector_matrix_path": str(frozen.selector_matrix_path),
                "candidate_selector_matrix_sha256": frozen.selector_matrix_sha256,
            }
            (pool_dir / f"{candidate_id}_pool_test_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
        pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
        frozen = replace(
            frozen,
            pool_checkpoints=tuple(
                SelectedCheckpoint(
                    candidate_id=str(row["candidate_id"]),
                    checkpoint_path=Path(str(row["checkpoint_path"])).resolve(),
                    checkpoint_sha256=sha256_file(
                        Path(str(row["checkpoint_path"]))
                    ),
                    selectors=(),
                )
                for row in metric_rows
            ),
        )
        output_csv = self.root / "pool_index.csv"
        result = aggregate_pool_test_summaries(
            config,
            self.root / "candidate_models",
            pool_dir,
            output_csv,
            self.root / "pool_index.json",
            source=self.source,
            frozen=frozen,
        )
        self.assertEqual(result["candidate_count"], 2)
        self.assertFalse(result["eligible_for_model_selection"])

        nonselected_checkpoint = checkpoints_dir / "epoch_002.pt"
        original_bytes = nonselected_checkpoint.read_bytes()
        nonselected_checkpoint.write_bytes(b"changed after Step 9 freeze")
        with self.assertRaisesRegex(GapAnalysisError, "incomplete"):
            aggregate_pool_test_summaries(
                config,
                self.root / "candidate_models",
                pool_dir,
                self.root / "changed_checkpoint.csv",
                self.root / "changed_checkpoint.json",
                source=self.source,
                frozen=frozen,
            )
        nonselected_checkpoint.write_bytes(original_bytes)

        (pool_dir / f"{run.candidate_id(2)}_pool_test_summary.json").unlink()
        with self.assertRaisesRegex(GapAnalysisError, "incomplete"):
            aggregate_pool_test_summaries(
                config,
                self.root / "candidate_models",
                pool_dir,
                self.root / "incomplete.csv",
                self.root / "incomplete.json",
                source=self.source,
                frozen=frozen,
            )


if __name__ == "__main__":
    unittest.main()
