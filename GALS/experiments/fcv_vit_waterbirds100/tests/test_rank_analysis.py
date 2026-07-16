from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.gap_analysis import POOL_TEST_COLUMNS, gap_analysis_fingerprint  # noqa: E402
from fcv.rank_analysis import (  # noqa: E402
    RankAnalysisError,
    analyze_rank_quality,
)
from fcv.test_evaluation import (  # noqa: E402
    FrozenSelection,
    SelectedCheckpoint,
    recompute_test_metrics_from_frame,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class RankAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        os.environ["MPLCONFIGDIR"] = str(self.root / "matplotlib_cache")
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
        self.config["candidate_pool"]["expected_candidate_checkpoints"] = 4
        self.config["evaluation"]["rank_analysis"]["top_k_values"] = [1, 2, 3, 4]
        self.config["evaluation"]["rank_analysis"]["clustered_bootstrap"][
            "replicates"
        ] = 100
        self.manifest = self.root / "test_manifest.csv"
        self.manifest.write_text("synthetic\n", encoding="utf-8")
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
            manifest_path=self.manifest.resolve(),
            manifest_sha256=sha256_file(self.manifest),
            manifest_bundle_path=self.root / "SYNTHETIC_TEST_MANIFEST_BUNDLE",
            manifest_bundle_sha256="SYNTHETIC_TEST_MODE",
            dataset=SimpleNamespace(frame=manifest_frame),
            sample_count=40,
            batch_size=128,
            num_workers=8,
        )
        self.frozen, self.pool_csv, self.pool_summary = self._make_inputs()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_inputs(self):
        candidate_ids = ["candidate_a", "candidate_b", "candidate_c", "candidate_d"]
        checkpoints = []
        for candidate_id in candidate_ids:
            path = self.root / f"{candidate_id}.pt"
            path.write_bytes(candidate_id.encode("utf-8"))
            checkpoints.append(str(path))
        identity = {
            "run_index": [0, 1, 2, 3],
            "candidate_id": candidate_ids,
            "epoch": [1, 1, 1, 1],
            "seed": [0, 1, 2, 3],
            "learning_rate": [1.0e-5, 2.0e-5, 3.0e-5, 4.0e-5],
            "weight_decay": [0.01, 0.02, 0.03, 0.04],
            "checkpoint_path": checkpoints,
            "checkpoint_sha256": [
                sha256_file(Path(path)) for path in checkpoints
            ],
        }
        selector_matrix = pd.DataFrame(
            {
                **identity,
                "biased_val_accuracy": [0.9, 0.8, 0.7, 0.6],
                "biased_val_loss": [0.4, 0.3, 0.2, 0.1],
                "primary_selector_score": [0.1, 0.2, 0.3, 0.4],
                "fcv_counterfactual_accuracy": [0.2, 0.3, 0.4, 0.5],
                "fcv_true_class_probability": [0.7, 0.6, 0.5, 0.4],
                "oracle_validation_balanced_group_accuracy": [0.1, 0.2, 0.3, 0.4],
            }
        )
        matrix_path = self.root / "candidate_selector_scores.csv"
        selector_matrix.to_csv(matrix_path, index=False)
        selector_names = [
            "biased_validation_accuracy",
            "biased_validation_loss",
            "equal_weight_original_and_opposite_fcv_accuracy",
            "opposite_context_counterfactual_accuracy",
            "opposite_context_true_class_probability",
            "oracle_validation_balanced_group_accuracy",
        ]
        selected_ids = [
            "candidate_a",
            "candidate_d",
            "candidate_d",
            "candidate_d",
            "candidate_a",
            "candidate_d",
        ]
        selected_paths = [
            checkpoints[candidate_ids.index(candidate_id)] for candidate_id in selected_ids
        ]
        selection_table = pd.DataFrame(
            {
                "selector_name": selector_names,
                "selector_family": ["biased", "biased", "fcv", "fcv", "fcv", "oracle"],
                "availability": ["public", "public", "public", "public", "public", "oracle"],
                "direction": [
                    "maximize",
                    "minimize",
                    "maximize",
                    "maximize",
                    "maximize",
                    "maximize",
                ],
                "selector_formula": selector_names,
                "selector_score": [0.9, 0.1, 0.4, 0.5, 0.7, 0.4],
                "selected_checkpoint_id": selected_ids,
                "selected_checkpoint_path": selected_paths,
                "selected_checkpoint_sha256": [
                    sha256_file(Path(path)) for path in selected_paths
                ],
                "selected_hparams": ["{}"] * 6,
            }
        )
        table_path = self.root / "selection_table.csv"
        selection_table.to_csv(table_path, index=False)
        selection_summary = self.root / "selection_summary.json"
        selection_summary.write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
        frozen = FrozenSelection(
            selection_table_path=table_path.resolve(),
            selection_table_sha256=sha256_file(table_path),
            selection_summary_path=selection_summary.resolve(),
            selector_matrix_path=matrix_path.resolve(),
            selector_matrix_sha256=sha256_file(matrix_path),
            table=selection_table,
            unique_checkpoints=tuple(),
            pool_checkpoints=tuple(
                SelectedCheckpoint(
                    candidate_id=candidate_id,
                    checkpoint_path=Path(path).resolve(),
                    checkpoint_sha256=sha256_file(Path(path)),
                    selectors=(),
                )
                for candidate_id, path in zip(candidate_ids, checkpoints)
            ),
        )

        pool_rows = []
        test_worst = [0.1, 0.2, 0.3, 0.4]
        for index, candidate_id in enumerate(candidate_ids):
            row = {
                key: values[index] for key, values in identity.items()
            }
            row.update(
                {
                    "test_loss": 0.5,
                    "test_accuracy": test_worst[index],
                    "test_balanced_group_accuracy": test_worst[index],
                    "test_worst_group_accuracy": test_worst[index],
                    "summary_path": str(self.root / f"{candidate_id}_summary.json"),
                }
            )
            correct_per_group = int(round(test_worst[index] * 10))
            raw_rows = []
            for sample in self.source.dataset.frame.itertuples(index=False):
                within_group = int(str(sample.sample_id).rsplit("_", 1)[1])
                prediction = int(sample.label) if within_group < correct_per_group else 1 - int(sample.label)
                probabilities = [0.2, 0.2]
                probabilities[prediction] = 0.8
                raw_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "sample_id": sample.sample_id,
                        "label": int(sample.label),
                        "group": int(sample.group),
                        "prediction": prediction,
                        "correct": int(prediction == int(sample.label)),
                        "true_class_probability": probabilities[int(sample.label)],
                        "loss": -float(np.log(probabilities[int(sample.label)])),
                        "logits": json.dumps(
                            [float(np.log(value)) for value in probabilities],
                            separators=(",", ":"),
                        ),
                        "probabilities": json.dumps(probabilities, separators=(",", ":")),
                    }
                )
            per_image_path = self.root / f"{candidate_id}_pool_per_image.csv"
            raw_frame = pd.DataFrame(raw_rows)
            raw_frame.to_csv(per_image_path, index=False)
            metrics = recompute_test_metrics_from_frame(
                raw_frame, self.source, candidate_id=candidate_id
            )
            row["test_loss"] = metrics["loss"]
            row["per_image_path"] = str(per_image_path.resolve())
            row["per_image_sha256"] = sha256_file(per_image_path)
            for group in range(4):
                row[f"test_group_{group}_accuracy"] = test_worst[index]
                row[f"test_group_{group}_count"] = 10
            pool_rows.append(row)
        pool_csv = self.root / "candidate_pool_test_scores.csv"
        pd.DataFrame(pool_rows, columns=POOL_TEST_COLUMNS).to_csv(pool_csv, index=False)
        pool_summary = self.root / "candidate_pool_test_scores_summary.json"
        pool_summary.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "artifact_type": "fcv_vit_posthoc_pool_test_index_summary",
                    "status": "complete",
                    "candidate_count": 4,
                    "expected_candidate_count": 4,
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
        return frozen, pool_csv, pool_summary

    def _analyze(self, *, create_plots: bool = True):
        output_results = self.root / "rank_correlation_results.csv"
        output_candidates = self.root / "candidate_rank_analysis.csv"
        output_summary = self.root / "rank_correlation_results_summary.json"
        plot_dir = self.root / "selector_scatter_plots"
        result = analyze_rank_quality(
            self.config,
            self.frozen,
            self.source,
            self.pool_csv,
            self.pool_summary,
            output_results,
            output_candidates,
            output_summary,
            plot_dir,
            create_plots=create_plots,
        )
        return result, pd.read_csv(output_results), pd.read_csv(output_candidates)

    def test_correlations_regret_top_k_and_plots(self) -> None:
        summary, results, candidates = self._analyze()
        by_name = results.set_index("selector_name")
        biased = by_name.loc["biased_validation_accuracy"]
        fcv = by_name.loc["equal_weight_original_and_opposite_fcv_accuracy"]
        loss = by_name.loc["biased_validation_loss"]
        self.assertAlmostEqual(biased["spearman_rho"], -1.0)
        self.assertAlmostEqual(fcv["spearman_rho"], 1.0)
        self.assertAlmostEqual(loss["spearman_rho"], 1.0)
        self.assertAlmostEqual(biased["selection_regret_to_pool_best"], 0.3)
        self.assertAlmostEqual(fcv["selection_regret_to_pool_best"], 0.0)
        self.assertAlmostEqual(biased["top_1_recall"], 0.0)
        self.assertAlmostEqual(fcv["top_1_recall"], 1.0)
        self.assertEqual(len(candidates), 4)
        self.assertEqual(summary["pool_best_candidate_id"], "candidate_d")
        self.assertFalse(summary["test_metrics_affected_selection"])
        self.assertEqual(len(summary["scatter_plot_paths"]), 8)
        self.assertTrue(all(Path(path).is_file() for path in summary["scatter_plot_paths"]))
        self.assertTrue(
            any(
                path.endswith("biased_val_vs_test_wga_colored_by_fcv.png")
                for path in summary["scatter_plot_paths"]
            )
        )
        self.assertIn("spearman_cluster_ci_low", results.columns)
        self.assertNotIn("spearman_pvalue", results.columns)
        self.assertFalse(
            summary["correlation_inference"]["naive_epoch_level_pvalues_reported"]
        )

    def test_recomputed_selection_must_match_frozen_step9(self) -> None:
        table = self.frozen.table.copy()
        mask = table["selector_name"] == "biased_validation_accuracy"
        table.loc[mask, "selected_checkpoint_id"] = "candidate_b"
        self.frozen.table = table
        with self.assertRaisesRegex(RankAnalysisError, "Recomputed selection"):
            self._analyze(create_plots=False)


if __name__ == "__main__":
    unittest.main()
