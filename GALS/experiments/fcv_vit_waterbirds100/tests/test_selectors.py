from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.candidate_training import (  # noqa: E402
    candidate_training_fingerprint,
    enumerate_sweep_runs,
)
from fcv.selectors import (  # noqa: E402
    ORACLE_METRIC_COLUMNS,
    OracleManifestDataset,
    SelectorError,
    compute_waterbirds_group_metrics,
    aggregate_oracle_summaries,
    build_selection_table,
    prepare_oracle_validation_source,
    recompute_oracle_metrics_from_frame,
    selector_analysis_fingerprint,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        config_path = (
            EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["paths"]["output_root"] = str(self.root / "outputs")
        self.config_path = self.root / "config.yaml"
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(raw, handle)
        self.config = load_and_validate_config(self.config_path)
        self.config["_synthetic_test_mode"] = True

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_oracle_manifest_is_analysis_only_and_has_all_groups(self) -> None:
        rows = []
        for group, (label, context) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            image_path = self.root / f"group_{group}.jpg"
            Image.fromarray(np.full((8, 8, 3), 50 + group * 40, dtype=np.uint8)).save(
                image_path
            )
            rows.append(
                {
                    "sample_id": f"sample_{group}",
                    "metadata_index": group,
                    "image_path": str(image_path),
                    "image_sha256": sha256_file(image_path),
                    "label": label,
                    "context": context,
                    "group": group,
                    "group_name": (
                        "Land_on_Land",
                        "Land_on_Water",
                        "Water_on_Land",
                        "Water_on_Water",
                    )[group],
                    "source_split": "original_validation",
                    "study_split": "oracle_validation_analysis_only",
                }
            )
        path = self.root / "oracle.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        dataset = OracleManifestDataset(path, transform=None)
        self.assertEqual(len(dataset), 4)
        self.assertEqual({dataset[index][2] for index in range(4)}, {0, 1, 2, 3})
        with self.assertRaisesRegex(SelectorError, "batch size"):
            prepare_oracle_validation_source(
                self.config, path, batch_size=1, check_images=False
            )
        with self.assertRaisesRegex(SelectorError, "worker count"):
            prepare_oracle_validation_source(
                self.config, path, num_workers=0, check_images=False
            )

        rows[0]["study_split"] = "test_analysis_only"
        leaked = self.root / "test.csv"
        pd.DataFrame(rows).to_csv(leaked, index=False)
        with self.assertRaisesRegex(SelectorError, "Oracle evaluator requires"):
            OracleManifestDataset(leaked, transform=None)

    def test_group_metric_computation(self) -> None:
        result = compute_waterbirds_group_metrics(
            loss_sum=4.0,
            correct=7,
            total=8,
            group_correct=[2, 1, 2, 2],
            group_total=[2, 2, 2, 2],
        )
        self.assertAlmostEqual(result["accuracy"], 0.875)
        self.assertAlmostEqual(result["balanced_group_accuracy"], 0.875)
        self.assertAlmostEqual(result["worst_group_accuracy"], 0.5)

    def test_oracle_aggregation_requires_complete_provenance(self) -> None:
        config = deepcopy(self.config)
        config["training"]["learning_rates"] = [1.0e-5]
        config["training"]["weight_decays"] = [0.01]
        config["training"]["seeds"] = [0]
        config["training"]["epochs"] = 2
        config["candidate_pool"]["candidate_epochs"] = [1, 2]
        config["candidate_pool"]["expected_training_runs"] = 1
        config["candidate_pool"]["expected_candidate_checkpoints"] = 2
        rows = []
        for group, (label, context) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            rows.append(
                {
                    "sample_id": f"oracle_{group}",
                    "metadata_index": group,
                    "image_path": str(self.root / f"unused_{group}.jpg"),
                    "image_sha256": "unused",
                    "label": label,
                    "context": context,
                    "group": group,
                    "group_name": (
                        "Land_on_Land",
                        "Land_on_Water",
                        "Water_on_Land",
                        "Water_on_Water",
                    )[group],
                    "source_split": "original_validation",
                    "study_split": "oracle_validation_analysis_only",
                }
            )
        manifest = self.root / "oracle_aggregate.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        source = prepare_oracle_validation_source(
            config, manifest, check_images=False
        )
        oracle_dir = self.root / "oracle_scores"
        oracle_dir.mkdir()
        run = enumerate_sweep_runs(config)[0]
        for epoch in (1, 2):
            candidate_id = run.candidate_id(epoch)
            checkpoint = self.root / f"{candidate_id}.pt"
            checkpoint.touch()
            raw_rows = []
            for group, label in enumerate((0, 0, 1, 1)):
                prediction = label if group < 3 else 0
                probabilities = (
                    [0.8, 0.2] if prediction == 0 else [0.2, 0.8]
                )
                logits = np.log(np.asarray(probabilities, dtype=np.float64))
                raw_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "sample_id": f"oracle_{group}",
                        "label": label,
                        "group": group,
                        "prediction": prediction,
                        "correct": int(prediction == label),
                        "true_class_probability": probabilities[label],
                        "loss": -float(np.log(probabilities[label])),
                        "logits": json.dumps(logits.tolist(), separators=(",", ":")),
                        "probabilities": json.dumps(
                            probabilities, separators=(",", ":")
                        ),
                    }
                )
            per_image = pd.DataFrame(raw_rows)
            per_image_path = oracle_dir / f"{candidate_id}_oracle_per_image.csv"
            per_image.to_csv(per_image_path, index=False)
            metrics = recompute_oracle_metrics_from_frame(
                per_image, source, candidate_id=candidate_id
            )
            summary = {
                "schema_version": 2,
                "artifact_type": "fcv_vit_oracle_validation_summary",
                "status": "complete",
                "candidate_id": candidate_id,
                "run": {
                    "run_index": run.run_index,
                    "learning_rate": run.learning_rate,
                    "weight_decay": run.weight_decay,
                    "seed": run.seed,
                },
                "epoch": epoch,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "training_fingerprint": candidate_training_fingerprint(config),
                "selector_analysis_fingerprint": selector_analysis_fingerprint(config),
                "oracle_manifest_path": str(source.manifest_path),
                "oracle_manifest_sha256": source.manifest_sha256,
                "manifest_bundle_path": str(source.manifest_bundle_path),
                "manifest_bundle_sha256": source.manifest_bundle_sha256,
                "oracle_sample_count": 4,
                "precision": "float32",
                "execution": {
                    "batch_size": source.batch_size,
                    "num_workers": source.num_workers,
                },
                "per_image_csv_path": str(per_image_path.resolve()),
                "per_image_csv_sha256": sha256_file(per_image_path),
                "metrics": metrics,
                "test_data_accessed": False,
            }
            with (
                oracle_dir / f"{candidate_id}_oracle_summary.json"
            ).open("w", encoding="utf-8") as handle:
                json.dump(summary, handle)
        output_csv = self.root / "oracle_index.csv"
        output_summary = self.root / "oracle_index_summary.json"
        result = aggregate_oracle_summaries(
            config,
            oracle_dir,
            output_csv,
            output_summary,
            source=source,
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(len(pd.read_csv(output_csv)), 2)
        self.assertTrue(result["oracle_per_image_metrics_recomputed"])

        first_summary = oracle_dir / f"{run.candidate_id(1)}_oracle_summary.json"
        tampered = json.loads(first_summary.read_text(encoding="utf-8"))
        tampered["metrics"]["balanced_group_accuracy"] += 0.1
        first_summary.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(SelectorError, "incomplete"):
            aggregate_oracle_summaries(
                config,
                oracle_dir,
                self.root / "oracle_tampered.csv",
                self.root / "oracle_tampered_summary.json",
                source=source,
            )

    def test_selector_table_uses_only_validation_metrics_and_fixed_ties(self) -> None:
        config = deepcopy(self.config)
        config["candidate_pool"]["expected_candidate_checkpoints"] = 3
        candidate_ids = ["candidate_a", "candidate_b", "candidate_c"]
        checkpoints = []
        for candidate_id in candidate_ids:
            path = self.root / f"{candidate_id}.pt"
            path.touch()
            checkpoints.append(str(path))

        common = {
            "candidate_id": candidate_ids,
            "run_index": [0, 1, 2],
            "epoch": [1, 1, 1],
            "seed": [0, 1, 2],
            "learning_rate": [1.0e-5, 3.0e-5, 1.0e-4],
            "weight_decay": [0.01, 0.05, 0.10],
            "checkpoint_path": checkpoints,
            "checkpoint_sha256": [
                sha256_file(Path(path)) for path in checkpoints
            ],
        }
        candidate = pd.DataFrame(
            {
                **common,
                "biased_val_loss": [0.30, 0.10, 0.10],
                "biased_val_accuracy": [0.90, 0.80, 0.80],
            }
        )
        original = np.asarray([0.90, 0.80, 0.80])
        counterfactual = np.asarray([0.20, 0.80, 0.70])
        score_paths = []
        ratios = [(0.8, 0.4), (0.8, 0.6), (0.6, 0.9)]
        for candidate_id, (p_original, p_counterfactual) in zip(candidate_ids, ratios):
            score_path = self.root / f"{candidate_id}_fcv.csv"
            pd.DataFrame(
                {
                    "candidate_id": [candidate_id, candidate_id],
                    "fcv_eligible": [True, True],
                    "p_y_original": [p_original, p_original],
                    "original_cross_entropy": [
                        -np.log(p_original),
                        -np.log(p_original),
                    ],
                    "p_y_counterfactual_mean": [p_counterfactual, p_counterfactual],
                }
            ).to_csv(score_path, index=False)
            score_paths.append(str(score_path))
        fcv = pd.DataFrame(
            {
                **common,
                "biased_validation_accuracy": original,
                "fcv_counterfactual_accuracy": counterfactual,
                "fcv_counterfactual_majority_accuracy": [0.20, 0.80, 0.70],
                "fcv_true_class_probability": [0.30, 0.70, 0.80],
                "fcv_confidence_drop": [0.40, 0.10, 0.20],
                "primary_selector_score": 0.5 * original + 0.5 * counterfactual,
                "score_csv_path": score_paths,
                "score_csv_sha256": [
                    sha256_file(Path(path)) for path in score_paths
                ],
            }
        )
        controls = pd.DataFrame(
            {
                **common,
                "same_context_counterfactual_accuracy": [0.80, 0.85, 0.82],
                "same_context_mean_confidence_drop": [0.10, 0.05, 0.10],
            }
        )
        oracle_rows = []
        for index, candidate_id in enumerate(candidate_ids):
            row = {
                **{key: value[index] for key, value in common.items()},
                "checkpoint_path": checkpoints[index],
                "oracle_validation_loss": [0.4, 0.3, 0.2][index],
                "oracle_validation_accuracy": [0.6, 0.8, 0.85][index],
                "oracle_validation_balanced_group_accuracy": [0.5, 0.7, 0.8][index],
                "oracle_validation_worst_group_accuracy": [0.4, 0.7, 0.6][index],
                "summary_path": str(self.root / f"{candidate_id}_oracle.json"),
            }
            for group in range(4):
                row[f"oracle_group_{group}_accuracy"] = 0.4 + 0.1 * index
                row[f"oracle_group_{group}_count"] = 10
            oracle_rows.append(row)
        oracle = pd.DataFrame(oracle_rows, columns=ORACLE_METRIC_COLUMNS)

        paths = {}
        for name, frame in (
            ("candidate", candidate),
            ("fcv", fcv),
            ("controls", controls),
            ("oracle", oracle),
        ):
            path = self.root / f"{name}.csv"
            frame.to_csv(path, index=False)
            paths[name] = path
        selection_path = self.root / "selection_table.csv"
        matrix_path = self.root / "candidate_selector_scores.csv"
        summary_path = self.root / "selection_table_summary.json"
        result = build_selection_table(
            config,
            paths["candidate"],
            paths["fcv"],
            paths["controls"],
            paths["oracle"],
            selection_path,
            matrix_path,
            summary_path,
        )
        selected = pd.read_csv(selection_path).set_index("selector_name")
        self.assertEqual(
            selected.loc["biased_validation_accuracy", "selected_checkpoint_id"],
            "candidate_a",
        )
        self.assertEqual(
            selected.loc["biased_validation_loss", "selected_checkpoint_id"],
            "candidate_b",
        )
        self.assertEqual(
            int(selected.loc["biased_validation_loss", "exact_tie_count"]), 2
        )
        self.assertEqual(
            selected.loc[
                "opposite_context_probability_retention_ratio",
                "selected_checkpoint_id",
            ],
            "candidate_c",
        )
        self.assertEqual(
            selected.loc["control_normalized_fcv", "selected_checkpoint_id"],
            "candidate_b",
        )
        self.assertEqual(
            selected.loc[
                "oracle_validation_worst_group_accuracy", "selected_checkpoint_id"
            ],
            "candidate_b",
        )
        self.assertEqual(
            selected.loc[
                "oracle_validation_balanced_group_accuracy", "selected_checkpoint_id"
            ],
            "candidate_c",
        )
        self.assertNotIn("test_avg_acc", selected.columns)
        self.assertFalse(result["test_data_accessed"])
        with summary_path.open("r", encoding="utf-8") as handle:
            saved_summary = json.load(handle)
        self.assertEqual(saved_summary["test_metrics_deferred_to_step"], 10)
        self.assertTrue(
            saved_summary["unprivileged_selection_frozen_before_oracle_join"]
        )
        self.assertTrue(
            Path(saved_summary["unprivileged_selection_table_path"]).is_file()
        )

        # Even a malformed privileged input is not read until the unprivileged
        # matrix and selections have been materialized and hashed.
        malformed_oracle = self.root / "malformed_oracle.csv"
        pd.DataFrame({"candidate_id": candidate_ids}).to_csv(
            malformed_oracle, index=False
        )
        frozen_table = self.root / "failed_join_selection.csv"
        frozen_matrix = self.root / "failed_join_matrix.csv"
        with self.assertRaisesRegex(SelectorError, "Oracle scores"):
            build_selection_table(
                config,
                paths["candidate"],
                paths["fcv"],
                paths["controls"],
                malformed_oracle,
                frozen_table,
                frozen_matrix,
                self.root / "failed_join_summary.json",
            )
        self.assertTrue(
            (self.root / "unprivileged_selections_frozen.csv").is_file()
        )
        self.assertTrue(
            (self.root / "unprivileged_candidate_matrix.csv").is_file()
        )

        # Step 9 must bind the selector's per-image probability input to the
        # hash recorded by Step 7, not merely trust its path.
        tampered_score = Path(score_paths[0])
        tampered_score.write_bytes(tampered_score.read_bytes() + b"\n")
        with self.assertRaisesRegex(SelectorError, "bytes changed"):
            build_selection_table(
                config,
                paths["candidate"],
                paths["fcv"],
                paths["controls"],
                paths["oracle"],
                self.root / "tampered_selection.csv",
                self.root / "tampered_matrix.csv",
                self.root / "tampered_summary.json",
            )


if __name__ == "__main__":
    unittest.main()
