from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
from PIL import Image


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.gap_analysis import validate_final_test_results  # noqa: E402
from fcv.candidate_training import (  # noqa: E402
    candidate_training_fingerprint,
    enumerate_sweep_runs,
)
from fcv.selectors import selector_analysis_fingerprint  # noqa: E402
from fcv.test_evaluation import (  # noqa: E402
    FinalTestError,
    TestManifestDataset,
    assemble_final_test_results,
    final_test_evaluation_fingerprint,
    load_frozen_selection,
    prepare_final_test_source,
    recompute_test_metrics_from_frame,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class FinalTestEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        config_path = (
            EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["paths"]["output_root"] = str(self.root / "outputs")
        self.config_path = self.root / "config.yaml"
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle)
        self.config = load_and_validate_config(self.config_path)
        self.config["_synthetic_test_mode"] = True
        self.config["training"]["learning_rates"] = [1.0e-5]
        self.config["training"]["weight_decays"] = [0.01]
        self.config["training"]["seeds"] = [0]
        self.config["training"]["epochs"] = 3
        self.config["candidate_pool"]["expected_training_runs"] = 1
        self.config["candidate_pool"]["expected_candidate_checkpoints"] = 3

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_frozen_artifacts(self) -> tuple[Path, Path, list[Path]]:
        run = enumerate_sweep_runs(self.config)[0]
        candidate_ids = [run.candidate_id(epoch) for epoch in (1, 2, 3)]
        checkpoints = [
            self.root / "epoch_001.pt",
            self.root / "epoch_002.pt",
            self.root / "epoch_003.pt",
        ]
        for checkpoint in checkpoints:
            checkpoint.touch()
        matrix_path = self.root / "candidate_selector_scores.csv"
        pd.DataFrame(
            {
                "candidate_id": candidate_ids,
                "checkpoint_path": [str(path) for path in checkpoints],
                "checkpoint_sha256": [sha256_file(path) for path in checkpoints],
            }
        ).to_csv(matrix_path, index=False)
        table_path = self.root / "selection_table.csv"
        table = pd.DataFrame(
            {
                "selector_name": [
                    "biased_validation_accuracy",
                    "equal_weight_original_and_opposite_fcv_accuracy",
                    "oracle_validation_balanced_group_accuracy",
                ],
                "selector_family": ["biased_validation", "fcv_primary", "oracle"],
                "availability": [
                    "unprivileged_train_holdout",
                    "unprivileged_train_holdout",
                    "privileged_analysis_only",
                ],
                "direction": ["maximize", "maximize", "maximize"],
                "selector_formula": ["biased", "fcv", "oracle"],
                "selector_score": [0.9, 0.8, 0.7],
                "selected_checkpoint_id": [
                    candidate_ids[0],
                    candidate_ids[1],
                    candidate_ids[1],
                ],
                "selected_checkpoint_path": [
                    str(checkpoints[0]),
                    str(checkpoints[1]),
                    str(checkpoints[1]),
                ],
                "selected_checkpoint_sha256": [
                    sha256_file(checkpoints[0]),
                    sha256_file(checkpoints[1]),
                    sha256_file(checkpoints[1]),
                ],
                "selected_hparams": ["{}", "{}", "{}"],
            }
        )
        table.to_csv(table_path, index=False)
        summary_path = self.root / "selection_table_summary.json"
        summary = {
            "schema_version": 1,
            "artifact_type": "fcv_vit_selection_table_summary",
            "status": "complete",
            "selector_count": 3,
            "selector_analysis_fingerprint": selector_analysis_fingerprint(self.config),
            "selection_table_path": str(table_path.resolve()),
            "selection_table_sha256": sha256_file(table_path),
            "candidate_selector_matrix_path": str(matrix_path.resolve()),
            "candidate_selector_matrix_sha256": sha256_file(matrix_path),
            "selected_candidates": dict(
                zip(table["selector_name"], table["selected_checkpoint_id"])
            ),
            "test_data_accessed": False,
            "test_metrics_deferred_to_step": 10,
        }
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return table_path, summary_path, checkpoints

    def test_test_manifest_requires_evaluation_only_split(self) -> None:
        rows = []
        for group, (label, context) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            image_path = self.root / f"test_{group}.jpg"
            Image.fromarray(np.full((8, 8, 3), group * 50, dtype=np.uint8)).save(
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
                    "source_split": "test",
                    "study_split": "test_analysis_only",
                }
            )
        manifest = self.root / "test_manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        dataset = TestManifestDataset(manifest, transform=None)
        self.assertEqual(len(dataset), 4)
        with self.assertRaisesRegex(FinalTestError, "batch size"):
            prepare_final_test_source(
                self.config, manifest, batch_size=1, check_images=False
            )
        with self.assertRaisesRegex(FinalTestError, "worker count"):
            prepare_final_test_source(
                self.config, manifest, num_workers=0, check_images=False
            )

        rows[0]["study_split"] = "oracle_validation_analysis_only"
        wrong = self.root / "oracle_manifest.csv"
        pd.DataFrame(rows).to_csv(wrong, index=False)
        with self.assertRaisesRegex(FinalTestError, "study_split"):
            TestManifestDataset(wrong, transform=None)

    def test_selection_is_deduplicated_and_rejects_prefilled_test_metrics(self) -> None:
        table_path, summary_path, _ = self._make_frozen_artifacts()
        frozen = load_frozen_selection(self.config, table_path, summary_path)
        self.assertEqual(len(frozen.table), 3)
        self.assertEqual(
            [item.candidate_id for item in frozen.unique_checkpoints],
            [enumerate_sweep_runs(self.config)[0].candidate_id(epoch) for epoch in (1, 2)],
        )
        self.assertEqual(len(frozen.unique_checkpoints[1].selectors), 2)
        self.assertEqual(len(frozen.pool_checkpoints), 3)

        table = pd.read_csv(table_path)
        table["test_worst_group_accuracy"] = [0.1, 0.9, 0.9]
        table.to_csv(table_path, index=False)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["selection_table_sha256"] = sha256_file(table_path)
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(FinalTestError, "before test metrics"):
            load_frozen_selection(self.config, table_path, summary_path)

    def test_frozen_selection_rejects_checkpoint_byte_changes(self) -> None:
        table_path, summary_path, checkpoints = self._make_frozen_artifacts()
        checkpoints[0].write_bytes(b"changed after selection")
        with self.assertRaisesRegex(FinalTestError, "changed|hash|differ"):
            load_frozen_selection(self.config, table_path, summary_path)

    def test_frozen_selection_rejects_incomplete_candidate_matrix(self) -> None:
        table_path, summary_path, _ = self._make_frozen_artifacts()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        matrix_path = Path(summary["candidate_selector_matrix_path"])
        matrix = pd.read_csv(matrix_path).iloc[:2].copy()
        matrix.to_csv(matrix_path, index=False)
        summary["candidate_selector_matrix_sha256"] = sha256_file(matrix_path)
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(FinalTestError, "complete locked candidate pool"):
            load_frozen_selection(self.config, table_path, summary_path)

    def test_final_results_preserve_selector_order_and_duplicate_metrics(self) -> None:
        table_path, summary_path, _ = self._make_frozen_artifacts()
        frozen = load_frozen_selection(self.config, table_path, summary_path)
        test_manifest = self.root / "synthetic_test_manifest.csv"
        test_manifest.write_text("synthetic\n", encoding="utf-8")
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
        source = SimpleNamespace(
            manifest_path=test_manifest.resolve(),
            manifest_sha256=sha256_file(test_manifest),
            manifest_bundle_path=self.root / "SYNTHETIC_TEST_MANIFEST_BUNDLE",
            manifest_bundle_sha256="SYNTHETIC_TEST_MODE",
            dataset=SimpleNamespace(frame=manifest_frame),
            sample_count=40,
            batch_size=128,
            num_workers=8,
        )
        candidate_summaries = {}
        for index, selected in enumerate(frozen.unique_checkpoints):
            summary_file = self.root / f"{selected.candidate_id}_test_summary.json"
            group_correct = ([4, 6, 8, 10], [6, 7, 8, 9])[index]
            per_image_rows = []
            for row in manifest_frame.itertuples(index=False):
                within_group = int(str(row.sample_id).rsplit("_", 1)[1])
                prediction = int(row.label) if within_group < group_correct[int(row.group)] else 1 - int(row.label)
                probabilities = [0.2, 0.2]
                probabilities[prediction] = 0.8
                logits = np.log(np.asarray(probabilities, dtype=np.float64))
                per_image_rows.append(
                    {
                        "candidate_id": selected.candidate_id,
                        "sample_id": row.sample_id,
                        "label": int(row.label),
                        "group": int(row.group),
                        "prediction": prediction,
                        "correct": int(prediction == int(row.label)),
                        "true_class_probability": probabilities[int(row.label)],
                        "loss": -float(np.log(probabilities[int(row.label)])),
                        "logits": json.dumps(logits.tolist(), separators=(",", ":")),
                        "probabilities": json.dumps(probabilities, separators=(",", ":")),
                    }
                )
            per_image = pd.DataFrame(per_image_rows)
            per_image_path = self.root / f"{selected.candidate_id}_test_per_image.csv"
            per_image.to_csv(per_image_path, index=False)
            metrics = recompute_test_metrics_from_frame(
                per_image, source, candidate_id=selected.candidate_id
            )
            persisted = {
                "schema_version": 2,
                "artifact_type": "fcv_vit_selected_test_summary",
                "status": "complete",
                "candidate_id": selected.candidate_id,
                "checkpoint_path": str(selected.checkpoint_path),
                "checkpoint_sha256": selected.checkpoint_sha256,
                "training_fingerprint": candidate_training_fingerprint(self.config),
                "final_test_evaluation_fingerprint": final_test_evaluation_fingerprint(
                    self.config
                ),
                "test_manifest_path": str(source.manifest_path),
                "test_manifest_sha256": source.manifest_sha256,
                "manifest_bundle_path": str(source.manifest_bundle_path),
                "manifest_bundle_sha256": source.manifest_bundle_sha256,
                "test_sample_count": source.sample_count,
                "per_image_csv_path": str(per_image_path.resolve()),
                "per_image_csv_sha256": sha256_file(per_image_path),
                "precision": "float32",
                "execution": {"batch_size": 128, "num_workers": 8},
                "metrics": metrics,
                "test_data_accessed": True,
                "selection_was_frozen_before_evaluation": True,
            }
            summary_file.write_text(json.dumps(persisted), encoding="utf-8")
            candidate_summaries[selected.candidate_id] = {
                **persisted,
                "summary_path": str(summary_file),
            }
        output_csv = self.root / "final_test_results.csv"
        output_summary = self.root / "final_test_results_summary.json"
        result = assemble_final_test_results(
            self.config,
            frozen,
            source,
            candidate_summaries,
            output_csv,
            output_summary,
        )
        final = pd.read_csv(output_csv)
        self.assertEqual(
            final["selector_name"].tolist(), frozen.table["selector_name"].tolist()
        )
        self.assertEqual(final["test_worst_group_accuracy"].tolist(), [0.4, 0.6, 0.6])
        self.assertEqual(result["unique_selected_checkpoint_count"], 2)
        self.assertTrue(result["selection_frozen_before_test"])
        self.assertFalse(result["test_metrics_affected_selection"])

        # Exercise the real producer-to-consumer boundary.  This guards the
        # Step-10 aggregate schema expected by Step 11 instead of relying on a
        # separately hand-written gap-analysis fixture.
        validated = validate_final_test_results(
            self.config,
            frozen,
            source,
            output_csv,
            output_summary,
        )
        self.assertEqual(
            validated["selector_name"].tolist(),
            frozen.table["selector_name"].tolist(),
        )
        self.assertEqual(
            validated["test_worst_group_accuracy"].tolist(), [0.4, 0.6, 0.6]
        )

        first_id = frozen.unique_checkpoints[0].candidate_id
        first_path = Path(candidate_summaries[first_id]["per_image_csv_path"])
        tampered = pd.read_csv(first_path)
        tampered.loc[0, "prediction"] = 1 - int(tampered.loc[0, "prediction"])
        tampered.to_csv(first_path, index=False)
        candidate_summaries[first_id]["per_image_csv_sha256"] = sha256_file(first_path)
        with self.assertRaisesRegex(FinalTestError, "reproduce|stale|correctness"):
            assemble_final_test_results(
                self.config,
                frozen,
                source,
                candidate_summaries,
                self.root / "tampered.csv",
                self.root / "tampered.json",
            )


if __name__ == "__main__":
    unittest.main()
