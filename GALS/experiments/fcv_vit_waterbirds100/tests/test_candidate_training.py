from __future__ import annotations

import sys
import tempfile
import unittest
import json
import hashlib
from copy import deepcopy
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

import fcv.candidate_training as candidate_training  # noqa: E402
from fcv.candidate_training import (  # noqa: E402
    METRIC_COLUMNS,
    CandidateTrainingError,
    PublicManifestDataset,
    aggregate_candidate_metrics,
    enumerate_sweep_runs,
    train_candidate_run,
    warmup_cosine_factor,
)
from fcv.config import ConfigError, load_and_validate_config  # noqa: E402


class CandidateTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        base_config_path = (
            EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
        )
        with base_config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["paths"]["output_root"] = str(self.root / "outputs")
        self.config_path = self.root / "config.yaml"
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle)
        self.config = load_and_validate_config(self.config_path)
        self.config["_synthetic_test_mode"] = True

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_sweep_order_and_scheduler(self) -> None:
        runs = enumerate_sweep_runs(self.config)
        self.assertEqual(len(runs), 27)
        self.assertEqual(
            (runs[0].learning_rate, runs[0].weight_decay, runs[0].seed),
            (1.0e-5, 0.01, 0),
        )
        self.assertEqual(runs[3].weight_decay, 0.05)
        self.assertEqual(runs[9].learning_rate, 3.0e-5)
        self.assertEqual(runs[-1].candidate_id(20).split("_")[-1], "020")

        self.assertAlmostEqual(
            warmup_cosine_factor(
                0, total_steps=200, warmup_steps=20, minimum_ratio=0.0
            ),
            0.05,
        )
        self.assertAlmostEqual(
            warmup_cosine_factor(
                19, total_steps=200, warmup_steps=20, minimum_ratio=0.0
            ),
            1.0,
        )
        self.assertAlmostEqual(
            warmup_cosine_factor(
                200, total_steps=200, warmup_steps=20, minimum_ratio=0.0
            ),
            0.0,
        )

    def test_production_config_locks_holdout_model_geometry_and_timm(self) -> None:
        cases = {
            "holdout": lambda cfg: cfg["data"]["biased_train_holdout"].update(
                {"train_fraction": 0.75, "validation_fraction": 0.25}
            ),
            "model": lambda cfg: cfg["model"].update({"name": "vit_tiny_patch16_224"}),
            "geometry": lambda cfg: cfg["model"].update(
                {"image_size": 192, "patch_grid_size": 12}
            ),
            "timm": lambda cfg: cfg["cluster"].update({"timm_version": "1.0.27"}),
            "batch_size": lambda cfg: cfg["training"].update({"batch_size": 64}),
            "workers": lambda cfg: cfg["training"].update({"num_workers": 4}),
            "warmup": lambda cfg: cfg["training"]["scheduler"].update(
                {"warmup_epochs": 3}
            ),
            "eval_resize": lambda cfg: cfg["training"]["augmentation"].update(
                {"eval_resize_size": 288}
            ),
            "horizontal_flip": lambda cfg: cfg["training"]["augmentation"].update(
                {"train_horizontal_flip_probability": 0.25}
            ),
            "crop_scale": lambda cfg: cfg["training"]["augmentation"]
            ["train_random_resized_crop"].update({"scale": [0.7, 1.0]}),
            "evidence_threshold": lambda cfg: cfg["fcv"].update(
                {"evidence_patch_threshold": 0.55}
            ),
            "background_threshold": lambda cfg: cfg["fcv"].update(
                {"background_patch_threshold": 0.05}
            ),
            "ambiguous_policy": lambda cfg: cfg["fcv"].update(
                {"ambiguous_patch_policy": "drop"}
            ),
            "donor_count": lambda cfg: cfg["fcv"].update(
                {"donor_samples_per_image": 4}
            ),
            "minimum_background": lambda cfg: cfg["fcv"].update(
                {"minimum_background_patches": 19}
            ),
            "donor_contract": lambda cfg: cfg["fcv"]["donor_bank"].update(
                {"sampling": "local_uniform"}
            ),
            "donor_seed": lambda cfg: cfg["reproducibility"].update(
                {"donor_sampling_seed": 2}
            ),
            "control_seed": lambda cfg: cfg["reproducibility"].update(
                {"control_sampling_seed": 2}
            ),
            "oracle_batch": lambda cfg: cfg["fcv"]["selector_analysis"].update(
                {"oracle_batch_size": 64}
            ),
            "final_batch": lambda cfg: cfg["evaluation"]["final_test"].update(
                {"batch_size": 64}
            ),
            "token_batch": lambda cfg: cfg["execution"].update(
                {"token_bank_batch_size": 64}
            ),
            "fcv_forward_batch": lambda cfg: cfg["execution"].update(
                {"fcv_counterfactual_forward_batch_size": 128}
            ),
            "control_forward_batch": lambda cfg: cfg["execution"].update(
                {"control_counterfactual_forward_batch_size": 128}
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                with self.config_path.open("r", encoding="utf-8") as handle:
                    raw = yaml.safe_load(handle)
                mutate(raw)
                path = self.root / f"invalid_{name}.yaml"
                with path.open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(raw, handle)
                with self.assertRaises(ConfigError):
                    load_and_validate_config(path)

    def test_torchvision_runtime_version_uses_module_build_tag(self) -> None:
        package_versions = {
            "timm": "1.0.28",
            "pandas": "2.3.3",
            "Pillow": "11.1.0",
        }
        with mock.patch.object(
            candidate_training.torchvision,
            "__version__",
            "0.26.0+cu130",
        ), mock.patch.object(
            candidate_training,
            "version",
            side_effect=lambda package: package_versions[package],
        ) as distribution_version:
            observed = candidate_training.software_versions()
        self.assertEqual(observed["torchvision"], "0.26.0+cu130")
        self.assertNotIn(
            "torchvision",
            [call.args[0] for call in distribution_version.call_args_list],
        )

    def test_public_dataset_rejects_group_columns(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "sample_id": "a",
                    "metadata_index": 0,
                    "image_path": str(self.root / "a.jpg"),
                    "image_sha256": "unused",
                    "label": 0,
                    "source_split": "train",
                    "study_split": "candidate_train",
                    "group": 0,
                },
                {
                    "sample_id": "b",
                    "metadata_index": 1,
                    "image_path": str(self.root / "b.jpg"),
                    "image_sha256": "unused",
                    "label": 1,
                    "source_split": "train",
                    "study_split": "candidate_train",
                    "group": 3,
                },
            ]
        )
        path = self.root / "leaked.csv"
        manifest.to_csv(path, index=False)
        with self.assertRaisesRegex(CandidateTrainingError, "analysis-only columns"):
            PublicManifestDataset(
                path, "candidate_train", transform=None, check_images=False
            )

    def test_full_candidate_aggregation(self) -> None:
        candidate_root = self.root / "candidate_models"
        pretrained_summary_path = self.root / "pretrained_model_summary.json"
        pretrained_summary_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "fcv_vit_pretrained_initialization",
                    "status": "cached_and_validated",
                    "model": self.config["model"]["name"],
                    "model_config": dict(self.config["model"]),
                    "runtime_versions": candidate_training.software_versions(),
                    "software_fingerprint": candidate_training.software_fingerprint(),
                    "source_tree_sha256": candidate_training.source_tree_provenance()[
                        "source_tree_sha256"
                    ],
                    "pretrained_backbone_sha256": "b" * 64,
                }
            ),
            encoding="utf-8",
        )
        pretrained_summary_sha256 = hashlib.sha256(
            pretrained_summary_path.read_bytes()
        ).hexdigest()
        rows_per_run = int(self.config["training"]["epochs"])
        for run in enumerate_sweep_runs(self.config):
            run_dir = candidate_root / run.run_id
            checkpoints = run_dir / "checkpoints"
            checkpoints.mkdir(parents=True)
            rows = []
            for epoch in range(1, rows_per_run + 1):
                checkpoint = checkpoints / f"epoch_{epoch:03d}.pt"
                row = {
                        "run_index": run.run_index,
                        "candidate_id": run.candidate_id(epoch),
                        "epoch": epoch,
                        "model_name": self.config["model"]["name"],
                        "seed": run.seed,
                        "learning_rate": run.learning_rate,
                        "weight_decay": run.weight_decay,
                        "train_loss": 1.0,
                        "train_accuracy": 0.5,
                        "biased_val_loss": 1.0,
                        "biased_val_accuracy": 0.5,
                        "lr_epoch_start": run.learning_rate,
                        "lr_epoch_end": run.learning_rate,
                        "checkpoint_path": str(checkpoint),
                        # The checkpoint contains this metric row, so its own
                        # content hash is necessarily filled after torch.save.
                        "checkpoint_sha256": "pending",
                        "epoch_seconds": 1.0,
                }
                checkpoint_payload = {
                    "schema_version": 1,
                    "artifact_type": "fcv_vit_candidate_checkpoint",
                    "candidate_id": run.candidate_id(epoch),
                    "run": {
                        "run_index": run.run_index,
                        "learning_rate": run.learning_rate,
                        "weight_decay": run.weight_decay,
                        "seed": run.seed,
                    },
                    "epoch": epoch,
                    "training_fingerprint": candidate_training.candidate_training_fingerprint(
                        self.config
                    ),
                    "software_versions": candidate_training.software_versions(),
                    "source_tree_sha256": candidate_training.source_tree_provenance()[
                        "source_tree_sha256"
                    ],
                    "initial_model_state_sha256": hashlib.sha256(
                        f"seed-{run.seed}".encode()
                    ).hexdigest(),
                    "pretrained_backbone_sha256": "b" * 64,
                    "pretrained_provenance_path": str(
                        pretrained_summary_path.resolve()
                    ),
                    "pretrained_provenance_sha256": pretrained_summary_sha256,
                    "manifest_sha256": {
                        "candidate_train": "a",
                        "biased_validation": "b",
                    },
                    "metrics": dict(row),
                    "model_state_dict": {},
                }
                torch.save(checkpoint_payload, checkpoint)
                row["checkpoint_sha256"] = hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest()
                rows.append(row)
            metrics_path = run_dir / "metrics.csv"
            pd.DataFrame(rows, columns=METRIC_COLUMNS).to_csv(metrics_path, index=False)
            summary = {
                "run": {
                    "run_index": run.run_index,
                    "learning_rate": run.learning_rate,
                    "weight_decay": run.weight_decay,
                    "seed": run.seed,
                },
                "training_fingerprint": candidate_training.candidate_training_fingerprint(
                    self.config
                ),
                "manifest_sha256": {"candidate_train": "a", "biased_validation": "b"},
                "software_versions": candidate_training.software_versions(),
                "software_fingerprint": candidate_training.software_fingerprint(),
                "source_tree_sha256": candidate_training.source_tree_provenance()[
                    "source_tree_sha256"
                ],
                "initial_model_state_sha256": hashlib.sha256(
                    f"seed-{run.seed}".encode()
                ).hexdigest(),
                "pretrained_backbone_sha256": "b" * 64,
                "pretrained_provenance_path": str(
                    pretrained_summary_path.resolve()
                ),
                "pretrained_provenance_sha256": pretrained_summary_sha256,
                "metrics_path": str(metrics_path.resolve()),
                "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            }
            with (run_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle)

        output_csv = candidate_root / "candidate_metrics_biased_val.csv"
        summary_path = candidate_root / "candidate_pool_summary.json"
        result = aggregate_candidate_metrics(
            self.config, candidate_root, output_csv, summary_path
        )
        combined = pd.read_csv(output_csv)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["candidate_count"], 540)
        self.assertEqual(len(combined), 540)
        self.assertFalse(combined["candidate_id"].duplicated().any())

        # Even if a mutable run summary is updated to bind a modified CSV,
        # checkpoint-embedded selector metrics remain authoritative.
        first_run = enumerate_sweep_runs(self.config)[0]
        first_run_dir = candidate_root / first_run.run_id
        first_metrics_path = first_run_dir / "metrics.csv"
        original_metrics_bytes = first_metrics_path.read_bytes()
        tampered_metrics = pd.read_csv(first_metrics_path)
        tampered_metrics.loc[0, "biased_val_loss"] = 0.123456789
        tampered_metrics.to_csv(first_metrics_path, index=False)
        first_summary_path = first_run_dir / "run_summary.json"
        with first_summary_path.open("r", encoding="utf-8") as handle:
            first_summary = json.load(handle)
        first_summary["metrics_sha256"] = hashlib.sha256(
            first_metrics_path.read_bytes()
        ).hexdigest()
        with first_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(first_summary, handle)
        with self.assertRaisesRegex(CandidateTrainingError, "Metric 'biased_val_loss'"):
            aggregate_candidate_metrics(
                self.config,
                candidate_root,
                candidate_root / "metric_tamper.csv",
                candidate_root / "metric_tamper_summary.json",
            )
        first_metrics_path.write_bytes(original_metrics_bytes)
        first_summary["metrics_sha256"] = hashlib.sha256(
            first_metrics_path.read_bytes()
        ).hexdigest()
        with first_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(first_summary, handle)

        with first_summary_path.open("r", encoding="utf-8") as handle:
            first_summary = json.load(handle)
        first_summary["software_versions"]["timm"] = "synthetic-mismatch"
        first_summary["software_fingerprint"] = candidate_training.software_fingerprint(
            first_summary["software_versions"]
        )
        with first_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(first_summary, handle)
        with self.assertRaisesRegex(
            CandidateTrainingError, "different software environments"
        ):
            aggregate_candidate_metrics(
                self.config,
                candidate_root,
                candidate_root / "mismatch.csv",
                candidate_root / "mismatch_summary.json",
            )

    def test_tiny_training_run_writes_and_reuses_epoch_candidates(self) -> None:
        image_root = self.root / "images"
        image_root.mkdir()
        rows = []
        for index, label in enumerate([0, 1, 0, 1, 0, 1, 0, 1]):
            image_path = image_root / f"image_{index}.jpg"
            array = np.full((32, 32, 3), 64 + label * 128, dtype=np.uint8)
            Image.fromarray(array).save(image_path)
            split = "candidate_train" if index < 4 else "biased_validation"
            rows.append(
                {
                    "sample_id": f"sample_{index}",
                    "metadata_index": index,
                    "image_path": str(image_path),
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "label": label,
                    "source_split": "train",
                    "study_split": split,
                }
            )
        train_manifest = self.root / "metadata_train.csv"
        validation_manifest = self.root / "metadata_val.csv"
        pd.DataFrame(rows[:4]).to_csv(train_manifest, index=False)
        pd.DataFrame(rows[4:]).to_csv(validation_manifest, index=False)

        config = deepcopy(self.config)
        config["training"]["epochs"] = 2
        config["training"]["batch_size"] = 2
        config["training"]["num_workers"] = 0
        config["training"]["precision"] = "float32"
        config["training"]["scheduler"]["warmup_epochs"] = 1
        config["model"]["image_size"] = 32
        config["training"]["augmentation"]["eval_resize_size"] = 32
        run = enumerate_sweep_runs(config)[0]
        def tiny_model() -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(3, 2, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

        uninterrupted_root = self.root / "tiny_candidates_uninterrupted"
        with mock.patch(
            "fcv.candidate_training.build_model", side_effect=lambda *args, **kwargs: tiny_model()
        ):
            train_candidate_run(
                config,
                run,
                train_manifest,
                validation_manifest,
                uninterrupted_root,
                device_name="cpu",
            )

        candidate_root = self.root / "tiny_candidates"
        with mock.patch(
            "fcv.candidate_training.build_model", side_effect=lambda *args, **kwargs: tiny_model()
        ), self.assertRaisesRegex(
            CandidateTrainingError, "Simulated abrupt interruption"
        ):
            train_candidate_run(
                config,
                run,
                train_manifest,
                validation_manifest,
                candidate_root,
                device_name="cpu",
                simulate_interruption_after_resume_epoch=1,
            )

        run_dir = candidate_root / run.run_id
        self.assertTrue((run_dir / "resume_state.pt").is_file())
        self.assertFalse((run_dir / "metrics.csv").exists())

        with mock.patch(
            "fcv.candidate_training.build_model", side_effect=lambda *args, **kwargs: tiny_model()
        ):
            result = train_candidate_run(
                config,
                run,
                train_manifest,
                validation_manifest,
                candidate_root,
                device_name="cpu",
            )
        self.assertEqual(result["status"], "complete")
        metrics = pd.read_csv(run_dir / "metrics.csv")
        self.assertEqual(len(metrics), 2)
        self.assertTrue((run_dir / "checkpoints" / "epoch_001.pt").is_file())
        self.assertTrue((run_dir / "checkpoints" / "epoch_002.pt").is_file())

        uninterrupted_dir = uninterrupted_root / run.run_id
        reference_metrics = pd.read_csv(uninterrupted_dir / "metrics.csv")
        metric_columns = [
            "train_loss",
            "train_accuracy",
            "biased_val_loss",
            "biased_val_accuracy",
            "lr_epoch_start",
            "lr_epoch_end",
        ]
        np.testing.assert_allclose(
            metrics[metric_columns].to_numpy(float),
            reference_metrics[metric_columns].to_numpy(float),
            rtol=0.0,
            atol=0.0,
        )
        resumed_checkpoint = torch.load(
            run_dir / "checkpoints" / "epoch_002.pt", map_location="cpu"
        )
        reference_checkpoint = torch.load(
            uninterrupted_dir / "checkpoints" / "epoch_002.pt", map_location="cpu"
        )
        for key, value in reference_checkpoint["model_state_dict"].items():
            self.assertTrue(torch.equal(value, resumed_checkpoint["model_state_dict"][key]))

        second = train_candidate_run(
            config,
            run,
            train_manifest,
            validation_manifest,
            candidate_root,
            device_name="cpu",
        )
        self.assertEqual(second["status"], "already_complete")

        # A completed-run reuse still revalidates the current source-image
        # bytes against the immutable hashes stored in the public manifest.
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(
            image_root / "image_0.jpg"
        )
        with self.assertRaisesRegex(CandidateTrainingError, "Image bytes differ"):
            train_candidate_run(
                config,
                run,
                train_manifest,
                validation_manifest,
                candidate_root,
                device_name="cpu",
            )

    def test_aggregation_rejects_checkpoint_tampering(self) -> None:
        candidate_root = self.root / "tampered_candidates"
        run = enumerate_sweep_runs(self.config)[0]
        run_dir = candidate_root / run.run_id
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        rows = []
        for epoch in range(1, 21):
            path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            path.write_bytes(f"checkpoint-{epoch}".encode())
            rows.append(
                {
                    "run_index": run.run_index,
                    "candidate_id": run.candidate_id(epoch),
                    "epoch": epoch,
                    "model_name": self.config["model"]["name"],
                    "seed": run.seed,
                    "learning_rate": run.learning_rate,
                    "weight_decay": run.weight_decay,
                    "train_loss": 1.0,
                    "train_accuracy": 0.5,
                    "biased_val_loss": 1.0,
                    "biased_val_accuracy": 0.5,
                    "lr_epoch_start": run.learning_rate,
                    "lr_epoch_end": run.learning_rate,
                    "checkpoint_path": str(path),
                    "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "epoch_seconds": 1.0,
                }
            )
        pd.DataFrame(rows, columns=METRIC_COLUMNS).to_csv(run_dir / "metrics.csv", index=False)
        with (run_dir / "run_summary.json").open("w") as handle:
            json.dump(
                {
                    "run": {
                        "run_index": run.run_index,
                        "learning_rate": run.learning_rate,
                        "weight_decay": run.weight_decay,
                        "seed": run.seed,
                    },
                    "training_fingerprint": candidate_training.candidate_training_fingerprint(
                        self.config
                    ),
                    "manifest_sha256": {"candidate_train": "a", "biased_validation": "b"},
                    "software_versions": candidate_training.software_versions(),
                    "software_fingerprint": candidate_training.software_fingerprint(),
                },
                handle,
            )
        (checkpoint_dir / "epoch_001.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(CandidateTrainingError, "bytes changed"):
            aggregate_candidate_metrics(
                self.config,
                candidate_root,
                candidate_root / "out.csv",
                candidate_root / "summary.json",
                allow_incomplete=True,
            )


if __name__ == "__main__":
    unittest.main()
