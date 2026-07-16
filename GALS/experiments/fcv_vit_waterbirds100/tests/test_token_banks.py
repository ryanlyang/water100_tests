from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.candidate_training import (  # noqa: E402
    candidate_training_fingerprint,
    enumerate_sweep_runs,
    get_sweep_run,
    software_versions,
)
from fcv.config import load_and_validate_config  # noqa: E402
from fcv.controls import (  # noqa: E402
    CONTROL_NAMES,
    FCVControlError,
    _canonical_probability_draw_statistics,
    aggregate_control_summaries,
    prepare_control_plan,
    recompute_control_metrics_from_frame,
    score_candidate_controls,
)
from fcv.fcv_scoring import (  # noqa: E402
    FCVScoringError,
    aggregate_fcv_score_summaries,
    load_background_bank,
    make_counterfactual_token_batch,
    prepare_opposite_donor_plan,
    score_candidate_fcv,
)
from fcv.streaming import _cleanup_candidate_banks  # noqa: E402
from fcv.token_banks import (  # noqa: E402
    CONTEXT_NAMES,
    TokenBankError,
    aggregate_token_bank_summaries,
    build_background_token_banks,
    prepare_token_bank_source,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def sha256_json(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


class TinyPatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_patches = 4
        self.patch_size = (2, 2)
        self.projection = nn.Conv2d(3, 8, kernel_size=2, stride=2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images).flatten(2).transpose(1, 2)


class TinyTimmLikeViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 8
        self.num_features = 8
        self.patch_embed = TinyPatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 8))
        self.pos_embed = nn.Parameter(torch.zeros(1, 5, 8))
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.blocks = nn.Identity()
        self.norm = nn.Identity()
        self.head = nn.Linear(8, 2)

    def _pos_embed(self, tokens: torch.Tensor) -> torch.Tensor:
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        return torch.cat((cls, tokens), dim=1) + self.pos_embed

    def forward_head(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens[:, 0])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self._pos_embed(self.patch_embed(images))
        return self.forward_head(self.norm(self.blocks(self.norm_pre(tokens))))


class TokenBanksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        base_config = (
            EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
        )
        with base_config.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["paths"]["output_root"] = str(self.root / "outputs")
        config["model"].update(
            {
                "name": "tiny_timm_like_vit",
                "image_size": 4,
                "patch_size": 2,
                "patch_grid_size": 2,
            }
        )
        config["training"]["batch_size"] = 2
        config["training"]["num_workers"] = 0
        config["execution"].update(
            {
                "token_bank_batch_size": 2,
                "token_bank_num_workers": 0,
                "fcv_counterfactual_forward_batch_size": 8,
                "control_target_batch_size": 2,
                "control_counterfactual_forward_batch_size": 8,
            }
        )
        config["training"]["augmentation"]["eval_resize_size"] = 4
        config["fcv"]["minimum_background_patches"] = 1
        self.config_path = self.root / "config.yaml"
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle)
        self.config = load_and_validate_config(
            self.config_path, strict_protocol=False
        )
        self.config["_synthetic_test_mode"] = True
        self.reconstruction_reports = {
            "pretrained": {"path": "synthetic_pretrained.json", "sha256": "a" * 64},
            "candidate": {"path": "synthetic_candidate.json", "sha256": "b" * 64},
        }

        image_root = self.root / "images"
        image_root.mkdir()
        manifest_rows = []
        records = []
        teacher_map_records = []
        labels = [0, 0, 1, 1]
        for index, label in enumerate(labels):
            sample_id = f"sample_{index}"
            image_path = image_root / f"{sample_id}.jpg"
            pixels = np.full((8, 8, 3), 32 + 50 * index, dtype=np.uint8)
            Image.fromarray(pixels).save(image_path)
            teacher_map_path = self.root / f"{sample_id}.png"
            Image.fromarray(
                np.full((8, 8, 3), 10 + index, dtype=np.uint8)
            ).save(teacher_map_path)
            teacher_map_sha256 = sha256_file(teacher_map_path)
            teacher_map_records.append(
                {"sample_id": sample_id, "sha256": teacher_map_sha256}
            )
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "metadata_index": index,
                    "image_path": str(image_path),
                    "image_sha256": sha256_file(image_path),
                    "image_rel_path": image_path.name,
                    "label": label,
                    "class_name": "Landbird" if label == 0 else "Waterbird",
                    "source_split": "train",
                    "study_split": "biased_validation",
                    "teacher_map_path": str(teacher_map_path),
                    "teacher_map_exists": True,
                }
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "metadata_index": index,
                    "label": label,
                    "teacher_map_path": str(teacher_map_path),
                    "teacher_map_sha256": teacher_map_sha256,
                    "patch_scores": torch.tensor([0.0, 0.5, 0.8, 0.1]),
                    "background_idx": torch.tensor([0, 3]),
                    "evidence_idx": torch.tensor([2]),
                    "fcv_eligible": True,
                    "evidence_control_eligible": True,
                }
            )
        self.manifest = self.root / "metadata_val.csv"
        pd.DataFrame(manifest_rows).to_csv(self.manifest, index=False)
        self.patch_masks = self.root / "patch_masks_val.pt"
        teacher_cfg = self.config["data"]["teacher_maps"]
        fcv_cfg = self.config["fcv"]
        model_cfg = self.config["model"]
        preprocessing_config = {
            "teacher_map_source": str(teacher_cfg["source"]),
            "teacher_map_format": str(teacher_cfg["format"]),
            "foreground_class_ids": [
                int(value) for value in teacher_cfg["foreground_class_ids"]
            ],
            "normalize_to_unit_interval": bool(
                teacher_cfg["normalize_to_unit_interval"]
            ),
            "interpolation": str(teacher_cfg["interpolation"]),
            "spatial_transform": str(teacher_cfg["spatial_transform"]),
            "eval_resize_size": int(
                self.config["training"]["augmentation"]["eval_resize_size"]
            ),
            "image_size": int(model_cfg["image_size"]),
            "patch_size": int(model_cfg["patch_size"]),
            "patch_grid_size": int(model_cfg["patch_grid_size"]),
            "evidence_threshold": float(fcv_cfg["evidence_patch_threshold"]),
            "background_threshold": float(fcv_cfg["background_patch_threshold"]),
            "minimum_background_patches": int(
                fcv_cfg["minimum_background_patches"]
            ),
            "minimum_eligible_fraction": float(fcv_cfg["minimum_eligible_fraction"]),
            "minimum_eligible_count_per_class": int(
                fcv_cfg["minimum_eligible_count_per_class"]
            ),
            "ambiguous_patch_policy": str(fcv_cfg["ambiguous_patch_policy"]),
        }
        preprocessing_sha256 = sha256_json(preprocessing_config)
        teacher_maps_sha256 = sha256_json({"teacher_maps": teacher_map_records})
        torch.save(
            {
                "schema_version": 2,
                "artifact_type": "fcv_vit_patch_masks",
                "split": "biased_validation",
                "manifest_sha256": sha256_file(self.manifest),
                "manifest_bundle_path": str(
                    self.root / "SYNTHETIC_TEST_MANIFEST_BUNDLE"
                ),
                "manifest_bundle_sha256": "SYNTHETIC_TEST_MODE",
                "teacher_maps_sha256": teacher_maps_sha256,
                "preprocessing_config": preprocessing_config,
                "preprocessing_config_sha256": preprocessing_sha256,
                "image_size": 4,
                "patch_size": 2,
                "patch_grid_size": 2,
                "patch_count": 4,
                "evidence_threshold": 0.6,
                "background_threshold": 0.1,
                "minimum_background_patches": 1,
                "records": records,
            },
            self.patch_masks,
        )
        audit_path = self.root / "patch_masks_val_audit.csv"
        overlay_index_path = self.root / "preflight_overlays.csv"
        pd.DataFrame(
            [{"sample_id": record["sample_id"]} for record in records]
        ).to_csv(audit_path, index=False)
        pd.DataFrame(
            columns=[
                "sample_id",
                "image_path",
                "teacher_map_path",
                "overlay_path",
                "fcv_eligible",
            ]
        ).to_csv(overlay_index_path, index=False)
        summary = {
            "schema_version": 2,
            "artifact_type": "fcv_vit_patch_masks",
            "status": "complete",
            "manifest_sha256": sha256_file(self.manifest),
            "manifest_bundle_path": str(
                self.root / "SYNTHETIC_TEST_MANIFEST_BUNDLE"
            ),
            "manifest_bundle_sha256": "SYNTHETIC_TEST_MODE",
            "patch_mask_path": str(self.patch_masks),
            "patch_mask_sha256": sha256_file(self.patch_masks),
            "preprocessing_config_sha256": preprocessing_sha256,
            "teacher_maps_sha256": teacher_maps_sha256,
            "audit_path": str(audit_path),
            "audit_sha256": sha256_file(audit_path),
            "preflight_overlay_index": str(overlay_index_path),
            "preflight_overlay_index_sha256": sha256_file(overlay_index_path),
        }
        with (self.root / "patch_masks_val_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle)

    @staticmethod
    def _validation_metrics(model: nn.Module, source) -> dict[str, float]:
        total_loss = 0.0
        total_correct = 0
        total = 0
        model.eval()
        with torch.inference_mode():
            for images, labels, _records in source.loader:
                logits = model(images).float()
                total_loss += float(
                    F.cross_entropy(logits, labels, reduction="sum").item()
                )
                total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += int(labels.numel())
        return {
            "biased_val_loss": total_loss / total,
            "biased_val_accuracy": total_correct / total,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_execution_batch_sizes_are_locked(self) -> None:
        with self.assertRaisesRegex(TokenBankError, "batch size"):
            prepare_token_bank_source(
                self.config,
                self.manifest,
                self.patch_masks,
                batch_size=1,
            )
        with self.assertRaisesRegex(FCVScoringError, "locked value"):
            score_candidate_fcv(
                self.config,
                "missing.pt",
                None,
                "missing_banks",
                "missing_plan.pt",
                "missing_output",
                reconstruction_reports=self.reconstruction_reports,
                counterfactual_forward_batch_size=7,
            )
        with self.assertRaisesRegex(FCVControlError, "locked value"):
            score_candidate_controls(
                self.config,
                "missing.pt",
                None,
                "missing_banks",
                "missing_plan.pt",
                "missing_control_plan.pt",
                "missing_step7",
                "missing_output",
                reconstruction_reports=self.reconstruction_reports,
                target_batch_size=3,
                counterfactual_forward_batch_size=8,
            )

    def test_control_probability_statistics_use_serialized_draw_values(self) -> None:
        probabilities = torch.tensor(
            [
                0.9515874981880188,
                0.6856323480606079,
                0.8291820883750916,
                0.5362794995307922,
                0.9977296590805054,
            ],
            dtype=torch.float32,
        )
        values, canonical_mean, canonical_std = (
            _canonical_probability_draw_statistics(probabilities)
        )
        recomputed = np.asarray(
            json.loads(json.dumps(values, separators=(",", ":"))),
            dtype=np.float64,
        )

        # This vector reproduces the GH200 failure mode: a float32 tensor
        # reduction differs from the canonical persisted mean by > 1e-7.
        self.assertGreater(
            abs(float(probabilities.mean().item()) - float(recomputed.mean())),
            1.0e-7,
        )
        self.assertEqual(canonical_mean, float(recomputed.mean()))
        self.assertEqual(canonical_std, float(recomputed.std(ddof=0)))

    def test_builds_separate_float32_banks_with_compact_provenance(self) -> None:
        source = prepare_token_bank_source(
            self.config,
            self.manifest,
            self.patch_masks,
        )
        self.assertEqual(source.eligible_counts_by_label, {0: 2, 1: 2})
        first_map = Path(
            source.records_by_sample_id["sample_0"]["teacher_map_path"]
        )
        original_map_bytes = first_map.read_bytes()
        first_map.write_bytes(original_map_bytes + b"tampered")
        with self.assertRaisesRegex(Exception, "Teacher-map bytes changed"):
            prepare_token_bank_source(self.config, self.manifest, self.patch_masks)
        first_map.write_bytes(original_map_bytes)
        checkpoint_path = self.root / "candidate.pt"
        checkpoint_path.write_bytes(b"candidate checkpoint placeholder")
        candidate_id = "run_000_tiny_epoch_001"
        artifact = {
            "candidate_id": candidate_id,
            "run": {"run_index": 0, "learning_rate": 1e-5, "weight_decay": 0.01, "seed": 0},
            "epoch": 1,
            "training_fingerprint": candidate_training_fingerprint(self.config),
            "software_versions": software_versions(),
            "model": dict(self.config["model"]),
            "manifest_sha256": {
                "candidate_train": "unused",
                "biased_validation": source.manifest_sha256,
                "manifest_bundle": source.manifest_bundle_sha256,
            },
        }
        model = TinyTimmLikeViT().eval()
        output_dir = self.root / "token_banks"
        with mock.patch(
            "fcv.token_banks.load_candidate_model",
            return_value=(model, artifact),
        ):
            summary = build_background_token_banks(
                self.config,
                checkpoint_path,
                source,
                output_dir,
                reconstruction_reports=self.reconstruction_reports,
                device="cpu",
        )
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["safe_background_donor_image_count"], 4)

        for label, context_name in CONTEXT_NAMES.items():
            bank_path = output_dir / f"{candidate_id}_{context_name}.pt"
            bank = torch.load(bank_path, map_location="cpu", weights_only=False)
            self.assertEqual(bank["context_label_proxy"], label)
            self.assertEqual(bank["tokens"].shape, (4, 8))
            self.assertEqual(bank["tokens"].dtype, torch.float32)
            self.assertEqual(bank["source_image_count"], 2)
            self.assertEqual(bank["token_source_image_index"].tolist(), [0, 0, 1, 1])
            self.assertEqual(bank["token_source_patch_idx"].tolist(), [0, 3, 0, 3])
            self.assertEqual(bank["token_source_patch_row"].tolist(), [0, 1, 0, 1])
            self.assertEqual(bank["token_source_patch_col"].tolist(), [0, 1, 0, 1])
            self.assertTrue((bank["token_source_class"] == label).all())
            self.assertTrue((bank["token_patch_score"] <= 0.1).all())

        with mock.patch(
            "fcv.token_banks.load_candidate_model",
            return_value=(model, artifact),
        ), mock.patch(
            "fcv.token_banks.extract_raw_patch_tokens",
            side_effect=AssertionError("completed banks should be reused"),
        ):
            reused = build_background_token_banks(
                self.config,
                checkpoint_path,
                source,
                output_dir,
                reconstruction_reports=self.reconstruction_reports,
                device="cpu",
            )
        self.assertEqual(reused["status"], "reused")

        land_bank = output_dir / f"{candidate_id}_{CONTEXT_NAMES[0]}.pt"
        land_bank.write_bytes(land_bank.read_bytes() + b"tampered")
        with mock.patch(
            "fcv.token_banks.load_candidate_model",
            return_value=(model, artifact),
        ), self.assertRaisesRegex(Exception, "hash|stale|provenance"):
            build_background_token_banks(
                self.config,
                checkpoint_path,
                source,
                output_dir,
                reconstruction_reports=self.reconstruction_reports,
                device="cpu",
            )

    def test_failed_step3_summary_is_rejected(self) -> None:
        summary_path = self.root / "patch_masks_val_summary.json"
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["status"] = "failed_preprocessing"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle)
        with self.assertRaisesRegex(Exception, "summary field 'status' is stale"):
            prepare_token_bank_source(self.config, self.manifest, self.patch_masks)

    def test_aggregate_requires_every_candidate_and_both_contexts(self) -> None:
        config = dict(self.config)
        config["training"] = dict(self.config["training"])
        config["training"]["epochs"] = 2
        config["candidate_pool"]["candidate_epochs"] = [1, 2]
        config["training"]["scheduler"] = dict(self.config["training"]["scheduler"])
        config["training"]["scheduler"]["warmup_epochs"] = 1
        config["candidate_pool"] = dict(self.config["candidate_pool"])
        config["candidate_pool"]["expected_candidate_checkpoints"] = 54
        token_root = self.root / "aggregate_banks"
        token_root.mkdir()
        fingerprint = candidate_training_fingerprint(config)
        for run in enumerate_sweep_runs(config):
            for epoch in (1, 2):
                candidate_id = run.candidate_id(epoch)
                checkpoint_path = token_root / f"{candidate_id}.pt"
                checkpoint_path.write_bytes(candidate_id.encode("utf-8"))
                banks = {}
                for label, context_name in CONTEXT_NAMES.items():
                    bank_path = token_root / f"{candidate_id}_{context_name}.pt"
                    bank_path.write_bytes(b"x")
                    banks[context_name] = {
                        "path": str(bank_path),
                        "file_size_bytes": 1,
                        "context_label_proxy": label,
                        "token_count": 4,
                        "source_image_count": 2,
                        "embedding_dim": 8,
                        "sha256": sha256_file(bank_path),
                    }
                with (token_root / f"{candidate_id}_summary.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(
                        {
                            "artifact_type": "fcv_vit_token_bank_summary",
                            "status": "complete",
                            "candidate_id": candidate_id,
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_sha256": sha256_file(checkpoint_path),
                            "training_fingerprint": fingerprint,
                            "validation_manifest_sha256": "m" * 64,
                            "patch_mask_sha256": "p" * 64,
                            "banks": banks,
                        },
                        handle,
                    )
        result = aggregate_token_bank_summaries(
            config,
            token_root,
            token_root / "index.csv",
            token_root / "summary.json",
            manifest_sha256="m" * 64,
            patch_mask_sha256="p" * 64,
        )
        index = pd.read_csv(token_root / "index.csv")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["candidate_count"], 54)
        self.assertEqual(result["bank_count"], 108)
        self.assertEqual(len(index), 108)

    def test_step7_uses_one_deterministic_opposite_context_plan(self) -> None:
        source = prepare_token_bank_source(
            self.config,
            self.manifest,
            self.patch_masks,
        )
        checkpoint_path = self.root / "candidate.pt"
        checkpoint_path.write_bytes(b"candidate checkpoint placeholder")
        candidate_epoch = self.config["candidate_pool"]["candidate_epochs"][0]
        candidate_id = get_sweep_run(self.config, 0).candidate_id(candidate_epoch)
        artifact = {
            "candidate_id": candidate_id,
            "run": {
                "run_index": 0,
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
                "seed": 0,
            },
            "epoch": candidate_epoch,
            "training_fingerprint": candidate_training_fingerprint(self.config),
            "software_versions": software_versions(),
            "model": dict(self.config["model"]),
            "manifest_sha256": {
                "candidate_train": "unused",
                "biased_validation": source.manifest_sha256,
                "manifest_bundle": source.manifest_bundle_sha256,
            },
        }
        model = TinyTimmLikeViT().eval()
        artifact["metrics"] = self._validation_metrics(model, source)
        token_dir = self.root / "step7_banks"
        with mock.patch(
            "fcv.token_banks.load_candidate_model",
            return_value=(model, artifact),
        ):
            build_background_token_banks(
                self.config,
                checkpoint_path,
                source,
                token_dir,
                reconstruction_reports=self.reconstruction_reports,
                device="cpu",
            )
        banks = {
            label: load_background_bank(
                self.config,
                token_dir / f"{candidate_id}_{context_name}.pt",
                source,
                expected_label=label,
                expected_candidate_id=candidate_id,
                expected_checkpoint_sha256=sha256_file(checkpoint_path),
            )
            for label, context_name in CONTEXT_NAMES.items()
        }
        plan_path = self.root / "scores" / "opposite_donor_plan.pt"
        plan = prepare_opposite_donor_plan(
            self.config,
            source,
            banks,
            plan_path,
        )
        reused_plan = prepare_opposite_donor_plan(
            self.config,
            source,
            banks,
            plan_path,
        )
        self.assertEqual(
            plan["plan_content_sha256"], reused_plan["plan_content_sha256"]
        )
        self.assertEqual(plan["eligible_sample_count"], 4)
        for record in plan["records"]:
            self.assertEqual(
                int(record["donor_context_label"]),
                1 - int(record["label"]),
            )
            self.assertEqual(tuple(record["donor_token_indices"].shape), (5, 2))
            donor_bank = banks[int(record["donor_context_label"])]
            donor_source_indices = donor_bank["token_source_image_index"].index_select(
                0, record["donor_token_indices"].flatten()
            )
            donor_source_labels = donor_bank["token_source_class"].index_select(
                0, record["donor_token_indices"].flatten()
            )
            self.assertTrue(
                torch.all(donor_source_labels == int(record["donor_context_label"]))
            )
            target_source_index = donor_bank["source_sample_id_to_index"].get(
                record["sample_id"]
            )
            if target_source_index is not None:
                self.assertFalse(torch.any(donor_source_indices == target_source_index))

        target = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        donors = 100 + torch.arange(32, dtype=torch.float32).reshape(4, 8)
        donor_indices = torch.tensor([[1, 2], [3, 0]], dtype=torch.long)
        counterfactual = make_counterfactual_token_batch(
            target,
            torch.tensor([0, 3]),
            donors,
            donor_indices,
        )
        self.assertTrue(torch.equal(counterfactual[:, 1], target[1].expand(2, -1)))
        self.assertTrue(torch.equal(counterfactual[:, 2], target[2].expand(2, -1)))
        self.assertTrue(torch.equal(counterfactual[0, 0], donors[1]))
        self.assertTrue(torch.equal(counterfactual[0, 3], donors[2]))
        self.assertTrue(torch.equal(counterfactual[1, 0], donors[3]))
        self.assertTrue(torch.equal(counterfactual[1, 3], donors[0]))

        score_dir = self.root / "scores"
        with mock.patch(
            "fcv.fcv_scoring.load_candidate_model",
            return_value=(model, artifact),
        ):
            summary = score_candidate_fcv(
                self.config,
                checkpoint_path,
                source,
                token_dir,
                plan_path,
                score_dir,
                reconstruction_reports=self.reconstruction_reports,
                device="cpu",
                counterfactual_forward_batch_size=8,
            )
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["fcv_eligible_sample_count"], 4)
        self.assertEqual(summary["identity_swap_max_abs_error"], 0.0)
        self.assertTrue(summary["identity_swap_uses_production_replacement_path"])
        self.assertEqual(summary["identity_swap_path_sample_count"], 4)
        self.assertEqual(summary["identity_swap_inside_token_max_abs_error"], 0.0)
        self.assertEqual(summary["identity_swap_outside_token_max_abs_error"], 0.0)
        swap = summary["real_swap_integrity_diagnostics"]
        self.assertEqual(swap["foreground_token_max_abs_error"], 0.0)
        self.assertEqual(swap["donor_reconstruction_max_abs_error"], 0.0)
        self.assertGreater(swap["replaced_token_changed_count"], 0)
        self.assertGreater(swap["replaced_token_changed_fraction"], 0.0)
        self.assertGreater(swap["replacement_delta_max"], 0.0)
        score_frame = pd.read_csv(score_dir / f"{candidate_id}.csv")
        self.assertEqual(len(score_frame), 4)
        self.assertTrue(score_frame["fcv_eligible"].all())
        self.assertTrue((score_frame["donor_draw_count"] == 5).all())
        self.assertTrue((score_frame["num_background_patches_swapped"] == 2).all())
        self.assertTrue(
            (score_frame["real_swap_foreground_max_abs_error"] == 0.0).all()
        )
        self.assertTrue(
            (
                score_frame["real_swap_donor_reconstruction_max_abs_error"]
                == 0.0
            ).all()
        )

        with mock.patch(
            "fcv.fcv_scoring.load_candidate_model",
            return_value=(model, artifact),
        ), mock.patch(
            "fcv.fcv_scoring.extract_raw_patch_tokens",
            side_effect=AssertionError("completed candidate score should be reused"),
        ):
            reused_summary = score_candidate_fcv(
                self.config,
                checkpoint_path,
                source,
                token_dir,
                plan_path,
                score_dir,
                reconstruction_reports=self.reconstruction_reports,
                device="cpu",
                counterfactual_forward_batch_size=8,
            )
        self.assertEqual(reused_summary["status"], "reused")

        aggregate = aggregate_fcv_score_summaries(
            self.config,
            score_dir,
            score_dir / "candidate_fcv_scores.csv",
            score_dir / "candidate_fcv_scores_summary.json",
            source=source,
            donor_plan_path=plan_path,
            allow_incomplete=True,
        )
        self.assertEqual(aggregate["status"], "incomplete")
        self.assertEqual(aggregate["candidate_count"], 1)
        self.assertTrue(aggregate["selection_metrics_recomputed_from_hashed_csvs"])
        self.assertEqual(aggregate["identity_swap_candidate_failure_count"], 0)
        self.assertEqual(aggregate["token_diagnostic_candidate_count"], 1)

        # A mutable JSON summary cannot alter FCV selection metrics while its
        # hashed per-image draw CSV remains unchanged.
        summary_path = score_dir / f"{candidate_id}_summary.json"
        with summary_path.open("r", encoding="utf-8") as handle:
            tampered = json.load(handle)
        tampered["primary_selector_score"] += 0.1
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(tampered, handle)
        tamper_aggregate = aggregate_fcv_score_summaries(
            self.config,
            score_dir,
            score_dir / "candidate_fcv_scores_tampered.csv",
            score_dir / "candidate_fcv_scores_tampered_summary.json",
            source=source,
            donor_plan_path=plan_path,
            allow_incomplete=True,
        )
        self.assertEqual(tamper_aggregate["candidate_count"], 0)
        self.assertEqual(tamper_aggregate["invalid_candidate_count"], 1)

    def test_step8_scores_all_controls_from_one_cached_plan(self) -> None:
        source = prepare_token_bank_source(
            self.config,
            self.manifest,
            self.patch_masks,
        )
        checkpoint_path = self.root / "control_candidate.pt"
        checkpoint_path.write_bytes(b"control candidate checkpoint placeholder")
        candidate_epoch = self.config["candidate_pool"]["candidate_epochs"][0]
        candidate_id = get_sweep_run(self.config, 0).candidate_id(candidate_epoch)
        artifact = {
            "candidate_id": candidate_id,
            "run": {
                "run_index": 0,
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
                "seed": 0,
            },
            "epoch": candidate_epoch,
            "training_fingerprint": candidate_training_fingerprint(self.config),
            "software_versions": software_versions(),
            "model": dict(self.config["model"]),
            "manifest_sha256": {
                "candidate_train": "unused",
                "biased_validation": source.manifest_sha256,
                "manifest_bundle": source.manifest_bundle_sha256,
            },
        }
        model = TinyTimmLikeViT().eval()
        artifact["metrics"] = self._validation_metrics(model, source)
        token_dir = self.root / "control_banks"
        with mock.patch(
            "fcv.token_banks.load_candidate_model",
            return_value=(model, artifact),
        ):
            build_background_token_banks(
                self.config,
                checkpoint_path,
                source,
                token_dir,
                reconstruction_reports=self.reconstruction_reports,
                device="cpu",
            )
        banks = {
            label: load_background_bank(
                self.config,
                token_dir / f"{candidate_id}_{context_name}.pt",
                source,
                expected_label=label,
                expected_candidate_id=candidate_id,
                expected_checkpoint_sha256=sha256_file(checkpoint_path),
            )
            for label, context_name in CONTEXT_NAMES.items()
        }
        step7_dir = self.root / "control_step7"
        opposite_plan_path = step7_dir / "opposite_donor_plan.pt"
        opposite_plan = prepare_opposite_donor_plan(
            self.config,
            source,
            banks,
            opposite_plan_path,
        )
        with mock.patch(
            "fcv.fcv_scoring.load_candidate_model",
            return_value=(model, artifact),
        ):
            step7_summary = score_candidate_fcv(
                self.config,
                checkpoint_path,
                source,
                token_dir,
                opposite_plan_path,
                step7_dir,
                reconstruction_reports=self.reconstruction_reports,
                device="cpu",
                counterfactual_forward_batch_size=8,
            )
        self.assertEqual(step7_summary["status"], "complete")

        control_dir = self.root / "controls"
        control_plan_path = control_dir / "control_plan.pt"
        control_plan = prepare_control_plan(
            self.config,
            source,
            banks,
            opposite_plan,
            opposite_plan_path,
            control_plan_path,
        )
        reused_plan = prepare_control_plan(
            self.config,
            source,
            banks,
            opposite_plan,
            opposite_plan_path,
            control_plan_path,
        )
        self.assertEqual(
            control_plan["plan_content_sha256"],
            reused_plan["plan_content_sha256"],
        )
        self.assertEqual(control_plan["fcv_eligible_sample_count"], 4)
        self.assertEqual(control_plan["evidence_eligible_sample_count"], 4)
        for record in control_plan["records"]:
            self.assertEqual(tuple(record["same_context_donor_token_indices"].shape), (5, 2))
            self.assertEqual(record["random_mask_idx"].numel(), 2)
            self.assertEqual(torch.unique(record["random_mask_idx"]).numel(), 2)
            self.assertNotEqual(
                record["sample_id"], record["shuffled_mask_source_sample_id"]
            )
            self.assertEqual(
                tuple(record["evidence_opposite_donor_token_indices"].shape),
                (5, 1),
            )
            same_bank = banks[int(record["label"])]
            target_source = same_bank["source_sample_id_to_index"].get(
                record["sample_id"]
            )
            if target_source is not None:
                donor_sources = same_bank["token_source_image_index"].index_select(
                    0, record["same_context_donor_token_indices"].flatten()
                )
                self.assertFalse(torch.any(donor_sources == target_source))

        with mock.patch(
            "fcv.controls.load_candidate_model",
            return_value=(model, artifact),
        ):
            summary = score_candidate_controls(
                self.config,
                checkpoint_path,
                source,
                token_dir,
                opposite_plan_path,
                control_plan_path,
                step7_dir,
                control_dir,
                reconstruction_reports=self.reconstruction_reports,
                device="cpu",
                target_batch_size=2,
                counterfactual_forward_batch_size=8,
            )
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(set(summary["controls"]), set(CONTROL_NAMES))
        self.assertIn(summary["diagnostic_status"], {"passed", "warning"})
        self.assertEqual(
            summary["diagnostic_warning_count"],
            len(summary["diagnostic_warnings"]),
        )
        for control_name in CONTROL_NAMES:
            csv_path = control_dir / f"{candidate_id}_{control_name}.csv"
            frame = pd.read_csv(csv_path)
            self.assertEqual(len(frame), 4)
            self.assertTrue(frame["control_eligible"].all())
            self.assertTrue((frame["donor_draw_count"] == 5).all())
            self.assertEqual(
                summary["controls"][control_name]["eligible_sample_count"], 4
            )
            integrity = summary["controls"][control_name]
            self.assertEqual(
                integrity["swap_preserved_token_max_abs_error"], 0.0
            )
            self.assertEqual(
                integrity["swap_donor_reconstruction_max_abs_error"], 0.0
            )
            self.assertGreater(integrity["swap_replaced_token_draw_count"], 0)
            self.assertGreater(integrity["swap_replaced_token_changed_count"], 0)
            self.assertGreater(
                integrity["swap_replaced_token_changed_fraction"], 0.0
            )
            self.assertGreater(integrity["swap_replacement_delta_max"], 0.0)
            self.assertTrue(
                (frame["swap_preserved_token_max_abs_error"] == 0.0).all()
            )
            self.assertTrue(
                (frame["swap_donor_reconstruction_max_abs_error"] == 0.0).all()
            )

        no_op = frame.copy()
        no_op["swap_replaced_token_changed_count"] = 0
        no_op["swap_replaced_token_changed_fraction"] = 0.0
        no_op["swap_replacement_delta_mean"] = 0.0
        no_op["swap_replacement_delta_max"] = 0.0
        with self.assertRaisesRegex(FCVControlError, "complete no-op"):
            recompute_control_metrics_from_frame(no_op)

        with mock.patch(
            "fcv.controls.load_candidate_model",
            return_value=(model, artifact),
        ), mock.patch(
            "fcv.controls.extract_raw_patch_tokens",
            side_effect=AssertionError("completed controls should be reused"),
        ):
            reused = score_candidate_controls(
                self.config,
                checkpoint_path,
                source,
                token_dir,
                opposite_plan_path,
                control_plan_path,
                step7_dir,
                control_dir,
                reconstruction_reports=self.reconstruction_reports,
                device="cpu",
                target_batch_size=2,
                counterfactual_forward_batch_size=8,
            )
        self.assertEqual(reused["status"], "reused")

        bank_summary_path = token_dir / f"{candidate_id}_summary.json"
        with bank_summary_path.open("r", encoding="utf-8") as handle:
            bank_summary = json.load(handle)
        receipt_path = (
            Path(self.config["paths"]["output_root"])
            / self.config["outputs"]["token_banks"]
            / "cleanup_receipts"
            / f"{candidate_id}.json"
        )
        receipt = _cleanup_candidate_banks(
            self.config,
            candidate_id=candidate_id,
            checkpoint_path=checkpoint_path.resolve(),
            checkpoint_sha256=sha256_file(checkpoint_path),
            bank_summary=bank_summary,
            fcv_summary_path=step7_dir / f"{candidate_id}_summary.json",
            control_summary_path=control_dir
            / f"{candidate_id}_controls_summary.json",
            receipt_path=receipt_path,
        )
        self.assertEqual(receipt["status"], "complete")
        for context_name in CONTEXT_NAMES.values():
            self.assertFalse(
                (token_dir / f"{candidate_id}_{context_name}.pt").exists()
            )

        fcv_aggregate = aggregate_fcv_score_summaries(
            self.config,
            step7_dir,
            step7_dir / "candidate_fcv_scores_after_cleanup.csv",
            step7_dir / "candidate_fcv_scores_after_cleanup_summary.json",
            source=source,
            donor_plan_path=opposite_plan_path,
            allow_incomplete=True,
        )
        self.assertEqual(fcv_aggregate["candidate_count"], 1)

        aggregate = aggregate_control_summaries(
            self.config,
            control_dir,
            control_dir / "candidate_control_scores.csv",
            control_dir / "candidate_control_scores_summary.json",
            source=source,
            opposite_plan_path=opposite_plan_path,
            control_plan_path=control_plan_path,
            allow_incomplete=True,
        )
        self.assertEqual(aggregate["status"], "incomplete")
        self.assertEqual(aggregate["candidate_count"], 1)
        self.assertTrue(aggregate["selection_metrics_recomputed_from_hashed_csvs"])

        # Control differences are likewise reproduced from all four raw
        # control CSVs plus the independently hashed Step 7 CSV.
        summary_path = control_dir / f"{candidate_id}_controls_summary.json"
        with summary_path.open("r", encoding="utf-8") as handle:
            tampered = json.load(handle)
        tampered["same_minus_opposite_accuracy"] += 0.1
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(tampered, handle)
        tamper_aggregate = aggregate_control_summaries(
            self.config,
            control_dir,
            control_dir / "candidate_control_scores_tampered.csv",
            control_dir / "candidate_control_scores_tampered_summary.json",
            source=source,
            opposite_plan_path=opposite_plan_path,
            control_plan_path=control_plan_path,
            allow_incomplete=True,
        )
        self.assertEqual(tamper_aggregate["candidate_count"], 0)
        self.assertEqual(tamper_aggregate["invalid_candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
