from __future__ import annotations

import sys
import tempfile
import unittest
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.patch_masks import PatchMaskError, prepare_patch_masks  # noqa: E402


class PatchMasksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.map_root = self.root / "maps"
        self.map_root.mkdir()

        # Real prediction_cmap encoding: class 1 is VOC red [128, 0, 0].
        # The non-square source also exercises Resize(256)+CenterCrop(224).
        centered = np.zeros((256, 320, 3), dtype=np.uint8)
        centered[64:176, 96:208] = (128, 0, 0)
        almost_full = np.full((256, 320, 3), (128, 0, 0), dtype=np.uint8)
        almost_full[16:32, 48:64] = 0
        Image.fromarray(centered).save(self.map_root / "centered.png")
        Image.fromarray(almost_full).save(self.map_root / "almost_full.png")
        for name in ("unused1.jpg", "unused2.jpg"):
            Image.fromarray(np.full((256, 320, 3), 120, dtype=np.uint8)).save(
                self.root / name
            )

        manifest = pd.DataFrame(
            [
                {
                    "sample_id": "wb100_00001",
                    "metadata_index": 1,
                    "image_path": str(self.root / "unused1.jpg"),
                    "image_sha256": hashlib.sha256(
                        (self.root / "unused1.jpg").read_bytes()
                    ).hexdigest(),
                    "image_rel_path": "001.Bird/Bird_1.jpg",
                    "label": 0,
                    "class_name": "Landbird",
                    "source_split": "train",
                    "study_split": "biased_validation",
                    "teacher_map_path": str(self.map_root / "centered.png"),
                    "teacher_map_exists": True,
                },
                {
                    "sample_id": "wb100_00002",
                    "metadata_index": 2,
                    "image_path": str(self.root / "unused2.jpg"),
                    "image_sha256": hashlib.sha256(
                        (self.root / "unused2.jpg").read_bytes()
                    ).hexdigest(),
                    "image_rel_path": "002.Bird/Bird_2.jpg",
                    "label": 1,
                    "class_name": "Waterbird",
                    "source_split": "train",
                    "study_split": "biased_validation",
                    "teacher_map_path": str(self.map_root / "almost_full.png"),
                    "teacher_map_exists": True,
                },
            ]
        )
        self.manifest_path = self.root / "metadata_val.csv"
        manifest.to_csv(self.manifest_path, index=False)

        base_config_path = (
            EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
        )
        with base_config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["paths"]["output_root"] = str(self.root / "outputs")
        self.config_path = self.root / "config.yaml"
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_patch_partition_and_audit(self) -> None:
        import torch

        config = load_and_validate_config(self.config_path)
        config["_synthetic_test_mode"] = True
        config["fcv"]["minimum_eligible_count_per_class"] = 0
        result = prepare_patch_masks(config, self.manifest_path, self.root / "patch_masks")
        payload = torch.load(result["artifacts"]["patch_masks"], map_location="cpu")
        audit = pd.read_csv(result["artifacts"]["audit"])

        self.assertEqual(payload["patch_count"], 196)
        self.assertEqual(len(payload["records"]), 2)
        self.assertEqual(len(payload["manifest_sha256"]), 64)
        self.assertEqual(len(payload["teacher_maps_sha256"]), 64)
        self.assertEqual(len(payload["preprocessing_config_sha256"]), 64)
        centered = payload["records"][0]
        self.assertEqual(centered["patch_scores"].shape, (196,))
        self.assertEqual(centered["evidence_idx"].numel(), 49)
        self.assertEqual(centered["background_idx"].numel(), 147)
        self.assertEqual(centered["ambiguous_idx"].numel(), 0)
        self.assertTrue(centered["fcv_eligible"])
        self.assertAlmostEqual(centered["coverage"]["evidence_frac"], 49 / 196)

        almost_full = payload["records"][1]
        self.assertEqual(almost_full["background_idx"].numel(), 1)
        self.assertFalse(almost_full["fcv_eligible"])
        self.assertEqual(
            almost_full["eligibility_reason"], "insufficient_background_patches"
        )
        self.assertEqual(int(audit["fcv_eligible"].sum()), 1)
        self.assertEqual(result["summary"]["eligible_count"], 1)
        self.assertEqual(payload["teacher_map_format"], "voc_colormap_class_ids")
        self.assertEqual(payload["spatial_transform"], "eval_resize_shorter_side_then_center_crop")
        self.assertEqual(result["summary"]["preflight_overlay_count"], 2)
        self.assertTrue(Path(result["summary"]["preflight_overlay_index"]).is_file())

    def test_failed_eligibility_still_persists_diagnostics(self) -> None:
        empty = np.zeros((256, 320, 3), dtype=np.uint8)
        Image.fromarray(empty).save(self.map_root / "centered.png")
        Image.fromarray(empty).save(self.map_root / "almost_full.png")
        config = load_and_validate_config(self.config_path)
        config["_synthetic_test_mode"] = True
        output = self.root / "failed_patch_masks"
        with self.assertRaisesRegex(PatchMaskError, "Diagnostics were written"):
            prepare_patch_masks(config, self.manifest_path, output)
        self.assertTrue((output / "patch_masks_val_audit.csv").is_file())
        self.assertTrue((output / "patch_masks_val_summary.json").is_file())
        self.assertTrue((output / "preflight_overlays.csv").is_file())

    def test_failed_overwrite_invalidates_previously_accepted_artifact(self) -> None:
        config = load_and_validate_config(self.config_path)
        config["_synthetic_test_mode"] = True
        config["fcv"]["minimum_eligible_count_per_class"] = 0
        output = self.root / "overwrite_patch_masks"
        prepare_patch_masks(config, self.manifest_path, output)
        artifact_path = output / "patch_masks_val.pt"
        self.assertTrue(artifact_path.is_file())

        empty = np.zeros((256, 320, 3), dtype=np.uint8)
        Image.fromarray(empty).save(self.map_root / "centered.png")
        Image.fromarray(empty).save(self.map_root / "almost_full.png")
        with self.assertRaisesRegex(PatchMaskError, "Diagnostics were written"):
            prepare_patch_masks(
                config,
                self.manifest_path,
                output,
                overwrite=True,
            )
        self.assertFalse(artifact_path.exists())
        with (output / "patch_masks_val_summary.json").open(
            "r", encoding="utf-8"
        ) as handle:
            summary = json.load(handle)
        self.assertEqual(summary["status"], "failed_acceptance")

    def test_decode_failure_is_persisted_and_fail_closed(self) -> None:
        invalid = np.zeros((256, 320, 3), dtype=np.uint8)
        invalid[0, 0] = (1, 2, 3)
        # Fail on the second image so the diagnostic prefix proves that the
        # first completed sample survives an early preprocessing error.
        Image.fromarray(invalid).save(self.map_root / "almost_full.png")
        config = load_and_validate_config(self.config_path)
        config["_synthetic_test_mode"] = True
        output = self.root / "decode_failure"
        stale_overlay_dir = output / "preflight_overlays"
        stale_overlay_dir.mkdir(parents=True)
        stale_overlay = stale_overlay_dir / "stale.png"
        stale_overlay.write_bytes(b"stale")
        with self.assertRaises(PatchMaskError):
            prepare_patch_masks(
                config, self.manifest_path, output, overwrite=True
            )
        self.assertFalse((output / "patch_masks_val.pt").exists())
        self.assertFalse(stale_overlay.exists())
        with (output / "patch_masks_val_summary.json").open(
            "r", encoding="utf-8"
        ) as handle:
            summary = json.load(handle)
        self.assertEqual(summary["status"], "failed_preprocessing")
        self.assertIn("VOC", summary["error"])
        self.assertTrue((output / "patch_masks_val_audit.csv").is_file())
        audit = pd.read_csv(output / "patch_masks_val_audit.csv")
        self.assertEqual(len(audit), 2)
        self.assertEqual(audit.iloc[0]["sample_id"], "wb100_00001")
        self.assertEqual(audit.iloc[1]["processing_status"], "failed")
        self.assertEqual(audit.iloc[1]["sample_id"], "wb100_00002")
        self.assertEqual(int(audit.iloc[1]["label"]), 1)
        self.assertEqual(
            audit.iloc[1]["processing_stage"], "decode_and_partition_teacher_map"
        )
        self.assertEqual(summary["sample_id"], "wb100_00002")
        self.assertEqual(
            summary["processing_stage"], "decode_and_partition_teacher_map"
        )
        self.assertEqual(summary["partial_audit_row_count"], 2)
        self.assertEqual(summary["completed_sample_count_before_failure"], 1)
        overlays = pd.read_csv(output / "preflight_overlays.csv")
        self.assertEqual(len(overlays), 1)

    def test_native_map_image_dimension_mismatch_is_rejected(self) -> None:
        mismatched = np.zeros((255, 320, 3), dtype=np.uint8)
        Image.fromarray(mismatched).save(self.map_root / "centered.png")
        config = load_and_validate_config(self.config_path)
        config["_synthetic_test_mode"] = True
        output = self.root / "dimension_failure"
        with self.assertRaisesRegex(PatchMaskError, "native size"):
            prepare_patch_masks(config, self.manifest_path, output)
        self.assertFalse((output / "patch_masks_val.pt").exists())
        with (output / "patch_masks_val_summary.json").open(
            "r", encoding="utf-8"
        ) as handle:
            summary = json.load(handle)
        self.assertEqual(summary["status"], "failed_preprocessing")

    def test_analysis_columns_are_rejected(self) -> None:
        manifest = pd.read_csv(self.manifest_path)
        manifest["group"] = [0, 3]
        leaked = self.root / "metadata_val_leaked.csv"
        manifest.to_csv(leaked, index=False)
        config = load_and_validate_config(self.config_path)
        config["_synthetic_test_mode"] = True
        with self.assertRaisesRegex(ValueError, "analysis-only columns"):
            prepare_patch_masks(config, leaked, self.root / "leaked_output")


if __name__ == "__main__":
    unittest.main()
