from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_data import expected_patch_intensity, locate_decoy_patch  # noqa: E402
from decoy_full_config import (  # noqa: E402
    canonical_config_sha256,
    load_and_validate_config,
)
from decoy_manifests import prepare_decoymnist_manifests  # noqa: E402
from decoy_teacher_masks import (  # noqa: E402
    TeacherMaskError,
    load_projected_teacher_masks,
    prepare_teacher_masks,
)


class DecoyTeacherMasksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "DecoyMNIST_png"
        self.teacher_root = self.root / "prediction_cmap"
        self.teacher_root.mkdir(parents=True)
        self.train_sizes = [10] * 10
        self.test_sizes = [2] * 10
        for split, sizes in (("train", self.train_sizes), ("test", self.test_sizes)):
            for label, size in enumerate(sizes):
                folder = self.data_root / split / str(label)
                folder.mkdir(parents=True, exist_ok=True)
                for index in range(size):
                    array = np.zeros((28, 28), dtype=np.uint8)
                    array[10:18, 10:18] = 100 + label
                    intensity = expected_patch_intensity(label, split)
                    corners = (
                        (slice(0, 5), slice(0, 5)),
                        (slice(0, 5), slice(23, 28)),
                        (slice(23, 28), slice(0, 5)),
                        (slice(23, 28), slice(23, 28)),
                    )
                    rows, columns = corners[index % 4]
                    array[rows, columns] = intensity
                    Image.fromarray(array, mode="L").save(
                        folder / f"{index:05d}_y{label}.png"
                    )

        config_path = (
            EXPERIMENT_ROOT
            / "configs"
            / "decoymnist_vit_s16_fcv_full_online.yaml"
        )
        self.config = load_and_validate_config(config_path)
        self.config["paths"]["data_root"] = str(self.data_root)
        self.config["paths"]["teacher_map_root"] = str(self.teacher_root)
        self.config["data"]["source_counts"] = {"train": 100, "test": 20}
        self.config["data"]["partition"].update(
            {
                "candidate_train_count": 60,
                "biased_validation_count": 20,
                "oracle_validation_source_count": 20,
            }
        )
        self.config["data"]["teacher_maps"]["preflight_overlay_count"] = 3
        self.config["fcv"]["minimum_eligible_fraction"] = 0.0
        self.config["fcv"]["minimum_eligible_per_class"] = 0
        self.config["_provenance"]["canonical_config_sha256"] = (
            canonical_config_sha256(self.config)
        )
        self.manifest_result = prepare_decoymnist_manifests(
            self.config, self.root / "split_manifests", workers=0
        )
        self.validation_path = Path(
            self.manifest_result["artifacts"]["biased_validation"]
        )
        self.validation = pd.read_csv(self.validation_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_maps(self, *, omit_first: bool = False, contaminate_first: bool = False) -> None:
        for index, row in enumerate(self.validation.itertuples(index=False)):
            if omit_first and index == 0:
                continue
            source_path = self.data_root / str(row.image_rel_path)
            with Image.open(source_path) as image:
                grayscale = np.asarray(image, dtype=np.uint8).copy()
            patch_rows, patch_columns = locate_decoy_patch(
                grayscale, int(row.label), "train"
            )
            digit = grayscale > 0
            digit[patch_rows, patch_columns] = False
            if contaminate_first and index == 0:
                digit[patch_rows, patch_columns] = True
            encoded = np.zeros((28, 28, 3), dtype=np.uint8)
            encoded[digit] = (128, 0, 0)
            stem = Path(str(row.image_rel_path)).stem
            Image.fromarray(encoded, mode="RGB").save(
                self.teacher_root / f"{int(row.label)}_{stem}.png"
            )

    def test_projects_maps_and_writes_small_nonpickle_artifact(self) -> None:
        self._write_maps()
        result = prepare_teacher_masks(
            self.config, self.validation_path, self.root / "teacher_audit"
        )
        self.assertEqual(result["summary"]["status"], "accepted")
        self.assertEqual(result["summary"]["teacher_map_count"], 20)
        self.assertEqual(result["summary"]["overlay_count"], 3)
        self.assertEqual(result["summary"]["decoy_evidence_target_count"], 0)
        self.assertAlmostEqual(
            result["summary"]["mean_teacher_exact_digit_iou_analysis_only"], 1.0
        )
        arrays, _binding = load_projected_teacher_masks(
            self.config, self.validation_path, result["artifacts"]["masks"]
        )
        self.assertEqual(arrays["patch_scores"].shape, (20, 196))
        self.assertEqual(arrays["patch_categories"].shape, (20, 196))
        self.assertTrue(arrays["fcv_eligible"].all())
        self.assertFalse(list((self.root / "teacher_audit").rglob("*.pt")))
        self.assertFalse(list((self.root / "teacher_audit").rglob("*.pth")))
        self.assertFalse(list((self.root / "teacher_audit").rglob("*.ckpt")))

    def test_missing_map_fails_with_exact_regeneration_manifest(self) -> None:
        self._write_maps(omit_first=True)
        output = self.root / "missing_audit"
        with self.assertRaisesRegex(FileNotFoundError, "coverage is incomplete"):
            prepare_teacher_masks(self.config, self.validation_path, output)
        missing = pd.read_csv(output / "missing_teacher_maps.csv")
        self.assertEqual(len(missing), 1)
        request = json.loads(
            (output / "missing_map_regeneration_request.json").read_text(
                encoding="utf-8"
            )
        )
        summary = json.loads(
            (output / "teacher_mask_preflight_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(request["status"], "required")
        self.assertTrue(request["do_not_substitute_another_teacher"])
        self.assertEqual(summary["status"], "failed_missing_teacher_maps")
        self.assertFalse((output / "projected_teacher_masks.npz").exists())

    def test_teacher_evidence_on_decoy_fails_closed(self) -> None:
        self._write_maps(contaminate_first=True)
        output = self.root / "contaminated_audit"
        with self.assertRaisesRegex(TeacherMaskError, "overlaps the decoy"):
            prepare_teacher_masks(self.config, self.validation_path, output)
        summary = json.loads(
            (output / "teacher_mask_preflight_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "failed_acceptance")
        self.assertEqual(summary["decoy_evidence_target_count"], 1)
        self.assertFalse((output / "projected_teacher_masks.npz").exists())


if __name__ == "__main__":
    unittest.main()
