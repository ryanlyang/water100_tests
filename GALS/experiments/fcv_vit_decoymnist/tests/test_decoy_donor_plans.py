from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_data import expected_patch_intensity  # noqa: E402
from decoy_donor_plans import (  # noqa: E402
    DonorPlanError,
    build_donor_records,
    load_and_validate_donor_plan,
    prepare_donor_plan,
)
from decoy_full_config import load_and_validate_config, sha256_file  # noqa: E402
from decoy_manifest_provenance import ManifestBinding  # noqa: E402


class DonorRecordTest(unittest.TestCase):
    def test_plan_is_deterministic_multiclass_and_same_corner(self) -> None:
        rows = []
        corners = {}
        eligible = {}
        for corner in ("top_left", "top_right"):
            for label in range(10):
                sample_id = f"{corner}_{label}"
                rows.append({"sample_id": sample_id, "label": label})
                corners[sample_id] = corner
                eligible[sample_id] = True
        frame = pd.DataFrame(rows)
        first = build_donor_records(
            frame, corners, eligible, seed=0, donors_per_target=5
        )
        second = build_donor_records(
            frame.sample(frac=1.0, random_state=7),
            corners,
            eligible,
            seed=0,
            donors_per_target=5,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(frame))
        for record in first:
            labels = [donor["label"] for donor in record["donors"]]
            self.assertEqual(len(labels), 5)
            self.assertEqual(len(set(labels)), 5)
            self.assertNotIn(record["target_label"], labels)
            self.assertTrue(
                all(donor["corner"] == record["corner"] for donor in record["donors"])
            )

    def test_plan_fails_without_five_non_target_labels(self) -> None:
        frame = pd.DataFrame(
            [{"sample_id": f"s{label}", "label": label} for label in range(5)]
        )
        corners = {sample_id: "top_left" for sample_id in frame["sample_id"]}
        eligible = {sample_id: True for sample_id in frame["sample_id"]}
        with self.assertRaisesRegex(DonorPlanError, "requires 5"):
            build_donor_records(
                frame, corners, eligible, seed=0, donors_per_target=5
            )

    def test_ineligible_samples_are_neither_targets_nor_donors(self) -> None:
        frame = pd.DataFrame(
            [{"sample_id": f"s{label}", "label": label} for label in range(10)]
        )
        corners = {sample_id: "top_left" for sample_id in frame["sample_id"]}
        eligible = {sample_id: sample_id != "s9" for sample_id in frame["sample_id"]}
        records = build_donor_records(
            frame, corners, eligible, seed=0, donors_per_target=5
        )
        self.assertNotIn("s9", {record["target_sample_id"] for record in records})
        self.assertNotIn(
            "s9",
            {
                donor["sample_id"]
                for record in records
                for donor in record["donors"]
            },
        )


class DonorArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = copy.deepcopy(
            load_and_validate_config(
                EXPERIMENT_ROOT
                / "configs"
                / "decoymnist_vit_s16_fcv_full_online.yaml"
            )
        )
        self.config["paths"]["data_root"] = str(self.root / "data")
        rows = []
        data_root = Path(self.config["paths"]["data_root"])
        for label in range(10):
            relative = Path("train") / str(label) / f"sample_{label}.png"
            path = data_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            image = np.zeros((28, 28), dtype=np.uint8)
            image[:5, :5] = expected_patch_intensity(label, "train")
            Image.fromarray(image, mode="L").save(path)
            rows.append(
                {
                    "sample_id": f"train_{label}_sample_{label}",
                    "image_rel_path": str(relative),
                    "label": label,
                    "source_split": "train",
                    "study_split": "biased_validation",
                    "image_sha256": sha256_file(path),
                }
            )
        self.frame = pd.DataFrame(rows)
        self.manifest = self.root / "metadata_val.csv"
        self.frame.to_csv(self.manifest, index=False)
        self.mask_path = self.root / "projected_teacher_masks.npz"
        self.mask_path.write_bytes(b"compact-mask-fixture")
        self.binding = ManifestBinding(
            role="biased_validation",
            manifest_path=self.manifest,
            manifest_sha256="a" * 64,
            bundle_path=self.root / "manifest_bundle.json",
            bundle_sha256="b" * 64,
            split_assignments_sha256="c" * 64,
            split_summary_sha256="d" * 64,
            config_sha256="e" * 64,
        )
        self.arrays = {
            "sample_ids": np.asarray(self.frame["sample_id"], dtype=str),
            "labels": np.asarray(self.frame["label"], dtype=np.int16),
            "fcv_eligible": np.ones(len(self.frame), dtype=np.bool_),
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_json_artifact_is_compact_reusable_and_tamper_evident(self) -> None:
        output = self.root / "donor_plans" / "plan.json"
        with patch(
            "decoy_donor_plans.load_projected_teacher_masks",
            return_value=(self.arrays, self.binding),
        ):
            payload = prepare_donor_plan(
                self.config, self.manifest, self.mask_path, output
            )
            loaded = load_and_validate_donor_plan(
                self.config, self.manifest, self.mask_path, output
            )
        self.assertEqual(payload, loaded)
        self.assertEqual(payload["target_count"], 10)
        self.assertLess(output.stat().st_size, 100_000)
        text = output.read_text(encoding="utf-8")
        self.assertNotIn("embedding", text.lower())
        self.assertNotIn("logits", text.lower())
        self.assertNotIn(str(self.root), text)

        tampered = output.read_text(encoding="utf-8").replace(
            '"target_label": 0', '"target_label": 9', 1
        )
        output.write_text(tampered, encoding="utf-8")
        with patch(
            "decoy_donor_plans.load_projected_teacher_masks",
            return_value=(self.arrays, self.binding),
        ):
            with self.assertRaisesRegex(DonorPlanError, "content hash"):
                load_and_validate_donor_plan(
                    self.config, self.manifest, self.mask_path, output
                )


if __name__ == "__main__":
    unittest.main()
