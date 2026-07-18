from __future__ import annotations

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

from decoy_full_config import (  # noqa: E402
    canonical_config_sha256,
    load_and_validate_config,
)
from decoy_manifest_provenance import (  # noqa: E402
    ManifestProvenanceError,
    PUBLIC_COLUMNS,
    validate_manifest_bundle,
)
from decoy_manifests import (  # noqa: E402
    largest_remainder_counts,
    prepare_decoymnist_manifests,
)
from decoy_data import expected_patch_intensity  # noqa: E402


class DecoyManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "DecoyMNIST_png"
        self.train_sizes = [11 + label for label in range(10)]
        self.test_sizes = [3] * 10
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
                        folder / f"{label}_{index:05d}_y{label}.png"
                    )

        base_config = (
            EXPERIMENT_ROOT
            / "configs"
            / "decoymnist_vit_s16_fcv_full_online.yaml"
        )
        self.config = load_and_validate_config(base_config)
        self.config["paths"]["data_root"] = str(self.data_root)
        self.config["data"]["source_counts"] = {
            "train": sum(self.train_sizes),
            "test": sum(self.test_sizes),
        }
        self.config["data"]["partition"].update(
            {
                "candidate_train_count": 93,
                "biased_validation_count": 31,
                "oracle_validation_source_count": 31,
            }
        )
        self.config["_provenance"]["canonical_config_sha256"] = (
            canonical_config_sha256(self.config)
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_largest_remainder_is_exact_and_label_stable(self) -> None:
        sizes = {label: size for label, size in enumerate(self.train_sizes)}
        first = largest_remainder_counts(sizes, 31)
        second = largest_remainder_counts(sizes, 31)
        self.assertEqual(first, second)
        self.assertEqual(sum(first.values()), 31)
        for label, count in first.items():
            expected = self.train_sizes[label] * 31 / sum(self.train_sizes)
            self.assertLessEqual(abs(count - expected), 1.0)

    def test_manifest_split_is_exact_reproducible_and_bound(self) -> None:
        before = (self.data_root / "train" / "0" / "0_00000_y0.png").read_bytes()
        first = prepare_decoymnist_manifests(
            self.config, self.root / "first", workers=0
        )
        second = prepare_decoymnist_manifests(
            self.config, self.root / "second", workers=2
        )
        after = (self.data_root / "train" / "0" / "0_00000_y0.png").read_bytes()
        self.assertEqual(before, after)

        expected_counts = {
            "candidate_train": 93,
            "biased_validation": 31,
            "oracle_validation": 31,
            "test": 30,
        }
        frames = {}
        for role, expected_count in expected_counts.items():
            first_path = Path(first["artifacts"][role])
            second_path = Path(second["artifacts"][role])
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            frame = pd.read_csv(first_path)
            frames[role] = frame
            self.assertEqual(len(frame), expected_count)
            self.assertEqual(frame.columns.tolist(), PUBLIC_COLUMNS)
            self.assertTrue(frame["image_sha256"].str.fullmatch(r"[0-9a-f]{64}").all())
            binding = validate_manifest_bundle(self.config, first_path, role)
            self.assertRegex(binding.bundle_sha256, r"^[0-9a-f]{64}$")

        train_ids = set(frames["candidate_train"]["sample_id"])
        val_ids = set(frames["biased_validation"]["sample_id"])
        oracle_ids = set(frames["oracle_validation"]["sample_id"])
        test_ids = set(frames["test"]["sample_id"])
        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & oracle_ids)
        self.assertFalse(val_ids & oracle_ids)
        self.assertFalse((train_ids | val_ids | oracle_ids) & test_ids)
        self.assertEqual(len(train_ids | val_ids | oracle_ids), sum(self.train_sizes))
        self.assertEqual(
            set(frames["candidate_train"]["study_split"]), {"candidate_train"}
        )
        self.assertEqual(
            set(frames["biased_validation"]["study_split"]), {"biased_validation"}
        )
        self.assertNotIn("oracle", " ".join(frames["biased_validation"].columns))
        self.assertNotIn("test", " ".join(frames["biased_validation"].columns))

    def test_bundle_rejects_manifest_tampering(self) -> None:
        result = prepare_decoymnist_manifests(
            self.config, self.root / "manifests", workers=0
        )
        validation_path = Path(result["artifacts"]["biased_validation"])
        frame = pd.read_csv(validation_path)
        frame.loc[0, "label"] = (int(frame.loc[0, "label"]) + 1) % 10
        frame.to_csv(validation_path, index=False)
        with self.assertRaisesRegex(ManifestProvenanceError, "altered"):
            validate_manifest_bundle(self.config, validation_path, "biased_validation")


if __name__ == "__main__":
    unittest.main()
