from __future__ import annotations

import sys
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.manifest_provenance import (  # noqa: E402
    ManifestProvenanceError,
    sha256_file,
    validate_manifest_bundle,
)
from fcv.waterbirds_metadata import prepare_waterbirds100_manifests  # noqa: E402


class WaterbirdsMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "waterbirds100"
        self.teacher_root = self.root / "teacher_maps"
        self.data_root.mkdir()
        self.teacher_root.mkdir()

        rows = []
        metadata_index = 0
        split_specs = {
            0: [(0, 0)] * 10 + [(1, 1)] * 10,
            1: [(0, 0), (0, 1), (1, 0), (1, 1)] * 2,
            2: [(0, 0), (0, 1), (1, 0), (1, 1)] * 2,
        }
        for split, label_context_pairs in split_specs.items():
            for label, context in label_context_pairs:
                folder = f"{metadata_index + 1:03d}.Synthetic_Bird"
                basename = f"Synthetic_Bird_{metadata_index:04d}.jpg"
                # Exercise the producer's complete relative-path flattening.
                # The old fixture had only one directory level, for which the
                # incomplete parent+basename resolver happened to work.
                relative = f"nested.images/{folder}/{basename}"
                image_path = self.data_root / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.touch()
                relative_without_extension = str(Path(relative).with_suffix(""))
                flattened = relative_without_extension.replace("/", "_")
                image_id = re.sub(r"[^A-Za-z0-9_-]+", "_", flattened).strip("_")
                # The real producer's eval list is metadata split 0 + 1. Test
                # masks are absent by contract unless generated separately.
                if split in {0, 1}:
                    (self.teacher_root / f"{image_id}.png").touch()
                rows.append(
                    {
                        "img_filename": relative,
                        "y": label,
                        "place": context,
                        "split": split,
                    }
                )
                metadata_index += 1

        self.metadata_path = self.data_root / "metadata.csv"
        pd.DataFrame(rows).to_csv(self.metadata_path, index=False)

        base_config_path = (
            EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
        )
        with base_config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["paths"]["data_root"] = str(self.data_root)
        config["paths"]["metadata"] = str(self.metadata_path)
        config["paths"]["teacher_map_root"] = str(self.teacher_root)
        config["paths"]["output_root"] = str(self.root / "outputs")
        self.config_path = self.root / "config.yaml"
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_split_is_stratified_disjoint_and_leakage_safe(self) -> None:
        config = load_and_validate_config(self.config_path)
        result = prepare_waterbirds100_manifests(
            config,
            self.root / "manifests",
            overwrite=False,
        )

        train = pd.read_csv(result["artifacts"]["candidate_train"])
        holdout = pd.read_csv(result["artifacts"]["biased_validation"])
        oracle = pd.read_csv(result["artifacts"]["oracle_validation"])
        test = pd.read_csv(result["artifacts"]["test"])
        self.assertTrue(Path(result["artifacts"]["bundle"]).is_file())
        for key in ("candidate_train", "biased_validation", "oracle_validation", "test"):
            binding = validate_manifest_bundle(config, result["artifacts"][key], key)
            self.assertRegex(binding.bundle_sha256, r"^[0-9a-f]{64}$")

        self.assertEqual(len(train), 16)
        self.assertEqual(len(holdout), 4)
        self.assertEqual(train["label"].value_counts().to_dict(), {0: 8, 1: 8})
        self.assertEqual(holdout["label"].value_counts().to_dict(), {0: 2, 1: 2})
        self.assertFalse(set(train["metadata_index"]) & set(holdout["metadata_index"]))

        for public in (train, holdout):
            self.assertNotIn("context", public.columns)
            self.assertNotIn("group", public.columns)
            self.assertNotIn("group_name", public.columns)
            self.assertIn("image_sha256", public.columns)
            self.assertTrue(public["image_sha256"].str.fullmatch(r"[0-9a-f]{64}").all())

        for analysis_only in (oracle, test):
            self.assertIn("context", analysis_only.columns)
            self.assertIn("group", analysis_only.columns)
            self.assertIn("group_name", analysis_only.columns)
            self.assertEqual(set(analysis_only["group"]), {0, 1, 2, 3})

        self.assertTrue(holdout["teacher_map_exists"].all())
        self.assertTrue(oracle["teacher_map_exists"].all())
        self.assertFalse(test["teacher_map_exists"].any())
        self.assertEqual(
            result["summary"]["splits"]["biased_validation"]["aligned_fraction"],
            1.0,
        )
        for split in result["summary"]["splits"].values():
            self.assertRegex(split["image_set_sha256"], r"^[0-9a-f]{64}$")

    def test_ambiguous_real_and_legacy_layouts_fail_closed(self) -> None:
        config = load_and_validate_config(self.config_path)
        metadata = pd.read_csv(self.metadata_path)
        row = metadata.loc[metadata["split"] == 0].iloc[0]
        relative = Path(row["img_filename"])
        legacy_name = (
            f"{relative.parent.name.replace('.', '_')}_{relative.stem}.png"
        )
        # The producer-flat map already exists from setUp. Adding a distinct
        # legacy-layout map must be diagnosed instead of silently choosing one.
        (self.teacher_root / legacy_name).touch()
        with self.assertRaisesRegex(ValueError, "Ambiguous teacher maps"):
            prepare_waterbirds100_manifests(
                config,
                self.root / "ambiguous_manifests",
            )

    def test_split_indices_are_reproducible(self) -> None:
        config = load_and_validate_config(self.config_path)
        first = prepare_waterbirds100_manifests(config, self.root / "first")
        second = prepare_waterbirds100_manifests(config, self.root / "second")
        first_val = pd.read_csv(first["artifacts"]["biased_validation"])
        second_val = pd.read_csv(second["artifacts"]["biased_validation"])
        self.assertEqual(
            first_val["metadata_index"].tolist(), second_val["metadata_index"].tolist()
        )

    def test_bundle_rejects_coherently_relabelled_test_rows_as_oracle(self) -> None:
        config = load_and_validate_config(self.config_path)
        result = prepare_waterbirds100_manifests(config, self.root / "manifests")
        oracle_path = Path(result["artifacts"]["oracle_validation"])
        forged = pd.read_csv(result["artifacts"]["test"])
        forged["source_split"] = "original_validation"
        forged["study_split"] = "oracle_validation_analysis_only"
        forged.to_csv(oracle_path, index=False)

        bundle_path = Path(result["artifacts"]["bundle"])
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        indices = sorted(forged["metadata_index"].astype(int).tolist())
        index_payload = ",".join(str(value) for value in indices).encode("ascii")
        bundle["manifests"]["oracle_validation"].update(
            {
                "sha256": sha256_file(oracle_path),
                "row_count": len(forged),
                "metadata_indices_sha256": hashlib.sha256(index_payload).hexdigest(),
            }
        )
        bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
        with self.assertRaisesRegex(ManifestProvenanceError, "wrong original split"):
            validate_manifest_bundle(config, oracle_path, "oracle_validation")


if __name__ == "__main__":
    unittest.main()
