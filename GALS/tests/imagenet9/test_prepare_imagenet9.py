import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "ImageNet9_Runs" / "prepare_imagenet9.py"
SPEC = importlib.util.spec_from_file_location("prepare_imagenet9", MODULE_PATH)
PREP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PREP
SPEC.loader.exec_module(PREP)


class PrepareImageNet9Tests(unittest.TestCase):
    def test_official_mapping_integrity(self):
        mapping_path = MODULE_PATH.parent / "assets" / "in_to_in9.json"
        mapping = PREP.load_mapping(mapping_path)
        self.assertEqual(len(mapping), 1000)
        self.assertEqual(
            tuple(sum(value == label for value in mapping.values()) for label in range(9)),
            PREP.EXPECTED_SUBCLASS_COUNTS,
        )

    def test_source_id_extraction(self):
        self.assertEqual(
            PREP.imagenet_ids_in_name(
                "fg_n02497673_05944_bg_n02536864_02721.JPEG"
            ),
            ("n02497673_05944", "n02536864_02721"),
        )

    def test_split_selection_is_balanced_deterministic_and_disjoint(self):
        candidates = []
        for label in range(9):
            for index in range(5):
                candidates.append(
                    PREP.Candidate(
                        sample_id=f"n{label:08d}_{index:05d}",
                        label=label,
                        class_name=PREP.CLASS_NAMES[label],
                        class_dir=PREP.CLASS_DIRS[label],
                        imagenet_index=label,
                        synset=f"n{label:08d}",
                        source_path=f"/images/{label}/{index}.JPEG",
                        annotation_path=f"/annotations/{label}/{index}.xml",
                        image_width=100,
                        image_height=100,
                        bbox_xmin=1,
                        bbox_ymin=2,
                        bbox_xmax=90,
                        bbox_ymax=91,
                    )
                )

        first, counts = PREP.select_splits(candidates, seed=7, train_per_class=3, val_per_class=1)
        second, _ = PREP.select_splits(candidates, seed=7, train_per_class=3, val_per_class=1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 36)
        self.assertTrue(all(value["train"] == 3 for value in counts.values()))
        self.assertTrue(all(value["val"] == 1 for value in counts.values()))
        train_ids = {item.sample_id for split, _rank, item in first if split == "train"}
        val_ids = {item.sample_id for split, _rank, item in first if split == "val"}
        self.assertFalse(train_ids & val_ids)

    def test_annotation_filtering_and_link_materialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_root = root / "train"
            annotation_root = root / "annotations"
            synset = "n00000001"
            sample_id = f"{synset}_00001"
            source = train_root / synset / f"{sample_id}.JPEG"
            annotation = annotation_root / f"{sample_id}.xml"
            source.parent.mkdir(parents=True)
            annotation.parent.mkdir(parents=True)
            source.write_bytes(b"fixture")
            annotation.write_text(
                "<annotation>"
                f"<filename>{sample_id}</filename>"
                "<size><width>100</width><height>80</height></size>"
                "<object><bndbox>"
                "<xmin>2</xmin><ymin>3</ymin><xmax>90</xmax><ymax>70</ymax>"
                "</bndbox></object>"
                "</annotation>"
            )

            candidate, rejection = PREP.parse_annotation(
                annotation,
                imagenet_index=0,
                synset=synset,
                label=0,
                train_root=train_root,
                official_test_ids=set(),
            )
            self.assertIsNone(rejection)
            self.assertIsNotNone(candidate)

            link_root = root / "links"
            PREP.materialize_links(link_root, [("train", 0, candidate)], overwrite=False)
            link = link_root / "train" / PREP.CLASS_DIRS[0] / source.name
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), source.resolve())

            excluded, rejection = PREP.parse_annotation(
                annotation,
                imagenet_index=0,
                synset=synset,
                label=0,
                train_root=train_root,
                official_test_ids={sample_id},
            )
            self.assertIsNone(excluded)
            self.assertEqual(rejection.reason, "official_test_overlap")

    def test_full_fixture_build_writes_manifests_and_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            imagenet_root = root / "imagenet"
            train_root = imagenet_root / "train"
            annotation_root = root / "annotations"
            official_root = root / "bg_challenge"
            output_root = root / "output"

            synsets = [f"n{index:08d}" for index in range(1000)]
            for synset in synsets:
                (train_root / synset).mkdir(parents=True)
            for variant in PREP.OFFICIAL_VARIANTS:
                for class_dir in PREP.CLASS_DIRS:
                    (official_root / variant / "val" / class_dir).mkdir(parents=True)

            mapping = PREP.load_mapping(MODULE_PATH.parent / "assets" / "in_to_in9.json")
            indices_by_label = {
                label: [index for index, value in mapping.items() if value == label][:2]
                for label in range(9)
            }
            for label, indices in indices_by_label.items():
                for ordinal, imagenet_index in enumerate(indices):
                    synset = synsets[imagenet_index]
                    sample_id = f"{synset}_{ordinal + 1:05d}"
                    source = train_root / synset / f"{sample_id}.JPEG"
                    annotation = annotation_root / synset / f"{sample_id}.xml"
                    source.write_bytes(b"fixture")
                    annotation.parent.mkdir(parents=True, exist_ok=True)
                    annotation.write_text(
                        "<annotation>"
                        f"<filename>{sample_id}</filename>"
                        "<size><width>100</width><height>80</height></size>"
                        "<object><bndbox>"
                        "<xmin>2</xmin><ymin>3</ymin><xmax>90</xmax><ymax>70</ymax>"
                        "</bndbox></object>"
                        "</annotation>"
                    )

            with contextlib.redirect_stdout(io.StringIO()):
                result = PREP.main(
                    [
                        "--imagenet-root", str(imagenet_root),
                        "--annotation-root", str(annotation_root),
                        "--official-test-root", str(official_root),
                        "--output-root", str(output_root),
                        "--train-per-class", "1",
                        "--val-per-class", "1",
                        "--allow-nonstandard-test-counts",
                        "--materialize-links",
                    ]
                )

            self.assertEqual(result, 0)
            metadata = output_root / "metadata" / "reconstructed_original_bbox1_v1"
            manifest_lines = (metadata / "manifest.csv").read_text().splitlines()
            self.assertEqual(len(manifest_lines), 19)
            summary = json.loads((metadata / "summary.json").read_text())
            self.assertEqual(summary["train_total"], 9)
            self.assertEqual(summary["val_total"], 9)
            links = list((output_root / "train_source").rglob("*.JPEG"))
            self.assertEqual(len(links), 18)
            self.assertTrue(all(path.is_symlink() for path in links))


if __name__ == "__main__":
    unittest.main()
