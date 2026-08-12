import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[2] / "ImageNet9_Runs" / "imagenet9_data.py"
SPEC = importlib.util.spec_from_file_location("imagenet9_data", MODULE_PATH)
DATA = None
IMPORT_ERROR = None
try:
    DATA = importlib.util.module_from_spec(SPEC)
    assert SPEC.loader is not None
    sys.modules[SPEC.name] = DATA
    SPEC.loader.exec_module(DATA)
except (ImportError, OSError) as error:
    IMPORT_ERROR = error
    DATA = None
    sys.modules.pop(SPEC.name, None)


@unittest.skipIf(DATA is None, f"PyTorch/torchvision unavailable: {IMPORT_ERROR}")
class ImageNet9DataTests(unittest.TestCase):
    def test_manifest_loader_transform_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            fields = [
                "split", "sample_id", "label", "class_name", "source_path",
                "imagenet_index", "synset",
            ]
            rows = []
            for label, class_name in enumerate(DATA.CLASS_NAMES):
                for split in ("train", "val"):
                    sample_id = f"n{label:08d}_{split}"
                    path = root / f"{sample_id}.JPEG"
                    Image.new("RGB", (300, 260), color=(label, label, label)).save(path)
                    rows.append(
                        {
                            "split": split,
                            "sample_id": sample_id,
                            "label": label,
                            "class_name": class_name,
                            "source_path": path,
                            "imagenet_index": label,
                            "synset": f"n{label:08d}",
                        }
                    )
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            train = DATA.load_original_samples(manifest, "train")
            val = DATA.load_original_samples(manifest, "val")
            self.assertEqual(DATA.class_counts(train), {name: 1 for name in DATA.CLASS_NAMES})
            self.assertFalse({row.sample_id for row in train} & {row.sample_id for row in val})

            dataset = DATA.ImageNet9Dataset(val, DATA.build_eval_transform())
            item = dataset[0]
            self.assertEqual(tuple(item["image"].shape), (3, 224, 224))
            self.assertEqual(item["sample_id"], val[0].sample_id)

            targets = DATA.torch.arange(DATA.NUM_CLASSES)
            perfect = DATA.classification_metrics(targets, targets)
            self.assertEqual(perfect["accuracy"], 1.0)
            self.assertEqual(perfect["macro_class_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
