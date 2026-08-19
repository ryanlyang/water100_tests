import importlib.util
import importlib.machinery
import sys
import types
from pathlib import Path

import numpy as np
import torch


# The lightweight local Python 3.8 compatibility environment does not include
# pandas. These unit tests exercise pure selection/inversion helpers only; the
# research-compute environment supplies real pandas for metadata loading.
try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.__spec__ = importlib.machinery.ModuleSpec("pandas", loader=None)
    sys.modules["pandas"] = pandas_stub


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "RightForTheRightRegions"
    / "repro_runs"
    / "r4rr"
    / "ablations"
    / "r4rr_redmeat_systematic_teacher_corruption.py"
)
SPEC = importlib.util.spec_from_file_location("redmeat_systematic_corruption", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_systematic_condition_selects_all_and_only_requested_class():
    labels = ["prime_rib", "steak", "prime_rib", "pork_chop", "prime_rib"]
    selected = MODULE.select_corruption_indices(
        condition="class_prime_rib",
        label_names=labels,
        corruption_seed=0,
    )

    assert selected.tolist() == [0, 2, 4]
    assert all(labels[index] == "prime_rib" for index in selected.tolist())


def test_random_control_is_exactly_count_matched_and_reproducible():
    labels = [MODULE.CLASS_ORDER[index % 5] for index in range(2500)]
    first = MODULE.select_corruption_indices(
        condition="random_20pct",
        label_names=labels,
        corruption_seed=0,
        random_count=500,
    )
    second = MODULE.select_corruption_indices(
        condition="random_20pct",
        label_names=labels,
        corruption_seed=0,
        random_count=500,
    )

    assert first.size == 500
    assert np.array_equal(first, second)
    assert len(set(first.tolist())) == 500
    assert int(first.min()) >= 0
    assert int(first.max()) < 2500


class _FakeRedMeatSplit:
    def __init__(self, split, per_class):
        self.classes = list(MODULE.CLASS_ORDER)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.label_names = []
        self.labels = []
        self.paths = []
        for class_name in self.classes:
            for index in range(per_class):
                self.label_names.append(class_name)
                self.labels.append(self.class_to_idx[class_name])
                self.paths.append(f"/tmp/redmeat/{split}/{class_name}/{index:04d}.jpg")
        self.labels = np.asarray(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.paths)


def test_manifest_is_exactly_class_and_count_matched():
    datasets = {
        "train": _FakeRedMeatSplit("train", 500),
        "val": _FakeRedMeatSplit("val", 250),
        "test": _FakeRedMeatSplit("test", 250),
    }
    manifest, selected, sample_ids = MODULE.build_manifest(
        condition="class_filet_mignon",
        datasets=datasets,
        data_root=Path("/tmp/redmeat"),
        corruption_seed=0,
        random_count=500,
    )

    assert selected.size == 500
    assert len(sample_ids) == 500
    assert manifest["corrupted_fraction_of_training"] == 0.2
    assert manifest["corrupted_class_counts"] == {
        class_name: (500 if class_name == "filet_mignon" else 0)
        for class_name in MODULE.CLASS_ORDER
    }


def test_locked_optimized_hparams_are_loaded_from_shared_config():
    config = (
        Path(__file__).resolve().parents[1]
        / "RightForTheRightRegions"
        / "configs"
        / "r4rr_optimized_hparams.yaml"
    )
    observed = MODULE.load_optimized_hparams(config)

    assert observed == {
        "attention_epoch": 2,
        "kl_lambda": 11.44,
        "base_lr": 2.40e-3,
        "classifier_lr": 2.33e-4,
        "lr2_mult": 1.567,
    }


def test_inversion_matches_one_minus_and_sum_normalization():
    mask = torch.tensor([[[0.0, 0.25], [0.5, 1.0]]], dtype=torch.float32)
    observed = MODULE.invert_and_renormalize_mask(mask)
    expected = 1.0 - mask
    expected = expected / expected.sum()

    assert torch.allclose(observed, expected)
    assert torch.isclose(observed.sum(), torch.tensor(1.0))


def test_inversion_uses_uniform_fallback_for_all_one_mask():
    mask = torch.ones((1, 2, 2), dtype=torch.float32)
    observed = MODULE.invert_and_renormalize_mask(mask)

    assert torch.allclose(observed, torch.full_like(mask, 0.25))
