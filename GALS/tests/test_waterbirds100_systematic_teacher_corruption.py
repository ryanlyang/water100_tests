import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch


# The lightweight local Python 3.8 environment does not include pandas. These
# tests exercise pure selection, manifest, and inversion helpers only.
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
    / "r4rr_waterbirds100_systematic_teacher_corruption.py"
)
SPEC = importlib.util.spec_from_file_location("waterbirds100_systematic_corruption", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _labels_with_exact_training_counts():
    return np.concatenate(
        [
            np.full(count, label, dtype=np.int64)
            for label, count in enumerate(MODULE.EXPECTED_CLASS_COUNTS["train"])
        ]
    )


def test_systematic_conditions_select_all_and_only_requested_class():
    labels = _labels_with_exact_training_counts()
    for class_index, class_key in enumerate(MODULE.CLASS_KEYS):
        selected = MODULE.select_corruption_indices(
            condition=f"class_{class_key}",
            labels=labels,
            corruption_seed=0,
            matched_class_counts=MODULE.EXPECTED_CLASS_COUNTS["train"],
        )

        assert selected.size == MODULE.EXPECTED_CLASS_COUNTS["train"][class_index]
        assert np.all(labels[selected] == class_index)


def test_random_controls_are_exactly_count_matched_and_reproducible():
    labels = _labels_with_exact_training_counts()
    for class_index, class_key in enumerate(MODULE.CLASS_KEYS):
        kwargs = dict(
            condition=f"random_matched_{class_key}",
            labels=labels,
            corruption_seed=0,
            matched_class_counts=MODULE.EXPECTED_CLASS_COUNTS["train"],
        )
        first = MODULE.select_corruption_indices(**kwargs)
        second = MODULE.select_corruption_indices(**kwargs)

        assert first.size == MODULE.EXPECTED_CLASS_COUNTS["train"][class_index]
        assert np.array_equal(first, second)
        assert np.unique(first).size == first.size
        assert int(first.min()) >= 0
        assert int(first.max()) < labels.size


class _FakeWaterbirdsSplit:
    def __init__(self, split):
        self.labels = []
        self.places = []
        self.paths = []
        counts = MODULE.EXPECTED_GROUP_COUNTS[split]
        for group, count in enumerate(counts):
            label = group // 2
            place = group % 2
            for index in range(count):
                self.labels.append(label)
                self.places.append(place)
                self.paths.append(
                    f"/tmp/waterbirds100/{split}/{MODULE.GROUP_KEYS[group]}/{index:05d}.jpg"
                )
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.places = np.asarray(self.places, dtype=np.int64)

    def __len__(self):
        return len(self.paths)


def _fake_splits():
    return {
        split: _FakeWaterbirdsSplit(split)
        for split in ("train", "val", "test")
    }


def test_systematic_manifest_is_class_pure_and_uses_exact_count():
    manifest, selected, sample_ids = MODULE.build_manifest(
        condition="class_waterbird",
        datasets=_fake_splits(),
        data_root=Path("/tmp/waterbirds100"),
        corruption_seed=0,
    )

    assert selected.size == 1111
    assert len(sample_ids) == 1111
    assert manifest["target_class_name"] == "Waterbird"
    assert manifest["corrupted_class_counts"] == {
        "Landbird": 0,
        "Waterbird": 1111,
    }
    assert manifest["corrupted_group_counts"] == {
        "Land_on_Land": 0,
        "Land_on_Water": 0,
        "Water_on_Land": 0,
        "Water_on_Water": 1111,
    }


def test_random_manifest_matches_corresponding_systematic_count():
    manifest, selected, sample_ids = MODULE.build_manifest(
        condition="random_matched_landbird",
        datasets=_fake_splits(),
        data_root=Path("/tmp/waterbirds100"),
        corruption_seed=0,
    )

    assert selected.size == 3684
    assert len(sample_ids) == 3684
    assert manifest["matched_class_count"] == 3684
    assert sum(manifest["corrupted_class_counts"].values()) == 3684
    assert manifest["condition_type"] == "matched_random_control"


def test_locked_optimized_hparams_are_loaded_from_shared_config():
    config = (
        Path(__file__).resolve().parents[1]
        / "RightForTheRightRegions"
        / "configs"
        / "r4rr_optimized_hparams.yaml"
    )
    observed = MODULE.load_optimized_hparams(config)

    assert observed == {
        "attention_epoch": 73,
        "kl_lambda": 495.61,
        "base_lr": 5.72e-5,
        "classifier_lr": 3.57e-3,
        "lr2_mult": 0.123,
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
