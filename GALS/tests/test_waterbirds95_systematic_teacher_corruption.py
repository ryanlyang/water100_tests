import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch


# The lightweight local Python 3.8 environment does not include pandas. The
# tests below exercise selection, manifest, and inversion helpers only.
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
    / "r4rr_waterbirds95_systematic_teacher_corruption.py"
)
SPEC = importlib.util.spec_from_file_location("waterbirds95_systematic_corruption", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _groups_with_exact_training_counts():
    return np.concatenate(
        [
            np.full(count, group, dtype=np.int64)
            for group, count in enumerate(MODULE.EXPECTED_GROUP_COUNTS["train"])
        ]
    )


def test_systematic_conditions_select_all_and_only_requested_group():
    groups = _groups_with_exact_training_counts()
    for group_index, group_key in enumerate(MODULE.GROUP_KEYS):
        selected = MODULE.select_corruption_indices(
            condition=f"group_{group_key}",
            groups=groups,
            corruption_seed=0,
            matched_group_counts=MODULE.EXPECTED_GROUP_COUNTS["train"],
        )

        assert selected.size == MODULE.EXPECTED_GROUP_COUNTS["train"][group_index]
        assert np.all(groups[selected] == group_index)


def test_random_controls_are_exactly_count_matched_and_reproducible():
    groups = _groups_with_exact_training_counts()
    for group_index, group_key in enumerate(MODULE.GROUP_KEYS):
        kwargs = dict(
            condition=f"random_matched_{group_key}",
            groups=groups,
            corruption_seed=0,
            matched_group_counts=MODULE.EXPECTED_GROUP_COUNTS["train"],
        )
        first = MODULE.select_corruption_indices(**kwargs)
        second = MODULE.select_corruption_indices(**kwargs)

        assert first.size == MODULE.EXPECTED_GROUP_COUNTS["train"][group_index]
        assert np.array_equal(first, second)
        assert np.unique(first).size == first.size
        assert int(first.min()) >= 0
        assert int(first.max()) < groups.size


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
                    f"/tmp/waterbirds95/{split}/{MODULE.GROUP_KEYS[group]}/{index:05d}.jpg"
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


def test_systematic_manifest_is_group_pure_and_uses_exact_count():
    manifest, selected, sample_ids = MODULE.build_manifest(
        condition="group_water_on_land",
        datasets=_fake_splits(),
        data_root=Path("/tmp/waterbirds95"),
        corruption_seed=0,
    )

    assert selected.size == 56
    assert len(sample_ids) == 56
    assert manifest["target_group_name"] == "Water_on_Land"
    assert manifest["corrupted_group_counts"] == {
        "Land_on_Land": 0,
        "Land_on_Water": 0,
        "Water_on_Land": 56,
        "Water_on_Water": 0,
    }


def test_random_manifest_matches_corresponding_systematic_count():
    manifest, selected, sample_ids = MODULE.build_manifest(
        condition="random_matched_water_on_water",
        datasets=_fake_splits(),
        data_root=Path("/tmp/waterbirds95"),
        corruption_seed=0,
    )

    assert selected.size == 1057
    assert len(sample_ids) == 1057
    assert manifest["matched_group_count"] == 1057
    assert sum(manifest["corrupted_group_counts"].values()) == 1057
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
        "attention_epoch": 109,
        "kl_lambda": 295.30,
        "base_lr": 4.82e-5,
        "classifier_lr": 2.93e-3,
        "lr2_mult": 0.409,
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
