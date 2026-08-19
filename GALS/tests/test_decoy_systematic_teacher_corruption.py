import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "RightForTheRightRegions"
    / "repro_runs"
    / "r4rr"
    / "ablations"
    / "r4rr_decoy_systematic_teacher_corruption.py"
)
SPEC = importlib.util.spec_from_file_location("decoy_systematic_corruption", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_systematic_condition_selects_only_requested_digit():
    targets = [index % 10 for index in range(100)]
    train_indices = list(range(10, 100))
    selected = MODULE.select_corruption_indices(
        condition="digit_3",
        sample_targets=targets,
        train_indices=train_indices,
        class_to_idx={str(digit): digit for digit in range(10)},
        corruption_seed=0,
    )

    assert selected.tolist() == [13, 23, 33, 43, 53, 63, 73, 83, 93]
    assert all(targets[index] == 3 for index in selected.tolist())


def test_clean_control_selects_no_corrupted_examples():
    selected = MODULE.select_corruption_indices(
        condition="clean",
        sample_targets=[index % 10 for index in range(100)],
        train_indices=list(range(10, 100)),
        class_to_idx={str(digit): digit for digit in range(10)},
        corruption_seed=0,
    )

    assert selected.dtype == np.int64
    assert selected.size == 0


def test_random_control_is_exactly_ten_percent_and_reproducible():
    targets = [index % 10 for index in range(1000)]
    train_indices = list(range(100, 1000))
    kwargs = dict(
        condition="random_10pct",
        sample_targets=targets,
        train_indices=train_indices,
        class_to_idx={str(digit): digit for digit in range(10)},
        corruption_seed=0,
        random_fraction=0.10,
    )
    first = MODULE.select_corruption_indices(**kwargs)
    second = MODULE.select_corruption_indices(**kwargs)

    assert first.size == 90
    assert np.array_equal(first, second)
    assert len(set(first.tolist())) == 90
    assert set(first.tolist()).issubset(set(train_indices))


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
