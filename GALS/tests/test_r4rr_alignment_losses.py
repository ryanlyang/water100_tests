import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "RightForTheRightRegions/repro_runs/r4rr/train/alignment_losses.py"
)
SPEC = importlib.util.spec_from_file_location("r4rr_alignment_losses", MODULE_PATH)
alignment_losses = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(alignment_losses)


def test_forward_kl_matches_original_r4rr_formula():
    student = torch.tensor([[[0.0, 0.5], [1.0, 0.25]]], requires_grad=True)
    teacher = torch.tensor([[[0.0, 2.0], [1.0, 3.0]]])
    expected = F.kl_div(
        F.log_softmax(student.flatten(1), dim=1),
        teacher.flatten(1) / (teacher.flatten(1).sum(dim=1, keepdim=True) + 1e-8),
        reduction="batchmean",
    )
    actual = alignment_losses.spatial_alignment_loss(student, teacher, "forward_kl")
    torch.testing.assert_close(actual, expected)


def test_all_alignment_losses_are_finite_and_differentiable():
    student = torch.randn(3, 4, 4, requires_grad=True)
    teacher = torch.rand(3, 4, 4)
    for name in alignment_losses.ALIGNMENT_LOSSES:
        loss = alignment_losses.spatial_alignment_loss(student, teacher, name)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        gradient = torch.autograd.grad(loss, student, retain_graph=True)[0]
        assert torch.isfinite(gradient).all()


def test_log_based_losses_handle_zero_teacher_maps():
    student = torch.randn(2, 3, 3, requires_grad=True)
    teacher = torch.zeros_like(student)
    for name in ("reverse_kl", "jensen_shannon"):
        loss = alignment_losses.spatial_alignment_loss(student, teacher, name)
        assert torch.isfinite(loss)
