"""Spatial alignment losses used by R4RR evidence-map training."""

import torch
import torch.nn.functional as F


ALIGNMENT_LOSSES = (
    "forward_kl",
    "reverse_kl",
    "jensen_shannon",
    "squared_l2",
    "cosine",
)

_ALIASES = {
    "kl": "forward_kl",
    "forward-kl": "forward_kl",
    "reverse-kl": "reverse_kl",
    "js": "jensen_shannon",
    "jensen-shannon": "jensen_shannon",
    "l2": "squared_l2",
    "squared-l2": "squared_l2",
    "cosine_distance": "cosine",
}


def normalize_alignment_loss_name(name):
    normalized = str(name).strip().lower()
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in ALIGNMENT_LOSSES:
        choices = ", ".join(ALIGNMENT_LOSSES)
        raise ValueError(f"Unknown alignment loss {name!r}; expected one of: {choices}")
    return normalized


def _teacher_distribution(teacher_flat, eps, smooth_zeros):
    teacher_flat = teacher_flat.clamp_min(0.0)
    teacher_prob = teacher_flat / (teacher_flat.sum(dim=1, keepdim=True) + eps)
    if not smooth_zeros:
        return teacher_prob

    # Reverse KL and JS require log(q). Epsilon smoothing also turns a rare
    # all-zero teacher map into a valid uniform spatial distribution.
    teacher_prob = teacher_prob.clamp_min(eps)
    return teacher_prob / teacher_prob.sum(dim=1, keepdim=True)


def spatial_alignment_loss(student_maps, teacher_maps, loss_name="forward_kl", eps=1e-8):
    """Compare flattened student and teacher evidence maps over spatial pixels.

    The student map is interpreted as logits, matching the original R4RR
    forward-KL implementation. The teacher map is nonnegative and normalized
    per image. All returned values are means over the batch.
    """
    if student_maps.shape != teacher_maps.shape:
        raise ValueError(
            f"Student and teacher maps must have equal shapes; got "
            f"{tuple(student_maps.shape)} and {tuple(teacher_maps.shape)}"
        )
    if student_maps.ndim < 2:
        raise ValueError(f"Expected batched spatial maps, got shape {tuple(student_maps.shape)}")

    loss_name = normalize_alignment_loss_name(loss_name)
    student_flat = student_maps.flatten(start_dim=1)
    teacher_flat = teacher_maps.flatten(start_dim=1).to(
        device=student_flat.device, dtype=student_flat.dtype
    )
    student_log_prob = F.log_softmax(student_flat, dim=1)

    if loss_name == "forward_kl":
        # Preserve the original R4RR formula exactly: KL(teacher || student).
        teacher_prob = _teacher_distribution(teacher_flat, eps, smooth_zeros=False)
        return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")

    student_prob = student_log_prob.exp()
    teacher_prob = _teacher_distribution(teacher_flat, eps, smooth_zeros=True)

    if loss_name == "reverse_kl":
        return (
            student_prob * (student_log_prob - teacher_prob.log())
        ).sum(dim=1).mean()

    if loss_name == "jensen_shannon":
        midpoint = 0.5 * (student_prob + teacher_prob)
        teacher_term = teacher_prob * (teacher_prob.log() - midpoint.log())
        student_term = student_prob * (student_log_prob - midpoint.log())
        return 0.5 * (teacher_term.sum(dim=1) + student_term.sum(dim=1)).mean()

    if loss_name == "squared_l2":
        return (student_prob - teacher_prob).square().sum(dim=1).mean()

    if loss_name == "cosine":
        similarity = F.cosine_similarity(student_prob, teacher_prob, dim=1, eps=eps)
        return (1.0 - similarity).mean()

    raise AssertionError(f"Unhandled alignment loss: {loss_name}")
