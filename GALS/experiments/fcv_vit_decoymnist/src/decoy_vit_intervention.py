"""Step-7 raw-patch FCV intervention for the locked timm ViT-S/16."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from decoy_teacher_masks import CATEGORY_BACKGROUND, CATEGORY_EVIDENCE


class ViTInterventionError(ValueError):
    """Raised when raw-token forwarding or spatial replacement is invalid."""


RECONSTRUCTION_TOLERANCE = 1.0e-5
REQUIRED_VIT_ATTRIBUTES = (
    "patch_embed",
    "_pos_embed",
    "patch_drop",
    "norm_pre",
    "blocks",
    "norm",
    "forward_head",
)


@dataclass(frozen=True)
class ReplacementAudit:
    replaced_patch_count: int
    preserved_token_max_abs_error: float
    donor_token_max_abs_error: float
    changed_token_count: int
    replacement_delta_mean: float
    replacement_delta_max: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_vit(model: Any) -> None:
    missing = [name for name in REQUIRED_VIT_ATTRIBUTES if not hasattr(model, name)]
    if missing:
        raise ViTInterventionError(
            f"FCV requires a timm VisionTransformer; missing {missing}."
        )
    if not hasattr(model.patch_embed, "num_patches"):
        raise ViTInterventionError("model.patch_embed has no num_patches.")


def _validate_tokens(model: Any, tokens: Any) -> None:
    import torch

    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        shape = tuple(tokens.shape) if isinstance(tokens, torch.Tensor) else None
        raise ViTInterventionError(f"Raw tokens must have shape [B,N,D], found {shape}.")
    if int(tokens.shape[1]) != int(model.patch_embed.num_patches):
        raise ViTInterventionError("Raw token patch count differs from the model.")
    if not torch.is_floating_point(tokens):
        raise ViTInterventionError("Raw patch tokens must be floating point.")


def extract_raw_patch_tokens(model: Any, images: Any):
    import torch

    _validate_vit(model)
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ViTInterventionError("Images must have shape [B,C,H,W].")
    tokens = model.patch_embed(images)
    _validate_tokens(model, tokens)
    return tokens


def forward_from_raw_patch_tokens(model: Any, raw_patch_tokens: Any):
    """Resume the native timm forward so donor content receives target positions."""

    import torch

    _validate_vit(model)
    _validate_tokens(model, raw_patch_tokens)
    values = model._pos_embed(raw_patch_tokens)
    values = model.patch_drop(values)
    values = model.norm_pre(values)
    values = model.blocks(values)
    values = model.norm(values)
    logits = model.forward_head(values)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ViTInterventionError("Reconstructed forward did not return [B,C] logits.")
    return logits


def verify_identity_forward(
    model: Any,
    images: Any,
    *,
    tolerance: float = RECONSTRUCTION_TOLERANCE,
) -> Dict[str, Any]:
    """Require the no-intervention raw-token path to reproduce native logits."""

    import torch

    if model.training:
        raise ViTInterventionError("Identity verification requires model.eval().")
    with torch.inference_mode():
        native = model(images)
        tokens = extract_raw_patch_tokens(model, images)
        resumed = forward_from_raw_patch_tokens(model, tokens)
    if native.shape != resumed.shape:
        raise ViTInterventionError("Native and resumed logits have different shapes.")
    error = (native.float() - resumed.float()).abs()
    maximum = float(error.max().item())
    mean = float(error.mean().item())
    if not np.isfinite(maximum) or maximum >= float(tolerance):
        raise ViTInterventionError(
            f"Raw-token identity forward failed: max_abs_error={maximum:.8g}."
        )
    return {
        "max_abs_error": maximum,
        "mean_abs_error": mean,
        "tolerance": float(tolerance),
        "passed": True,
    }


def mutually_safe_positions(
    target_categories: np.ndarray,
    donor_categories: np.ndarray,
    *,
    category: int = CATEGORY_BACKGROUND,
) -> np.ndarray:
    target = np.asarray(target_categories, dtype=np.uint8).reshape(-1)
    donor = np.asarray(donor_categories, dtype=np.uint8).reshape(-1)
    if target.shape != donor.shape or target.size == 0:
        raise ViTInterventionError("Target and donor masks must have equal nonempty shape.")
    if int(category) not in (CATEGORY_BACKGROUND, CATEGORY_EVIDENCE):
        raise ViTInterventionError("Only background or evidence positions may be swapped.")
    return np.flatnonzero((target == int(category)) & (donor == int(category))).astype(
        np.int64
    )


def require_positions_present(
    replacement_positions: Iterable[int], required_positions: Iterable[int]
) -> None:
    replacement = {int(value) for value in replacement_positions}
    required = {int(value) for value in required_positions}
    missing = sorted(required - replacement)
    if missing:
        raise ViTInterventionError(
            f"FCV replacement excludes required decoy patch cells: {missing}."
        )


def replace_spatially_aligned_tokens(
    target_tokens: Any,
    donor_tokens: Any,
    positions: Iterable[int],
):
    """Replace donor content at identical patch indices and audit the result."""

    import torch

    if not isinstance(target_tokens, torch.Tensor) or not isinstance(
        donor_tokens, torch.Tensor
    ):
        raise ViTInterventionError("Target and donor tokens must be tensors.")
    if target_tokens.ndim != 2 or donor_tokens.ndim != 2:
        raise ViTInterventionError("Single-image tokens must have shape [N,D].")
    if target_tokens.shape != donor_tokens.shape:
        raise ViTInterventionError("Target and donor token shapes differ.")
    indices = torch.as_tensor(
        list(int(value) for value in positions),
        dtype=torch.long,
        device=target_tokens.device,
    )
    if indices.numel() == 0:
        raise ViTInterventionError("FCV cannot perform an empty intervention.")
    if int(indices.min()) < 0 or int(indices.max()) >= int(target_tokens.shape[0]):
        raise ViTInterventionError("Replacement position is outside the patch grid.")
    if int(torch.unique(indices).numel()) != int(indices.numel()):
        raise ViTInterventionError("Replacement positions must be unique.")
    donor = donor_tokens.to(device=target_tokens.device, dtype=target_tokens.dtype)
    result = target_tokens.clone()
    result.index_copy_(0, indices, donor.index_select(0, indices))

    replaced = torch.zeros(target_tokens.shape[0], dtype=torch.bool, device=target_tokens.device)
    replaced[indices] = True
    preserved_error = float(
        (result[~replaced].float() - target_tokens[~replaced].float()).abs().max().item()
    ) if bool((~replaced).any()) else 0.0
    donor_error = float(
        (result[replaced].float() - donor[replaced].float()).abs().max().item()
    )
    deltas = (result[replaced].float() - target_tokens[replaced].float()).abs()
    per_token_delta = deltas.flatten(1).amax(dim=1)
    audit = ReplacementAudit(
        replaced_patch_count=int(indices.numel()),
        preserved_token_max_abs_error=preserved_error,
        donor_token_max_abs_error=donor_error,
        changed_token_count=int((per_token_delta > 0).sum().item()),
        replacement_delta_mean=float(deltas.mean().item()),
        replacement_delta_max=float(deltas.max().item()),
    )
    if preserved_error != 0.0 or donor_error != 0.0:
        raise ViTInterventionError("Spatial replacement failed its token-integrity audit.")
    return result, audit
