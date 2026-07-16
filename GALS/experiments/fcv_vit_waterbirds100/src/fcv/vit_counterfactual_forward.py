"""Raw-patch extraction and resumed forwards for timm Vision Transformers.

FCV swaps patch *content* before positional embeddings are applied.  This
module deliberately delegates all model-specific token/position handling to
the timm VisionTransformer methods used by its normal forward path.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn

from .candidate_training import (
    CandidateTrainingError,
    build_model,
    candidate_training_fingerprint,
    software_versions,
    source_tree_provenance,
)


class ViTPatchForwardError(ValueError):
    """Raised when a model or token tensor violates the Step 5 contract."""


RECONSTRUCTION_TOLERANCE = 1.0e-5


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reconstruction_gate(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Require stored successful checks for the locked model and a real candidate."""

    preflight = Path(config["paths"]["output_root"]) / "preflight"
    reports = {
        "pretrained": preflight / "reconstruction_pretrained.json",
        "candidate": preflight / "reconstruction_candidate.json",
    }
    validated: Dict[str, Any] = {}
    for kind, path in reports.items():
        if not path.is_file():
            raise ViTPatchForwardError(
                f"Missing mandatory {kind} reconstruction report: {path}"
            )
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        max_abs_error = float(report.get("max_abs_error", float("inf")))
        mean_abs_error = float(report.get("mean_abs_error", float("inf")))
        if (
            report.get("artifact_type") != "fcv_vit_reconstruction_report"
            or report.get("schema_version") != 1
            or report.get("status") != "passed"
            or report.get("model") != config["model"]["name"]
            or report.get("training_fingerprint")
            != candidate_training_fingerprint(config)
            or float(report.get("tolerance", 0.0)) != RECONSTRUCTION_TOLERANCE
            or not math.isfinite(max_abs_error)
            or not math.isfinite(mean_abs_error)
            or max_abs_error >= RECONSTRUCTION_TOLERANCE
        ):
            raise ViTPatchForwardError(f"Invalid reconstruction report: {path}")
        if kind == "pretrained":
            if (
                report.get("source") != "pretrained"
                or report.get("checkpoint_path") is not None
                or report.get("checkpoint_sha256") is not None
            ):
                raise ViTPatchForwardError(
                    "Pretrained reconstruction report did not use locked weights."
                )
        else:
            checkpoint_path = Path(str(report.get("checkpoint_path", "")))
            checkpoint_sha256 = str(report.get("checkpoint_sha256", ""))
            if (
                not report.get("candidate_id")
                or not checkpoint_path.is_file()
                or len(checkpoint_sha256) != 64
                or _sha256_file(checkpoint_path) != checkpoint_sha256
            ):
                raise ViTPatchForwardError(
                    "The real-candidate reconstruction checkpoint has changed."
                )
        validated[kind] = {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
        }
    return validated


REQUIRED_TIMM_VIT_ATTRIBUTES = (
    "patch_embed",
    "_pos_embed",
    "patch_drop",
    "norm_pre",
    "blocks",
    "norm",
    "forward_head",
)


@dataclass(frozen=True)
class ReconstructionReport:
    """Numerical equivalence result for the normal and resumed forwards."""

    normal_logits_shape: tuple[int, ...]
    raw_patch_tokens_shape: tuple[int, ...]
    reconstructed_logits_shape: tuple[int, ...]
    max_abs_error: float
    mean_abs_error: float
    tolerance: float
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_timm_vit(model: nn.Module) -> None:
    missing = [name for name in REQUIRED_TIMM_VIT_ATTRIBUTES if not hasattr(model, name)]
    if missing:
        raise ViTPatchForwardError(
            "FCV raw-patch forwarding requires a timm VisionTransformer-like "
            f"model; missing attributes: {missing}."
        )
    patch_embed = model.patch_embed
    if not hasattr(patch_embed, "num_patches"):
        raise ViTPatchForwardError("model.patch_embed does not expose num_patches.")


def _expected_embedding_dim(model: nn.Module) -> int | None:
    for name in ("embed_dim", "num_features"):
        value = getattr(model, name, None)
        if value is not None:
            return int(value)
    return None


def _validate_raw_patch_tokens(model: nn.Module, raw_patch_tokens: torch.Tensor) -> None:
    if not isinstance(raw_patch_tokens, torch.Tensor):
        raise ViTPatchForwardError("raw_patch_tokens must be a torch.Tensor.")
    if raw_patch_tokens.ndim != 3:
        raise ViTPatchForwardError(
            "raw_patch_tokens must have shape [batch, patches, embedding_dim], "
            f"found {tuple(raw_patch_tokens.shape)}."
        )
    expected_patches = int(model.patch_embed.num_patches)
    if int(raw_patch_tokens.shape[1]) != expected_patches:
        raise ViTPatchForwardError(
            f"Expected {expected_patches} patch tokens, found "
            f"{raw_patch_tokens.shape[1]}."
        )
    expected_dim = _expected_embedding_dim(model)
    if expected_dim is not None and int(raw_patch_tokens.shape[2]) != expected_dim:
        raise ViTPatchForwardError(
            f"Expected patch embedding dimension {expected_dim}, found "
            f"{raw_patch_tokens.shape[2]}."
        )
    if not torch.is_floating_point(raw_patch_tokens):
        raise ViTPatchForwardError("raw_patch_tokens must use a floating-point dtype.")


def extract_raw_patch_tokens(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return patch embeddings before CLS/position embeddings, shape ``[B,N,D]``.

    The operation remains differentiable.  Callers performing FCV scoring are
    responsible for setting ``model.eval()`` and using an inference context.
    """

    _validate_timm_vit(model)
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        shape = tuple(images.shape) if isinstance(images, torch.Tensor) else None
        raise ViTPatchForwardError(
            f"images must be a tensor with shape [B,C,H,W], found {shape}."
        )
    raw_patch_tokens = model.patch_embed(images)
    _validate_raw_patch_tokens(model, raw_patch_tokens)
    return raw_patch_tokens


def forward_from_patch_tokens(
    model: nn.Module,
    raw_patch_tokens: torch.Tensor,
) -> torch.Tensor:
    """Resume a timm ViT forward from raw patches and return classifier logits.

    ``model._pos_embed`` adds the model's own prefix/CLS and positional
    embeddings.  Thus a donor token inserted before this function receives the
    target position, not its original donor position.
    """

    _validate_timm_vit(model)
    _validate_raw_patch_tokens(model, raw_patch_tokens)
    x = model._pos_embed(raw_patch_tokens)
    x = model.patch_drop(x)
    x = model.norm_pre(x)
    x = model.blocks(x)
    x = model.norm(x)
    logits = model.forward_head(x)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        shape = tuple(logits.shape) if isinstance(logits, torch.Tensor) else None
        raise ViTPatchForwardError(
            f"Expected classifier logits with shape [B,C], found {shape}."
        )
    return logits


def verify_reconstructed_forward(
    model: nn.Module,
    images: torch.Tensor,
    *,
    tolerance: float = 1.0e-5,
) -> ReconstructionReport:
    """Verify that the raw-token route reproduces the normal eval forward."""

    if model.training:
        raise ViTPatchForwardError(
            "Reconstruction verification requires model.eval(); dropout and "
            "stochastic-depth behavior is not deterministic in training mode."
        )
    if tolerance <= 0:
        raise ViTPatchForwardError("tolerance must be positive.")
    with torch.inference_mode():
        normal_logits = model(images)
        raw_patch_tokens = extract_raw_patch_tokens(model, images)
        reconstructed_logits = forward_from_patch_tokens(model, raw_patch_tokens)
    if normal_logits.shape != reconstructed_logits.shape:
        raise ViTPatchForwardError(
            "Normal and reconstructed logits have different shapes: "
            f"{tuple(normal_logits.shape)} versus "
            f"{tuple(reconstructed_logits.shape)}."
        )
    absolute_error = (normal_logits - reconstructed_logits).abs().float()
    max_abs_error = float(absolute_error.max().item())
    mean_abs_error = float(absolute_error.mean().item())
    return ReconstructionReport(
        normal_logits_shape=tuple(int(value) for value in normal_logits.shape),
        raw_patch_tokens_shape=tuple(int(value) for value in raw_patch_tokens.shape),
        reconstructed_logits_shape=tuple(
            int(value) for value in reconstructed_logits.shape
        ),
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        tolerance=float(tolerance),
        passed=max_abs_error < float(tolerance),
    )


class TimmViTPatchTokenAdapter(nn.Module):
    """Small module wrapper exposing both image and raw-token forward routes."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        _validate_timm_vit(model)
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def extract_raw_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        return extract_raw_patch_tokens(self.model, images)

    def forward_from_patch_tokens(self, raw_patch_tokens: torch.Tensor) -> torch.Tensor:
        return forward_from_patch_tokens(self.model, raw_patch_tokens)


def _load_trusted_torch_artifact(path: Path) -> Mapping[str, Any]:
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, Mapping):
        raise CandidateTrainingError(f"Checkpoint is not a mapping: {path}")
    return artifact


def load_candidate_model(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, Mapping[str, Any]]:
    """Strictly restore one Step 4 candidate for FCV extraction/scoring."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Candidate checkpoint does not exist: {path}")
    artifact = _load_trusted_torch_artifact(path)
    if artifact.get("artifact_type") != "fcv_vit_candidate_checkpoint":
        raise CandidateTrainingError(
            f"Not a Step 4 FCV candidate checkpoint: {path}"
        )
    if int(artifact.get("schema_version", -1)) != 1:
        raise CandidateTrainingError(
            f"Unsupported candidate checkpoint schema: {artifact.get('schema_version')}"
        )
    expected_fingerprint = candidate_training_fingerprint(config)
    if artifact.get("training_fingerprint") != expected_fingerprint:
        raise CandidateTrainingError(
            "Candidate checkpoint training fingerprint does not match the active config."
        )
    if artifact.get("software_versions") != software_versions():
        raise CandidateTrainingError(
            "Candidate checkpoint software provenance differs from the active runtime."
        )
    if artifact.get("source_tree_sha256") != source_tree_provenance()[
        "source_tree_sha256"
    ]:
        raise CandidateTrainingError("Candidate checkpoint source-tree hash is stale.")
    for field in ("initial_model_state_sha256", "pretrained_backbone_sha256"):
        if not isinstance(artifact.get(field), str) or len(artifact[field]) != 64:
            raise CandidateTrainingError(
                f"Candidate checkpoint is missing initialization provenance: {field}."
            )
    saved_model = artifact.get("model")
    if not isinstance(saved_model, Mapping):
        raise CandidateTrainingError("Candidate checkpoint is missing model metadata.")
    for key in ("name", "num_classes", "patch_size", "patch_grid_size"):
        if saved_model.get(key) != config["model"].get(key):
            raise CandidateTrainingError(
                f"Candidate model field {key!r} does not match the active config: "
                f"{saved_model.get(key)!r} versus {config['model'].get(key)!r}."
            )
    state_dict = artifact.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise CandidateTrainingError("Candidate checkpoint has no model_state_dict.")
    model = build_model(config, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.to(torch.device(device))
    model.eval()
    _validate_timm_vit(model)
    return model, artifact
