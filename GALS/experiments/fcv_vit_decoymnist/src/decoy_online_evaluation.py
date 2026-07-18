"""Step-9 visibility-separated Oracle and official-test evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from decoy_candidate_training import (
    ManifestImageDataset,
    build_evaluation_transform,
    evaluate_classifier,
    seed_worker,
)
from decoy_oracle_view import OracleViewDataset


class OnlineEvaluationError(ValueError):
    """Raised when Oracle/test analysis violates split or provenance boundaries."""


def build_analysis_loaders(
    config: Mapping[str, Any],
    oracle_manifest: str | Path,
    test_manifest: str | Path,
    *,
    seed: int,
):
    """Build independently seeded, deterministic analysis-only loaders."""

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - exercised on Tigris
        raise RuntimeError("Step 9 requires PyTorch.") from exc
    transform = build_evaluation_transform(config)
    oracle = OracleViewDataset(config, oracle_manifest, transform=transform)
    test = ManifestImageDataset(config, test_manifest, "test", transform)
    if oracle.binding.bundle_sha256 != test.binding.bundle_sha256:
        raise OnlineEvaluationError("Oracle and test do not share one split bundle.")
    oracle_generator = torch.Generator()
    oracle_generator.manual_seed(20_000 + int(seed))
    test_generator = torch.Generator()
    test_generator.manual_seed(30_000 + int(seed))
    common = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,
        "worker_init_fn": seed_worker,
        "drop_last": False,
        "shuffle": False,
    }
    loaders = {
        "oracle_analysis_only": DataLoader(
            oracle, generator=oracle_generator, **common
        ),
        "test_analysis_only": DataLoader(test, generator=test_generator, **common),
    }
    return loaders, {"oracle_analysis_only": oracle, "test_analysis_only": test}


def evaluate_oracle_online(
    model: Any,
    oracle_loader: Any,
    device: Any,
    *,
    precision: str,
    num_classes: int,
) -> Dict[str, Any]:
    """Evaluate the privileged reversed-decoy validation view in memory."""

    metrics = evaluate_classifier(
        model, oracle_loader, device, precision, int(num_classes)
    ).as_dict()
    return {
        "visibility": "oracle_analysis_only",
        "selector_authorization": "oracle_only",
        "metrics": metrics,
        "per_image_predictions_persisted": False,
    }


def evaluate_test_analysis_only(
    model: Any,
    test_loader: Any,
    device: Any,
    *,
    precision: str,
    num_classes: int,
) -> Dict[str, Any]:
    """Evaluate the official reversed-decoy test without exposing selector access."""

    metrics = evaluate_classifier(
        model, test_loader, device, precision, int(num_classes)
    ).as_dict()
    return {
        "visibility": "test_analysis_only",
        "selector_authorization": "posthoc_only",
        "metrics": metrics,
        "per_image_predictions_persisted": False,
        "training_or_stopping_authorized": False,
    }
