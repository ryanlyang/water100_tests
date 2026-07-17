from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.vit_counterfactual_forward import (  # noqa: E402
    TimmViTPatchTokenAdapter,
    ViTPatchForwardError,
    extract_raw_patch_tokens,
    forward_from_patch_tokens,
    load_candidate_model,
    validate_reconstruction_gate,
    verify_reconstructed_forward,
)
from fcv.candidate_training import (  # noqa: E402
    candidate_training_fingerprint,
    software_versions,
    source_tree_provenance,
)
from fcv.config import load_and_validate_config  # noqa: E402


class TinyPatchEmbed(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.patch_size = (2, 2)
        self.num_patches = 4
        self.projection = nn.Conv2d(3, embedding_dim, kernel_size=2, stride=2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images).flatten(2).transpose(1, 2)


class MixPrefixWithPatches(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mixed = tokens.clone()
        mixed[:, 0] = tokens[:, 0] + tokens[:, 1:].mean(dim=1)
        return mixed


class TinyTimmLikeViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 8
        self.num_features = 8
        self.patch_embed = TinyPatchEmbed(self.embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 5, self.embed_dim))
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.LayerNorm(self.embed_dim)
        self.blocks = nn.Sequential(MixPrefixWithPatches())
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, 2)

    def _pos_embed(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        cls = self.cls_token.expand(patch_tokens.shape[0], -1, -1)
        return torch.cat((cls, patch_tokens), dim=1) + self.pos_embed

    def forward_head(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens[:, 0])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images)
        tokens = self._pos_embed(tokens)
        tokens = self.patch_drop(tokens)
        tokens = self.norm_pre(tokens)
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        return self.forward_head(tokens)


class ViTCounterfactualForwardTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = TinyTimmLikeViT().eval()
        self.images = torch.randn(3, 3, 4, 4)

    def test_raw_tokens_reconstruct_normal_logits_exactly(self) -> None:
        report = verify_reconstructed_forward(self.model, self.images)
        self.assertTrue(report.passed)
        self.assertEqual(report.raw_patch_tokens_shape, (3, 4, 8))
        self.assertEqual(report.normal_logits_shape, (3, 2))
        self.assertEqual(report.max_abs_error, 0.0)

        adapter = TimmViTPatchTokenAdapter(self.model)
        raw = adapter.extract_raw_patch_tokens(self.images)
        self.assertTrue(
            torch.equal(adapter(self.images), adapter.forward_from_patch_tokens(raw))
        )

    def test_raw_patch_intervention_changes_logits(self) -> None:
        raw = extract_raw_patch_tokens(self.model, self.images)
        changed = raw.clone()
        changed[:, 2, 0] = changed[:, 2, 0] + 3.0
        original_logits = forward_from_patch_tokens(self.model, raw)
        changed_logits = forward_from_patch_tokens(self.model, changed)
        self.assertFalse(torch.allclose(original_logits, changed_logits))

    def test_verification_requires_eval_mode(self) -> None:
        self.model.train()
        with self.assertRaisesRegex(ViTPatchForwardError, "model.eval"):
            verify_reconstructed_forward(self.model, self.images)

    def test_invalid_patch_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ViTPatchForwardError, "Expected 4 patch tokens"):
            forward_from_patch_tokens(self.model, torch.randn(2, 3, 8))

    def test_candidate_loader_restores_strict_state_and_eval_mode(self) -> None:
        config_path = (
            EXPERIMENT_ROOT
            / "configs"
            / "waterbirds100_vit_s16_first_study.yaml"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config = deepcopy(config)
        config["model"].update(
            {
                "name": "tiny_timm_like_vit",
                "num_classes": 2,
                "patch_size": 2,
                "patch_grid_size": 2,
                "image_size": 4,
            }
        )
        config["training"]["augmentation"]["eval_resize_size"] = 4
        config["fcv"]["minimum_background_patches"] = 1
        validated = load_and_validate_config_dict(config)
        artifact = {
            "schema_version": 1,
            "artifact_type": "fcv_vit_candidate_checkpoint",
            "candidate_id": "tiny_epoch_001",
            "model": dict(validated["model"]),
            "training_fingerprint": candidate_training_fingerprint(validated),
            "software_versions": software_versions(),
            "source_tree_sha256": source_tree_provenance()["source_tree_sha256"],
            "initial_model_state_sha256": "a" * 64,
            "pretrained_backbone_sha256": "b" * 64,
            "model_state_dict": self.model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "candidate.pt"
            torch.save(artifact, checkpoint)
            restored = TinyTimmLikeViT()
            restored.train()
            with mock.patch(
                "fcv.vit_counterfactual_forward.build_model",
                return_value=restored,
            ):
                loaded_model, loaded_artifact = load_candidate_model(
                    validated,
                    checkpoint,
                )
        self.assertFalse(loaded_model.training)
        self.assertEqual(loaded_artifact["candidate_id"], "tiny_epoch_001")
        for expected, actual in zip(
            self.model.state_dict().values(), loaded_model.state_dict().values()
        ):
            self.assertTrue(torch.equal(expected, actual))

    def test_reconstruction_gate_binds_pretrained_and_candidate_reports(self) -> None:
        config_path = (
            EXPERIMENT_ROOT
            / "configs"
            / "waterbirds100_vit_s16_first_study.yaml"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config["paths"]["output_root"] = str(root / "outputs")
            validated = load_and_validate_config_dict(config)
            preflight = Path(validated["paths"]["output_root"]) / "preflight"
            preflight.mkdir(parents=True)
            checkpoint = root / "candidate.pt"
            checkpoint.write_bytes(b"candidate")
            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            common = {
                "schema_version": 1,
                "artifact_type": "fcv_vit_reconstruction_report",
                "status": "passed",
                "model": validated["model"]["name"],
                "training_fingerprint": candidate_training_fingerprint(validated),
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
                "tolerance": 1.0e-5,
            }
            pretrained = {
                **common,
                "source": "pretrained",
                "candidate_id": None,
                "checkpoint_path": None,
                "checkpoint_sha256": None,
            }
            candidate = {
                **common,
                "source": str(checkpoint),
                "candidate_id": "candidate_epoch_001",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
            }
            (preflight / "reconstruction_pretrained.json").write_text(
                json.dumps(pretrained), encoding="utf-8"
            )
            (preflight / "reconstruction_candidate.json").write_text(
                json.dumps(candidate), encoding="utf-8"
            )
            reports = validate_reconstruction_gate(validated)
            self.assertEqual(set(reports), {"pretrained", "candidate"})

            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(
                ViTPatchForwardError, "checkpoint has changed"
            ):
                validate_reconstruction_gate(validated)


def load_and_validate_config_dict(config: dict) -> dict:
    """Round-trip a test config through the public YAML config loader."""

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "config.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle)
        return load_and_validate_config(path, strict_protocol=False)


def _real_timm_is_available() -> bool:
    if importlib.util.find_spec("timm") is None:
        return False
    import timm

    return getattr(timm, "__version__", "") != "test-stub"


@unittest.skipUnless(_real_timm_is_available(), "real timm is not installed")
class RealTimmViTForwardTest(unittest.TestCase):
    def test_locked_model_reconstructs_logits(self) -> None:
        import timm

        torch.manual_seed(11)
        model = timm.create_model(
            "vit_small_patch16_224.augreg_in21k_ft_in1k",
            pretrained=False,
            num_classes=2,
        ).eval()
        images = torch.randn(1, 3, 224, 224)
        report = verify_reconstructed_forward(model, images, tolerance=1.0e-5)
        self.assertTrue(report.passed, report.to_dict())
        self.assertEqual(report.raw_patch_tokens_shape[1:], (196, 384))


if __name__ == "__main__":
    unittest.main()
