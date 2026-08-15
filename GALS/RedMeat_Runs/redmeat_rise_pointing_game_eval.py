#!/usr/bin/env python3
"""Evaluate a RedMeat model with a shared, deterministic RISE Pointing Game.

This evaluator is intentionally test-only. It validates the reviewed mask
package against ``all_images.csv``, applies method-correct deterministic image
preprocessing and geometrically matching mask preprocessing, and explains the
ground-truth class by default. All methods use the same saved RISE mask bank.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models
from torchvision import transforms


THIS_DIR = Path(__file__).resolve().parent
GALS_ROOT = THIS_DIR.parent
if str(GALS_ROOT) not in sys.path:
    sys.path.insert(0, str(GALS_ROOT))

from gals_rise_utils import load_or_create_mask_bank, rise_from_probabilities_batch  # noqa: E402
from models.resnet import resnet50 as gals_resnet50  # noqa: E402
from models.resnet_abn import resnet50 as gals_resnet50_abn  # noqa: E402
from RedMeat_Runs.validate_redmeat_pointing_masks import (  # noqa: E402
    load_csv,
    metadata_image_index,
    sha256_file,
    validate_package,
)


CLASS_NAMES = (
    "prime_rib",
    "pork_chop",
    "steak",
    "baby_back_ribs",
    "filet_mignon",
)
SUPPORTED_METHODS = (
    "vanilla",
    "elrep",
    "upweight",
    "abn",
    "gals",
    "afr",
    "r4rr",
    "clip_lr",
    "clip_zs",
)
IMAGE_SIZE = 224
MASK_PROTOCOL_VERSION = 1
PRIMARY_PG_PROTOCOL = "rise_pixel_argmax"
EXPLAINER = "rise"


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finite_float(value: object) -> Optional[float]:
    result = float(value)
    return result if np.isfinite(result) else None


def torch_load_compat(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        if "Weights only load failed" not in str(exc):
            raise
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "algorithm"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(key, str) for key in checkpoint):
            if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
                return checkpoint  # type: ignore[return-value]
    raise RuntimeError("Could not extract a tensor state_dict from checkpoint")


def candidate_state_dicts(state_dict: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    def strip_prefix(values: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {
            (key[len(prefix) :] if key.startswith(prefix) else key): value
            for key, value in values.items()
        }

    def add_prefix(values: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {prefix + key: value for key, value in values.items()}

    candidates = [state_dict]
    for prefix in ("module.", "model.", "net.", "base."):
        candidates.append(strip_prefix(state_dict, prefix))
    for prefix in ("module.", "model.", "net.", "base."):
        candidates.append(add_prefix(state_dict, prefix))

    unique: List[Dict[str, torch.Tensor]] = []
    signatures: Set[Tuple[str, ...]] = set()
    for candidate in candidates:
        signature = tuple(sorted(candidate))
        if signature not in signatures:
            signatures.add(signature)
            unique.append(candidate)
    return unique


def load_state_flexible(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[int, List[str], List[str]]:
    model_state = model.state_dict()
    best: Optional[Dict[str, torch.Tensor]] = None
    best_overlap = -1
    for candidate in candidate_state_dicts(state_dict):
        overlap = sum(
            key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
            for key, value in candidate.items()
        )
        if overlap > best_overlap:
            best = candidate
            best_overlap = overlap
    if best is None or best_overlap <= 0:
        return 0, list(model_state), list(state_dict)

    compatible = {
        key: value
        for key, value in best.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    source_unexpected = [key for key in best if key not in compatible]
    return best_overlap, list(missing), sorted(set(unexpected).union(source_unexpected))


def make_torchvision_resnet50(num_classes: int) -> nn.Module:
    try:
        model = tv_models.resnet50(weights=None)
    except TypeError:
        model = tv_models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def make_gals_resnet50(num_classes: int) -> nn.Module:
    model = gals_resnet50(pretrained=False, return_fmaps=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def make_abn_resnet50(num_classes: int) -> nn.Module:
    return gals_resnet50_abn(
        pretrained=False,
        num_classes=num_classes,
        add_after_attention=True,
    )


def load_best_builder(
    checkpoint_path: Path,
    builders: Sequence[Tuple[str, Callable[[], nn.Module]]],
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, object]]:
    state = extract_state_dict(torch_load_compat(checkpoint_path, device))
    selected: Optional[nn.Module] = None
    selected_meta: Optional[Dict[str, object]] = None
    selected_overlap = -1
    for builder_name, builder in builders:
        model = builder().to(device)
        overlap, missing, unexpected = load_state_flexible(model, state)
        total_keys = len(model.state_dict())
        if overlap > selected_overlap:
            selected = model
            selected_overlap = overlap
            selected_meta = {
                "builder": builder_name,
                "loaded_key_overlap": int(overlap),
                "model_key_count": int(total_keys),
                "missing_key_count": len(missing),
                "unexpected_key_count": len(unexpected),
                "missing_keys_preview": missing[:20],
                "unexpected_keys_preview": unexpected[:20],
            }
    if selected is None or selected_meta is None or selected_overlap <= 0:
        raise RuntimeError(f"Could not load checkpoint into a candidate model: {checkpoint_path}")
    loaded_fraction = selected_overlap / float(max(len(selected.state_dict()), 1))
    selected_meta["loaded_key_fraction"] = loaded_fraction
    if loaded_fraction < 0.90:
        raise RuntimeError(
            f"Unsafe partial checkpoint load for {checkpoint_path}: {selected_meta}"
        )
    selected.eval()
    return selected, selected_meta


def extract_last_layer(
    state_dict: Dict[str, torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    for candidate in candidate_state_dicts(state_dict):
        for weight_key, bias_key in (("weight", "bias"), ("fc.weight", "fc.bias")):
            weight = candidate.get(weight_key)
            bias = candidate.get(bias_key)
            if (
                isinstance(weight, torch.Tensor)
                and isinstance(bias, torch.Tensor)
                and weight.ndim == 2
                and bias.ndim == 1
            ):
                return weight, bias
    return None


def load_afr_model(
    stage1_checkpoint: Path,
    stage2_checkpoint: Path,
    num_classes: int,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, object]]:
    model, metadata = load_best_builder(
        stage1_checkpoint,
        (
            ("gals_resnet50", lambda: make_gals_resnet50(num_classes)),
            ("torchvision_resnet50", lambda: make_torchvision_resnet50(num_classes)),
        ),
        device,
    )
    stage2_state = extract_state_dict(torch_load_compat(stage2_checkpoint, device))
    last_layer = extract_last_layer(stage2_state)
    if last_layer is None:
        # Some AFR exports contain a full stage-2 model rather than only its head.
        return load_best_builder(
            stage2_checkpoint,
            (
                ("gals_resnet50", lambda: make_gals_resnet50(num_classes)),
                ("torchvision_resnet50", lambda: make_torchvision_resnet50(num_classes)),
            ),
            device,
        )
    weight, bias = last_layer
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise RuntimeError("AFR reconstruction requires an nn.Linear model.fc")
    if tuple(model.fc.weight.shape) != tuple(weight.shape) or tuple(model.fc.bias.shape) != tuple(bias.shape):
        raise RuntimeError(
            "AFR stage-2 head shape mismatch: "
            f"model={tuple(model.fc.weight.shape)}/{tuple(model.fc.bias.shape)} "
            f"checkpoint={tuple(weight.shape)}/{tuple(bias.shape)}"
        )
    with torch.no_grad():
        model.fc.weight.copy_(weight.to(model.fc.weight.device, dtype=model.fc.weight.dtype))
        model.fc.bias.copy_(bias.to(model.fc.bias.device, dtype=model.fc.bias.dtype))
    metadata = dict(metadata)
    metadata.update(
        {
            "afr_load_mode": "stage1_plus_stage2_last_layer",
            "afr_stage1_checkpoint": str(stage1_checkpoint),
            "afr_stage2_checkpoint": str(stage2_checkpoint),
        }
    )
    model.eval()
    return model, metadata


class RedMeatProbabilityModel(nn.Module):
    """Normalize repository model outputs to five class probabilities."""

    def __init__(self, method: str, model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.method = method
        self.model = model
        self.num_classes = int(num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images)
        if self.method == "abn":
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError("ABN model did not return attention/classification outputs")
            logits = output[1]
        elif isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise RuntimeError(f"Invalid logits from {self.method}: {type(logits)}")
        if logits.shape[1] != self.num_classes:
            raise RuntimeError(
                f"Expected {self.num_classes} logits from {self.method}, got {tuple(logits.shape)}"
            )
        return torch.softmax(logits.float(), dim=1)


class CLIPZeroShotProbabilityModel(nn.Module):
    def __init__(self, model: nn.Module, text_features: np.ndarray) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "text_features",
            torch.from_numpy(np.asarray(text_features, dtype=np.float32)),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(images).float()
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = self.model.logit_scale.exp().float().clamp(max=100.0)
        return torch.softmax(scale * features @ self.text_features.t(), dim=1)


class CLIPLinearProbabilityModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        coefficients: np.ndarray,
        intercept: np.ndarray,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "coefficients",
            torch.from_numpy(np.asarray(coefficients, dtype=np.float32)),
            persistent=False,
        )
        self.register_buffer(
            "intercept",
            torch.from_numpy(np.asarray(intercept, dtype=np.float32)),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(images).float()
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        logits = features @ self.coefficients.t() + self.intercept
        return torch.softmax(logits, dim=1)


def build_cnn_preprocess() -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_cnn_mask_preprocess() -> Callable[[Image.Image], Image.Image]:
    return transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.NEAREST,
    )


def build_clip_mask_preprocess() -> Callable[[Image.Image], Image.Image]:
    return transforms.Compose(
        [
            transforms.Resize(
                IMAGE_SIZE,
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.CenterCrop(IMAGE_SIZE),
        ]
    )


def build_checkpoint_probability_model(
    method: str,
    checkpoint: Path,
    afr_stage1_checkpoint: Optional[Path],
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, object]]:
    num_classes = len(CLASS_NAMES)
    if method in ("vanilla", "elrep"):
        model, metadata = load_best_builder(
            checkpoint,
            (("torchvision_resnet50", lambda: make_torchvision_resnet50(num_classes)),),
            device,
        )
    elif method in ("gals", "upweight"):
        model, metadata = load_best_builder(
            checkpoint,
            (("gals_resnet50", lambda: make_gals_resnet50(num_classes)),),
            device,
        )
    elif method == "abn":
        model, metadata = load_best_builder(
            checkpoint,
            (("gals_resnet50_abn", lambda: make_abn_resnet50(num_classes)),),
            device,
        )
    elif method == "r4rr":
        # This training module imports the shared Waterbirds runner from the
        # parent repository. Keep it lazy so other evaluators do not depend on
        # that training-only module being importable.
        from RedMeat_Runs import run_guided_redmeat as guided_redmeat

        model = guided_redmeat.make_redmeat_cam_model(
            num_classes=num_classes,
            model_name="resnet50",
            pretrained=False,
            clip_model="RN50",
        ).to(device)
        state = extract_state_dict(torch_load_compat(checkpoint, device))
        overlap, missing, unexpected = load_state_flexible(model, state)
        if overlap != len(model.state_dict()) or missing or unexpected:
            raise RuntimeError(
                "R4RR checkpoint did not load exactly: "
                f"overlap={overlap}/{len(model.state_dict())} "
                f"missing={missing} unexpected={unexpected}"
            )
        metadata = {
            "builder": "redmeat_cam_resnet50",
            "loaded_key_overlap": len(model.state_dict()),
            "loaded_key_fraction": 1.0,
            "missing_key_count": 0,
            "unexpected_key_count": 0,
        }
    elif method == "afr":
        if afr_stage1_checkpoint is None:
            raise ValueError("AFR requires --afr-stage1-checkpoint")
        model, metadata = load_afr_model(
            stage1_checkpoint=afr_stage1_checkpoint,
            stage2_checkpoint=checkpoint,
            num_classes=num_classes,
            device=device,
        )
    else:
        raise ValueError(f"Checkpoint model is unsupported for method={method}")
    model.eval()
    return RedMeatProbabilityModel(method, model, num_classes), metadata


def build_clip_probability_model(
    method: str,
    data_root: Path,
    clip_model_name: str,
    clip_c: float,
    feature_batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, Callable[[Image.Image], torch.Tensor], Dict[str, object]]:
    import run_clip_zeroshot_redmeat as clip_zs
    from RedMeat_Runs import run_clip_lr_sweep_redmeat as clip_lr

    clip_module = clip_lr._try_import_clip()
    try:
        model, preprocess = clip_module.load(clip_model_name, device=str(device), jit=False)
    except TypeError:
        model, preprocess = clip_module.load(clip_model_name, device=str(device))
    model.eval()

    details: Dict[str, object] = {
        "clip_model": clip_model_name,
        "clip_class_names": "|".join(CLASS_NAMES),
        "clip_c": float(clip_c) if method == "clip_lr" else "",
        "clip_penalty": "l2" if method == "clip_lr" else "",
        "clip_solver": "lbfgs" if method == "clip_lr" else "",
        "clip_fit_intercept": True if method == "clip_lr" else "",
        "clip_feature_mode": "l2" if method == "clip_lr" else "",
    }
    if method == "clip_zs":
        templates = clip_zs._default_templates()
        text_features = clip_zs._build_text_features(
            clip_module=clip_module,
            model=model,
            device=str(device),
            class_names=CLASS_NAMES,
            templates=templates,
        )
        details["clip_num_templates"] = len(templates)
        return CLIPZeroShotProbabilityModel(model, text_features), preprocess, details

    from sklearn.linear_model import LogisticRegression

    classes, train_samples, _val_samples, _test_samples = clip_lr._build_splits(
        dataset_path=str(data_root),
        split_col="split",
        label_col="label",
        path_col="abs_file_path",
        train_value="train",
        val_value="val",
        test_value="test",
        classes=list(CLASS_NAMES),
    )
    if tuple(classes) != CLASS_NAMES:
        raise RuntimeError(f"CLIP-LR class order mismatch: {classes} != {CLASS_NAMES}")
    features, labels = clip_lr._extract_features(
        train_samples,
        model,
        preprocess,
        str(device),
        int(feature_batch_size),
        int(num_workers),
    )
    features = np.ascontiguousarray(clip_lr._l2_normalize(features), dtype=np.float64)
    classifier = LogisticRegression(
        random_state=int(seed),
        C=float(clip_c),
        penalty="l2",
        solver="lbfgs",
        fit_intercept=True,
        max_iter=5000,
        n_jobs=1,
        verbose=0,
    )
    clip_lr._safe_fit(classifier, features, labels)
    details["clip_train_samples"] = int(labels.shape[0])
    return (
        CLIPLinearProbabilityModel(model, classifier.coef_, classifier.intercept_),
        preprocess,
        details,
    )


class RedMeatPointingDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        mask_root: Path,
        preprocess: Callable[[Image.Image], torch.Tensor],
        mask_preprocess: Callable[[Image.Image], Image.Image],
        mask_threshold: int,
        max_samples: int,
        sample_seed: int,
        verify_source_checksum: bool,
    ) -> None:
        metadata_csv = data_root / "all_images.csv"
        external_images = metadata_image_index(
            data_root=data_root,
            metadata_csv=metadata_csv,
            split="test",
            split_col="split",
            label_col="label",
            path_col="abs_file_path",
        )
        validation = validate_package(
            package_root=mask_root,
            expected_images=1250,
            expected_per_class=250,
            classes=tuple(sorted(CLASS_NAMES)),
            external_images=external_images,
            verify_source_checksum=verify_source_checksum,
        )
        if not validation["valid"]:
            raise RuntimeError(f"Invalid RedMeat mask package: {validation['errors'][:20]}")
        self.validation = validation
        self.preprocess = preprocess
        self.mask_preprocess = mask_preprocess
        self.mask_threshold = int(mask_threshold)

        class_to_idx = {name: index for index, name in enumerate(CLASS_NAMES)}
        records: List[Dict[str, object]] = []
        for row in load_csv(mask_root / "manifest.csv"):
            key = (row["class_name"], row["image_id"])
            image_path = external_images[key]
            records.append(
                {
                    "image_id": row["image_id"],
                    "class_name": row["class_name"],
                    "label": class_to_idx[row["class_name"]],
                    "image_path": image_path,
                    "mask_path": mask_root / row["mask_relative_path"],
                }
            )
        records.sort(key=lambda row: (int(row["label"]), int(str(row["image_id"]))))
        if max_samples > 0 and max_samples < len(records):
            rng = np.random.RandomState(int(sample_seed))
            selected = sorted(rng.choice(len(records), size=int(max_samples), replace=False).tolist())
            records = [records[index] for index in selected]
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")
            image_tensor = self.preprocess(image)
        with Image.open(record["mask_path"]) as source_mask:
            transformed_mask = self.mask_preprocess(source_mask.convert("L"))
            mask = (
                np.asarray(transformed_mask, dtype=np.uint8) > self.mask_threshold
            ).astype(np.uint8)
        if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"Transformed mask has shape={mask.shape}")
        return (
            image_tensor,
            torch.from_numpy(mask),
            int(record["label"]),
            str(record["class_name"]),
            str(record["image_id"]),
            str(record["image_path"]),
            str(record["mask_path"]),
        )


def pointing_result(saliency: np.ndarray, mask: np.ndarray) -> Tuple[bool, int, int, bool]:
    if saliency.shape != mask.shape:
        raise ValueError(f"Saliency/mask shape mismatch: {saliency.shape} != {mask.shape}")
    if not np.isfinite(saliency).all():
        raise ValueError("RISE saliency contains non-finite values")
    is_zero = float(np.max(saliency)) <= 1e-12
    if is_zero:
        return False, -1, -1, True
    peak_row, peak_col = np.unravel_index(int(np.argmax(saliency)), saliency.shape)
    return bool(mask[peak_row, peak_col] > 0), int(peak_row), int(peak_col), False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--afr-stage1-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-mode", choices=("label", "prediction"), default="label")
    parser.add_argument("--mask-threshold", type=int, default=0)
    parser.add_argument("--skip-source-checksum", action="store_true")
    parser.add_argument("--clip-model", default="RN50")
    parser.add_argument("--clip-c", type=float, default=1.329346323656201)
    parser.add_argument("--clip-feature-batch-size", type=int, default=256)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--max-masked-batch", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--rise-num-masks", type=int, default=2000)
    parser.add_argument("--rise-grid-size", type=int, default=8)
    parser.add_argument("--rise-p1", type=float, default=0.1)
    parser.add_argument("--rise-seed", type=int, default=0)
    parser.add_argument("--rise-masks-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    random.seed(int(args.sample_seed))
    np.random.seed(int(args.sample_seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    data_root = args.data_root.expanduser().resolve()
    mask_root = args.mask_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rise_masks_path = args.rise_masks_path.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    afr_stage1 = (
        args.afr_stage1_checkpoint.expanduser().resolve()
        if args.afr_stage1_checkpoint
        else None
    )
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if not mask_root.is_dir():
        raise FileNotFoundError(mask_root)

    is_clip = args.method in ("clip_lr", "clip_zs")
    if not is_clip and (checkpoint is None or not checkpoint.is_file()):
        raise FileNotFoundError(f"Missing checkpoint for method={args.method}: {checkpoint}")
    if args.method == "afr" and (afr_stage1 is None or not afr_stage1.is_file()):
        raise FileNotFoundError(f"Missing AFR stage-1 checkpoint: {afr_stage1}")

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    clip_details: Dict[str, object] = {
        "clip_model": "",
        "clip_class_names": "",
        "clip_c": "",
        "clip_penalty": "",
        "clip_solver": "",
        "clip_fit_intercept": "",
        "clip_feature_mode": "",
    }
    load_metadata: Dict[str, object] = {}
    if is_clip:
        probability_model, preprocess, clip_details = build_clip_probability_model(
            method=args.method,
            data_root=data_root,
            clip_model_name=str(args.clip_model),
            clip_c=float(args.clip_c),
            feature_batch_size=int(args.clip_feature_batch_size),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
            device=device,
        )
        mask_preprocess = build_clip_mask_preprocess()
        preprocessing = "openai_clip_resize_center_crop_224"
    else:
        assert checkpoint is not None
        probability_model, load_metadata = build_checkpoint_probability_model(
            method=args.method,
            checkpoint=checkpoint,
            afr_stage1_checkpoint=afr_stage1,
            device=device,
        )
        preprocess = build_cnn_preprocess()
        mask_preprocess = build_cnn_mask_preprocess()
        preprocessing = "imagenet_resize_224x224"

    dataset = RedMeatPointingDataset(
        data_root=data_root,
        mask_root=mask_root,
        preprocess=preprocess,
        mask_preprocess=mask_preprocess,
        mask_threshold=int(args.mask_threshold),
        max_samples=int(args.max_samples),
        sample_seed=int(args.sample_seed),
        verify_source_checksum=not args.skip_source_checksum,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.image_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory and device.type == "cuda"),
    )
    probability_model = probability_model.to(device).eval()
    masks_np, masks_sha256 = load_or_create_mask_bank(
        path=rise_masks_path,
        num_masks=int(args.rise_num_masks),
        grid_size=int(args.rise_grid_size),
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        p1=float(args.rise_p1),
        seed=int(args.rise_seed),
    )
    rise_masks = torch.from_numpy(masks_np).to(device=device, dtype=torch.float32)

    class_count = len(CLASS_NAMES)
    class_hits = np.zeros(class_count, dtype=np.int64)
    class_totals = np.zeros(class_count, dtype=np.int64)
    class_correct = np.zeros(class_count, dtype=np.int64)
    class_mass_sum = np.zeros(class_count, dtype=np.float64)
    correct_only_hits = 0
    correct_only_total = 0
    mask_pixels_total = 0
    zero_maps = 0
    processed = 0
    rows: List[Dict[str, object]] = []

    print(
        f"[INFO] dataset=redmeat method={args.method} seed={args.seed} samples={len(dataset)} "
        f"device={device} target_mode={args.target_mode} explainer={EXPLAINER}",
        flush=True,
    )
    print(f"[INFO] class_order={list(CLASS_NAMES)}", flush=True)
    print(f"[INFO] preprocessing={preprocessing}", flush=True)
    print(f"[INFO] checkpoint={checkpoint or ''}", flush=True)
    print(f"[INFO] afr_stage1_checkpoint={afr_stage1 or ''}", flush=True)
    print(f"[INFO] load_metadata={load_metadata}", flush=True)
    print(f"[INFO] clip_details={clip_details}", flush=True)
    print(
        f"[INFO] mask_package={mask_root} validated={dataset.validation['valid']} "
        f"manifest_sha256={sha256_file(mask_root / 'manifest.csv')}",
        flush=True,
    )
    print(
        f"[INFO] rise_masks={rise_masks_path} sha256={masks_sha256} "
        f"N={args.rise_num_masks} grid={args.rise_grid_size} "
        f"p1={args.rise_p1} seed={args.rise_seed}",
        flush=True,
    )

    for batch in loader:
        images, meat_masks, labels, class_names, image_ids, image_paths, mask_paths = batch
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        with torch.no_grad():
            probabilities = probability_model(images)
            predictions = probabilities.argmax(dim=1)
        targets = labels_device if args.target_mode == "label" else predictions
        saliency_batch = rise_from_probabilities_batch(
            probability_fn=probability_model,
            images=images,
            targets=targets,
            masks=rise_masks,
            p1=float(args.rise_p1),
            max_masked_batch=int(args.max_masked_batch),
        ).cpu().numpy()

        masks_batch = meat_masks.numpy()
        labels_batch = labels.numpy()
        predictions_batch = predictions.cpu().numpy()
        targets_batch = targets.cpu().numpy()
        for (
            saliency,
            meat_mask,
            label,
            prediction,
            target,
            class_name,
            image_id,
            image_path,
            mask_path,
        ) in zip(
            saliency_batch,
            masks_batch,
            labels_batch,
            predictions_batch,
            targets_batch,
            class_names,
            image_ids,
            image_paths,
            mask_paths,
        ):
            hit, peak_row, peak_col, is_zero = pointing_result(saliency, meat_mask)
            class_index = int(label)
            correct = int(int(prediction) == class_index)
            positive_saliency = np.maximum(saliency.astype(np.float64), 0.0)
            saliency_sum = float(positive_saliency.sum())
            mass_inside = (
                float(positive_saliency[meat_mask > 0].sum() / saliency_sum)
                if saliency_sum > 0.0
                else 0.0
            )
            class_totals[class_index] += 1
            class_hits[class_index] += int(hit)
            class_correct[class_index] += correct
            class_mass_sum[class_index] += mass_inside
            if correct:
                correct_only_total += 1
                correct_only_hits += int(hit)
            mask_pixels = int(np.count_nonzero(meat_mask))
            mask_pixels_total += mask_pixels
            zero_maps += int(is_zero)
            rows.append(
                {
                    "dataset": "redmeat",
                    "method": args.method,
                    "seed": int(args.seed),
                    "split": "test",
                    "target_mode": args.target_mode,
                    "explainer": EXPLAINER,
                    "image_id": image_id,
                    "class_name": class_name,
                    "label": class_index,
                    "prediction": int(prediction),
                    "classification_correct": correct,
                    "saliency_target": int(target),
                    "pointing_hit": int(hit),
                    "peak_row": peak_row,
                    "peak_col": peak_col,
                    "saliency_max": float(np.max(saliency)),
                    "saliency_mass_in_meat": mass_inside,
                    "zero_saliency": int(is_zero),
                    "meat_mask_pixels": mask_pixels,
                    "image_path": image_path,
                    "mask_path": mask_path,
                }
            )
        processed += len(labels_batch)
        if processed == len(dataset) or processed % 100 <= len(labels_batch):
            print(f"[PROGRESS] {processed}/{len(dataset)}", flush=True)

    if not rows:
        raise RuntimeError("No RedMeat samples were evaluated")
    class_pg = np.divide(
        class_hits,
        class_totals,
        out=np.full(class_count, np.nan, dtype=np.float64),
        where=class_totals > 0,
    )
    class_acc = np.divide(
        class_correct,
        class_totals,
        out=np.full(class_count, np.nan, dtype=np.float64),
        where=class_totals > 0,
    )
    class_mass = np.divide(
        class_mass_sum,
        class_totals,
        out=np.full(class_count, np.nan, dtype=np.float64),
        where=class_totals > 0,
    )
    worst_pg_index = int(np.nanargmin(class_pg))
    total = int(class_totals.sum())
    mask_manifest_sha256 = sha256_file(mask_root / "manifest.csv")
    summary: Dict[str, object] = {
        "dataset": "redmeat",
        "method": args.method,
        "seed": int(args.seed),
        "split": "test",
        "target_mode": args.target_mode,
        "explainer": EXPLAINER,
        "primary_pg_protocol": PRIMARY_PG_PROTOCOL,
        "mask_protocol_version": MASK_PROTOCOL_VERSION,
        "map_height": IMAGE_SIZE,
        "map_width": IMAGE_SIZE,
        "pg_hits": int(class_hits.sum()),
        "pg_total": total,
        "pg_acc": float(class_hits.sum() / max(total, 1)),
        "pg_macro_class_acc": float(np.nanmean(class_pg)),
        "pg_worst_class_acc": float(class_pg[worst_pg_index]),
        "pg_worst_class": CLASS_NAMES[worst_pg_index],
        "pg_correct_only_hits": int(correct_only_hits),
        "pg_correct_only_total": int(correct_only_total),
        "pg_correct_only_acc": float(correct_only_hits / max(correct_only_total, 1)),
        "pg_random_acc": float(mask_pixels_total / max(total * IMAGE_SIZE * IMAGE_SIZE, 1)),
        "classification_acc": float(class_correct.sum() / max(total, 1)),
        "classification_balanced_class_acc": float(np.nanmean(class_acc)),
        "classification_worst_class_acc": float(np.nanmin(class_acc)),
        "saliency_mass_in_meat": float(class_mass_sum.sum() / max(total, 1)),
        "zero_saliency_maps": int(zero_maps),
        "preprocessing": preprocessing,
        "mask_source": "reviewed_redmeat_coco_union_v1",
        "mask_root": str(mask_root),
        "mask_manifest_sha256": mask_manifest_sha256,
        "mask_threshold": int(args.mask_threshold),
        "mask_package_validated": bool(dataset.validation["valid"]),
        "source_checksums_verified": bool(dataset.validation["source_checksums_verified"]),
        "max_samples": int(args.max_samples),
        "sample_seed": int(args.sample_seed),
        "checkpoint": str(checkpoint) if checkpoint else "",
        "checkpoint_load_metadata": json.dumps(load_metadata, sort_keys=True),
        "afr_stage1_checkpoint": str(afr_stage1) if afr_stage1 else "",
        **clip_details,
        "rise_num_masks": int(args.rise_num_masks),
        "rise_grid_size": int(args.rise_grid_size),
        "rise_p1": float(args.rise_p1),
        "rise_seed": int(args.rise_seed),
        "rise_masks_path": str(rise_masks_path),
        "rise_masks_sha256": masks_sha256,
        "image_batch_size": int(args.image_batch_size),
        "max_masked_batch": int(args.max_masked_batch),
        "missing_images": 0,
        "missing_masks": 0,
        "errors": 0,
        "seconds": int(time.time() - started),
    }
    for index, class_name in enumerate(CLASS_NAMES):
        summary[f"class_{class_name}_pg_hits"] = int(class_hits[index])
        summary[f"class_{class_name}_total"] = int(class_totals[index])
        summary[f"class_{class_name}_pg_acc"] = finite_float(class_pg[index])
        summary[f"class_{class_name}_classification_acc"] = finite_float(class_acc[index])
        summary[f"class_{class_name}_saliency_mass_in_meat"] = finite_float(class_mass[index])

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pointing_game_per_image.csv", rows)
    write_csv(output_dir / "pointing_game_summary.csv", [summary])
    write_json(output_dir / "run_summary.json", summary)
    print(
        f"[RESULT] method={args.method} seed={args.seed} "
        f"rise_pg={100.0 * float(summary['pg_acc']):.2f}% "
        f"macro_class={100.0 * float(summary['pg_macro_class_acc']):.2f}% "
        f"worst_class={summary['pg_worst_class']} "
        f"worst={100.0 * float(summary['pg_worst_class_acc']):.2f}% "
        f"random={100.0 * float(summary['pg_random_acc']):.2f}% "
        f"classification={100.0 * float(summary['classification_acc']):.2f}% "
        f"zero_maps={zero_maps}/{total}",
        flush=True,
    )
    print(f"[DONE] {output_dir / 'pointing_game_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
