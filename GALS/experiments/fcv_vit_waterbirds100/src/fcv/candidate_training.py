"""Vanilla ViT candidate-pool training for the first FCV study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .config import candidate_epochs
from .manifest_provenance import (
    ManifestProvenanceError,
    validate_manifest_bundle,
)
from .storage import assert_storage_budget


PUBLIC_MANIFEST_REQUIRED = {
    "sample_id",
    "metadata_index",
    "image_path",
    "image_sha256",
    "label",
    "source_split",
    "study_split",
}
FORBIDDEN_SELECTOR_COLUMNS = {"context", "group", "group_name", "place"}
METRIC_COLUMNS = [
    "run_index",
    "candidate_id",
    "epoch",
    "model_name",
    "seed",
    "learning_rate",
    "weight_decay",
    "train_loss",
    "train_accuracy",
    "biased_val_loss",
    "biased_val_accuracy",
    "lr_epoch_start",
    "lr_epoch_end",
    "checkpoint_path",
    "checkpoint_sha256",
    "epoch_seconds",
]


class CandidateTrainingError(ValueError):
    """Raised when candidate training would violate the locked protocol."""


@dataclass(frozen=True)
class SweepRun:
    run_index: int
    learning_rate: float
    weight_decay: float
    seed: int

    @property
    def run_id(self) -> str:
        lr = _float_slug(self.learning_rate)
        wd = _float_slug(self.weight_decay)
        return f"run_{self.run_index:03d}_lr_{lr}_wd_{wd}_seed_{self.seed}"

    def candidate_id(self, epoch: int) -> str:
        return f"{self.run_id}_epoch_{epoch:03d}"


def _float_slug(value: float) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace("+", "").replace(".", "p")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_tree_provenance() -> Dict[str, Any]:
    """Hash Python and launch sources for one immutable campaign identity."""

    experiment_root = Path(__file__).resolve().parents[2]
    source_paths = sorted(
        [
            *experiment_root.glob("src/**/*.py"),
            *experiment_root.glob("scripts/*.py"),
            *experiment_root.glob("scripts/*.sh"),
            *experiment_root.glob("slurm/*.sbatch"),
        ],
        key=lambda path: path.relative_to(experiment_root).as_posix(),
    )
    entries = [
        {
            "path": path.relative_to(experiment_root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in source_paths
        if "__pycache__" not in path.parts
    ]
    return {
        "source_tree_sha256": _sha256_json({"files": entries}),
        "source_file_count": len(entries),
    }


def enumerate_sweep_runs(config: Mapping[str, Any]) -> List[SweepRun]:
    """Return the locked LR x weight-decay x seed ordering used by Slurm."""

    training = config["training"]
    runs = []
    run_index = 0
    for learning_rate in training["learning_rates"]:
        for weight_decay in training["weight_decays"]:
            for seed in training["seeds"]:
                runs.append(
                    SweepRun(
                        run_index=run_index,
                        learning_rate=float(learning_rate),
                        weight_decay=float(weight_decay),
                        seed=int(seed),
                    )
                )
                run_index += 1
    expected = int(config["candidate_pool"]["expected_training_runs"])
    if len(runs) != expected:
        raise CandidateTrainingError(
            f"Sweep produced {len(runs)} runs, but config expects {expected}."
        )
    return runs


def get_sweep_run(config: Mapping[str, Any], run_index: int) -> SweepRun:
    runs = enumerate_sweep_runs(config)
    if run_index < 0 or run_index >= len(runs):
        raise IndexError(f"run_index must be in [0, {len(runs) - 1}], found {run_index}.")
    return runs[run_index]


def candidate_training_fingerprint(config: Mapping[str, Any]) -> str:
    relevant = {
        "study": config["study"],
        "model": config["model"],
        "training": config["training"],
        "candidate_pool": config["candidate_pool"],
        "holdout": config["data"]["biased_train_holdout"],
        "selector_visibility": config["data"]["selector_visibility"],
        "source_tree": source_tree_provenance(),
    }
    return _sha256_json(relevant)


def software_versions() -> Dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        # Distribution metadata strips local CUDA build tags.  The module
        # version is the authoritative value for the locked binary runtime.
        "torchvision": str(torchvision.__version__),
        "source_tree_sha256": str(source_tree_provenance()["source_tree_sha256"]),
    }
    for package in ("timm", "pandas", "Pillow"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def software_fingerprint(versions: Mapping[str, str] | None = None) -> str:
    """Return a stable identity for the complete recorded software runtime."""

    return _sha256_json(dict(software_versions() if versions is None else versions))


def validate_runtime_software(config: Mapping[str, Any]) -> Dict[str, str]:
    """Reject a production job whose core packages differ from the locked runtime."""

    observed = software_versions()
    expected = {
        "torch": str(config["cluster"]["torch_version"]),
        "torchvision": str(config["cluster"]["torchvision_version"]),
        "timm": str(config["cluster"]["timm_version"]),
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise CandidateTrainingError(
            f"Locked runtime software mismatch: {mismatches}"
        )
    return observed


class PublicManifestDataset(Dataset):
    """Image dataset that refuses analysis-only group/context columns."""

    def __init__(
        self,
        manifest_path: str | Path,
        expected_study_split: str,
        transform: Any,
        *,
        check_images: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Missing public manifest: {self.manifest_path}")
        frame = pd.read_csv(self.manifest_path)
        missing = sorted(PUBLIC_MANIFEST_REQUIRED.difference(frame.columns))
        if missing:
            raise CandidateTrainingError(
                f"Manifest {self.manifest_path} is missing columns: {missing}"
            )
        leaked = sorted(FORBIDDEN_SELECTOR_COLUMNS.intersection(frame.columns))
        if leaked:
            raise CandidateTrainingError(
                f"Manifest {self.manifest_path} contains analysis-only columns: {leaked}"
            )
        if frame.empty:
            raise CandidateTrainingError(f"Manifest {self.manifest_path} is empty.")
        if frame["sample_id"].duplicated().any():
            raise CandidateTrainingError(
                f"Manifest {self.manifest_path} contains duplicate sample IDs."
            )
        splits = set(frame["study_split"].astype(str))
        if splits != {expected_study_split}:
            raise CandidateTrainingError(
                f"Expected study_split={expected_study_split!r}, found {sorted(splits)}."
            )
        source_splits = set(frame["source_split"].astype(str))
        if source_splits != {"train"}:
            raise CandidateTrainingError(
                "Candidate training and biased validation must originate from "
                f"source_split='train', found {sorted(source_splits)}."
            )
        labels = set(frame["label"].astype(int).unique().tolist())
        if not labels.issubset({0, 1}) or labels != {0, 1}:
            raise CandidateTrainingError(
                f"The binary Waterbirds candidate split must contain labels 0 and 1; "
                f"found {sorted(labels)}."
            )

        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.image_paths = [Path(str(path)) for path in self.frame["image_path"]]
        if check_images:
            missing_images = [str(path) for path in self.image_paths if not path.is_file()]
            if missing_images:
                raise FileNotFoundError(
                    f"Manifest {self.manifest_path} has {len(missing_images)} missing images. "
                    f"First paths: {missing_images[:5]}"
                )
            changed_images = []
            for path, expected in zip(
                self.image_paths, self.frame["image_sha256"].astype(str)
            ):
                observed = _sha256_file(path)
                if observed != expected:
                    changed_images.append(
                        {"path": str(path), "expected": expected, "observed": observed}
                    )
            if changed_images:
                raise CandidateTrainingError(
                    "Image bytes differ from the frozen manifest. First mismatches: "
                    f"{changed_images[:3]}"
                )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.frame.iloc[index]
        path = self.image_paths[index]
        try:
            with Image.open(path) as image:
                image.load()
                image = image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read image {path}: {exc}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"]), str(row["sample_id"])


def build_transforms(config: Mapping[str, Any]) -> Dict[str, Any]:
    model_cfg = config["model"]
    augmentation = config["training"]["augmentation"]
    image_size = int(model_cfg["image_size"])
    normalization = str(augmentation["normalization"])
    if normalization != "imagenet":
        raise CandidateTrainingError(
            f"Unsupported normalization {normalization!r}; first study locks ImageNet."
        )
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    crop = augmentation["train_random_resized_crop"]
    if not crop.get("enabled", False):
        raise CandidateTrainingError("The locked candidate pool requires RandomResizedCrop.")
    scale = tuple(float(value) for value in crop["scale"])
    eval_resize_size = int(augmentation["eval_resize_size"])
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=scale,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(
                p=float(augmentation["train_horizontal_flip_probability"])
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(eval_resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return {"train": train_transform, "eval": eval_transform}


def seed_everything(seed: int, *, deterministic_algorithms: bool, cudnn_benchmark: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        torch.backends.cudnn.deterministic = bool(deterministic_algorithms)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(
    config: Mapping[str, Any],
    train_manifest: str | Path,
    validation_manifest: str | Path,
    train_generator: torch.Generator,
) -> tuple[Dict[str, DataLoader], Dict[str, PublicManifestDataset]]:
    try:
        train_binding = validate_manifest_bundle(
            config, train_manifest, "candidate_train"
        )
        validation_binding = validate_manifest_bundle(
            config, validation_manifest, "biased_validation"
        )
    except ManifestProvenanceError as exc:
        raise CandidateTrainingError(str(exc)) from exc
    if train_binding.bundle_sha256 != validation_binding.bundle_sha256:
        raise CandidateTrainingError(
            "Training and biased-validation manifests do not share one Step-2 bundle."
        )
    transform_map = build_transforms(config)
    datasets = {
        "train": PublicManifestDataset(
            train_manifest, "candidate_train", transform_map["train"]
        ),
        "biased_val": PublicManifestDataset(
            validation_manifest, "biased_validation", transform_map["eval"]
        ),
    }
    training = config["training"]
    batch_size = int(training["batch_size"])
    num_workers = int(training["num_workers"])
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,
        "worker_init_fn": seed_worker,
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            generator=train_generator,
            drop_last=False,
            **common,
        ),
        "biased_val": DataLoader(
            datasets["biased_val"],
            shuffle=False,
            drop_last=False,
            **common,
        ),
    }
    return loaders, datasets


def build_model(config: Mapping[str, Any], *, pretrained: bool) -> nn.Module:
    model_cfg = config["model"]
    if model_cfg["library"] != "timm":
        raise CandidateTrainingError("Step 4 currently supports only the locked timm ViT.")
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            "Step 4 requires timm. Install it in fcv_gh200 with: "
            "python -m pip install 'timm==1.0.28'"
        ) from exc

    model = timm.create_model(
        str(model_cfg["name"]),
        pretrained=pretrained,
        num_classes=int(model_cfg["num_classes"]),
    )
    if not hasattr(model, "patch_embed"):
        raise CandidateTrainingError("Configured timm model does not expose patch_embed.")
    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, int):
        patch_size_tuple = (patch_size, patch_size)
    else:
        patch_size_tuple = tuple(int(value) for value in patch_size)
    expected_patch = int(model_cfg["patch_size"])
    if patch_size_tuple != (expected_patch, expected_patch):
        raise CandidateTrainingError(
            f"Model patch size {patch_size_tuple} does not match config {expected_patch}."
        )
    expected_patches = int(model_cfg["patch_grid_size"]) ** 2
    if int(model.patch_embed.num_patches) != expected_patches:
        raise CandidateTrainingError(
            f"Model exposes {model.patch_embed.num_patches} patches; expected {expected_patches}."
        )
    return model


def warmup_cosine_factor(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> float:
    if total_steps <= 0:
        raise CandidateTrainingError("total_steps must be positive.")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise CandidateTrainingError("warmup_steps must satisfy 0 <= warmup < total.")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise CandidateTrainingError("minimum_ratio must be in [0, 1].")
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    cosine_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / cosine_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    training = config["training"]
    scheduler_cfg = training["scheduler"]
    if scheduler_cfg["name"] != "cosine":
        raise CandidateTrainingError("The first candidate pool locks a cosine scheduler.")
    epochs = int(training["epochs"])
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(scheduler_cfg["warmup_epochs"]) * steps_per_epoch
    base_lr = float(optimizer.defaults["lr"])
    minimum_lr = float(scheduler_cfg["minimum_learning_rate"])
    minimum_ratio = minimum_lr / base_lr if base_lr > 0 else 0.0
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_factor(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            minimum_ratio=minimum_ratio,
        ),
    )


def _autocast_context(device: torch.device, precision: str):
    if precision == "amp_bfloat16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "float32":
        return nullcontext()
    if precision == "amp_bfloat16" and device.type != "cuda":
        return nullcontext()
    raise CandidateTrainingError(f"Unsupported precision mode: {precision}")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    precision: str,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    grad_context = torch.enable_grad if is_training else torch.inference_mode

    with grad_context():
        for images, labels, _sample_ids in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, precision):
                logits = model(images)
                loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            batch_size = int(labels.size(0))
            total_loss += float(loss.detach()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += batch_size

    if total_examples == 0:
        raise CandidateTrainingError("Encountered an empty data loader.")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def _state_dict_cpu(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state_dict.items()}


def _recursive_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _recursive_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_recursive_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_recursive_cpu(item) for item in value)
    return value


def state_dict_sha256(
    state_dict: Mapping[str, torch.Tensor],
    *,
    excluded_keys: set[str] | None = None,
) -> str:
    """Hash tensor names, dtypes, shapes, and exact bytes deterministically."""

    excluded = set() if excluded_keys is None else set(excluded_keys)
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        if key in excluded:
            continue
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def classifier_state_keys(model: nn.Module) -> set[str]:
    """Return state-dict keys belonging to timm's task-specific classifier."""

    if not hasattr(model, "get_classifier"):
        raise CandidateTrainingError("Locked timm model does not expose get_classifier().")
    classifier = model.get_classifier()
    parameter_ids = {id(parameter) for parameter in classifier.parameters()}
    buffer_ids = {id(buffer) for buffer in classifier.buffers()}
    keys = {
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in parameter_ids
    }
    keys.update(
        name for name, buffer in model.named_buffers() if id(buffer) in buffer_ids
    )
    if not keys:
        raise CandidateTrainingError("Could not identify classifier state keys.")
    return keys


def pretrained_backbone_sha256(model: nn.Module) -> str:
    """Hash pretrained weights while excluding the newly initialized task head."""

    return state_dict_sha256(
        model.state_dict(), excluded_keys=classifier_state_keys(model)
    )


def load_pretrained_cache_provenance(
    config: Mapping[str, Any], path: str | Path
) -> Dict[str, Any]:
    """Validate the persisted pretrained-backbone root used by production jobs."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing pretrained provenance artifact: {path}. Run Step 4 cache first."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    valid = (
        payload.get("schema_version") == 1
        and payload.get("artifact_type") == "fcv_vit_pretrained_initialization"
        and payload.get("status") == "cached_and_validated"
        and payload.get("model") == config["model"]["name"]
        and payload.get("model_config") == dict(config["model"])
        and payload.get("runtime_versions") == software_versions()
        and payload.get("software_fingerprint") == software_fingerprint()
        and payload.get("source_tree_sha256")
        == source_tree_provenance()["source_tree_sha256"]
        and isinstance(payload.get("pretrained_backbone_sha256"), str)
        and len(payload["pretrained_backbone_sha256"]) == 64
    )
    if not valid:
        raise CandidateTrainingError(
            f"Pretrained initialization provenance is stale or incompatible: {path}"
        )
    result = dict(payload)
    result["artifact_path"] = str(path)
    result["artifact_sha256"] = _sha256_file(path)
    return result


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _rng_state(train_generator: torch.Generator) -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_generator": train_generator.get_state(),
    }


def _restore_rng_state(payload: Mapping[str, Any], train_generator: torch.Generator) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and payload.get("torch_cuda"):
        torch.cuda.set_rng_state_all(payload["torch_cuda"])
    train_generator.set_state(payload["train_generator"])


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _candidate_checkpoint_payload(
    model: nn.Module,
    config: Mapping[str, Any],
    run: SweepRun,
    epoch: int,
    metric_row: Mapping[str, Any],
    manifest_hashes: Mapping[str, str],
    training_fingerprint: str,
    versions: Mapping[str, str],
    initial_model_state_sha256: str,
    pretrained_backbone_state_sha256: str,
    pretrained_provenance: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "fcv_vit_candidate_checkpoint",
        "candidate_id": run.candidate_id(epoch),
        "run": asdict(run),
        "epoch": epoch,
        "model": dict(config["model"]),
        "training_fingerprint": training_fingerprint,
        "software_versions": dict(versions),
        "source_tree_sha256": source_tree_provenance()["source_tree_sha256"],
        "initial_model_state_sha256": initial_model_state_sha256,
        "pretrained_backbone_sha256": pretrained_backbone_state_sha256,
        "pretrained_provenance_path": (
            pretrained_provenance.get("artifact_path") if pretrained_provenance else None
        ),
        "pretrained_provenance_sha256": (
            pretrained_provenance.get("artifact_sha256") if pretrained_provenance else None
        ),
        "manifest_sha256": dict(manifest_hashes),
        "metrics": dict(metric_row),
        "model_state_dict": _state_dict_cpu(model.state_dict()),
    }


def _load_trusted_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_checkpoint_metric_row(
    checkpoint: Mapping[str, Any],
    csv_row: Mapping[str, Any],
    run: SweepRun,
) -> None:
    """Prove that mutable CSV selector values equal checkpoint-bound values."""

    candidate_id = str(csv_row["candidate_id"])
    epoch = int(csv_row["epoch"])
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("artifact_type") != "fcv_vit_candidate_checkpoint"
        or checkpoint.get("candidate_id") != candidate_id
        or checkpoint.get("run") != asdict(run)
        or int(checkpoint.get("epoch", -1)) != epoch
    ):
        raise CandidateTrainingError(
            f"Checkpoint identity does not match metrics row for {candidate_id}."
        )
    embedded = checkpoint.get("metrics")
    if not isinstance(embedded, Mapping):
        raise CandidateTrainingError(
            f"Checkpoint {candidate_id} does not contain its metric row."
        )
    for column in METRIC_COLUMNS:
        if column == "checkpoint_sha256":
            # A file cannot recursively contain its own SHA-256.  This value is
            # bound externally and independently verified against the bytes.
            continue
        expected = embedded.get(column)
        observed = csv_row[column]
        if column in {
            "run_index",
            "epoch",
            "seed",
        }:
            matches = int(expected) == int(observed)
        elif column in {
            "learning_rate",
            "weight_decay",
            "train_loss",
            "train_accuracy",
            "biased_val_loss",
            "biased_val_accuracy",
            "lr_epoch_start",
            "lr_epoch_end",
            "epoch_seconds",
        }:
            matches = bool(
                np.isclose(float(expected), float(observed), rtol=0.0, atol=1.0e-12)
            )
        else:
            matches = str(expected) == str(observed)
        if not matches:
            raise CandidateTrainingError(
                f"Metric {column!r} differs between CSV and checkpoint for "
                f"{candidate_id}: csv={observed!r}, checkpoint={expected!r}."
            )


def _completed_run_from_disk(
    metrics_path: Path,
    checkpoints_dir: Path,
    summary_path: Path,
    expected_training_epochs: int,
    expected_candidate_epochs: List[int],
    run: SweepRun,
    training_fingerprint: str,
    manifest_hashes: Mapping[str, str],
    expected_software_versions: Mapping[str, str],
    pretrained_provenance: Mapping[str, Any] | None,
) -> bool:
    if not metrics_path.is_file() or not summary_path.is_file():
        return False
    metrics = pd.read_csv(metrics_path)
    if len(metrics) != len(expected_candidate_epochs) or metrics[
        "epoch"
    ].astype(int).tolist() != expected_candidate_epochs:
        return False
    checkpoint_provenance = []
    expected_source_tree = source_tree_provenance()["source_tree_sha256"]
    for row in metrics.itertuples(index=False):
        checkpoint_path = Path(str(row.checkpoint_path))
        if (
            not checkpoint_path.is_file()
            or _sha256_file(checkpoint_path) != str(row.checkpoint_sha256)
        ):
            return False
        checkpoint = _load_trusted_checkpoint(checkpoint_path)
        _validate_checkpoint_metric_row(checkpoint, row._asdict(), run)
        if (
            checkpoint.get("training_fingerprint") != training_fingerprint
            or checkpoint.get("software_versions")
            != dict(expected_software_versions)
            or checkpoint.get("source_tree_sha256") != expected_source_tree
            or checkpoint.get("manifest_sha256") != dict(manifest_hashes)
        ):
            raise CandidateTrainingError(
                f"Completed checkpoint provenance is stale: {checkpoint_path}"
            )
        checkpoint_provenance.append(
            {
                "initial_model_state_sha256": checkpoint.get(
                    "initial_model_state_sha256"
                ),
                "pretrained_backbone_sha256": checkpoint.get(
                    "pretrained_backbone_sha256"
                ),
                "pretrained_provenance_path": checkpoint.get(
                    "pretrained_provenance_path"
                ),
                "pretrained_provenance_sha256": checkpoint.get(
                    "pretrained_provenance_sha256"
                ),
            }
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("run") != asdict(run):
        raise CandidateTrainingError("Completed-run summary has a different run specification.")
    if int(summary.get("epochs", -1)) != expected_training_epochs:
        raise CandidateTrainingError("Completed-run summary has a different training length.")
    if summary.get("candidate_epochs") != expected_candidate_epochs:
        raise CandidateTrainingError("Completed-run summary has different candidate epochs.")
    if summary.get("training_fingerprint") != training_fingerprint:
        raise CandidateTrainingError("Completed-run summary has a stale training fingerprint.")
    if summary.get("manifest_sha256") != dict(manifest_hashes):
        raise CandidateTrainingError("Completed-run summary references different manifests.")
    if summary.get("software_versions") != dict(expected_software_versions):
        raise CandidateTrainingError(
            "Completed-run summary software differs from the active environment."
        )
    if summary.get("software_fingerprint") != software_fingerprint(
        expected_software_versions
    ):
        raise CandidateTrainingError("Completed-run software fingerprint is stale.")
    if summary.get("metrics_path") != str(metrics_path.resolve()) or summary.get(
        "metrics_sha256"
    ) != _sha256_file(metrics_path):
        raise CandidateTrainingError("Completed-run metrics CSV is stale.")
    if summary.get("source_tree_sha256") != expected_source_tree:
        raise CandidateTrainingError("Completed-run source-tree provenance is stale.")
    if pretrained_provenance is not None:
        if (
            summary.get("pretrained_provenance_path")
            != pretrained_provenance["artifact_path"]
            or summary.get("pretrained_provenance_sha256")
            != pretrained_provenance["artifact_sha256"]
            or summary.get("pretrained_backbone_sha256")
            != pretrained_provenance["pretrained_backbone_sha256"]
        ):
            raise CandidateTrainingError(
                "Completed-run pretrained initialization provenance is stale."
            )
    expected_checkpoint_provenance = {
        "initial_model_state_sha256": summary.get("initial_model_state_sha256"),
        "pretrained_backbone_sha256": summary.get("pretrained_backbone_sha256"),
        "pretrained_provenance_path": summary.get("pretrained_provenance_path"),
        "pretrained_provenance_sha256": summary.get(
            "pretrained_provenance_sha256"
        ),
    }
    if any(
        provenance != expected_checkpoint_provenance
        for provenance in checkpoint_provenance
    ):
        raise CandidateTrainingError(
            "Completed-run summary disagrees with checkpoint initialization provenance."
        )
    return True


def train_candidate_run(
    config: Mapping[str, Any],
    run: SweepRun,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    candidate_root: str | Path,
    *,
    device_name: str = "cuda",
    stop_after_epoch: int | None = None,
    simulate_interruption_after_resume_epoch: int | None = None,
    pretrained_provenance_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Train or resume one of the 27 locked candidate-pool runs."""

    train_manifest = Path(train_manifest)
    validation_manifest = Path(validation_manifest)
    candidate_root = Path(candidate_root)
    run_dir = candidate_root / run.run_id
    checkpoints_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "run_summary.json"
    resume_path = run_dir / "resume_state.pt"
    epochs = int(config["training"]["epochs"])
    selected_epochs = candidate_epochs(config)
    selected_epoch_set = set(selected_epochs)
    output_root = Path(config["paths"]["output_root"])
    if stop_after_epoch is not None and not (1 <= stop_after_epoch < epochs):
        raise CandidateTrainingError(
            f"stop_after_epoch must be in [1, {epochs - 1}] for a resumable smoke run."
        )
    if simulate_interruption_after_resume_epoch is not None and not (
        1 <= simulate_interruption_after_resume_epoch < epochs
    ):
        raise CandidateTrainingError(
            "simulate_interruption_after_resume_epoch must identify a non-final epoch."
        )
    training_fingerprint = candidate_training_fingerprint(config)
    try:
        train_binding = validate_manifest_bundle(
            config, train_manifest, "candidate_train"
        )
        validation_binding = validate_manifest_bundle(
            config, validation_manifest, "biased_validation"
        )
    except ManifestProvenanceError as exc:
        raise CandidateTrainingError(str(exc)) from exc
    if train_binding.bundle_sha256 != validation_binding.bundle_sha256:
        raise CandidateTrainingError(
            "Training and biased-validation manifests do not share one Step-2 bundle."
        )
    manifest_hashes = {
        "candidate_train": train_binding.manifest_sha256,
        "biased_validation": validation_binding.manifest_sha256,
        "manifest_bundle": train_binding.bundle_sha256,
        "original_metadata": train_binding.original_metadata_sha256,
        "split_indices": train_binding.split_indices_sha256,
        "split_summary": train_binding.split_summary_sha256,
    }
    versions = software_versions()
    pretrained_provenance = (
        load_pretrained_cache_provenance(config, pretrained_provenance_path)
        if pretrained_provenance_path is not None
        else None
    )
    # Revalidate the immutable image roots even on a completed-run reuse.  A
    # manifest hash alone binds the expected image hashes, but this check also
    # proves that the bytes currently reachable through those paths still
    # match before we accept or extend any campaign artifact.
    PublicManifestDataset(
        train_manifest, "candidate_train", transform=None, check_images=True
    )
    PublicManifestDataset(
        validation_manifest,
        "biased_validation",
        transform=None,
        check_images=True,
    )
    if _completed_run_from_disk(
        metrics_path,
        checkpoints_dir,
        summary_path,
        epochs,
        selected_epochs,
        run,
        training_fingerprint,
        manifest_hashes,
        versions,
        pretrained_provenance,
    ):
        return {
            "status": "already_complete",
            "run_id": run.run_id,
            "metrics": str(metrics_path.resolve()),
            "summary": str(summary_path.resolve()),
        }
    if metrics_path.exists() and not resume_path.exists():
        raise CandidateTrainingError(
            f"Partial metrics exist without a resume state in {run_dir}. Refusing to "
            "restart this run under the same ID."
        )
    assert_storage_budget(config, output_root, stage=f"candidate_run:{run.run_id}")

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(device_name)
    reproducibility = config["reproducibility"]
    seed_everything(
        run.seed,
        deterministic_algorithms=bool(reproducibility["deterministic_algorithms"]),
        cudnn_benchmark=bool(reproducibility["cudnn_benchmark"]),
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(run.seed)
    loaders, datasets = build_dataloaders(
        config, train_manifest, validation_manifest, train_generator
    )

    resuming = resume_path.is_file()
    model = build_model(config, pretrained=bool(config["model"]["pretrained"]) and not resuming)
    initial_model_state_sha256 = ""
    pretrained_backbone_state_sha256 = ""
    if not resuming:
        initial_model_state_sha256 = state_dict_sha256(model.state_dict())
        pretrained_backbone_state_sha256 = (
            pretrained_backbone_sha256(model)
            if pretrained_provenance is not None
            else initial_model_state_sha256
        )
        if (
            pretrained_provenance is not None
            and pretrained_backbone_state_sha256
            != pretrained_provenance["pretrained_backbone_sha256"]
        ):
            raise CandidateTrainingError(
                "Loaded pretrained backbone bytes differ from the persisted Step 4 "
                "cache provenance artifact."
            )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run.learning_rate,
        weight_decay=run.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config, len(loaders["train"]))
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["training"]["augmentation"]["label_smoothing"])
    )

    metric_rows: List[Dict[str, Any]] = []
    start_epoch = 1
    if resuming:
        resume = _load_trusted_checkpoint(resume_path)
        if resume.get("artifact_type") != "fcv_vit_candidate_resume_state":
            raise CandidateTrainingError(f"Unexpected resume artifact: {resume_path}")
        if resume.get("run") != asdict(run):
            raise CandidateTrainingError("Resume state run specification does not match.")
        if resume.get("training_fingerprint") != training_fingerprint:
            raise CandidateTrainingError("Resume state training configuration has changed.")
        if resume.get("software_versions") != versions:
            raise CandidateTrainingError(
                "Resume state software versions differ from the active environment."
            )
        if resume.get("manifest_sha256") != manifest_hashes:
            raise CandidateTrainingError("Resume state manifests have changed.")
        if resume.get("source_tree_sha256") != source_tree_provenance()[
            "source_tree_sha256"
        ]:
            raise CandidateTrainingError("Resume state source tree has changed.")
        initial_model_state_sha256 = str(
            resume.get("initial_model_state_sha256", "")
        )
        pretrained_backbone_state_sha256 = str(
            resume.get("pretrained_backbone_sha256", "")
        )
        if len(initial_model_state_sha256) != 64 or len(
            pretrained_backbone_state_sha256
        ) != 64:
            raise CandidateTrainingError(
                "Resume state lacks initialization provenance hashes."
            )
        if pretrained_provenance is not None and (
            resume.get("pretrained_provenance_path")
            != pretrained_provenance["artifact_path"]
            or resume.get("pretrained_provenance_sha256")
            != pretrained_provenance["artifact_sha256"]
            or pretrained_backbone_state_sha256
            != pretrained_provenance["pretrained_backbone_sha256"]
        ):
            raise CandidateTrainingError(
                "Resume state pretrained initialization provenance has changed."
            )
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        metric_rows = list(resume["metric_rows"])
        completed_epoch = int(resume["completed_epoch"])
        expected_completed_candidates = [
            epoch for epoch in selected_epochs if epoch <= completed_epoch
        ]
        if [int(row["epoch"]) for row in metric_rows] != expected_completed_candidates:
            raise CandidateTrainingError(
                "Resume candidate rows disagree with its completed training epoch."
            )
        if metrics_path.is_file():
            disk_metrics = pd.read_csv(metrics_path)
            if len(disk_metrics) > len(metric_rows):
                raise CandidateTrainingError(
                    "metrics.csv is ahead of the committed resume state. Refusing "
                    "an ambiguous recovery."
                )
            for index, disk_row in enumerate(disk_metrics.itertuples(index=False)):
                expected_row = metric_rows[index]
                if str(disk_row.candidate_id) != str(expected_row["candidate_id"]):
                    raise CandidateTrainingError(
                        "metrics.csv is not a prefix of the committed resume state."
                    )
                checkpoint_path = Path(str(disk_row.checkpoint_path))
                if (
                    not checkpoint_path.is_file()
                    or _sha256_file(checkpoint_path)
                    != str(expected_row["checkpoint_sha256"])
                ):
                    raise CandidateTrainingError(
                        "A recovery checkpoint is missing or has changed bytes."
                    )
                checkpoint = _load_trusted_checkpoint(checkpoint_path)
                _validate_checkpoint_metric_row(checkpoint, disk_row._asdict(), run)
            # The resume state is the epoch commit marker.  Repair a CSV that
            # was interrupted after resume commit but before CSV publication.
            if len(disk_metrics) < len(metric_rows):
                _atomic_csv(
                    pd.DataFrame(metric_rows, columns=METRIC_COLUMNS), metrics_path
                )
        elif metric_rows:
            _atomic_csv(pd.DataFrame(metric_rows, columns=METRIC_COLUMNS), metrics_path)
        start_epoch = completed_epoch + 1
        _restore_rng_state(resume["rng_state"], train_generator)

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    precision = str(config["training"]["precision"])
    started = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.time()
        lr_epoch_start = float(optimizer.param_groups[0]["lr"])
        train_metrics = _run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            precision,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        candidate_payload = None
        if epoch in selected_epoch_set:
            assert_storage_budget(
                config,
                output_root,
                stage=f"candidate_checkpoint:{run.candidate_id(epoch)}",
            )
            # Only the three epochs fixed before training enter any selector.
            # Their ordinary holdout pass is always float32.
            validation_metrics = _run_epoch(
                model,
                loaders["biased_val"],
                criterion,
                device,
                "float32",
            )
            checkpoint_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
            metric_row = {
                "run_index": run.run_index,
                "candidate_id": run.candidate_id(epoch),
                "epoch": epoch,
                "model_name": str(config["model"]["name"]),
                "seed": run.seed,
                "learning_rate": run.learning_rate,
                "weight_decay": run.weight_decay,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "biased_val_loss": validation_metrics["loss"],
                "biased_val_accuracy": validation_metrics["accuracy"],
                "lr_epoch_start": lr_epoch_start,
                "lr_epoch_end": float(optimizer.param_groups[0]["lr"]),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": "pending",
                "epoch_seconds": time.time() - epoch_started,
            }
            candidate_payload = _candidate_checkpoint_payload(
                model,
                config,
                run,
                epoch,
                metric_row,
                manifest_hashes,
                training_fingerprint,
                versions,
                initial_model_state_sha256,
                pretrained_backbone_state_sha256,
                pretrained_provenance,
            )
            _atomic_torch_save(candidate_payload, checkpoint_path)
            metric_row["checkpoint_sha256"] = _sha256_file(checkpoint_path)
            metric_rows.append(metric_row)
        resume_payload = {
            "schema_version": 1,
            "artifact_type": "fcv_vit_candidate_resume_state",
            "run": asdict(run),
            "completed_epoch": epoch,
            "training_fingerprint": training_fingerprint,
            "software_versions": versions,
            "source_tree_sha256": source_tree_provenance()["source_tree_sha256"],
            "initial_model_state_sha256": initial_model_state_sha256,
            "pretrained_backbone_sha256": pretrained_backbone_state_sha256,
            "pretrained_provenance_path": (
                pretrained_provenance.get("artifact_path")
                if pretrained_provenance
                else None
            ),
            "pretrained_provenance_sha256": (
                pretrained_provenance.get("artifact_sha256")
                if pretrained_provenance
                else None
            ),
            "manifest_sha256": manifest_hashes,
            "model_state_dict": (
                candidate_payload["model_state_dict"]
                if candidate_payload is not None
                else _state_dict_cpu(model.state_dict())
            ),
            "optimizer_state_dict": _recursive_cpu(optimizer.state_dict()),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": _rng_state(train_generator),
            "metric_rows": metric_rows,
        }
        _atomic_torch_save(resume_payload, resume_path)
        if simulate_interruption_after_resume_epoch == epoch:
            raise CandidateTrainingError(
                "Simulated abrupt interruption after resume-state commit and before "
                "metrics.csv commit. Reinvoke the run to exercise recovery."
            )
        if metric_rows:
            _atomic_csv(pd.DataFrame(metric_rows, columns=METRIC_COLUMNS), metrics_path)
        if candidate_payload is not None:
            print(
                f"[CANDIDATE] run={run.run_index:02d} epoch={epoch:02d}/{epochs} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"biased_val_acc={validation_metrics['accuracy']:.4f} "
                f"biased_val_loss={validation_metrics['loss']:.6f} "
                f"lr={metric_row['lr_epoch_end']:.3e}",
                flush=True,
            )
        else:
            print(
                f"[TRAIN] run={run.run_index:02d} epoch={epoch:02d}/{epochs} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"lr={float(optimizer.param_groups[0]['lr']):.3e}",
                flush=True,
            )
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            return {
                "status": "paused_for_smoke_resume",
                "run_id": run.run_id,
                "completed_epoch": epoch,
                "metrics": str(metrics_path.resolve()),
                "resume_state": str(resume_path.resolve()),
            }

    metrics = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    best_accuracy = metrics.sort_values(
        ["biased_val_accuracy", "biased_val_loss", "epoch"],
        ascending=[False, True, True],
    ).iloc[0]
    best_loss = metrics.sort_values(
        ["biased_val_loss", "biased_val_accuracy", "epoch"],
        ascending=[True, False, True],
    ).iloc[0]
    summary = {
        "status": "complete",
        "run": asdict(run),
        "run_id": run.run_id,
        "training_fingerprint": training_fingerprint,
        "software_versions": versions,
        "software_fingerprint": software_fingerprint(versions),
        "source_tree_sha256": source_tree_provenance()["source_tree_sha256"],
        "initial_model_state_sha256": initial_model_state_sha256,
        "pretrained_backbone_sha256": pretrained_backbone_state_sha256,
        "pretrained_provenance_path": (
            pretrained_provenance.get("artifact_path") if pretrained_provenance else None
        ),
        "pretrained_provenance_sha256": (
            pretrained_provenance.get("artifact_sha256") if pretrained_provenance else None
        ),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "manifest_sha256": manifest_hashes,
        "dataset_sizes": {key: len(value) for key, value in datasets.items()},
        "epochs": epochs,
        "candidate_epochs": selected_epochs,
        "candidate_count": int(len(metrics)),
        "best_biased_val_accuracy_candidate": str(best_accuracy["candidate_id"]),
        "best_biased_val_accuracy": float(best_accuracy["biased_val_accuracy"]),
        "best_biased_val_loss_candidate": str(best_loss["candidate_id"]),
        "best_biased_val_loss": float(best_loss["biased_val_loss"]),
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": _sha256_file(metrics_path),
        "resume_state_path": str(resume_path.resolve()),
        "seconds_this_invocation": time.time() - started,
    }
    _atomic_json(summary, summary_path)
    return summary


def aggregate_candidate_metrics(
    config: Mapping[str, Any],
    candidate_root: str | Path,
    output_csv: str | Path,
    summary_path: str | Path,
    *,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    """Combine per-run metric files after the Slurm array has finished."""

    candidate_root = Path(candidate_root)
    output_csv = Path(output_csv)
    summary_path = Path(summary_path)
    selected_epochs = candidate_epochs(config)
    frames = []
    missing_runs = []
    incomplete_runs = []
    missing_checkpoint_paths = []
    manifest_hashes_seen = []
    software_versions_seen = []
    pretrained_backbones_seen = []
    initialization_by_seed: Dict[int, set[str]] = {}
    pretrained_provenance_seen = []
    expected_fingerprint = candidate_training_fingerprint(config)
    current_software_versions = software_versions()
    current_source_tree_sha256 = source_tree_provenance()["source_tree_sha256"]
    for run in enumerate_sweep_runs(config):
        run_dir = candidate_root / run.run_id
        metrics_path = run_dir / "metrics.csv"
        run_summary_path = run_dir / "run_summary.json"
        if not metrics_path.is_file():
            missing_runs.append(run.run_index)
            continue
        frame = pd.read_csv(metrics_path)
        missing_columns = sorted(set(METRIC_COLUMNS).difference(frame.columns))
        if missing_columns:
            raise CandidateTrainingError(
                f"{metrics_path} is missing metric columns: {missing_columns}"
            )
        expected_epochs = set(selected_epochs)
        expected_candidate_ids = {run.candidate_id(epoch) for epoch in selected_epochs}
        run_is_incomplete = (
            len(frame) != len(selected_epochs)
            or set(frame["epoch"].astype(int)) != expected_epochs
            or set(frame["candidate_id"].astype(str)) != expected_candidate_ids
        )
        if set(frame["run_index"].astype(int)) != {run.run_index}:
            raise CandidateTrainingError(f"{metrics_path} has the wrong run_index values.")
        if set(frame["seed"].astype(int)) != {run.seed}:
            raise CandidateTrainingError(f"{metrics_path} has the wrong seed values.")
        if not np.allclose(frame["learning_rate"], run.learning_rate, rtol=0.0, atol=0.0):
            raise CandidateTrainingError(f"{metrics_path} has the wrong learning rate.")
        if not np.allclose(frame["weight_decay"], run.weight_decay, rtol=0.0, atol=0.0):
            raise CandidateTrainingError(f"{metrics_path} has the wrong weight decay.")
        missing_for_run = []
        checkpoint_provenance = []
        for row in frame.itertuples(index=False):
            checkpoint_path = Path(str(row.checkpoint_path))
            if not checkpoint_path.is_file():
                missing_for_run.append(str(checkpoint_path))
            elif _sha256_file(checkpoint_path) != str(row.checkpoint_sha256):
                raise CandidateTrainingError(
                    f"Checkpoint bytes changed after training: {checkpoint_path}"
                )
            else:
                checkpoint = _load_trusted_checkpoint(checkpoint_path)
                _validate_checkpoint_metric_row(checkpoint, row._asdict(), run)
                if checkpoint.get("training_fingerprint") != expected_fingerprint:
                    raise CandidateTrainingError(
                        f"Checkpoint has stale source/config provenance: {checkpoint_path}"
                    )
                if checkpoint.get("software_versions") != current_software_versions:
                    raise CandidateTrainingError(
                        f"Checkpoint software provenance is stale: {checkpoint_path}"
                    )
                checkpoint_provenance.append(
                    {
                        "source_tree_sha256": checkpoint.get("source_tree_sha256"),
                        "initial_model_state_sha256": checkpoint.get(
                            "initial_model_state_sha256"
                        ),
                        "pretrained_backbone_sha256": checkpoint.get(
                            "pretrained_backbone_sha256"
                        ),
                        "pretrained_provenance_path": checkpoint.get(
                            "pretrained_provenance_path"
                        ),
                        "pretrained_provenance_sha256": checkpoint.get(
                            "pretrained_provenance_sha256"
                        ),
                        "manifest_sha256": checkpoint.get("manifest_sha256"),
                    }
                )
        if missing_for_run:
            missing_checkpoint_paths.extend(missing_for_run)
            run_is_incomplete = True
        if not run_summary_path.is_file():
            run_is_incomplete = True
        else:
            with run_summary_path.open("r", encoding="utf-8") as handle:
                run_summary = json.load(handle)
            if run_summary.get("run") != asdict(run):
                raise CandidateTrainingError(
                    f"{run_summary_path} has the wrong run specification."
                )
            if run_summary.get("candidate_epochs") != selected_epochs:
                raise CandidateTrainingError(
                    f"{run_summary_path} has the wrong candidate epochs."
                )
            if run_summary.get("training_fingerprint") != expected_fingerprint:
                raise CandidateTrainingError(
                    f"{run_summary_path} has a stale training fingerprint."
                )
            run_manifest_hashes = run_summary.get("manifest_sha256")
            if not isinstance(run_manifest_hashes, dict):
                raise CandidateTrainingError(
                    f"{run_summary_path} does not contain manifest hashes."
                )
            manifest_hashes_seen.append(run_manifest_hashes)
            run_software_versions = run_summary.get("software_versions")
            if not isinstance(run_software_versions, dict):
                raise CandidateTrainingError(
                    f"{run_summary_path} does not contain software versions."
                )
            if run_summary.get("software_fingerprint") != software_fingerprint(
                run_software_versions
            ):
                raise CandidateTrainingError(
                    f"{run_summary_path} has a stale software fingerprint."
                )
            software_versions_seen.append(run_software_versions)
            if (
                run_summary.get("metrics_path") != str(metrics_path.resolve())
                or run_summary.get("metrics_sha256") != _sha256_file(metrics_path)
            ):
                raise CandidateTrainingError(
                    f"{run_summary_path} does not bind the current metrics CSV."
                )
            if run_summary.get("source_tree_sha256") != current_source_tree_sha256:
                raise CandidateTrainingError(
                    f"{run_summary_path} has stale source-tree provenance."
                )
            backbone_hash = str(run_summary.get("pretrained_backbone_sha256", ""))
            initialization_hash = str(
                run_summary.get("initial_model_state_sha256", "")
            )
            if len(backbone_hash) != 64 or len(initialization_hash) != 64:
                raise CandidateTrainingError(
                    f"{run_summary_path} lacks initialization hashes."
                )
            pretrained_backbones_seen.append(backbone_hash)
            initialization_by_seed.setdefault(run.seed, set()).add(
                initialization_hash
            )
            pretrained_provenance_seen.append(
                {
                    "path": run_summary.get("pretrained_provenance_path"),
                    "sha256": run_summary.get("pretrained_provenance_sha256"),
                }
            )
            expected_checkpoint_provenance = {
                "source_tree_sha256": run_summary.get("source_tree_sha256"),
                "initial_model_state_sha256": initialization_hash,
                "pretrained_backbone_sha256": backbone_hash,
                "pretrained_provenance_path": run_summary.get(
                    "pretrained_provenance_path"
                ),
                "pretrained_provenance_sha256": run_summary.get(
                    "pretrained_provenance_sha256"
                ),
                "manifest_sha256": run_manifest_hashes,
            }
            if any(
                provenance != expected_checkpoint_provenance
                for provenance in checkpoint_provenance
            ):
                raise CandidateTrainingError(
                    f"{run_summary_path} disagrees with checkpoint initialization "
                    "or source provenance."
                )
        if run_is_incomplete:
            incomplete_runs.append(run.run_index)
        frames.append(frame[METRIC_COLUMNS])

    if (missing_runs or incomplete_runs) and not allow_incomplete:
        raise CandidateTrainingError(
            f"Candidate pool is incomplete: missing_runs={missing_runs}, "
            f"incomplete_runs={incomplete_runs}."
        )
    if not frames:
        raise CandidateTrainingError("No per-run candidate metrics were found.")

    combined = pd.concat(frames, ignore_index=True)
    if combined["candidate_id"].duplicated().any():
        raise CandidateTrainingError("Aggregated candidate IDs are not unique.")
    manifest_fingerprints = {
        json.dumps(item, sort_keys=True) for item in manifest_hashes_seen if item is not None
    }
    if len(manifest_fingerprints) > 1:
        raise CandidateTrainingError("Candidate runs reference different manifest files.")
    software_fingerprints = {
        software_fingerprint(item) for item in software_versions_seen
    }
    if len(software_fingerprints) > 1:
        raise CandidateTrainingError(
            "Candidate runs were produced by different software environments."
        )
    if len(set(pretrained_backbones_seen)) > 1:
        raise CandidateTrainingError(
            "Candidate runs used different pretrained backbone initializations."
        )
    inconsistent_seed_initializations = {
        seed: sorted(values)
        for seed, values in initialization_by_seed.items()
        if len(values) > 1
    }
    if inconsistent_seed_initializations:
        raise CandidateTrainingError(
            "Runs sharing a seed used different full initial states: "
            f"{inconsistent_seed_initializations}"
        )
    provenance_fingerprints = {
        json.dumps(item, sort_keys=True) for item in pretrained_provenance_seen
    }
    if len(provenance_fingerprints) > 1:
        raise CandidateTrainingError(
            "Candidate runs reference different pretrained cache artifacts."
        )
    if bool(config["model"].get("pretrained", False)):
        if not pretrained_provenance_seen:
            raise CandidateTrainingError(
                "Pretrained candidate pool has no persisted initialization provenance."
            )
        provenance = pretrained_provenance_seen[0]
        if not provenance.get("path") or not provenance.get("sha256"):
            raise CandidateTrainingError(
                "Pretrained candidate pool has an unbound initialization artifact."
            )
        validated_pretrained = load_pretrained_cache_provenance(
            config, provenance["path"]
        )
        if (
            validated_pretrained["artifact_sha256"] != provenance["sha256"]
            or validated_pretrained["pretrained_backbone_sha256"]
            != pretrained_backbones_seen[0]
        ):
            raise CandidateTrainingError(
                "Candidate pool pretrained initialization artifact changed."
            )
    combined = combined.sort_values(["run_index", "epoch"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv(combined, output_csv)
    expected_candidates = int(config["candidate_pool"]["expected_candidate_checkpoints"])
    summary = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_candidate_pool_summary",
        "status": "complete" if not missing_runs and not incomplete_runs else "incomplete",
        "candidate_count": int(len(combined)),
        "expected_candidate_count": expected_candidates,
        "candidate_epochs": selected_epochs,
        "complete_run_count": int(
            len(enumerate_sweep_runs(config)) - len(missing_runs) - len(incomplete_runs)
        ),
        "expected_run_count": int(config["candidate_pool"]["expected_training_runs"]),
        "missing_runs": missing_runs,
        "incomplete_runs": incomplete_runs,
        "missing_checkpoint_count": len(missing_checkpoint_paths),
        "missing_checkpoint_preview": missing_checkpoint_paths[:10],
        "training_fingerprint": expected_fingerprint,
        "manifest_sha256": manifest_hashes_seen[0] if manifest_hashes_seen else None,
        "software_versions": software_versions_seen[0] if software_versions_seen else None,
        "software_fingerprint": (
            next(iter(software_fingerprints)) if software_fingerprints else None
        ),
        "source_tree_sha256": current_source_tree_sha256,
        "pretrained_backbone_sha256": (
            pretrained_backbones_seen[0] if pretrained_backbones_seen else None
        ),
        "pretrained_provenance": (
            pretrained_provenance_seen[0] if pretrained_provenance_seen else None
        ),
        "initial_model_state_sha256_by_seed": {
            str(seed): next(iter(values))
            for seed, values in sorted(initialization_by_seed.items())
            if values
        },
        "output_csv": str(output_csv.resolve()),
        "output_csv_sha256": _sha256_file(output_csv),
        "checkpoint_hashes_bound": True,
        "checkpoint_metric_rows_reproduced": True,
        "source_images_bound_by_manifest_hashes": True,
        "pretrained_initialization_artifact_validated": bool(
            pretrained_provenance_seen
        ),
    }
    _atomic_json(summary, summary_path)
    return summary
