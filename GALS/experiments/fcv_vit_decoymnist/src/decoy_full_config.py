"""Locked configuration loading for the full online DecoyMNIST FCV campaign."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml


class ConfigError(ValueError):
    """Raised when configuration violates the precommitted campaign protocol."""


@dataclass(frozen=True)
class CampaignRun:
    run_index: int
    learning_rate: float
    weight_decay: float
    crop_scale_min: float
    seed: int

    @staticmethod
    def _slug(value: float) -> str:
        return f"{value:.8g}".replace("-", "m").replace(".", "p")

    @property
    def run_id(self) -> str:
        return (
            f"run_{self.run_index:03d}"
            f"_lr_{self._slug(self.learning_rate)}"
            f"_wd_{self._slug(self.weight_decay)}"
            f"_crop_{self._slug(self.crop_scale_min)}"
            f"_seed_{self.seed}"
        )

    def candidate_id(self, epoch: int) -> str:
        if epoch < 1:
            raise ValueError("epoch must be positive")
        return f"{self.run_id}_epoch_{int(epoch):03d}"


def _expand_paths(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_paths(item) for key, item in value.items()}
    return value


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if not isinstance(mapping, Mapping):
        raise ConfigError(f"{context} must be a mapping.")
    if key not in mapping:
        raise ConfigError(f"Missing required key: {context}.{key}")
    return mapping[key]


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _require(config, key, "config")
    if not isinstance(value, Mapping):
        raise ConfigError(f"config.{key} must be a mapping.")
    return value


def _exact_sequence(
    mapping: Mapping[str, Any], key: str, expected: Sequence[Any], context: str
) -> None:
    value = _require(mapping, key, context)
    if not isinstance(value, list) or value != list(expected):
        raise ConfigError(f"{context}.{key} must equal {list(expected)!r}.")


def _positive_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = _require(mapping, key, context)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{context}.{key} must be a positive integer.")
    return int(value)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if key != "_provenance"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def candidate_epochs(config: Mapping[str, Any]) -> List[int]:
    pool = _mapping(config, "candidate_pool")
    training = _mapping(config, "training")
    values = _require(pool, "candidate_epochs", "candidate_pool")
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise ConfigError("candidate_pool.candidate_epochs must be integer epochs.")
    epochs = [int(value) for value in values]
    if not epochs or epochs != sorted(set(epochs)):
        raise ConfigError("candidate epochs must be nonempty, unique, and increasing.")
    total_epochs = _positive_int(training, "epochs", "training")
    if epochs[0] < 1 or epochs[-1] > total_epochs:
        raise ConfigError("candidate epochs must lie within the training schedule.")
    return epochs


def enumerate_runs(config: Mapping[str, Any]) -> List[CampaignRun]:
    training = _mapping(config, "training")
    learning_rates = [float(value) for value in training["learning_rates"]]
    weight_decays = [float(value) for value in training["weight_decays"]]
    crop_scales = [float(value) for value in training["crop_scale_mins"]]
    seeds = [int(value) for value in training["seeds"]]
    runs = []
    for learning_rate, weight_decay, crop_scale, seed in itertools.product(
        learning_rates, weight_decays, crop_scales, seeds
    ):
        runs.append(
            CampaignRun(
                run_index=len(runs),
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                crop_scale_min=crop_scale,
                seed=seed,
            )
        )
    return runs


def validate_config(config: Mapping[str, Any]) -> None:
    study = _mapping(config, "study")
    cluster = _mapping(config, "cluster")
    paths = _mapping(config, "paths")
    data = _mapping(config, "data")
    model = _mapping(config, "model")
    training = _mapping(config, "training")
    pool = _mapping(config, "candidate_pool")
    fcv = _mapping(config, "fcv")
    evaluation = _mapping(config, "evaluation")
    storage = _mapping(config, "storage")
    execution = _mapping(config, "execution")
    reproducibility = _mapping(config, "reproducibility")

    expected_study = {
        "id": "fcv_vit_decoymnist_full_online",
        "protocol_version": 1,
        "dataset": "decoymnist",
        "objective": "online_model_selection_under_reversed_corner_shortcut",
    }
    for key, expected in expected_study.items():
        if study.get(key) != expected:
            raise ConfigError(f"study.{key} is locked to {expected!r}.")

    expected_cluster = {
        "name": "tigris",
        "account": "reu-aisocial",
        "partition": "tigris",
        "gres": "gpu:gh200:1",
        "torch_version": "2.11.0+cu130",
        "torchvision_version": "0.26.0+cu130",
        "timm_version": "1.0.28",
    }
    for key, expected in expected_cluster.items():
        if cluster.get(key) != expected:
            raise ConfigError(f"cluster.{key} is locked to {expected!r}.")
    for key in ("conda_environment", "python"):
        if not isinstance(_require(cluster, key, "cluster"), str):
            raise ConfigError(f"cluster.{key} must be a path string.")

    for key in (
        "repository_root",
        "data_root",
        "teacher_map_root",
        "output_root",
        "split_manifest_dir",
    ):
        if not isinstance(_require(paths, key, "paths"), str):
            raise ConfigError(f"paths.{key} must be a path string.")

    source_counts = _require(data, "source_counts", "data")
    if data.get("source_directories") != {"train": "train", "test": "test"}:
        raise ConfigError("Source directories are locked to train/ and test/.")
    if source_counts != {"train": 60000, "test": 10000}:
        raise ConfigError("data.source_counts is locked to train=60000/test=10000.")
    if _positive_int(data, "num_classes", "data") != 10:
        raise ConfigError("DecoyMNIST has exactly ten classes.")
    image_shape = _require(data, "image_shape", "data")
    if image_shape != {"height": 28, "width": 28, "channels": 1}:
        raise ConfigError("The unmodified source image shape is locked to 28x28x1.")
    shortcut = _require(data, "shortcut_encoding", "data")
    if shortcut.get("patch_size") != 5:
        raise ConfigError("The original DecoyMNIST patch is 5x5.")
    _exact_sequence(
        shortcut,
        "corners",
        ["top_left", "top_right", "bottom_left", "bottom_right"],
        "data.shortcut_encoding",
    )
    if shortcut.get("train_intensity_formula") != "255_minus_25_times_label":
        raise ConfigError("The training patch formula is locked to 255-25*y.")
    if shortcut.get("test_intensity_formula") != "25_times_label":
        raise ConfigError("The test patch formula is locked to 25*y.")
    if float(shortcut.get("tolerance", -1.0)) != 1.0:
        raise ConfigError("The encoding-audit tolerance is locked to 1.0.")
    if shortcut.get("audit_every_source_image") is not True:
        raise ConfigError("Every source PNG must pass the encoding audit.")

    partition = _require(data, "partition", "data")
    expected_partition = {
        "source_split": "train",
        "algorithm": "class_stratified_largest_remainder",
        "split_seed": 0,
        "candidate_train_count": 48000,
        "biased_validation_count": 6000,
        "oracle_validation_source_count": 6000,
        "preserve_official_test": True,
    }
    for key, expected in expected_partition.items():
        if partition.get(key) != expected:
            raise ConfigError(f"data.partition.{key} is locked to {expected!r}.")
    if sum(
        int(partition[key])
        for key in (
            "candidate_train_count",
            "biased_validation_count",
            "oracle_validation_source_count",
        )
    ) != int(source_counts["train"]):
        raise ConfigError("The three study partitions must consume all training PNGs.")

    visibility = _require(data, "selector_visibility", "data")
    expected_visibility = {
        "vanilla": ["biased_validation"],
        "fcv": ["biased_validation", "fcv_counterfactuals"],
        "oracle": ["oracle_validation_analysis_only"],
        "posthoc": ["test_analysis_only"],
    }
    if visibility != expected_visibility:
        raise ConfigError("Selector visibility differs from the locked protocol.")
    oracle_view = _require(data, "oracle_view", "data")
    expected_oracle = {
        "construct_in_memory": True,
        "source": "oracle_validation_source_analysis_only",
        "replace_training_patch_with_test_encoding": True,
        "persist_transformed_images": False,
    }
    if oracle_view != expected_oracle:
        raise ConfigError("The Oracle view must be in-memory and analysis-only.")
    preprocessing = _require(data, "preprocessing", "data")
    if preprocessing.get("grayscale_to_rgb") != "identical_channels":
        raise ConfigError("ViT input must repeat grayscale into identical RGB channels.")
    if preprocessing.get("normalization") != "imagenet":
        raise ConfigError("Evaluation normalization is locked to ImageNet statistics.")
    if float(preprocessing.get("horizontal_flip_probability", -1.0)) != 0.0:
        raise ConfigError("Horizontal flips are forbidden for digit labels.")
    if preprocessing.get("resize") != {
        "size": [224, 224],
        "interpolation": "bicubic",
    }:
        raise ConfigError("Evaluation preprocessing is locked to bicubic 224x224.")
    teacher_maps = _require(data, "teacher_maps", "data")
    if teacher_maps != {
        "source": "existing_r4rr_openclip_dinovit",
        "required_for": "biased_validation",
        "require_complete_coverage_before_training": True,
        "format": "voc_colormap_class_ids",
        "foreground_class_ids": [1],
        "normalize_to_unit_interval": True,
        "resize_interpolation": "nearest",
        "patch_projection_grid": 14,
        "preflight_overlay_count": 20,
        "missing_map_policy": "fail_with_exact_regeneration_manifest",
    }:
        raise ConfigError("Teacher-map provenance and coverage policy are locked.")

    expected_model = {
        "library": "timm",
        "name": "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "pretrained": True,
        "num_classes": 10,
        "image_size": 224,
        "patch_size": 16,
        "patch_grid_size": 14,
        "classification_head": "cls_token",
        "fine_tune_mode": "full",
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ConfigError(f"model.{key} is locked to {expected!r}.")
    if int(model["image_size"]) // int(model["patch_size"]) != int(
        model["patch_grid_size"]
    ):
        raise ConfigError("The configured ViT patch geometry is inconsistent.")

    if training.get("optimizer") != "AdamW":
        raise ConfigError("The optimizer is locked to AdamW.")
    _exact_sequence(training, "learning_rates", [1.0e-5, 3.0e-5, 1.0e-4], "training")
    _exact_sequence(training, "weight_decays", [0.01, 0.05, 0.10], "training")
    _exact_sequence(training, "crop_scale_mins", [1.0, 0.8, 0.6, 0.4], "training")
    _exact_sequence(training, "seeds", [0, 1, 2], "training")
    if _positive_int(training, "epochs", "training") != 10:
        raise ConfigError("Training length is locked to ten epochs.")
    if _positive_int(training, "batch_size", "training") != 256:
        raise ConfigError("Production batch size is locked to 256.")
    if training.get("precision") != "amp_bfloat16":
        raise ConfigError("GH200 training uses bfloat16 autocast.")
    if training.get("evaluate_every_epoch") is not True:
        raise ConfigError("Every epoch must be evaluated online.")
    scheduler = _require(training, "scheduler", "training")
    if scheduler != {
        "name": "linear_warmup_cosine",
        "warmup_epochs": 1,
        "minimum_learning_rate": 0.0,
    }:
        raise ConfigError("The scheduler is locked to one-epoch warmup plus cosine.")
    augmentation = _require(training, "augmentation", "training")
    for disabled in (
        "horizontal_flip_probability",
        "mixup",
        "cutmix",
        "label_smoothing",
    ):
        if float(augmentation.get(disabled, -1.0)) != 0.0:
            raise ConfigError(f"training.augmentation.{disabled} must be zero.")
    if augmentation.get("train_random_resized_crop") != {
        "enabled": True,
        "maximum_scale": 1.0,
        "ratio": [1.0, 1.0],
        "interpolation": "bicubic",
    }:
        raise ConfigError("RandomResizedCrop geometry differs from the locked grid.")

    epochs = candidate_epochs(config)
    if epochs != list(range(1, 11)):
        raise ConfigError("Every epoch 1..10 is a precommitted candidate.")
    runs = enumerate_runs(config)
    expected_runs = 3 * 3 * 4 * 3
    expected_candidates = expected_runs * 10
    if len(runs) != expected_runs or len({run.run_id for run in runs}) != expected_runs:
        raise ConfigError("The candidate grid must enumerate 108 unique runs.")
    if int(pool.get("expected_training_runs", -1)) != expected_runs:
        raise ConfigError("candidate_pool.expected_training_runs must be 108.")
    if int(pool.get("expected_candidate_states", -1)) != expected_candidates:
        raise ConfigError("candidate_pool.expected_candidate_states must be 1080.")
    if _positive_int(execution, "array_size", "execution") != expected_runs:
        raise ConfigError("execution.array_size must match the 108-run grid.")
    expected_order = ["learning_rate", "weight_decay", "crop_scale_min", "seed"]
    if pool.get("deterministic_order") != expected_order:
        raise ConfigError("Candidate-grid nesting order is locked.")
    for key in (
        "persist_model_checkpoints",
        "persist_optimizer_states",
        "persist_resume_states",
        "retain_selector_winners",
    ):
        if pool.get(key) is not False:
            raise ConfigError(f"candidate_pool.{key} must remain false.")
    if pool.get("persist_aggregate_metrics_only") is not True:
        raise ConfigError("Only aggregate candidate metrics may persist.")

    if fcv.get("intervention_layer") != "raw_patch_embeddings_before_position":
        raise ConfigError("FCV intervention must use raw patch embeddings.")
    background = float(fcv.get("background_patch_threshold", -1.0))
    evidence = float(fcv.get("evidence_patch_threshold", -1.0))
    if (background, evidence) != (0.10, 0.60):
        raise ConfigError("FCV thresholds are locked to background=.10/evidence=.60.")
    if fcv.get("ambiguous_patch_policy") != "keep_target":
        raise ConfigError("Ambiguous target patches must be preserved.")
    expected_mask_acceptance = {
        "minimum_background_patches": 20,
        "minimum_eligible_fraction": 0.10,
        "minimum_eligible_per_class": 20,
        "require_decoy_region_safe_background": True,
        "decoy_unsafe_target_policy": "exclude_and_audit",
    }
    for key, expected in expected_mask_acceptance.items():
        if fcv.get(key) != expected:
            raise ConfigError(f"fcv.{key} is locked to {expected!r}.")
    if int(fcv.get("donor_samples_per_target", -1)) != 5:
        raise ConfigError("Each FCV target must use five donors.")
    if int(fcv.get("donor_plan_seed", -1)) != 0:
        raise ConfigError("The donor plan seed is locked to zero.")
    donor_rules = _require(fcv, "donor_rules", "fcv")
    if donor_rules != {
        "source": "biased_validation_only",
        "require_non_target_label": True,
        "require_same_corner": True,
        "require_distinct_donor_labels": True,
        "exclude_target_sample": True,
        "replacement_rule": "mutually_safe_spatial_background",
        "apply_target_position_embeddings_after_swap": True,
    }:
        raise ConfigError("FCV donor rules differ from the locked multiclass protocol.")
    primary = _require(fcv, "primary_selector", "fcv")
    if primary != {
        "name": "harmonic_original_counterfactual_accuracy",
        "epsilon": 1.0e-12,
        "formula": "2ab_over_a_plus_b",
    }:
        raise ConfigError("The parameter-free harmonic FCV selector is locked.")
    controls = _require(fcv, "controls", "fcv")
    if controls != {
        "same_context_donor": True,
        "random_mask": True,
        "shuffled_teacher_mask": True,
        "evidence_swap": True,
        "exact_synthetic_mask_analysis_only": True,
        "warning_only": True,
    }:
        raise ConfigError("The online FCV control set differs from the locked protocol.")

    if evaluation.get("online_every_epoch") is not True:
        raise ConfigError("Evaluation must remain online at every epoch.")
    if evaluation.get("primary_test_metric") != "accuracy":
        raise ConfigError("The primary DecoyMNIST test metric is overall accuracy.")
    if evaluation.get("test_metrics_must_not_affect_selection") is not True:
        raise ConfigError("Test metrics cannot affect any deployable selector.")
    if evaluation.get("tie_break") != "candidate_id_ascending":
        raise ConfigError("Exact ties use ascending candidate ID.")
    if evaluation.get("selectors") != {
        "vanilla": "biased_validation_accuracy",
        "fcv": "harmonic_original_counterfactual_accuracy",
        "oracle": "oracle_validation_accuracy",
        "posthoc": "test_accuracy_analysis_only",
    }:
        raise ConfigError("Selector definitions or visibility names are stale.")
    if evaluation.get("secondary_test_metrics") != [
        "balanced_class_accuracy",
        "worst_class_accuracy",
        "per_class_accuracy",
    ]:
        raise ConfigError("Secondary test metrics differ from the locked protocol.")
    if evaluation.get("gap_closure") != {
        "metric": "test_accuracy",
        "denominator_epsilon": 1.0e-12,
        "clip_fraction": False,
    }:
        raise ConfigError("Gap closure is locked to unclipped test accuracy.")

    if float(storage.get("persistent_output_budget_gib", -1.0)) != 1.0:
        raise ConfigError("Persistent campaign output is budgeted below 1 GiB.")
    required_true = (
        "node_local_ephemeral_token_banks",
        "delete_ephemeral_token_banks_after_each_epoch",
    )
    for key in required_true:
        if storage.get(key) is not True:
            raise ConfigError(f"storage.{key} must remain true.")
    required_false = (
        "persist_token_banks",
        "persist_patch_embeddings",
        "persist_per_image_logits",
        "persist_per_donor_predictions",
        "persist_duplicate_datasets",
    )
    for key in required_false:
        if storage.get(key) is not False:
            raise ConfigError(f"storage.{key} must remain false.")
    _exact_sequence(storage, "forbidden_suffixes", [".ckpt", ".pth", ".pt"], "storage")
    _exact_sequence(
        storage,
        "allowed_persistent_artifacts",
        [
            "manifests",
            "provenance",
            "projected_teacher_masks",
            "donor_plans",
            "aggregate_csv",
            "aggregate_json",
            "logs",
            "plots",
        ],
        "storage",
    )

    if _positive_int(execution, "max_concurrent_runs", "execution") != 8:
        raise ConfigError("At most eight GH200 runs may execute concurrently.")
    for key, expected in {
        "cpus_per_task": 12,
        "memory_gib": 64,
        "fcv_counterfactual_batch_size": 256,
        "control_target_batch_size": 16,
    }.items():
        if _positive_int(execution, key, "execution") != expected:
            raise ConfigError(f"execution.{key} is locked to {expected}.")
    if execution.get("time_limit") != "1-00:00:00":
        raise ConfigError("Each array task is limited to one day.")
    if reproducibility.get("deterministic_algorithms") is not True:
        raise ConfigError("Deterministic algorithms are required.")
    if reproducibility.get("cudnn_benchmark") is not False:
        raise ConfigError("cuDNN benchmarking must remain disabled.")
    if reproducibility.get("hash_algorithm") != "sha256":
        raise ConfigError("Source and artifact provenance uses SHA-256.")
    if reproducibility.get("manifest_schema_version") != 1:
        raise ConfigError("Manifest schema version is locked to one.")
    if reproducibility.get("split_algorithm_version") != (
        "decoy_stratified_largest_remainder_v1"
    ):
        raise ConfigError("Split algorithm version differs from the locked protocol.")


def load_and_validate_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError("The top-level YAML value must be a mapping.")
    config = _expand_paths(loaded)
    validate_config(config)
    config["_provenance"] = {
        "config_path": str(config_path),
        "config_file_sha256": sha256_file(config_path),
        "canonical_config_sha256": canonical_config_sha256(config),
    }
    return config
