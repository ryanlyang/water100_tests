"""Configuration loading and protocol validation for the ViT-FCV first study."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a study configuration violates the locked protocol."""


def candidate_epochs(config: Mapping[str, Any]) -> List[int]:
    """Return the ordered, precommitted checkpoint epochs in the candidate pool."""

    training = _require(config, "training", "config")
    pool = _require(config, "candidate_pool", "config")
    epochs = int(_require(training, "epochs", "training"))
    values = [int(value) for value in _require(pool, "candidate_epochs", "candidate_pool")]
    if not values or values != sorted(set(values)):
        raise ConfigError(
            "candidate_pool.candidate_epochs must be a nonempty, strictly increasing set."
        )
    if values[0] < 1 or values[-1] > epochs:
        raise ConfigError(
            "candidate_pool.candidate_epochs must lie inside the training schedule."
        )
    return values


def _expand_paths(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_paths(item) for key, item in value.items()}
    return value


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required key: {context}.{key}")
    return mapping[key]


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ConfigError("The top-level YAML value must be a mapping.")
    return _expand_paths(loaded)


def validate_config(
    config: Mapping[str, Any],
    *,
    strict_protocol: bool = True,
) -> None:
    study = _require(config, "study", "config")
    cluster = _require(config, "cluster", "config")
    data = _require(config, "data", "config")
    model = _require(config, "model", "config")
    training = _require(config, "training", "config")
    pool = _require(config, "candidate_pool", "config")
    execution = _require(config, "execution", "config")
    fcv = _require(config, "fcv", "config")
    evaluation = _require(config, "evaluation", "config")
    storage = _require(config, "storage", "config")

    if study.get("dataset") != "waterbirds100":
        raise ConfigError("The first-study configuration must target waterbirds100.")

    if cluster.get("account") != "reu-aisocial":
        raise ConfigError("Tigris jobs must use Slurm account 'reu-aisocial'.")
    if cluster.get("partition") != "tigris":
        raise ConfigError("The first study must target the Tigris partition.")
    if cluster.get("gres") != "gpu:gh200:1":
        raise ConfigError("The first study expects exactly one GH200 GPU per job.")
    if cluster.get("torch_version") != "2.11.0+cu130":
        raise ConfigError("The first study locks torch=2.11.0+cu130.")
    if cluster.get("torchvision_version") != "0.26.0+cu130":
        raise ConfigError("The first study locks torchvision=0.26.0+cu130.")
    if cluster.get("timm_version") != "1.0.28":
        raise ConfigError("The first study locks timm=1.0.28.")

    holdout = _require(data, "biased_train_holdout", "data")
    train_fraction = float(_require(holdout, "train_fraction", "data.biased_train_holdout"))
    val_fraction = float(
        _require(holdout, "validation_fraction", "data.biased_train_holdout")
    )
    if train_fraction != 0.80 or val_fraction != 0.20:
        raise ConfigError("The locked train-derived holdout is exactly 80/20.")
    if holdout.get("source_split") != "train":
        raise ConfigError("The biased validation holdout must be sliced from train.")
    if int(holdout.get("split_seed", -1)) != 0:
        raise ConfigError("The locked holdout split seed is 0.")
    if holdout.get("stratify_by") != "y":
        raise ConfigError("The holdout must be stratified by target class y only.")
    if not holdout.get("require_complete_shortcut_correlation", False):
        raise ConfigError("The held-out validation data must remain fully correlated.")

    visibility = _require(data, "selector_visibility", "data")
    expected_holdout = "biased_train_holdout_only"
    if visibility.get("vanilla") != expected_holdout:
        raise ConfigError("Vanilla selection must use only the biased train holdout.")
    if visibility.get("fcv") != expected_holdout:
        raise ConfigError("FCV selection must use only the biased train holdout.")
    if visibility.get("oracle") != "original_mixed_validation_only":
        raise ConfigError(
            "Oracle selection must use only the original mixed validation split."
        )
    if visibility.get("test") != "evaluation_only":
        raise ConfigError("The test split must remain evaluation-only.")

    image_size = int(_require(model, "image_size", "model"))
    patch_size = int(_require(model, "patch_size", "model"))
    patch_grid_size = int(_require(model, "patch_grid_size", "model"))
    if image_size % patch_size != 0:
        raise ConfigError("Image size must be divisible by patch size.")
    if image_size // patch_size != patch_grid_size:
        raise ConfigError("patch_grid_size is inconsistent with image_size/patch_size.")
    if model.get("library") != "timm":
        raise ConfigError("The first study locks the candidate model library to timm.")
    if model.get("fine_tune_mode") != "full":
        raise ConfigError("The first candidate pool uses full ViT fine-tuning.")
    if model.get("classification_head") != "cls_token":
        raise ConfigError("The first candidate pool uses the CLS-token head.")
    if strict_protocol:
        expected_model = {
            "name": "vit_small_patch16_224.augreg_in21k_ft_in1k",
            "pretrained": True,
            "image_size": 224,
            "patch_size": 16,
            "patch_grid_size": 14,
            "num_classes": 2,
        }
        for key, expected in expected_model.items():
            if model.get(key) != expected:
                raise ConfigError(
                    f"The production first study locks model.{key}={expected!r}."
                )

    learning_rates = [float(value) for value in _require(training, "learning_rates", "training")]
    weight_decays = [float(value) for value in _require(training, "weight_decays", "training")]
    seeds = [int(value) for value in _require(training, "seeds", "training")]
    epochs = int(_require(training, "epochs", "training"))
    if learning_rates != [1.0e-5, 3.0e-5, 1.0e-4]:
        raise ConfigError("The locked learning-rate grid is [1e-5, 3e-5, 1e-4].")
    if weight_decays != [0.01, 0.05, 0.10]:
        raise ConfigError("The locked weight-decay grid is [0.01, 0.05, 0.10].")
    if seeds != [0, 1, 2]:
        raise ConfigError("The locked candidate seeds are [0, 1, 2].")
    if epochs != 20:
        raise ConfigError("The locked first-study training length is 20 epochs.")
    expected_runs = len(learning_rates) * len(weight_decays) * len(seeds)
    selected_epochs = candidate_epochs(config)
    if strict_protocol and selected_epochs != [5, 10, 20]:
        raise ConfigError(
            "The reduced first study locks candidate epochs to [5, 10, 20]."
        )
    expected_candidates = expected_runs * len(selected_epochs)
    if int(pool.get("expected_training_runs", -1)) != expected_runs:
        raise ConfigError(
            "candidate_pool.expected_training_runs does not match the sweep product."
        )
    if int(pool.get("expected_candidate_checkpoints", -1)) != expected_candidates:
        raise ConfigError(
            "candidate_pool.expected_candidate_checkpoints does not match "
            "runs*len(candidate_epochs)."
        )
    if training.get("optimizer") != "AdamW":
        raise ConfigError("The first candidate pool locks the optimizer to AdamW.")
    if training.get("precision") != "amp_bfloat16":
        raise ConfigError("Tigris candidate training is locked to bfloat16 autocast.")
    scheduler = _require(training, "scheduler", "training")
    if scheduler.get("name") != "cosine":
        raise ConfigError("The first candidate pool uses cosine scheduling.")
    warmup_epochs = int(_require(scheduler, "warmup_epochs", "training.scheduler"))
    if not 0 <= warmup_epochs < epochs:
        raise ConfigError("scheduler.warmup_epochs must be in [0, epochs).")
    augmentation = _require(training, "augmentation", "training")
    if int(augmentation.get("eval_resize_size", 0)) < image_size:
        raise ConfigError("augmentation.eval_resize_size must be at least image_size.")
    for disabled in ("mixup", "cutmix", "label_smoothing"):
        if float(augmentation.get(disabled, -1.0)) != 0.0:
            raise ConfigError(f"The first candidate pool locks {disabled}=0.")
    if strict_protocol:
        locked_training = {
            "batch_size": 128,
            "num_workers": 8,
            "evaluate_every_epoch": True,
        }
        for key, expected in locked_training.items():
            if training.get(key) != expected:
                raise ConfigError(
                    f"The production first study locks training.{key}={expected!r}."
                )
        locked_scheduler = {
            "warmup_epochs": 2,
            "minimum_learning_rate": 0.0,
        }
        for key, expected in locked_scheduler.items():
            if scheduler.get(key) != expected:
                raise ConfigError(
                    "The production first study locks "
                    f"training.scheduler.{key}={expected!r}."
                )
        crop = _require(
            augmentation,
            "train_random_resized_crop",
            "training.augmentation",
        )
        locked_augmentation = {
            "train_horizontal_flip_probability": 0.50,
            "eval_resize_size": 256,
            "normalization": "imagenet",
        }
        for key, expected in locked_augmentation.items():
            if augmentation.get(key) != expected:
                raise ConfigError(
                    "The production first study locks "
                    f"training.augmentation.{key}={expected!r}."
                )
        if crop.get("enabled") is not True or [
            float(value) for value in crop.get("scale", [])
        ] != [0.80, 1.00]:
            raise ConfigError(
                "The production first study locks random-resized crop to enabled "
                "with scale=[0.80, 1.00]."
            )
    reproducibility = _require(config, "reproducibility", "config")
    if reproducibility.get("deterministic_algorithms") is not True:
        raise ConfigError("Candidate training requires deterministic algorithms.")
    if reproducibility.get("cudnn_benchmark") is not False:
        raise ConfigError("cuDNN benchmarking must be disabled for exact resume.")
    if pool.get("unit") != "fixed_epoch_checkpoint":
        raise ConfigError("The reduced study uses fixed candidate checkpoint epochs.")
    if pool.get("checkpoint_retention") != (
        "candidate_epochs_until_step12_complete_then_prune"
    ):
        raise ConfigError(
            "All reduced-pool candidate checkpoints must remain through Step 12."
        )
    if pool.get("checkpoint_dtype") != "float32":
        raise ConfigError("Candidate checkpoint weights must remain float32.")
    if pool.get("save_optimizer_state") is not False:
        raise ConfigError("Candidate checkpoints must not contain optimizer state.")
    hard_budget = float(_require(storage, "hard_budget_gib", "storage"))
    launch_guard = float(_require(storage, "launch_guard_gib", "storage"))
    if hard_budget != 40.0 or launch_guard != 35.0:
        raise ConfigError("The reduced study locks storage limits to 35/40 GiB.")
    if not 0.0 < launch_guard < hard_budget:
        raise ConfigError("storage.launch_guard_gib must be below hard_budget_gib.")
    locked_storage = {
        "streaming_token_banks": True,
        "delete_token_banks_after_fcv_and_controls": True,
        "delete_completed_resume_states_after_pool_validation": True,
        "max_concurrent_streaming_runs": 4,
    }
    for key, expected in locked_storage.items():
        if storage.get(key) != expected:
            raise ConfigError(f"The reduced study locks storage.{key}={expected!r}.")
    locked_execution = {
        "token_bank_batch_size": 128,
        "token_bank_num_workers": 8,
        "fcv_counterfactual_forward_batch_size": 256,
        "control_target_batch_size": 16,
        "control_counterfactual_forward_batch_size": 256,
    }
    for key, expected in locked_execution.items():
        value = int(_require(execution, key, "execution"))
        if key.endswith("num_workers"):
            if value < 0:
                raise ConfigError(f"execution.{key} must be nonnegative.")
        elif value <= 0:
            raise ConfigError(f"execution.{key} must be positive.")
        if strict_protocol and value != expected:
            raise ConfigError(
                f"The production first study locks execution.{key}={expected}."
            )

    evidence_threshold = float(
        _require(fcv, "evidence_patch_threshold", "fcv")
    )
    background_threshold = float(
        _require(fcv, "background_patch_threshold", "fcv")
    )
    if not 0.0 <= background_threshold < evidence_threshold <= 1.0:
        raise ConfigError(
            "Patch thresholds must satisfy 0 <= background < evidence <= 1."
        )
    if fcv.get("intervention_layer") != "raw_patch_embeddings_before_position":
        raise ConfigError("The first intervention must use raw patch embeddings.")
    if int(fcv.get("donor_samples_per_image", 0)) <= 0:
        raise ConfigError("donor_samples_per_image must be positive.")
    if reproducibility.get("cache_donor_sample_indices") is not True:
        raise ConfigError(
            "Step 7 requires one cached donor-index plan shared by all candidates."
        )
    donor_sampling_seed = _require(
        reproducibility,
        "donor_sampling_seed",
        "reproducibility",
    )
    if not isinstance(donor_sampling_seed, int) or isinstance(
        donor_sampling_seed, bool
    ):
        raise ConfigError("reproducibility.donor_sampling_seed must be an integer.")
    control_sampling_seed = _require(
        reproducibility,
        "control_sampling_seed",
        "reproducibility",
    )
    if not isinstance(control_sampling_seed, int) or isinstance(
        control_sampling_seed, bool
    ):
        raise ConfigError("reproducibility.control_sampling_seed must be an integer.")
    patch_count = patch_grid_size * patch_grid_size
    minimum_background = int(
        _require(fcv, "minimum_background_patches", "fcv")
    )
    if not 1 <= minimum_background <= patch_count:
        raise ConfigError(
            "minimum_background_patches must be between 1 and the patch count."
        )
    if strict_protocol:
        locked_fcv = {
            "evidence_patch_threshold": 0.60,
            "background_patch_threshold": 0.10,
            "ambiguous_patch_policy": "keep_target",
            "minimum_background_patches": 20,
            "minimum_eligible_fraction": 0.10,
            "minimum_eligible_count_per_class": 20,
            "preflight_overlay_count": 20,
            "donor_samples_per_image": 5,
        }
        for key, expected in locked_fcv.items():
            if fcv.get(key) != expected:
                raise ConfigError(
                    f"The production first study locks fcv.{key}={expected!r}."
                )
        expected_donor_contract = {
            "scope": "model_specific",
            "source": "biased_train_holdout_only",
            "context_key": "class_label_proxy",
            "sampling": "global_uniform",
            "sample_with_replacement": True,
            "exclude_target_image": True,
            "donor_context": "opposite",
            "token_positioning": "apply_target_position_embedding_after_swap",
        }
        if dict(_require(fcv, "donor_bank", "fcv")) != expected_donor_contract:
            raise ConfigError(
                "The production first study locks the exact FCV donor-bank contract."
            )
        if donor_sampling_seed != 0 or control_sampling_seed != 1:
            raise ConfigError(
                "The production first study locks donor/control seeds to 0/1."
            )

    teacher_maps = _require(data, "teacher_maps", "data")
    if not teacher_maps.get("normalize_to_unit_interval", False):
        raise ConfigError("Step 3 requires teacher maps normalized to [0, 1].")
    if teacher_maps.get("format") != "voc_colormap_class_ids":
        raise ConfigError("Waterbirds teacher maps must be decoded as VOC class IDs.")
    if [int(value) for value in teacher_maps.get("foreground_class_ids", [])] != [1]:
        raise ConfigError("Waterbirds foreground class IDs are locked to [1].")
    if teacher_maps.get("interpolation") != "nearest":
        raise ConfigError("Categorical teacher maps require nearest interpolation.")
    if teacher_maps.get("spatial_transform") != (
        "eval_resize_shorter_side_then_center_crop"
    ):
        raise ConfigError("Teacher masks must use the exact evaluation resize/crop geometry.")
    if teacher_maps.get("interpolation") not in {"nearest", "bilinear", "bicubic"}:
        raise ConfigError("Unsupported teacher-map interpolation mode.")
    minimum_fraction = float(_require(fcv, "minimum_eligible_fraction", "fcv"))
    minimum_per_class = int(_require(fcv, "minimum_eligible_count_per_class", "fcv"))
    overlay_count = int(_require(fcv, "preflight_overlay_count", "fcv"))
    if not 0.0 < minimum_fraction <= 1.0:
        raise ConfigError("minimum_eligible_fraction must be in (0, 1].")
    if minimum_per_class < 2:
        raise ConfigError("minimum_eligible_count_per_class must be at least 2.")
    if overlay_count <= 0:
        raise ConfigError("preflight_overlay_count must be positive.")

    outputs = _require(config, "outputs", "config")
    if not outputs.get("patch_masks"):
        raise ConfigError("outputs.patch_masks must name the Step 3 artifact directory.")

    primary = _require(fcv, "primary_selector", "fcv")
    original_weight = float(
        _require(primary, "original_accuracy_weight", "fcv.primary_selector")
    )
    counterfactual_weight = float(
        _require(
            primary,
            "counterfactual_accuracy_weight",
            "fcv.primary_selector",
        )
    )
    if abs(original_weight + counterfactual_weight - 1.0) > 1e-9:
        raise ConfigError("Primary selector weights must sum to 1.0.")
    if original_weight != 0.5 or counterfactual_weight != 0.5:
        raise ConfigError("The locked primary selector uses equal 0.5/0.5 weights.")

    selector_analysis = _require(fcv, "selector_analysis", "fcv")
    lambda_values = [
        float(value)
        for value in _require(
            selector_analysis,
            "fcv_accuracy_lambdas",
            "fcv.selector_analysis",
        )
    ]
    if lambda_values != [0.25, 0.5, 1.0]:
        raise ConfigError(
            "The first Step 9 analysis locks FCV accuracy lambdas to [0.25, 0.5, 1.0]."
        )
    if float(selector_analysis.get("control_normalized_lambda", -1.0)) != 1.0:
        raise ConfigError("The control-normalized FCV lambda must be fixed to 1.0.")
    if float(selector_analysis.get("probability_ratio_epsilon", 0.0)) <= 0.0:
        raise ConfigError("The FCV probability-ratio epsilon must be positive.")
    if strict_protocol and float(
        selector_analysis.get("probability_ratio_epsilon")
    ) != 1.0e-8:
        raise ConfigError("The production probability-ratio epsilon is locked to 1e-8.")
    if selector_analysis.get("oracle_precision") != "float32":
        raise ConfigError("Step 9 Oracle validation is locked to float32 evaluation.")
    if int(selector_analysis.get("oracle_batch_size", 0)) <= 0:
        raise ConfigError("The Oracle validation batch size must be positive.")
    if strict_protocol and int(selector_analysis.get("oracle_batch_size")) != 128:
        raise ConfigError("The production Oracle validation batch size is locked to 128.")
    if selector_analysis.get("deterministic_tie_break") != "candidate_id_ascending":
        raise ConfigError("Step 9 ties must be broken by ascending candidate ID only.")
    if selector_analysis.get("exact_ties_only") is not True:
        raise ConfigError("Step 9 must use exact numeric ties, without isclose tolerance.")

    expected_secondary_selectors = {
        "biased_validation_accuracy",
        "biased_validation_loss",
        "opposite_context_counterfactual_accuracy",
        "opposite_context_true_class_probability",
        "control_normalized_fcv",
        "oracle_validation_worst_group_accuracy",
        "oracle_validation_balanced_group_accuracy",
    }
    if set(fcv.get("secondary_selectors", [])) != expected_secondary_selectors:
        raise ConfigError("Step 9 secondary selector definitions are incomplete or stale.")

    controls = _require(fcv, "controls", "fcv")
    for control_name in (
        "same_context_background_swap",
        "random_patch_mask_swap",
        "shuffled_teacher_mask_swap",
        "evidence_token_swap",
    ):
        if controls.get(control_name) is not True:
            raise ConfigError(f"The first study requires fcv.controls.{control_name}=true.")
    warning_policy = _require(
        controls, "diagnostic_warning_policy", "fcv.controls"
    )
    if warning_policy.get("warning_only") is not True:
        raise ConfigError("Control diagnostics must warn without changing selection.")
    for name in (
        "same_minus_opposite_accuracy_min",
        "opposite_minus_same_confidence_drop_min",
        "evidence_vs_background_sensitivity_gap_min",
    ):
        value = float(_require(warning_policy, name, "control warning policy"))
        if not -1.0 <= value <= 1.0:
            raise ConfigError(f"Control warning threshold {name} is invalid.")
        if strict_protocol and value != -0.02:
            raise ConfigError(
                f"The production first study locks control warning {name}=-0.02."
            )

    if not evaluation.get("test_metrics_must_not_affect_selection", False):
        raise ConfigError("Test metrics must not affect model selection.")
    final_test = _require(evaluation, "final_test", "evaluation")
    if final_test.get("precision") != "float32":
        raise ConfigError("Step 10 final-test inference is locked to float32.")
    if int(final_test.get("batch_size", 0)) <= 0:
        raise ConfigError("Step 10 final-test batch size must be positive.")
    if strict_protocol and int(final_test.get("batch_size")) != 128:
        raise ConfigError("The production final-test batch size is locked to 128.")
    for key in (
        "evaluate_unique_checkpoints_once",
        "require_complete_step9_selection",
        "preserve_selector_order",
    ):
        if final_test.get(key) is not True:
            raise ConfigError(f"Step 10 requires evaluation.final_test.{key}=true.")
    gap_closure = _require(evaluation, "gap_closure", "evaluation")
    expected_gap_closure = {
        "metric": "test_worst_group_accuracy",
        "biased_selector": "biased_validation_accuracy",
        "fcv_selector": "equal_weight_original_and_opposite_fcv_accuracy",
        "oracle_selector": "oracle_validation_balanced_group_accuracy",
    }
    for key, expected in expected_gap_closure.items():
        if gap_closure.get(key) != expected:
            raise ConfigError(
                f"Step 11 locks evaluation.gap_closure.{key}={expected!r}."
            )
    if float(gap_closure.get("denominator_epsilon", 0.0)) <= 0.0:
        raise ConfigError("Step 11 gap denominator epsilon must be positive.")
    if gap_closure.get("clip_fraction") is not False:
        raise ConfigError("Step 11 must report the raw, unclipped gap-closure fraction.")
    for key in (
        "evaluate_full_pool_posthoc",
        "full_pool_scores_must_not_affect_selection",
    ):
        if gap_closure.get(key) is not True:
            raise ConfigError(f"Step 11 requires evaluation.gap_closure.{key}=true.")
    rank_analysis = _require(evaluation, "rank_analysis", "evaluation")
    expected_rank_settings = {
        "target_metric": "test_worst_group_accuracy",
        "fcv_stability_definition": "mean_counterfactual_true_class_probability",
        "top_k_values": [1, 5, 10, 25],
        "deterministic_tie_break": "candidate_id_ascending",
        "spearman_rank_method": "average",
        "kendall_variant": "b",
        "orient_minimize_scores": True,
        "create_scatter_plots": True,
        "posthoc_only": True,
    }
    for key, expected in expected_rank_settings.items():
        if rank_analysis.get(key) != expected:
            raise ConfigError(
                f"Step 12 locks evaluation.rank_analysis.{key}={expected!r}."
            )
    bootstrap = _require(rank_analysis, "clustered_bootstrap", "rank_analysis")
    if bootstrap.get("enabled") is not True:
        raise ConfigError("Step 12 requires clustered bootstrap inference.")
    if bootstrap.get("cluster_column") != "run_index":
        raise ConfigError("Step 12 bootstrap clusters must be candidate runs.")
    if int(bootstrap.get("replicates", 0)) < 1000:
        raise ConfigError("Step 12 requires at least 1,000 bootstrap replicates.")
    if int(bootstrap.get("seed", -1)) < 0:
        raise ConfigError("Step 12 bootstrap seed must be nonnegative.")
    if strict_protocol:
        locked_bootstrap = {
            "replicates": 2000,
            "seed": 0,
            "confidence_level": 0.95,
        }
        for key, expected in locked_bootstrap.items():
            if bootstrap.get(key) != expected:
                raise ConfigError(
                    "The production first study locks "
                    f"evaluation.rank_analysis.clustered_bootstrap.{key}={expected!r}."
                )
    confidence_level = float(bootstrap.get("confidence_level", 0.0))
    if not 0.0 < confidence_level < 1.0:
        raise ConfigError("Step 12 bootstrap confidence level must be in (0, 1).")
    expected_rank_selectors = [
        {
            "name": "biased_validation_accuracy",
            "display_name": "Biased validation accuracy",
            "score_column": "biased_val_accuracy",
            "direction": "maximize",
        },
        {
            "name": "biased_validation_loss",
            "display_name": "Negative biased validation loss",
            "score_column": "biased_val_loss",
            "direction": "minimize",
        },
        {
            "name": "equal_weight_original_and_opposite_fcv_accuracy",
            "display_name": "FCV main score",
            "score_column": "primary_selector_score",
            "direction": "maximize",
        },
        {
            "name": "opposite_context_counterfactual_accuracy",
            "display_name": "FCV counterfactual accuracy",
            "score_column": "fcv_counterfactual_accuracy",
            "direction": "maximize",
        },
        {
            "name": "opposite_context_true_class_probability",
            "display_name": "FCV stability",
            "score_column": "fcv_true_class_probability",
            "direction": "maximize",
        },
        {
            "name": "oracle_validation_balanced_group_accuracy",
            "display_name": "Oracle group validation",
            "score_column": "oracle_validation_balanced_group_accuracy",
            "direction": "maximize",
        },
    ]
    if rank_analysis.get("selectors") != expected_rank_selectors:
        raise ConfigError("Step 12 rank-analysis selector definitions are stale.")


def load_and_validate_config(
    path: str | Path,
    *,
    strict_protocol: bool = True,
) -> Dict[str, Any]:
    config = load_config(path)
    validate_config(config, strict_protocol=strict_protocol)
    return config


def config_summary(config: Mapping[str, Any]) -> Dict[str, Any]:
    training = config["training"]
    holdout = config["data"]["biased_train_holdout"]
    run_count = (
        len(training["learning_rates"])
        * len(training["weight_decays"])
        * len(training["seeds"])
    )
    return {
        "study_id": config["study"]["id"],
        "cluster": config["cluster"]["name"],
        "slurm_account": config["cluster"]["account"],
        "model": config["model"]["name"],
        "train_fraction": holdout["train_fraction"],
        "biased_validation_fraction": holdout["validation_fraction"],
        "holdout_split_seed": holdout["split_seed"],
        "training_runs": run_count,
        "candidate_checkpoints": run_count * len(candidate_epochs(config)),
        "candidate_epochs": candidate_epochs(config),
        "fcv_donor_samples": config["fcv"]["donor_samples_per_image"],
        "primary_selector": config["fcv"]["primary_selector"]["name"],
    }
