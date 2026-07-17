"""Schemas shared by online training and leakage-separated analysis.

This module deliberately imports no dataset or evaluation implementation.  In
particular, the validation-selection process can import these schemas without
importing the test-evaluation module.
"""

from __future__ import annotations


ONLINE_VALIDATION_COLUMNS = [
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
    "biased_val_loss_batch_reduced_diagnostic",
    "biased_val_accuracy",
    "lr_epoch_start",
    "lr_epoch_end",
    "checkpoint_sha256",
    "epoch_train_seconds",
    "epoch_online_score_seconds",
    "fcv_counterfactual_accuracy",
    "fcv_counterfactual_majority_accuracy",
    "fcv_true_class_probability",
    "fcv_probability_retention_ratio",
    "fcv_confidence_drop",
    "primary_selector_score",
    "same_context_counterfactual_accuracy",
    "same_context_mean_confidence_drop",
    "random_mask_counterfactual_accuracy",
    "shuffled_mask_counterfactual_accuracy",
    "evidence_swap_counterfactual_accuracy",
    "control_diagnostic_warning_count",
    "control_diagnostic_status",
    "target_donor_cosine_similarity_mean",
    "target_nearest_donor_cosine_mean",
    "donor_unique_source_images_mean",
    "donor_max_source_fraction_mean",
    "real_swap_replaced_token_changed_fraction",
    "real_swap_replacement_delta_mean",
    "real_swap_replacement_delta_max",
    "real_swap_foreground_token_max_abs_error",
    "real_swap_donor_reconstruction_max_abs_error",
    "shortcut_sensitivity",
    "control_normalized_fcv_score",
    "oracle_validation_loss",
    "oracle_validation_accuracy",
    "oracle_validation_balanced_group_accuracy",
    "oracle_validation_worst_group_accuracy",
    "oracle_group_0_accuracy",
    "oracle_group_1_accuracy",
    "oracle_group_2_accuracy",
    "oracle_group_3_accuracy",
    "fcv_summary_path",
    "fcv_summary_sha256",
    "controls_summary_path",
    "controls_summary_sha256",
    "oracle_summary_path",
    "oracle_summary_sha256",
]


ONLINE_TEST_COLUMNS = [
    "run_index",
    "candidate_id",
    "epoch",
    "seed",
    "learning_rate",
    "weight_decay",
    "checkpoint_sha256",
    "epoch_online_total_seconds",
    "test_loss",
    "test_accuracy",
    "test_balanced_group_accuracy",
    "test_worst_group_accuracy",
    "test_group_0_accuracy",
    "test_group_0_count",
    "test_group_1_accuracy",
    "test_group_1_count",
    "test_group_2_accuracy",
    "test_group_2_count",
    "test_group_3_accuracy",
    "test_group_3_count",
    "per_image_path",
    "per_image_sha256",
    "summary_path",
    "summary_sha256",
]


RETAINED_SELECTOR_SPECS = {
    "biased_validation_accuracy": ("biased_val_accuracy", "maximize"),
    "equal_weight_original_and_opposite_fcv_accuracy": (
        "primary_selector_score",
        "maximize",
    ),
    "oracle_validation_balanced_group_accuracy": (
        "oracle_validation_balanced_group_accuracy",
        "maximize",
    ),
}
