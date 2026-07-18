"""Visibility-separated aggregate schemas for DecoyMNIST online evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd


class OnlineSchemaError(ValueError):
    """Raised when aggregate rows cross a selector visibility boundary."""


NAMESPACES = (
    "biased_validation",
    "fcv",
    "controls",
    "oracle_analysis_only",
    "test_analysis_only",
)

COMMON_COLUMNS = [
    "run_index",
    "run_id",
    "candidate_id",
    "epoch",
    "seed",
    "learning_rate",
    "weight_decay",
    "crop_scale_min",
]

CLASS_ACCURACY_SUFFIXES = [f"class_{label}_accuracy" for label in range(10)]

BIASED_VALIDATION_COLUMNS = COMMON_COLUMNS + [
    "train_loss",
    "train_accuracy",
    "lr_epoch_start",
    "lr_epoch_end",
    "biased_validation_loss",
    "biased_validation_accuracy",
    "biased_validation_balanced_class_accuracy",
    "biased_validation_worst_class_accuracy",
] + [f"biased_validation_{name}" for name in CLASS_ACCURACY_SUFFIXES] + [
    "epoch_train_seconds",
]

FCV_COLUMNS = COMMON_COLUMNS + [
    "original_biased_validation_accuracy",
    "harmonic_fcv_score",
    "fcv_eligible_target_count",
    "fcv_eligible_target_fraction",
    "fcv_donor_draw_count",
    "fcv_counterfactual_accuracy",
    "fcv_counterfactual_majority_accuracy",
    "fcv_mean_true_class_probability",
    "fcv_mean_confidence_drop",
    "fcv_mean_replaced_patch_count",
    "fcv_changed_replacement_fraction",
    "identity_forward_max_abs_error",
    "epoch_fcv_seconds",
]

CONTROL_METRIC_NAMES = [
    "status",
    "eligible_target_count",
    "eligible_target_fraction",
    "donor_draw_count",
    "counterfactual_accuracy",
    "counterfactual_majority_accuracy",
    "mean_true_class_counterfactual_probability",
    "mean_original_to_counterfactual_confidence_drop",
    "mean_replaced_patch_count",
]

CONTROL_NAMES = (
    "same_context",
    "random_mask",
    "shuffled_teacher_mask",
    "evidence_swap",
    "exact_synthetic_mask_analysis_only",
)

CONTROLS_COLUMNS = COMMON_COLUMNS + [
    "control_diagnostics_warning_only",
    "control_warning_count",
    "control_warning_reason_counts_json",
] + [
    f"control_{control}_{metric}"
    for control in CONTROL_NAMES
    for metric in CONTROL_METRIC_NAMES
]

ORACLE_COLUMNS = COMMON_COLUMNS + [
    "oracle_validation_loss",
    "oracle_validation_accuracy",
    "oracle_validation_balanced_class_accuracy",
    "oracle_validation_worst_class_accuracy",
] + [f"oracle_validation_{name}" for name in CLASS_ACCURACY_SUFFIXES] + [
    "epoch_oracle_seconds",
]

TEST_COLUMNS = COMMON_COLUMNS + [
    "test_loss",
    "test_accuracy",
    "test_balanced_class_accuracy",
    "test_worst_class_accuracy",
] + [f"test_{name}" for name in CLASS_ACCURACY_SUFFIXES] + [
    "epoch_test_seconds",
]

NAMESPACE_COLUMNS: Dict[str, Sequence[str]] = {
    "biased_validation": BIASED_VALIDATION_COLUMNS,
    "fcv": FCV_COLUMNS,
    "controls": CONTROLS_COLUMNS,
    "oracle_analysis_only": ORACLE_COLUMNS,
    "test_analysis_only": TEST_COLUMNS,
}

SELECTOR_NAMESPACE_ACCESS = {
    "vanilla": ("biased_validation",),
    "fcv": ("biased_validation", "fcv"),
    "oracle": ("oracle_analysis_only",),
    "posthoc": ("test_analysis_only",),
}


def namespace_columns(namespace: str) -> list[str]:
    if namespace not in NAMESPACE_COLUMNS:
        raise OnlineSchemaError(f"Unknown online namespace: {namespace!r}")
    return list(NAMESPACE_COLUMNS[namespace])


def validate_namespace_row(namespace: str, row: Mapping[str, Any]) -> None:
    expected = namespace_columns(namespace)
    observed = list(row.keys())
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        raise OnlineSchemaError(
            f"{namespace} row schema mismatch; missing={missing}, unexpected={unexpected}."
        )
    if int(row["epoch"]) < 1 or not str(row["candidate_id"]):
        raise OnlineSchemaError("Online rows require a positive epoch and candidate ID.")
    if namespace in {"biased_validation", "fcv", "controls"}:
        forbidden = [key for key in row if key.startswith(("test_", "oracle_"))]
        if forbidden:
            raise OnlineSchemaError(
                f"Unprivileged namespace contains analysis-only values: {forbidden}."
            )
    if namespace != "test_analysis_only" and any(
        key.startswith("test_") for key in row
    ):
        raise OnlineSchemaError("Test values escaped their analysis-only namespace.")


def authorized_namespaces(selector: str) -> tuple[str, ...]:
    if selector not in SELECTOR_NAMESPACE_ACCESS:
        raise OnlineSchemaError(f"Unknown selector role: {selector!r}")
    return tuple(SELECTOR_NAMESPACE_ACCESS[selector])


def require_selector_access(selector: str, namespace: str) -> None:
    if namespace not in authorized_namespaces(selector):
        raise OnlineSchemaError(
            f"Selector {selector!r} is not authorized to read {namespace!r}."
        )


def namespace_output_path(
    output_root: str | Path, namespace: str, run_id: str
) -> Path:
    namespace_columns(namespace)
    return (
        Path(output_root).expanduser().resolve()
        / "online_metrics"
        / namespace
        / f"{run_id}.csv"
    )


def atomic_write_namespace_rows(
    namespace: str, rows: Sequence[Mapping[str, Any]], path: str | Path
) -> Path:
    columns = namespace_columns(namespace)
    normalized = []
    for row in rows:
        validate_namespace_row(namespace, row)
        normalized.append({column: row[column] for column in columns})
    if not normalized:
        raise OnlineSchemaError(f"Cannot write an empty {namespace} index.")
    candidate_ids = [str(row["candidate_id"]) for row in normalized]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise OnlineSchemaError(f"{namespace} contains duplicate candidate IDs.")
    epochs = [int(row["epoch"]) for row in normalized]
    if epochs != list(range(1, len(epochs) + 1)):
        raise OnlineSchemaError(f"{namespace} rows must be a contiguous epoch prefix.")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    pd.DataFrame(normalized, columns=columns).to_csv(temporary, index=False)
    temporary.replace(destination)
    return destination


def load_selector_namespace(
    output_root: str | Path, selector: str, namespace: str, run_id: str
) -> pd.DataFrame:
    """The only supported selector-facing namespace loader."""

    require_selector_access(selector, namespace)
    path = namespace_output_path(output_root, namespace, run_id)
    if not path.is_file():
        raise OnlineSchemaError(f"Missing {namespace} aggregate: {path}")
    frame = pd.read_csv(path)
    if frame.columns.tolist() != namespace_columns(namespace):
        raise OnlineSchemaError(f"Stored {namespace} header is stale.")
    for row in frame.to_dict("records"):
        validate_namespace_row(namespace, row)
    return frame
