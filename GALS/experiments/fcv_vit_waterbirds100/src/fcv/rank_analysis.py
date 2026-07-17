"""Step 12 post-hoc selector correlation and rank-quality analysis."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

from .gap_analysis import (
    gap_analysis_fingerprint,
    load_complete_pool_index,
)
from .selectors import selector_analysis_fingerprint
from .test_evaluation import FinalTestSource, FrozenSelection


IDENTITY_COLUMNS = [
    "run_index",
    "candidate_id",
    "epoch",
    "seed",
    "learning_rate",
    "weight_decay",
    "checkpoint_path",
    "checkpoint_sha256",
]


class RankAnalysisError(ValueError):
    """Raised when Step 12 inputs or rank calculations are invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def rank_analysis_fingerprint(config: Mapping[str, Any]) -> str:
    payload = {
        "study": config["study"],
        "model": config["model"],
        "candidate_pool": config["candidate_pool"],
        "selector_analysis_fingerprint": selector_analysis_fingerprint(config),
        "gap_analysis_fingerprint": gap_analysis_fingerprint(config),
        "rank_analysis": config["evaluation"]["rank_analysis"],
    }
    return _sha256_json(payload)


def _require_columns(frame: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise RankAnalysisError(f"{name} is missing required columns: {missing}")


def _oriented_scores(values: np.ndarray, direction: str) -> tuple[np.ndarray, float]:
    if direction == "maximize":
        return values, 1.0
    if direction == "minimize":
        return -values, -1.0
    raise RankAnalysisError(f"Unsupported selector direction: {direction}")


def _deterministic_order(
    candidate_ids: Sequence[str], oriented_scores: np.ndarray
) -> List[str]:
    order = pd.DataFrame(
        {
            "candidate_id": [str(value) for value in candidate_ids],
            "oriented_score": oriented_scores,
        }
    ).sort_values(
        ["oriented_score", "candidate_id"],
        ascending=[False, True],
        kind="stable",
    )
    return order["candidate_id"].astype(str).tolist()


def _correlations(
    oriented_scores: np.ndarray,
    target: np.ndarray,
    *,
    kendall_variant: str,
) -> Dict[str, Any]:
    if np.unique(target).size < 2:
        return {
            "correlation_status": "undefined_constant_test_metric",
            "spearman_rho": None,
            "kendall_tau_b": None,
        }
    if np.unique(oriented_scores).size < 2:
        return {
            "correlation_status": "undefined_constant_selector_score",
            "spearman_rho": None,
            "kendall_tau_b": None,
        }
    spearman = spearmanr(oriented_scores, target, nan_policy="raise")
    kendall = kendalltau(
        oriented_scores,
        target,
        variant=kendall_variant,
        nan_policy="raise",
    )
    values = [float(spearman.statistic), float(kendall.statistic)]
    if not np.isfinite(values).all():
        raise RankAnalysisError("Rank-correlation calculation returned non-finite values.")
    return {
        "correlation_status": "defined",
        "spearman_rho": values[0],
        "kendall_tau_b": values[1],
        "inference_note": "descriptive_fixed_pool; uncertainty uses run-cluster bootstrap",
    }


def _cluster_bootstrap_correlations(
    oriented_scores: np.ndarray,
    target: np.ndarray,
    clusters: np.ndarray,
    *,
    kendall_variant: str,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> Dict[str, Any]:
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < 2 or replicates <= 0:
        raise RankAnalysisError("Cluster bootstrap requires at least two runs and replicates.")
    rng = np.random.default_rng(seed)
    spearman_values: List[float] = []
    kendall_values: List[float] = []
    cluster_rows = {value: np.flatnonzero(clusters == value) for value in unique_clusters}
    for _ in range(replicates):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        indices = np.concatenate([cluster_rows[value] for value in sampled])
        x = oriented_scores[indices]
        y = target[indices]
        if np.unique(x).size < 2 or np.unique(y).size < 2:
            continue
        spearman_values.append(float(spearmanr(x, y).statistic))
        kendall_values.append(
            float(kendalltau(x, y, variant=kendall_variant).statistic)
        )
    if not spearman_values or not kendall_values:
        raise RankAnalysisError("All clustered bootstrap correlation replicates failed.")
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "cluster_bootstrap_replicates_requested": replicates,
        "cluster_bootstrap_replicates_valid": min(
            len(spearman_values), len(kendall_values)
        ),
        "cluster_bootstrap_seed": seed,
        "cluster_bootstrap_confidence_level": confidence_level,
        "spearman_cluster_ci_low": float(np.quantile(spearman_values, alpha)),
        "spearman_cluster_ci_high": float(np.quantile(spearman_values, 1.0 - alpha)),
        "kendall_cluster_ci_low": float(np.quantile(kendall_values, alpha)),
        "kendall_cluster_ci_high": float(np.quantile(kendall_values, 1.0 - alpha)),
    }


def _plot_selector_scatters(
    analysis: pd.DataFrame,
    selector_specs: Sequence[Mapping[str, str]],
    selected_ids: Mapping[str, str],
    pool_best_id: str,
    plot_dir: Path,
) -> List[Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_dir / ".matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths: List[Path] = []

    def draw(ax: Any, spec: Mapping[str, str]) -> None:
        name = str(spec["name"])
        x = analysis[f"oriented_score__{name}"].to_numpy(float)
        y = analysis["test_worst_group_accuracy"].to_numpy(float)
        ax.scatter(x, y, s=13, alpha=0.55, color="#3b6ea8", linewidths=0)
        if np.unique(x).size > 1:
            slope, intercept = np.polyfit(x, y, deg=1)
            x_line = np.linspace(float(x.min()), float(x.max()), 100)
            ax.plot(x_line, slope * x_line + intercept, color="#303030", lw=1.2)
        selected = analysis[analysis["candidate_id"] == selected_ids[name]].iloc[0]
        pool_best = analysis[analysis["candidate_id"] == pool_best_id].iloc[0]
        ax.scatter(
            [float(selected[f"oriented_score__{name}"])],
            [float(selected["test_worst_group_accuracy"])],
            s=52,
            marker="D",
            color="#d95f02",
            edgecolor="white",
            linewidth=0.7,
            label="Selected",
            zorder=4,
        )
        ax.scatter(
            [float(pool_best[f"oriented_score__{name}"])],
            [float(pool_best["test_worst_group_accuracy"])],
            s=62,
            marker="*",
            color="#238b45",
            edgecolor="white",
            linewidth=0.7,
            label="Pool best",
            zorder=5,
        )
        ax.set_title(str(spec["display_name"]), fontsize=10)
        ax.set_xlabel("Oriented selector score", fontsize=9)
        ax.set_ylabel("Test worst-group accuracy", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.22, linestyle=":")

    for spec in selector_specs:
        figure, axis = plt.subplots(figsize=(5.0, 3.8), constrained_layout=True)
        draw(axis, spec)
        axis.legend(loc="best", fontsize=8, frameon=False)
        output = plot_dir / f"{spec['name']}_scatter.png"
        figure.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(figure)
        plot_paths.append(output)

    ncols = min(3, max(1, len(selector_specs)))
    nrows = int(np.ceil(len(selector_specs) / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.35 * ncols, 3.55 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, spec in zip(axes.flat, selector_specs):
        draw(axis, spec)
    for axis in list(axes.flat)[len(selector_specs):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    grid_path = plot_dir / "selector_rank_scatter_grid.png"
    figure.savefig(grid_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    plot_paths.append(grid_path)
    figure, axis = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    colored = axis.scatter(
        analysis["biased_val_accuracy"],
        analysis["test_worst_group_accuracy"],
        c=analysis["fcv_main_score"],
        cmap="viridis",
        s=18,
        alpha=0.72,
        linewidths=0,
    )
    axis.set_xlabel("Biased validation accuracy")
    axis.set_ylabel("Test worst-group accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.22, linestyle=":")
    colorbar = figure.colorbar(colored, ax=axis)
    colorbar.set_label("FCV main score")
    proof_path = plot_dir / "biased_val_vs_test_wga_colored_by_fcv.png"
    figure.savefig(proof_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    plot_paths.append(proof_path)
    return plot_paths


def analyze_rank_quality(
    config: Mapping[str, Any],
    frozen: FrozenSelection,
    source: FinalTestSource,
    pool_csv: str | Path,
    pool_summary: str | Path,
    output_results_csv: str | Path,
    output_candidates_csv: str | Path,
    output_summary: str | Path,
    plot_dir: str | Path,
    *,
    create_plots: bool | None = None,
) -> Dict[str, Any]:
    """Join pre-test selectors to post-hoc outcomes and analyze rank quality."""

    rank_cfg = config["evaluation"]["rank_analysis"]
    selector_specs = list(rank_cfg["selectors"])
    matrix_path = frozen.selector_matrix_path
    if not matrix_path.is_file() or _sha256_file(matrix_path) != frozen.selector_matrix_sha256:
        raise RankAnalysisError("Frozen Step 9 selector matrix is missing or stale.")
    selector_matrix = pd.read_csv(matrix_path)
    test_columns = [
        column for column in selector_matrix.columns if str(column).startswith("test_")
    ]
    if test_columns:
        raise RankAnalysisError(
            f"Step 9 selector matrix contains forbidden test columns: {test_columns}"
        )
    required_selector_columns = IDENTITY_COLUMNS + [
        str(spec["score_column"]) for spec in selector_specs
    ]
    _require_columns(selector_matrix, required_selector_columns, "Step 9 selector matrix")
    if selector_matrix["candidate_id"].astype(str).duplicated().any():
        raise RankAnalysisError("Step 9 selector matrix has duplicate candidate IDs.")

    pool = load_complete_pool_index(config, source, pool_csv, pool_summary, frozen)
    expected_count = int(config["candidate_pool"]["expected_candidate_checkpoints"])
    if len(selector_matrix) != expected_count or len(pool) != expected_count:
        raise RankAnalysisError("Step 12 requires the complete locked candidate pool.")
    selector_ids = set(selector_matrix["candidate_id"].astype(str))
    pool_ids = set(pool["candidate_id"].astype(str))
    if selector_ids != pool_ids:
        raise RankAnalysisError("Step 9 and Step 11 candidate sets differ.")

    selector_matrix = selector_matrix.sort_values("candidate_id").reset_index(drop=True)
    pool = pool.set_index("candidate_id").loc[
        selector_matrix["candidate_id"].astype(str)
    ].reset_index()
    for column in IDENTITY_COLUMNS:
        left = selector_matrix[column].to_numpy()
        right = pool[column].to_numpy()
        if np.issubdtype(np.asarray(left).dtype, np.number):
            matches = np.allclose(left, right, rtol=0.0, atol=0.0)
        else:
            matches = np.array_equal(left.astype(str), right.astype(str))
        if not matches:
            raise RankAnalysisError(f"Step 9 and Step 11 {column} values differ.")

    analysis = selector_matrix[IDENTITY_COLUMNS].copy()
    analysis["biased_val_accuracy"] = selector_matrix["biased_val_accuracy"].to_numpy(
        float
    )
    analysis["biased_val_loss"] = selector_matrix["biased_val_loss"].to_numpy(float)
    analysis["fcv_main_score"] = selector_matrix["primary_selector_score"].to_numpy(
        float
    )
    analysis["fcv_counterfactual_accuracy"] = selector_matrix[
        "fcv_counterfactual_accuracy"
    ].to_numpy(float)
    analysis["fcv_stability"] = selector_matrix[
        "fcv_true_class_probability"
    ].to_numpy(float)
    analysis["oracle_group_val_score"] = selector_matrix[
        "oracle_validation_balanced_group_accuracy"
    ].to_numpy(float)
    for column in (
        "test_accuracy",
        "test_balanced_group_accuracy",
        "test_worst_group_accuracy",
    ):
        analysis[column] = pool[column].to_numpy(float)
    numeric_columns = [
        "biased_val_accuracy",
        "biased_val_loss",
        "fcv_main_score",
        "fcv_counterfactual_accuracy",
        "fcv_stability",
        "oracle_group_val_score",
        "test_accuracy",
        "test_balanced_group_accuracy",
        "test_worst_group_accuracy",
    ]
    if not np.isfinite(analysis[numeric_columns].to_numpy(float)).all():
        raise RankAnalysisError("Step 12 inputs contain non-finite metrics.")

    target = analysis["test_worst_group_accuracy"].to_numpy(float)
    candidate_ids = analysis["candidate_id"].astype(str).tolist()
    test_order = _deterministic_order(candidate_ids, target)
    pool_best_id = test_order[0]
    pool_best_value = float(target[candidate_ids.index(pool_best_id)])
    test_rank_by_id = {
        candidate_id: rank
        for candidate_id, rank in zip(
            candidate_ids, rankdata(-target, method="average")
        )
    }
    analysis["test_robust_rank"] = [
        test_rank_by_id[candidate_id] for candidate_id in candidate_ids
    ]

    frozen_rows = frozen.table.set_index("selector_name", drop=False)
    selected_ids: Dict[str, str] = {}
    result_rows: List[Dict[str, Any]] = []
    bootstrap_cfg = rank_cfg["clustered_bootstrap"]
    if bootstrap_cfg.get("enabled") is not True:
        raise RankAnalysisError("Run-cluster bootstrap is mandatory for Step 12.")
    cluster_column = str(bootstrap_cfg["cluster_column"])
    if cluster_column not in analysis.columns:
        raise RankAnalysisError(f"Missing bootstrap cluster column: {cluster_column}")
    clusters = analysis[cluster_column].to_numpy()
    top_k_values = [int(value) for value in rank_cfg["top_k_values"]]
    if (
        top_k_values != sorted(set(top_k_values))
        or any(value <= 0 or value > len(analysis) for value in top_k_values)
    ):
        raise RankAnalysisError(
            "Step 12 top-k values must be unique, ascending, positive, and no "
            "larger than the candidate pool."
        )
    for spec in selector_specs:
        name = str(spec["name"])
        score_column = str(spec["score_column"])
        direction = str(spec["direction"])
        if name not in frozen_rows.index:
            raise RankAnalysisError(f"Frozen selection lacks Step 12 selector {name}.")
        frozen_row = frozen_rows.loc[name]
        if str(frozen_row["direction"]) != direction:
            raise RankAnalysisError(f"Frozen direction differs for selector {name}.")
        raw_scores = selector_matrix[score_column].to_numpy(float)
        if not np.isfinite(raw_scores).all():
            raise RankAnalysisError(f"Selector {name} contains non-finite scores.")
        oriented, multiplier = _oriented_scores(raw_scores, direction)
        order = _deterministic_order(candidate_ids, oriented)
        selected_id = str(frozen_row["selected_checkpoint_id"])
        if order[0] != selected_id:
            raise RankAnalysisError(
                f"Recomputed selection for {name} is {order[0]}, not frozen {selected_id}."
            )
        selected_ids[name] = selected_id
        analysis[f"raw_score__{name}"] = raw_scores
        analysis[f"oriented_score__{name}"] = oriented
        selector_ranks = rankdata(-oriented, method=str(rank_cfg["spearman_rank_method"]))
        analysis[f"rank__{name}"] = selector_ranks
        rank_by_id = dict(zip(candidate_ids, selector_ranks))
        selected_test = float(
            analysis.loc[
                analysis["candidate_id"] == selected_id,
                "test_worst_group_accuracy",
            ].iloc[0]
        )
        row: Dict[str, Any] = {
            "selector_name": name,
            "display_name": str(spec["display_name"]),
            "selector_score_column": score_column,
            "direction": direction,
            "orientation_multiplier": multiplier,
            "candidate_count": len(analysis),
            **_correlations(
                oriented,
                target,
                kendall_variant=str(rank_cfg["kendall_variant"]),
            ),
            **_cluster_bootstrap_correlations(
                oriented,
                target,
                clusters,
                kendall_variant=str(rank_cfg["kendall_variant"]),
                replicates=int(bootstrap_cfg["replicates"]),
                seed=int(bootstrap_cfg["seed"]) + len(result_rows),
                confidence_level=float(bootstrap_cfg["confidence_level"]),
            ),
            "selected_candidate_id": selected_id,
            "selected_test_worst_group_accuracy": selected_test,
            "selected_test_robust_rank": float(test_rank_by_id[selected_id]),
            "selection_regret_to_pool_best": pool_best_value - selected_test,
            "pool_best_candidate_id": pool_best_id,
            "pool_best_test_worst_group_accuracy": pool_best_value,
            "pool_best_selector_rank": float(rank_by_id[pool_best_id]),
        }
        for k in top_k_values:
            selector_top = set(order[:k])
            robust_top = set(test_order[:k])
            overlap = len(selector_top.intersection(robust_top))
            row[f"top_{k}_overlap_count"] = overlap
            row[f"top_{k}_recall"] = float(overlap / k)
            row[f"top_{k}_hit"] = bool(overlap > 0)
        result_rows.append(row)

    results = pd.DataFrame(result_rows)
    output_results_csv = Path(output_results_csv).expanduser().resolve()
    output_candidates_csv = Path(output_candidates_csv).expanduser().resolve()
    output_summary = Path(output_summary).expanduser().resolve()
    plot_dir = Path(plot_dir).expanduser().resolve()
    _atomic_csv(results, output_results_csv)
    _atomic_csv(analysis, output_candidates_csv)
    should_plot = (
        bool(rank_cfg["create_scatter_plots"])
        if create_plots is None
        else bool(create_plots)
    )
    plot_paths = (
        _plot_selector_scatters(
            analysis,
            selector_specs,
            selected_ids,
            pool_best_id,
            plot_dir,
        )
        if should_plot
        else []
    )
    pool_csv = Path(pool_csv).expanduser().resolve()
    pool_summary = Path(pool_summary).expanduser().resolve()
    summary = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_rank_analysis_summary",
        "status": "complete",
        "rank_analysis_fingerprint": rank_analysis_fingerprint(config),
        "candidate_count": len(analysis),
        "selector_count": len(results),
        "selector_names": [str(spec["name"]) for spec in selector_specs],
        "target_metric": str(rank_cfg["target_metric"]),
        "fcv_stability_definition": str(rank_cfg["fcv_stability_definition"]),
        "top_k_values": top_k_values,
        "correlation_inference": {
            "coefficients": "descriptive_fixed_candidate_pool",
            "uncertainty": "run_cluster_bootstrap",
            "cluster_column": cluster_column,
            "replicates": int(bootstrap_cfg["replicates"]),
            "seed": int(bootstrap_cfg["seed"]),
            "confidence_level": float(bootstrap_cfg["confidence_level"]),
            "naive_epoch_level_pvalues_reported": False,
        },
        "selection_table_path": str(frozen.selection_table_path),
        "selection_table_sha256": frozen.selection_table_sha256,
        "candidate_selector_matrix_path": str(matrix_path),
        "candidate_selector_matrix_sha256": frozen.selector_matrix_sha256,
        "candidate_pool_test_scores_path": str(pool_csv),
        "candidate_pool_test_scores_sha256": _sha256_file(pool_csv),
        "candidate_pool_test_scores_summary_path": str(pool_summary),
        "candidate_pool_test_scores_summary_sha256": _sha256_file(pool_summary),
        "test_manifest_path": str(source.manifest_path),
        "test_manifest_sha256": source.manifest_sha256,
        "test_sample_count": source.sample_count,
        "rank_correlation_results_path": str(output_results_csv),
        "rank_correlation_results_sha256": _sha256_file(output_results_csv),
        "candidate_rank_analysis_path": str(output_candidates_csv),
        "candidate_rank_analysis_sha256": _sha256_file(output_candidates_csv),
        "scatter_plot_paths": [str(path) for path in plot_paths],
        "scatter_plot_sha256": {
            str(path): _sha256_file(path) for path in plot_paths
        },
        "pool_best_candidate_id": pool_best_id,
        "pool_best_test_worst_group_accuracy": pool_best_value,
        "posthoc_analysis_only": True,
        "selection_was_frozen_before_test": True,
        "test_metrics_affected_selection": False,
    }
    _atomic_json(summary, output_summary)
    return summary
