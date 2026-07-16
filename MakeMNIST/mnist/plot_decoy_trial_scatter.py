#!/usr/bin/env python3
"""Plot DecoyMNIST trial-level selector diagnostics from trial_eval.csv."""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def _to_array(rows, key):
    vals = []
    for row in rows:
        v = row.get(key, "")
        if v in ("", None):
            continue
        vals.append(float(v))
    return np.asarray(vals, dtype=np.float64)


def _rankdata(x):
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x, y):
    if len(x) < 2:
        return np.nan
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    if len(x) < 2:
        return np.nan
    return _pearson(_rankdata(x), _rankdata(y))


def _annotate_corr(ax, x, y):
    r = _pearson(x, y)
    rho = _spearman(x, y)
    ax.text(
        0.03,
        0.97,
        f"Pearson r = {r:.3f}\nSpearman rho = {rho:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"),
    )


def _scatter(ax, x, y, x_label, y_label, title):
    ax.scatter(x, y, alpha=0.85, s=28)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    _annotate_corr(ax, x, y)


def main():
    parser = argparse.ArgumentParser(description="Plot scatter diagnostics from trial_eval.csv")
    parser.add_argument("--eval-csv", required=True, help="CSV from eval_decoy_trial_checkpoints.py")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: alongside eval CSV).")
    parser.add_argument("--prefix", default="decoy_selector", help="Output filename prefix.")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.dirname(args.eval_csv) or "."
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.eval_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {args.eval_csv}")

    # Use log-space optim metric when available (numerically stable).
    use_log_optim = all(row.get("best_log_optim", "") not in ("", None) for row in rows)
    optim_key = "best_log_optim" if use_log_optim else "best_optim_value"
    optim_label = "Best log_optim (trial)" if use_log_optim else "Best optim_value (trial)"

    # Filter rows that have all needed values.
    filtered = []
    for row in rows:
        required = ["best_val_acc", optim_key, "test_acc_valSel", "test_acc_optimSel"]
        if any(row.get(k, "") in ("", None) for k in required):
            continue
        filtered.append(row)
    if not filtered:
        raise RuntimeError("No valid rows with required columns for plotting.")

    best_val_acc = _to_array(filtered, "best_val_acc")
    best_optim_metric = _to_array(filtered, optim_key)
    test_acc_val_sel = _to_array(filtered, "test_acc_valSel")
    test_acc_optim_sel = _to_array(filtered, "test_acc_optimSel")
    selection_gain = test_acc_optim_sel - test_acc_val_sel

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    _scatter(
        axes[0],
        best_val_acc,
        test_acc_val_sel,
        "Best val_acc (trial)",
        "Test acc @ val selector (%)",
        "Val Selector Predictiveness",
    )

    _scatter(
        axes[1],
        best_optim_metric,
        test_acc_optim_sel,
        optim_label,
        "Test acc @ optim selector (%)",
        "Optim Selector Predictiveness",
    )

    axes[2].scatter(test_acc_val_sel, test_acc_optim_sel, alpha=0.85, s=28)
    lo = min(float(test_acc_val_sel.min()), float(test_acc_optim_sel.min()))
    hi = max(float(test_acc_val_sel.max()), float(test_acc_optim_sel.max()))
    pad = max(1e-6, 0.03 * (hi - lo))
    lo -= pad
    hi += pad
    axes[2].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="black", alpha=0.9)
    axes[2].set_xlim(lo, hi)
    axes[2].set_ylim(lo, hi)
    axes[2].set_xlabel("Test acc @ val selector (%)")
    axes[2].set_ylabel("Test acc @ optim selector (%)")
    axes[2].set_title("Selection Gain (above diagonal = optim wins)")
    axes[2].grid(alpha=0.2)
    axes[2].text(
        0.03,
        0.97,
        f"mean gain = {float(selection_gain.mean()):.3f}\nmedian gain = {float(np.median(selection_gain)):.3f}",
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"),
    )

    fig.suptitle("DecoyMNIST Trial-Level Selection Diagnostics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = os.path.join(args.out_dir, f"{args.prefix}_all_scatter.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print(f"Wrote plot: {out_png}")
    print(f"rows={len(filtered)}")


if __name__ == "__main__":
    main()
