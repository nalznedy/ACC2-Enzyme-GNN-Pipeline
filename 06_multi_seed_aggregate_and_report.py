#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 06_multi_seed_aggregate_and_report.py

Purpose:
    Aggregate multi-seed model outputs and generate reporting tables and figures.

Workflow step:
    Step 6 - cross-seed summary and statistical comparison.

Main inputs:
    - metrics/gnn_metrics_seed###.csv
    - metrics/gnn_fusion_metrics_seed###.csv
    - Optional prediction CSVs for overlays

Main outputs:
    - release_gnn_reporting/tables/test_metrics_summary_mean_sd.csv
    - release_gnn_reporting/tables/paired_statistics_test.csv
    - release_gnn_reporting/tables/merged_metrics_all_models_all_splits.csv
    - release_gnn_reporting/figures/*.svg|png
    - release_gnn_reporting/manifest_script7.json
    - release_gnn_reporting/README.txt

Notes:
    Reports mean +- SD across seeds and paired statistical tests with FDR correction.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

# Optional SciPy (if available)
try:
    from scipy.stats import ttest_rel, wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# -----------------------------
# Publication style
# -----------------------------
@dataclass
class FigStyle:
    font_family: str = "Times New Roman"
    title_size: int = 18
    label_size: int = 16
    tick_size: int = 14
    legend_size: int = 14
    line_width: float = 2.0
    axis_width: float = 2.0
    dpi_png: int = 300

    def apply(self) -> None:
        mpl.rcParams["font.family"] = self.font_family
        mpl.rcParams["font.weight"] = "bold"
        mpl.rcParams["axes.titleweight"] = "bold"
        mpl.rcParams["axes.labelweight"] = "bold"
        mpl.rcParams["axes.titlesize"] = self.title_size
        mpl.rcParams["axes.labelsize"] = self.label_size
        mpl.rcParams["xtick.labelsize"] = self.tick_size
        mpl.rcParams["ytick.labelsize"] = self.tick_size
        mpl.rcParams["legend.fontsize"] = self.legend_size
        mpl.rcParams["axes.linewidth"] = self.axis_width
        mpl.rcParams["lines.linewidth"] = self.line_width
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42


def _save_fig(fig: plt.Figure, out_svg: Path, out_png: Path, dpi: int) -> None:
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def ensure_dirs(out_root: Path, task: str) -> Dict[str, Path]:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    release_dir = out_root / f"release_gnn_reporting_{task}_{stamp}"
    tables_dir = release_dir / "tables"
    figures_dir = release_dir / "figures"
    merged_dir = release_dir / "merged"
    env_dir = release_dir / "environment"
    for d in [release_dir, tables_dir, figures_dir, merged_dir, env_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {
        "release": release_dir,
        "tables": tables_dir,
        "figures": figures_dir,
        "merged": merged_dir,
        "env": env_dir,
    }


# -----------------------------
# Loading utilities
# -----------------------------
def read_metrics_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {path}")
    df = pd.read_csv(path)
    required = {"seed", "split", "model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metrics file {path.name} missing columns: {sorted(missing)}")
    return df


def find_metrics_files(out_root: Path, seeds: List[int]) -> Tuple[List[Path], List[Path]]:
    """
    Supports:
      - Seed-separated layout: out_root/seed_####/metrics/*.csv
      - Flat layout:          out_root/metrics/*.csv
    """
    gnn_files: List[Path] = []
    fusion_files: List[Path] = []

    for s in seeds:
        # Preferred: per-seed
        seed_root = out_root / f"seed_{s:04d}"
        metrics_dir_seed = seed_root / "metrics"
        g_seed = metrics_dir_seed / f"gnn_metrics_seed{s:03d}.csv"
        f_seed = metrics_dir_seed / f"gnn_fusion_metrics_seed{s:03d}.csv"

        # Fallback: flat
        metrics_dir_flat = out_root / "metrics"
        g_flat = metrics_dir_flat / f"gnn_metrics_seed{s:03d}.csv"
        f_flat = metrics_dir_flat / f"gnn_fusion_metrics_seed{s:03d}.csv"

        if g_seed.exists():
            gnn_files.append(g_seed)
        elif g_flat.exists():
            gnn_files.append(g_flat)

        if f_seed.exists():
            fusion_files.append(f_seed)
        elif f_flat.exists():
            fusion_files.append(f_flat)

    if len(gnn_files) == 0:
        raise FileNotFoundError("No GNN metrics found (checked seed folders and flat metrics/).")
    if len(fusion_files) == 0:
        raise FileNotFoundError("No GNN_FUSION metrics found (checked seed folders and flat metrics/).")

    return gnn_files, fusion_files


def read_predictions(out_root: Path, model_key: str, seed: int, split: str = "test") -> Optional[pd.DataFrame]:
    """
    Supports:
      - Seed-separated: out_root/seed_####/predictions/*.csv
      - Flat:           out_root/predictions/*.csv
    Requires columns: y_cls_true, y_cls_prob for overlays.
    """
    if model_key == "gnn":
        fname = f"gnn_seed{seed:03d}_{split}.csv"
    elif model_key == "gnn_fusion":
        fname = f"gnn_fusion_seed{seed:03d}_{split}.csv"
    else:
        raise ValueError("Unknown model_key")

    p1 = out_root / f"seed_{seed:04d}" / "predictions" / fname
    if p1.exists():
        return pd.read_csv(p1)

    p2 = out_root / "predictions" / fname
    if p2.exists():
        return pd.read_csv(p2)

    return None


# -----------------------------
# Statistics
# -----------------------------
def mean_sd(arr: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def paired_t_test(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Paired comparison of y vs x (same seeds).
    Returns p-value for H0: mean(y-x)=0.
    Fallback if SciPy not available uses normal approximation (conservative for small n).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return {"n": float(x.size), "t": float("nan"), "p": float("nan")}

    if _HAVE_SCIPY:
        res = ttest_rel(y, x, nan_policy="omit")
        return {"n": float(x.size), "t": float(res.statistic), "p": float(res.pvalue)}

    d = y - x
    n = d.size
    md = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return {"n": float(n), "t": float("inf"), "p": 0.0}
    t = md / (sd / math.sqrt(n))

    # normal-approx p-value (fallback)
    z = abs(t)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return {"n": float(n), "t": float(t), "p": float(p)}


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return {"n": float(x.size), "stat": float("nan"), "p": float("nan")}
    if not _HAVE_SCIPY:
        return {"n": float(x.size), "stat": float("nan"), "p": float("nan")}
    res = wilcoxon(y, x, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
    return {"n": float(x.size), "stat": float(res.statistic), "p": float(res.pvalue)}


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR correction.
    """
    pvals = np.asarray(pvals, dtype=float)
    q = np.full_like(pvals, np.nan, dtype=float)
    mask = ~np.isnan(pvals)
    pv = pvals[mask]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranks = np.arange(1, pv.size + 1)
    qv = pv[order] * pv.size / ranks
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    qv = np.clip(qv, 0.0, 1.0)
    out = np.empty_like(pv)
    out[order] = qv
    q[mask] = out
    return q


# -----------------------------
# Plotting summary
# -----------------------------
def barplot_with_error(
    df_summary: pd.DataFrame,
    metric: str,
    higher_is_better: bool,
    figs_dir: Path,
    style: FigStyle,
    title: str,
    fname_prefix: str,
) -> None:
    """
    df_summary columns expected:
      model, metric, mean, sd
    """
    sub = df_summary[df_summary["metric"] == metric].copy()
    if sub.empty:
        return

    order = ["GNN", "GNN_FUSION"]
    sub["model"] = pd.Categorical(sub["model"], categories=order, ordered=True)
    sub = sub.sort_values("model")

    fig = plt.figure(figsize=(6.8, 5.6))
    ax = fig.add_subplot(111)
    x = np.arange(len(sub))
    ax.bar(x, sub["mean"].values, yerr=sub["sd"].values, capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["model"].astype(str).tolist(), fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(metric, fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    for i, (m, s) in enumerate(zip(sub["mean"].values, sub["sd"].values)):
        txt = f"{m:.3f} ± {s:.3f}"
        ax.text(i, m + (0.02 * (abs(m) + 1e-6)), txt, ha="center", va="bottom", fontweight="bold")

    if higher_is_better:
        ax.set_ylim(bottom=min(0.0, float(np.nanmin(sub["mean"].values - sub["sd"].values)) - 0.05))

    _save_fig(fig, figs_dir / f"{fname_prefix}_{metric}.svg", figs_dir / f"{fname_prefix}_{metric}.png", dpi=style.dpi_png)


def overlay_pr_curves(out_root: Path, seeds: List[int], model_key: str, figs_dir: Path, style: FigStyle, fname: str) -> None:
    """
    Overlay PR curves across seeds for a given model, using test predictions.
    Requires columns: y_cls_true, y_cls_prob
    """
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)
    plotted = 0

    # local import to avoid hard dependency at script import time
    from sklearn.metrics import precision_recall_curve, average_precision_score

    for s in seeds:
        df = read_predictions(out_root, model_key=model_key, seed=s, split="test")
        if df is None:
            continue
        if ("y_cls_true" not in df.columns) or ("y_cls_prob" not in df.columns):
            continue
        y_true = df["y_cls_true"].to_numpy()
        y_prob = df["y_cls_prob"].to_numpy()
        mask = (y_true >= 0) & (~np.isnan(y_prob))
        y_true = y_true[mask].astype(int)
        y_prob = y_prob[mask].astype(float)
        if y_true.size == 0 or len(np.unique(y_true)) < 2:
            continue

        p, r, _ = precision_recall_curve(y_true, y_prob)
        ap = float(average_precision_score(y_true, y_prob))
        ax.plot(r, p, label=f"Seed {s} (PR-AUC={ap:.3f})")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_title(f"Precision–Recall (Test): {model_key.upper()} (multi-seed)", fontweight="bold")
    ax.set_xlabel("Recall", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.legend(frameon=False)
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)
    _save_fig(fig, figs_dir / f"{fname}.svg", figs_dir / f"{fname}.png", dpi=style.dpi_png)


def overlay_roc_curves(out_root: Path, seeds: List[int], model_key: str, figs_dir: Path, style: FigStyle, fname: str) -> None:
    """
    Overlay ROC curves across seeds for a given model, plotting TRAIN and TEST in the same figure.

    For each seed:
      - Test curve: solid line, legend includes AUROC.
      - Train curve: dotted line, legend includes AUROC.

    Requires per-split prediction CSVs with columns: y_cls_true, y_cls_prob.
    """
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)
    plotted = 0

    from sklearn.metrics import roc_curve, roc_auc_score

    def _plot_one(df: pd.DataFrame, label: str, linestyle: str) -> bool:
        if df is None:
            return False
        if ("y_cls_true" not in df.columns) or ("y_cls_prob" not in df.columns):
            return False
        y_true = df["y_cls_true"].to_numpy()
        y_prob = df["y_cls_prob"].to_numpy()
        mask = (y_true >= 0) & (~np.isnan(y_prob))
        y_true = y_true[mask].astype(int)
        y_prob = y_prob[mask].astype(float)
        if y_true.size == 0 or len(np.unique(y_true)) < 2:
            return False
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = float(roc_auc_score(y_true, y_prob))
        ax.plot(fpr, tpr, linestyle=linestyle, label=f"{label} (AUROC={auc:.3f})")
        return True

    for s in seeds:
        df_test = read_predictions(out_root, model_key=model_key, seed=s, split="test")
        df_train = read_predictions(out_root, model_key=model_key, seed=s, split="train")

        ok_test = _plot_one(df_test, label=f"Seed {s} Test", linestyle="-")
        ok_train = _plot_one(df_train, label=f"Seed {s} Train", linestyle=":")

        if ok_test or ok_train:
            plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_title(f"ROC (Train + Test): {model_key.upper()} (multi-seed)", fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.legend(frameon=False)
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)
    _save_fig(fig, figs_dir / f"{fname}.svg", figs_dir / f"{fname}.png", dpi=style.dpi_png)


# -----------------------------
# Environment capture
# -----------------------------
def capture_pip_freeze(out_path: Path) -> None:
    try:
        res = subprocess.run(["python", "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
        out_path.write_text(res.stdout, encoding="utf-8")
    except Exception:
        out_path.write_text("pip freeze failed.\n", encoding="utf-8")


# -----------------------------
# Core aggregation
# -----------------------------
def build_long_metrics(gnn_files: List[Path], fusion_files: List[Path]) -> pd.DataFrame:
    parts = []
    for p in gnn_files:
        df = read_metrics_csv(p)
        df["source_file"] = p.name
        df["source_path"] = str(p)
        parts.append(df)
    for p in fusion_files:
        df = read_metrics_csv(p)
        df["source_file"] = p.name
        df["source_path"] = str(p)
        parts.append(df)

    all_df = pd.concat(parts, ignore_index=True)

    # normalize
    all_df["model"] = all_df["model"].astype(str)
    all_df["split"] = all_df["split"].astype(str).str.lower()
    return all_df


def summarize_test_metrics(long_df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    Returns summary df with columns:
      model, metric, mean, sd, n
    """
    rows = []
    test_df = long_df[long_df["split"] == "test"].copy()

    for model in ["GNN", "GNN_FUSION"]:
        sub_m = test_df[test_df["model"] == model]
        for metric in metrics:
            if metric not in sub_m.columns:
                continue
            vals = sub_m[metric].to_numpy(dtype=float)
            m, s = mean_sd(vals)
            n = int(np.sum(~np.isnan(vals)))
            rows.append({"model": model, "metric": metric, "mean": m, "sd": s, "n": n})

    return pd.DataFrame(rows)


def paired_stats_test(long_df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    Paired comparison between models per seed for test split.
    """
    test_df = long_df[long_df["split"] == "test"].copy()
    rows = []

    for metric in metrics:
        if metric not in test_df.columns:
            continue
        pivot = test_df.pivot_table(index="seed", columns="model", values=metric, aggfunc="mean")
        if ("GNN" not in pivot.columns) or ("GNN_FUSION" not in pivot.columns):
            continue
        x = pivot["GNN"].to_numpy(dtype=float)
        y = pivot["GNN_FUSION"].to_numpy(dtype=float)

        t_res = paired_t_test(x, y)
        w_res = paired_wilcoxon(x, y)

        rows.append({
            "metric": metric,
            "n": int(t_res["n"]),
            "paired_t_t": float(t_res["t"]),
            "paired_t_p": float(t_res["p"]),
            "wilcoxon_stat": float(w_res["stat"]),
            "wilcoxon_p": float(w_res["p"]),
        })

    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        if "paired_t_p" in stats_df.columns:
            stats_df["paired_t_q_fdr"] = fdr_bh(stats_df["paired_t_p"].to_numpy(dtype=float))
        if "wilcoxon_p" in stats_df.columns:
            stats_df["wilcoxon_q_fdr"] = fdr_bh(stats_df["wilcoxon_p"].to_numpy(dtype=float))
    return stats_df


def default_metric_sets(task: str) -> Tuple[List[str], Dict[str, bool]]:
    """
    Returns:
      metrics list
      higher_is_better map
    """
    higher_is_better: Dict[str, bool] = {}
    metrics: List[str] = []

    if task in ("regression", "both"):
        metrics += ["rmse", "mae", "r2", "pearson_r", "spearman_rho"]
        higher_is_better.update({
            "rmse": False,
            "mae": False,
            "r2": True,
            "pearson_r": True,
            "spearman_rho": True
        })

    if task in ("classification", "both"):
        metrics += ["auroc", "pr_auc", "mcc", "bal_acc", "f1", "brier"]
        higher_is_better.update({
            "auroc": True,
            "pr_auc": True,
            "mcc": True,
            "bal_acc": True,
            "f1": True,
            "brier": False
        })

    return metrics, higher_is_better


# -----------------------------
# README bundle
# -----------------------------
def write_release_readme(
    out_path: Path,
    out_root: Path,
    seeds: List[int],
    task: str,
    gnn_files: List[Path],
    fusion_files: List[Path],
) -> None:
    lines: List[str] = []
    lines.append("GNN QSAR Reporting Package (Multi-seed)\n")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z\n")
    lines.append(f"Project root: {out_root}\n")
    lines.append(f"Seeds: {', '.join(str(s) for s in seeds)}\n")
    lines.append(f"Task: {task}\n")
    lines.append(f"SciPy available: {_HAVE_SCIPY}\n")
    lines.append("\nInputs collected:\n")
    lines.append("  Script-5 (GNN) metrics:\n")
    for p in gnn_files:
        lines.append(f"    - {p}\n")
    lines.append("  Script-6 (GNN+Morgan fusion) metrics:\n")
    for p in fusion_files:
        lines.append(f"    - {p}\n")

    lines.append("\nOutputs in this release folder:\n")
    lines.append("  merged/\n")
    lines.append("    - merged_metrics_all_models_all_splits.csv\n")
    lines.append("  tables/\n")
    lines.append("    - test_metrics_summary_mean_sd.csv\n")
    lines.append("    - paired_statistics_test.csv\n")
    lines.append("  figures/\n")
    lines.append("    - barplots for selected test metrics (SVG + PNG)\n")
    lines.append("    - optional multi-seed ROC/PR overlays (SVG + PNG)\n")
    lines.append("  environment/\n")
    lines.append("    - pip_freeze.txt\n")

    lines.append("\nRecommended Methods text:\n")
    lines.append("  Models were evaluated on scaffold-split test sets across multiple random seeds and reported as mean ± SD.\n")
    lines.append("  Hyperparameter tuning and threshold selection were performed using train/validation only; test sets were not used for tuning.\n")
    lines.append("  Paired tests were computed across seeds, with Benjamini–Hochberg FDR correction across metrics.\n")

    out_path.write_text("".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate multi-seed GNN QSAR results into publication-ready reporting.")
    ap.add_argument("--out_root", type=str, required=True, help="Project root directory (QSAR_GNN_Project).")
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="List of seeds used (e.g., 2023 2024 2025).")
    ap.add_argument("--task", type=str, choices=["regression", "classification", "both"], default="both")
    ap.add_argument("--make_overlays", action="store_true", help="Create multi-seed ROC/PR overlays if predictions exist.")
    ap.add_argument("--barplot_metrics", type=str, nargs="*", default=None,
                    help="Which metrics to plot; default selects standard metrics for the task.")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    seeds = [int(s) for s in args.seeds]

    dirs = ensure_dirs(out_root, args.task)
    style = FigStyle()
    style.apply()

    # Collect metrics files
    gnn_files, fusion_files = find_metrics_files(out_root, seeds)

    # Build long metrics table
    long_df = build_long_metrics(gnn_files, fusion_files)

    # Save merged table
    merged_csv = dirs["merged"] / "merged_metrics_all_models_all_splits.csv"
    long_df.to_csv(merged_csv, index=False)

    # Decide metrics set
    metrics, hib_map = default_metric_sets(args.task)

    if args.barplot_metrics is not None and len(args.barplot_metrics) > 0:
        metrics_to_plot = args.barplot_metrics
    else:
        metrics_to_plot: List[str] = []
        if args.task in ("classification", "both"):
            metrics_to_plot += ["pr_auc", "auroc", "mcc"]
        if args.task in ("regression", "both"):
            metrics_to_plot += ["rmse", "r2", "mae"]
        # unique preserving order
        seen = set()
        metrics_to_plot = [m for m in metrics_to_plot if not (m in seen or seen.add(m))]

    # Summaries
    summary_df = summarize_test_metrics(long_df, metrics=metrics)
    summary_csv = dirs["tables"] / "test_metrics_summary_mean_sd.csv"
    summary_df.to_csv(summary_csv, index=False)

    stats_df = paired_stats_test(long_df, metrics=metrics)
    stats_csv = dirs["tables"] / "paired_statistics_test.csv"
    stats_df.to_csv(stats_csv, index=False)

    # Barplots for selected metrics
    for m in metrics_to_plot:
        if m not in hib_map:
            continue
        barplot_with_error(
            df_summary=summary_df,
            metric=m,
            higher_is_better=hib_map[m],
            figs_dir=dirs["figures"],
            style=style,
            title=f"Test performance (mean ± SD): {m}",
            fname_prefix="30_test_barplot",
        )

    # Optional overlays
    if args.make_overlays and (args.task in ("classification", "both")):
        overlay_pr_curves(out_root, seeds, model_key="gnn", figs_dir=dirs["figures"], style=style, fname="31_pr_overlay_gnn_test")
        overlay_pr_curves(out_root, seeds, model_key="gnn_fusion", figs_dir=dirs["figures"], style=style, fname="32_pr_overlay_gnn_fusion_test")
        overlay_roc_curves(out_root, seeds, model_key="gnn", figs_dir=dirs["figures"], style=style, fname="33_roc_overlay_gnn_test")
        overlay_roc_curves(out_root, seeds, model_key="gnn_fusion", figs_dir=dirs["figures"], style=style, fname="34_roc_overlay_gnn_fusion_test")

    # Environment capture
    pip_freeze_path = dirs["env"] / "pip_freeze.txt"
    capture_pip_freeze(pip_freeze_path)

    # README
    readme_path = dirs["release"] / "README.txt"
    write_release_readme(readme_path, out_root, seeds, args.task, gnn_files, fusion_files)

    # Manifest
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "project_root": str(out_root),
        "seeds": seeds,
        "task": args.task,
        "inputs": {
            "gnn_metrics_files": [str(p) for p in gnn_files],
            "gnn_fusion_metrics_files": [str(p) for p in fusion_files],
        },
        "outputs": {
            "release_dir": str(dirs["release"]),
            "merged_metrics_csv": str(merged_csv),
            "summary_csv": str(summary_csv),
            "paired_stats_csv": str(stats_csv),
            "figures_dir": str(dirs["figures"]),
            "pip_freeze": str(pip_freeze_path),
            "readme": str(readme_path),
        },
        "scipy_available": bool(_HAVE_SCIPY),
    }
    manifest_path = dirs["release"] / "manifest_script7.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== DONE: Multi-seed aggregation and reporting package ===")
    print(f"Release folder:  {dirs['release']}")
    print(f"Merged metrics:  {merged_csv}")
    print(f"Summary table:   {summary_csv}")
    print(f"Paired stats:    {stats_csv}")
    print(f"Figures (SVG):   {dirs['figures']}")
    print(f"README:          {readme_path}")
    print(f"Manifest:        {manifest_path}")
    print("=========================================================\n")


if __name__ == "__main__":
    main()
