#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 06b_probability_calibration_val_only.py

Purpose:
    Fit and apply validation-only probability calibration for classification outputs.

Workflow step:
    Step 6b - post hoc calibration for screening-ready probabilities.

Main inputs:
    - predictions/{model_key}_seed###_val.csv
    - predictions/{model_key}_seed###_test.csv

Main outputs:
    - predictions/{model_key}_seed###_{val,test}_calibrated_{method}.csv
    - metrics/{model_key}_calibrated_{method}_metrics_seed###.*
    - figures/calibration_*.svg|png

Notes:
    Calibration models are fit on validation data only to avoid information leakage.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


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


def ensure_dirs(out_root: Path) -> Dict[str, Path]:
    preds_dir = out_root / "predictions"
    metrics_dir = out_root / "metrics"
    figs_dir = out_root / "figures"
    reports_dir = out_root / "reports"
    for d in [preds_dir, metrics_dir, figs_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {"predictions": preds_dir, "metrics": metrics_dir, "figures": figs_dir, "reports": reports_dir}


# -----------------------------
# Utilities
# -----------------------------
def _load_pred(out_root: Path, model_key: str, seed: int, split: str) -> pd.DataFrame:
    preds_dir = out_root / "predictions"
    if model_key == "gnn":
        fname = f"gnn_seed{seed:03d}_{split}.csv"
    elif model_key == "gnn_fusion":
        fname = f"gnn_fusion_seed{seed:03d}_{split}.csv"
    else:
        raise ValueError("model_key must be gnn or gnn_fusion")

    path = preds_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")

    df = pd.read_csv(path)
    needed = {"y_cls_true", "y_cls_prob"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

    return df


def _filter_valid(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    y = df["y_cls_true"].to_numpy()
    p = df["y_cls_prob"].to_numpy(dtype=float)
    mask = (y >= 0) & (~np.isnan(p))
    y = y[mask].astype(int)
    p = np.clip(p[mask].astype(float), 0.0, 1.0)
    if y.size == 0:
        raise ValueError("No valid labeled samples (y_cls_true>=0) found.")
    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes in validation to calibrate and evaluate.")
    return y, p


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Simple ECE with uniform bins over [0,1].
    """
    y_true = y_true.astype(int)
    y_prob = np.clip(y_prob.astype(float), 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = y_true.size

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if m.sum() == 0:
            continue
        acc = float(y_true[m].mean())
        conf = float(y_prob[m].mean())
        ece += (m.sum() / n) * abs(acc - conf)

    return float(ece)


def select_threshold_by_mcc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    grid = np.linspace(0.0, 1.0, 1001)
    best_t, best_m = 0.5, -1.0
    for t in grid:
        m = matthews_corrcoef(y_true, (y_prob >= t).astype(int))
        if m > best_m:
            best_m = m
            best_t = float(t)
    return float(best_t)


def compute_cls_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out: Dict[str, float] = {}
    out["auroc"] = float(roc_auc_score(y_true, y_prob))
    out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    out["bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    out["ece"] = float(expected_calibration_error(y_true, y_prob, n_bins=10))
    out["threshold"] = float(threshold)
    return out


# -----------------------------
# Calibrators (VAL-only)
# -----------------------------
class PlattCalibrator:
    """
    Fits LogisticRegression on logit(p_uncal), then outputs calibrated probability.
    """
    def __init__(self) -> None:
        self.lr = LogisticRegression(solver="lbfgs", max_iter=2000)

    def fit(self, y: np.ndarray, p: np.ndarray) -> None:
        x = _logit(p).reshape(-1, 1)
        self.lr.fit(x, y)

    def predict(self, p: np.ndarray) -> np.ndarray:
        x = _logit(p).reshape(-1, 1)
        return self.lr.predict_proba(x)[:, 1].astype(float)


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, y: np.ndarray, p: np.ndarray) -> None:
        self.iso.fit(p.astype(float), y.astype(int))

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.clip(self.iso.predict(p.astype(float)).astype(float), 0.0, 1.0)


def make_calibrator(method: str):
    method = method.lower().strip()
    if method == "platt":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    raise ValueError("method must be platt or isotonic")


# -----------------------------
# Figures
# -----------------------------
def plot_reliability_before_after(
    y_val: np.ndarray,
    p_val_before: np.ndarray,
    p_val_after: np.ndarray,
    figs_dir: Path,
    style: FigStyle,
    prefix: str,
    bins: int,
) -> None:
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)

    frac_b, mean_b = calibration_curve(y_val, p_val_before, n_bins=bins, strategy="uniform")
    frac_a, mean_a = calibration_curve(y_val, p_val_after, n_bins=bins, strategy="uniform")

    ax.plot(mean_b, frac_b, marker="o", label="Before")
    ax.plot(mean_a, frac_a, marker="o", label="After")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal")

    ax.set_title("Calibration (VAL): Reliability Diagram", fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax.set_ylabel("Fraction of Positives", fontweight="bold")
    ax.legend(frameon=False)

    ax.tick_params(width=style.axis_width)
    for s in ax.spines.values():
        s.set_linewidth(style.axis_width)

    _save_fig(fig, figs_dir / f"{prefix}_reliability_val.svg", figs_dir / f"{prefix}_reliability_val.png", dpi=style.dpi_png)


def plot_prob_hist_before_after(
    p_before: np.ndarray,
    p_after: np.ndarray,
    figs_dir: Path,
    style: FigStyle,
    prefix: str,
) -> None:
    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_subplot(111)
    ax.hist(p_before, bins=30, alpha=0.60, label="Before")
    ax.hist(p_after, bins=30, alpha=0.60, label="After")
    ax.set_title("Probability Distribution (VAL)", fontweight="bold")
    ax.set_xlabel("Predicted Probability", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.legend(frameon=False)

    ax.tick_params(width=style.axis_width)
    for s in ax.spines.values():
        s.set_linewidth(style.axis_width)

    _save_fig(fig, figs_dir / f"{prefix}_prob_hist_val.svg", figs_dir / f"{prefix}_prob_hist_val.png", dpi=style.dpi_png)


def plot_roc_pr_before_after(
    y: np.ndarray,
    p_before: np.ndarray,
    p_after: np.ndarray,
    figs_dir: Path,
    style: FigStyle,
    prefix: str,
    split_name: str,
) -> None:
    # ROC
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)

    auc_b = float(roc_auc_score(y, p_before))
    auc_a = float(roc_auc_score(y, p_after))

    fpr_b, tpr_b, _ = roc_curve(y, p_before)
    fpr_a, tpr_a, _ = roc_curve(y, p_after)

    ax.plot(fpr_b, tpr_b, label=f"Before (AUROC={auc_b:.3f})")
    ax.plot(fpr_a, tpr_a, label=f"After (AUROC={auc_a:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")

    ax.set_title(f"ROC ({split_name})", fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.legend(frameon=False)
    ax.tick_params(width=style.axis_width)
    for s in ax.spines.values():
        s.set_linewidth(style.axis_width)

    _save_fig(fig, figs_dir / f"{prefix}_roc_{split_name.lower()}.svg", figs_dir / f"{prefix}_roc_{split_name.lower()}.png", dpi=style.dpi_png)

    # PR
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)
    prec_b, rec_b, _ = precision_recall_curve(y, p_before)
    prec_a, rec_a, _ = precision_recall_curve(y, p_after)

    ax.plot(rec_b, prec_b, label="Before")
    ax.plot(rec_a, prec_a, label="After")

    ax.set_title(f"Precision–Recall ({split_name})", fontweight="bold")
    ax.set_xlabel("Recall", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.legend(frameon=False)
    ax.tick_params(width=style.axis_width)
    for s in ax.spines.values():
        s.set_linewidth(style.axis_width)

    _save_fig(fig, figs_dir / f"{prefix}_pr_{split_name.lower()}.svg", figs_dir / f"{prefix}_pr_{split_name.lower()}.png", dpi=style.dpi_png)

def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        if y.size == 0 or len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def plot_roc_val_test_overlay(
    y_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    figs_dir: Path,
    style: FigStyle,
    prefix: str,
    title: str,
) -> None:
    """
    Single ROC plot overlaying VAL and TEST curves, with AUROC values in legend.
    """
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)

    try:
        fpr_v, tpr_v, _ = roc_curve(y_val, p_val)
        auc_v = _safe_auc(y_val, p_val)
        ax.plot(fpr_v, tpr_v, label=f"VAL (AUROC={auc_v:.3f})" if np.isfinite(auc_v) else "VAL (AUROC=NA)")

        fpr_t, tpr_t, _ = roc_curve(y_test, p_test)
        auc_t = _safe_auc(y_test, p_test)
        ax.plot(fpr_t, tpr_t, linestyle=":", label=f"TEST (AUROC={auc_t:.3f})" if np.isfinite(auc_t) else "TEST (AUROC=NA)")

        ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontweight="bold")
        ax.legend(frameon=False)
    except Exception:
        ax.text(0.05, 0.5, "ROC not defined", fontweight="bold")
        ax.set_axis_off()

    ax.tick_params(width=style.axis_width)
    for s in ax.spines.values():
        s.set_linewidth(style.axis_width)

    _save_fig(fig, figs_dir / f"{prefix}.svg", figs_dir / f"{prefix}.png", dpi=style.dpi_png)


def plot_confusion_matrix(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    figs_dir: Path,
    style: FigStyle,
    prefix: str,
    title: str,
) -> None:
    yp = (p >= threshold).astype(int)
    cm = confusion_matrix(y, yp, labels=[0, 1])
    fig = plt.figure(figsize=(6.2, 5.6))
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Observed", fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for s in ax.spines.values():
        s.set_linewidth(style.axis_width)
    _save_fig(fig, figs_dir / f"{prefix}_confusion.svg", figs_dir / f"{prefix}_confusion.png", dpi=style.dpi_png)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="VAL-only probability calibration for QSAR classification probabilities.")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model_key", type=str, choices=["gnn", "gnn_fusion"], required=True)
    ap.add_argument("--method", type=str, choices=["platt", "isotonic"], default="platt")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--threshold_metric", type=str, choices=["mcc"], default="mcc")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    dirs = ensure_dirs(out_root)

    style = FigStyle()
    style.apply()

    # Load VAL and TEST predictions
    df_val = _load_pred(out_root, args.model_key, args.seed, "val")
    df_test = _load_pred(out_root, args.model_key, args.seed, "test")

    y_val, p_val_before = _filter_valid(df_val)
    y_test, p_test_before = _filter_valid(df_test)

    # Fit calibrator on VAL only
    calibrator = make_calibrator(args.method)
    calibrator.fit(y_val, p_val_before)

    # Calibrate VAL and TEST
    p_val_after = calibrator.predict(p_val_before)
    p_test_after = calibrator.predict(p_test_before)

    # Threshold lock on CALIBRATED VAL (for threshold-based metrics)
    thr_after = select_threshold_by_mcc(y_val, p_val_after)
    thr_before = select_threshold_by_mcc(y_val, p_val_before)

    # Metrics (VAL and TEST, before/after)
    rows = []
    rows.append({
        "seed": args.seed, "split": "val", "model_key": args.model_key, "calibration": "before",
        **compute_cls_metrics(y_val, p_val_before, thr_before),
    })
    rows.append({
        "seed": args.seed, "split": "val", "model_key": args.model_key, "calibration": f"after_{args.method}",
        **compute_cls_metrics(y_val, p_val_after, thr_after),
    })
    rows.append({
        "seed": args.seed, "split": "test", "model_key": args.model_key, "calibration": "before",
        **compute_cls_metrics(y_test, p_test_before, thr_before),
    })
    rows.append({
        "seed": args.seed, "split": "test", "model_key": args.model_key, "calibration": f"after_{args.method}",
        **compute_cls_metrics(y_test, p_test_after, thr_after),
    })

    metrics_df = pd.DataFrame(rows)
    metrics_csv = dirs["metrics"] / f"{args.model_key}_calibrated_{args.method}_metrics_seed{args.seed:03d}.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    metrics_json = dirs["metrics"] / f"{args.model_key}_calibrated_{args.method}_metrics_seed{args.seed:03d}.json"
    metrics_json.write_text(json.dumps({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": args.seed,
        "model_key": args.model_key,
        "method": args.method,
        "fit_split": "val",
        "threshold_before_val_mcc": float(thr_before),
        "threshold_after_val_mcc": float(thr_after),
        "metrics": rows,
    }, indent=2), encoding="utf-8")

    # Save calibrated predictions (keep original df rows, calibrate only where labels exist)
    def _write_calibrated(df_in: pd.DataFrame, split: str, method: str) -> Path:
        df = df_in.copy()
        p = df["y_cls_prob"].to_numpy(dtype=float)
        # apply calibrator for all rows (even unlabeled), but keep numeric stability
        p = np.clip(p, 0.0, 1.0)
        df[f"y_cls_prob_calibrated_{method}"] = calibrator.predict(p)
        # also add prediction using val-locked calibrated threshold
        df[f"y_cls_pred_calibrated_{method}_thr_val_mcc"] = (df[f"y_cls_prob_calibrated_{method}"] >= thr_after).astype(int)
        out_path = dirs["predictions"] / f"{args.model_key}_seed{args.seed:03d}_{split}_calibrated_{method}.csv"
        df.to_csv(out_path, index=False)
        return out_path

    out_val_csv = _write_calibrated(df_val, "val", args.method)
    out_test_csv = _write_calibrated(df_test, "test", args.method)

    # Figures
    prefix = f"40_calibration_{args.model_key}_seed{args.seed:03d}_{args.method}"

    plot_reliability_before_after(
        y_val=y_val,
        p_val_before=p_val_before,
        p_val_after=p_val_after,
        figs_dir=dirs["figures"],
        style=style,
        prefix=prefix,
        bins=int(args.bins),
    )
    plot_prob_hist_before_after(
        p_before=p_val_before,
        p_after=p_val_after,
        figs_dir=dirs["figures"],
        style=style,
        prefix=prefix,
    )
    plot_roc_pr_before_after(
        y=y_val,
        p_before=p_val_before,
        p_after=p_val_after,
        figs_dir=dirs["figures"],
        style=style,
        prefix=prefix,
        split_name="VAL",
    )
    plot_roc_pr_before_after(
        y=y_test,
        p_before=p_test_before,
        p_after=p_test_after,
        figs_dir=dirs["figures"],
        style=style,
        prefix=prefix,
        split_name="TEST",
    )

    # Extra ROC overlays: VAL vs TEST in one plot, with AUROC values in legend (before and after)
    plot_roc_val_test_overlay(
        y_val=y_val,
        p_val=p_val_before,
        y_test=y_test,
        p_test=p_test_before,
        figs_dir=dirs["figures"],
        style=style,
        prefix=f"{prefix}_roc_val_test_before",
        title="ROC (VAL vs TEST): Before Calibration",
    )
    plot_roc_val_test_overlay(
        y_val=y_val,
        p_val=p_val_after,
        y_test=y_test,
        p_test=p_test_after,
        figs_dir=dirs["figures"],
        style=style,
        prefix=f"{prefix}_roc_val_test_after",
        title=f"ROC (VAL vs TEST): After {args.method}",
    )

    plot_confusion_matrix(
        y=y_val,
        p=p_val_before,
        threshold=thr_before,
        figs_dir=dirs["figures"],
        style=style,
        prefix=f"{prefix}_val_before",
        title=f"VAL Confusion (Before) thr={thr_before:.3f}",
    )
    plot_confusion_matrix(
        y=y_val,
        p=p_val_after,
        threshold=thr_after,
        figs_dir=dirs["figures"],
        style=style,
        prefix=f"{prefix}_val_after",
        title=f"VAL Confusion (After {args.method}) thr={thr_after:.3f}",
    )
    plot_confusion_matrix(
        y=y_test,
        p=p_test_before,
        threshold=thr_before,
        figs_dir=dirs["figures"],
        style=style,
        prefix=f"{prefix}_test_before",
        title=f"TEST Confusion (Before) thr={thr_before:.3f}",
    )
    plot_confusion_matrix(
        y=y_test,
        p=p_test_after,
        threshold=thr_after,
        figs_dir=dirs["figures"],
        style=style,
        prefix=f"{prefix}_test_after",
        title=f"TEST Confusion (After {args.method}) thr={thr_after:.3f}",
    )

    # Manifest
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": int(args.seed),
        "model_key": args.model_key,
        "method": args.method,
        "fit_split": "val_only",
        "inputs": {
            "val_predictions": str((out_root / "predictions").resolve() / (out_val_csv.name.replace(f"_calibrated_{args.method}.csv", ".csv"))),
            "test_predictions": str((out_root / "predictions").resolve() / (out_test_csv.name.replace(f"_calibrated_{args.method}.csv", ".csv"))),
        },
        "outputs": {
            "val_calibrated_csv": str(out_val_csv),
            "test_calibrated_csv": str(out_test_csv),
            "metrics_csv": str(metrics_csv),
            "metrics_json": str(metrics_json),
            "figures_dir": str(dirs["figures"]),
        },
        "thresholds": {
            "val_mcc_before": float(thr_before),
            "val_mcc_after": float(thr_after),
        },
    }
    manifest_path = dirs["reports"] / f"run_manifest_calibration_{args.model_key}_seed{args.seed:03d}_{args.method}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== DONE: VAL-only probability calibration ===")
    print(f"Model:                 {args.model_key}")
    print(f"Seed:                  {args.seed}")
    print(f"Method:                {args.method}")
    print(f"Calibrated VAL CSV:    {out_val_csv}")
    print(f"Calibrated TEST CSV:   {out_test_csv}")
    print(f"Metrics CSV:           {metrics_csv}")
    print(f"Figures dir (SVG):     {dirs['figures']}")
    print(f"Manifest:              {manifest_path}")
    print("============================================\n")


if __name__ == "__main__":
    main()
