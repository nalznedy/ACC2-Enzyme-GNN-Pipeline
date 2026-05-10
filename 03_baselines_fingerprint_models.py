#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 03_baselines_fingerprint_models.py

Purpose:
        Train Morgan fingerprint baselines for ACC2 pIC50 regression and activity classification.

Workflow step:
        Step 3 - baseline machine-learning models.

Main inputs:
        - data/03_dedup.csv
        - splits/scaffold_seed###.csv

Main outputs:
        - features/morgan_seed###.npz
        - models/baseline_*_seed###_*.pkl
        - predictions/baseline_*_seed###_*.csv
        - metrics/baseline_*_seed###.*
        - figures/baseline_*.svg|png

Notes:
        Classification threshold selection is performed on validation predictions only.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# fallback gradient boosting
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

# Matplotlib only
import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------
# Publication figure styling
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


def ensure_dirs(out_root: Path) -> Dict[str, Path]:
    features_dir = out_root / "features"
    models_dir = out_root / "models"
    preds_dir = out_root / "predictions"
    metrics_dir = out_root / "metrics"
    reports_dir = out_root / "reports"
    tables_dir = out_root / "tables"
    figs_dir = out_root / "figures"
    logs_dir = out_root / "logs"
    for d in [features_dir, models_dir, preds_dir, metrics_dir, reports_dir, tables_dir, figs_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {
        "features": features_dir,
        "models": models_dir,
        "predictions": preds_dir,
        "metrics": metrics_dir,
        "reports": reports_dir,
        "tables": tables_dir,
        "figures": figs_dir,
        "logs": logs_dir,
    }


def _save_fig(fig: plt.Figure, out_svg: Path, out_png: Path, dpi: int) -> None:
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Fingerprints
# -----------------------------
def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    if smi is None or not isinstance(smi, str) or smi.strip() == "":
        return None
    try:
        m = Chem.MolFromSmiles(smi, sanitize=True)
        return m
    except Exception:
        return None


def morgan_fp_array(smiles_list: List[str], radius: int = 2, n_bits: int = 2048) -> Tuple[np.ndarray, List[int]]:
    """
    Returns (X, bad_indices) where X is uint8 matrix shape [N, n_bits].
    """
    X = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    bad = []
    for i, smi in enumerate(smiles_list):
        mol = mol_from_smiles(smi)
        if mol is None:
            bad.append(i)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i, :] = arr.astype(np.uint8)
    return X, bad


# -----------------------------
# Metrics
# -----------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    # Pearson/Spearman without scipy: compute Pearson via np.corrcoef, Spearman via rank transform
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else float("nan")
    ranks_true = pd.Series(y_true).rank(method="average").values
    ranks_pred = pd.Series(y_pred).rank(method="average").values
    spearman = float(np.corrcoef(ranks_true, ranks_pred)[0, 1]) if y_true.size > 1 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson_r": pearson, "spearman_rho": spearman}


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    out = {}
    # Some splits can have only one class; handle safely
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["pr_auc"] = float("nan")

    out["mcc"] = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    out["bal_acc"] = float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    # confusion matrix components
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    # brier (calibration)
    try:
        out["brier"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        out["brier"] = float("nan")
    return out


def select_threshold_by_mcc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Select threshold maximizing MCC on validation set.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    thresholds = np.unique(np.clip(y_prob, 0, 1))
    # also add a standard grid to avoid edge cases
    grid = np.linspace(0.0, 1.0, 501)
    thresholds = np.unique(np.concatenate([thresholds, grid]))

    best_t = 0.5
    best_mcc = -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        # mcc defined even if one class predicted, but y_true must have both classes to be meaningful
        if len(np.unique(y_true)) < 2:
            continue
        mcc = matthews_corrcoef(y_true, pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_t = float(t)
    return float(best_t)


# -----------------------------
# Figures
# -----------------------------
def plot_pred_vs_obs(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_svg: Path, out_png: Path, style: FigStyle) -> None:
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)

    colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    c_scatter = colors[0] if len(colors) > 0 else "C0"
    c_line = colors[1] if len(colors) > 1 else "C1"

    ax.scatter(y_true, y_pred, alpha=0.70, color=c_scatter, edgecolors="black", linewidths=0.4)
    mn = float(np.min([np.min(y_true), np.min(y_pred)]))
    mx = float(np.max([np.max(y_true), np.max(y_pred)]))
    ax.plot([mn, mx], [mn, mx], color=c_line, linestyle="--")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Observed", fontweight="bold")
    ax.set_ylabel("Predicted", fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(fig, out_svg, out_png, dpi=style.dpi_png)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_svg: Path, out_png: Path, style: FigStyle) -> None:
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_subplot(111)

    colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
    c_hist = colors[0] if len(colors) > 0 else "C0"
    c_zero = colors[1] if len(colors) > 1 else "C1"

    ax.hist(resid, bins=30, color=c_hist, edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.axvline(0.0, color=c_zero, linestyle="--")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Residual (Observed - Predicted)", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(fig, out_svg, out_png, dpi=style.dpi_png)


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, prefix: str, figs_dir: Path, style: FigStyle) -> None:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    c_curve = colors[0] if len(colors) > 0 else "C0"
    c_chance = colors[1] if len(colors) > 1 else "C1"
    c_pr = colors[2] if len(colors) > 2 else "C2"

    # ----------------
    # ROC
    # ----------------
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=c_curve, label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color=c_chance, label="Chance")
        ax.set_title("ROC Curve", fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontweight="bold")
        ax.legend(frameon=False)
    except Exception:
        ax.text(0.05, 0.5, "ROC not defined (single-class y_true)", fontweight="bold")
        ax.set_axis_off()

    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(
        fig,
        figs_dir / f"{prefix}_roc.svg",
        figs_dir / f"{prefix}_roc.png",
        dpi=style.dpi_png
    )

    # ----------------
    # PR
    # ----------------
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, color=c_pr, label=f"PR (AP = {ap:.3f})")
        ax.set_title("Precision–Recall Curve", fontweight="bold")
        ax.set_xlabel("Recall", fontweight="bold")
        ax.set_ylabel("Precision", fontweight="bold")
        ax.legend(frameon=False)
    except Exception:
        ax.text(0.05, 0.5, "PR not defined (single-class y_true)", fontweight="bold")
        ax.set_axis_off()

    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(
        fig,
        figs_dir / f"{prefix}_pr.svg",
        figs_dir / f"{prefix}_pr.png",
        dpi=style.dpi_png
    )

def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, title: str, out_svg: Path, out_png: Path, style: FigStyle) -> None:
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)

    colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
    c_model = colors[0] if len(colors) > 0 else "C0"
    c_ideal = colors[1] if len(colors) > 1 else "C1"

    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", color=c_model, label="Model")
        ax.plot([0, 1], [0, 1], linestyle="--", color=c_ideal, label="Ideal")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
        ax.set_ylabel("Fraction of Positives", fontweight="bold")
        ax.legend(frameon=False)
    except Exception:
        ax.text(0.05, 0.5, "Calibration not defined (single-class y_true)", fontweight="bold")
        ax.set_axis_off()
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(fig, out_svg, out_png, dpi=style.dpi_png)


def plot_confusion(cm: np.ndarray, title: str, out_svg: Path, out_png: Path, style: FigStyle) -> None:
    fig = plt.figure(figsize=(6.2, 5.6))
    ax = fig.add_subplot(111)

    colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    c_text = colors[0] if len(colors) > 0 else "C0"

    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Observed", fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontweight="bold", color=c_text)

    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(fig, out_svg, out_png, dpi=style.dpi_png)

    # Annotate
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontweight="bold")

    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(fig, out_svg, out_png, dpi=style.dpi_png)


# -----------------------------
# Model builders
# -----------------------------
def build_rf_reg(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=600,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )


def build_rf_cls(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=800,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
    )


def build_hgb_reg(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        random_state=seed,
        max_depth=None,
        learning_rate=0.05,
        max_iter=800,
        l2_regularization=1e-3,
    )


def build_hgb_cls(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        random_state=seed,
        max_depth=None,
        learning_rate=0.05,
        max_iter=800,
        l2_regularization=1e-3,
    )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline QSAR models: Morgan fingerprints + RF/GBDT.")
    ap.add_argument("--dedup_csv", type=str, required=True, help="Script-1 output: data/03_dedup.csv")
    ap.add_argument("--split_csv", type=str, required=True, help="Script-2 split manifest: splits/scaffold_seed###.csv")
    ap.add_argument("--out_root", type=str, required=True, help="Project root directory.")
    ap.add_argument("--task", type=str, choices=["regression", "classification", "both"], default="both")
    ap.add_argument("--smiles_col", type=str, default="canonical_smiles")
    ap.add_argument("--reg_col", type=str, default="pIC50")
    ap.add_argument("--cls_col", type=str, default="Active")
    ap.add_argument("--fp_radius", type=int, default=2)
    ap.add_argument("--fp_bits", type=int, default=2048)
    ap.add_argument("--fail_on_missing_labels", action="store_true")
    ap.add_argument("--save_train_predictions", action="store_true",
                    help="If set, saves train predictions (large but useful for diagnostics).")
    args = ap.parse_args()

    style = FigStyle()
    style.apply()

    dedup_path = Path(args.dedup_csv).resolve()
    split_path = Path(args.split_csv).resolve()
    out_root = Path(args.out_root).resolve()
    dirs = ensure_dirs(out_root)

    # Load
    df = pd.read_csv(dedup_path)
    split_df = pd.read_csv(split_path)

    for c in [args.smiles_col]:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' in dedup CSV: {dedup_path}")
        if c not in split_df.columns:
            raise ValueError(f"Missing '{c}' in split CSV: {split_path}")
    if "split" not in split_df.columns:
        raise ValueError("split_csv must contain 'split' column")
    if "seed" not in split_df.columns:
        split_df["seed"] = -1

    # Join
    joined = df.merge(split_df[[args.smiles_col, "split", "seed"]], on=args.smiles_col, how="inner", validate="one_to_one")

    # Coverage check
    if joined.shape[0] != df.shape[0]:
        miss = df[~df[args.smiles_col].isin(set(joined[args.smiles_col].tolist()))][[args.smiles_col]].copy()
        miss_path = dirs["reports"] / "baseline_missing_from_split.csv"
        miss.to_csv(miss_path, index=False)
        raise RuntimeError(f"Split manifest does not cover all dedup rows. Missing saved to {miss_path}")

    seed_vals = sorted(set(joined["seed"].astype(int).tolist()))
    if len(seed_vals) != 1:
        raise RuntimeError(f"Expected a single seed in split file; found {seed_vals}")
    seed = int(seed_vals[0])

    # Label checks
    if args.task in ("regression", "both"):
        if args.reg_col not in joined.columns and args.fail_on_missing_labels:
            raise ValueError(f"Regression requires '{args.reg_col}' column.")
    if args.task in ("classification", "both"):
        if args.cls_col not in joined.columns and args.fail_on_missing_labels:
            raise ValueError(f"Classification requires '{args.cls_col}' column.")

    # Build fingerprints
    smiles = joined[args.smiles_col].astype(str).tolist()
    X, bad_idx = morgan_fp_array(smiles, radius=args.fp_radius, n_bits=args.fp_bits)
    if len(bad_idx) > 0:
        bad_rows = joined.iloc[bad_idx][[args.smiles_col, "split", "seed"]].copy()
        bad_path = dirs["reports"] / f"baseline_fp_failures_seed{seed:03d}.csv"
        bad_rows.to_csv(bad_path, index=False)
        raise RuntimeError(f"Fingerprint generation failed for {len(bad_idx)} molecules. See {bad_path}")

    # Save FP cache
    fp_cache_path = dirs["features"] / f"morgan_seed{seed:03d}_r{args.fp_radius}_b{args.fp_bits}.npz"
    np.savez_compressed(
        fp_cache_path,
        X=X,
        smiles=np.array(smiles, dtype=object),
        split=np.array(joined["split"].astype(str).tolist(), dtype=object),
        seed=np.array(joined["seed"].astype(int).tolist(), dtype=int),
    )

    # Split indices
    idx_train = np.where(joined["split"].values == "train")[0]
    idx_val = np.where(joined["split"].values == "val")[0]
    idx_test = np.where(joined["split"].values == "test")[0]

    # Summary collectors
    metrics_records = []

    # Helper to save predictions
    def save_preds(model_name: str, split_name: str, ids: np.ndarray, y_true: Optional[np.ndarray], y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Path:
        out = pd.DataFrame({
            "seed": seed,
            "split": split_name,
            args.smiles_col: joined.iloc[ids][args.smiles_col].values,
        })
        if "molecule_chembl_id" in joined.columns:
            out["molecule_chembl_id"] = joined.iloc[ids]["molecule_chembl_id"].values

        if y_true is not None:
            out["y_true"] = y_true
        out["y_pred"] = y_pred
        if y_prob is not None:
            out["y_prob"] = y_prob

        p = dirs["predictions"] / f"{model_name}_seed{seed:03d}_{split_name}.csv"
        out.to_csv(p, index=False)
        return p

    # -----------------------------
    # REGRESSION BASELINES
    # -----------------------------
    if args.task in ("regression", "both") and args.reg_col in joined.columns:
        y = pd.to_numeric(joined[args.reg_col], errors="coerce").values.astype(float)

        # Remove NaN labels per split (strictly)
        def filter_valid(ids: np.ndarray) -> np.ndarray:
            return ids[~np.isnan(y[ids])]

        tr = filter_valid(idx_train)
        va = filter_valid(idx_val)
        te = filter_valid(idx_test)

        # RF regressor
        rf_reg = build_rf_reg(seed)
        rf_reg.fit(X[tr], y[tr])

        # Predict
        pred_tr = rf_reg.predict(X[tr])
        pred_va = rf_reg.predict(X[va])
        pred_te = rf_reg.predict(X[te])

        # Save model
        rf_path = dirs["models"] / f"baseline_rf_seed{seed:03d}_reg.pkl"
        with open(rf_path, "wb") as f:
            pickle.dump(rf_reg, f)

        # Save predictions
        if args.save_train_predictions:
            save_preds("baseline_rf_reg", "train", tr, y[tr], pred_tr)
        p_val = save_preds("baseline_rf_reg", "val", va, y[va], pred_va)
        p_test = save_preds("baseline_rf_reg", "test", te, y[te], pred_te)

        # Metrics
        m_tr = regression_metrics(y[tr], pred_tr)
        m_va = regression_metrics(y[va], pred_va)
        m_te = regression_metrics(y[te], pred_te)

        for split_name, m in [("train", m_tr), ("val", m_va), ("test", m_te)]:
            metrics_records.append({"seed": seed, "model": "RF", "task": "regression", "split": split_name, **m})

        # Figures (test)
        plot_pred_vs_obs(
            y[te], pred_te,
            title="RF (Morgan) Regression: Test Predicted vs Observed",
            out_svg=dirs["figures"] / f"07_rf_reg_seed{seed:03d}_pred_vs_obs_test.svg",
            out_png=dirs["figures"] / f"07_rf_reg_seed{seed:03d}_pred_vs_obs_test.png",
            style=style
        )
        plot_residuals(
            y[te], pred_te,
            title="RF (Morgan) Regression: Test Residuals",
            out_svg=dirs["figures"] / f"08_rf_reg_seed{seed:03d}_residuals_test.svg",
            out_png=dirs["figures"] / f"08_rf_reg_seed{seed:03d}_residuals_test.png",
            style=style
        )

        # HGB regressor (GBDT-like baseline)
        hgb_reg = build_hgb_reg(seed)
        hgb_reg.fit(X[tr], y[tr])

        pred_tr = hgb_reg.predict(X[tr])
        pred_va = hgb_reg.predict(X[va])
        pred_te = hgb_reg.predict(X[te])

        hgb_path = dirs["models"] / f"baseline_hgb_seed{seed:03d}_reg.pkl"
        with open(hgb_path, "wb") as f:
            pickle.dump(hgb_reg, f)

        if args.save_train_predictions:
            save_preds("baseline_hgb_reg", "train", tr, y[tr], pred_tr)
        save_preds("baseline_hgb_reg", "val", va, y[va], pred_va)
        save_preds("baseline_hgb_reg", "test", te, y[te], pred_te)

        m_tr = regression_metrics(y[tr], pred_tr)
        m_va = regression_metrics(y[va], pred_va)
        m_te = regression_metrics(y[te], pred_te)

        for split_name, m in [("train", m_tr), ("val", m_va), ("test", m_te)]:
            metrics_records.append({"seed": seed, "model": "HGB", "task": "regression", "split": split_name, **m})

        plot_pred_vs_obs(
            y[te], pred_te,
            title="HGB (Morgan) Regression: Test Predicted vs Observed",
            out_svg=dirs["figures"] / f"09_hgb_reg_seed{seed:03d}_pred_vs_obs_test.svg",
            out_png=dirs["figures"] / f"09_hgb_reg_seed{seed:03d}_pred_vs_obs_test.png",
            style=style
        )
        plot_residuals(
            y[te], pred_te,
            title="HGB (Morgan) Regression: Test Residuals",
            out_svg=dirs["figures"] / f"10_hgb_reg_seed{seed:03d}_residuals_test.svg",
            out_png=dirs["figures"] / f"10_hgb_reg_seed{seed:03d}_residuals_test.png",
            style=style
        )

    # -----------------------------
    # CLASSIFICATION BASELINES
    # -----------------------------
    if args.task in ("classification", "both") and args.cls_col in joined.columns:
        y = pd.to_numeric(joined[args.cls_col], errors="coerce").values
        # strict valid labels: 0/1
        valid_mask = np.isin(y, [0, 1])
        y = y.astype(int, copy=False)

        def filter_valid(ids: np.ndarray) -> np.ndarray:
            return ids[valid_mask[ids]]

        tr = filter_valid(idx_train)
        va = filter_valid(idx_val)
        te = filter_valid(idx_test)

        # RF classifier
        rf_cls = build_rf_cls(seed)
        rf_cls.fit(X[tr], y[tr])

        prob_tr = rf_cls.predict_proba(X[tr])[:, 1]
        prob_va = rf_cls.predict_proba(X[va])[:, 1]
        prob_te = rf_cls.predict_proba(X[te])[:, 1]

        # threshold selection on validation only
        thr = select_threshold_by_mcc(y[va], prob_va)

        rf_path = dirs["models"] / f"baseline_rf_seed{seed:03d}_cls.pkl"
        with open(rf_path, "wb") as f:
            pickle.dump(rf_cls, f)

        if args.save_train_predictions:
            save_preds("baseline_rf_cls", "train", tr, y[tr], (prob_tr >= thr).astype(int), y_prob=prob_tr)
        save_preds("baseline_rf_cls", "val", va, y[va], (prob_va >= thr).astype(int), y_prob=prob_va)
        save_preds("baseline_rf_cls", "test", te, y[te], (prob_te >= thr).astype(int), y_prob=prob_te)

        m_tr = classification_metrics(y[tr], prob_tr, thr)
        m_va = classification_metrics(y[va], prob_va, thr)
        m_te = classification_metrics(y[te], prob_te, thr)
        for split_name, m in [("train", m_tr), ("val", m_va), ("test", m_te)]:
            metrics_records.append({"seed": seed, "model": "RF", "task": "classification", "split": split_name, "threshold": thr, **m})

        # Figures on test
        plot_roc_pr(y[te], prob_te, prefix=f"11_rf_cls_seed{seed:03d}_test", figs_dir=dirs["figures"], style=style)
        plot_calibration(
            y[te], prob_te,
            title="RF (Morgan) Calibration: Test",
            out_svg=dirs["figures"] / f"12_rf_cls_seed{seed:03d}_calibration_test.svg",
            out_png=dirs["figures"] / f"12_rf_cls_seed{seed:03d}_calibration_test.png",
            style=style
        )
        cm = confusion_matrix(y[te], (prob_te >= thr).astype(int), labels=[0, 1])
        plot_confusion(
            cm,
            title=f"RF (Morgan) Confusion Matrix: Test (thr={thr:.3f})",
            out_svg=dirs["figures"] / f"13_rf_cls_seed{seed:03d}_confusion_test.svg",
            out_png=dirs["figures"] / f"13_rf_cls_seed{seed:03d}_confusion_test.png",
            style=style
        )

        # HGB classifier
        hgb_cls = build_hgb_cls(seed)
        hgb_cls.fit(X[tr], y[tr])

        # HGB has predict_proba in sklearn
        prob_tr = hgb_cls.predict_proba(X[tr])[:, 1]
        prob_va = hgb_cls.predict_proba(X[va])[:, 1]
        prob_te = hgb_cls.predict_proba(X[te])[:, 1]

        thr = select_threshold_by_mcc(y[va], prob_va)

        hgb_path = dirs["models"] / f"baseline_hgb_seed{seed:03d}_cls.pkl"
        with open(hgb_path, "wb") as f:
            pickle.dump(hgb_cls, f)

        if args.save_train_predictions:
            save_preds("baseline_hgb_cls", "train", tr, y[tr], (prob_tr >= thr).astype(int), y_prob=prob_tr)
        save_preds("baseline_hgb_cls", "val", va, y[va], (prob_va >= thr).astype(int), y_prob=prob_va)
        save_preds("baseline_hgb_cls", "test", te, y[te], (prob_te >= thr).astype(int), y_prob=prob_te)

        m_tr = classification_metrics(y[tr], prob_tr, thr)
        m_va = classification_metrics(y[va], prob_va, thr)
        m_te = classification_metrics(y[te], prob_te, thr)
        for split_name, m in [("train", m_tr), ("val", m_va), ("test", m_te)]:
            metrics_records.append({"seed": seed, "model": "HGB", "task": "classification", "split": split_name, "threshold": thr, **m})

        plot_roc_pr(y[te], prob_te, prefix=f"14_hgb_cls_seed{seed:03d}_test", figs_dir=dirs["figures"], style=style)
        plot_calibration(
            y[te], prob_te,
            title="HGB (Morgan) Calibration: Test",
            out_svg=dirs["figures"] / f"15_hgb_cls_seed{seed:03d}_calibration_test.svg",
            out_png=dirs["figures"] / f"15_hgb_cls_seed{seed:03d}_calibration_test.png",
            style=style
        )
        cm = confusion_matrix(y[te], (prob_te >= thr).astype(int), labels=[0, 1])
        plot_confusion(
            cm,
            title=f"HGB (Morgan) Confusion Matrix: Test (thr={thr:.3f})",
            out_svg=dirs["figures"] / f"16_hgb_cls_seed{seed:03d}_confusion_test.svg",
            out_png=dirs["figures"] / f"16_hgb_cls_seed{seed:03d}_confusion_test.png",
            style=style
        )

    # Save metrics
    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv = dirs["metrics"] / f"baseline_metrics_seed{seed:03d}.csv"
    metrics_json = dirs["metrics"] / f"baseline_metrics_seed{seed:03d}.json"
    metrics_df.to_csv(metrics_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "records": metrics_records}, f, indent=2)

    # Save a compact summary table for paper
    summary_rows = []
    for task in sorted(metrics_df["task"].unique()) if not metrics_df.empty else []:
        for model in sorted(metrics_df.loc[metrics_df["task"] == task, "model"].unique()):
            te = metrics_df[(metrics_df["task"] == task) & (metrics_df["model"] == model) & (metrics_df["split"] == "test")]
            if te.empty:
                continue
            r = te.iloc[0].to_dict()
            keep = {"seed": seed, "task": task, "model": model}
            if task == "regression":
                keep.update({k: r.get(k) for k in ["rmse", "mae", "r2", "pearson_r", "spearman_rho"]})
            else:
                keep.update({k: r.get(k) for k in ["auroc", "pr_auc", "mcc", "bal_acc", "f1", "brier", "threshold"]})
            summary_rows.append(keep)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = dirs["tables"] / f"baseline_summary_test_seed{seed:03d}.csv"
    summary_df.to_csv(summary_path, index=False)

    # Run manifest
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "inputs": {"dedup_csv": str(dedup_path), "split_csv": str(split_path)},
        "outputs": {
            "fp_cache": str(fp_cache_path),
            "models_dir": str(dirs["models"]),
            "predictions_dir": str(dirs["predictions"]),
            "metrics_csv": str(metrics_csv),
            "metrics_json": str(metrics_json),
            "summary_table": str(summary_path),
            "figures_dir": str(dirs["figures"]),
        },
        "fingerprint": {"radius": args.fp_radius, "bits": args.fp_bits},
    }
    manifest_path = dirs["reports"] / f"run_manifest_baselines_seed{seed:03d}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== DONE: Baseline fingerprint models ===")
    print(f"Seed:             {seed}")
    print(f"FP cache:         {fp_cache_path}")
    print(f"Models:           {dirs['models']}")
    print(f"Predictions:      {dirs['predictions']}")
    print(f"Metrics:          {metrics_csv}")
    print(f"Summary (test):   {summary_path}")
    print(f"Figures (SVG):    {dirs['figures']}")
    print(f"Manifest:         {manifest_path}")
    print("========================================\n")


if __name__ == "__main__":
    main()
