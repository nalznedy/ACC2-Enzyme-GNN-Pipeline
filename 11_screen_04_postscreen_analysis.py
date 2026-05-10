#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_screen_04_postscreen_analysis.py

Post-screening analysis (Script-11): generate a paper-ready analysis package from Script-10 outputs.

This script reads the outputs produced by:
  Script-10: QSAR_GNN_Project/screen/output/
    - screening_scores_full.csv
    - screening_topN.csv
    Script: 11_screen_04_postscreen_analysis.py

    Purpose:
        Summarize ranked screening outputs using scaffold diversity, uncertainty, and property analyses.

    Workflow step:
        Step 11 - post-screening analysis and report generation.

    Main inputs:
        - screen/output/screen_scores_full.csv
        - screen/output/screen_topN.csv
        - Optional training SMILES CSV for similarity analysis

    Main outputs:
        - screen/analysis/tables/*.csv
        - screen/analysis/figures/*.svg|png
        - screen/analysis/report/screening_analysis_summary.json
        - screen/analysis/report/README.txt

    Notes:
        Includes optional similarity-to-training analysis and scaffold-level diversity summaries.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

# RDKit (required for scaffold + properties + similarity)
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold


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


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# RDKit helpers
# -----------------------------
def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    if smi is None or not isinstance(smi, str) or smi.strip() == "":
        return None
    try:
        return Chem.MolFromSmiles(smi, sanitize=True)
    except Exception:
        return None


def bemis_murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return ""
        return Chem.MolToSmiles(scaf, isomericSmiles=False)
    except Exception:
        return ""


def morgan_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> DataStructs.ExplicitBitVect:
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def calc_basic_props(mol: Chem.Mol) -> Dict[str, float]:
    # Keep a tight, standard set
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Descriptors.MolLogP(mol)),
        "HBD": float(rdMolDescriptors.CalcNumHBD(mol)),
        "HBA": float(rdMolDescriptors.CalcNumHBA(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "RotB": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "Rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "HeavyAtoms": float(mol.GetNumHeavyAtoms()),
    }


# -----------------------------
# Plotting
# -----------------------------
def plot_hist(
    x: np.ndarray,
    title: str,
    xlabel: str,
    out_prefix: Path,
    style: FigStyle,
    bins: int = 50,
    alpha: float = 0.85,
) -> None:
    fig = plt.figure(figsize=(7.4, 5.6))
    ax = fig.add_subplot(111)
    ax.hist(x, bins=bins, alpha=alpha)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for sp in ax.spines.values():
        sp.set_linewidth(style.axis_width)
    _save_fig(fig, out_prefix.with_suffix(".svg"), out_prefix.with_suffix(".png"), dpi=style.dpi_png)


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_prefix: Path,
    style: FigStyle,
    alpha: float = 0.35,
) -> None:
    fig = plt.figure(figsize=(7.4, 6.2))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, alpha=alpha)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for sp in ax.spines.values():
        sp.set_linewidth(style.axis_width)
    _save_fig(fig, out_prefix.with_suffix(".svg"), out_prefix.with_suffix(".png"), dpi=style.dpi_png)


def plot_top_bar(
    values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_prefix: Path,
    style: FigStyle,
    k: int = 25,
) -> None:
    k = int(min(k, values.size))
    fig = plt.figure(figsize=(10.5, 6.0))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(k), values[:k])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for sp in ax.spines.values():
        sp.set_linewidth(style.axis_width)
    _save_fig(fig, out_prefix.with_suffix(".svg"), out_prefix.with_suffix(".png"), dpi=style.dpi_png)


def plot_scaffold_sizes(scaf_counts: pd.Series, out_prefix: Path, style: FigStyle) -> None:
    fig = plt.figure(figsize=(7.4, 5.6))
    ax = fig.add_subplot(111)
    ax.hist(scaf_counts.values.astype(float), bins=40, alpha=0.85)
    ax.set_title("Scaffold size distribution (TopN)", fontweight="bold")
    ax.set_xlabel("Compounds per scaffold", fontweight="bold")
    ax.set_ylabel("Count of scaffolds", fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for sp in ax.spines.values():
        sp.set_linewidth(style.axis_width)
    _save_fig(fig, out_prefix.with_suffix(".svg"), out_prefix.with_suffix(".png"), dpi=style.dpi_png)


# -----------------------------
# Core analysis
# -----------------------------
def bin_by_quantiles(values: np.ndarray, q: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      edges: quantile edges
      bin_id: integer bin assignment (0..len(edges)-2)
    """
    edges = np.quantile(values, q)
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([values.min(), values.max() + 1e-12], dtype=float)
    bin_id = np.digitize(values, edges[1:-1], right=False)
    return edges, bin_id


def compute_similarity_to_train(
    hit_smiles: List[str],
    train_smiles: List[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    # Build training fps
    train_fps = []
    for s in train_smiles:
        m = mol_from_smiles(s)
        if m is None:
            continue
        train_fps.append(morgan_fp(m, radius=radius, n_bits=n_bits))
    if len(train_fps) == 0:
        return np.full((len(hit_smiles),), np.nan, dtype=float)

    out = np.full((len(hit_smiles),), np.nan, dtype=float)
    for i, s in enumerate(hit_smiles):
        m = mol_from_smiles(s)
        if m is None:
            continue
        fp = morgan_fp(m, radius=radius, n_bits=n_bits)
        # max similarity to train
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        out[i] = float(np.max(sims)) if len(sims) else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-screening analysis on Script-10 outputs (diversity, uncertainty, properties).")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--task", type=str, choices=["classification", "regression", "both"], default="classification")

    ap.add_argument("--topN", type=int, default=5000)

    # Script-10 output columns
    ap.add_argument("--score_col", type=str, default="ens_prob_mean")
    ap.add_argument("--unc_col", type=str, default="ens_prob_std")
    ap.add_argument("--smiles_col", type=str, default="canonical_smiles")
    ap.add_argument("--id_col", type=str, default="screen_id")
    ap.add_argument("--name_col", type=str, default="screen_name")

    # Filtering knobs
    ap.add_argument("--high_score_quantile", type=float, default=0.99, help="Keep hits above this quantile of score among TopN.")
    ap.add_argument("--low_unc_quantile", type=float, default=0.25, help="Keep hits below this quantile of uncertainty among TopN.")
    ap.add_argument("--representatives_per_scaffold", type=int, default=1)

    # Similarity-to-train (optional)
    ap.add_argument("--train_smiles_csv", type=str, default=None, help="Optional: training set CSV for similarity analysis.")
    ap.add_argument("--train_smiles_col", type=str, default="canonical_smiles")
    ap.add_argument("--fp_radius", type=int, default=2)
    ap.add_argument("--fp_bits", type=int, default=2048)

    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()

    # Inputs from Script-10
    screen_out_dir = out_root / "screen" / "output"
    full_csv = screen_out_dir / "screening_scores_full.csv"
    top_csv = screen_out_dir / "screening_topN.csv"
    summary_json = screen_out_dir / "screening_summary.json"

    if not full_csv.exists():
        raise FileNotFoundError(f"Missing Script-10 output: {full_csv}")
    if not top_csv.exists():
        raise FileNotFoundError(f"Missing Script-10 output: {top_csv}")

    # Output dirs for Script-11
    analysis_dir = out_root / "screen" / "analysis"
    tables_dir = analysis_dir / "tables"
    figs_dir = analysis_dir / "figures"
    report_dir = analysis_dir / "report"
    for d in [analysis_dir, tables_dir, figs_dir, report_dir]:
        safe_mkdir(d)

    style = FigStyle()
    style.apply()

    df_full = pd.read_csv(full_csv)
    df_top = pd.read_csv(top_csv)

    # Defensive checks
    for col in [args.score_col, args.smiles_col, args.id_col]:
        if col not in df_full.columns:
            raise ValueError(f"Column '{col}' not found in {full_csv.name}. Found: {list(df_full.columns)[:20]} ...")
    if args.unc_col not in df_full.columns:
        # allow running without uncertainty column
        print(f"[WARN] Uncertainty column '{args.unc_col}' not found. Uncertainty analyses will be skipped.")
    if args.name_col not in df_full.columns:
        print(f"[WARN] Name column '{args.name_col}' not found. Will proceed without names.")

    # ---- Full distribution figures (from full scored set)
    score_full = df_full[args.score_col].to_numpy(dtype=float)
    plot_hist(
        score_full,
        title=f"Screening score distribution ({args.score_col})",
        xlabel=args.score_col,
        out_prefix=figs_dir / "01_score_hist_full",
        style=style,
        bins=60,
    )

    if args.unc_col in df_full.columns:
        unc_full = df_full[args.unc_col].to_numpy(dtype=float)
        plot_hist(
            unc_full,
            title=f"Screening uncertainty distribution ({args.unc_col})",
            xlabel=args.unc_col,
            out_prefix=figs_dir / "02_uncertainty_hist_full",
            style=style,
            bins=60,
        )
        plot_scatter(
            score_full,
            unc_full,
            title="Score vs uncertainty (full screening set)",
            xlabel=args.score_col,
            ylabel=args.unc_col,
            out_prefix=figs_dir / "03_score_vs_uncertainty_full",
            style=style,
            alpha=0.25,
        )

    # ---- TopN enrichment table
    topN = int(min(args.topN, df_top.shape[0]))
    df_top = df_top.head(topN).copy()

    # Score/unc columns in top
    if args.score_col not in df_top.columns:
        # If top file lost columns, merge from full using id
        df_top = df_top.merge(
            df_full[[args.id_col, args.score_col] + ([args.unc_col] if args.unc_col in df_full.columns else [])],
            on=args.id_col,
            how="left",
        )

    # Add bins for quick reporting
    top_scores = df_top[args.score_col].to_numpy(dtype=float)
    score_edges, score_bin = bin_by_quantiles(top_scores, q=[0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0])
    df_top["score_bin"] = score_bin

    if args.unc_col in df_top.columns:
        top_unc = df_top[args.unc_col].to_numpy(dtype=float)
        unc_edges, unc_bin = bin_by_quantiles(top_unc, q=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
        df_top["uncertainty_bin"] = unc_bin

    df_top.to_csv(tables_dir / "topN_enriched.csv", index=False)

    # Bin summaries
    bin_summary = (
        df_top.groupby("score_bin", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("score_bin")
    )
    bin_summary["score_bin_edges"] = str(score_edges.tolist())
    bin_summary.to_csv(tables_dir / "score_bins_summary.csv", index=False)

    if args.unc_col in df_top.columns:
        ubin_summary = (
            df_top.groupby("uncertainty_bin", dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("uncertainty_bin")
        )
        ubin_summary["uncertainty_bin_edges"] = str(unc_edges.tolist())
        ubin_summary.to_csv(tables_dir / "uncertainty_bins_summary.csv", index=False)

    # ---- Uncertainty-filtered high-confidence subset (within TopN)
    df_conf = None
    if args.unc_col in df_top.columns:
        score_thr = float(np.quantile(top_scores, args.high_score_quantile))
        unc_thr = float(np.quantile(df_top[args.unc_col].to_numpy(dtype=float), args.low_unc_quantile))
        df_conf = df_top[(df_top[args.score_col] >= score_thr) & (df_top[args.unc_col] <= unc_thr)].copy()
        df_conf = df_conf.sort_values([args.score_col, args.unc_col], ascending=[False, True]).reset_index(drop=True)
        df_conf.to_csv(tables_dir / "topN_uncertainty_filtered.csv", index=False)

    # ---- TopN figures
    df_top_sorted = df_top.sort_values(args.score_col, ascending=False).reset_index(drop=True)
    plot_top_bar(
        df_top_sorted[args.score_col].to_numpy(dtype=float),
        title=f"Top hits by {args.score_col} (TopN={topN})",
        xlabel="Rank",
        ylabel=args.score_col,
        out_prefix=figs_dir / "10_topN_score_bar",
        style=style,
        k=25,
    )
    if args.unc_col in df_top_sorted.columns:
        plot_top_bar(
            df_top_sorted[args.unc_col].to_numpy(dtype=float),
            title=f"Top hits uncertainty ({args.unc_col}) (TopN={topN})",
            xlabel="Rank",
            ylabel=args.unc_col,
            out_prefix=figs_dir / "11_topN_uncertainty_bar",
            style=style,
            k=25,
        )

    # ---- Scaffold clustering for diversity
    scaffolds = []
    mol_ok = []
    for s in df_top_sorted[args.smiles_col].astype(str).tolist():
        m = mol_from_smiles(s)
        if m is None:
            scaffolds.append("")
            mol_ok.append(False)
        else:
            scaffolds.append(bemis_murcko_scaffold_smiles(m))
            mol_ok.append(True)
    df_top_sorted["murcko_scaffold"] = scaffolds
    df_top_sorted["rdkit_mol_ok"] = mol_ok

    # scaffold groups
    scaf_counts = df_top_sorted[df_top_sorted["murcko_scaffold"] != ""].groupby("murcko_scaffold").size().sort_values(ascending=False)
    scaf_counts.to_frame("count").reset_index().rename(columns={"index": "murcko_scaffold"}).to_csv(
        tables_dir / "topN_scaffold_clusters.csv", index=False
    )
    if scaf_counts.size > 0:
        plot_scaffold_sizes(scaf_counts, figs_dir / "12_scaffold_size_distribution", style=style)

    # representatives (one or more per scaffold)
    reps = []
    reps_per = int(max(1, args.representatives_per_scaffold))
    for scaf, g in df_top_sorted[df_top_sorted["murcko_scaffold"] != ""].groupby("murcko_scaffold", sort=False):
        g2 = g.sort_values(args.score_col, ascending=False).head(reps_per)
        reps.append(g2)
    if len(reps) > 0:
        df_reps = pd.concat(reps, ignore_index=True)
        df_reps = df_reps.sort_values(args.score_col, ascending=False).reset_index(drop=True)
    else:
        df_reps = df_top_sorted.head(min(200, df_top_sorted.shape[0])).copy()
    df_reps.to_csv(tables_dir / "topN_scaffold_representatives.csv", index=False)

    # ---- Chemical properties (TopN)
    props_rows = []
    for idx, row in df_top_sorted.iterrows():
        smi = str(row[args.smiles_col])
        m = mol_from_smiles(smi)
        if m is None:
            continue
        pr = calc_basic_props(mol=m)
        pr[args.id_col] = row.get(args.id_col, "")
        pr[args.name_col] = row.get(args.name_col, "") if args.name_col in df_top_sorted.columns else ""
        pr[args.smiles_col] = smi
        pr["score"] = float(row[args.score_col])
        if args.unc_col in df_top_sorted.columns:
            pr["uncertainty"] = float(row[args.unc_col])
        props_rows.append(pr)
        if idx >= topN - 1:
            break

    if len(props_rows) > 0:
        df_props = pd.DataFrame(props_rows)
        df_props.to_csv(tables_dir / "chem_properties_topN.csv", index=False)

    # ---- Optional: similarity to training set
    sim_done = False
    if args.train_smiles_csv is not None:
        train_csv = Path(args.train_smiles_csv).resolve()
        if not train_csv.exists():
            raise FileNotFoundError(f"train_smiles_csv not found: {train_csv}")
        train_df = pd.read_csv(train_csv)
        if args.train_smiles_col not in train_df.columns:
            raise ValueError(f"train_smiles_col '{args.train_smiles_col}' not in {train_csv.name}")
        train_smiles = train_df[args.train_smiles_col].astype(str).tolist()

        hit_smiles = df_top_sorted[args.smiles_col].astype(str).tolist()
        max_sim = compute_similarity_to_train(
            hit_smiles=hit_smiles,
            train_smiles=train_smiles,
            radius=int(args.fp_radius),
            n_bits=int(args.fp_bits),
        )
        df_sim = df_top_sorted[[args.id_col, args.smiles_col, args.score_col] + ([args.unc_col] if args.unc_col in df_top_sorted.columns else [])].copy()
        df_sim["max_tanimoto_to_train"] = max_sim
        df_sim.to_csv(tables_dir / "topN_similarity_to_train.csv", index=False)

        # figures
        ms = df_sim["max_tanimoto_to_train"].to_numpy(dtype=float)
        ms = ms[~np.isnan(ms)]
        if ms.size > 0:
            plot_hist(
                ms,
                title="Max Tanimoto similarity to training set (TopN)",
                xlabel="Max Tanimoto (Morgan)",
                out_prefix=figs_dir / "20_similarity_to_train_hist",
                style=style,
                bins=50,
            )
            plot_scatter(
                df_sim[args.score_col].to_numpy(dtype=float),
                df_sim["max_tanimoto_to_train"].to_numpy(dtype=float),
                title="Similarity to training set vs screening score (TopN)",
                xlabel=args.score_col,
                ylabel="Max Tanimoto (Morgan)",
                out_prefix=figs_dir / "21_similarity_to_train_vs_score",
                style=style,
                alpha=0.35,
            )
        sim_done = True

    # ---- Summary JSON + README
    extra = {}
    if summary_json.exists():
        try:
            extra["screening_summary_json"] = json.loads(summary_json.read_text(encoding="utf-8"))
        except Exception:
            extra["screening_summary_json"] = "failed_to_parse"

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "screening_scores_full_csv": str(full_csv),
            "screening_topN_csv": str(top_csv),
            "screening_summary_json": str(summary_json) if summary_json.exists() else None,
        },
        "columns": {
            "id_col": args.id_col,
            "name_col": args.name_col,
            "smiles_col": args.smiles_col,
            "score_col": args.score_col,
            "unc_col": args.unc_col if args.unc_col in df_full.columns else None,
        },
        "topN": topN,
        "filters": {
            "high_score_quantile_within_topN": float(args.high_score_quantile),
            "low_unc_quantile_within_topN": float(args.low_unc_quantile),
        },
        "outputs": {
            "analysis_dir": str(analysis_dir),
            "tables_dir": str(tables_dir),
            "figures_dir": str(figs_dir),
            "report_dir": str(report_dir),
        },
        "counts": {
            "n_scored_full": int(df_full.shape[0]),
            "n_topN": int(df_top.shape[0]),
            "n_confident_subset": int(df_conf.shape[0]) if df_conf is not None else None,
            "n_unique_scaffolds_topN": int(scaf_counts.shape[0]) if scaf_counts is not None else 0,
        },
        "optional_similarity_to_train": {
            "enabled": bool(sim_done),
            "train_smiles_csv": str(Path(args.train_smiles_csv).resolve()) if sim_done else None,
            "train_smiles_col": args.train_smiles_col if sim_done else None,
            "fp_radius": int(args.fp_radius),
            "fp_bits": int(args.fp_bits),
        },
        "extra": extra,
    }
    (report_dir / "screening_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme_lines = [
        "Post-screening analysis (Script-11)\n",
        f"Generated: {summary['timestamp']}\n",
        "\nInputs (from Script-10):\n",
        f"  - {full_csv}\n",
        f"  - {top_csv}\n",
        "\nKey outputs:\n",
        f"  - tables/: enriched TopN tables, uncertainty-filtered subset, scaffold clusters, representatives, properties\n",
        f"  - figures/: score/uncertainty distributions, top bars, scaffold distribution\n",
        f"  - report/screening_analysis_summary.json\n",
        "\nInterpretation tips:\n",
        "  - Prefer hits with high score and low uncertainty (ensemble std).\n",
        "  - Use scaffold representatives for diverse experimental testing.\n",
        "  - If similarity-to-train is enabled, very high similarity can indicate analogs; lower similarity indicates novelty.\n",
    ]
    (report_dir / "README.txt").write_text("".join(readme_lines), encoding="utf-8")

    print("\n=== DONE: Post-screening analysis package created ===")
    print(f"Analysis dir: {analysis_dir}")
    print(f"Tables dir:   {tables_dir}")
    print(f"Figures dir:  {figs_dir}")
    print(f"Report dir:   {report_dir}")
    print("===============================================\n")


if __name__ == "__main__":
    main()
