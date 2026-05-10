#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 01_scaffold_split_generation.py

Purpose:
        Generate seed-controlled scaffold train/validation/test splits.

Workflow step:
        Step 1 - Bemis-Murcko scaffold split generation.

Main inputs:
        - data/03_dedup.csv

Main outputs:
        - splits/scaffold_seed###.csv
        - reports/scaffold_split_stats_seed###.json
        - tables/scaffold_stats_seed###.csv
        - figures/scaffold_distribution_*.svg|png

Notes:
        Scaffold grouping is used to evaluate chemical-series generalization.
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

# RDKit
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Matplotlib only (no seaborn)
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
    splits_dir = out_root / "splits"
    reports_dir = out_root / "reports"
    tables_dir = out_root / "tables"
    figs_dir = out_root / "figures"
    logs_dir = out_root / "logs"
    for d in [splits_dir, reports_dir, tables_dir, figs_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {"splits": splits_dir, "reports": reports_dir, "tables": tables_dir, "figures": figs_dir, "logs": logs_dir}


def _save_fig(fig: plt.Figure, out_svg: Path, out_png: Path, dpi: int) -> None:
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# RDKit scaffold helpers
# -----------------------------
def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    if smi is None or not isinstance(smi, str) or smi.strip() == "":
        return None
    try:
        m = Chem.MolFromSmiles(smi, sanitize=True)
        return m
    except Exception:
        return None


def bemis_murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    """
    Returns canonical scaffold SMILES. If scaffold extraction fails, returns empty string.
    """
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return ""
        return Chem.MolToSmiles(scaf, isomericSmiles=True, canonical=True)
    except Exception:
        return ""


# -----------------------------
# Split algorithm (scaffold groups)
# -----------------------------
def build_scaffold_groups(df: pd.DataFrame, smiles_col: str = "canonical_smiles") -> pd.DataFrame:
    """
    Adds a 'scaffold' column. Any failure produces scaffold="" and is logged externally if needed.
    """
    scaffolds: List[str] = []
    failures = 0
    for smi in df[smiles_col].astype(str).tolist():
        mol = mol_from_smiles(smi)
        if mol is None:
            scaffolds.append("")
            failures += 1
        else:
            scaffolds.append(bemis_murcko_scaffold_smiles(mol))
    out = df.copy()
    out["scaffold"] = scaffolds
    out["scaffold_failed"] = (out["scaffold"].astype(str).str.len() == 0).astype(int)
    return out


def scaffold_split(
    df: pd.DataFrame,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    smiles_col: str = "canonical_smiles",
) -> pd.DataFrame:
    """
    Deterministic scaffold split:
      1) Group molecules by scaffold.
      2) Shuffle scaffold groups with seed.
      3) Assign whole scaffold groups to train/val/test by cumulative size.

    Returns a copy with added columns: split, seed, scaffold_size
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    df2 = df.copy()

    # Ensure scaffold exists
    if "scaffold" not in df2.columns:
        df2 = build_scaffold_groups(df2, smiles_col=smiles_col)

    # Filter out scaffold failures? For production robustness:
    # Keep them, but treat each molecule as its own scaffold by using its canonical_smiles as fallback.
    # This prevents silent drops and preserves reproducibility.
    bad = df2["scaffold"].astype(str).str.len() == 0
    df2.loc[bad, "scaffold"] = df2.loc[bad, smiles_col].astype(str)

    # Group indices by scaffold
    scaffold_to_indices: Dict[str, List[int]] = {}
    for idx, scaf in enumerate(df2["scaffold"].astype(str).tolist()):
        scaffold_to_indices.setdefault(scaf, []).append(idx)

    # Create a list of (scaffold, size)
    scaffold_items = [(s, len(idxs)) for s, idxs in scaffold_to_indices.items()]

    # Shuffle scaffolds deterministically
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffold_items)

    # Sort by size descending AFTER shuffle for stability? Two approaches exist.
    # For better balance, sort by size descending but break ties by shuffled order:
    # We'll implement: shuffle -> stable sort by size descending.
    scaffold_items = sorted(scaffold_items, key=lambda x: x[1], reverse=True)

    n_total = df2.shape[0]
    n_train_target = int(round(train_frac * n_total))
    n_val_target = int(round(val_frac * n_total))
    # test gets the remainder
    n_test_target = n_total - n_train_target - n_val_target

    split_assign = np.array([""] * n_total, dtype=object)

    n_train = n_val = n_test = 0
    for scaf, size in scaffold_items:
        idxs = scaffold_to_indices[scaf]

        # Greedy assignment to match targets
        if n_train < n_train_target:
            split = "train"
            n_train += size
        elif n_val < n_val_target:
            split = "val"
            n_val += size
        else:
            split = "test"
            n_test += size

        split_assign[idxs] = split

    df2["split"] = split_assign
    df2["seed"] = int(seed)
    df2["scaffold_size"] = df2["scaffold"].map(lambda s: len(scaffold_to_indices[str(s)])).astype(int)

    # Sanity
    if (df2["split"] == "").any():
        raise RuntimeError("Split assignment failed: some rows have empty split.")

    return df2


# -----------------------------
# Stats and tables
# -----------------------------
def split_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {
        "train": int((df["split"] == "train").sum()),
        "val": int((df["split"] == "val").sum()),
        "test": int((df["split"] == "test").sum()),
    }


def compute_label_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute label distribution per split for both regression and classification if present.
    """
    out: Dict[str, Dict[str, float]] = {}
    for sp in ["train", "val", "test"]:
        sub = df[df["split"] == sp].copy()
        stats: Dict[str, float] = {}
        if "pIC50" in sub.columns:
            x = pd.to_numeric(sub["pIC50"], errors="coerce").dropna()
            if not x.empty:
                stats.update({
                    "pIC50_n": float(x.shape[0]),
                    "pIC50_mean": float(x.mean()),
                    "pIC50_std": float(x.std(ddof=1)) if x.shape[0] > 1 else 0.0,
                    "pIC50_min": float(x.min()),
                    "pIC50_max": float(x.max()),
                })
        if "Active" in sub.columns:
            y = pd.to_numeric(sub["Active"], errors="coerce").dropna()
            if not y.empty:
                y = y.astype(int)
                stats.update({
                    "Active_n": float(y.shape[0]),
                    "Active_pos": float((y == 1).sum()),
                    "Active_neg": float((y == 0).sum()),
                    "Active_pos_frac": float((y == 1).mean()),
                })
        out[sp] = stats
    return out


def compute_scaffold_stats(df: pd.DataFrame) -> Dict[str, object]:
    """
    Scaffold-level diagnostics for generalization reporting.
    """
    # Unique scaffolds per split
    scaf_counts = {}
    for sp in ["train", "val", "test"]:
        scaf_counts[f"{sp}_unique_scaffolds"] = int(df.loc[df["split"] == sp, "scaffold"].nunique())

    # How many test scaffolds also appear in train? (should be 0 in pure scaffold split)
    train_scafs = set(df.loc[df["split"] == "train", "scaffold"].astype(str).tolist())
    test_scafs = set(df.loc[df["split"] == "test", "scaffold"].astype(str).tolist())
    overlap = train_scafs.intersection(test_scafs)

    # Scaffold size distribution summary
    scaf_sizes = df.groupby("scaffold")["canonical_smiles"].count().values
    scaf_sizes = np.asarray(scaf_sizes, dtype=float)

    scaf_size_summary = {
        "n_scaffolds_total": int(df["scaffold"].nunique()),
        "scaffold_size_mean": float(scaf_sizes.mean()) if scaf_sizes.size else 0.0,
        "scaffold_size_median": float(np.median(scaf_sizes)) if scaf_sizes.size else 0.0,
        "scaffold_size_max": float(scaf_sizes.max()) if scaf_sizes.size else 0.0,
        "scaffold_size_p95": float(np.percentile(scaf_sizes, 95)) if scaf_sizes.size else 0.0,
    }

    return {
        **scaf_counts,
        "train_test_scaffold_overlap_count": int(len(overlap)),
        "train_test_scaffold_overlap_fraction": float(len(overlap) / max(1, len(test_scafs))),
        "scaffold_size_summary": scaf_size_summary,
    }


# -----------------------------
# Figures
# -----------------------------
def plot_split_counts_bar(df: pd.DataFrame, figs_dir: Path, style: FigStyle, prefix: str) -> None:
    c = split_counts(df)
    fig = plt.figure(figsize=(7.0, 5.5))
    ax = fig.add_subplot(111)
    ax.bar(["Train", "Validation", "Test"], [c["train"], c["val"], c["test"]])
    ax.set_title("Scaffold Split Counts", fontweight="bold")
    ax.set_ylabel("Number of Compounds", fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(
        fig,
        figs_dir / f"{prefix}_split_counts.svg",
        figs_dir / f"{prefix}_split_counts.png",
        dpi=style.dpi_png
    )


def plot_scaffold_size_hist(df: pd.DataFrame, figs_dir: Path, style: FigStyle, prefix: str) -> None:
    # scaffold sizes
    sizes = df.groupby("scaffold")["canonical_smiles"].count().values.astype(float)
    if sizes.size == 0:
        return

    fig = plt.figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)
    ax.hist(sizes, bins=30)
    ax.set_title("Scaffold Size Distribution", fontweight="bold")
    ax.set_xlabel("Number of Compounds per Scaffold", fontweight="bold")
    ax.set_ylabel("Number of Scaffolds", fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(
        fig,
        figs_dir / f"{prefix}_scaffold_size_hist.svg",
        figs_dir / f"{prefix}_scaffold_size_hist.png",
        dpi=style.dpi_png
    )


def plot_pic50_by_split(df: pd.DataFrame, figs_dir: Path, style: FigStyle, prefix: str) -> None:
    if "pIC50" not in df.columns:
        return
    fig = plt.figure(figsize=(8.2, 5.6))
    ax = fig.add_subplot(111)

    for sp in ["train", "val", "test"]:
        x = pd.to_numeric(df.loc[df["split"] == sp, "pIC50"], errors="coerce").dropna().values
        if x.size == 0:
            continue
        ax.hist(x, bins=25, alpha=0.35, label=sp.capitalize())

    ax.set_title("pIC50 Distribution by Split", fontweight="bold")
    ax.set_xlabel("pIC50", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.legend(frameon=False)
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(
        fig,
        figs_dir / f"{prefix}_pic50_by_split.svg",
        figs_dir / f"{prefix}_pic50_by_split.png",
        dpi=style.dpi_png
    )


def plot_active_ratio_by_split(df: pd.DataFrame, figs_dir: Path, style: FigStyle, prefix: str) -> None:
    if "Active" not in df.columns:
        return
    fig = plt.figure(figsize=(7.8, 5.6))
    ax = fig.add_subplot(111)

    splits = ["train", "val", "test"]
    pos_fracs = []
    for sp in splits:
        y = pd.to_numeric(df.loc[df["split"] == sp, "Active"], errors="coerce").dropna()
        if y.empty:
            pos_fracs.append(np.nan)
        else:
            y = y.astype(int)
            pos_fracs.append(float((y == 1).mean()))

    ax.bar(["Train", "Validation", "Test"], pos_fracs)
    ax.set_title("Active Positive Fraction by Split", fontweight="bold")
    ax.set_ylabel("Positive Fraction (Active=1)", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(
        fig,
        figs_dir / f"{prefix}_active_pos_frac_by_split.svg",
        figs_dir / f"{prefix}_active_pos_frac_by_split.png",
        dpi=style.dpi_png
    )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Production scaffold split generator (Bemis–Murcko).")
    ap.add_argument("--input_csv", type=str, required=True, help="Path to deduplicated CSV (03_dedup.csv).")
    ap.add_argument("--out_root", type=str, required=True, help="Project root directory (same as script-1 out_root).")
    ap.add_argument("--n_seeds", type=int, default=5, help="Number of seeds to generate.")
    ap.add_argument("--base_seed", type=int, default=2025, help="Base seed; actual seeds = base_seed + i.")
    ap.add_argument("--train_frac", type=float, default=0.8, help="Train fraction.")
    ap.add_argument("--val_frac", type=float, default=0.1, help="Validation fraction.")
    ap.add_argument("--test_frac", type=float, default=0.1, help="Test fraction.")
    ap.add_argument("--smiles_col", type=str, default="canonical_smiles", help="SMILES column name.")
    ap.add_argument("--write_split_datasets", action="store_true",
                    help="If set, export per-seed train/val/test CSV datasets (heavier but useful).")
    ap.add_argument("--max_scaffold_table", type=int, default=5000,
                    help="Max number of scaffolds to save in scaffold table (for very large datasets).")
    args = ap.parse_args()

    in_path = Path(args.input_csv).resolve()
    out_root = Path(args.out_root).resolve()
    dirs = ensure_dirs(out_root)

    style = FigStyle()
    style.apply()

    # Load
    df = pd.read_csv(in_path)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Missing smiles_col '{args.smiles_col}' in {in_path}. Found columns: {list(df.columns)}")

    # Compute scaffolds once (reused across seeds)
    df_scaf = build_scaffold_groups(df, smiles_col=args.smiles_col)

    # Global scaffold table (all scaffolds and sizes)
    scaffold_table = (
        df_scaf.groupby("scaffold")[args.smiles_col]
        .count()
        .reset_index()
        .rename(columns={args.smiles_col: "scaffold_size"})
        .sort_values("scaffold_size", ascending=False)
    )
    scaffold_table_limited = scaffold_table.head(int(args.max_scaffold_table)).copy()
    scaffold_table_path = dirs["tables"] / "scaffold_table_global.csv"
    scaffold_table_limited.to_csv(scaffold_table_path, index=False)

    # Global report
    global_report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input_csv": str(in_path),
        "out_root": str(out_root),
        "n_rows": int(df_scaf.shape[0]),
        "n_scaffolds_total": int(df_scaf["scaffold"].nunique()),
        "scaffold_failed_rows": int(df_scaf["scaffold_failed"].sum()),
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "n_seeds": int(args.n_seeds),
        "base_seed": int(args.base_seed),
        "smiles_col": args.smiles_col,
    }
    with open(dirs["reports"] / "scaffold_global_report.json", "w", encoding="utf-8") as f:
        json.dump(global_report, f, indent=2)

    # Seed-wise summary table
    seed_rows = []

    # Generate per-seed splits
    for i in range(args.n_seeds):
        seed = int(args.base_seed + i)
        df_split = scaffold_split(
            df_scaf,
            seed=seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            smiles_col=args.smiles_col,
        )

        # Write assignment CSV (light and immutable)
        assign_cols = ["seed", args.smiles_col, "scaffold", "scaffold_size", "split"]
        # Keep labels if present for convenience, but assignment file remains a “manifest”
        for extra in ["pIC50", "Active", "n_records_smiles", "molecule_chembl_id"]:
            if extra in df_split.columns and extra not in assign_cols:
                assign_cols.append(extra)

        assign_path = dirs["splits"] / f"scaffold_seed{seed:03d}.csv"
        df_split[assign_cols].to_csv(assign_path, index=False)

        # Optionally write per-seed datasets (train/val/test CSVs)
        if args.write_split_datasets:
            for sp in ["train", "val", "test"]:
                sp_path = dirs["splits"] / f"scaffold_seed{seed:03d}_{sp}.csv"
                df_split[df_split["split"] == sp].to_csv(sp_path, index=False)

        # Reports and tables
        counts = split_counts(df_split)
        label_stats = compute_label_stats(df_split)
        scaf_stats = compute_scaffold_stats(df_split)

        seed_report = {
            "seed": seed,
            "counts": counts,
            "label_stats": label_stats,
            "scaffold_stats": scaf_stats,
        }
        with open(dirs["reports"] / f"scaffold_seed{seed:03d}_report.json", "w", encoding="utf-8") as f:
            json.dump(seed_report, f, indent=2)

        # Add row for seed summary table
        row = {
            "seed": seed,
            "n_train": counts["train"],
            "n_val": counts["val"],
            "n_test": counts["test"],
            "train_unique_scaffolds": scaf_stats["train_unique_scaffolds"],
            "val_unique_scaffolds": scaf_stats["val_unique_scaffolds"],
            "test_unique_scaffolds": scaf_stats["test_unique_scaffolds"],
            "train_test_scaffold_overlap_count": scaf_stats["train_test_scaffold_overlap_count"],
            "train_test_scaffold_overlap_fraction": scaf_stats["train_test_scaffold_overlap_fraction"],
            "n_scaffolds_total": scaf_stats["scaffold_size_summary"]["n_scaffolds_total"],
            "scaffold_size_mean": scaf_stats["scaffold_size_summary"]["scaffold_size_mean"],
            "scaffold_size_median": scaf_stats["scaffold_size_summary"]["scaffold_size_median"],
            "scaffold_size_max": scaf_stats["scaffold_size_summary"]["scaffold_size_max"],
            "scaffold_size_p95": scaf_stats["scaffold_size_summary"]["scaffold_size_p95"],
        }

        # Add pIC50/Active summary if present
        if "pIC50" in df_split.columns:
            for sp in ["train", "val", "test"]:
                if f"pIC50_mean_{sp}" not in row:
                    row[f"pIC50_mean_{sp}"] = label_stats.get(sp, {}).get("pIC50_mean", np.nan)
                    row[f"pIC50_std_{sp}"] = label_stats.get(sp, {}).get("pIC50_std", np.nan)

        if "Active" in df_split.columns:
            for sp in ["train", "val", "test"]:
                row[f"active_pos_frac_{sp}"] = label_stats.get(sp, {}).get("Active_pos_frac", np.nan)

        seed_rows.append(row)

        # Figures per seed (use prefix to prevent overwriting)
        prefix = f"04_seed{seed:03d}"
        plot_split_counts_bar(df_split, dirs["figures"], style, prefix=prefix)
        plot_scaffold_size_hist(df_split, dirs["figures"], style, prefix=prefix)
        plot_pic50_by_split(df_split, dirs["figures"], style, prefix=prefix)
        plot_active_ratio_by_split(df_split, dirs["figures"], style, prefix=prefix)

    # Save seed summary table
    seed_summary_df = pd.DataFrame(seed_rows)
    seed_summary_path = dirs["tables"] / "scaffold_seed_summary.csv"
    seed_summary_df.to_csv(seed_summary_path, index=False)

    # Save mean ± SD summary (numeric columns)
    numeric_cols = seed_summary_df.select_dtypes(include=[np.number]).columns.tolist()
    summary = []
    for c in numeric_cols:
        if c == "seed":
            continue
        vals = pd.to_numeric(seed_summary_df[c], errors="coerce").dropna()
        if vals.empty:
            continue
        summary.append({
            "metric": c,
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.shape[0] > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
        })
    summary_df = pd.DataFrame(summary)
    summary_path = dirs["tables"] / "scaffold_seed_summary_mean_std.csv"
    summary_df.to_csv(summary_path, index=False)

    # Global figure: distribution of split sizes across seeds
    # (use train/val/test counts from seed_summary_df)
    fig = plt.figure(figsize=(8.2, 5.8))
    ax = fig.add_subplot(111)
    x = np.arange(seed_summary_df.shape[0])
    ax.plot(x, seed_summary_df["n_train"].values, marker="o", label="Train")
    ax.plot(x, seed_summary_df["n_val"].values, marker="o", label="Validation")
    ax.plot(x, seed_summary_df["n_test"].values, marker="o", label="Test")
    ax.set_title("Split Size Stability Across Seeds", fontweight="bold")
    ax.set_xlabel("Seed Index", fontweight="bold")
    ax.set_ylabel("Number of Compounds", fontweight="bold")
    ax.legend(frameon=False)
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(
        fig,
        dirs["figures"] / "05_split_size_stability_across_seeds.svg",
        dirs["figures"] / "05_split_size_stability_across_seeds.png",
        dpi=style.dpi_png
    )

    # Final run manifest
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input_csv": str(in_path),
        "out_root": str(out_root),
        "split_files": [str(p) for p in sorted(dirs["splits"].glob("scaffold_seed*.csv"))],
        "reports": [str(p) for p in sorted(dirs["reports"].glob("scaffold_seed*_report.json"))],
        "tables": [str(seed_summary_path), str(summary_path), str(scaffold_table_path)],
        "figures": [str(p) for p in sorted(dirs["figures"].glob("04_seed*.svg"))] + [
            str(dirs["figures"] / "05_split_size_stability_across_seeds.svg")
        ],
    }
    with open(dirs["reports"] / "run_manifest_scaffold_splits.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== DONE: Scaffold split generation ===")
    print(f"Input:            {in_path}")
    print(f"Scaffold table:   {scaffold_table_path}")
    print(f"Split files:      {dirs['splits']}")
    print(f"Reports:          {dirs['reports']}")
    print(f"Tables:           {dirs['tables']}")
    print(f"Figures (SVG):    {dirs['figures']}")
    print(f"Manifest:         {dirs['reports'] / 'run_manifest_scaffold_splits.json'}")
    print("======================================\n")


if __name__ == "__main__":
    main()
