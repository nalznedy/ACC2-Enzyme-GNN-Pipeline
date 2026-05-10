#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 00_data_audit_and_curation.py

Purpose:
    Audit and curate ACC2 bioactivity records before model development.

Workflow step:
    Step 0 - data audit and curation.

Main inputs:
    - Input CSV with canonical_smiles
    - Optional labels: pIC50 and/or Active

Main outputs:
    - data/01_raw_copy.csv
    - data/02_clean_valid.csv
    - data/03_dedup.csv
    - reports/data_audit.json
    - reports/structure_failures.csv
    - figures/01_pic50_hist.svg|png
    - figures/02_active_ratio.svg|png
    - figures/03_property_distributions.svg|png

Notes:
    Supports pIC50 regression and activity classification labels.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# RDKit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Matplotlib (no seaborn)
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")


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
    data_dir = out_root / "data"
    reports_dir = out_root / "reports"
    figs_dir = out_root / "figures"
    logs_dir = out_root / "logs"
    for d in [data_dir, reports_dir, figs_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {"data": data_dir, "reports": reports_dir, "figures": figs_dir, "logs": logs_dir}


# -----------------------------
# RDKit utilities
# -----------------------------
def smiles_to_mol(smiles: str) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """
    Returns (mol, error_message). Sanitizes molecules.
    """
    if smiles is None or not isinstance(smiles, str) or smiles.strip() == "":
        return None, "empty_or_non_string_smiles"
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return None, "rdkit_molfromsmiles_returned_none"
        return mol, None
    except Exception as e:
        return None, f"rdkit_exception:{type(e).__name__}"


def canonicalize_smiles(mol: Chem.Mol) -> str:
    """
    Isomeric canonical SMILES (retains stereochem where defined).
    """
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def compute_basic_rdkit_props(mol: Chem.Mol) -> Dict[str, float]:
    """
    Minimal additional sanity features (not used as training labels here),
    useful for QA and plots if original columns are missing.
    """
    return {
        "RDKit_MolWt": float(rdMolDescriptors.CalcExactMolWt(mol)),
        "RDKit_RingCount": float(rdMolDescriptors.CalcNumRings(mol)),
        "RDKit_HeavyAtomCount": float(mol.GetNumHeavyAtoms()),
    }


# -----------------------------
# Curation core
# -----------------------------
def validate_columns(df: pd.DataFrame, require_pic50: bool, require_active: bool) -> None:
    required = ["canonical_smiles"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Required at least: {required}")

    if require_pic50 and "pIC50" not in df.columns:
        raise ValueError("require_pic50=True but column 'pIC50' is missing.")
    if require_active and "Active" not in df.columns:
        raise ValueError("require_active=True but column 'Active' is missing.")


def deduplicate(
    df: pd.DataFrame,
    label_mode: str,
    agg_pic50: str,
    keep_first_meta: bool = True
) -> pd.DataFrame:
    """
    Deduplicate by canonical_smiles_canon.
    For pIC50, aggregation is configurable:
      - "max": best potency
      - "median": robust central tendency
      - "mean": average

    For Active, aggregation uses max (if any record is active -> active).
    """
    group_key = "canonical_smiles_canon"
    df = df.copy()

    # keep counts for transparency
    df["_n_records_smiles"] = df.groupby(group_key)[group_key].transform("count")

    agg_dict = {}

    if label_mode in ("regression", "both") and "pIC50" in df.columns:
        if agg_pic50 not in ("max", "median", "mean"):
            raise ValueError("agg_pic50 must be one of: max, median, mean")

        if agg_pic50 == "max":
            agg_dict["pIC50"] = "max"
        elif agg_pic50 == "median":
            agg_dict["pIC50"] = "median"
        else:
            agg_dict["pIC50"] = "mean"

    if label_mode in ("classification", "both") and "Active" in df.columns:
        # conservative rule: if any entry active => active
        agg_dict["Active"] = "max"

    # For properties: take first (or median if numeric) to preserve a representative record
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # avoid re-aggregating labels already defined above
    numeric_cols = [c for c in numeric_cols if c not in agg_dict and c != "_n_records_smiles"]

    for c in numeric_cols:
        agg_dict[c] = "median"  # robust

    # keep some metadata cols (first)
    meta_cols = [c for c in df.columns if c not in agg_dict and c not in [group_key]]
    if keep_first_meta:
        for c in meta_cols:
            agg_dict[c] = "first"

    out = df.groupby(group_key, as_index=False).agg(agg_dict)

    # rename back for clarity
    out = out.rename(columns={group_key: "canonical_smiles"})
    # keep record count (median is okay because it is identical inside group)
    out["n_records_smiles"] = out["_n_records_smiles"].astype(int)
    out = out.drop(columns=["_n_records_smiles"], errors="ignore")

    return out


# -----------------------------
# Figures (SVG + PNG)
# -----------------------------
def _save_fig(fig: plt.Figure, out_svg: Path, out_png: Path, dpi: int) -> None:
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pic50_hist(df: pd.DataFrame, figs_dir: Path, style: FigStyle) -> None:
    if "pIC50" not in df.columns:
        return
    x = df["pIC50"].dropna().astype(float).values
    if x.size == 0:
        return

    fig = plt.figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)

    color = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])[0]
    ax.hist(x, bins=30, color=color, edgecolor="black", linewidth=0.8)
    ax.set_title("pIC50 Distribution", fontweight="bold")
    ax.set_xlabel("pIC50", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")

    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(
        fig,
        figs_dir / "01_pic50_hist.svg",
        figs_dir / "01_pic50_hist.png",
        dpi=style.dpi_png
    )


def plot_active_ratio(df: pd.DataFrame, figs_dir: Path, style: FigStyle) -> None:
    if "Active" not in df.columns:
        return

    s = df["Active"].dropna().astype(int)
    if s.empty:
        return

    counts = s.value_counts().to_dict()
    inactive = int(counts.get(0, 0))
    active = int(counts.get(1, 0))

    fig = plt.figure(figsize=(6.8, 5.5))
    ax = fig.add_subplot(111)

    colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
    c1 = colors[0] if len(colors) > 0 else "C0"
    c2 = colors[1] if len(colors) > 1 else "C1"
    ax.bar(["Inactive (0)", "Active (1)"], [inactive, active], color=[c1, c2], edgecolor="black", linewidth=0.8)
    ax.set_title("Class Balance (Active)", fontweight="bold")
    ax.set_xlabel("Class", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.tick_params(width=style.axis_width)

    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(
        fig,
        figs_dir / "02_active_ratio.svg",
        figs_dir / "02_active_ratio.png",
        dpi=style.dpi_png
    )


def plot_property_distributions(df: pd.DataFrame, figs_dir: Path, style: FigStyle) -> None:
    # Prefer your existing columns if present
    preferred = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotB", "Rings", "Ro5_Violations"]
    cols = [c for c in preferred if c in df.columns]

    # If none exist, try RDKit computed minimal props (if present)
    if not cols:
        fallback = ["RDKit_MolWt", "RDKit_RingCount", "RDKit_HeavyAtomCount"]
        cols = [c for c in fallback if c in df.columns]

    if not cols:
        return

    # Plot up to 6 per figure for readability
    cols = cols[:6]

    # Publication-friendly x-axis labels (name + unit)
    xlabels = {
        "MolWt": "Molecular weight (g/mol)",
        "LogP": "LogP",
        "TPSA": "Topological polar surface area (Å²)",
        "HBD": "H-bond donors (count)",
        "HBA": "H-bond acceptors (count)",
        "RotB": "Rotatable bonds (count)",
        "Rings": "Ring count (count)",
        "Ro5_Violations": "Ro5 violations (count)",
        "RDKit_MolWt": "Exact molecular weight (g/mol)",
        "RDKit_RingCount": "Ring count (count)",
        "RDKit_HeavyAtomCount": "Heavy atom count (count)",
    }

    colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])

    n = len(cols)
    fig = plt.figure(figsize=(12, 6))
    for i, c in enumerate(cols, 1):
        ax = fig.add_subplot(2, 3, i)
        x = pd.to_numeric(df[c], errors="coerce").dropna().values
        if x.size == 0:
            ax.set_axis_off()
            continue

        color = colors[(i - 1) % len(colors)]
        ax.hist(x, bins=25, color=color, edgecolor="black", linewidth=0.7)
        ax.set_title(c, fontweight="bold")
        ax.set_xlabel(xlabels.get(c, c), fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
        ax.tick_params(width=style.axis_width)
        for spine in ax.spines.values():
            spine.set_linewidth(style.axis_width)

    fig.suptitle("Property Distributions", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    _save_fig(
        fig,
        figs_dir / "03_property_distributions.svg",
        figs_dir / "03_property_distributions.png",
        dpi=style.dpi_png
    )



# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Production data audit + curation for GNN-QSAR.")
    ap.add_argument("--input_csv", type=str, required=True, help="Path to input CSV.")
    ap.add_argument("--out_root", type=str, default="qsar_gnn_project", help="Output root directory.")
    ap.add_argument("--label_mode", type=str, choices=["regression", "classification", "both"], default="both",
                    help="Which label(s) you plan to model.")
    ap.add_argument("--agg_pic50", type=str, choices=["max", "median", "mean"], default="max",
                    help="How to aggregate pIC50 for duplicate SMILES.")
    ap.add_argument("--require_pic50", action="store_true", help="Fail if pIC50 is missing.")
    ap.add_argument("--require_active", action="store_true", help="Fail if Active is missing.")
    args = ap.parse_args()

    in_path = Path(args.input_csv).resolve()
    out_root = Path(args.out_root).resolve()
    dirs = ensure_dirs(out_root)

    style = FigStyle()
    style.apply()

    # Load
    df = pd.read_csv(in_path)
    validate_columns(df, require_pic50=args.require_pic50, require_active=args.require_active)

    # Save immutable raw copy
    raw_copy_path = dirs["data"] / "01_raw_copy.csv"
    df.to_csv(raw_copy_path, index=False)

    # Validate structures
    failures: List[Dict[str, str]] = []
    mols: List[Optional[Chem.Mol]] = []
    canon_smiles: List[Optional[str]] = []
    rdkit_props_list: List[Dict[str, float]] = []

    for idx, smi in enumerate(df["canonical_smiles"].astype(str).tolist()):
        mol, err = smiles_to_mol(smi)
        if mol is None:
            failures.append({"row_index": str(idx), "canonical_smiles": str(smi), "reason": str(err)})
            mols.append(None)
            canon_smiles.append(None)
            rdkit_props_list.append({})
        else:
            mols.append(mol)
            canon = canonicalize_smiles(mol)
            canon_smiles.append(canon)
            rdkit_props_list.append(compute_basic_rdkit_props(mol))

    df["canonical_smiles_canon"] = canon_smiles

    # Attach minimal rdkit props if user columns are missing (helpful QA)
    rdkit_props_df = pd.DataFrame(rdkit_props_list)
    for c in rdkit_props_df.columns:
        if c not in df.columns:
            df[c] = rdkit_props_df[c]

    valid_df = df[df["canonical_smiles_canon"].notna()].copy()

    # Log failures
    fail_path = dirs["reports"] / "structure_failures.csv"
    pd.DataFrame(failures).to_csv(fail_path, index=False)

    # Clean valid dataset
    clean_path = dirs["data"] / "02_clean_valid.csv"
    valid_df.to_csv(clean_path, index=False)

    # Deduplicate
    dedup_df = deduplicate(valid_df, label_mode=args.label_mode, agg_pic50=args.agg_pic50, keep_first_meta=True)
    dedup_path = dirs["data"] / "03_dedup.csv"
    dedup_df.to_csv(dedup_path, index=False)

    # Audit report
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input_csv": str(in_path),
        "out_root": str(out_root),
        "n_input_rows": int(df.shape[0]),
        "n_valid_rows": int(valid_df.shape[0]),
        "n_invalid_rows": int(df.shape[0] - valid_df.shape[0]),
        "n_unique_canonical_smiles": int(valid_df["canonical_smiles_canon"].nunique()),
        "n_dedup_rows": int(dedup_df.shape[0]),
        "label_mode": args.label_mode,
        "agg_pic50": args.agg_pic50,
        "columns_present": list(df.columns),
    }
    with open(dirs["reports"] / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Figures generated from deduplicated dataset (recommended for downstream ML)
    plot_pic50_hist(dedup_df, dirs["figures"], style)
    plot_active_ratio(dedup_df, dirs["figures"], style)
    plot_property_distributions(dedup_df, dirs["figures"], style)

    print("\n=== DONE: Data audit + curation ===")
    print(f"Raw copy:        {raw_copy_path}")
    print(f"Valid clean:     {clean_path}")
    print(f"Deduplicated:    {dedup_path}")
    print(f"Failures log:    {fail_path}")
    print(f"Audit report:    {dirs['reports'] / 'data_audit.json'}")
    print(f"Figures (SVG):   {dirs['figures']}")
    print("==================================\n")


if __name__ == "__main__":
    main()
