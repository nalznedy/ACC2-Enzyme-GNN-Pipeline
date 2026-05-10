#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_graph_dataset_builder.py

Production-grade molecular graph builder + cache for GNN-QSAR (atom/bond features).
This is Script-3 in the workflow.

Inputs:
  1) Deduplicated dataset from Script-1:
       QSAR_GNN_Project/data/03_dedup.csv
     Must include:
       - canonical_smiles
     Optional but recommended:
       - pIC50 (regression)
    Script: 02_graph_dataset_builder.py

    Purpose:
            Build molecular graph caches for GNN and graph-Morgan fusion modeling.

    Workflow step:
            Step 2 - graph feature construction from curated ACC2 compounds.

    Main inputs:
            - data/03_dedup.csv
            - splits/scaffold_seed###.csv

    Main outputs:
            - features/graphs_seed###.pt
            - features/graphs_seed###_index.csv
            - reports/graph_build_seed###.json
            - tables/graph_stats_seed###.csv
            - figures/graph_feature_*.svg|png

    Notes:
            Graph tensors preserve split assignments for leakage-safe training.
    --task both
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
from rdkit.Chem.rdchem import HybridizationType, BondType, ChiralType

# Torch
import torch

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
    features_dir = out_root / "features"
    reports_dir = out_root / "reports"
    tables_dir = out_root / "tables"
    figs_dir = out_root / "figures"
    logs_dir = out_root / "logs"
    for d in [features_dir, reports_dir, tables_dir, figs_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {"features": features_dir, "reports": reports_dir, "tables": tables_dir, "figures": figs_dir, "logs": logs_dir}


def _save_fig(fig: plt.Figure, out_svg: Path, out_png: Path, dpi: int) -> None:
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# One-hot helpers
# -----------------------------
def one_hot(value: int, allowable: List[int]) -> List[int]:
    return [1 if value == a else 0 for a in allowable]


def one_hot_str(value: str, allowable: List[str]) -> List[int]:
    return [1 if value == a else 0 for a in allowable]


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


# -----------------------------
# Feature encoding spec
# -----------------------------
@dataclass
class GraphFeaturizerConfig:
    max_atomic_num: int = 100
    max_degree: int = 5
    max_h: int = 4
    max_valence: int = 6
    formal_charges: Tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
    hybridizations: Tuple[HybridizationType, ...] = (
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
    )
    chiral_tags: Tuple[ChiralType, ...] = (
        ChiralType.CHI_UNSPECIFIED,
        ChiralType.CHI_TETRAHEDRAL_CW,
        ChiralType.CHI_TETRAHEDRAL_CCW,
        ChiralType.CHI_OTHER,
    )
    bond_types: Tuple[BondType, ...] = (
        BondType.SINGLE,
        BondType.DOUBLE,
        BondType.TRIPLE,
        BondType.AROMATIC,
    )
    stereo_allow: Tuple[str, ...] = ("STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "STEREOANY")

    def node_feature_dim(self) -> int:
        # atomic_num one-hot (1..max_atomic_num) + degree (0..max_degree) +
        # formal_charge + hybridization + aromatic + ring + chiral +
        # implicitH + totalValence + atomicMassScaled
        return (
            self.max_atomic_num
            + (self.max_degree + 1)
            + len(self.formal_charges)
            + len(self.hybridizations)
            + 1  # aromatic
            + 1  # ring
            + len(self.chiral_tags)
            + (self.max_h + 1)
            + (self.max_valence + 1)
            + 1  # mass scaled
        )

    def edge_feature_dim(self) -> int:
        # bond type one-hot + conjugated + in ring + aromatic bond + stereo one-hot
        return len(self.bond_types) + 1 + 1 + 1 + len(self.stereo_allow)


# -----------------------------
# RDKit graph building
# -----------------------------
def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    if smi is None or not isinstance(smi, str) or smi.strip() == "":
        return None
    try:
        m = Chem.MolFromSmiles(smi, sanitize=True)
        return m
    except Exception:
        return None


def atom_features(atom: Chem.Atom, cfg: GraphFeaturizerConfig) -> np.ndarray:
    # Atomic number: 1..max_atomic_num (unknown -> clipped)
    z = clamp_int(atom.GetAtomicNum(), 1, cfg.max_atomic_num)
    atomic_num_oh = one_hot(z, list(range(1, cfg.max_atomic_num + 1)))

    # Degree: 0..max_degree (clip)
    deg = clamp_int(atom.GetDegree(), 0, cfg.max_degree)
    degree_oh = one_hot(deg, list(range(0, cfg.max_degree + 1)))

    # Formal charge
    fc = int(atom.GetFormalCharge())
    # map to nearest allowed (clip by bounding, then exact match if possible)
    if fc not in cfg.formal_charges:
        fc = clamp_int(fc, min(cfg.formal_charges), max(cfg.formal_charges))
    formal_charge_oh = one_hot(fc, list(cfg.formal_charges))

    # Hybridization
    hyb = atom.GetHybridization()
    hyb_oh = [1 if hyb == h else 0 for h in cfg.hybridizations]

    aromatic = int(atom.GetIsAromatic())
    ring = int(atom.IsInRing())

    # Chirality tag
    ct = atom.GetChiralTag()
    chiral_oh = [1 if ct == c else 0 for c in cfg.chiral_tags]

    # Implicit H count: 0..max_h
    ih = clamp_int(atom.GetTotalNumHs(includeNeighbors=True), 0, cfg.max_h)
    ih_oh = one_hot(ih, list(range(0, cfg.max_h + 1)))

    # Total valence: 0..max_valence
    tv = clamp_int(atom.GetTotalValence(), 0, cfg.max_valence)
    tv_oh = one_hot(tv, list(range(0, cfg.max_valence + 1)))

    # Atomic mass scaled (simple scaling to keep magnitude stable)
    mass_scaled = float(atom.GetMass()) / 200.0  # typical masses << 200
    mass_feat = [mass_scaled]

    feats = (
        atomic_num_oh
        + degree_oh
        + formal_charge_oh
        + hyb_oh
        + [aromatic]
        + [ring]
        + chiral_oh
        + ih_oh
        + tv_oh
        + mass_feat
    )
    return np.asarray(feats, dtype=np.float32)


def bond_features(bond: Chem.Bond, cfg: GraphFeaturizerConfig) -> np.ndarray:
    bt = bond.GetBondType()
    bt_oh = [1 if bt == b else 0 for b in cfg.bond_types]

    conj = int(bond.GetIsConjugated())
    ring = int(bond.IsInRing())
    aromatic_bond = int(bond.GetIsAromatic())

    st = str(bond.GetStereo())  # e.g., 'STEREONONE'
    if st not in cfg.stereo_allow:
        st = "STEREOANY"
    st_oh = one_hot_str(st, list(cfg.stereo_allow))

    feats = bt_oh + [conj] + [ring] + [aromatic_bond] + st_oh
    return np.asarray(feats, dtype=np.float32)


def mol_to_graph_tensors(
    mol: Chem.Mol,
    cfg: GraphFeaturizerConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (x, edge_index, edge_attr) with bidirectional edges.
    """
    n = mol.GetNumAtoms()
    x = np.zeros((n, cfg.node_feature_dim()), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        x[i, :] = atom_features(atom, cfg)

    # Collect edges in both directions
    edge_index_list: List[List[int]] = []
    edge_attr_list: List[np.ndarray] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond, cfg)

        edge_index_list.append([i, j])
        edge_attr_list.append(bf)

        edge_index_list.append([j, i])
        edge_attr_list.append(bf)

    if len(edge_index_list) == 0:
        # Single-atom molecules: no bonds; create empty tensors
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, cfg.edge_feature_dim()), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).T.contiguous()
        edge_attr = torch.tensor(np.stack(edge_attr_list, axis=0), dtype=torch.float32)

    x_t = torch.tensor(x, dtype=torch.float32)
    return x_t, edge_index, edge_attr


# -----------------------------
# Stats and figures
# -----------------------------
def compute_graph_stats(rows: List[Dict]) -> pd.DataFrame:
    """
    rows: list of index rows with at least num_nodes, num_edges, split
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df


def plot_num_nodes_by_split(stats_df: pd.DataFrame, figs_dir: Path, style: FigStyle, prefix: str) -> None:
    if stats_df.empty:
        return

    fig = plt.figure(figsize=(8.2, 5.6))
    ax = fig.add_subplot(111)

    for sp in ["train", "val", "test"]:
        x = stats_df.loc[stats_df["split"] == sp, "num_nodes"].dropna().values.astype(float)
        if x.size == 0:
            continue
        ax.hist(x, bins=25, alpha=0.35, label=sp.capitalize())

    ax.set_title("Number of Atoms per Molecule by Split", fontweight="bold")
    ax.set_xlabel("Number of Atoms", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.legend(frameon=False)
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(
        fig,
        figs_dir / f"{prefix}_num_atoms_by_split.svg",
        figs_dir / f"{prefix}_num_atoms_by_split.png",
        dpi=style.dpi_png
    )


def plot_num_edges_by_split(stats_df: pd.DataFrame, figs_dir: Path, style: FigStyle, prefix: str) -> None:
    if stats_df.empty:
        return

    fig = plt.figure(figsize=(8.2, 5.6))
    ax = fig.add_subplot(111)

    for sp in ["train", "val", "test"]:
        x = stats_df.loc[stats_df["split"] == sp, "num_edges"].dropna().values.astype(float)
        if x.size == 0:
            continue
        ax.hist(x, bins=25, alpha=0.35, label=sp.capitalize())

    ax.set_title("Number of Directed Edges per Molecule by Split", fontweight="bold")
    ax.set_xlabel("Number of Directed Edges", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.legend(frameon=False)
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(
        fig,
        figs_dir / f"{prefix}_num_edges_by_split.svg",
        figs_dir / f"{prefix}_num_edges_by_split.png",
        dpi=style.dpi_png
    )


def plot_label_distributions(joined: pd.DataFrame, figs_dir: Path, style: FigStyle, prefix: str) -> None:
    # pIC50
    if "pIC50" in joined.columns:
        fig = plt.figure(figsize=(8.2, 5.6))
        ax = fig.add_subplot(111)
        for sp in ["train", "val", "test"]:
            x = pd.to_numeric(joined.loc[joined["split"] == sp, "pIC50"], errors="coerce").dropna().values
            if x.size == 0:
                continue
            ax.hist(x, bins=25, alpha=0.35, label=sp.capitalize())
        ax.set_title("pIC50 Distribution by Split (Joined Dataset)", fontweight="bold")
        ax.set_xlabel("pIC50", fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
        ax.legend(frameon=False)
        ax.tick_params(width=style.axis_width)
        for spine in ax.spines.values():
            spine.set_linewidth(style.axis_width)

        _save_fig(
            fig,
            figs_dir / f"{prefix}_pic50_by_split_joined.svg",
            figs_dir / f"{prefix}_pic50_by_split_joined.png",
            dpi=style.dpi_png
        )

    # Active
    if "Active" in joined.columns:
        fig = plt.figure(figsize=(7.8, 5.6))
        ax = fig.add_subplot(111)
        splits = ["train", "val", "test"]
        pos_fracs = []
        for sp in splits:
            y = pd.to_numeric(joined.loc[joined["split"] == sp, "Active"], errors="coerce").dropna()
            if y.empty:
                pos_fracs.append(np.nan)
            else:
                y = y.astype(int)
                pos_fracs.append(float((y == 1).mean()))
        ax.bar(["Train", "Validation", "Test"], pos_fracs)
        ax.set_title("Active Positive Fraction by Split (Joined Dataset)", fontweight="bold")
        ax.set_ylabel("Positive Fraction (Active=1)", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.tick_params(width=style.axis_width)
        for spine in ax.spines.values():
            spine.set_linewidth(style.axis_width)

        _save_fig(
            fig,
            figs_dir / f"{prefix}_active_pos_frac_joined.svg",
            figs_dir / f"{prefix}_active_pos_frac_joined.png",
            dpi=style.dpi_png
        )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build and cache molecular graphs for GNN-QSAR.")
    ap.add_argument("--dedup_csv", type=str, required=True, help="Path to Script-1 output: data/03_dedup.csv")
    ap.add_argument("--split_csv", type=str, required=True, help="Path to Script-2 split manifest: splits/scaffold_seed###.csv")
    ap.add_argument("--out_root", type=str, required=True, help="Project root directory.")
    ap.add_argument("--task", type=str, choices=["regression", "classification", "both"], default="both",
                    help="Which labels to package in cached graphs.")
    ap.add_argument("--smiles_col", type=str, default="canonical_smiles", help="SMILES column name.")
    ap.add_argument("--reg_col", type=str, default="pIC50", help="Regression label column.")
    ap.add_argument("--cls_col", type=str, default="Active", help="Classification label column.")
    ap.add_argument("--max_atomic_num", type=int, default=100, help="Max atomic number for one-hot (clip above).")
    ap.add_argument("--max_degree", type=int, default=5, help="Max degree for one-hot (clip above).")
    ap.add_argument("--max_h", type=int, default=4, help="Max implicit H count for one-hot (clip above).")
    ap.add_argument("--max_valence", type=int, default=6, help="Max total valence for one-hot (clip above).")
    ap.add_argument("--fail_on_missing_labels", action="store_true",
                    help="If set, will error if required labels are missing for the selected task.")
    args = ap.parse_args()

    dedup_path = Path(args.dedup_csv).resolve()
    split_path = Path(args.split_csv).resolve()
    out_root = Path(args.out_root).resolve()
    dirs = ensure_dirs(out_root)

    style = FigStyle()
    style.apply()

    # Load inputs
    dedup = pd.read_csv(dedup_path)
    split_df = pd.read_csv(split_path)

    # Validate columns
    for c in [args.smiles_col]:
        if c not in dedup.columns:
            raise ValueError(f"Missing '{c}' in dedup_csv: {dedup_path}")
        if c not in split_df.columns:
            raise ValueError(f"Missing '{c}' in split_csv: {split_path}")

    if "split" not in split_df.columns:
        raise ValueError(f"split_csv must contain 'split' column. Found: {list(split_df.columns)}")
    if "seed" not in split_df.columns:
        # Script-2 always writes seed, but we handle robustly
        split_df["seed"] = -1

    # Label presence checks
    if args.task in ("regression", "both"):
        if args.reg_col not in dedup.columns and args.fail_on_missing_labels:
            raise ValueError(f"task={args.task} requires '{args.reg_col}' but it is missing in dedup CSV.")
    if args.task in ("classification", "both"):
        if args.cls_col not in dedup.columns and args.fail_on_missing_labels:
            raise ValueError(f"task={args.task} requires '{args.cls_col}' but it is missing in dedup CSV.")

    # Join split manifest to dedup dataset by canonical_smiles (strict)
    joined = dedup.merge(
        split_df[[args.smiles_col, "split", "seed"]],
        on=args.smiles_col,
        how="inner",
        validate="one_to_one"
    )

    # Check coverage
    n_dedup = int(dedup.shape[0])
    n_join = int(joined.shape[0])
    if n_join != n_dedup:
        # In production we do not silently proceed; write diagnostic then raise.
        missing = dedup[~dedup[args.smiles_col].isin(set(joined[args.smiles_col].tolist()))][[args.smiles_col]].copy()
        miss_path = dirs["reports"] / "graph_build_missing_from_split.csv"
        missing.to_csv(miss_path, index=False)
        raise RuntimeError(
            f"Split manifest does not cover all dedup rows: dedup={n_dedup}, joined={n_join}. "
            f"Missing saved to: {miss_path}"
        )

    # Determine seed (single seed file expected)
    seed_vals = sorted(set(joined["seed"].astype(int).tolist()))
    if len(seed_vals) != 1:
        raise RuntimeError(f"Expected a single seed in split_csv, found: {seed_vals}")
    seed = int(seed_vals[0])

    # Configure featurizer
    cfg = GraphFeaturizerConfig(
        max_atomic_num=args.max_atomic_num,
        max_degree=args.max_degree,
        max_h=args.max_h,
        max_valence=args.max_valence,
    )

    # Build graphs
    graphs: List[Dict] = []
    index_rows: List[Dict] = []
    failures: List[Dict] = []

    for idx, row in joined.reset_index(drop=True).iterrows():
        smi = str(row[args.smiles_col])
        sp = str(row["split"])
        mol = mol_from_smiles(smi)
        if mol is None:
            failures.append({"row_index": int(idx), args.smiles_col: smi, "reason": "rdkit_molfromsmiles_none"})
            continue

        try:
            x, edge_index, edge_attr = mol_to_graph_tensors(mol, cfg)
        except Exception as e:
            failures.append({"row_index": int(idx), args.smiles_col: smi, "reason": f"graph_build_exception:{type(e).__name__}"})
            continue

        # Labels
        y_reg = None
        y_cls = None

        if args.task in ("regression", "both") and args.reg_col in joined.columns:
            v = pd.to_numeric(row.get(args.reg_col, np.nan), errors="coerce")
            if pd.isna(v):
                y_reg = None
            else:
                y_reg = torch.tensor(float(v), dtype=torch.float32)

        if args.task in ("classification", "both") and args.cls_col in joined.columns:
            v = pd.to_numeric(row.get(args.cls_col, np.nan), errors="coerce")
            if pd.isna(v):
                y_cls = None
            else:
                y_cls = torch.tensor(int(v), dtype=torch.long)

        meta = {
            "smiles": smi,
            "split": sp,
            "seed": seed,
        }
        # keep optional ids if present
        for extra in ["molecule_chembl_id", "n_records_smiles"]:
            if extra in joined.columns:
                meta[extra] = row.get(extra)

        g = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y_reg": y_reg,
            "y_cls": y_cls,
            "meta": meta,
        }
        graphs.append(g)

        index_rows.append({
            "graph_id": int(len(graphs) - 1),
            "seed": seed,
            "split": sp,
            args.smiles_col: smi,
            "num_nodes": int(x.shape[0]),
            "num_edges": int(edge_index.shape[1]),
            "has_y_reg": int(y_reg is not None),
            "has_y_cls": int(y_cls is not None),
            "molecule_chembl_id": meta.get("molecule_chembl_id", ""),
            "n_records_smiles": meta.get("n_records_smiles", ""),
        })

    # Failures saved
    failures_path = dirs["reports"] / f"graph_build_seed{seed:03d}_failures.csv"
    pd.DataFrame(failures).to_csv(failures_path, index=False)

    # If any failures occurred, we fail hard to avoid silent leakage/inconsistency.
    # This is a strict production choice: you can relax later, but for publication keep strict.
    if len(failures) > 0:
        raise RuntimeError(
            f"Graph building failed for {len(failures)} molecules. "
            f"Failures saved to: {failures_path}. Fix inputs and rerun."
        )

    # Save index table
    index_df = pd.DataFrame(index_rows)
    index_path = dirs["features"] / f"graphs_seed{seed:03d}_index.csv"
    index_df.to_csv(index_path, index=False)

    # Save graphs cache
    cache = {
        "seed": seed,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "featurizer_config": {
            "max_atomic_num": cfg.max_atomic_num,
            "max_degree": cfg.max_degree,
            "max_h": cfg.max_h,
            "max_valence": cfg.max_valence,
            "node_feature_dim": cfg.node_feature_dim(),
            "edge_feature_dim": cfg.edge_feature_dim(),
            "formal_charges": list(cfg.formal_charges),
            "hybridizations": [str(h) for h in cfg.hybridizations],
            "chiral_tags": [str(c) for c in cfg.chiral_tags],
            "bond_types": [str(b) for b in cfg.bond_types],
            "stereo_allow": list(cfg.stereo_allow),
        },
        "graphs": graphs,
    }
    cache_path = dirs["features"] / f"graphs_seed{seed:03d}.pt"
    torch.save(cache, cache_path)

    # Stats tables
    stats_df = compute_graph_stats(index_rows)
    stats_path = dirs["tables"] / f"graph_stats_seed{seed:03d}.csv"
    stats_df.to_csv(stats_path, index=False)

    # Per-split summary stats
    summary_rows = []
    for sp in ["train", "val", "test"]:
        sub = stats_df[stats_df["split"] == sp].copy()
        if sub.empty:
            continue
        summary_rows.append({
            "seed": seed,
            "split": sp,
            "n_graphs": int(sub.shape[0]),
            "num_nodes_mean": float(sub["num_nodes"].mean()),
            "num_nodes_std": float(sub["num_nodes"].std(ddof=1)) if sub.shape[0] > 1 else 0.0,
            "num_edges_mean": float(sub["num_edges"].mean()),
            "num_edges_std": float(sub["num_edges"].std(ddof=1)) if sub.shape[0] > 1 else 0.0,
            "y_reg_coverage": float(sub["has_y_reg"].mean()) if "has_y_reg" in sub.columns else np.nan,
            "y_cls_coverage": float(sub["has_y_cls"].mean()) if "has_y_cls" in sub.columns else np.nan,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = dirs["tables"] / f"graph_stats_seed{seed:03d}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Figures
    prefix = f"06_seed{seed:03d}"
    plot_num_nodes_by_split(stats_df, dirs["figures"], style, prefix=prefix)
    plot_num_edges_by_split(stats_df, dirs["figures"], style, prefix=prefix)
    plot_label_distributions(joined, dirs["figures"], style, prefix=prefix)

    # Audit report
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dedup_csv": str(dedup_path),
        "split_csv": str(split_path),
        "out_root": str(out_root),
        "seed": seed,
        "n_graphs": int(len(graphs)),
        "node_feature_dim": int(cfg.node_feature_dim()),
        "edge_feature_dim": int(cfg.edge_feature_dim()),
        "cache_path": str(cache_path),
        "index_path": str(index_path),
        "stats_path": str(stats_path),
        "stats_summary_path": str(summary_path),
        "failures_path": str(failures_path),
        "figures": {
            "num_atoms_by_split_svg": str(dirs["figures"] / f"{prefix}_num_atoms_by_split.svg"),
            "num_edges_by_split_svg": str(dirs["figures"] / f"{prefix}_num_edges_by_split.svg"),
        },
        "label_columns": {
            "task": args.task,
            "reg_col": args.reg_col if args.reg_col in joined.columns else None,
            "cls_col": args.cls_col if args.cls_col in joined.columns else None,
        },
    }
    report_path = dirs["reports"] / f"graph_build_seed{seed:03d}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Run manifest
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "outputs": {
            "graphs_cache": str(cache_path),
            "graphs_index_csv": str(index_path),
            "graph_stats_csv": str(stats_path),
            "graph_stats_summary_csv": str(summary_path),
            "audit_json": str(report_path),
            "figures_dir": str(dirs["figures"]),
        },
        "inputs": {
            "dedup_csv": str(dedup_path),
            "split_csv": str(split_path),
        },
    }
    manifest_path = dirs["reports"] / f"run_manifest_graphs_seed{seed:03d}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== DONE: Graph dataset build + cache ===")
    print(f"Seed:             {seed}")
    print(f"Graphs cache:     {cache_path}")
    print(f"Graphs index:     {index_path}")
    print(f"Stats table:      {stats_path}")
    print(f"Stats summary:    {summary_path}")
    print(f"Audit report:     {report_path}")
    print(f"Run manifest:     {manifest_path}")
    print(f"Figures (SVG):    {dirs['figures']}")
    print("========================================\n")


if __name__ == "__main__":
    main()
