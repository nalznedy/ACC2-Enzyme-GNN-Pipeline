#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 09b_screen_02_make_graph_cache.py

Purpose:
        Build screening graph caches using the same feature schema as model training.

Workflow step:
        Step 9b - graph-cache preparation for screening compounds.

Main inputs:
        - screen/01_screen_refined.csv

Main outputs:
        - screen/02_graphs_seed{seed}_screen.pt
        - screen/02_graphs_seed{seed}_screen_stats.json

Notes:
        Feature dimensions are aligned to Script 02 to keep inference compatible.
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
import torch

from rdkit import Chem
from rdkit.Chem.rdchem import BondType, ChiralType, HybridizationType


# -----------------------------
# One-hot helpers (same style as training)
# -----------------------------
def one_hot(value: int, allowable: List[int]) -> List[int]:
    return [1 if value == a else 0 for a in allowable]


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


# -----------------------------
# Feature config (MATCHES 02_graph_dataset_builder.py)
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
    stereo_allow: Tuple[str, ...] = (
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    )

    def node_feature_dim(self) -> int:
        # atomic_num one-hot (1..max_atomic_num) +
        # degree (0..max_degree) +
        # formal_charge (over predefined set) +
        # hybridization +
        # aromatic + ring +
        # chirality +
        # implicitH (0..max_h) +
        # totalValence (0..max_valence) +
        # atomicMassScaled
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
# RDKit helpers
# -----------------------------
def safe_mol(smi: str) -> Optional[Chem.Mol]:
    if smi is None:
        return None
    smi = str(smi).strip()
    if smi == "":
        return None
    try:
        return Chem.MolFromSmiles(smi, sanitize=True)
    except Exception:
        return None


# -----------------------------
# Feature builders (MATCH training)
# -----------------------------
def atom_features(atom: Chem.Atom, cfg: GraphFeaturizerConfig) -> np.ndarray:
    # Atomic number: 1..max_atomic_num (unknown -> clipped)
    z = clamp_int(atom.GetAtomicNum(), 1, cfg.max_atomic_num)
    atomic_num_oh = one_hot(z, list(range(1, cfg.max_atomic_num + 1)))

    # Degree: 0..max_degree
    deg = clamp_int(atom.GetDegree(), 0, cfg.max_degree)
    degree_oh = one_hot(deg, list(range(0, cfg.max_degree + 1)))

    # Formal charge over cfg.formal_charges (clip into range if outside)
    fc = int(atom.GetFormalCharge())
    if fc not in cfg.formal_charges:
        fc = clamp_int(fc, min(cfg.formal_charges), max(cfg.formal_charges))
    formal_charge_oh = one_hot(fc, list(cfg.formal_charges))

    # Hybridization (unknown -> all zeros)
    hyb = atom.GetHybridization()
    hyb_oh = [1 if hyb == h else 0 for h in cfg.hybridizations]

    aromatic = int(atom.GetIsAromatic())
    ring = int(atom.IsInRing())

    # Chirality tag (unknown -> all zeros)
    ct = atom.GetChiralTag()
    chiral_oh = [1 if ct == c else 0 for c in cfg.chiral_tags]

    # Total Hs: 0..max_h
    ih = clamp_int(atom.GetTotalNumHs(includeNeighbors=True), 0, cfg.max_h)
    ih_oh = one_hot(ih, list(range(0, cfg.max_h + 1)))

    # Total valence: 0..max_valence
    tv = clamp_int(atom.GetTotalValence(), 0, cfg.max_valence)
    tv_oh = one_hot(tv, list(range(0, cfg.max_valence + 1)))

    # Atomic mass (scaled)
    # (Same simple scaling used in many GNN baselines; training builder uses this convention.)
    mass = atom.GetMass() * 0.01

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
        + [mass]
    )
    return np.asarray(feats, dtype=np.float32)


def bond_features(bond: Chem.Bond, cfg: GraphFeaturizerConfig) -> np.ndarray:
    bt = bond.GetBondType()
    bt_oh = [1 if bt == b else 0 for b in cfg.bond_types]

    conj = int(bond.GetIsConjugated())
    ring = int(bond.IsInRing())
    aromatic_bond = int(bond.GetIsAromatic())

    st = str(bond.GetStereo())
    stereo_oh = [1 if st == s else 0 for s in cfg.stereo_allow]

    feats = bt_oh + [conj] + [ring] + [aromatic_bond] + stereo_oh
    return np.asarray(feats, dtype=np.float32)


def mol_to_graph(mol: Chem.Mol, cfg: GraphFeaturizerConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns x, edge_index, edge_attr
    - Adds both directions for each bond.
    """
    n = mol.GetNumAtoms()
    x = np.stack([atom_features(mol.GetAtomWithIdx(i), cfg) for i in range(n)], axis=0).astype(np.float32)

    srcs: List[int] = []
    dsts: List[int] = []
    eattrs: List[np.ndarray] = []

    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bf = bond_features(b, cfg)

        # i -> j
        srcs.append(i)
        dsts.append(j)
        eattrs.append(bf)
        # j -> i
        srcs.append(j)
        dsts.append(i)
        eattrs.append(bf)

    if len(srcs) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, cfg.edge_feature_dim()), dtype=np.float32)
    else:
        edge_index = np.asarray([srcs, dsts], dtype=np.int64)
        edge_attr = np.stack(eattrs, axis=0).astype(np.float32)

    return x, edge_index, edge_attr


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dirs(out_root: Path) -> Dict[str, Path]:
    screen_dir = out_root / "screen"
    screen_dir.mkdir(parents=True, exist_ok=True)
    return {"screen": screen_dir}


def main() -> None:
    ap = argparse.ArgumentParser(description="Step B: Convert refined screening library to training-aligned graph cache.")
    ap.add_argument("--refined_csv", type=str, required=True, help="Step A output CSV.")
    ap.add_argument("--out_root", type=str, required=True, help="QSAR project root.")
    ap.add_argument("--seed", type=int, default=2025, help="Seed id used for naming (not used for splitting).")

    ap.add_argument("--smiles_col", type=str, default="canonical_smiles", help="Canonical SMILES column.")
    ap.add_argument("--id_col", type=str, default="id", help="Optional id column.")
    ap.add_argument("--public_id_col", type=str, default="public_id", help="Optional public id column.")
    ap.add_argument("--name_col", type=str, default="name", help="Optional name column.")

    ap.add_argument("--max_atoms", type=int, default=120, help="Drop molecules with more than this many atoms.")
    ap.add_argument("--keep_invalid", action="store_true", help="Keep invalid as dropped list only.")
    args = ap.parse_args()

    refined_csv = Path(args.refined_csv).resolve()
    if not refined_csv.exists():
        raise FileNotFoundError(f"Refined CSV not found: {refined_csv}")

    out_root = Path(args.out_root).resolve()
    dirs = ensure_dirs(out_root)

    df = pd.read_csv(refined_csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Missing smiles_col '{args.smiles_col}' in {refined_csv}. Columns: {list(df.columns)}")

    cfg = GraphFeaturizerConfig()

    graphs: List[Dict] = []
    dropped: List[Dict] = []

    n_total = int(len(df))
    for idx, row in df.iterrows():
        smi = str(row[args.smiles_col]).strip()
        mol = safe_mol(smi)
        if mol is None:
            dropped.append({"row": int(idx), "reason": "invalid_smiles", "smiles": smi})
            continue

        n_atoms = mol.GetNumAtoms()
        if n_atoms <= 0:
            dropped.append({"row": int(idx), "reason": "zero_atoms", "smiles": smi})
            continue
        if n_atoms > int(args.max_atoms):
            dropped.append(
                {"row": int(idx), "reason": f"too_many_atoms>{args.max_atoms}", "smiles": smi, "n_atoms": int(n_atoms)}
            )
            continue

        try:
            x, edge_index, edge_attr = mol_to_graph(mol, cfg)
        except Exception as e:
            dropped.append({"row": int(idx), "reason": "graph_build_failed", "smiles": smi, "error": str(e)})
            continue

        meta = {
            "smiles": smi,
            "split": "screen",
            "seed": int(args.seed),
        }
        for col, key in [(args.id_col, "id"), (args.public_id_col, "public_id"), (args.name_col, "name")]:
            if col and col in df.columns:
                v = row[col]
                if pd.notna(v):
                    meta[key] = str(v)

        g = {
            "x": torch.tensor(x, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
            "meta": meta,
        }
        graphs.append(g)

    out_pt = dirs["screen"] / f"02_graphs_seed{int(args.seed)}_screen.pt"
    torch.save(
        {
            "graphs": graphs,
            "seed": int(args.seed),
            "mode": "screen",
            "node_in": int(cfg.node_feature_dim()),
            "edge_in": int(cfg.edge_feature_dim()),
            "feature_schema": "MATCHES 02_graph_dataset_builder.py",
        },
        out_pt,
    )

    stats = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "input_refined_csv": str(refined_csv),
        "out_pt": str(out_pt),
        "seed": int(args.seed),
        "counts": {
            "n_total_rows": n_total,
            "n_graphs_ok": int(len(graphs)),
            "n_dropped": int(len(dropped)),
        },
        "max_atoms": int(args.max_atoms),
        "dims": {"node_in": int(cfg.node_feature_dim()), "edge_in": int(cfg.edge_feature_dim())},
        "feature_schema": {
            "atom": "atomic_num(1..100 one-hot, clipped); degree(0..5); formal_charge(-3..3); "
                    "hybridization(5); aromatic; ring; chirality(4); implicitH(0..4); totalValence(0..6); mass_scaled",
            "bond": "bond_type(4); conjugated; ring; aromatic_bond; stereo(6)",
        },
        "dropped_preview": dropped[:50],
    }
    out_stats = dirs["screen"] / f"02_graphs_seed{int(args.seed)}_screen_stats.json"
    out_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n=== DONE: Graph cache for screening (Step B, training-aligned) ===")
    print(f"Input refined:   {refined_csv}")
    print(f"Graphs saved:    {out_pt}")
    print(f"Stats JSON:      {out_stats}")
    print(f"node_in/edge_in: {cfg.node_feature_dim()}/{cfg.edge_feature_dim()}")
    print(f"Total rows:      {n_total}")
    print(f"Graphs OK:       {len(graphs)}")
    print(f"Dropped:         {len(dropped)}")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
