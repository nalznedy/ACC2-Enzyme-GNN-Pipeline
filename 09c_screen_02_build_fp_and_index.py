#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 09c_screen_02_build_fp_and_index.py

Purpose:
        Build Morgan fingerprints and aligned index files for ensemble screening inference.

Workflow step:
        Step 9c - fingerprint and index preparation for screening.

Main inputs:
        - screen/01_screen_refined.csv
        - screen/02_graphs_seed###_screen.pt

Main outputs:
        - screen/features/screen_graphs.pt
        - screen/features/screen_fp.npy
        - screen/features/screen_index.csv

Notes:
        Row order is aligned across graph, fingerprint, and index files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


# -----------------------------
# RDKit helpers
# -----------------------------
def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    if smi is None or not isinstance(smi, str):
        return None
    smi = smi.strip()
    if smi == "":
        return None
    try:
        return Chem.MolFromSmiles(smi, sanitize=True)
    except Exception:
        return None


def canon_smiles(smi: str) -> Optional[str]:
    mol = mol_from_smiles(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        return None


def morgan_fp_bits(canon_smi: str, radius: int, n_bits: int) -> np.ndarray:
    mol = mol_from_smiles(canon_smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES for fingerprint: {canon_smi}")
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)


# -----------------------------
# IO helpers
# -----------------------------
def load_graphs_pt(path: Path) -> Tuple[List[Dict], Dict]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "graphs" in obj:
        graphs = obj["graphs"]
        meta = {k: v for k, v in obj.items() if k != "graphs"}
        return graphs, meta
    raise ValueError(f"Invalid graphs_pt format (expected dict with key 'graphs'): {path}")


def ensure_features_dir(out_root: Path) -> Path:
    feat_dir = out_root / "screen" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    return feat_dir


def _pick_meta_id(meta: Dict) -> Optional[str]:
    """
    Try several common keys that may exist in screen graph meta.
    """
    for k in ("id", "screen_id", "record_id", "compound_id", "public_id", "moldb_id"):
        if k in meta and meta[k] is not None:
            return str(meta[k])
    return None


def _pick_meta_name(meta: Dict) -> Optional[str]:
    for k in ("name", "screen_name", "compound_name", "moldb_name"):
        if k in meta and meta[k] is not None:
            return str(meta[k])
    return None


def _pick_meta_smiles(meta: Dict) -> Optional[str]:
    for k in ("canonical_smiles", "smiles", "moldb_smiles"):
        if k in meta and meta[k] is not None:
            return str(meta[k])
    return None


# -----------------------------
# Alignment logic
# -----------------------------
def build_row_maps(df: pd.DataFrame, id_col: str, name_col: str, smiles_col: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Returns:
      id_to_row: map id -> row index (first occurrence)
      smi_to_row: map canonical_smiles -> row index (first occurrence)
    """
    id_to_row: Dict[str, int] = {}
    smi_to_row: Dict[str, int] = {}

    for i, r in df.iterrows():
        rid = None
        if id_col in df.columns and pd.notna(r.get(id_col, None)):
            rid = str(r[id_col])

        smi_raw = r.get(smiles_col, None)
        smi_can = canon_smiles(str(smi_raw)) if pd.notna(smi_raw) else None

        if rid is not None and rid not in id_to_row:
            id_to_row[rid] = int(i)
        if smi_can is not None and smi_can not in smi_to_row:
            smi_to_row[smi_can] = int(i)

    return id_to_row, smi_to_row


def align_graph_to_csv_row(
    g: Dict,
    df: pd.DataFrame,
    id_to_row: Dict[str, int],
    smi_to_row: Dict[str, int],
    id_col: str,
    name_col: str,
    smiles_col: str,
) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[str]]:
    """
    Returns:
      row_index, screen_id, screen_name, canonical_smiles
    """
    meta = g.get("meta", {}) if isinstance(g, dict) else {}
    g_id = _pick_meta_id(meta)
    g_name = _pick_meta_name(meta)
    g_smi = _pick_meta_smiles(meta)

    g_can = canon_smiles(g_smi) if g_smi else None

    # 1) Prefer ID match (most reliable)
    if g_id is not None and g_id in id_to_row:
        ri = id_to_row[g_id]
        row = df.loc[ri]
        sid = str(row[id_col]) if id_col in df.columns and pd.notna(row.get(id_col, None)) else g_id
        sname = str(row[name_col]) if name_col in df.columns and pd.notna(row.get(name_col, None)) else (g_name or "")
        smi_raw = row.get(smiles_col, None)
        scan = canon_smiles(str(smi_raw)) if pd.notna(smi_raw) else g_can
        return ri, sid, sname, scan

    # 2) Fallback: canonical SMILES match
    if g_can is not None and g_can in smi_to_row:
        ri = smi_to_row[g_can]
        row = df.loc[ri]
        sid = str(row[id_col]) if id_col in df.columns and pd.notna(row.get(id_col, None)) else (g_id or "")
        sname = str(row[name_col]) if name_col in df.columns and pd.notna(row.get(name_col, None)) else (g_name or "")
        smi_raw = row.get(smiles_col, None)
        scan = canon_smiles(str(smi_raw)) if pd.notna(smi_raw) else g_can
        return ri, sid, sname, scan

    # 3) No match found
    return None, (g_id or ""), (g_name or ""), g_can


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build screening fingerprints + index (compatible with Script-10).")
    ap.add_argument("--refined_csv", type=str, required=True)
    ap.add_argument("--graphs_pt", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)

    ap.add_argument("--smiles_col", type=str, default="moldb_smiles")
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--name_col", type=str, default="name")

    ap.add_argument("--fp_radius", type=int, default=2)
    ap.add_argument("--fp_bits", type=int, default=2048)

    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    refined_csv = Path(args.refined_csv).resolve()
    graphs_pt = Path(args.graphs_pt).resolve()

    if not refined_csv.exists():
        raise FileNotFoundError(f"Missing refined_csv: {refined_csv}")
    if not graphs_pt.exists():
        raise FileNotFoundError(f"Missing graphs_pt: {graphs_pt}")

    feat_dir = ensure_features_dir(out_root)
    out_graphs = feat_dir / "screen_graphs.pt"
    out_fp = feat_dir / "screen_fp.npy"
    out_index = feat_dir / "screen_index.csv"
    out_manifest = feat_dir / "screen_features_manifest.json"

    df = pd.read_csv(refined_csv)
    for col in (args.smiles_col, args.id_col, args.name_col):
        if col not in df.columns:
            # name_col can be missing; keep it optional, but id/smiles are required
            if col in (args.smiles_col, args.id_col):
                raise ValueError(f"Missing column in refined_csv: {col}")

    graphs, graphs_meta = load_graphs_pt(graphs_pt)
    if not isinstance(graphs, list) or len(graphs) == 0:
        raise ValueError("graphs_pt contains no graphs.")

    # Build lookup maps from CSV
    id_to_row, smi_to_row = build_row_maps(df, args.id_col, args.name_col, args.smiles_col)

    # Align and build outputs in graph order
    fps = np.zeros((len(graphs), int(args.fp_bits)), dtype=np.float32)
    index_rows = []
    n_bad = 0
    n_nomatch = 0

    for i, g in enumerate(graphs):
        meta = g.get("meta", {}) if isinstance(g, dict) else {}
        ri, sid, sname, scan = align_graph_to_csv_row(
            g, df, id_to_row, smi_to_row, args.id_col, args.name_col, args.smiles_col
        )

        if scan is None:
            n_bad += 1
            scan = ""  # keep placeholder

        # Compute fp (if possible)
        if scan != "":
            try:
                fps[i, :] = morgan_fp_bits(scan, radius=int(args.fp_radius), n_bits=int(args.fp_bits))
            except Exception:
                n_bad += 1
                fps[i, :] = 0.0
        else:
            fps[i, :] = 0.0

        if ri is None:
            n_nomatch += 1

        # Enrich graph meta for downstream traceability
        meta = dict(meta)
        meta["canonical_smiles"] = scan
        meta["screen_id"] = sid
        meta["screen_name"] = sname
        meta["split"] = meta.get("split", "screen")
        g["meta"] = meta

        index_rows.append(
            {
                "row_index": i,
                "screen_id": sid,
                "name": sname,
                "canonical_smiles": scan,
                "matched_csv_row": (int(ri) if ri is not None else -1),
            }
        )

    # Save outputs
    np.save(out_fp, fps.astype(np.float32))
    pd.DataFrame(index_rows).to_csv(out_index, index=False)

    torch.save(
        {
            "graphs": graphs,
            "source_graphs_pt": str(graphs_pt),
            "source_refined_csv": str(refined_csv),
            "fp_radius": int(args.fp_radius),
            "fp_bits": int(args.fp_bits),
            "n_graphs": int(len(graphs)),
            "notes": "This file is written to match 10_screen_03_ensemble_infer_rank.py expectations.",
            **graphs_meta,
        },
        out_graphs,
    )

    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "inputs": {"refined_csv": str(refined_csv), "graphs_pt": str(graphs_pt)},
        "outputs": {"screen_graphs": str(out_graphs), "screen_fp": str(out_fp), "screen_index": str(out_index)},
        "fp": {"radius": int(args.fp_radius), "bits": int(args.fp_bits)},
        "counts": {"n_graphs": int(len(graphs)), "n_bad_smiles_or_fp": int(n_bad), "n_no_csv_match": int(n_nomatch)},
    }
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== DONE: Screening features built (Script-10 compatible) ===")
    print(f"Features dir:     {feat_dir}")
    print(f"screen_graphs.pt: {out_graphs}")
    print(f"screen_fp.npy:    {out_fp}")
    print(f"screen_index.csv: {out_index}")
    print(f"Manifest:         {out_manifest}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
