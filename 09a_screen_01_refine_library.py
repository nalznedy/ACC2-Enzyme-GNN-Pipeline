#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 09a_screen_01_refine_library.py

Purpose:
    Refine screening-library structures (including FoodDB exports) for model inference.

Workflow step:
    Step 9a - library cleaning and canonicalization before screening.

Main inputs:
    - Input TSV/CSV with a SMILES column (default: moldb_smiles)

Main outputs:
    - screen/01_screen_refined.csv
    - screen/01_screen_refined_stats.json
    - screen/01_screen_refined_dropped.csv (optional)

Notes:
    Optional neutralization should match how training compounds were standardized.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd

from rdkit import Chem

# Optional (recommended) standardization utilities in RDKit
_HAVE_MOLSTD = False
try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
    _HAVE_MOLSTD = True
except Exception:
    _HAVE_MOLSTD = False


def _safe_mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    if smi is None:
        return None
    if not isinstance(smi, str):
        smi = str(smi)
    smi = smi.strip()
    if smi == "" or smi.lower() in {"nan", "none"}:
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        return mol
    except Exception:
        return None


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Return largest fragment by heavy atom count."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return mol
    best = None
    best_ha = -1
    for f in frags:
        ha = f.GetNumHeavyAtoms()
        if ha > best_ha:
            best = f
            best_ha = ha
    # sanitize after selection
    Chem.SanitizeMol(best)
    return best


def _neutralize_if_possible(mol: Chem.Mol) -> Chem.Mol:
    """
    Neutralize charges using RDKit MolStandardize Uncharger when available.
    If MolStandardize is not available, return original mol.
    """
    if not _HAVE_MOLSTD:
        return mol
    try:
        uncharger = rdMolStandardize.Uncharger()
        return uncharger.uncharge(mol)
    except Exception:
        return mol


def _canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def refine_one(
    raw_smiles: str,
    do_largest_fragment: bool,
    do_neutralize: bool,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (canonical_smiles, drop_reason). If OK, drop_reason=None.
    """
    mol = _safe_mol_from_smiles(raw_smiles)
    if mol is None:
        return None, "invalid_smiles"

    if do_largest_fragment:
        try:
            mol = _largest_fragment(mol)
        except Exception:
            return None, "fragment_handling_failed"

    if do_neutralize:
        mol = _neutralize_if_possible(mol)

    try:
        can = _canonical_smiles(mol)
    except Exception:
        return None, "canonicalization_failed"

    if can is None or can.strip() == "":
        return None, "empty_canonical"

    return can, None


def ensure_dirs(out_root: Path) -> Dict[str, Path]:
    screen_dir = out_root / "screen"
    screen_dir.mkdir(parents=True, exist_ok=True)
    return {"screen": screen_dir}


def main() -> None:
    ap = argparse.ArgumentParser(description="Step A: Refine screening library to match QSAR GNN pipeline.")
    ap.add_argument("--in_file", type=str, required=True, help="Input database file (TSV/CSV).")
    ap.add_argument("--out_root", type=str, required=True, help="Project root (QSAR_GNN_Project).")
    ap.add_argument("--sep", type=str, default="\t", help="Delimiter: '\\t' for TSV, ',' for CSV.")
    ap.add_argument("--smiles_col", type=str, default="moldb_smiles", help="Column containing SMILES.")
    ap.add_argument("--id_col", type=str, default="id", help="Identifier column (optional but recommended).")
    ap.add_argument("--name_col", type=str, default="name", help="Name column (optional).")
    ap.add_argument("--keep_cols", type=str, nargs="*", default=None,
                    help="Additional columns to keep (space-separated). If not set, keeps id/name/smiles + canonical.")
    ap.add_argument("--largest_fragment", action="store_true",
                    help="Keep only the largest fragment (recommended for salts/mixtures).")
    ap.add_argument("--neutralize", action="store_true",
                    help="Neutralize charges using RDKit MolStandardize (ONLY if training used neutralization).")
    ap.add_argument("--write_dropped", action="store_true", help="Write dropped rows for auditing.")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    dirs = ensure_dirs(out_root)

    in_path = Path(args.in_file).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Load
    df = pd.read_csv(in_path, sep=args.sep, low_memory=False)
    if args.smiles_col not in df.columns:
        raise ValueError(f"SMILES column not found: '{args.smiles_col}'. Available: {list(df.columns)[:30]} ...")

    # Determine columns to carry forward
    base_cols: List[str] = []
    for c in [args.id_col, args.name_col, args.smiles_col]:
        if c and c in df.columns and c not in base_cols:
            base_cols.append(c)
    if args.keep_cols:
        for c in args.keep_cols:
            if c in df.columns and c not in base_cols:
                base_cols.append(c)

    work = df[base_cols].copy() if base_cols else df[[args.smiles_col]].copy()
    work = work.rename(columns={args.smiles_col: "input_smiles"})

    # Refine
    canon_list = []
    reason_list = []
    for smi in work["input_smiles"].tolist():
        can, reason = refine_one(
            raw_smiles=smi,
            do_largest_fragment=bool(args.largest_fragment),
            do_neutralize=bool(args.neutralize),
        )
        canon_list.append(can)
        reason_list.append(reason)

    work["canonical_smiles"] = canon_list
    work["drop_reason"] = reason_list

    n_total = int(len(work))
    ok = work["canonical_smiles"].notna()
    n_ok = int(ok.sum())
    n_bad = n_total - n_ok

    refined = work.loc[ok].copy()
    # Deduplicate by canonical SMILES
    before_dups = int(len(refined))
    refined = refined.drop_duplicates(subset=["canonical_smiles"], keep="first").reset_index(drop=True)
    after_dups = int(len(refined))
    n_dups_removed = before_dups - after_dups

    out_refined = dirs["screen"] / "01_screen_refined.csv"
    refined.to_csv(out_refined, index=False)

    if args.write_dropped:
        dropped = work.loc[~ok].copy()
        out_dropped = dirs["screen"] / "01_screen_refined_dropped.csv"
        dropped.to_csv(out_dropped, index=False)

    stats = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "input_file": str(in_path),
        "out_root": str(out_root),
        "settings": {
            "sep": args.sep,
            "smiles_col": args.smiles_col,
            "largest_fragment": bool(args.largest_fragment),
            "neutralize": bool(args.neutralize),
            "molstandardize_available": bool(_HAVE_MOLSTD),
        },
        "counts": {
            "n_total_rows": n_total,
            "n_valid_after_refine": n_ok,
            "n_dropped_invalid": n_bad,
            "n_duplicates_removed": n_dups_removed,
            "n_final_unique": after_dups,
        },
        "drop_reason_counts": work.loc[~ok, "drop_reason"].value_counts(dropna=False).to_dict(),
        "outputs": {
            "refined_csv": str(out_refined),
            "dropped_csv": str((dirs["screen"] / "01_screen_refined_dropped.csv")) if args.write_dropped else None,
        },
    }
    out_stats = dirs["screen"] / "01_screen_refined_stats.json"
    out_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n=== DONE: Screening library refinement (Step A) ===")
    print(f"Input:            {in_path}")
    print(f"Refined CSV:      {out_refined}")
    print(f"Stats JSON:       {out_stats}")
    if args.write_dropped:
        print(f"Dropped CSV:      {dirs['screen'] / '01_screen_refined_dropped.csv'}")
    print("Counts:")
    print(f"  Total rows:     {n_total}")
    print(f"  Valid rows:     {n_ok}")
    print(f"  Dropped rows:   {n_bad}")
    print(f"  Dups removed:   {n_dups_removed}")
    print(f"  Final unique:   {after_dups}")
    print("===============================================\n")


if __name__ == "__main__":
    main()
