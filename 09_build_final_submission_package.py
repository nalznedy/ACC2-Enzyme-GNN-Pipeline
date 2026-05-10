#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 09_build_final_submission_package.py

Purpose:
    Assemble final manuscript submission folders from generated modeling outputs.

Workflow step:
    Step 9 - packaging figures, tables, manifests, and optional prediction exports.

Main inputs:
    - release_gnn_reporting/*
    - reports/*
    - interpretability_ad/*

Main outputs:
    - submission_package_NC/
    - submission_package_NC/figure_index.csv
    - submission_package_NC/table_index.csv
    - submission_package_NC/supplementary_data_index.csv

Notes:
    File copies preserve original names unless alias generation is requested.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    safe_mkdir(dst.parent)
    shutil.copy2(src, dst)
    return True


def copy_glob(src_dir: Path, pattern: str, dst_dir: Path) -> List[Path]:
    if not src_dir.exists():
        return []
    safe_mkdir(dst_dir)
    copied = []
    for f in sorted(src_dir.glob(pattern)):
        if f.is_file():
            dst = dst_dir / f.name
            shutil.copy2(f, dst)
            copied.append(dst)
    return copied


def read_text_if_exists(p: Path) -> str:
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def write_text(p: Path, txt: str) -> None:
    safe_mkdir(p.parent)
    p.write_text(txt, encoding="utf-8")


def json_if_exists(p: Path) -> Optional[Dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def to_index_csv(items: List[Dict], out_csv: Path) -> None:
    safe_mkdir(out_csv.parent)
    pd.DataFrame(items).to_csv(out_csv, index=False)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


# -----------------------------
# Package layout
# -----------------------------
@dataclass
class PackageDirs:
    root: Path
    manifests: Path
    main_figs: Path
    supp_figs: Path
    main_tables: Path
    supp_tables: Path
    preds: Path
    cfg_env: Path
    atom: Path

    @staticmethod
    def create(base: Path) -> "PackageDirs":
        d = PackageDirs(
            root=base,
            manifests=base / "01_Manifests",
            main_figs=base / "02_Main_Figures",
            supp_figs=base / "03_Supplementary_Figures",
            main_tables=base / "04_Main_Tables",
            supp_tables=base / "05_Supplementary_Tables",
            preds=base / "06_Predictions_Test",
            cfg_env=base / "07_Configs_and_Env",
            atom=base / "08_Atom_Attribution_Examples",
        )
        for p in [
            d.root, d.manifests, d.main_figs, d.supp_figs, d.main_tables, d.supp_tables, d.preds, d.cfg_env, d.atom
        ]:
            safe_mkdir(p)
        return d


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Script-10: Build final Nature Communications submission package.")
    ap.add_argument("--out_root", type=str, required=True, help="Project root (QSAR_GNN_Project).")
    ap.add_argument("--package_name", type=str, default="submission_package_NC", help="Output folder name.")
    ap.add_argument("--copy_predictions", action="store_true", help="Copy per-seed test predictions CSVs.")
    ap.add_argument("--make_aliases", action="store_true", help="Create standardized aliases (Fig_1.svg etc.).")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    pkg_root = out_root / args.package_name
    pkg = PackageDirs.create(pkg_root)

    timestamp = datetime.utcnow().isoformat() + "Z"

    # Expected sources
    s7_release = out_root / "release_gnn_reporting"
    s7_tables = s7_release / "tables"
    s7_figs = s7_release / "figures"
    s8_reports = out_root / "reports"
    s9_root = out_root / "interpretability_ad"
    s9_tables = s9_root / "tables"
    s9_figs = s9_root / "figures"
    s9_atom = s9_root / "atom_svgs"
    s9_reports = s9_root / "reports"

    # -------------------------
    # Copy manifests
    # -------------------------
    manifests_copied = []
    for p in [
        s7_release / "manifest_script7.json",
        s8_reports / "manifest_script8.json",
        s9_reports / "manifest_script9.json",
    ]:
        dst = pkg.manifests / p.name
        if copy_if_exists(p, dst):
            manifests_copied.append(dst)

    # Also copy README from Script-7 release if present
    copy_if_exists(s7_release / "README.txt", pkg.manifests / "README_script7_release.txt")

    # -------------------------
    # Copy core tables (Main + Supplementary)
    # -------------------------
    main_table_files = [
        s7_tables / "test_metrics_summary_mean_sd.csv",
        s7_tables / "paired_statistics_test.csv",
    ]
    supp_table_files = [
        s7_tables / "merged_metrics_all_models_all_splits.csv",
        out_root / "tables" / "results_sentence_blocks.csv",
        s9_tables / "51_inside_AD_performance_by_seed.csv",
        s9_tables / "seed_all_test_embeddings_similarity_merged.csv",
    ]

    main_tables_copied = []
    for p in main_table_files:
        if copy_if_exists(p, pkg.main_tables / p.name):
            main_tables_copied.append(pkg.main_tables / p.name)

    supp_tables_copied = []
    for p in supp_table_files:
        if copy_if_exists(p, pkg.supp_tables / p.name):
            supp_tables_copied.append(pkg.supp_tables / p.name)

    # Copy all Script-9 per-seed tables (errors, bins) as supplementary
    supp_tables_copied += copy_glob(s9_tables, "*.csv", pkg.supp_tables)

    # -------------------------
    # Copy figures
    # -------------------------
    # Heuristic: Script-7 figures are "main", Script-9 figures are "supplementary" by default.
    main_figs_copied = copy_glob(s7_figs, "*.svg", pkg.main_figs)
    # also copy png for convenience
    copy_glob(s7_figs, "*.png", pkg.main_figs)

    supp_figs_copied = copy_glob(s9_figs, "*.svg", pkg.supp_figs)
    copy_glob(s9_figs, "*.png", pkg.supp_figs)

    # Atom attribution examples (SVG)
    atom_copied = copy_glob(s9_atom, "*.svg", pkg.atom)

    # -------------------------
    # Copy results text + methods + captions from Script-8
    # -------------------------
    copied_texts = []
    for p in [
        s8_reports / "results_nature_comm_style.txt",
        s8_reports / "results_nature_comm_style.md",
        s8_reports / "figure_captions_nc.txt",
        s8_reports / "modeling_methods_qsar_nc.txt",
    ]:
        if copy_if_exists(p, pkg.manifests / p.name):
            copied_texts.append(pkg.manifests / p.name)

    # -------------------------
    # Copy configs + environment capture
    # -------------------------
    # Bring pip freeze from Script-7 release if present
    copy_if_exists(s7_release / "environment" / "pip_freeze.txt", pkg.cfg_env / "pip_freeze.txt")

    # Copy any best-config JSONs saved by Script-5/6
    copied_cfgs = []
    for p in sorted((out_root / "reports").glob("*best_config_seed*.json")):
        if copy_if_exists(p, pkg.cfg_env / p.name):
            copied_cfgs.append(pkg.cfg_env / p.name)

    # Copy script manifests and pipeline manifests if present
    for p in sorted((out_root / "reports").glob("manifest_script*.json")):
        copy_if_exists(p, pkg.manifests / p.name)

    # -------------------------
    # Copy test predictions (optional)
    # -------------------------
    preds_copied = []
    if args.copy_predictions:
        pred_dir = out_root / "predictions"
        if pred_dir.exists():
            preds_copied += copy_glob(pred_dir, "*_test.csv", pkg.preds)

    # -------------------------
    # Generate indices
    # -------------------------
    fig_index = []
    for f in sorted(pkg.main_figs.glob("*.svg")):
        fig_index.append({
            "category": "Main",
            "suggested_label": "",  # fill later (Figure 1, 2, etc.)
            "file": f.name,
            "relative_path": _rel(f, pkg.root),
            "caption": "",
            "notes": "Generated by Script-7 (multi-seed summary).",
        })
    for f in sorted(pkg.supp_figs.glob("*.svg")):
        fig_index.append({
            "category": "Supplementary",
            "suggested_label": "",  # fill later (Figure S1, S2, etc.)
            "file": f.name,
            "relative_path": _rel(f, pkg.root),
            "caption": "",
            "notes": "Generated by Script-9 (AD, chemical space, error analysis).",
        })
    for f in sorted(pkg.atom.glob("*.svg")):
        fig_index.append({
            "category": "Supplementary",
            "suggested_label": "",
            "file": f.name,
            "relative_path": _rel(f, pkg.root),
            "caption": "Atom-level attribution example (test set).",
            "notes": "Generated by Script-9 (gradient-based attribution).",
        })
    to_index_csv(fig_index, pkg.root / "Figure_Index.csv")

    table_index = []
    for f in sorted(pkg.main_tables.glob("*.csv")):
        table_index.append({
            "category": "Main",
            "suggested_label": "",
            "file": f.name,
            "relative_path": _rel(f, pkg.root),
            "title": "",
            "notes": "Core performance reporting.",
        })
    for f in sorted(pkg.supp_tables.glob("*.csv")):
        table_index.append({
            "category": "Supplementary",
            "suggested_label": "",
            "file": f.name,
            "relative_path": _rel(f, pkg.root),
            "title": "",
            "notes": "Extended reporting / AD / error analysis.",
        })
    to_index_csv(table_index, pkg.root / "Table_Index.csv")

    # Supplementary Data index (NC-style)
    supp_data_items = []
    # include all supplementary CSVs + test predictions (if copied)
    for f in sorted(pkg.supp_tables.glob("*.csv")):
        supp_data_items.append({
            "supp_data_label": "",  # Supplementary Data 1, 2, ...
            "file": f.name,
            "relative_path": _rel(f, pkg.root),
            "description": "",
        })
    for f in sorted(pkg.preds.glob("*.csv")):
        supp_data_items.append({
            "supp_data_label": "",
            "file": f.name,
            "relative_path": _rel(f, pkg.root),
            "description": "Per-compound test predictions (seed-specific).",
        })
    to_index_csv(supp_data_items, pkg.root / "Supplementary_Data_Index.csv")

    # -------------------------
    # Optional standardized aliases (keeps originals)
    # -------------------------
    alias_log = []
    if args.make_aliases:
        # Main figures: Fig_1.svg, Fig_2.svg ...
        for i, f in enumerate(sorted(pkg.main_figs.glob("*.svg")), start=1):
            alias = pkg.main_figs / f"Fig_{i}.svg"
            if not alias.exists():
                shutil.copy2(f, alias)
                alias_log.append({"original": f.name, "alias": alias.name, "category": "Main"})
        # Supplementary figures: Fig_S1.svg ...
        for i, f in enumerate(sorted(pkg.supp_figs.glob("*.svg")), start=1):
            alias = pkg.supp_figs / f"Fig_S{i}.svg"
            if not alias.exists():
                shutil.copy2(f, alias)
                alias_log.append({"original": f.name, "alias": alias.name, "category": "Supplementary"})
        # Atom attribution: Fig_SA1.svg ...
        for i, f in enumerate(sorted(pkg.atom.glob("*.svg")), start=1):
            alias = pkg.atom / f"Fig_SA{i}.svg"
            if not alias.exists():
                shutil.copy2(f, alias)
                alias_log.append({"original": f.name, "alias": alias.name, "category": "Atom_Attribution"})
        to_index_csv(alias_log, pkg.root / "Alias_Log.csv")

    # -------------------------
    # README
    # -------------------------
    readme = []
    readme.append("Nature Communications submission package (QSAR GNN)\n")
    readme.append(f"Generated: {timestamp}\n")
    readme.append(f"Project root: {out_root}\n\n")

    readme.append("Folder structure:\n")
    readme.append("  01_Manifests/                  script manifests and text outputs\n")
    readme.append("  02_Main_Figures/               SVG/PNG summary figures (multi-seed performance)\n")
    readme.append("  03_Supplementary_Figures/      SVG/PNG AD and chemical space figures\n")
    readme.append("  04_Main_Tables/                primary performance tables\n")
    readme.append("  05_Supplementary_Tables/       AD, binning, hard errors, merged diagnostics\n")
    readme.append("  06_Predictions_Test/           per-compound test predictions (optional)\n")
    readme.append("  07_Configs_and_Env/            best configs and environment capture\n")
    readme.append("  08_Atom_Attribution_Examples/  test-only atom attribution SVGs\n\n")

    readme.append("Index files:\n")
    readme.append("  Figure_Index.csv\n")
    readme.append("  Table_Index.csv\n")
    readme.append("  Supplementary_Data_Index.csv\n")
    if args.make_aliases:
        readme.append("  Alias_Log.csv (standardized alias filenames)\n")

    readme.append("\nNotes:\n")
    readme.append("- Main vs supplementary grouping is a default. You can move files as needed.\n")
    readme.append("- Captions are provided as templates in 01_Manifests/figure_captions_nc.txt.\n")
    readme.append("- Use the index CSVs to assign final figure numbers and titles.\n")

    write_text(pkg.root / "00_README_submission_package.txt", "".join(readme))

    # -------------------------
    # Package manifest
    # -------------------------
    pkg_manifest = {
        "timestamp": timestamp,
        "project_root": str(out_root),
        "package_root": str(pkg.root),
        "copied": {
            "manifests": [str(p) for p in manifests_copied],
            "main_tables": [str(p) for p in main_tables_copied],
            "supp_tables_count": int(len(list(pkg.supp_tables.glob("*.csv")))),
            "main_figs_count_svg": int(len(list(pkg.main_figs.glob("*.svg")))),
            "supp_figs_count_svg": int(len(list(pkg.supp_figs.glob("*.svg")))),
            "atom_svg_count": int(len(list(pkg.atom.glob("*.svg")))),
            "predictions_count": int(len(preds_copied)),
            "configs_count": int(len(copied_cfgs)),
        },
        "indices": {
            "figure_index": str(pkg.root / "Figure_Index.csv"),
            "table_index": str(pkg.root / "Table_Index.csv"),
            "supplementary_data_index": str(pkg.root / "Supplementary_Data_Index.csv"),
            "alias_log": str(pkg.root / "Alias_Log.csv") if args.make_aliases else None,
        },
        "source_paths": {
            "script7_release": str(s7_release),
            "script8_reports": str(s8_reports),
            "script9_root": str(s9_root),
        },
    }
    write_text(pkg.manifests / "manifest_script10.json", json.dumps(pkg_manifest, indent=2))

    print("\n=== DONE: Script-10 submission package created ===")
    print(f"Package folder: {pkg.root}")
    print(f"README:         {pkg.root / '00_README_submission_package.txt'}")
    print(f"Figure index:   {pkg.root / 'Figure_Index.csv'}")
    print(f"Table index:    {pkg.root / 'Table_Index.csv'}")
    print(f"Supp data idx:  {pkg.root / 'Supplementary_Data_Index.csv'}")
    if args.make_aliases:
        print(f"Aliases:        {pkg.root / 'Alias_Log.csv'}")
    print("=================================================\n")


if __name__ == "__main__":
    main()
