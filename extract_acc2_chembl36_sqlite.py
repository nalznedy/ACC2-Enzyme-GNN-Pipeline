#!/usr/bin/env python3
"""
Script: extract_acc2_chembl36_sqlite.py

Purpose:
        Extract ACC2 (CHEMBL4829) activity records from a local ChEMBL 36 SQLite database.

Workflow step:
        Utility pre-step before Script 00 when raw ACC2 extracts are needed.

Main inputs:
        - ChEMBL SQLite database file (for example chembl_36.db)
        - Target ID (ACC2: CHEMBL4829)

Main outputs:
        - <TARGET>_raw.csv
        - <TARGET>_qsar_ready.csv

Notes:
        QSAR-ready output keeps records with standard_relation '=', valid SMILES, and positive activity values.
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
from datetime import datetime
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract ChEMBL target activities from SQLite .db")
    p.add_argument("--db", required=True, help="Path to ChEMBL SQLite .db file (e.g., chembl_36.db)")
    p.add_argument("--target", required=True, help="Target ChEMBL ID (for ACC2 use CHEMBL4829).")
    p.add_argument("--outdir", default="out", help="Output directory")
    p.add_argument("--confidence_min", type=int, default=7, help="Minimum assay confidence_score (default: 7)")
    p.add_argument("--units", default="nM,uM", help="Allowed standard_units (default: nM,uM)")
    p.add_argument("--types", default="IC50,Ki,Kd,EC50", help="Allowed standard_type list")
    p.add_argument("--require_standard_flag", action="store_true",
                   help="If set, require standard_flag=1 (otherwise allow NULL too).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    units: Tuple[str, ...] = tuple(u.strip() for u in args.units.split(",") if u.strip())
    types: Tuple[str, ...] = tuple(t.strip() for t in args.types.split(",") if t.strip())

    raw_csv = os.path.join(args.outdir, f"{args.target}_raw.csv")
    qsar_csv = os.path.join(args.outdir, f"{args.target}_qsar_ready.csv")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    placeholders_units = ",".join("?" for _ in units)
    placeholders_types = ",".join("?" for _ in types)

    standard_flag_clause = "AND a.standard_flag = 1" if args.require_standard_flag else "AND (a.standard_flag = 1 OR a.standard_flag IS NULL)"

    sql = f"""
    SELECT
        md.chembl_id              AS molecule_chembl_id,
        cs.canonical_smiles       AS canonical_smiles,

        a.standard_type           AS standard_type,
        a.standard_relation       AS standard_relation,
        a.standard_value          AS standard_value,
        a.standard_units          AS standard_units,
        a.pchembl_value           AS pchembl_value,

        ass.chembl_id             AS assay_chembl_id,
        ass.assay_type            AS assay_type,
        ass.confidence_score      AS confidence_score,

        td.chembl_id              AS target_chembl_id,
        td.pref_name              AS target_name,
        d.year                    AS year
    FROM activities a
    JOIN assays ass              ON ass.assay_id = a.assay_id
    JOIN target_dictionary td    ON td.tid = ass.tid
    JOIN molecule_dictionary md  ON md.molregno = a.molregno
    LEFT JOIN compound_structures cs ON cs.molregno = md.molregno
    LEFT JOIN docs d             ON d.doc_id = ass.doc_id
    WHERE td.chembl_id = ?
      AND a.standard_value IS NOT NULL
      AND a.standard_units IN ({placeholders_units})
      AND a.standard_type IN ({placeholders_types})
      AND ass.confidence_score >= ?
      {standard_flag_clause}
    ORDER BY md.chembl_id, a.standard_type, a.standard_units, a.standard_value;
    """

    params: List = [args.target, *units, *types, args.confidence_min]

    t0 = datetime.now()
    print(f"[{t0.isoformat(timespec='seconds')}] Querying {args.target} from {args.db}")

    cur.execute(sql, params)
    rows = cur.fetchall()

    print(f"[INFO] Retrieved {len(rows):,} rows")

    if not rows:
        conn.close()
        print("[WARN] No rows found. Check target ID, units/types, and confidence_min.")
        return 0

    fieldnames = rows[0].keys()

    # Write raw
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))

    # QSAR-ready subset
    qsar_rows = []
    for r in rows:
        try:
            val = float(r["standard_value"])
        except Exception:
            continue

        if r["standard_relation"] != "=":
            continue
        if val <= 0:
            continue
        if r["canonical_smiles"] is None or str(r["canonical_smiles"]).strip() == "":
            continue

        qsar_rows.append(r)

    with open(qsar_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in qsar_rows:
            w.writerow(dict(r))

    conn.close()
    t1 = datetime.now()
    dt = (t1 - t0).total_seconds()

    print(f"[DONE] {args.target} completed in {dt:.1f}s")
    print(f"Raw:        {raw_csv}")
    print(f"QSAR-ready: {qsar_csv}")
    print("Units retained as nM/uM (convert uM → nM later ×1000).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
