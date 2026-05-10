#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 10_screen_03_ensemble_infer_rank.py

Purpose:
    Run multi-seed ensemble inference and rank screening compounds by predicted ACC2 interaction potential.

Workflow step:
    Step 10 - ensemble inference and candidate ranking.

Main inputs:
    - screen/features/screen_graphs.pt
    - screen/features/screen_fp.npy
    - screen/features/screen_index.csv
    - models/gnn_fusion_seed###_best.pt

Main outputs:
    - screen/output/screen_scores_full.csv
    - screen/output/screen_topN.csv
    - screen/output/screen_summary.json
    - screen/output/figures/*.svg|png

Notes:
    Reports ensemble mean and ensemble standard deviation for uncertainty-aware ranking.
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
import torch.nn as nn
import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.pyplot as plt


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
# Minimal scatter ops
# -----------------------------
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.dim() == 1:
        out = torch.zeros((dim_size,), device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        return out
    out = torch.zeros((dim_size, src.size(-1)), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = scatter_sum(src, index, dim_size)
    ones = torch.ones((index.size(0),), device=src.device, dtype=src.dtype)
    cnt = scatter_sum(ones, index, dim_size).clamp(min=1.0).unsqueeze(-1)
    return out / cnt


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    order = torch.argsort(index)
    idx_sorted = index[order]
    src_sorted = src[order]
    out = torch.full((dim_size, src.size(-1)), -1e9, device=src.device, dtype=src.dtype)
    unique_ids = torch.unique_consecutive(idx_sorted)
    for gid in unique_ids.tolist():
        mask = (idx_sorted == gid)
        out[gid] = torch.max(src_sorted[mask], dim=0).values
    return out


# -----------------------------
# Model (FusionNet; consistent with Script-9)
# -----------------------------
@dataclass
class Batch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor
    fp: torch.Tensor
    smiles: List[str]
    screen_id: List[str]
    screen_name: List[str]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GINEBlock(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.msg_mlp = MLP(node_dim + edge_dim, hidden_dim, node_dim, dropout)
        self.upd = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        xj = x[src]
        m = self.msg_mlp(torch.cat([xj, edge_attr], dim=-1))
        m = F.relu(m)
        agg = scatter_sum(m, dst, dim_size=x.size(0))
        return self.upd(x + agg)


class GraphEncoder(nn.Module):
    def __init__(self, node_in: int, edge_in: int, node_dim: int, depth: int, hidden_dim: int, dropout: float, pooling: str):
        super().__init__()
        self.node_proj = nn.Linear(node_in, node_dim)
        self.blocks = nn.ModuleList([GINEBlock(node_dim, edge_in, hidden_dim, dropout) for _ in range(depth)])
        self.dropout = dropout
        self.pooling = pooling
        self.out_dim = node_dim * 2 if pooling == "meanmax" else node_dim

    def pool(self, x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        if self.pooling == "mean":
            return scatter_mean(x, batch, dim_size=num_graphs)
        if self.pooling == "max":
            return scatter_max(x, batch, dim_size=num_graphs)
        if self.pooling == "meanmax":
            return torch.cat(
                [scatter_mean(x, batch, dim_size=num_graphs), scatter_max(x, batch, dim_size=num_graphs)],
                dim=-1,
            )
        raise ValueError("Unknown pooling")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.node_proj(x)
        for blk in self.blocks:
            x = blk(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)
        num_graphs = int(batch.max().item() + 1)
        return self.pool(x, batch, num_graphs)


class FusionNet(nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        fp_bits: int,
        task: str,
        node_dim: int,
        depth: int,
        gnn_hidden: int,
        pooling: str,
        fp_hidden: int,
        fp_dropout: float,
        fusion_hidden: int,
        dropout: float,
        head_hidden: int,
    ):
        super().__init__()
        self.task = task
        self.encoder = GraphEncoder(node_in, edge_in, node_dim, depth, gnn_hidden, dropout, pooling)
        g_dim = self.encoder.out_dim

        self.fp_proj = nn.Sequential(
            nn.LayerNorm(fp_bits),
            nn.Dropout(fp_dropout),
            nn.Linear(fp_bits, fp_hidden),
            nn.ReLU(),
            nn.Dropout(fp_dropout),
            nn.Linear(fp_hidden, fp_hidden),
            nn.ReLU(),
        )

        fusion_in = g_dim + fp_hidden
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Dropout(dropout),
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head_reg = None
        self.head_cls = None
        if task in ("regression", "both"):
            self.head_reg = nn.Sequential(
                nn.LayerNorm(fusion_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
        if task in ("classification", "both"):
            self.head_cls = nn.Sequential(
                nn.LayerNorm(fusion_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )

    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        g = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        f = self.fp_proj(batch.fp)
        z = self.fusion(torch.cat([g, f], dim=-1))
        out = {"embedding": z}
        if self.head_reg is not None:
            out["y_reg"] = self.head_reg(z).squeeze(-1)
        if self.head_cls is not None:
            out["logits"] = self.head_cls(z).squeeze(-1)
        return out


def load_fusion_model(model_path: Path, device: torch.device, task: str) -> FusionNet:
    ckpt = torch.load(model_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    node_in = int(ckpt["node_in"])
    edge_in = int(ckpt["edge_in"])

    model = FusionNet(
        node_in=node_in,
        edge_in=edge_in,
        fp_bits=int(cfg.get("fp_bits", 2048)),
        task=task,
        node_dim=int(cfg.get("node_dim", 256)),
        depth=int(cfg.get("depth", 5)),
        gnn_hidden=int(cfg.get("gnn_hidden", 256)),
        pooling=str(cfg.get("pooling", "meanmax")),
        fp_hidden=int(cfg.get("fp_hidden", 512)),
        fp_dropout=float(cfg.get("fp_dropout", 0.10)),
        fusion_hidden=int(cfg.get("fusion_hidden", 512)),
        dropout=float(cfg.get("dropout", 0.15)),
        head_hidden=int(cfg.get("head_hidden", 256)),
    ).to(device)

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def collate(
    graphs: List[Dict],
    fps: np.ndarray,
    device: torch.device,
    base_index: int = 0,   # NEW
) -> Batch:
    xs, eis, eas, batches = [], [], [], []
    fp_list = []
    smiles = []
    screen_id = []
    screen_name = []

    node_offset = 0
    for gid, g in enumerate(graphs):
        x = g["x"].to(device)
        ei = g["edge_index"].to(device)
        ea = g["edge_attr"].to(device)

        xs.append(x)
        eis.append(ei + node_offset)
        eas.append(ea)
        batches.append(torch.full((x.size(0),), gid, device=device, dtype=torch.long))
        node_offset += x.size(0)

        meta = g.get("meta", {})
        # If row_index is missing, assume graphs and fps are in the same order
        ridx = meta.get("row_index", base_index + gid)
        ridx = int(ridx)

        fp_list.append(torch.tensor(fps[ridx, :], device=device, dtype=torch.float32))

        smi = str(meta.get("smiles", ""))
        smiles.append(smi)
        screen_id.append(str(meta.get("screen_id", ridx)))
        screen_name.append(str(meta.get("screen_name", "")))

    return Batch(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(eis, dim=1),
        edge_attr=torch.cat(eas, dim=0),
        batch=torch.cat(batches, dim=0),
        fp=torch.stack(fp_list, dim=0),
        smiles=smiles,
        screen_id=screen_id,
        screen_name=screen_name,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Ensemble inference and ranking for screening database.")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--task", type=str, choices=["classification", "regression", "both"], default="classification")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--topN", type=int, default=5000)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    feat_dir = out_root / "screen" / "features"
    out_dir = out_root / "screen" / "output"
    fig_dir = out_dir / "figures"
    safe_mkdir(out_dir)
    safe_mkdir(fig_dir)

    style = FigStyle()
    style.apply()

    graphs_path = feat_dir / "screen_graphs.pt"
    fp_path = feat_dir / "screen_fp.npy"
    index_path = feat_dir / "screen_index.csv"

    if not graphs_path.exists():
        raise FileNotFoundError(f"Missing: {graphs_path}")
    if not fp_path.exists():
        raise FileNotFoundError(f"Missing: {fp_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing: {index_path}")

    obj = torch.load(graphs_path, map_location="cpu")
    graphs = obj["graphs"]
    fps = np.load(fp_path).astype(np.float32)
    idx_df = pd.read_csv(index_path)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    # Load ensemble
    models: List[FusionNet] = []
    for s in args.seeds:
        mp = out_root / "models" / f"gnn_fusion_seed{s:03d}_best.pt"
        if not mp.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {mp}")
        models.append(load_fusion_model(mp, device=device, task=args.task))

    # Inference
    rows = []
    n = len(graphs)
    for i in range(0, n, args.batch_size):
        chunk = graphs[i:i + args.batch_size]
        b = collate(chunk, fps=fps, device=device, base_index=i)
        with torch.no_grad():
            # collect per-model outputs
            probs = []
            regs = []
            for m in models:
                out = m(b)
                if args.task in ("classification", "both") and "logits" in out:
                    probs.append(torch.sigmoid(out["logits"]).detach().cpu().numpy())
                if args.task in ("regression", "both") and "y_reg" in out:
                    regs.append(out["y_reg"].detach().cpu().numpy())

        prob_mean = prob_std = None
        reg_mean = reg_std = None
        if len(probs) > 0:
            P = np.stack(probs, axis=0)  # [n_models, batch]
            prob_mean = P.mean(axis=0)
            prob_std = P.std(axis=0, ddof=1) if P.shape[0] > 1 else np.zeros_like(prob_mean)
        if len(regs) > 0:
            R = np.stack(regs, axis=0)
            reg_mean = R.mean(axis=0)
            reg_std = R.std(axis=0, ddof=1) if R.shape[0] > 1 else np.zeros_like(reg_mean)

        for j in range(len(chunk)):
            r = {
                "screen_id": b.screen_id[j],
                "screen_name": b.screen_name[j],
                "canonical_smiles": b.smiles[j],
            }
            if prob_mean is not None:
                r["ens_prob_mean"] = float(prob_mean[j])
                r["ens_prob_std"] = float(prob_std[j])
            if reg_mean is not None:
                r["ens_reg_mean"] = float(reg_mean[j])
                r["ens_reg_std"] = float(reg_std[j])
            rows.append(r)

        if (i + args.batch_size) % (args.batch_size * 50) == 0:
            print(f"Inferred {min(i+args.batch_size, n):,} / {n:,}")

    out_df = pd.DataFrame(rows)

    # Rank
    if args.task in ("classification", "both") and "ens_prob_mean" in out_df.columns:
        out_df = out_df.sort_values(["ens_prob_mean", "ens_prob_std"], ascending=[False, True]).reset_index(drop=True)
        score_col = "ens_prob_mean"
        unc_col = "ens_prob_std"
    elif args.task in ("regression", "both") and "ens_reg_mean" in out_df.columns:
        out_df = out_df.sort_values(["ens_reg_mean", "ens_reg_std"], ascending=[False, True]).reset_index(drop=True)
        score_col = "ens_reg_mean"
        unc_col = "ens_reg_std"
    else:
        raise RuntimeError("No scores produced. Check task and model heads.")

    full_path = out_dir / "screening_scores_full.csv"
    out_df.to_csv(full_path, index=False)

    topN = int(min(args.topN, out_df.shape[0]))
    top_df = out_df.head(topN).copy()
    top_path = out_dir / "screening_topN.csv"
    top_df.to_csv(top_path, index=False)

    # Figures
    # Histogram
    fig = plt.figure(figsize=(7.4, 5.6))
    ax = fig.add_subplot(111)
    ax.hist(out_df[score_col].to_numpy(dtype=float), bins=40, alpha=0.85)
    ax.set_title(f"Screening score distribution ({score_col})", fontweight="bold")
    ax.set_xlabel(score_col, fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for sp in ax.spines.values():
        sp.set_linewidth(style.axis_width)
    _save_fig(fig, fig_dir / "score_histogram.svg", fig_dir / "score_histogram.png", dpi=style.dpi_png)

    # Top hits bar
    k = min(25, top_df.shape[0])
    fig = plt.figure(figsize=(10.5, 6.0))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(k), top_df[score_col].to_numpy(dtype=float)[:k])
    ax.set_title(f"Top {k} hits by {score_col}", fontweight="bold")
    ax.set_xlabel("Rank", fontweight="bold")
    ax.set_ylabel(score_col, fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for sp in ax.spines.values():
        sp.set_linewidth(style.axis_width)
    _save_fig(fig, fig_dir / "top_hits_bar.svg", fig_dir / "top_hits_bar.png", dpi=style.dpi_png)

    # Uncertainty vs score
    if unc_col in out_df.columns:
        fig = plt.figure(figsize=(7.4, 6.2))
        ax = fig.add_subplot(111)
        ax.scatter(out_df[score_col].to_numpy(dtype=float), out_df[unc_col].to_numpy(dtype=float), alpha=0.35)
        ax.set_title("Ensemble uncertainty vs score", fontweight="bold")
        ax.set_xlabel(score_col, fontweight="bold")
        ax.set_ylabel(unc_col, fontweight="bold")
        ax.tick_params(width=style.axis_width)
        for sp in ax.spines.values():
            sp.set_linewidth(style.axis_width)
        _save_fig(fig, fig_dir / "uncertainty_vs_score.svg", fig_dir / "uncertainty_vs_score.png", dpi=style.dpi_png)

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "task": args.task,
        "seeds": [int(s) for s in args.seeds],
        "inputs": {
            "screen_graphs_pt": str(graphs_path),
            "screen_fp_npy": str(fp_path),
            "screen_index_csv": str(index_path),
        },
        "outputs": {
            "scores_full": str(full_path),
            "topN": str(top_path),
            "figures_dir": str(fig_dir),
        },
        "ranking": {
            "score_col": score_col,
            "uncertainty_col": unc_col,
            "topN": topN,
        },
        "counts": {
            "n_scored": int(out_df.shape[0]),
        },
    }
    (out_dir / "screening_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== DONE: Screening inference completed ===")
    print(f"Full scores: {full_path}")
    print(f"TopN:        {top_path}")
    print(f"Figures:     {fig_dir}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
