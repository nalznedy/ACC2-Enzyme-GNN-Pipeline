#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 04_gnn_train_and_evaluate.py

Purpose:
        Train and evaluate graph neural-network models on scaffold-split ACC2 data.

Workflow step:
        Step 4 - GNN model training and evaluation.

Main inputs:
        - features/graphs_seed###.pt

Main outputs:
        - models/gnn_seed###_best.pt
        - predictions/gnn_seed###_{train,val,test}.csv
        - metrics/gnn_metrics_seed###.*
        - tables/gnn_training_history_seed###.csv
        - reports/run_manifest_gnn_seed###.json
        - figures/gnn_*.svg|png

Notes:
        Hyperparameter tuning and threshold selection use training/validation data only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# Matplotlib only
import matplotlib as mpl
import matplotlib.pyplot as plt

# Sklearn metrics for evaluation
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score

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
    models_dir = out_root / "models"
    preds_dir = out_root / "predictions"
    metrics_dir = out_root / "metrics"
    reports_dir = out_root / "reports"
    tables_dir = out_root / "tables"
    figs_dir = out_root / "figures"
    logs_dir = out_root / "logs"
    for d in [models_dir, preds_dir, metrics_dir, reports_dir, tables_dir, figs_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {
        "models": models_dir,
        "predictions": preds_dir,
        "metrics": metrics_dir,
        "reports": reports_dir,
        "tables": tables_dir,
        "figures": figs_dir,
        "logs": logs_dir,
    }


def _save_fig(fig: plt.Figure, out_svg: Path, out_png: Path, dpi: int) -> None:
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Reproducibility
# -----------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Tensor utilities
# -----------------------------
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    src: [E, D] or [E]
    index: [E] values in [0..dim_size-1]
    returns: [dim_size, D] or [dim_size]
    """
    if src.dim() == 1:
        out = torch.zeros((dim_size,), device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        return out
    else:
        out = torch.zeros((dim_size, src.size(-1)), device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = scatter_sum(src, index, dim_size)
    if src.dim() == 1:
        count = scatter_sum(torch.ones_like(index, dtype=src.dtype), index, dim_size).clamp(min=1.0)
        return out / count
    else:
        ones = torch.ones((index.size(0),), device=src.device, dtype=src.dtype)
        count = scatter_sum(ones, index, dim_size).clamp(min=1.0).unsqueeze(-1)
        return out / count


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Simple scatter max. Works for src [N, D].
    """
    out = torch.full((dim_size, src.size(-1)), -1e9, device=src.device, dtype=src.dtype)
    out = out.index_put_((index,), src, accumulate=True)  # accumulate=True sums, not max
    # We need true max; implement with loop-free alternative:
    # Use scatter_sum on exp weights can be unstable. We'll do safe fallback using segment max by sorting.
    # For QSAR sizes, sorting is fine and deterministic.
    order = torch.argsort(index)
    idx_sorted = index[order]
    src_sorted = src[order]

    out = torch.full((dim_size, src.size(-1)), -1e9, device=src.device, dtype=src.dtype)
    # iterate unique segments in a vectorized-ish manner:
    unique_ids = torch.unique_consecutive(idx_sorted)
    for gid in unique_ids.tolist():
        mask = (idx_sorted == gid)
        out[gid] = torch.max(src_sorted[mask], dim=0).values
    return out


# -----------------------------
# Graph batching
# -----------------------------
@dataclass
class Batch:
    x: torch.Tensor              # [N, F]
    edge_index: torch.Tensor     # [2, E]
    edge_attr: torch.Tensor      # [E, Fe]
    batch: torch.Tensor          # [N] graph id for each node
    y_reg: Optional[torch.Tensor]  # [B] float
    y_cls: Optional[torch.Tensor]  # [B] long
    smiles: List[str]
    split: List[str]
    seed: int


def collate_graphs(graphs: List[Dict], device: torch.device, task: str) -> Batch:
    """
    Collate a list of graphs (as saved in cache) into a single batch.
    """
    xs, eis, eas, batches = [], [], [], []
    y_reg_list, y_cls_list = [], []
    smiles, splits = [], []

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

        if task in ("regression", "both"):
            yreg = g.get("y_reg", None)
            y_reg_list.append(torch.tensor(float("nan"), device=device) if yreg is None else yreg.to(device))
        if task in ("classification", "both"):
            ycls = g.get("y_cls", None)
            y_cls_list.append(torch.tensor(-1, device=device, dtype=torch.long) if ycls is None else ycls.to(device))

        meta = g.get("meta", {})
        smiles.append(str(meta.get("smiles", "")))
        splits.append(str(meta.get("split", "")))

    x = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1) if len(eis) else torch.zeros((2, 0), device=device, dtype=torch.long)
    edge_attr = torch.cat(eas, dim=0) if len(eas) else torch.zeros((0, 1), device=device, dtype=torch.float32)
    batch = torch.cat(batches, dim=0)

    y_reg = None
    y_cls = None
    if task in ("regression", "both"):
        y_reg = torch.stack(y_reg_list, dim=0)
    if task in ("classification", "both"):
        y_cls = torch.stack(y_cls_list, dim=0)

    seed = int(graphs[0].get("meta", {}).get("seed", -1)) if graphs else -1
    return Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, y_reg=y_reg, y_cls=y_cls, smiles=smiles, split=splits, seed=seed)


# -----------------------------
# Model: Edge-aware message passing (GINE-like)
# -----------------------------
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
    """
    GINE-like block:
      m_ij = MLP_edge(e_ij) + x_j
      agg_i = sum_{j in N(i)} ReLU(MLP_msg([x_j, e_ij]))
      x_i' = MLP_upd(x_i + agg_i)
    """
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
        src = edge_index[0]  # j
        dst = edge_index[1]  # i

        xj = x[src]
        m_in = torch.cat([xj, edge_attr], dim=-1)
        m = self.msg_mlp(m_in)
        m = F.relu(m)
        agg = scatter_sum(m, dst, dim_size=x.size(0))
        out = x + agg
        out = self.upd(out)
        return out


class GraphEncoder(nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        node_dim: int,
        depth: int,
        hidden_dim: int,
        dropout: float,
        pooling: str = "meanmax",
    ):
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
            m1 = scatter_mean(x, batch, dim_size=num_graphs)
            m2 = scatter_max(x, batch, dim_size=num_graphs)
            return torch.cat([m1, m2], dim=-1)
        raise ValueError("Unknown pooling")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.node_proj(x)
        for blk in self.blocks:
            x = blk(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)
        num_graphs = int(batch.max().item() + 1) if batch.numel() else 0
        g = self.pool(x, batch, num_graphs=num_graphs)
        return g


class QSARNet(nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        node_dim: int,
        depth: int,
        hidden_dim: int,
        dropout: float,
        pooling: str,
        task: str,
        head_hidden: int,
    ):
        super().__init__()
        self.task = task
        self.encoder = GraphEncoder(
            node_in=node_in,
            edge_in=edge_in,
            node_dim=node_dim,
            depth=depth,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pooling=pooling,
        )
        enc_out = self.encoder.out_dim

        self.head_reg = None
        self.head_cls = None

        if task in ("regression", "both"):
            self.head_reg = nn.Sequential(
                nn.LayerNorm(enc_out),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(enc_out, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
        if task in ("classification", "both"):
            self.head_cls = nn.Sequential(
                nn.LayerNorm(enc_out),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(enc_out, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),  # logits
            )

    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        g = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        out = {}
        if self.head_reg is not None:
            out["y_reg"] = self.head_reg(g).squeeze(-1)
        if self.head_cls is not None:
            out["logits"] = self.head_cls(g).squeeze(-1)
        return out


# -----------------------------
# Losses / evaluation
# -----------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else float("nan")
    ranks_true = pd.Series(y_true).rank(method="average").values
    ranks_pred = pd.Series(y_pred).rank(method="average").values
    spearman = float(np.corrcoef(ranks_true, ranks_pred)[0, 1]) if y_true.size > 1 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson_r": pearson, "spearman_rho": spearman}


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {}
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["pr_auc"] = float("nan")
    if len(np.unique(y_true)) > 1:
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        out["bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    else:
        out["mcc"] = float("nan")
        out["bal_acc"] = float("nan")
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    try:
        out["brier"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        out["brier"] = float("nan")
    return out


def select_threshold_by_mcc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.unique(np.clip(y_prob, 0, 1))
    grid = np.linspace(0.0, 1.0, 1001)
    thresholds = np.unique(np.concatenate([thresholds, grid]))
    best_t = 0.5
    best_mcc = -1.0
    if len(np.unique(y_true)) < 2:
        return 0.5
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        mcc = matthews_corrcoef(y_true, pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_t = float(t)
    return float(best_t)


# -----------------------------
# Data loading from cache
# -----------------------------
def load_graph_cache(path: Path) -> Dict:
    obj = torch.load(path, map_location="cpu")
    if "graphs" not in obj:
        raise ValueError("Invalid cache: missing 'graphs'")
    return obj


def split_graphs(graphs: List[Dict]) -> Dict[str, List[Dict]]:
    out = {"train": [], "val": [], "test": []}
    for g in graphs:
        sp = str(g.get("meta", {}).get("split", ""))
        if sp not in out:
            raise ValueError(f"Unknown split '{sp}' in graph meta")
        out[sp].append(g)
    return out


# -----------------------------
# Training helpers
# -----------------------------
@dataclass
class TrainConfig:
    # architecture
    node_dim: int = 256
    depth: int = 5
    hidden_dim: int = 256
    head_hidden: int = 256
    pooling: str = "meanmax"

    # optimization
    lr: float = 3e-4
    weight_decay: float = 1e-5
    dropout: float = 0.15
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 25
    grad_clip: float = 2.0

    # task
    task: str = "both"
    reg_loss: str = "huber"       # huber is stable
    huber_delta: float = 1.0
    cls_pos_weight: float = 1.0   # computed from train if auto_pos_weight

    # tuning
    tune_trials: int = 20
    tune_seed: int = 2025
    auto_pos_weight: bool = True

    # reporting
    eval_every: int = 1


def make_minibatches(items: List[Dict], batch_size: int, rng: np.random.RandomState, shuffle: bool) -> List[List[Dict]]:
    idx = np.arange(len(items))
    if shuffle:
        rng.shuffle(idx)
    batches = []
    for i in range(0, len(items), batch_size):
        sel = idx[i:i + batch_size]
        batches.append([items[j] for j in sel.tolist()])
    return batches


def compute_pos_weight(train_graphs: List[Dict]) -> float:
    ys = []
    for g in train_graphs:
        y = g.get("y_cls", None)
        if y is None:
            continue
        ys.append(int(y))
    ys = np.asarray(ys, dtype=int)
    if ys.size == 0:
        return 1.0
    pos = float((ys == 1).sum())
    neg = float((ys == 0).sum())
    if pos <= 0:
        return 1.0
    return float(neg / pos)


def regression_loss_fn(pred: torch.Tensor, target: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    pred = pred[mask]
    target = target[mask]
    if cfg.reg_loss == "mse":
        return F.mse_loss(pred, target)
    # huber
    return F.huber_loss(pred, target, delta=cfg.huber_delta)


def classification_loss_fn(logits: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    # target: -1 indicates missing
    mask = target >= 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    logits = logits[mask]
    target = target[mask].float()
    pw = torch.tensor(pos_weight, device=logits.device, dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)


@torch.no_grad()
def predict_split(model: nn.Module, graphs: List[Dict], device: torch.device, task: str, batch_size: int) -> Dict[str, np.ndarray]:
    model.eval()
    rng = np.random.RandomState(0)
    batches = make_minibatches(graphs, batch_size=batch_size, rng=rng, shuffle=False)

    y_reg_true, y_reg_pred = [], []
    y_cls_true, y_cls_prob = [], []
    smiles = []

    for b in batches:
        batch = collate_graphs(b, device=device, task=task)
        out = model(batch)
        smiles.extend(batch.smiles)

        if task in ("regression", "both") and "y_reg" in out:
            y_true = batch.y_reg.detach().cpu().numpy()
            y_pred = out["y_reg"].detach().cpu().numpy()
            y_reg_true.append(y_true)
            y_reg_pred.append(y_pred)

        if task in ("classification", "both") and "logits" in out:
            y_true = batch.y_cls.detach().cpu().numpy()
            prob = torch.sigmoid(out["logits"]).detach().cpu().numpy()
            y_cls_true.append(y_true)
            y_cls_prob.append(prob)

    res = {"smiles": np.array(smiles, dtype=object)}
    if task in ("regression", "both"):
        res["y_reg_true"] = np.concatenate(y_reg_true) if y_reg_true else np.array([], dtype=float)
        res["y_reg_pred"] = np.concatenate(y_reg_pred) if y_reg_pred else np.array([], dtype=float)
    if task in ("classification", "both"):
        res["y_cls_true"] = np.concatenate(y_cls_true) if y_cls_true else np.array([], dtype=int)
        res["y_cls_prob"] = np.concatenate(y_cls_prob) if y_cls_prob else np.array([], dtype=float)
    return res


def compute_objective(val_pred: Dict[str, np.ndarray], task: str) -> float:
    """
    Objective for tuning (higher is better):
      - regression: negative RMSE
      - classification: PR-AUC (robust for imbalance)
      - both: average of normalized components (PR-AUC + (-RMSE scaled))
    """
    if task == "regression":
        y_true = val_pred["y_reg_true"]
        y_pred = val_pred["y_reg_pred"]
        mask = ~np.isnan(y_true)
        if mask.sum() == 0:
            return -1e9
        rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        return -float(rmse)

    if task == "classification":
        y_true = val_pred["y_cls_true"]
        prob = val_pred["y_cls_prob"]
        mask = y_true >= 0
        if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
            return -1e9
        return float(average_precision_score(y_true[mask], prob[mask]))

    # both
    score = 0.0
    # classification component
    y_true_c = val_pred["y_cls_true"]
    prob = val_pred["y_cls_prob"]
    mask_c = y_true_c >= 0
    if mask_c.sum() > 0 and len(np.unique(y_true_c[mask_c])) >= 2:
        score += float(average_precision_score(y_true_c[mask_c], prob[mask_c]))
    else:
        score += 0.0

    # regression component scaled
    y_true_r = val_pred["y_reg_true"]
    y_pred_r = val_pred["y_reg_pred"]
    mask_r = ~np.isnan(y_true_r)
    if mask_r.sum() > 0:
        rmse = float(np.sqrt(mean_squared_error(y_true_r[mask_r], y_pred_r[mask_r])))
        score += float(-rmse / 2.0)  # scale for pIC50 typical range
    else:
        score += -1.0
    return float(score / 2.0)


# -----------------------------
# Plotting (publication style)
# -----------------------------
def plot_training_curves(hist: pd.DataFrame, figs_dir: Path, style: FigStyle, prefix: str) -> None:
    # Loss
    fig = plt.figure(figsize=(8.2, 5.6))
    ax = fig.add_subplot(111)
    ax.plot(hist["epoch"], hist["train_loss"], marker="o", alpha=0.9, label="Train")
    ax.plot(hist["epoch"], hist["val_loss"], marker="o", alpha=0.9, label="Validation")
    ax.set_title("Training Curves: Loss", fontweight="bold")
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")
    ax.legend(frameon=False)
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)
    _save_fig(fig, figs_dir / f"{prefix}_loss.svg", figs_dir / f"{prefix}_loss.png", dpi=style.dpi_png)

    # Objective
    fig = plt.figure(figsize=(8.2, 5.6))
    ax = fig.add_subplot(111)
    ax.plot(hist["epoch"], hist["val_objective"], marker="o", label="Val objective")
    ax.set_title("Validation Objective Over Epochs", fontweight="bold")
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Objective", fontweight="bold")
    ax.legend(frameon=False)
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)
    _save_fig(fig, figs_dir / f"{prefix}_objective.svg", figs_dir / f"{prefix}_objective.png", dpi=style.dpi_png)


def plot_roc_pr_multi(curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
                      title_prefix: str,
                      out_svg: Path,
                      out_png: Path,
                      style: FigStyle,
                      kind: str = "roc") -> None:
    """
    curves: dict like {"Train": (y_true, y_prob), "Val": (...), "Test": (...)}
    kind: "roc" or "pr"
    """
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)

    for name, (y_true, y_prob) in curves.items():
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        mask = (y_true >= 0)
        y_true = y_true[mask].astype(int)
        y_prob = y_prob[mask].astype(float)

        if y_true.size == 0 or len(np.unique(y_true)) < 2:
            continue

        if kind == "roc":
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        else:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")

    if kind == "roc":
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set_title(f"{title_prefix}: ROC (Train/Val/Test)", fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontweight="bold")
    else:
        ax.set_title(f"{title_prefix}: PR (Train/Val/Test)", fontweight="bold")
        ax.set_xlabel("Recall", fontweight="bold")
        ax.set_ylabel("Precision", fontweight="bold")

    ax.legend(frameon=False)
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)

    _save_fig(fig, out_svg, out_png, dpi=style.dpi_png)


def plot_regression_figures(y_true: np.ndarray, y_pred: np.ndarray, figs_dir: Path, style: FigStyle, prefix: str, title_prefix: str) -> None:
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return

    # Pred vs obs
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, alpha=0.6)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    ax.plot([mn, mx], [mn, mx])
    ax.set_title(f"{title_prefix}: Predicted vs Observed", fontweight="bold")
    ax.set_xlabel("Observed", fontweight="bold")
    ax.set_ylabel("Predicted", fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)
    _save_fig(fig, figs_dir / f"{prefix}_pred_vs_obs.svg", figs_dir / f"{prefix}_pred_vs_obs.png", dpi=style.dpi_png)

    # Residuals
    resid = y_true - y_pred
    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_subplot(111)
    ax.hist(resid, bins=30)
    ax.set_title(f"{title_prefix}: Residuals", fontweight="bold")
    ax.set_xlabel("Residual (Observed - Predicted)", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)
    _save_fig(fig, figs_dir / f"{prefix}_residuals.svg", figs_dir / f"{prefix}_residuals.png", dpi=style.dpi_png)


def plot_classification_figures(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, figs_dir: Path, style: FigStyle, prefix: str, title_prefix: str) -> None:
    mask = y_true >= 0
    y_true = y_true[mask].astype(int)
    y_prob = y_prob[mask].astype(float)
    if y_true.size == 0:
        return

    # ROC
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")  # AUC in legend
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set_title(f"{title_prefix}: ROC Curve", fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontweight="bold")
        ax.legend(frameon=False)
    except Exception:
        ax.text(0.05, 0.5, "ROC not defined (single-class)", fontweight="bold")
        ax.set_axis_off()
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)
    _save_fig(fig, figs_dir / f"{prefix}_roc.svg", figs_dir / f"{prefix}_roc.png", dpi=style.dpi_png)

    # PR
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f"PR (AP={ap:.3f})")
        ax.set_title(f"{title_prefix}: Precision–Recall Curve", fontweight="bold")
        ax.set_xlabel("Recall", fontweight="bold")
        ax.set_ylabel("Precision", fontweight="bold")
        ax.legend(frameon=False)
    except Exception:
        ax.text(0.05, 0.5, "PR not defined (single-class)", fontweight="bold")
        ax.set_axis_off()
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)
    _save_fig(fig, figs_dir / f"{prefix}_pr.svg", figs_dir / f"{prefix}_pr.png", dpi=style.dpi_png)

    # Calibration
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", label="Model")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
        ax.set_title(f"{title_prefix}: Calibration", fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
        ax.set_ylabel("Fraction of Positives", fontweight="bold")
        ax.legend(frameon=False)
    except Exception:
        ax.text(0.05, 0.5, "Calibration not defined", fontweight="bold")
        ax.set_axis_off()
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)
    _save_fig(fig, figs_dir / f"{prefix}_calibration.svg", figs_dir / f"{prefix}_calibration.png", dpi=style.dpi_png)

    # Confusion matrix
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = plt.figure(figsize=(6.2, 5.6))
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"{title_prefix}: Confusion (thr={threshold:.3f})", fontweight="bold")
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Observed", fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)
    _save_fig(fig, figs_dir / f"{prefix}_confusion.svg", figs_dir / f"{prefix}_confusion.png", dpi=style.dpi_png)


# -----------------------------
# Hyperparameter tuning (train/val only)
# -----------------------------
def sample_config(base: TrainConfig, rng: np.random.RandomState) -> TrainConfig:
    cfg = TrainConfig(**asdict(base))
    # log-uniform LR and weight decay
    cfg.lr = float(10 ** rng.uniform(np.log10(1e-4), np.log10(3e-3)))
    cfg.weight_decay = float(10 ** rng.uniform(np.log10(1e-7), np.log10(3e-4)))
    cfg.dropout = float(rng.uniform(0.05, 0.35))
    cfg.node_dim = int(rng.choice([128, 192, 256, 320]))
    cfg.hidden_dim = int(rng.choice([128, 192, 256, 320]))
    cfg.head_hidden = int(rng.choice([128, 192, 256, 320]))
    cfg.depth = int(rng.choice([3, 4, 5, 6, 7]))
    cfg.batch_size = int(rng.choice([32, 64, 96, 128]))
    cfg.reg_loss = str(rng.choice(["huber", "mse"]))
    cfg.huber_delta = float(rng.choice([0.5, 1.0, 1.5]))
    cfg.grad_clip = float(rng.choice([1.0, 2.0, 5.0]))
    cfg.patience = int(rng.choice([15, 20, 25, 30]))
    cfg.max_epochs = int(rng.choice([120, 160, 200, 240]))
    cfg.pooling = str(rng.choice(["meanmax", "mean", "max"]))
    return cfg


def train_one(
    graphs_train: List[Dict],
    graphs_val: List[Dict],
    node_in: int,
    edge_in: int,
    cfg: TrainConfig,
    device: torch.device,
    run_seed: int,
) -> Tuple[nn.Module, pd.DataFrame, Dict[str, float], Dict[str, np.ndarray], float]:
    """
    Returns:
      - best_model (weights restored)
      - history dataframe
      - best_val_metrics (dict)
      - best_val_predictions (dict arrays)
      - best_objective
    """
    set_all_seeds(run_seed)

    # class weight from train only (prevents leakage)
    pos_weight = 1.0
    if cfg.task in ("classification", "both") and cfg.auto_pos_weight:
        pos_weight = compute_pos_weight(graphs_train)

    model = QSARNet(
        node_in=node_in,
        edge_in=edge_in,
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
        task=cfg.task,
        head_hidden=cfg.head_hidden,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Early stopping
    best_obj = -1e18
    best_state = None
    best_epoch = -1
    wait = 0

    hist_rows = []
    rng = np.random.RandomState(run_seed + 123)

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        batches = make_minibatches(graphs_train, cfg.batch_size, rng=rng, shuffle=True)

        train_losses = []
        for b in batches:
            batch = collate_graphs(b, device=device, task=cfg.task)
            out = model(batch)

            loss = torch.tensor(0.0, device=device)
            if cfg.task in ("regression", "both") and "y_reg" in out:
                loss = loss + regression_loss_fn(out["y_reg"], batch.y_reg, cfg)
            if cfg.task in ("classification", "both") and "logits" in out:
                loss = loss + classification_loss_fn(out["logits"], batch.y_cls, pos_weight=pos_weight)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            opt.step()

            train_losses.append(float(loss.detach().cpu().item()))

        # Validation evaluation each epoch
        if epoch % cfg.eval_every == 0:
            val_pred = predict_split(model, graphs_val, device=device, task=cfg.task, batch_size=cfg.batch_size)
            val_obj = compute_objective(val_pred, cfg.task)

            # val loss (same as training loss definition, computed in eval mode)
            model.eval()
            with torch.no_grad():
                vbatches = make_minibatches(graphs_val, cfg.batch_size, rng=np.random.RandomState(0), shuffle=False)
                vlosses = []
                for b in vbatches:
                    batch = collate_graphs(b, device=device, task=cfg.task)
                    out = model(batch)
                    loss = torch.tensor(0.0, device=device)
                    if cfg.task in ("regression", "both") and "y_reg" in out:
                        loss = loss + regression_loss_fn(out["y_reg"], batch.y_reg, cfg)
                    if cfg.task in ("classification", "both") and "logits" in out:
                        loss = loss + classification_loss_fn(out["logits"], batch.y_cls, pos_weight=pos_weight)
                    vlosses.append(float(loss.detach().cpu().item()))
                val_loss = float(np.mean(vlosses)) if vlosses else float("nan")

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

            hist_rows.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_objective": float(val_obj),
            })

            # Early stopping check
            if val_obj > best_obj:
                best_obj = float(val_obj)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= cfg.patience:
                    break

    # Restore best
    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    # Compute best val metrics and lock threshold on val for classification
    val_pred = predict_split(model, graphs_val, device=device, task=cfg.task, batch_size=cfg.batch_size)

    best_val_metrics = {}
    if cfg.task in ("regression", "both"):
        y_true = val_pred["y_reg_true"]
        y_pred = val_pred["y_reg_pred"]
        mask = ~np.isnan(y_true)
        if mask.sum() > 0:
            best_val_metrics.update({f"val_{k}": v for k, v in regression_metrics(y_true[mask], y_pred[mask]).items()})

    thr = None
    if cfg.task in ("classification", "both"):
        y_true = val_pred["y_cls_true"]
        y_prob = val_pred["y_cls_prob"]
        mask = y_true >= 0
        if mask.sum() > 0 and len(np.unique(y_true[mask])) >= 2:
            thr = select_threshold_by_mcc(y_true[mask], y_prob[mask])
            best_val_metrics.update({f"val_{k}": v for k, v in classification_metrics(y_true[mask], y_prob[mask], thr).items()})
            best_val_metrics["val_threshold"] = float(thr)
        else:
            best_val_metrics["val_threshold"] = 0.5

    history = pd.DataFrame(hist_rows)
    best_val_metrics["best_epoch"] = int(best_epoch)
    best_val_metrics["best_objective"] = float(best_obj)
    best_val_metrics["pos_weight_used"] = float(pos_weight)
    return model, history, best_val_metrics, val_pred, float(best_obj)


# -----------------------------
# Main evaluation (train/val/test)
# -----------------------------
def save_predictions_csv(
    out_path: Path,
    seed: int,
    split: str,
    smiles: np.ndarray,
    task: str,
    y_reg_true: Optional[np.ndarray],
    y_reg_pred: Optional[np.ndarray],
    y_cls_true: Optional[np.ndarray],
    y_cls_prob: Optional[np.ndarray],
    threshold: Optional[float],
) -> None:
    df = pd.DataFrame({"seed": seed, "split": split, "canonical_smiles": smiles})
    if task in ("regression", "both") and y_reg_true is not None:
        df["y_reg_true"] = y_reg_true
        df["y_reg_pred"] = y_reg_pred
    if task in ("classification", "both") and y_cls_true is not None:
        df["y_cls_true"] = y_cls_true
        df["y_cls_prob"] = y_cls_prob
        if threshold is not None:
            df["y_cls_pred"] = (y_cls_prob >= threshold).astype(int)
    df.to_csv(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate production GNN for QSAR.")
    ap.add_argument("--graph_cache", type=str, required=True, help="Script-3 output: features/graphs_seed###.pt")
    ap.add_argument("--out_root", type=str, required=True, help="Project root directory.")
    ap.add_argument("--task", type=str, choices=["regression", "classification", "both"], default="both")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--tune_trials", type=int, default=20, help="Number of hyperparameter trials (train/val only).")
    ap.add_argument("--tune_seed", type=int, default=2025, help="Seed for tuning reproducibility.")
    ap.add_argument("--final_seed_offset", type=int, default=777, help="Seed offset for final training run.")
    ap.add_argument("--no_tuning", action="store_true", help="Skip tuning and train only default config.")
    args = ap.parse_args()

    graph_cache_path = Path(args.graph_cache).resolve()
    out_root = Path(args.out_root).resolve()
    dirs = ensure_dirs(out_root)

    style = FigStyle()
    style.apply()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    cache = load_graph_cache(graph_cache_path)
    graphs = cache["graphs"]
    seed = int(cache.get("seed", -1))

    # Split graphs (leakage control: fixed split)
    split_map = split_graphs(graphs)
    graphs_train = split_map["train"]
    graphs_val = split_map["val"]
    graphs_test = split_map["test"]

    if len(graphs_train) == 0 or len(graphs_val) == 0 or len(graphs_test) == 0:
        raise RuntimeError("One of the splits is empty. Check your scaffold split generation.")

    # Determine feature dims
    node_in = int(graphs_train[0]["x"].shape[1])
    edge_in = int(graphs_train[0]["edge_attr"].shape[1]) if graphs_train[0]["edge_attr"].numel() else int(graphs[0]["edge_attr"].shape[1])

    # Base config
    base = TrainConfig(task=args.task, tune_trials=int(args.tune_trials), tune_seed=int(args.tune_seed))

    # Tuning (train/val only)
    tuning_rows = []
    best_cfg = None
    best_obj = -1e18
    best_hist = None
    best_model = None
    best_val_metrics = None

    rng = np.random.RandomState(base.tune_seed)

    trials = 1 if args.no_tuning else base.tune_trials
    for t in range(trials):
        cfg = base if args.no_tuning else sample_config(base, rng)
        run_seed = int(base.tune_seed + 1000 + t)

        model, history, val_metrics, _, obj = train_one(
            graphs_train, graphs_val, node_in=node_in, edge_in=edge_in, cfg=cfg, device=device, run_seed=run_seed
        )

        row = {"trial": t, "objective": obj, **asdict(cfg), **val_metrics}
        tuning_rows.append(row)

        if obj > best_obj:
            best_obj = obj
            best_cfg = cfg
            best_hist = history
            best_model = model
            best_val_metrics = val_metrics

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_csv = dirs["tables"] / f"gnn_tuning_seed{seed:03d}.csv"
    tuning_df.to_csv(tuning_csv, index=False)

    # Save best config JSON
    best_cfg_json = dirs["reports"] / f"gnn_best_config_seed{seed:03d}.json"
    with open(best_cfg_json, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "best_objective": best_obj, "config": asdict(best_cfg), "val_metrics": best_val_metrics}, f, indent=2)

    # Train curves CSV
    hist_csv = dirs["tables"] / f"gnn_training_history_seed{seed:03d}.csv"
    best_hist.to_csv(hist_csv, index=False)

    # Plot training curves
    prefix = f"17_gnn_seed{seed:03d}"
    plot_training_curves(best_hist, dirs["figures"], style, prefix=prefix)

    # Final evaluation on train/val/test using best_model (already best weights by val objective)
    # For classification, lock threshold on val predictions only
    val_pred = predict_split(best_model, graphs_val, device=device, task=args.task, batch_size=best_cfg.batch_size)
    threshold = None
    if args.task in ("classification", "both"):
        y_true = val_pred["y_cls_true"]
        y_prob = val_pred["y_cls_prob"]
        mask = y_true >= 0
        if mask.sum() > 0 and len(np.unique(y_true[mask])) >= 2:
            threshold = select_threshold_by_mcc(y_true[mask], y_prob[mask])
        else:
            threshold = 0.5

    # ---- Collect predictions for ROC/PR overlays (Train/Val/Test in one figure) ----
    pred_by_split: Dict[str, Dict[str, np.ndarray]] = {}
    for sp_name, g_list in [("train", graphs_train), ("val", graphs_val), ("test", graphs_test)]:
        pred_by_split[sp_name] = predict_split(
            best_model,
            g_list,
            device=device,
            task=args.task,
            batch_size=best_cfg.batch_size,
        )

    if args.task in ("classification", "both"):
        curves = {
            "Train": (pred_by_split["train"]["y_cls_true"], pred_by_split["train"]["y_cls_prob"]),
            "Val":   (pred_by_split["val"]["y_cls_true"],   pred_by_split["val"]["y_cls_prob"]),
            "Test":  (pred_by_split["test"]["y_cls_true"],  pred_by_split["test"]["y_cls_prob"]),
        }

        plot_roc_pr_multi(
            curves=curves,
            title_prefix="GNN Classification",
            out_svg=dirs["figures"] / f"19_gnn_cls_seed{seed:03d}_roc_train_val_test.svg",
            out_png=dirs["figures"] / f"19_gnn_cls_seed{seed:03d}_roc_train_val_test.png",
            style=style,
            kind="roc",
        )

        plot_roc_pr_multi(
            curves=curves,
            title_prefix="GNN Classification",
            out_svg=dirs["figures"] / f"19_gnn_cls_seed{seed:03d}_pr_train_val_test.svg",
            out_png=dirs["figures"] / f"19_gnn_cls_seed{seed:03d}_pr_train_val_test.png",
            style=style,
            kind="pr",
        )

    # Predictions per split
    for sp_name, g_list in [("train", graphs_train), ("val", graphs_val), ("test", graphs_test)]:
        pred = predict_split(best_model, g_list, device=device, task=args.task, batch_size=best_cfg.batch_size)
        out_p = dirs["predictions"] / f"gnn_seed{seed:03d}_{sp_name}.csv"
        save_predictions_csv(
            out_p,
            seed=seed,
            split=sp_name,
            smiles=pred["smiles"],
            task=args.task,
            y_reg_true=pred.get("y_reg_true", None),
            y_reg_pred=pred.get("y_reg_pred", None),
            y_cls_true=pred.get("y_cls_true", None),
            y_cls_prob=pred.get("y_cls_prob", None),
            threshold=threshold,
        )

    # Compute metrics for each split (report only; no tuning on test)
    metrics_rows = []
    for sp_name, g_list in [("train", graphs_train), ("val", graphs_val), ("test", graphs_test)]:
        pred = predict_split(best_model, g_list, device=device, task=args.task, batch_size=best_cfg.batch_size)
        row = {"seed": seed, "split": sp_name, "model": "GNN", "task": args.task}

        if args.task in ("regression", "both"):
            y_true = pred["y_reg_true"]
            y_hat = pred["y_reg_pred"]
            mask = ~np.isnan(y_true)
            if mask.sum() > 0:
                row.update(regression_metrics(y_true[mask], y_hat[mask]))

        if args.task in ("classification", "both"):
            y_true = pred["y_cls_true"]
            y_prob = pred["y_cls_prob"]
            mask = y_true >= 0
            if mask.sum() > 0 and len(np.unique(y_true[mask])) >= 2:
                row["threshold"] = float(threshold)
                row.update(classification_metrics(y_true[mask], y_prob[mask], threshold))
            else:
                row["threshold"] = float(threshold)

        metrics_rows.append(row)

        # Figures only for TEST (paper-quality)
        if sp_name == "test":
            if args.task in ("regression", "both"):
                plot_regression_figures(
                    y_true=pred["y_reg_true"], y_pred=pred["y_reg_pred"],
                    figs_dir=dirs["figures"], style=style,
                    prefix=f"18_gnn_reg_seed{seed:03d}_test",
                    title_prefix="GNN Regression (Test)"
                )
            if args.task in ("classification", "both"):
                plot_classification_figures(
                    y_true=pred["y_cls_true"], y_prob=pred["y_cls_prob"], threshold=float(threshold),
                    figs_dir=dirs["figures"], style=style,
                    prefix=f"19_gnn_cls_seed{seed:03d}_test",
                    title_prefix="GNN Classification (Test)"
                )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = dirs["metrics"] / f"gnn_metrics_seed{seed:03d}.csv"
    metrics_json = dirs["metrics"] / f"gnn_metrics_seed{seed:03d}.json"
    metrics_df.to_csv(metrics_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "best_config": asdict(best_cfg), "threshold": threshold, "metrics": metrics_rows}, f, indent=2)

    # Save best model weights
    model_path = dirs["models"] / f"gnn_seed{seed:03d}_best.pt"
    torch.save({"state_dict": best_model.state_dict(), "config": asdict(best_cfg), "seed": seed, "node_in": node_in, "edge_in": edge_in}, model_path)

    # Final run manifest for reproducibility
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "inputs": {"graph_cache": str(graph_cache_path)},
        "outputs": {
            "best_model": str(model_path),
            "tuning_csv": str(tuning_csv),
            "best_config_json": str(best_cfg_json),
            "training_history_csv": str(hist_csv),
            "metrics_csv": str(metrics_csv),
            "metrics_json": str(metrics_json),
            "predictions_dir": str(dirs["predictions"]),
            "figures_dir": str(dirs["figures"]),
        },
        "protocol": {
            "leakage_control": "Scaffold split fixed; tuning uses train/val only; threshold locked on val; test never used for tuning.",
            "overfitting_control": "Early stopping on val objective; dropout; weight decay; gradient clipping; limited depth/width search.",
            "tuning": {"trials": int(trials), "tune_seed": int(args.tune_seed), "objective": "PR-AUC for classification; -RMSE for regression; combined for both."},
        },
        "environment": {
            "torch_version": torch.__version__,
            "device_used": str(device),
        },
    }
    manifest_path = dirs["reports"] / f"run_manifest_gnn_seed{seed:03d}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== DONE: GNN training + evaluation ===")
    print(f"Seed:              {seed}")
    print(f"Best model:        {model_path}")
    print(f"Tuning table:      {tuning_csv}")
    print(f"Best config:       {best_cfg_json}")
    print(f"Training history:  {hist_csv}")
    print(f"Metrics:           {metrics_csv}")
    print(f"Predictions dir:   {dirs['predictions']}")
    print(f"Figures (SVG):     {dirs['figures']}")
    print(f"Manifest:          {manifest_path}")
    print("======================================\n")


if __name__ == "__main__":
    main()
