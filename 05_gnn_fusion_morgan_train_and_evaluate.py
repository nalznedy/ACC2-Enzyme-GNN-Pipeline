#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 05_gnn_fusion_morgan_train_and_evaluate.py

Purpose:
        Train graph-Morgan fusion models for joint ACC2 pIC50 regression and activity classification.

Workflow step:
        Step 5 - fusion model training and evaluation.

Main inputs:
        - features/graphs_seed###.pt
        - data/03_dedup.csv

Main outputs:
        - models/gnn_fusion_seed###_best.pt
        - predictions/gnn_fusion_seed###_{train,val,test}.csv
        - metrics/gnn_fusion_metrics_seed###.*
        - tables/gnn_fusion_tuning_seed###.csv
        - tables/gnn_fusion_training_history_seed###.csv
        - reports/run_manifest_gnn_fusion_seed###.json
        - figures/gnn_fusion_*.svg|png

Notes:
        Includes calibration diagnostics and validation-only threshold selection.
"""

from __future__ import annotations

import argparse
import json
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

# RDKit for Morgan
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Matplotlib only
import matplotlib as mpl
import matplotlib.pyplot as plt

# Sklearn metrics
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
# EMA (Exponential Moving Average)
# -----------------------------
class EMA:
    """
    Maintains an EMA of model parameters.
    Use EMA weights for validation selection and final evaluation to reduce overfitting/variance.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999, start_step: int = 0):
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        if step < self.start_step:
            return
        self.num_updates += 1
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.backup:
                continue
            p.data.copy_(self.backup[name].data)
        self.backup = {}

    def state_dict(self) -> Dict[str, object]:
        return {"decay": self.decay, "start_step": self.start_step, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, d: Dict[str, object]) -> None:
        self.decay = float(d.get("decay", self.decay))
        self.start_step = int(d.get("start_step", self.start_step))
        shadow = d.get("shadow", {})
        if isinstance(shadow, dict):
            self.shadow = {k: v.clone() for k, v in shadow.items()}


# -----------------------------
# Scatter ops (self-contained)
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
# Batch object
# -----------------------------
@dataclass
class Batch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor
    fp: torch.Tensor                 # [B, fp_bits]
    y_reg: Optional[torch.Tensor]
    y_cls: Optional[torch.Tensor]
    smiles: List[str]
    split: List[str]
    seed: int


def collate_graphs_with_fp(graphs: List[Dict], fp_map: Dict[str, np.ndarray], device: torch.device, task: str) -> Batch:
    xs, eis, eas, batches = [], [], [], []
    fps = []
    y_reg_list, y_cls_list = [], []
    smiles, splits = [], []

    node_offset = 0
    for gid, g in enumerate(graphs):
        meta = g.get("meta", {})
        smi = str(meta.get("smiles", ""))
        if smi not in fp_map:
            raise RuntimeError(f"Missing fingerprint for SMILES in fp_map: {smi}")

        x = g["x"].to(device)
        ei = g["edge_index"].to(device)
        ea = g["edge_attr"].to(device)

        xs.append(x)
        eis.append(ei + node_offset)
        eas.append(ea)
        batches.append(torch.full((x.size(0),), gid, device=device, dtype=torch.long))
        node_offset += x.size(0)

        fps.append(torch.tensor(fp_map[smi], device=device, dtype=torch.float32))

        if task in ("regression", "both"):
            yreg = g.get("y_reg", None)
            y_reg_list.append(torch.tensor(float("nan"), device=device) if yreg is None else yreg.to(device))
        if task in ("classification", "both"):
            ycls = g.get("y_cls", None)
            y_cls_list.append(torch.tensor(-1, device=device, dtype=torch.long) if ycls is None else ycls.to(device))

        smiles.append(smi)
        splits.append(str(meta.get("split", "")))

    x = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1) if len(eis) else torch.zeros((2, 0), device=device, dtype=torch.long)
    edge_attr = torch.cat(eas, dim=0) if len(eas) else torch.zeros((0, 1), device=device, dtype=torch.float32)
    batch = torch.cat(batches, dim=0)
    fp = torch.stack(fps, dim=0)

    y_reg = torch.stack(y_reg_list, dim=0) if task in ("regression", "both") else None
    y_cls = torch.stack(y_cls_list, dim=0) if task in ("classification", "both") else None
    seed = int(graphs[0].get("meta", {}).get("seed", -1)) if graphs else -1
    return Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, fp=fp, y_reg=y_reg, y_cls=y_cls, smiles=smiles, split=splits, seed=seed)


# -----------------------------
# Morgan fingerprints
# -----------------------------
def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    if smi is None or not isinstance(smi, str) or smi.strip() == "":
        return None
    try:
        return Chem.MolFromSmiles(smi, sanitize=True)
    except Exception:
        return None


def morgan_fp(smi: str, radius: int, n_bits: int) -> np.ndarray:
    mol = mol_from_smiles(smi)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smi}")
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)


def build_fp_map_from_graphs(graphs: List[Dict], radius: int, n_bits: int) -> Dict[str, np.ndarray]:
    fp_map: Dict[str, np.ndarray] = {}
    for g in graphs:
        smi = str(g.get("meta", {}).get("smiles", ""))
        if smi in fp_map:
            continue
        fp_map[smi] = morgan_fp(smi, radius=radius, n_bits=n_bits)
    return fp_map


# -----------------------------
# Model blocks
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
        out = self.upd(x + agg)
        return out


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
                dim=-1
            )
        raise ValueError("Unknown pooling")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.node_proj(x)
        for blk in self.blocks:
            x = blk(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)
        num_graphs = int(batch.max().item() + 1) if batch.numel() else 0
        return self.pool(x, batch, num_graphs)


class FusionNet(nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        fp_bits: int,
        task: str,
        # gnn
        node_dim: int,
        depth: int,
        gnn_hidden: int,
        pooling: str,
        # fp branch
        fp_hidden: int,
        fp_dropout: float,
        fp_bit_dropout: float,
        # fusion
        fusion_hidden: int,
        dropout: float,
        head_hidden: int,
    ):
        super().__init__()
        self.task = task
        self.fp_bit_dropout = float(fp_bit_dropout)

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

        fp = batch.fp
        if self.training and self.fp_bit_dropout > 0:
            keep = (torch.rand_like(fp) > self.fp_bit_dropout).float()
            fp = fp * keep

        f = self.fp_proj(fp)
        z = self.fusion(torch.cat([g, f], dim=-1))
        out: Dict[str, torch.Tensor] = {}
        if self.head_reg is not None:
            out["y_reg"] = self.head_reg(z).squeeze(-1)
        if self.head_cls is not None:
            out["logits"] = self.head_cls(z).squeeze(-1)
        return out


# -----------------------------
# Metrics and thresholding
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


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Simple ECE (expected calibration error) on [0,1] bins.
    """
    y_true = y_true.astype(int)
    y_prob = np.clip(y_prob.astype(float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(y_true))
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(m):
            continue
        acc = float(np.mean(y_true[m]))
        conf = float(np.mean(y_prob[m]))
        ece += (float(np.sum(m)) / n) * abs(acc - conf)
    return float(ece)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out: Dict[str, float] = {}
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
    out["ece"] = float(expected_calibration_error(y_true, y_prob, n_bins=15))
    return out


def select_threshold_by_mcc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    thresholds = np.unique(np.clip(y_prob, 0, 1))
    grid = np.linspace(0.0, 1.0, 1001)
    thresholds = np.unique(np.concatenate([thresholds, grid]))
    best_t, best_mcc = 0.5, -1.0
    for t in thresholds:
        mcc = matthews_corrcoef(y_true, (y_prob >= t).astype(int))
        if mcc > best_mcc:
            best_mcc = mcc
            best_t = float(t)
    return float(best_t)


def compute_objective(val_pred: Dict[str, np.ndarray], task: str) -> float:
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
        y_prob = val_pred["y_cls_prob"]
        mask = y_true >= 0
        if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
            return -1e9
        return float(average_precision_score(y_true[mask], y_prob[mask]))

    # both
    score = 0.0
    y_true_c = val_pred["y_cls_true"]
    y_prob = val_pred["y_cls_prob"]
    mask_c = y_true_c >= 0
    if mask_c.sum() > 0 and len(np.unique(y_true_c[mask_c])) >= 2:
        score += float(average_precision_score(y_true_c[mask_c], y_prob[mask_c]))
    else:
        score += 0.0

    y_true_r = val_pred["y_reg_true"]
    y_pred_r = val_pred["y_reg_pred"]
    mask_r = ~np.isnan(y_true_r)
    if mask_r.sum() > 0:
        rmse = float(np.sqrt(mean_squared_error(y_true_r[mask_r], y_pred_r[mask_r])))
        score += float(-rmse / 2.0)
    else:
        score += -1.0

    return float(score / 2.0)


# -----------------------------
# Loss functions
# -----------------------------
@dataclass
class TrainConfig:
    task: str = "both"

    # Morgan
    fp_radius: int = 2
    fp_bits: int = 2048
    fp_hidden: int = 256
    fp_dropout: float = 0.35
    fp_bit_dropout: float = 0.20

    # GNN
    node_dim: int = 256
    depth: int = 5
    gnn_hidden: int = 256
    pooling: str = "meanmax"

    # Fusion + heads
    fusion_hidden: int = 512
    head_hidden: int = 256
    dropout: float = 0.25

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 1e-4
    fp_weight_decay_mult: float = 5.0
    batch_size: int = 64
    max_epochs: int = 220
    patience: int = 25
    grad_clip: float = 2.0

    # EMA (new)
    ema_decay: float = 0.999
    ema_start_step: int = 200

    # Loss settings
    reg_loss: str = "huber"
    huber_delta: float = 1.0
    auto_pos_weight: bool = True
    cls_pos_weight: float = 1.0
    cls_label_smoothing: float = 0.05

    # Tuning
    tune_trials: int = 20
    tune_seed: int = 2025


def make_minibatches(items: List[Dict], batch_size: int, rng: np.random.RandomState, shuffle: bool) -> List[List[Dict]]:
    idx = np.arange(len(items))
    if shuffle:
        rng.shuffle(idx)
    return [[items[j] for j in idx[i:i + batch_size].tolist()] for i in range(0, len(items), batch_size)]


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


def regression_loss(pred: torch.Tensor, target: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    pred = pred[mask]
    target = target[mask]
    if cfg.reg_loss == "mse":
        return F.mse_loss(pred, target)
    return F.huber_loss(pred, target, delta=cfg.huber_delta)


def classification_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float, cfg: TrainConfig) -> torch.Tensor:
    mask = target >= 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    logits = logits[mask]
    t = target[mask].float()

    if cfg.cls_label_smoothing and cfg.cls_label_smoothing > 0:
        eps = float(cfg.cls_label_smoothing)
        t = t * (1.0 - eps) + 0.5 * eps

    pw = torch.tensor(pos_weight, device=logits.device, dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(logits, t, pos_weight=pw)


# -----------------------------
# Cache loading and splitting
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
# Prediction helper
# -----------------------------
@torch.no_grad()
def predict_split(
    model: nn.Module,
    graphs: List[Dict],
    fp_map: Dict[str, np.ndarray],
    device: torch.device,
    task: str,
    batch_size: int
) -> Dict[str, np.ndarray]:
    model.eval()
    rng = np.random.RandomState(0)
    batches = make_minibatches(graphs, batch_size=batch_size, rng=rng, shuffle=False)

    y_reg_true, y_reg_pred = [], []
    y_cls_true, y_cls_prob = [], []
    smiles = []

    for b in batches:
        batch = collate_graphs_with_fp(b, fp_map=fp_map, device=device, task=task)
        out = model(batch)
        smiles.extend(batch.smiles)

        if task in ("regression", "both"):
            y_reg_true.append(batch.y_reg.detach().cpu().numpy())
            y_reg_pred.append(out["y_reg"].detach().cpu().numpy())

        if task in ("classification", "both"):
            y_cls_true.append(batch.y_cls.detach().cpu().numpy())
            y_cls_prob.append(torch.sigmoid(out["logits"]).detach().cpu().numpy())

    res: Dict[str, np.ndarray] = {"smiles": np.array(smiles, dtype=object)}
    if task in ("regression", "both"):
        res["y_reg_true"] = np.concatenate(y_reg_true) if y_reg_true else np.array([], dtype=float)
        res["y_reg_pred"] = np.concatenate(y_reg_pred) if y_reg_pred else np.array([], dtype=float)
    if task in ("classification", "both"):
        res["y_cls_true"] = np.concatenate(y_cls_true) if y_cls_true else np.array([], dtype=int)
        res["y_cls_prob"] = np.concatenate(y_cls_prob) if y_cls_prob else np.array([], dtype=float)
    return res


def predict_split_with_ema(
    model: nn.Module,
    ema: EMA,
    graphs: List[Dict],
    fp_map: Dict[str, np.ndarray],
    device: torch.device,
    task: str,
    batch_size: int
) -> Dict[str, np.ndarray]:
    ema.apply_shadow(model)
    try:
        pred = predict_split(model, graphs, fp_map, device, task, batch_size)
    finally:
        ema.restore(model)
    return pred


# -----------------------------
# Figures (publication style)
# -----------------------------
def plot_training_curves(hist: pd.DataFrame, figs_dir: Path, style: FigStyle, prefix: str) -> None:
    fig = plt.figure(figsize=(8.2, 5.6))
    ax = fig.add_subplot(111)
    ax.plot(hist["epoch"], hist["train_loss"], marker="o", label="Train")
    ax.plot(hist["epoch"], hist["val_loss"], marker="o", label="Validation")
    ax.set_title("Training Curves: Loss", fontweight="bold")
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")
    ax.legend(frameon=False)
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)
    _save_fig(fig, figs_dir / f"{prefix}_loss.svg", figs_dir / f"{prefix}_loss.png", dpi=style.dpi_png)

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


def plot_regression(y_true: np.ndarray, y_pred: np.ndarray, figs_dir: Path, style: FigStyle, prefix: str, title_prefix: str) -> None:
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return

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


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if y_true.size == 0 or len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def plot_classification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    figs_dir: Path,
    style: FigStyle,
    prefix: str,
    title_prefix: str,
    y_true_train: Optional[np.ndarray] = None,
    y_prob_train: Optional[np.ndarray] = None,
) -> None:
    mask = y_true >= 0
    y_true = y_true[mask].astype(int)
    y_prob = y_prob[mask].astype(float)
    if y_true.size == 0:
        return

    # prepare optional TRAIN curve (ROC only)
    train_ok = False
    if y_true_train is not None and y_prob_train is not None:
        mt = y_true_train >= 0
        yt = y_true_train[mt].astype(int)
        pt = y_prob_train[mt].astype(float)
        train_ok = (yt.size > 0 and len(np.unique(yt)) >= 2)
    else:
        yt, pt = None, None

    # ROC (Train + Test in one plot; AUC values in legend)
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_test = _safe_auc(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"Test (AUC={auc_test:.3f})" if np.isfinite(auc_test) else "Test (AUC=NA)")

        if train_ok and yt is not None and pt is not None:
            fpr_t, tpr_t, _ = roc_curve(yt, pt)
            auc_tr = _safe_auc(yt, pt)
            ax.plot(
                fpr_t, tpr_t, linestyle=":",
                label=f"Train (AUC={auc_tr:.3f})" if np.isfinite(auc_tr) else "Train (AUC=NA)"
            )

        ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
        ax.set_title(f"{title_prefix}: ROC", fontweight="bold")
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
        ax.plot(rec, prec, label="PR")
        ax.set_title(f"{title_prefix}: PR", fontweight="bold")
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

    # Confusion
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = plt.figure(figsize=(6.2, 5.6))
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"{title_prefix}: Confusion (thr={threshold:.3f})", fontweight="bold")
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Observed", fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)
    _save_fig(fig, figs_dir / f"{prefix}_confusion.svg", figs_dir / f"{prefix}_confusion.png", dpi=style.dpi_png)


# -----------------------------
# Training and tuning
# -----------------------------
def sample_config(base: TrainConfig, rng: np.random.RandomState) -> TrainConfig:
    cfg = TrainConfig(**asdict(base))

    # Optimization: avoid too small regularization
    cfg.lr = float(10 ** rng.uniform(np.log10(1e-4), np.log10(3e-3)))
    cfg.weight_decay = float(10 ** rng.uniform(np.log10(1e-5), np.log10(3e-4)))
    cfg.dropout = float(rng.uniform(0.15, 0.40))
    cfg.batch_size = int(rng.choice([32, 64, 96, 128]))
    cfg.patience = int(rng.choice([15, 20, 25, 30]))
    cfg.max_epochs = int(rng.choice([150, 190, 220, 260]))
    cfg.grad_clip = float(rng.choice([1.0, 2.0, 5.0]))

    # EMA
    cfg.ema_decay = float(rng.choice([0.999, 0.9993, 0.9995]))
    cfg.ema_start_step = int(rng.choice([100, 200, 400]))

    # GNN
    cfg.node_dim = int(rng.choice([128, 192, 256, 320]))
    cfg.gnn_hidden = int(rng.choice([128, 192, 256, 320]))
    cfg.depth = int(rng.choice([3, 4, 5, 6, 7]))
    cfg.pooling = str(rng.choice(["meanmax", "mean", "max"]))

    # FP branch: safer (no huge fp_hidden, stronger dropout, add bit dropout)
    cfg.fp_hidden = int(rng.choice([128, 192, 256, 384]))
    cfg.fp_dropout = float(rng.uniform(0.20, 0.55))
    cfg.fp_bit_dropout = float(rng.uniform(0.10, 0.35))

    # Fusion/head
    cfg.fusion_hidden = int(rng.choice([384, 512, 768, 1024]))
    cfg.head_hidden = int(rng.choice([192, 256, 384]))

    # Loss
    cfg.reg_loss = str(rng.choice(["huber", "mse"]))
    cfg.huber_delta = float(rng.choice([0.5, 1.0, 1.5]))
    cfg.cls_label_smoothing = float(rng.choice([0.00, 0.03, 0.05, 0.08]))

    # FP weight decay multiplier
    cfg.fp_weight_decay_mult = float(rng.choice([3.0, 5.0, 7.0]))

    return cfg


def train_one(
    graphs_train: List[Dict],
    graphs_val: List[Dict],
    fp_map: Dict[str, np.ndarray],
    node_in: int,
    edge_in: int,
    cfg: TrainConfig,
    device: torch.device,
    run_seed: int,
) -> Tuple[nn.Module, pd.DataFrame, Dict[str, float], float, float, Dict[str, torch.Tensor]]:
    set_all_seeds(run_seed)

    pos_weight = 1.0
    if cfg.task in ("classification", "both") and cfg.auto_pos_weight:
        pos_weight = compute_pos_weight(graphs_train)

    model = FusionNet(
        node_in=node_in,
        edge_in=edge_in,
        fp_bits=cfg.fp_bits,
        task=cfg.task,
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        gnn_hidden=cfg.gnn_hidden,
        pooling=cfg.pooling,
        fp_hidden=cfg.fp_hidden,
        fp_dropout=cfg.fp_dropout,
        fp_bit_dropout=cfg.fp_bit_dropout,
        fusion_hidden=cfg.fusion_hidden,
        dropout=cfg.dropout,
        head_hidden=cfg.head_hidden,
    ).to(device)

    # EMA (new)
    ema = EMA(model, decay=cfg.ema_decay, start_step=cfg.ema_start_step)

    # Targeted regularization: stronger weight decay on fingerprint projection layers
    fp_params: List[torch.nn.Parameter] = []
    gnn_params: List[torch.nn.Parameter] = []
    head_params: List[torch.nn.Parameter] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("fp_proj"):
            fp_params.append(p)
        elif n.startswith("encoder"):
            gnn_params.append(p)
        else:
            head_params.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": gnn_params, "weight_decay": cfg.weight_decay},
            {"params": head_params, "weight_decay": cfg.weight_decay},
            {"params": fp_params, "weight_decay": cfg.weight_decay * float(cfg.fp_weight_decay_mult)},
        ],
        lr=cfg.lr,
    )

    best_obj = -1e18
    best_epoch = -1
    wait = 0

    # best EMA shadow weights (store tensors)
    best_ema_shadow: Optional[Dict[str, torch.Tensor]] = None

    hist = []
    rng = np.random.RandomState(run_seed + 123)

    global_step = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        batches = make_minibatches(graphs_train, cfg.batch_size, rng=rng, shuffle=True)
        tr_losses = []

        for b in batches:
            global_step += 1
            batch = collate_graphs_with_fp(b, fp_map=fp_map, device=device, task=cfg.task)
            out = model(batch)

            loss = torch.tensor(0.0, device=device)
            if cfg.task in ("regression", "both"):
                loss = loss + regression_loss(out["y_reg"], batch.y_reg, cfg)
            if cfg.task in ("classification", "both"):
                loss = loss + classification_loss(out["logits"], batch.y_cls, pos_weight, cfg)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            opt.step()

            # EMA update after optimizer step
            ema.update(model, step=global_step)

            tr_losses.append(float(loss.detach().cpu().item()))

        # val eval (EMA weights)
        with torch.no_grad():
            vp = predict_split_with_ema(model, ema, graphs_val, fp_map, device, cfg.task, cfg.batch_size)
            obj = compute_objective(vp, cfg.task)

            # compute val_loss using EMA weights for consistency
            ema.apply_shadow(model)
            try:
                vbatches = make_minibatches(graphs_val, cfg.batch_size, rng=np.random.RandomState(0), shuffle=False)
                vlosses = []
                model.eval()
                for b in vbatches:
                    batch = collate_graphs_with_fp(b, fp_map=fp_map, device=device, task=cfg.task)
                    out = model(batch)
                    vloss = torch.tensor(0.0, device=device)
                    if cfg.task in ("regression", "both"):
                        vloss = vloss + regression_loss(out["y_reg"], batch.y_reg, cfg)
                    if cfg.task in ("classification", "both"):
                        vloss = vloss + classification_loss(out["logits"], batch.y_cls, pos_weight, cfg)
                    vlosses.append(float(vloss.detach().cpu().item()))
                val_loss = float(np.mean(vlosses)) if vlosses else float("nan")
            finally:
                ema.restore(model)

        hist.append({
            "epoch": epoch,
            "train_loss": float(np.mean(tr_losses)) if tr_losses else float("nan"),
            "val_loss": val_loss,
            "val_objective": float(obj),
        })

        if obj > best_obj:
            best_obj = float(obj)
            best_epoch = epoch
            wait = 0
            # store EMA shadow as the "best" weights
            best_ema_shadow = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_ema_shadow is None:
        best_ema_shadow = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}

    # Load best EMA weights into model parameters
    model_sd = model.state_dict()
    for k in model_sd.keys():
        if k in best_ema_shadow:
            model_sd[k] = best_ema_shadow[k]
    model.load_state_dict(model_sd)

    # Final val metrics computed using EMA-best-loaded model (plain predict)
    val_pred = predict_split(model, graphs_val, fp_map, device, cfg.task, cfg.batch_size)
    val_metrics: Dict[str, float] = {
        "best_epoch": int(best_epoch),
        "best_objective": float(best_obj),
        "pos_weight_used": float(pos_weight),
        "ema_decay": float(cfg.ema_decay),
        "ema_start_step": int(cfg.ema_start_step),
    }

    if cfg.task in ("regression", "both"):
        y_true = val_pred["y_reg_true"]
        y_hat = val_pred["y_reg_pred"]
        mask = ~np.isnan(y_true)
        if mask.sum() > 0:
            val_metrics.update({f"val_{k}": v for k, v in regression_metrics(y_true[mask], y_hat[mask]).items()})

    if cfg.task in ("classification", "both"):
        y_true = val_pred["y_cls_true"]
        y_prob = val_pred["y_cls_prob"]
        mask = y_true >= 0
        if mask.sum() > 0 and len(np.unique(y_true[mask])) >= 2:
            thr = select_threshold_by_mcc(y_true[mask], y_prob[mask])
            val_metrics.update({f"val_{k}": v for k, v in classification_metrics(y_true[mask], y_prob[mask], thr).items()})
            val_metrics["val_threshold"] = float(thr)
        else:
            val_metrics["val_threshold"] = 0.5

    return model, pd.DataFrame(hist), val_metrics, float(best_obj), float(pos_weight), best_ema_shadow


# -----------------------------
# Save predictions
# -----------------------------
def save_predictions_csv(
    out_path: Path,
    seed: int,
    split: str,
    smiles: np.ndarray,
    task: str,
    pred: Dict[str, np.ndarray],
    threshold: Optional[float],
) -> None:
    df = pd.DataFrame({"seed": seed, "split": split, "canonical_smiles": smiles})
    if task in ("regression", "both"):
        df["y_reg_true"] = pred["y_reg_true"]
        df["y_reg_pred"] = pred["y_reg_pred"]
    if task in ("classification", "both"):
        df["y_cls_true"] = pred["y_cls_true"]
        df["y_cls_prob"] = pred["y_cls_prob"]
        if threshold is not None:
            yhat = (pred["y_cls_prob"] >= threshold).astype(int)
            df["y_cls_pred"] = yhat
    df.to_csv(out_path, index=False)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate GNN + Morgan late-fusion model (overfitting-controlled, EMA).")
    ap.add_argument("--graph_cache", type=str, required=True)
    ap.add_argument("--dedup_csv", type=str, required=True, help="Script-1 output (sanity checks).")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--task", type=str, choices=["regression", "classification", "both"], default="both")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tune_trials", type=int, default=20)
    ap.add_argument("--tune_seed", type=int, default=2025)
    ap.add_argument("--no_tuning", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    dirs = ensure_dirs(out_root)

    style = FigStyle()
    style.apply()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    cache = load_graph_cache(Path(args.graph_cache).resolve())
    graphs = cache["graphs"]
    seed = int(cache.get("seed", -1))

    # Split graphs
    split_map = split_graphs(graphs)
    graphs_train = split_map["train"]
    graphs_val = split_map["val"]
    graphs_test = split_map["test"]
    if len(graphs_train) == 0 or len(graphs_val) == 0 or len(graphs_test) == 0:
        raise RuntimeError("Empty split found. Fix scaffold split generation.")

    # Dims
    node_in = int(graphs_train[0]["x"].shape[1])
    edge_in = int(graphs_train[0]["edge_attr"].shape[1]) if graphs_train[0]["edge_attr"].numel() else int(graphs[0]["edge_attr"].shape[1])

    # Base config (revised defaults)
    base = TrainConfig(task=args.task, tune_trials=int(args.tune_trials), tune_seed=int(args.tune_seed))

    # Fingerprints map (deterministic)
    fp_map = build_fp_map_from_graphs(graphs, radius=base.fp_radius, n_bits=base.fp_bits)

    # Tuning (train/val only)
    rng = np.random.RandomState(base.tune_seed)
    trials = 1 if args.no_tuning else base.tune_trials

    tuning_rows = []
    best_obj = -1e18
    best_cfg: Optional[TrainConfig] = None
    best_model: Optional[nn.Module] = None
    best_hist: Optional[pd.DataFrame] = None
    best_val_metrics: Optional[Dict[str, float]] = None
    best_ema_shadow: Optional[Dict[str, torch.Tensor]] = None
    best_pos_weight: float = 1.0

    for t in range(trials):
        cfg = base if args.no_tuning else sample_config(base, rng)
        run_seed = int(base.tune_seed + 2000 + t)

        model, hist, val_metrics, obj, posw, ema_shadow = train_one(
            graphs_train, graphs_val, fp_map, node_in, edge_in, cfg, device, run_seed
        )

        row = {"trial": t, "objective": obj, **asdict(cfg), **val_metrics}
        tuning_rows.append(row)

        if obj > best_obj:
            best_obj = float(obj)
            best_cfg = cfg
            best_model = model
            best_hist = hist
            best_val_metrics = val_metrics
            best_ema_shadow = ema_shadow
            best_pos_weight = float(posw)

    if best_cfg is None or best_model is None or best_hist is None or best_val_metrics is None:
        raise RuntimeError("Training failed to produce a best model/config.")

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_csv = dirs["tables"] / f"gnn_fusion_tuning_seed{seed:03d}.csv"
    tuning_df.to_csv(tuning_csv, index=False)

    # Save best config
    best_cfg_json = dirs["reports"] / f"gnn_fusion_best_config_seed{seed:03d}.json"
    with open(best_cfg_json, "w", encoding="utf-8") as f:
        json.dump(
            {"seed": seed, "best_objective": best_obj, "config": asdict(best_cfg), "val_metrics": best_val_metrics},
            f, indent=2
        )

    # Save training history
    hist_csv = dirs["tables"] / f"gnn_fusion_training_history_seed{seed:03d}.csv"
    best_hist.to_csv(hist_csv, index=False)

    # Plot curves
    prefix = f"20_gnn_fusion_seed{seed:03d}"
    plot_training_curves(best_hist, dirs["figures"], style, prefix=prefix)

    # Lock threshold on VAL only (EMA-best model already loaded)
    threshold: Optional[float] = None
    if args.task in ("classification", "both"):
        vp = predict_split(best_model, graphs_val, fp_map, device, args.task, best_cfg.batch_size)
        y_true = vp["y_cls_true"]
        y_prob = vp["y_cls_prob"]
        mask = y_true >= 0
        threshold = select_threshold_by_mcc(y_true[mask], y_prob[mask]) if (mask.sum() > 0 and len(np.unique(y_true[mask])) >= 2) else 0.5

    # Store TRAIN predictions for ROC overlay on TEST ROC
    train_pred_for_roc: Optional[Dict[str, np.ndarray]] = None

    # Predictions and metrics
    metrics_rows = []
    for sp, g_list in [("train", graphs_train), ("val", graphs_val), ("test", graphs_test)]:
        pred = predict_split(best_model, g_list, fp_map, device, args.task, best_cfg.batch_size)
        save_predictions_csv(
            dirs["predictions"] / f"gnn_fusion_seed{seed:03d}_{sp}.csv",
            seed, sp, pred["smiles"], args.task, pred, threshold
        )

        if sp == "train":
            train_pred_for_roc = pred

        row: Dict[str, object] = {"seed": seed, "split": sp, "model": "GNN_FUSION", "task": args.task}

        if args.task in ("regression", "both"):
            y_true_r = pred["y_reg_true"]
            y_hat_r = pred["y_reg_pred"]
            mask_r = ~np.isnan(y_true_r)
            if mask_r.sum() > 0:
                row.update(regression_metrics(y_true_r[mask_r], y_hat_r[mask_r]))

        if args.task in ("classification", "both"):
            y_true_c = pred["y_cls_true"]
            y_prob_c = pred["y_cls_prob"]
            mask_c = y_true_c >= 0
            row["threshold"] = float(threshold if threshold is not None else 0.5)
            if mask_c.sum() > 0 and len(np.unique(y_true_c[mask_c])) >= 2:
                row.update(classification_metrics(y_true_c[mask_c].astype(int), y_prob_c[mask_c].astype(float), float(row["threshold"])) )

        metrics_rows.append(row)

        # Test figures (paper)
        if sp == "test":
            if args.task in ("regression", "both"):
                plot_regression(
                    pred["y_reg_true"], pred["y_reg_pred"],
                    dirs["figures"], style,
                    prefix=f"21_gnn_fusion_reg_seed{seed:03d}_test",
                    title_prefix="GNN+Morgan Fusion Regression (Test)"
                )
            if args.task in ("classification", "both"):
                ytt = None
                ppt = None
                if train_pred_for_roc is not None:
                    ytt = train_pred_for_roc["y_cls_true"]
                    ppt = train_pred_for_roc["y_cls_prob"]
                plot_classification(
                    pred["y_cls_true"], pred["y_cls_prob"], float(threshold if threshold is not None else 0.5),
                    dirs["figures"], style,
                    prefix=f"22_gnn_fusion_cls_seed{seed:03d}_test",
                    title_prefix="GNN+Morgan Fusion Classification (Test)",
                    y_true_train=ytt,
                    y_prob_train=ppt,
                )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = dirs["metrics"] / f"gnn_fusion_metrics_seed{seed:03d}.csv"
    metrics_json = dirs["metrics"] / f"gnn_fusion_metrics_seed{seed:03d}.json"
    metrics_df.to_csv(metrics_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "best_config": asdict(best_cfg),
                "threshold": threshold,
                "metrics": metrics_rows,
            },
            f, indent=2
        )

    # Save best model weights (EMA-best already loaded)
    model_path = dirs["models"] / f"gnn_fusion_seed{seed:03d}_best.pt"
    payload = {
        "state_dict": best_model.state_dict(),
        "config": asdict(best_cfg),
        "seed": seed,
        "node_in": node_in,
        "edge_in": edge_in,
    }
    if best_ema_shadow is not None:
        payload["ema_shadow"] = {k: v.cpu() for k, v in best_ema_shadow.items()}
    torch.save(payload, model_path)

    # Manifest
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "inputs": {
            "graph_cache": str(Path(args.graph_cache).resolve()),
            "dedup_csv": str(Path(args.dedup_csv).resolve())
        },
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
            "leakage_control": "Scaffold split fixed; tuning uses train/val only; threshold locked on val; test not used for tuning.",
            "fusion": "Late fusion of graph embedding and Morgan branch embedding.",
            "overfitting_controls": [
                "Reduced Morgan capacity (fp_hidden), stronger dropout (fp_dropout, dropout).",
                "Stochastic fingerprint bit dropout during training (fp_bit_dropout).",
                "Targeted weight decay (higher WD on fp_proj via parameter groups).",
                "Safer hyperparameter search space (prevents weak regularization).",
                "Optional label smoothing for classification (cls_label_smoothing).",
                "EMA of weights used for validation selection and final evaluation.",
            ],
        },
        "environment": {"torch_version": torch.__version__, "device_used": str(device)},
    }
    manifest_path = dirs["reports"] / f"run_manifest_gnn_fusion_seed{seed:03d}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== DONE: GNN + Morgan fusion training + evaluation (overfitting-controlled, EMA) ===")
    print(f"Seed:              {seed}")
    print(f"Best model:        {model_path}")
    print(f"Tuning table:      {tuning_csv}")
    print(f"Best config:       {best_cfg_json}")
    print(f"Training history:  {hist_csv}")
    print(f"Metrics:           {metrics_csv}")
    print(f"Predictions dir:   {dirs['predictions']}")
    print(f"Figures (SVG):     {dirs['figures']}")
    print(f"Manifest:          {manifest_path}")
    print("==========================================================================\n")


if __name__ == "__main__":
    main()
