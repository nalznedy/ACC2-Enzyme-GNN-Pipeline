#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 08_interpretability_applicability_domain_error_analysis.py

Purpose:
    Analyze interpretability, applicability domain, and error patterns for trained models.

Workflow step:
    Step 8 - model interpretation and applicability-domain assessment.

Main inputs:
    - features/graphs_seed###.pt
    - models/gnn_fusion_seed###_best.pt
    - predictions/gnn_fusion_seed###_{val,test}.csv

Main outputs:
    - interpretability_ad/tables/*.csv
    - interpretability_ad/figures/*.svg|png
    - interpretability_ad/atom_svgs/*.svg
    - interpretability_ad/reports/manifest_script9.json

Notes:
    Includes atom-level attribution and chemical-space applicability-domain summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Draw import rdMolDraw2D

# Matplotlib only
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib as mpl
import matplotlib.pyplot as plt

# Sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Optional UMAP
try:
    import umap  # type: ignore
    _HAVE_UMAP = True
except Exception:
    _HAVE_UMAP = False


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


def ensure_dirs(out_root: Path) -> Dict[str, Path]:
    d = {}
    d["reports"] = out_root / "reports"
    d["tables"] = out_root / "tables"
    d["figures"] = out_root / "figures"
    d["metrics"] = out_root / "metrics"
    d["predictions"] = out_root / "predictions"
    d["models"] = out_root / "models"
    d["features"] = out_root / "features"

    d["interp_root"] = out_root / "interpretability_ad"
    d["interp_tables"] = d["interp_root"] / "tables"
    d["interp_figs"] = d["interp_root"] / "figures"
    d["interp_svgs"] = d["interp_root"] / "atom_svgs"
    d["interp_csvs"] = d["interp_root"] / "atom_csvs"
    d["interp_reports"] = d["interp_root"] / "reports"

    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


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
# Batch
# -----------------------------
@dataclass
class Batch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor
    fp: torch.Tensor
    y_reg: Optional[torch.Tensor]
    y_cls: Optional[torch.Tensor]
    smiles: List[str]


def collate_graphs_with_fp(
    graphs: List[Dict],
    fp_map: Dict[str, np.ndarray],
    device: torch.device,
    task: str,
) -> Batch:
    xs, eis, eas, batches = [], [], [], []
    fps = []
    y_reg_list, y_cls_list = [], []
    smiles = []

    node_offset = 0
    for gid, g in enumerate(graphs):
        meta = g.get("meta", {})
        smi = str(meta.get("smiles", ""))
        if smi not in fp_map:
            raise RuntimeError(f"Fingerprint missing for SMILES: {smi}")

        x = g["x"].to(device)
        ei = g["edge_index"].to(device)
        ea = g["edge_attr"].to(device)

        xs.append(x)
        eis.append(ei + node_offset)
        eas.append(ea)
        batches.append(torch.full((x.size(0),), gid, device=device, dtype=torch.long))
        node_offset += x.size(0)

        fps.append(torch.tensor(fp_map[smi], device=device, dtype=torch.float32))
        smiles.append(smi)

        if task in ("regression", "both"):
            yreg = g.get("y_reg", None)
            y_reg_list.append(torch.tensor(float("nan"), device=device) if yreg is None else yreg.to(device))
        if task in ("classification", "both"):
            ycls = g.get("y_cls", None)
            y_cls_list.append(torch.tensor(-1, device=device, dtype=torch.long) if ycls is None else ycls.to(device))

    x = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1)
    edge_attr = torch.cat(eas, dim=0)
    batch = torch.cat(batches, dim=0)
    fp = torch.stack(fps, dim=0)

    y_reg = torch.stack(y_reg_list, dim=0) if task in ("regression", "both") else None
    y_cls = torch.stack(y_cls_list, dim=0) if task in ("classification", "both") else None
    return Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, fp=fp, y_reg=y_reg, y_cls=y_cls, smiles=smiles)


# -----------------------------
# Morgan + similarity
# -----------------------------
def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    if not isinstance(smi, str) or smi.strip() == "":
        return None
    try:
        return Chem.MolFromSmiles(smi, sanitize=True)
    except Exception:
        return None


def morgan_bitvect(smi: str, radius: int, n_bits: int) -> DataStructs.ExplicitBitVect:
    mol = mol_from_smiles(smi)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smi}")
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def morgan_fp_array(smi: str, radius: int, n_bits: int) -> np.ndarray:
    bv = morgan_bitvect(smi, radius, n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)


def build_fp_maps(
    graphs: List[Dict], radius: int, n_bits: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, DataStructs.ExplicitBitVect]]:
    fp_arr_map: Dict[str, np.ndarray] = {}
    fp_bv_map: Dict[str, DataStructs.ExplicitBitVect] = {}
    for g in graphs:
        smi = str(g.get("meta", {}).get("smiles", ""))
        if smi in fp_arr_map:
            continue
        fp_bv_map[smi] = morgan_bitvect(smi, radius, n_bits)
        fp_arr_map[smi] = morgan_fp_array(smi, radius, n_bits)
    return fp_arr_map, fp_bv_map


def max_tanimoto_to_train(
    test_smiles: List[str],
    train_bv: List[DataStructs.ExplicitBitVect],
    test_bv_map: Dict[str, DataStructs.ExplicitBitVect],
) -> np.ndarray:
    sims = []
    for smi in test_smiles:
        bv = test_bv_map[smi]
        s = DataStructs.BulkTanimotoSimilarity(bv, train_bv)
        sims.append(float(np.max(s)) if len(s) else float("nan"))
    return np.asarray(sims, dtype=float)


# -----------------------------
# Model (FusionNet, extracted from Script-6)
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


def load_fusion_model(model_path: Path, device: torch.device, task: str) -> Tuple[FusionNet, Dict]:
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
    return model, cfg


# -----------------------------
# Metrics helpers (AD bins)
# -----------------------------
def cls_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= thr).astype(int)
    out = {}
    if len(np.unique(y_true)) >= 2:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        out["bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    else:
        out["auroc"] = float("nan")
        out["pr_auc"] = float("nan")
        out["mcc"] = float("nan")
        out["bal_acc"] = float("nan")
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out["tn"] = int(tn)
    out["fp"] = int(fp)
    out["fn"] = int(fn)
    out["tp"] = int(tp)
    return out


def reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = {}
    out["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    out["mae"] = float(mean_absolute_error(y_true, y_pred))
    out["r2"] = float(r2_score(y_true, y_pred))
    return out


# -----------------------------
# Embeddings projection
# -----------------------------
def project_2d(emb: np.ndarray, method: str, seed: int = 0) -> np.ndarray:
    if method == "umap" and _HAVE_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=25, min_dist=0.15, metric="cosine")
        return reducer.fit_transform(emb)
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(emb)


def plot_chemical_space(points: np.ndarray, color: np.ndarray, title: str, cbar_label: str, out_svg: Path, out_png: Path, style: FigStyle) -> None:
    fig = plt.figure(figsize=(7.4, 6.2))
    ax = fig.add_subplot(111)
    sc = ax.scatter(points[:, 0], points[:, 1], c=color, alpha=0.65)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Component 1", fontweight="bold")
    ax.set_ylabel("Component 2", fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(cbar_label, fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)
    _save_fig(fig, out_svg, out_png, dpi=style.dpi_png)


# -----------------------------
# Atom attribution (gradient w.r.t node features)
# -----------------------------
@torch.no_grad()
def sanity_check_atom_counts(smiles: str, n_nodes: int) -> bool:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return False
    return mol.GetNumAtoms() == n_nodes


def compute_atom_importance_for_single(
    model: FusionNet,
    g: Dict,
    fp_arr_map: Dict[str, np.ndarray],
    device: torch.device,
    task: str,
    which: str = "classification",
) -> Tuple[np.ndarray, Dict[str, float]]:
    model.eval()

    smi = str(g.get("meta", {}).get("smiles", ""))
    x = g["x"].to(device).clone().detach().requires_grad_(True)
    edge_index = g["edge_index"].to(device)
    edge_attr = g["edge_attr"].to(device)
    batch = torch.zeros((x.size(0),), device=device, dtype=torch.long)
    fp = torch.tensor(fp_arr_map[smi], device=device, dtype=torch.float32).unsqueeze(0)

    b = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, fp=fp, y_reg=None, y_cls=None, smiles=[smi])

    out = model(b)
    extra = {}
    if which == "classification":
        if "logits" not in out:
            raise RuntimeError("Model has no classification head.")
        score = out["logits"][0]
        extra["logit"] = float(score.detach().cpu().item())
        extra["prob"] = float(torch.sigmoid(score).detach().cpu().item())
    else:
        if "y_reg" not in out:
            raise RuntimeError("Model has no regression head.")
        score = out["y_reg"][0]
        extra["y_reg_pred"] = float(score.detach().cpu().item())

    model.zero_grad(set_to_none=True)
    score.backward()

    grad = x.grad.detach()
    gi = (grad * x.detach()).abs().sum(dim=1)
    gi = gi / (gi.max() + 1e-12)
    return gi.detach().cpu().numpy(), extra


# -----------------------------
# PubChem label helpers (ADDED)
# -----------------------------
def _escape_xml(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&apos;")
    )


def _normalize_pubchem_label(raw: str) -> str:
    raw = "" if raw is None else str(raw).strip()
    if raw == "" or raw.lower() in {"nan", "none", "null"}:
        return ""
    # If user provides CID-only, show as "CID: ####"
    if raw.isdigit():
        return f"CID: {raw}"
    # If already contains CID/PCID, keep as-is
    if "cid" in raw.lower():
        return raw
    return raw


def _build_pubchem_map_from_predictions(pred_csv_path: Path) -> Dict[str, str]:
    """
    Create SMILES -> PubChem label map from predictions CSV if it contains any known PubChem columns.
    This does not change any other behavior.
    """
    if not pred_csv_path.exists():
        return {}
    try:
        df = pd.read_csv(pred_csv_path)
    except Exception:
        return {}

    if "canonical_smiles" not in df.columns:
        return {}

    # Common column candidates
    candidates = [
        "pubchem_cid", "PubChemCID", "cid", "CID",
        "pubchem_id", "PubChemID",
        "pubchem_name", "PubChemName", "compound_name", "name",
    ]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        return {}

    m: Dict[str, str] = {}
    for smi, val in zip(df["canonical_smiles"].astype(str).tolist(), df[col].tolist()):
        lab = _normalize_pubchem_label(val)
        if lab:
            m[smi] = lab
    return m


def _get_pubchem_label_for_graph(g: Dict, smi: str, pred_map: Dict[str, str]) -> str:
    """
    Priority:
      1) graph meta keys (common variants)
      2) predictions CSV map (if present)
      3) fallback to short SMILES label
    """
    meta = g.get("meta", {}) if isinstance(g, dict) else {}
    # common keys in graph meta
    keys = [
        "pubchem_cid", "PubChemCID", "cid", "CID",
        "pubchem_id", "PubChemID",
        "pubchem_name", "PubChemName", "compound_name", "name",
    ]
    for k in keys:
        if k in meta:
            lab = _normalize_pubchem_label(meta.get(k))
            if lab:
                return lab

    if smi in pred_map and pred_map[smi].strip():
        return pred_map[smi].strip()

    # fallback
    short = (smi[:42] + "…") if len(smi) > 43 else smi
    return f"SMILES: {short}"


def draw_atom_highlight_svg(
    smiles: str,
    atom_weights: np.ndarray,
    out_path: Path,
    label: str = "",  # <-- ADDED
    width: int = 500,
    height: int = 380
) -> None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return

    # Draw without adding Hs for stable atom mapping
    Chem.rdDepictor.Compute2DCoords(mol)

    w = np.asarray(atom_weights, dtype=float)
    w = np.clip(w, 0.0, 1.0)
    if mol.GetNumAtoms() != w.size:
        return

    highlight_atoms = list(range(mol.GetNumAtoms()))
    atom_colors = {i: (1.0, 1.0 - float(w[i]), 1.0 - float(w[i])) for i in highlight_atoms}

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # Inject label as vector text at top-left (ADDED)
    label = _escape_xml(str(label).strip())
    if label:
        label_svg = (
            f'<text x="12" y="26" font-family="Times New Roman" font-size="18" '
            f'font-weight="bold">{label}</text>\n'
        )
        svg = re.sub(r"(<svg[^>]*>\s*)", r"\1" + label_svg, svg, count=1, flags=re.DOTALL)

    out_path.write_text(svg, encoding="utf-8")


# -----------------------------
# Main per-seed processing
# -----------------------------
def load_graph_cache(path: Path) -> Dict:
    obj = torch.load(path, map_location="cpu")
    if "graphs" not in obj:
        raise ValueError(f"Invalid cache: {path}")
    return obj


def split_graphs(graphs: List[Dict]) -> Dict[str, List[Dict]]:
    out = {"train": [], "val": [], "test": []}
    for g in graphs:
        sp = str(g.get("meta", {}).get("split", ""))
        if sp not in out:
            raise ValueError(f"Unknown split '{sp}' in graph meta")
        out[sp].append(g)
    return out


@torch.no_grad()
def predict_embeddings_and_outputs(
    model: FusionNet,
    graphs: List[Dict],
    fp_arr_map: Dict[str, np.ndarray],
    device: torch.device,
    task: str,
    batch_size: int,
) -> pd.DataFrame:
    rows = []
    for i in range(0, len(graphs), batch_size):
        chunk = graphs[i:i + batch_size]
        batch = collate_graphs_with_fp(chunk, fp_arr_map, device=device, task=task)
        out = model(batch)
        emb = out["embedding"].detach().cpu().numpy()
        for j, smi in enumerate(batch.smiles):
            r = {
                "canonical_smiles": smi,
                "emb_dim": emb.shape[1],
            }
            r["embedding"] = ";".join([f"{v:.6g}" for v in emb[j].tolist()])
            if task in ("classification", "both") and "logits" in out:
                r["y_cls_prob"] = float(torch.sigmoid(out["logits"][j]).detach().cpu().item())
                r["y_cls_true"] = int(batch.y_cls[j].detach().cpu().item()) if batch.y_cls is not None else -1
            if task in ("regression", "both") and "y_reg" in out:
                r["y_reg_pred"] = float(out["y_reg"][j].detach().cpu().item())
                r["y_reg_true"] = float(batch.y_reg[j].detach().cpu().item()) if batch.y_reg is not None else float("nan")
            rows.append(r)
    return pd.DataFrame(rows)


def make_similarity_bins(sim: np.ndarray, n_bins: int = 10) -> np.ndarray:
    sim = np.asarray(sim, dtype=float)
    qs = np.nanquantile(sim, np.linspace(0, 1, n_bins + 1))
    qs = np.maximum.accumulate(qs)
    bins = np.digitize(sim, qs[1:-1], right=True)
    return bins.astype(int)


def plot_similarity_hist(sim: np.ndarray, out_svg: Path, out_png: Path, style: FigStyle, title: str) -> None:
    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_subplot(111)
    ax.hist(sim[~np.isnan(sim)], bins=30, alpha=0.85)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Max Tanimoto similarity to TRAIN", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)
    _save_fig(fig, out_svg, out_png, dpi=style.dpi_png)


def plot_perf_vs_similarity(bin_df: pd.DataFrame, metric: str, out_svg: Path, out_png: Path, style: FigStyle, title: str) -> None:
    fig = plt.figure(figsize=(7.6, 5.8))
    ax = fig.add_subplot(111)
    ax.plot(bin_df["bin_center"], bin_df[metric], marker="o")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Similarity bin center (max Tanimoto to TRAIN)", fontweight="bold")
    ax.set_ylabel(metric, fontweight="bold")
    ax.tick_params(width=style.axis_width)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_width)
    _save_fig(fig, out_svg, out_png, dpi=style.dpi_png)


def main() -> None:
    ap = argparse.ArgumentParser(description="Script-9: Interpretation + AD + error analysis for GNN QSAR (fusion model).")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--model_kind", type=str, choices=["gnn_fusion"], default="gnn_fusion")
    ap.add_argument("--task", type=str, choices=["regression", "classification", "both"], default="both")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fp_radius", type=int, default=2)
    ap.add_argument("--fp_bits", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--ad_similarity_threshold", type=float, default=0.35)
    ap.add_argument("--similarity_bins", type=int, default=10)
    ap.add_argument("--projection", type=str, choices=["umap", "pca"], default="umap")
    ap.add_argument("--topk", type=int, default=15, help="How many compounds to export for hard-error tables and atom attribution.")
    ap.add_argument("--which_attr", type=str, choices=["classification", "regression"], default="classification")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    dirs = ensure_dirs(out_root)

    style = FigStyle()
    style.apply()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    proj_method = args.projection if (args.projection == "pca" or _HAVE_UMAP) else "pca"

    run_manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "out_root": str(out_root),
        "seeds": [int(s) for s in args.seeds],
        "task": args.task,
        "model_kind": args.model_kind,
        "fp_radius": args.fp_radius,
        "fp_bits": args.fp_bits,
        "projection": proj_method,
        "umap_available": bool(_HAVE_UMAP),
        "ad_similarity_threshold": float(args.ad_similarity_threshold),
        "similarity_bins": int(args.similarity_bins),
        "device": str(device),
    }

    all_seed_rows = []

    for seed in args.seeds:
        set_all_seeds(seed)

        cache_path = dirs["features"] / f"graphs_seed{seed:03d}.pt"
        model_path = dirs["models"] / f"gnn_fusion_seed{seed:03d}_best.pt"
        pred_path = dirs["predictions"] / f"gnn_fusion_seed{seed:03d}_test.csv"

        if not cache_path.exists():
            raise FileNotFoundError(f"Missing graph cache: {cache_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions: {pred_path}")

        # PubChem map from predictions (ADDED)
        pred_pubchem_map = _build_pubchem_map_from_predictions(pred_path)

        cache = load_graph_cache(cache_path)
        graphs = cache["graphs"]
        split_map = split_graphs(graphs)
        g_train = split_map["train"]
        g_test = split_map["test"]

        fp_arr_map, fp_bv_map = build_fp_maps(graphs, radius=args.fp_radius, n_bits=args.fp_bits)

        train_smiles = [str(g.get("meta", {}).get("smiles", "")) for g in g_train]
        test_smiles = [str(g.get("meta", {}).get("smiles", "")) for g in g_test]
        train_bv = [fp_bv_map[s] for s in train_smiles]
        max_sim = max_tanimoto_to_train(test_smiles, train_bv, fp_bv_map)

        model, cfg = load_fusion_model(model_path, device=device, task=args.task)

        test_df = predict_embeddings_and_outputs(
            model, g_test, fp_arr_map, device=device, task=args.task, batch_size=args.batch_size
        )

        sim_df = pd.DataFrame({"canonical_smiles": test_smiles, "max_tanimoto_to_train": max_sim})
        test_df = test_df.merge(sim_df, on="canonical_smiles", how="left")

        per_seed_table = dirs["interp_tables"] / f"seed{seed:03d}_test_embeddings_similarity.csv"
        test_df.to_csv(per_seed_table, index=False)

        plot_similarity_hist(
            max_sim,
            dirs["interp_figs"] / f"40_similarity_hist_seed{seed:03d}.svg",
            dirs["interp_figs"] / f"40_similarity_hist_seed{seed:03d}.png",
            style,
            title=f"Test similarity to training set (seed {seed})",
        )

        emb = np.vstack([np.fromstring(s, sep=";") for s in test_df["embedding"].values])
        pts = project_2d(emb, method=proj_method, seed=seed)

        plot_chemical_space(
            pts,
            test_df["max_tanimoto_to_train"].to_numpy(dtype=float),
            title=f"Chemical space (TEST) colored by train-similarity (seed {seed})",
            cbar_label="Max Tanimoto to TRAIN",
            out_svg=dirs["interp_figs"] / f"41_space_similarity_seed{seed:03d}.svg",
            out_png=dirs["interp_figs"] / f"41_space_similarity_seed{seed:03d}.png",
            style=style,
        )

        if args.task in ("classification", "both") and "y_cls_prob" in test_df.columns:
            plot_chemical_space(
                pts,
                test_df["y_cls_prob"].to_numpy(dtype=float),
                title=f"Chemical space (TEST) colored by predicted probability (seed {seed})",
                cbar_label="Predicted probability",
                out_svg=dirs["interp_figs"] / f"42_space_prob_seed{seed:03d}.svg",
                out_png=dirs["interp_figs"] / f"42_space_prob_seed{seed:03d}.png",
                style=style,
            )
        if args.task in ("regression", "both") and "y_reg_pred" in test_df.columns:
            plot_chemical_space(
                pts,
                test_df["y_reg_pred"].to_numpy(dtype=float),
                title=f"Chemical space (TEST) colored by predicted activity (seed {seed})",
                cbar_label="Predicted activity",
                out_svg=dirs["interp_figs"] / f"43_space_regpred_seed{seed:03d}.svg",
                out_png=dirs["interp_figs"] / f"43_space_regpred_seed{seed:03d}.png",
                style=style,
            )

        inside = test_df["max_tanimoto_to_train"].to_numpy(dtype=float) >= float(args.ad_similarity_threshold)
        test_df["inside_AD"] = inside.astype(int)

        bins = make_similarity_bins(test_df["max_tanimoto_to_train"].to_numpy(dtype=float), n_bins=args.similarity_bins)
        test_df["sim_bin"] = bins
        bin_rows = []
        for b in range(args.similarity_bins):
            sub = test_df[test_df["sim_bin"] == b].copy()
            if sub.empty:
                continue
            lo = float(np.nanmin(sub["max_tanimoto_to_train"]))
            hi = float(np.nanmax(sub["max_tanimoto_to_train"]))
            center = 0.5 * (lo + hi)

            row = {"seed": seed, "bin": b, "bin_lo": lo, "bin_hi": hi, "bin_center": center, "n": int(sub.shape[0])}

            if args.task in ("classification", "both") and ("y_cls_true" in sub.columns) and ("y_cls_prob" in sub.columns):
                yy = sub["y_cls_true"].to_numpy()
                pp = sub["y_cls_prob"].to_numpy()
                mask = yy >= 0
                if mask.sum() >= 10 and len(np.unique(yy[mask])) >= 2:
                    metrics_json = dirs["metrics"] / f"gnn_fusion_metrics_seed{seed:03d}.json"
                    mjs = json.loads(metrics_json.read_text(encoding="utf-8")) if metrics_json.exists() else {}
                    thr = float(mjs.get("threshold", 0.5))
                    row.update({f"cls_{k}": v for k, v in cls_metrics(yy[mask], pp[mask], thr).items()})
                else:
                    row.update({f"cls_{k}": float("nan") for k in ["auroc", "pr_auc", "mcc", "bal_acc", "f1"]})

            if args.task in ("regression", "both") and ("y_reg_true" in sub.columns) and ("y_reg_pred" in sub.columns):
                yt = sub["y_reg_true"].to_numpy(dtype=float)
                yp = sub["y_reg_pred"].to_numpy(dtype=float)
                mask = ~np.isnan(yt)
                if mask.sum() >= 10:
                    row.update({f"reg_{k}": v for k, v in reg_metrics(yt[mask], yp[mask]).items()})
                else:
                    row.update({f"reg_{k}": float("nan") for k in ["rmse", "mae", "r2"]})

            bin_rows.append(row)

        bins_df = pd.DataFrame(bin_rows)
        bins_csv = dirs["interp_tables"] / f"44_perf_vs_similarity_bins_seed{seed:03d}.csv"
        bins_df.to_csv(bins_csv, index=False)

        if args.task in ("classification", "both") and "cls_pr_auc" in bins_df.columns:
            plot_perf_vs_similarity(
                bins_df,
                metric="cls_pr_auc",
                out_svg=dirs["interp_figs"] / f"45_pr_auc_vs_similarity_seed{seed:03d}.svg",
                out_png=dirs["interp_figs"] / f"45_pr_auc_vs_similarity_seed{seed:03d}.png",
                style=style,
                title=f"PR-AUC vs similarity bins (seed {seed})",
            )
        if args.task in ("regression", "both") and "reg_rmse" in bins_df.columns:
            plot_perf_vs_similarity(
                bins_df,
                metric="reg_rmse",
                out_svg=dirs["interp_figs"] / f"46_rmse_vs_similarity_seed{seed:03d}.svg",
                out_png=dirs["interp_figs"] / f"46_rmse_vs_similarity_seed{seed:03d}.png",
                style=style,
                title=f"RMSE vs similarity bins (seed {seed})",
            )

        per_seed_errors = test_df.copy()

        if args.task in ("classification", "both") and ("y_cls_true" in per_seed_errors.columns) and ("y_cls_prob" in per_seed_errors.columns):
            yy = per_seed_errors["y_cls_true"].to_numpy()
            pp = per_seed_errors["y_cls_prob"].to_numpy()
            mask = yy >= 0
            metrics_json = dirs["metrics"] / f"gnn_fusion_metrics_seed{seed:03d}.json"
            mjs = json.loads(metrics_json.read_text(encoding="utf-8")) if metrics_json.exists() else {}
            thr = float(mjs.get("threshold", 0.5))
            yhat = (pp >= thr).astype(int)
            per_seed_errors["y_cls_pred"] = yhat
            per_seed_errors["cls_error_type"] = "correct"
            per_seed_errors.loc[(mask) & (yy == 0) & (yhat == 1), "cls_error_type"] = "false_positive"
            per_seed_errors.loc[(mask) & (yy == 1) & (yhat == 0), "cls_error_type"] = "false_negative"

            fps = per_seed_errors[(per_seed_errors["cls_error_type"] == "false_positive")].copy()
            fps = fps.sort_values(["y_cls_prob", "max_tanimoto_to_train"], ascending=[False, True]).head(args.topk)
            fps_out = dirs["interp_tables"] / f"47_hard_false_positives_seed{seed:03d}.csv"
            fps.to_csv(fps_out, index=False)

            fns = per_seed_errors[(per_seed_errors["cls_error_type"] == "false_negative")].copy()
            fns = fns.sort_values(["y_cls_prob", "max_tanimoto_to_train"], ascending=[True, True]).head(args.topk)
            fns_out = dirs["interp_tables"] / f"48_hard_false_negatives_seed{seed:03d}.csv"
            fns.to_csv(fns_out, index=False)

        if args.task in ("regression", "both") and ("y_reg_true" in per_seed_errors.columns) and ("y_reg_pred" in per_seed_errors.columns):
            yt = per_seed_errors["y_reg_true"].to_numpy(dtype=float)
            yp = per_seed_errors["y_reg_pred"].to_numpy(dtype=float)
            ae = np.abs(yt - yp)
            per_seed_errors["abs_error"] = ae
            big = per_seed_errors[~np.isnan(per_seed_errors["abs_error"])].copy()
            big = big.sort_values(["abs_error", "max_tanimoto_to_train"], ascending=[False, True]).head(args.topk)
            big_out = dirs["interp_tables"] / f"49_largest_regression_errors_seed{seed:03d}.csv"
            big.to_csv(big_out, index=False)

        # Atom attribution (selected test compounds)
        attr_rows = []
        if args.which_attr == "classification" and args.task in ("classification", "both") and ("y_cls_prob" in test_df.columns):
            cand = test_df.copy()
            cand = cand.sort_values(["max_tanimoto_to_train", "y_cls_prob"], ascending=[True, False]).head(args.topk)
            chosen_smiles = cand["canonical_smiles"].tolist()
        elif args.which_attr == "regression" and args.task in ("regression", "both") and ("y_reg_pred" in test_df.columns):
            cand = test_df.copy()
            cand = cand.sort_values(["max_tanimoto_to_train", "y_reg_pred"], ascending=[True, False]).head(args.topk)
            chosen_smiles = cand["canonical_smiles"].tolist()
        else:
            chosen_smiles = []

        test_map = {str(g.get("meta", {}).get("smiles", "")): g for g in g_test}
        for smi in chosen_smiles:
            g = test_map.get(smi, None)
            if g is None:
                continue
            n_nodes = int(g["x"].shape[0])
            if not sanity_check_atom_counts(smi, n_nodes):
                continue

            try:
                atom_imp, extra = compute_atom_importance_for_single(
                    model, g, fp_arr_map, device=device, task=args.task,
                    which=("classification" if args.which_attr == "classification" else "regression"),
                )
            except Exception:
                continue

            atom_csv = dirs["interp_csvs"] / f"atom_importance_seed{seed:03d}_{args.which_attr}_{hash(smi) & 0xffffffff:08x}.csv"
            pd.DataFrame({
                "canonical_smiles": [smi] * len(atom_imp),
                "atom_index": np.arange(len(atom_imp), dtype=int),
                "atom_importance_0_1": atom_imp.astype(float),
            }).to_csv(atom_csv, index=False)

            # Save SVG highlight (NOW with PubChem label)
            svg_path = dirs["interp_svgs"] / f"atom_importance_seed{seed:03d}_{args.which_attr}_{hash(smi) & 0xffffffff:08x}.svg"
            pubchem_label = _get_pubchem_label_for_graph(g, smi, pred_pubchem_map)
            draw_atom_highlight_svg(smi, atom_imp, svg_path, label=pubchem_label)

            row = {
                "seed": seed,
                "canonical_smiles": smi,
                "which_attr": args.which_attr,
                "atom_csv": str(atom_csv),
                "atom_svg": str(svg_path),
                "pubchem_label": pubchem_label,  # added column for traceability
                **extra,
            }
            attr_rows.append(row)

        attr_index = pd.DataFrame(attr_rows)
        attr_index_out = dirs["interp_tables"] / f"50_atom_attribution_index_seed{seed:03d}.csv"
        attr_index.to_csv(attr_index_out, index=False)

        test_df["seed"] = seed
        all_seed_rows.append(test_df)

    merged = pd.concat(all_seed_rows, ignore_index=True)
    merged_out = dirs["interp_tables"] / "seed_all_test_embeddings_similarity_merged.csv"
    merged.to_csv(merged_out, index=False)

    ad_thr = float(args.ad_similarity_threshold)
    ad_rows = []
    for seed in args.seeds:
        sub = merged[merged["seed"] == seed].copy()
        inside = sub["max_tanimoto_to_train"].to_numpy(dtype=float) >= ad_thr
        row = {"seed": seed, "ad_threshold": ad_thr, "n_test": int(sub.shape[0]), "n_inside_AD": int(inside.sum())}

        if args.task in ("classification", "both") and ("y_cls_true" in sub.columns) and ("y_cls_prob" in sub.columns):
            yy = sub["y_cls_true"].to_numpy()
            pp = sub["y_cls_prob"].to_numpy()
            mask = (yy >= 0) & inside
            metrics_json = (out_root / "metrics" / f"gnn_fusion_metrics_seed{seed:03d}.json")
            mjs = json.loads(metrics_json.read_text(encoding="utf-8")) if metrics_json.exists() else {}
            thr = float(mjs.get("threshold", 0.5))
            if mask.sum() >= 10 and len(np.unique(yy[mask])) >= 2:
                row.update({f"insideAD_cls_{k}": v for k, v in cls_metrics(yy[mask], pp[mask], thr).items()})
        if args.task in ("regression", "both") and ("y_reg_true" in sub.columns) and ("y_reg_pred" in sub.columns):
            yt = sub["y_reg_true"].to_numpy(dtype=float)
            yp = sub["y_reg_pred"].to_numpy(dtype=float)
            mask = (~np.isnan(yt)) & inside
            if mask.sum() >= 10:
                row.update({f"insideAD_reg_{k}": v for k, v in reg_metrics(yt[mask], yp[mask]).items()})
        ad_rows.append(row)

    ad_df = pd.DataFrame(ad_rows)
    ad_out = dirs["interp_tables"] / "51_inside_AD_performance_by_seed.csv"
    ad_df.to_csv(ad_out, index=False)

    run_manifest["outputs"] = {
        "interpretability_root": str(dirs["interp_root"]),
        "merged_test_table": str(merged_out),
        "inside_AD_table": str(ad_out),
    }
    (dirs["interp_reports"] / "manifest_script9.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    report_lines = []
    report_lines.append("Script-9 completed.\n")
    report_lines.append(f"Projection method: {proj_method} (UMAP available: {bool(_HAVE_UMAP)})\n")
    report_lines.append(f"AD threshold (max Tanimoto to TRAIN): {ad_thr:.3f}\n")
    report_lines.append(f"Outputs root: {dirs['interp_root']}\n")
    (dirs["interp_reports"] / "report_script9.txt").write_text("".join(report_lines), encoding="utf-8")

    print("\n=== DONE: Script-9 (Interpretability + AD + error analysis) ===")
    print(f"Outputs folder:      {dirs['interp_root']}")
    print(f"Merged test table:   {merged_out}")
    print(f"Inside-AD summary:   {ad_out}")
    print(f"Manifest:            {dirs['interp_reports'] / 'manifest_script9.json'}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
