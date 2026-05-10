# Graph Neural Network-Based Structure-Function Analysis of ACC2 Enzyme Interactions with Food-Derived Compounds

## Study overview
This repository contains the computational workflow used to prioritize food-derived compounds with predicted ACC2 interaction potential in a nutritional biochemistry context. The pipeline starts from curated ACC2 bioactivity records, builds scaffold-split machine-learning and deep-learning models, performs uncertainty-aware screening on FoodDB compounds, and prepares analysis outputs for downstream structural and biological follow-up.

## Scientific aim
The goal is computational prioritization of candidate ACC2 binders that may inform lipid metabolic regulation hypotheses and support later biochemical and nutritional validation.

The repository supports:
- ChEMBL ACC2 bioactivity curation
- pIC50 regression and activity classification
- scaffold-based splitting
- Morgan fingerprint baseline models
- graph neural-network models
- graph-Morgan fusion models
- validation-only probability calibration
- interpretability and applicability-domain analysis
- FoodDB screening and ranking
- docking prioritization (manuscript-described downstream step)
- molecular dynamics analysis (manuscript-described downstream step)
- MM-GBSA binding free-energy estimation (manuscript-described downstream step)
- post-screening scaffold and uncertainty analysis

## Repository structure
Current source repository layout:

```text
Code/
├── 00_data_audit_and_curation.py
├── 01_scaffold_split_generation.py
├── 02_graph_dataset_builder.py
├── 03_baselines_fingerprint_models.py
├── 04_gnn_train_and_evaluate.py
├── 05_gnn_fusion_morgan_train_and_evaluate.py
├── 06_multi_seed_aggregate_and_report.py
├── 06b_probability_calibration_val_only.py
├── 07_generate_manuscript_results_nc.py
├── 08_interpretability_applicability_domain_error_analysis.py
├── 09_build_final_submission_package.py
├── 09a_screen_01_refine_library.py
├── 09b_screen_02_make_graph_cache.py
├── 09c_screen_02_build_fp_and_index.py
├── 10_screen_03_ensemble_infer_rank.py
├── 11_screen_04_postscreen_analysis.py
├── extract_acc2_chembl36_sqlite.py
├── qsar_postprocess.py
├── README.md
├── README.txt
└── docs/
    ├── workflow.md
    ├── scripts.md
    └── reproducibility.md
```

Typical generated output layout after running the workflow:

```text
<out_root>/
├── data/
├── splits/
├── features/
├── models/
├── predictions/
├── metrics/
├── tables/
├── figures/
├── reports/
├── release_gnn_reporting/
├── interpretability_ad/
├── screen/
│   ├── features/
│   ├── output/
│   └── analysis/
└── submission_package_NC/
```

## Workflow summary
1. Curate ACC2 records and remove invalid/duplicate structures.
2. Generate scaffold splits across fixed random seeds.
3. Build graph caches and Morgan fingerprints.
4. Train/evaluate baseline, GNN, and graph-Morgan fusion models for pIC50 and Active labels.
5. Aggregate multi-seed performance and run paired statistical comparisons.
6. Apply validation-only probability calibration when required.
7. Generate interpretability and applicability-domain outputs.
8. Refine FoodDB library, run ensemble inference, and rank candidates.
9. Produce post-screening scaffold/uncertainty/property analyses and final package files.

## Data inputs
Primary expected inputs:
- ACC2 activity records with canonical_smiles and activity columns
- ChEMBL target identifier: CHEMBL4829
- FoodDB screening library table with SMILES column

Utility preprocessing scripts are available for local ChEMBL SQLite extraction and pIC50 post-processing.

## Environment and dependencies
Core Python dependencies used across scripts:
- Python 3.9+
- pandas, numpy, matplotlib
- rdkit
- scikit-learn
- torch
- xgboost (optional fallback logic exists in baseline script)
- umap-learn (optional in Script 08)

Example environment setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas matplotlib scikit-learn torch rdkit-pypi xgboost umap-learn
```

## How to run each stage
All scripts are CLI-driven. Replace placeholders such as <out_root>, <seed>, and input file paths with local values.

### Optional ACC2 extraction and post-processing
```bash
python extract_acc2_chembl36_sqlite.py \
  --db /path/to/chembl_36.db \
  --target CHEMBL4829 \
  --outdir ./raw_acc2

python qsar_postprocess.py \
  --input ./raw_acc2/CHEMBL4829_qsar_ready.csv \
  --outdir ./processed_acc2 \
  --endpoint IC50 \
  --threshold 6.0 \
  --aggregate best \
  --svg --png
```

### Core modeling workflow
```bash
python 00_data_audit_and_curation.py \
  --input_csv /path/to/acc2_input.csv \
  --out_root <out_root> \
  --label_mode both \
  --agg_pic50 max

python 01_scaffold_split_generation.py \
  --input_csv <out_root>/data/03_dedup.csv \
  --out_root <out_root> \
  --n_seeds 3 \
  --base_seed 2025
```

For each scaffold split seed:

```bash
python 02_graph_dataset_builder.py \
  --dedup_csv <out_root>/data/03_dedup.csv \
  --split_csv <out_root>/splits/scaffold_seed<seed>.csv \
  --out_root <out_root> \
  --task both

python 03_baselines_fingerprint_models.py \
  --dedup_csv <out_root>/data/03_dedup.csv \
  --split_csv <out_root>/splits/scaffold_seed<seed>.csv \
  --out_root <out_root> \
  --task both

python 04_gnn_train_and_evaluate.py \
  --graph_cache <out_root>/features/graphs_seed<seed>.pt \
  --out_root <out_root> \
  --task both \
  --device cuda \
  --tune_trials 20

python 05_gnn_fusion_morgan_train_and_evaluate.py \
  --graph_cache <out_root>/features/graphs_seed<seed>.pt \
  --dedup_csv <out_root>/data/03_dedup.csv \
  --out_root <out_root> \
  --task both \
  --device cuda \
  --tune_trials 20
```

Aggregate across seeds:

```bash
python 06_multi_seed_aggregate_and_report.py \
  --out_root <out_root> \
  --seeds 2025 2026 2027 \
  --task both \
  --make_overlays
```

Optional calibration:

```bash
python 06b_probability_calibration_val_only.py \
  --out_root <out_root> \
  --seed 2025 \
  --model_key gnn_fusion \
  --method platt
```

Reporting and interpretability:

```bash
python 07_generate_manuscript_results_nc.py \
  --out_root <out_root> \
  --task both \
  --target_name ACC2 \
  --activity_name pIC50 \
  --split_name "scaffold split" \
  --seeds 2025 2026 2027

python 08_interpretability_applicability_domain_error_analysis.py \
  --out_root <out_root> \
  --seeds 2025 2026 2027 \
  --model_kind gnn_fusion \
  --task both \
  --projection umap \
  --device cuda

python 09_build_final_submission_package.py \
  --out_root <out_root> \
  --package_name submission_package_NC \
  --copy_predictions \
  --make_aliases
```

### FoodDB screening workflow
```bash
python 09a_screen_01_refine_library.py \
  --in_file /path/to/FoodDB_subset.tsv \
  --out_root <out_root> \
  --sep "\t" \
  --smiles_col moldb_smiles \
  --id_col id \
  --name_col name \
  --write_dropped

python 09b_screen_02_make_graph_cache.py \
  --refined_csv <out_root>/screen/01_screen_refined.csv \
  --out_root <out_root> \
  --seed 2025

python 09c_screen_02_build_fp_and_index.py \
  --refined_csv <out_root>/screen/01_screen_refined.csv \
  --graphs_pt <out_root>/screen/02_graphs_seed2025_screen.pt \
  --out_root <out_root> \
  --smiles_col canonical_smiles \
  --id_col id \
  --name_col name \
  --fp_radius 2 \
  --fp_bits 2048

python 10_screen_03_ensemble_infer_rank.py \
  --out_root <out_root> \
  --seeds 2025 2026 2027 \
  --task classification \
  --topN 5000 \
  --device cuda

python 11_screen_04_postscreen_analysis.py \
  --out_root <out_root> \
  --task classification \
  --topN 5000 \
  --score_col ens_prob_mean \
  --unc_col ens_prob_std \
  --smiles_col canonical_smiles
```

## Scripts
A complete script-by-script description (purpose, inputs, outputs, workflow position, and usage status) is provided in docs/scripts.md.

Status labels used:
- core workflow script
- optional analysis
- utility script
- legacy script

## Expected outputs
Main output groups:
- Curated datasets: data/*.csv
- Scaffold splits: splits/scaffold_seed*.csv
- Graph/fingerprint features: features/*.pt, *.npz and screen/features/*
- Model checkpoints: models/*.pt and baseline *.pkl
- Predictions: predictions/*.csv and screen/output/*.csv
- Calibration outputs: calibrated predictions, metrics, reliability figures
- Applicability-domain outputs: interpretability_ad/tables, interpretability_ad/figures, atom_svgs
- FoodDB screening outputs: screen/output and screen/analysis
- Manuscript package outputs: release_gnn_reporting and submission_package_NC

## Figure and table outputs
Common destinations:
- figures/: per-script training/evaluation figures
- tables/: per-script analysis tables
- release_gnn_reporting/tables and release_gnn_reporting/figures: multi-seed summary outputs
- interpretability_ad/figures and interpretability_ad/tables: interpretability and AD outputs
- screen/analysis/figures and screen/analysis/tables: post-screening analysis outputs
- submission_package_NC: consolidated review package

## Reproducibility notes
Reproducibility controls are implemented through:
- scaffold-based split definitions
- explicit seed arguments across training and aggregation steps
- validation-only threshold selection for classification
- separate test-set evaluation after model selection
- seed-wise ensemble inference and cross-seed uncertainty estimates
- optional validation-only calibration (Platt or isotonic)

## Contact
Corresponding author (manuscript):
Nada A. Alzunaidy
Department of Food Science and Human Nutrition, Qassim University
Email: n.alznedy@qu.edu.sa
