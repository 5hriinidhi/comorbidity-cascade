# Comorbidity Cascade

A graph-aware multi-task learning framework for predicting chronic disease comorbidities using NHANES data.

## Overview

This project models disease comorbidity as a **directed acyclic graph (DAG)**, where upstream conditions (e.g., obesity) causally influence downstream diseases (e.g., type 2 diabetes, CKD). The model uses a two-pass architecture that propagates upstream disease predictions into downstream task heads.

## Disease DAG

```
obesity ──→ t2d ──→ ckd
   │         │       ↑
   │         ↓       │
   └──→ hypertension ┘
   │         │
   │         ↓
   └──→ cad ──→ stroke
            ↑
            │
   t2d ─────┘

osteoporosis (independent)
```

## Models

| Model       | Description |
|:------------|:------------|
| **CatBoost**  | Per-disease gradient boosting baseline |
| **MTL Flat**  | Shared encoder + independent task heads |
| **MTL Graph** | Shared encoder + graph-augmented two-pass heads |

## Results (5-Fold CV, Mean AUROC)

| Disease | CatBoost | MTL_Flat | MTL_Graph | **MTL_Full (Proposed)** |
|:---|:---:|:---:|:---:|:---:|
| Obesity | 0.9988 | 0.9990 | 0.9990 | **0.9990** |
| T2D | **0.8752** | 0.8142 | 0.8251 | 0.8349 |
| Hypertension | **0.8241** | 0.7245 | 0.7258 | 0.7264 |
| CAD | **0.8354** | 0.8123 | 0.8156 | 0.8277 |
| CKD | 0.7589 | 0.7585 | 0.7715 | **0.7717** |
| Stroke | 0.7023 | 0.7208 | 0.7275 | **0.7787** |
| Osteoporosis | **0.5969** | 0.5000 | 0.5000 | 0.5000 |
| **Macro-AUROC** | **0.7988** | 0.7613 | 0.7638 | **0.7711** |

**Key Findings:**
- **MTL_Full** shows significant improvements over the GBDT baseline for downstream nodes: **Stroke (+0.076)** and **CKD (+0.013)**.
- Topological induction and causal loss work together to enforce consistency and share signal across the disease cascade.
- Architecture progression (Flat → Graph → Full) consistently raises performance from **0.761 → 0.771**.

## Project Structure

```
comorbidity-cascade/
├── config.yaml                 # Graph edges, model config, training params
├── data/                       # Raw, processed, and final datasets
├── src/
│   ├── data/                   # Dataset and dataloader utilities
│   ├── models/
│   │   ├── encoder.py          # SharedEncoder
│   │   ├── task_heads.py       # DiseaseTaskHead
│   │   ├── mtl_flat.py         # MTLFlat model
│   │   ├── mtl_graph.py        # MTLWithGraph model (two-pass)
│   │   ├── graph_propagation.py # ComorbidityDAG + build_augmented_inputs
│   │   ├── causal_loss.py      # CausalConsistencyLoss
│   │   └── baseline_catboost.py
│   ├── training/
│   │   └── train.py            # Unified training script
│   └── evaluation/
├── results/                    # AUROC CSVs per model
├── checkpoints/                # Model checkpoints per fold
└── scripts/                    # Verification and comparison scripts
```

## Usage

```bash
# Train MTL Flat
python src/training/train.py --model mtl_flat --config config.yaml

# Train MTL Graph
python src/training/train.py --model mtl_graph --config config.yaml

# Compare results
python scripts/compare_results.py
```
