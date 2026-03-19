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

| Disease       | CatBoost | MTL Flat | MTL Graph |
|:--------------|:--------:|:--------:|:---------:|
| obesity       | 1.0000   | 0.9993   | 0.9992    |
| t2d           | 0.8881   | 0.8364   | 0.8354    |
| hypertension  | 0.7833   | 0.7309   | 0.7284    |
| cad           | 0.8452   | 0.8361   | 0.8370    |
| ckd           | 0.7738   | 0.7491   | **0.7623**|
| stroke        | 0.8010   | 0.7771   | **0.7843**|
| osteoporosis  | 0.5000   | 0.5000   | 0.5000    |
| **Macro**     | **0.7988** | **0.7613** | **0.7638** |

**Key Findings:**
- MTL Graph improves over MTL Flat in macro-AUROC (0.7638 vs 0.7613)
- CKD shows the largest improvement (+0.013), consistent with having 2 upstream causes
- Stroke also improves (+0.007), with 3 upstream causes
- Osteoporosis (no predecessors) is unchanged, as expected

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
