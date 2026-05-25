# Hierarchical News Cataloging via Graph Partitioning and LLM Self-Supervision

Course project: unsupervised + self-supervised news headline cataloging pipeline.

## Quick Start

```bash
pip install sentence-transformers leidenalg scikit-learn numpy pandas
python pipeline.py
```

## Pipeline Overview

1. **CLUSTER** — k-NN graph + Leiden community detection (29 initial topics)
2. **FSM Loop** — Rule Scorer (6 rules) + LLM Judge (MiniMax-M2.7) grades A/B/C/D
3. **SPLIT** — Recursive sub-clustering with dynamic resolution γ
4. **EXPORT** — 3-level tree → `tree.json`, `catalog.csv`, `summary.json`

**Results:** 94.3% coverage, 585 leaf topics, zero outliers, 60% LLM cost reduction.

## Key Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Main pipeline orchestrator (FSM) |
| `catalog_controller.py` | Catalog state management & export |
| `catalog_state.py` | Catalog data structures |
| `rule_scorer.py` | 6 deterministic rules (R1-R6) |
| `llm_judge.py` | LLM grade assignment (A/B/C/D) |
| `split_engine.py` | Sub-clustering with dynamic γ |
| `cluster_leiden_fast.py` | Leiden community detection |
| `experiment_*.py` | Experiments A-K (see paper §3.3) |
| `benchmark_all_methods.py` | Benchmark all 5 competing methods |
| `paper/paper.tex` | Final paper (LaTeX) |

## Dataset

[A Million News Headlines](https://www.kaggle.com/datasets/therohk/million-headlines) (Kaggle) — 1.2M ABC News headlines 2003-2019, 50K sampled.
