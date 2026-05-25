# 📄 Academic Paper — News Topic Cataloging Pipeline

**Title:** *Hierarchical News Topic Cataloging via Iterative Graph Clustering and LLM-Augmented Assessment: A Comprehensive Study*

## Paper Files

|| File | Description |
||------|-------------|
|| `paper.tex` | Full LaTeX source (381 lines, 2-column journal format) |
|| `fig1_architecture.tex` | **NEW** TikZ architecture figure — 3-layer pipeline |
|| `fig2_evolution_v2.svg` | **IMPROVED** Research evolution w/ coverage + LLM costs |
|| `fig3_benchmark_v2.svg` | **IMPROVED** Quantitative benchmark (6×4 grouped bars) |
|| `fig4_resolution_v2.svg` | **IMPROVED** Resolution scan w/ dual-axis (comm. + modularity) |
|| `fig7_grades_v2.svg` | **IMPROVED** Grade distribution (bar + donut combined) |
|| `fig_dataset_stats.svg` | **NEW** Dataset summary statistics & distribution |
|| `fig2_evolution.svg` | Legacy evolution timeline |
|| `fig3_benchmark.svg` | Legacy benchmark chart |
|| `fig4_resolution.svg` | Legacy resolution scan |
|| `fig5_sizedist.svg` | Community size distribution (log-log) |
|| `fig6_fsm.svg` | Pipeline finite-state machine |
|| `fig7_grades.svg` | Legacy grade distribution |
|| `fig8_hierarchical_tree.svg` | Hierarchical topic tree (3 groups) |
|| `fig9_ablation.svg` | Optimization ablation study |
|| `fig10_comparison.svg` | Comprehensive method comparison table |
|| `architecture.svg.html` | System architecture (interactive SVG) |
|| `architecture.png` | Rendered architecture (1280×900) |

## Paper Structure

1. **Abstract** — 150-word summary
2. **§1 Introduction** — Motivation, 3 contributions
3. **§2 Related Work** — LDA, BERTopic, Leiden, LLM annotation
4. **§3 Methodology** — Dataset, 13-experiment program, final pipeline (3 phases)
5. **§4 Experiments** — Final performance, benchmark, in-depth analysis, ablation
6. **§5 Discussion** — Why Leiden, Rule vs LLM, limitations
7. **§6 Conclusion** — Key findings, future work
8. **References** — 10 citations

## Key Coverage

✅ 13 experiments (A→K + Final)  
✅ 6 clustering paradigms benchmarked  
✅ 4 evaluation metrics  
✅ Hierarchical tree visualization  
✅ Complete ablation study  
✅ FSM architecture with all equations  
✅ 10 figures + 2 tables  
✅ $k$-NN, Leiden, GSDMM, BERTopic formulas  
✅ Deterministic rule mathematical definitions  

## How to Compile

Upload to Overleaf: compile with `pdflatex` (2 runs for refs).

```bash
xelatex paper.tex && xelatex paper.tex
```
