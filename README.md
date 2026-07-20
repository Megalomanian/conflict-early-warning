# Unsupervised Risk Forecasting of Conflict Escalation in Social Media Comment Streams

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](./main.pdf)
[![Slides](https://img.shields.io/badge/Slides-HTML-orange)](./slides.html)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

**Authors**: Linli Zhu (朱林立)<sup>1</sup>, Ziqiang Ma (马自强)<sup>2</sup>  
<sup>1</sup> School of Computer Science, Ningxia University  
<sup>2</sup> School of Information Engineering, Ningxia University  
📧 violina2333@gmail.com · maziqiang@nxu.edu.cn

## Overview

An **unsupervised** framework for risk forecasting of conflict escalation in social media comment streams. The framework combines **attack intensity**, **negative high-arousal emotion**, and **stance polarization** into a conflict index without manual annotation, then uses deep sequential forecasters for short-horizon temporal prediction.

### Key Results (V2 — strict temporal split, 5 seeds)

| Metric | Value |
|--------|-------|
| CNN-BiLSTM R² | **0.812 ± 0.008** |
| BiGRU R² | **0.811 ± 0.009** |
| Transformer R² | **0.806 ± 0.007** |
| Best Esc-F1 (CNN-BiLSTM) | **0.718** |
| Lead Time (mean ± std) | **-1.7 ± 3.2 bins** |
| Early Warning Rate (lead>0) | **21%** |

### Architecture

```
Raw Comments → Conflict Index → Bin-Level Trajectory → CNN-BiLSTM → Risk Monitoring
            (3 signals, unsupervised)   (top-k aggregation)   (L=12, H=6)   (5 trigger rules)
```

## Repository Structure

```
├── main.tex / main_cn.tex    # English & Chinese paper (IEEEtran)
├── refs.bib                  # Bibliography (29 entries)
├── slides.html               # reveal.js presentation
├── compile.sh                # Build: ./compile.sh [en|cn|clean]
├── AGENTS.md                 # Contributor guide
│
├── run_experiments_v2.py           # **Primary**: synthetic + full baselines (temporal split, 5 seeds)
├── experiment_real_model_v2.py     # **Primary**: BERT conflict index + CNN-BiLSTM on Zhihu data
├── case_study.py                   # Weibo reversal event case study (V2 fixed)
│
├── run_experiments.py / experiment_real_model.py  # Deprecated (random-split leakage)
├── run_contributions.py / c3_leadtime.py          # Deprecated
├── experiment.py / experiment_extended.py         # Deprecated
│
├── figures/                  # Paper figures (8 PDFs)
├── experiment_results/       # Output .pkl files from experiments
├── reading_list/             # Curated literature + summaries
├── zhihu_topics/             # Zhihu dataset (140 topics) + weibo_reversal/ sub-dataset
│
├── pyproject.toml            # Python deps (uv-managed)
├── .vscode/settings.json     # LaTeX Workshop auto-build on save (Tectonic)
└── .latexmkrc                # Alternative XeLaTeX+BibTeX config
```

**⚠️ `run_experiments_v2.py` and `experiment_real_model_v2.py` are the authoritative scripts.** Older scripts without `_v2` use random-split data leakage and are deprecated.

## Build & Run

### Paper

```bash
./compile.sh en          # English → main.pdf (Tectonic, auto BibTeX)
./compile.sh cn          # Chinese → main_cn.pdf
./compile.sh clean       # Remove auxiliary files
```

Requires **Tectonic** at `~/.local/bin/tectonic`.

### Experiments

```bash
uv sync                                          # Install dependencies
uv run python3 run_experiments_v2.py             # Primary experiment (GPU recommended, ~90 min)
uv run python3 experiment_real_model_v2.py       # Real-data experiment (GPU required)
uv run python3 case_study.py                     # Case study
```

**Requirements**: Python 3.13+, PyTorch 2.13+, Transformers, Sentence-Transformers, XGBoost, scikit-learn, statsmodels.

### Presentation

Open `slides.html` in any browser — powered by [reveal.js](https://revealjs.com/). Press `F` for fullscreen, `?` for shortcuts.

## Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| **Zhihu** (知乎) | Chinese Q&A platform comment streams | 140 topics, 45K comments |
| **Synthetic** | Logistic escalation events + seasonality + pink noise | 60 trajectories × 200 bins |
| **Weibo Reversal** | Public opinion reversal events (Zhang et al. 2023) | 27 events, 245K comments |

## Baselines & Results (V2)

| Model | Type | R² (mean ± std) | Esc-F1 |
|-------|------|-----------------|--------|
| Persistence | Statistical | 0.638 ± 0.015 | 0.637 |
| AR(6) | Statistical | 0.715 ± 0.011 | 0.644 |
| SVR (RBF) | ML | 0.759 ± 0.024 | 0.667 |
| XGBoost | ML | 0.781 ± 0.013 | 0.684 |
| TCN | Deep | 0.787 ± 0.010 | 0.695 |
| Informer-Lite | Deep | 0.796 ± 0.012 | 0.662 |
| BiLSTM | Deep | 0.805 ± 0.011 | 0.711 |
| Transformer + PE | Deep | 0.806 ± 0.007 | 0.697 |
| BiGRU | Deep | 0.811 ± 0.009 | 0.716 |
| **CNN-BiLSTM** | **Ours** | **0.812 ± 0.008** | **0.718** |

Results under strict temporal ordering (no shuffling), ECDF fitted on training data only, 5 random seeds.

## Citation

```bibtex
@article{zhu2025conflict,
  title={Unsupervised Risk Forecasting of Conflict Escalation in Social Media Comment Streams},
  author={Zhu, Linli and Ma, Ziqiang},
  journal={IEEE Transactions on Computational Social Systems},
  year={2025},
  note={Under review}
}
```

## License

MIT License — see [LICENSE](./LICENSE) file.
