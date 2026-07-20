# Unsupervised Risk Forecasting of Conflict Escalation in Social Media Comment Streams

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](./main.pdf)
[![Slides](https://img.shields.io/badge/Slides-HTML-orange)](./slides.html)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

**Authors**: Linli Zhu (朱林立)<sup>1</sup>, Ziqiang Ma (马自强)<sup>2</sup>  
<sup>1</sup> School of Computer Science, Ningxia University  
<sup>2</sup> School of Information Engineering, Ningxia University  
📧 violina2333@gmail.com · maziqiang@nxu.edu.cn

## Overview

An **unsupervised** framework for risk forecasting of conflict escalation in social media comment streams. The framework constructs a comment-level conflict index from **attack intensity**, **negative high-arousal emotion**, and **stance polarization** without manual annotation, then uses deep sequential forecasters for short-horizon temporal prediction. The paper evaluates this framework across three complementary scenarios, reporting both positive and negative results under strict temporal-split protocols.

### Key Findings (V2 — strict temporal split)

| Scenario | Target | R² | Takeaway |
|----------|--------|-----|----------|
| Synthetic conflict trajectories | Conflict index | **0.809 ± 0.010** | Architecture works under ideal conditions |
| Weibo reversal events (27 pooled) | Comment volume | **0.715** | Forecaster generalizes to real dense events |
| Zhihu discussions (20 topics) | Conflict index | **≈ 0** | Text-derived conflict at 12h bins is unpredictable |

**Core insight**: The conflict index reveals that text-based conflict signals at coarse temporal granularity lack short-term predictability. This negative result is informative—it defines boundary conditions that random-split protocols systematically obscure.

### Architecture

```
Raw Comments → Conflict Index → Bin-Level Trajectory → Deep Forecaster → Risk Monitoring
            (3 signals)           (top-k aggregation)    (L=12, H=6)
```

## Repository Structure

```
├── main.tex / main_cn.tex    # English & Chinese paper (IEEEtran)
├── srep_submission/           # Scientific Reports version
├── refs.bib                  # Bibliography (29 entries)
├── slides.html               # reveal.js presentation
├── compile.sh                # Build: ./compile.sh [en|cn|clean]
├── AGENTS.md                 # Contributor guide
│
├── run_experiments_v2.py           # Primary: synthetic + full baselines (temporal split, 5 seeds)
├── experiment_real_model_v2.py     # BERT conflict index + CNN-BiLSTM on Zhihu data
├── case_study.py                   # Weibo reversal event case study (V2 fixed)
├── eval_triggers.py                # Trigger rule evaluation
│
├── reproduce_competitors/          # Demo: random vs temporal split inflation
│
├── run_experiments.py / experiment_real_model.py  # Deprecated (random-split leakage)
│
├── figures/                  # Paper figures (8 PDFs)
├── experiment_results_v2/    # V2 experiment outputs (.pkl files)
├── reading_list/             # Curated literature + summaries
├── zhihu_topics/             # Zhihu dataset (140 topics) + weibo_reversal/ sub-dataset
│
├── pyproject.toml            # Python deps (uv-managed)
└── .vscode/settings.json     # LaTeX Workshop auto-build on save (Tectonic)
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
uv run python3 run_experiments_v2.py             # Primary experiment (GPU, ~90 min, 5 seeds)
uv run python3 experiment_real_model_v2.py       # Real-data conflict index (GPU required)
uv run python3 eval_triggers.py                  # Trigger evaluation
uv run python3 reproduce_competitors/demo_random_vs_temporal.py  # Leakage demo
```

**Requirements**: Python 3.13+, PyTorch 2.13+, Transformers, Sentence-Transformers, XGBoost, scikit-learn.

### Presentation

Open `slides.html` in any browser — powered by [reveal.js](https://revealjs.com/). Press `F` for fullscreen, `?` for shortcuts.

## Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| **Synthetic** | Logistic escalation events + seasonality + pink noise | 60 trajectories × 200 bins |
| **Zhihu** (知乎) | Chinese Q&A platform, sparse comment streams | 140 topics, 45K comments |
| **Weibo Reversal** | Public opinion reversal events, dense commenting | 27 events, 245K comments |

## Baselines & Results (V2 — Synthetic)

| Model | Type | R² (mean ± std) | Esc-F1 |
|-------|------|-----------------|--------|
| Persistence | Statistical | 0.638 ± 0.015 | 0.637 |
| Moving Avg (k=3) | Statistical | 0.540 ± 0.020 | 0.547 |
| Exp. Smoothing | Statistical | 0.498 ± 0.022 | 0.430 |
| AR(6) | Statistical | 0.715 ± 0.011 | 0.644 |
| SVR (RBF) | ML | 0.759 ± 0.024 | 0.667 |
| XGBoost | ML | 0.781 ± 0.013 | 0.684 |
| TCN | Deep | 0.787 ± 0.010 | 0.695 |
| Informer-Lite | Deep | 0.796 ± 0.014 | 0.659 |
| BiLSTM | Deep | 0.805 ± 0.011 | 0.711 |
| Transformer + PE | Deep | 0.806 ± 0.007 | 0.697 |
| **BiGRU** | Deep | **0.811 ± 0.009** | **0.716** |
| **CNN-BiLSTM (Ours)** | Deep | **0.809 ± 0.010** | **0.710** |

Results under strict temporal ordering (no shuffling), ECDF fitted on training data only, 5 independent seeds. BiGRU and CNN-BiLSTM are statistically indistinguishable (ΔR² = -0.002, within 1σ).

## Real-Data Results (V2 — Temporal Split)

| Dataset | Target | R² |
|---------|--------|-----|
| Weibo 27 events, 1h bins | Comment volume (pooled) | **0.715** |
| Zhihu 20 topics, 12h bins | Conflict index | ≈ 0 |
| Zhihu 20 topics (Persistence) | Conflict index | -0.90 |
| Zhihu 20 topics (AR(6)) | Conflict index | -0.00 |

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
