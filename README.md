# Weakly Supervised Early Warning of Conflict Escalation in Social Media Comment Streams

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](./main.pdf)
[![Slides](https://img.shields.io/badge/Slides-HTML-orange)](./slides.html)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

**Authors**: Linli Zhu (朱林立)<sup>1</sup>, Ziqiang Ma (马自强)<sup>2</sup>  
<sup>1</sup> School of Computer Science, Ningxia University  
<sup>2</sup> School of Information Engineering, Ningxia University  
📧 violina2333@gmail.com · maziqiang@nxu.edu.cn

## Overview

A weakly supervised LSTM-based framework for early warning of conflict escalation in social media comment streams. The framework combines **attack intensity**, **negative high-arousal emotion**, and **stance polarization** into a conflict index, then uses a **CNN-BiLSTM** hybrid forecaster for short-horizon temporal prediction.

### Key Results

| Metric | Value |
|--------|-------|
| Synthetic R² | **0.833** |
| Synthetic Esc-F1 | **0.710** |
| Real Zhihu Volume R² | **0.831** |
| Case Study (Huolala) Esc-F1 | **0.889** |
| Best Trigger F1 (Threshold) | **0.688** @ 2.9% alert rate |
| Lead Time | **+0.3 ± 1.6 bins** |

### Architecture

```
Raw Comments → Conflict Index → Bin-Level Trajectory → CNN-BiLSTM → Early Warning
                  (3 signals)      (top-k aggregation)   (L=12, H=6)   (5 trigger rules)
```

## Repository Structure

```
├── main.tex                  # English paper (IEEEtran journal)
├── main_cn.tex               # Chinese version (partial)
├── refs.bib                  # Bibliography (26 entries)
├── slides.html               # Presentation slides (reveal.js)
├── compile.sh                # Build script
│
├── experiment_real_model.py  # Main experiment: BERT conflict index + CNN-BiLSTM
├── run_experiments.py        # Full pipeline: synthetic data + all baselines
├── experiment_extended.py    # Extended model comparison
├── experiment.py             # Keyword-based conflict index (original)
├── case_study.py             # Weibo reversal event case study
├── run_contributions.py      # C1/C2/C3 contribution validation
├── c3_leadtime.py            # Lead time analysis
├── fix_figure.py             # Figure generation with CJK fonts
│
├── figures/                  # Paper figures (8 PDFs)
├── experiment_results/       # Experiment outputs (.pkl files)
├── reading_list/             # Curated literature with summaries
│
├── zhihu_topics/             # Real-world Zhihu dataset (140 topics)
│   └── weibo_reversal/       # Weibo public opinion reversal dataset
│
└── .vscode/                  # VS Code LaTeX Workshop config
```

## Build & Run

### Paper

```bash
# English version
./compile.sh en

# Chinese version
./compile.sh cn

# Clean auxiliary files
./compile.sh clean
```

Uses **Tectonic** (Rust-based LaTeX engine) with automatic BibTeX and package fetching.

### Experiments

```bash
# Install dependencies
uv sync

# Run with GPU (recommended)
uv run python3 run_experiments.py

# Run without GPU
python3 run_experiments.py
```

**Requirements**: Python 3.10+, PyTorch 2.5+, Transformers, Sentence-Transformers, XGBoost, scikit-learn.

### Presentation

Open `slides.html` in any browser — powered by [reveal.js](https://revealjs.com/) (MIT license).  
Press `F` for fullscreen, `Esc` for overview, `?` for shortcuts.

## Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| **Zhihu** (知乎) | Chinese Q&A platform comment streams | 140 topics, 45K comments |
| **Synthetic** | Logistic escalation events + seasonality + pink noise | 60 trajectories × 200 bins |
| **Weibo Reversal** | Public opinion reversal events with annotated windows | 27 events, 245K comments |

The Weibo dataset is from Zhang et al. (2023), *Information Studies: Theory & Application*.

## Baselines

| Model | Type | R² | Esc-F1 |
|-------|------|-----|--------|
| Persistence | Statistical | 0.640 | 0.638 |
| AR(6) | Statistical | 0.712 | 0.628 |
| SVR (RBF) | ML | 0.784 | 0.663 |
| XGBoost | ML | 0.793 | 0.679 |
| BiGRU | Deep | 0.829 | 0.698 |
| TCN | Deep | 0.819 | 0.651 |
| Transformer | Deep | 0.532 | 0.138 |
| **CNN-BiLSTM** | **Ours** | **0.833** | **0.710** |

## Citation

```bibtex
@article{zhu2025conflict,
  title={Weakly Supervised Early Warning of Conflict Escalation in Social Media Comment Streams},
  author={Zhu, Linli and Ma, Ziqiang},
  journal={IEEE Transactions on Computational Social Systems},
  year={2025},
  note={Under review}
}
```

## License

MIT License — see [LICENSE](./LICENSE) file.
