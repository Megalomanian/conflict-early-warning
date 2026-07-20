# Repository Guidelines

## Project Structure & Module Organization

```
├── main.tex / main_cn.tex    # LaTeX paper (English primary, Chinese partial)
├── refs.bib                  # Shared BibTeX bibliography
├── compile.sh                # Build script: ./compile.sh [en|cn|clean]
├── slides.html               # reveal.js presentation
│
├── run_experiments_v2.py           # **Primary**: synthetic + full baselines (temporal split, multi-seed)
├── experiment_real_model_v2.py     # **Primary**: BERT conflict index + CNN-BiLSTM on Zhihu data
├── case_study.py                   # Weibo reversal event case study
├── run_experiments.py / run_contributions.py / c3_leadtime.py  # Deprecated (random-split leakage)
├── experiment.py / experiment_real_model.py / experiment_extended.py  # Deprecated
│
├── figures/                  # Paper figures (8 PDFs)
├── experiment_results/       # Output .pkl files from experiments
├── reading_list/             # Curated literature + detailed summaries (README.md, summary.md)
├── zhihu_topics/             # Zhihu dataset (140 topics) + weibo_reversal/ sub-dataset
│
├── pyproject.toml            # Python deps (uv-managed)
├── .vscode/settings.json     # LaTeX Workshop auto-build on save (Tectonic)
└── .latexmkrc                # Alternative XeLaTeX+BibTeX config (not used by compile.sh)
```

**Key rule**: `run_experiments_v2.py` and `experiment_real_model_v2.py` are the authoritative experiment scripts. The older scripts (without `_v2`) contain random-split data leakage and are deprecated for paper results.

## Build, Test, and Development Commands

```bash
# --- Paper ---
./compile.sh en          # Compile English paper → main.pdf (Tectonic, auto BibTeX)
./compile.sh cn          # Compile Chinese paper → main_cn.pdf
./compile.sh clean       # Remove LaTeX auxiliary files

# --- Python environment ---
uv sync                  # Install all dependencies from pyproject.toml
uv run python3 run_experiments_v2.py           # Run primary experiment (GPU recommended)
uv run python3 experiment_real_model_v2.py     # Run real-data experiment
uv run python3 case_study.py                   # Run case study
```

- Requires **Tectonic** at `~/.local/bin/tectonic` for LaTeX compilation.
- Requires **Python 3.13+** and a CUDA-capable GPU for deep learning models (falls back to CPU).
- VS Code with LaTeX Workshop auto-builds on save (configured in `.vscode/settings.json`). Saving any `.tex` file triggers recompilation.

## Coding Style & Naming Conventions

- **Language**: Python 3.13, formatted with no specific auto-formatter. Keep imports grouped: stdlib → numpy/pandas → torch → sklearn/xgboost → matplotlib.
- **Constants**: `UPPER_CASE` at module top (e.g., `L, H = 12, 6`, `HIDDEN = 64`, `DEVICE = "cuda"`).
- **Functions/variables**: `snake_case`. Classes: `PascalCase`.
- **Section dividers**: Use `# ═══ Section Name ═══` comments for visual separation in scripts.
- **Plots**: Use `matplotlib.use("Agg")`, IEEE-compatible rcParams (serif font, 9pt, 150/300 dpi).
- **LaTeX**: Use `\noindent\textbf{(I-C1) ...}` for contributions, `\noindent\textbf{RQ1:}` for research questions. Do not remove the `\AtBeginDocument` font override block in `main.tex` — it prevents `TU/ptm` errors under XeLaTeX.

## Testing Guidelines

This is a research paper repository; there is no formal test suite. Validation is done by running experiments and verifying:

- Run `run_experiments_v2.py` and check output R² and Esc-F1 in `experiment_results_v2/`.
- Results must report **mean ± std over 5+ random seeds** (`N_SEEDS = 5`).
- All train/test splits must use **temporal ordering** (first 75% bins = train), not random shuffling.
- All ECDF calibration (`QuantileTransformer`) must be **fitted on training data only**, per topic.

## Commit & Pull Request Guidelines

- **Commit style**: `Action: short description` — e.g., `Fix: paper revisions`, `Add: Transformer baseline`, `Audit: BERT case study`.
- Keep commits focused: paper text changes, experiment code, and data should be in separate commits.
- Rebuild the PDF (`./compile.sh en`) and verify it compiles cleanly before committing LaTeX changes.
- Before submitting: audit `refs.bib` for uncited entries.

## Terminology & Paper Conventions

- Use **unsupervised** (not "weakly supervised") throughout the paper text.
- Conflict index notation: `c_{t,i}` for per-comment, `\bar{c}_t` for bin-level, `\hat{c}_{t+h}` for forecast.
- The English paper uses `\documentclass[journal]{IEEEtran}`; the Chinese paper uses `\documentclass[conference]{IEEEtran}`.

## Environment & Tooling

- **Package manager**: `uv` with `pyproject.toml`. Dependencies include PyTorch, Transformers, Sentence-Transformers, XGBoost, scikit-learn, statsmodels.
- **LaTeX engine**: Tectonic (Rust-based, auto-fetches packages, auto BibTeX). Alternative: `latexmk` with XeLaTeX (see `.latexmkrc`).
- **Fonts**: TeX Gyre Termes (body), Noto Serif CJK SC (Chinese), JetBrains Mono (monospace). XeLaTeX required for Unicode support.
