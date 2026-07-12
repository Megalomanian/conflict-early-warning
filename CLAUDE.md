# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

IEEE conference/journal paper about a weakly supervised LSTM-based framework for early warning of conflict escalation in social media comment streams. The framework combines attack intensity, negative high-arousal emotion, and stance polarization into a conflict index, then uses LSTM for temporal forecasting.

## Build

```bash
# English version (main.tex)
./compile.sh en

# Chinese version (main_cn.tex)
./compile.sh cn

# Clean auxiliary files
./compile.sh clean
```

Compilation uses **Tectonic** at `~/.local/bin/tectonic`, a Rust-based LaTeX engine with automatic package fetching and BibTeX. VS Code LaTeX Workshop (`.vscode/settings.json`) is configured to auto-build on save using Tectonic — saving any `.tex` file triggers recompilation. The viewer opens PDFs in a VS Code tab.

An alternative is `latexmk` (configured in `.latexmkrc` for XeLaTeX), though `compile.sh` uses Tectonic by default.

## File structure

- `main.tex` — English paper, `\documentclass[journal]{IEEEtran}`. Requires XeLaTeX for Unicode/font support (TeX Gyre Termes, Noto Serif CJK SC, JetBrains Mono). Bibliography via `refs.bib`. This is the **primary and most complete** version.
- `main_cn.tex` — Chinese version, `\documentclass[conference]{IEEEtran}`, uses `ctex` package. **Partial translation**: contains only Section III (Methods) and lacks Introduction, Related Work, Experiments, and bibliography entirely. No `\cite` commands or `\bibliography` block. When working on the Chinese version, content from `main.tex` sections I, II, and IV will need to be translated.
- `refs.bib` — Shared bibliography (BibTeX format). All entries are cited in `main.tex` except **`Tong2025MEUV`** which is unused — remove or cite before submission.
- `figures/` — Directory for figures (currently empty).
- `compile.sh` — Build script; pass `en`, `cn`, or `clean`.
- `.latexmkrc` — Alternative XeLaTeX + BibTeX build configuration (not used by `compile.sh`).
- `.vscode/settings.json` — LaTeX Workshop config: Tectonic auto-build on save, PDF viewer in tab.
- `reading_list/README.md` — Curated reading list organized by topic (conflict prediction, weakly-supervised signals, stance detection, LSTM forecasting, cascade prediction, early-warning systems, surveys).
- `reading_list/summary.md` — Detailed summaries of each paper in the reading list, including architecture descriptions, key results, and specific guidance on how each paper relates to this work. **Consult this before writing Related Work or looking for additional citations.**

## Writing guidelines

- The paper uses `\noindent\textbf{(I-C1) ...}` style for listing contributions and `\noindent\textbf{RQ1:}` for research questions.
- Sections IV (Experiments) has placeholder headers with **no content yet** in subsections B–H: Dataset and Preprocessing, Implementation Details, Baselines, Evaluation Metrics, Main Results, Ablation Studies, and Early-Warning Case Analysis. These are the main areas needing writing.
- English version uses `\documentclass[journal]{IEEEtran}`; Chinese version uses `\documentclass[conference]{IEEEtran}` — note the different document class options.
- Mathematical notation: conflict index uses `c_{t,i}`, bin-level aggregation uses `\bar{c}_t`, LSTM forecast uses `\hat{c}_{t+h}`.
- Algorithms use `\usepackage{algorithm}` + `\usepackage{algorithmic}`.
- Font configuration in `main.tex` uses `\AtBeginDocument` to override IEEEtran's default `ptm` (Times) settings — do not remove this block; removing it causes `TU/ptm` font errors under XeLaTeX.
