# Unsupervised Risk Forecasting of Conflict Escalation in Social Media Comment Streams

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](./main.pdf)
[![中文说明](https://img.shields.io/badge/README-中文-red)](./README_CN.md)
[![Slides](https://img.shields.io/badge/Slides-HTML-orange)](./slides.html)

This repository contains the paper, data-processing pipeline, leakage-free experiments, and reproducibility artifacts for forecasting conflict escalation in social-media comment streams. The upstream conflict index is **unsupervised**: it combines attack intensity, negative high-arousal emotion, and stance polarization without conflict labels. Downstream forecasting is evaluated separately as continuous regression and high-risk-state classification.

## Main Findings

| Scenario | Task | Held-out result | Interpretation |
|---|---|---:|---|
| Synthetic trajectories | Continuous forecasting | CNN-BiLSTM R² **0.809 ± 0.010** | The architecture learns controlled dynamics |
| Weibo, 27 reversal events | Comment-volume forecasting | R² **0.715** | Positive result on dense real event streams |
| Zhihu, 20 topics | Raw conflict-index forecasting | R² **≈ 0** | Exact sparse text-derived trajectories are not reliably predictable |
| Zhihu, final 25% | Cleaned high-risk classification | F1 **0.778**, recall **0.822** | Better F1/recall than persistence, but not better AUC |

The result is deliberately bounded: on Zhihu, logistic regression reaches AUC 0.891 versus 0.896 for persistence. The contribution is improved detection balance and recall, not universal superiority or accurate point forecasting.

## Experimental Architecture

```text
Raw comments
  ├─ Chinese BERT sentiment → attack intensity + high-arousal negativity
  └─ MiniLM embeddings → train-only K-means → stance polarization
             ↓
Per-topic, train-only ECDF calibration
             ↓
Weighted conflict score (0.5 attack + 0.3 emotion + 0.2 stance)
             ↓
Top-15% aggregation in regular time bins → conflict trajectory
             ↓
Temporal windows (12 observed bins → next 6 bins)
  ├─ continuous forecasters: statistical, ML, and neural baselines
  └─ risk classifier: causal state-history features → logistic regression
```
The proposed CNN-BiLSTM uses two 1-D convolution layers (32 channels; kernels 3 and 5), a two-layer bidirectional LSTM (64 hidden units per direction, dropout 0.2), and a linear six-step output head. The 12-step context and six-step horizon correspond to six days of history and three days ahead for 12-hour Zhihu bins.

## Leakage-Free Evaluation Protocol

The authoritative V2 pipeline preserves temporal causality:

1. Each topic is placed on a regular time grid and divided chronologically into 60% train, 15% validation, and 25% final test segments.
2. K-means, per-topic `QuantileTransformer` ECDFs, winsorization, imputation, scaling, and risk thresholds are fitted from training observations only.
3. Validation F1 selects the causal EWMA span, feature group, logistic `C`, and decision threshold. The selected configuration is span 24 bins (12 days), state-only features, `C=0.1`, and threshold 0.4.
4. Windows crossing a split boundary are discarded. Samples are never shuffled, and the final test segment is not used for model selection.
5. `preprocessing_audit.json` records the fit boundary for every evaluated topic; all 20 topics report zero test comments used during preprocessing fit.

The final Zhihu classification set contains 1,428 training, 680 validation, and 1,706 test windows. Topic-level bootstrap estimates give logistic-minus-persistence ΔF1 **+0.062** (95% CI 0.018–0.108) and Δrecall **+0.241** (0.173–0.317); AUC is slightly lower, ΔAUC **−0.0046** (−0.0079 to −0.0013). F1 improves on 15/20 topics and recall on 18/20.

## Repository Layout

```text
main.tex, main_cn.tex              Paper sources
srep_submission/                   Scientific Reports submission version
run_experiments_v2.py              Synthetic multi-seed benchmark (authoritative)
experiment_real_model_v2.py        Text signals and real-data forecasting (authoritative)
optimize_real_signal_v2.py         Leakage-free Zhihu risk-model selection
case_study.py, eval_triggers.py     Weibo case study and warning rules
figures/                            Publication figures
experiment_results_v2/             Metrics, audits, logs, and serialized outputs
zhihu_topics/                       Zhihu and Weibo-reversal datasets
reading_list/                       Literature notes
```

Scripts without the `_v2` suffix are deprecated because they use random splits that leak future information. Do not use their outputs in the paper.

## Setup and Reproduction

The supported environment is Python 3.13+ managed by [`uv`](https://docs.astral.sh/uv/). CUDA is detected automatically; CPU fallback is available but substantially slower for transformer encoding and neural baselines.

```bash
uv sync
uv run python3 run_experiments_v2.py
uv run python3 experiment_real_model_v2.py
uv run python3 optimize_real_signal_v2.py
uv run python3 case_study.py
uv run python3 eval_triggers.py
```

Run `experiment_real_model_v2.py` before `optimize_real_signal_v2.py`; the latter consumes `experiment_results_v2/trajectories_real_model.pkl`. Hugging Face models must be cached or downloadable. Set `HF_HUB_OFFLINE=1` to force cached-only loading.

Key reproducibility outputs are:

- `aggregated_results.pkl` and `seed_results.pkl`: five-seed synthetic metrics and individual runs.
- `preprocessing_audit.json`: topic-level leakage audit.
- `signal_cleaning_results.json`: selected hyperparameters, test metrics, sensitivity analysis, and bootstrap intervals.
- `leakage_free_gpu_run_20260803.tar.gz`: archived leakage-free GPU run, including logs and data products.

## Baselines and Metrics

The synthetic benchmark compares persistence, moving average, exponential smoothing, AR(6), SVR, XGBoost, TCN, Informer-Lite, BiLSTM, Transformer, BiGRU, and CNN-BiLSTM over five seeds. It reports R², MAE, RMSE, escalation F1, and event-based lead time. Continuous real-data evaluation additionally reports the persistence reference; classification reports AUC, precision, recall, and F1.

At the primary 80th-percentile risk definition, logistic regression obtains precision 0.738, recall 0.822, and F1 0.778. Persistence obtains precision 0.926, recall 0.584, and F1 0.716. Sensitivity checks at the 70th, 80th, and 90th percentiles yield logistic F1 values of 0.779, 0.778, and 0.761, respectively.

## Build the Paper

```bash
./compile.sh en       # main.tex → main.pdf
./compile.sh cn       # main_cn.tex → main_cn.pdf
./compile.sh clean    # remove auxiliary files
```

Compilation uses Tectonic at `~/.local/bin/tectonic` with automatic BibTeX handling. Before committing paper changes, rebuild the relevant PDF and check that citations, figures, and tables resolve.

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
