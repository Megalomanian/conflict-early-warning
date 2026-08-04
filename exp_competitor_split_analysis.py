#!/usr/bin/env python3
"""Competitor-protocol reproduction: random split vs temporal split.

Published opinion-forecasting papers (e.g. IPSO-LSTM-style setups) commonly
split pooled windows randomly (k-fold without temporal blocking). Because
successive windows overlap and windows from the same series appear in both
partitions, this leaks the recent past into the "test" set and inflates R².

This script trains the same models under BOTH protocols on (a) synthetic
trajectories with known event regimes and (b) the real Zhihu conflict-index
trajectories, and reports the R² / Esc-F1 inflation of random vs temporal
splitting:
  - RANDOM:  all windows pooled and shuffled, 70/30 split
  - TEMPORAL: per-series first-70%-of-windows vs the rest (no cross-window leak)
Models: LSTM (SimpleLSTM, the architecture used in competitor papers),
CNN-BiLSTM (our model) and XGBoost (per-horizon).
"""
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

from experiment_real_model_v2 import CNNBiLSTM, gen_traj  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L, H = 12, 6
R_DIR = Path("experiment_results_v2")


class SimpleLSTM(nn.Module):
    """Single-layer LSTM used in the competitor-style demo."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, 1, batch_first=True)
        self.proj = nn.Linear(64, H)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.proj(o[:, -1, :])


def make_windows(series_list):
    X, y = [], []
    for s in series_list:
        s = np.asarray(s, dtype=float)
        for i in range(len(s) - L - H + 1):
            X.append(s[i:i + L])
            y.append(s[i + L:i + L + H])
    return np.asarray(X, dtype=np.float32).reshape(len(X), L, 1), \
        np.asarray(y, dtype=np.float32)


def split_random(X, y, frac=0.7, seed=0):
    rng = np.random.RandomState(seed)
    n = len(X)
    idx = rng.permutation(n)
    cut = int(n * frac)
    return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]


def split_temporal_per_series(series_list, frac=0.7):
    """Per-series temporal split: first frac of each series' windows = train."""
    Xtr, ytr, Xte, yte = [], [], [], []
    for s in series_list:
        X, y = make_windows([s])
        cut = int(len(X) * frac)
        Xtr.append(X[:cut]); ytr.append(y[:cut])
        Xte.append(X[cut:]); yte.append(y[cut:])
    return (np.concatenate(Xtr), np.concatenate(ytr),
            np.concatenate(Xte), np.concatenate(yte))


def fit_lstm(model_factory, Xtr, ytr, Xte, yte, epochs=100, patience=20):
    Xtr_d = torch.tensor(Xtr).to(DEVICE)
    ytr_d = torch.tensor(ytr).to(DEVICE)
    Xte_d = torch.tensor(Xte).to(DEVICE)
    yte_d = torch.tensor(yte).to(DEVICE)
    loader = DataLoader(TensorDataset(Xtr_d, ytr_d), 64, True)
    model = model_factory().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    huber = nn.HuberLoss(delta=0.5)
    best_val, best_state, patience_count = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            huber(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = huber(model(Xte_d[:200]), yte_d[:200]).item()
        if val < best_val:
            best_val = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        yp = model(Xte_d).cpu().numpy()
    return yp


def fit_xgb(Xtr, ytr, Xte):
    flat = Xtr.reshape(len(Xtr), -1)
    preds = []
    for h in range(H):
        m = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                         n_jobs=-1, random_state=42)
        m.fit(flat, ytr[:, h])
        preds.append(m.predict(Xte.reshape(len(Xte), -1)))
    return np.stack(preds, axis=1)


def metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    esc_pred = (y_pred.max(1) >= 0.65).astype(int)
    esc_true = (y_true.max(1) >= 0.65).astype(int)
    tp = (esc_pred & esc_true).sum()
    fp = (esc_pred & (1 - esc_true)).sum()
    fn = ((1 - esc_pred) & esc_true).sum()
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"r2": r2, "mae": mae, "esc_f1": float(f1), "n_test": len(y_true)}


def evaluate_protocols(name, series_list, models=("LSTM", "CNN-BiLSTM", "XGBoost")):
    X_all, y_all = make_windows(series_list)
    Xtr_r, ytr_r, Xte_r, yte_r = split_random(X_all, y_all)
    Xtr_t, ytr_t, Xte_t, yte_t = split_temporal_per_series(series_list)
    results = {protocol: {} for protocol in ("random", "temporal")}
    for model_name in models:
        if model_name == "LSTM":
            factory = SimpleLSTM
        elif model_name == "CNN-BiLSTM":
            factory = lambda: CNNBiLSTM()  # noqa: E731
        else:
            factory = None
        for protocol, (Xtr, ytr, Xte, yte) in {
                "random": (Xtr_r, ytr_r, Xte_r, yte_r),
                "temporal": (Xtr_t, ytr_t, Xte_t, yte_t)}.items():
            if factory is not None:
                yp = fit_lstm(factory, Xtr, ytr, Xte, yte)
            else:
                yp = fit_xgb(Xtr, ytr, Xte)
            results[protocol][model_name] = metrics(yte, yp)
    return results


def load_real_series():
    with (R_DIR / "trajectories_real_model.pkl").open("rb") as f:
        traj = pickle.load(f)
    series = {}
    for topic, frame in traj.items():
        s = frame["c_bar"].to_numpy(dtype=float)
        mean, std = s.mean(), s.std()
        series[topic] = (s - mean) / (std or 1.0)
    return series


def main():
    print(f"Device: {DEVICE}")
    output = {"protocols": "random (competitor-style pooled shuffle) vs "
                           "temporal (per-series first-70%)",
              "note": "z-score fitted on each full series; Esc-F1 uses the "
                      "0.65 peak-exceedance threshold of the v2 pipeline"}

    # Synthetic trajectories with event regimes
    print("\n═══ Synthetic data ═══")
    synth = [gen_traj(n=200, s=seed)[0] for seed in range(12)]
    output["synthetic"] = evaluate_protocols("synthetic", synth)
    for protocol, rows in output["synthetic"].items():
        for model, m in rows.items():
            print(f"  {protocol:8s} {model:10s} R²={m['r2']:.4f} "
                  f"Esc-F1={m['esc_f1']:.4f} (n={m['n_test']})")

    # Real Zhihu conflict-index trajectories
    print("\n═══ Real Zhihu trajectories ═══")
    real = load_real_series()
    output["real_zhihu"] = evaluate_protocols("real_zhihu", list(real.values()))
    for protocol, rows in output["real_zhihu"].items():
        for model, m in rows.items():
            print(f"  {protocol:8s} {model:10s} R²={m['r2']:.4f} "
                  f"Esc-F1={m['esc_f1']:.4f} (n={m['n_test']})")

    # Inflation summary
    print("\n═══ Random-split R² inflation (random − temporal) ═══")
    summary = {}
    for dataset in ("synthetic", "real_zhihu"):
        summary[dataset] = {}
        for model in ("LSTM", "CNN-BiLSTM", "XGBoost"):
            d = (output[dataset]["random"][model]["r2"]
                 - output[dataset]["temporal"][model]["r2"])
            summary[dataset][model] = round(d, 4)
            print(f"  {dataset:10s} {model:10s} ΔR² = {d:+.4f}")
    output["r2_inflation_random_minus_temporal"] = summary

    out = R_DIR / "competitor_split_analysis_results.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    with (R_DIR / "competitor_split_analysis_results.pkl").open("wb") as f:
        pickle.dump(output, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
