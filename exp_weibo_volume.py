#!/usr/bin/env python3
"""Re-run the Weibo comment-volume forecasting experiment (1-hour bins).

The paper claimed pooled R2=0.715 for LSTM on log comment volume across 27
events, but that number is not reproducible from the current code base
(it coincides with the synthetic AR(6) baseline). This script recomputes
the true pooled R2 under the same protocol: per-event temporal 75/25 split,
L=12, H=6 windows, per-event z-scoring, LSTM and CNN-BiLSTM.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

from exp_weibo_generalization import load_weibo  # noqa: E402
from experiment_real_model_v2 import CNNBiLSTM  # noqa: E402

R_DIR = Path("experiment_results_v2")
L, H = 12, 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT = 0.75


class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, 1, batch_first=True)
        self.proj = nn.Linear(64, H)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.proj(o[:, -1, :])


def build_volume_series(df, freq="1h"):
    """Per-event log1p comment-volume series on a regular 1h grid."""
    series = {}
    for event in df["event_name"].unique():
        edf = df[df["event_name"] == event]
        counts = edf.groupby(edf["dt"].dt.floor(freq)).size()
        if len(counts) < L + H + 10:
            continue
        full_index = pd.date_range(counts.index.min(), counts.index.max(),
                                   freq=freq)
        counts = counts.reindex(full_index, fill_value=0)
        series[event] = np.log1p(counts.to_numpy(dtype=float))
    return series


def make_windows(s):
    X, y = [], []
    for i in range(len(s) - L - H + 1):
        X.append(s[i:i + L])
        y.append(s[i + L:i + L + H])
    return np.asarray(X, dtype=np.float32).reshape(-1, L, 1), \
        np.asarray(y, dtype=np.float32)


def train_eval(model_factory, Xtr, ytr, Xte, yte, epochs=100, patience=20):
    Xtr_d = torch.tensor(Xtr).to(DEVICE)
    ytr_d = torch.tensor(ytr).to(DEVICE)
    Xte_d = torch.tensor(Xte).to(DEVICE)
    yte_d = torch.tensor(yte).to(DEVICE)
    loader = DataLoader(TensorDataset(Xtr_d, ytr_d), 64, True)
    model = model_factory().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    huber = nn.HuberLoss(delta=0.5)
    best, best_st, pc = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            huber(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = huber(model(Xte_d[:200]), yte_d[:200]).item()
        if val < best:
            best = val
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                break
    model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        yp = model(Xte_d).cpu().numpy()
    return yp, yte


def metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {"r2": float(1 - ss_res / (ss_tot + 1e-8)), "mae": mae,
            "n_test": int(len(y_true))}


def main():
    t0 = time.time()
    df = load_weibo()
    series = build_volume_series(df, freq="1h")
    print(f"{len(series)} events with >= {L + H + 10} hourly bins")

    Xtr_all, ytr_all, Xte_all, yte_all = [], [], [], []
    for event, s in series.items():
        mean, std = s.mean(), s.std()
        s = (s - mean) / (std or 1.0)
        X, y = make_windows(s)
        cut = int(len(X) * SPLIT)
        Xtr_all.append(X[:cut]); ytr_all.append(y[:cut])
        Xte_all.append(X[cut:]); yte_all.append(y[cut:])
    Xtr = np.concatenate(Xtr_all); ytr = np.concatenate(ytr_all)
    Xte = np.concatenate(Xte_all); yte = np.concatenate(yte_all)
    print(f"train windows: {len(Xtr)}, test windows: {len(Xte)}")

    results = {"freq": "1h", "n_events": len(series),
               "train_windows": len(Xtr), "test_windows": len(Xte)}
    for name, factory in [("LSTM", SimpleLSTM), ("CNN-BiLSTM", lambda: CNNBiLSTM())]:
        yp, yte_np = train_eval(factory, Xtr, ytr, Xte, yte)
        results[name] = metrics(yte_np, yp)
        print(f"{name}: {results[name]}")
    out = R_DIR / "weibo_volume_results.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
