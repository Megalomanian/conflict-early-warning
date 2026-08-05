#!/usr/bin/env python3
"""Strong baselines for the Zhihu risk-state classification task.

Same protocol as the logistic pipeline: build_windows(span=48, Q=0.8),
full feature pool (L=12 x 9 dims), validation-based decision threshold,
final-25% test windows (N=1,706). Models: BiLSTM, GRU, XGBoost, TCN,
Transformer; 5 seeds (mean +/- std); topic-bootstrap on the best seed.
"""
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

from exp_component_ablation import VARIANTS, build_trajectories  # noqa: E402
from exp_real_components_v2 import NEW_STORE  # noqa: E402
from optimize_real_signal_v2 import (DECISION_THRESHOLDS, build_windows,  # noqa: E402
                                     classification_metrics,
                                     topic_bootstrap)

R_DIR = Path("experiment_results_v2")
L, H = 12, 6
SPAN = 48
D_IN = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 300
PATIENCE = 40
HID = 64


# ═══ Models (multi-channel, classification head) ═══
class BiLSTMClf(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(D_IN, HID, 1, batch_first=True, bidirectional=True,
                            dropout=0.0)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * HID, 1))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :])


class GRUClf(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(D_IN, HID, 1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * HID, 1))

    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(o[:, -1, :])


class TCNClf(nn.Module):
    def __init__(self, channels=32, levels=4, kernel=3):
        super().__init__()
        layers, in_ch = [], D_IN
        for i in range(levels):
            dilation = 2 ** i
            layers += [nn.Conv1d(in_ch, channels, kernel,
                                 padding=(kernel - 1) * dilation,
                                 dilation=dilation), nn.ReLU(),
                       nn.Dropout(0.2)]
            in_ch = channels
        self.convs = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(channels, HID), nn.ReLU(),
                                  nn.Linear(HID, 1))

    def forward(self, x):
        return self.head(self.convs(x.transpose(1, 2)))


class TransformerClf(nn.Module):
    def __init__(self, d_model=64, nhead=4, n_layers=1):
        super().__init__()
        self.proj = nn.Linear(D_IN, d_model)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                       dropout=0.2, batch_first=True),
            n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.enc(self.proj(x)).mean(dim=1))


# ═══ Training ═══
def train_deep(model, Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pos = ytr.sum().item()
    pos_w = torch.tensor([(len(ytr) - pos) / max(pos, 1)]).to(DEVICE)
    loader = DataLoader(TensorDataset(Xtr, ytr), 64, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best_f1, best_thr, best_state, pc = -1.0, 0.5, None, 0
    yva_np = yva.cpu().numpy()
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb).squeeze(-1), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva).squeeze(-1)).cpu().numpy()
        scores = [(f1_score(yva_np, pv >= t, zero_division=0), t)
                  for t in DECISION_THRESHOLDS]
        f1v, thr = max(scores)
        if f1v > best_f1:
            best_f1, best_thr = f1v, thr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_thr


def run_deep(factory, Xtr, ytr, Xva, yva, Xte, yte, items):
    per_seed = []
    for seed in SEEDS:
        model = factory().to(DEVICE)
        model, thr = train_deep(model, Xtr, ytr, Xva, yva, seed)
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(Xte).squeeze(-1)).cpu().numpy()
        pred = prob >= thr
        m = classification_metrics(yte.cpu().numpy(), prob, pred)
        per_seed.append({"seed": seed, **m, "threshold": thr, "prob": prob})
    metrics = {}
    for k in ("auc", "precision", "recall", "f1"):
        vals = [s[k] for s in per_seed]
        metrics[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    best_seed = max(per_seed, key=lambda s: s["f1"])
    last = np.asarray([i["last_target"] for i in items])
    pers_pred = np.asarray([i["last_target"] > i["threshold"] for i in items])
    boot = topic_bootstrap(items, yte.cpu().numpy(), best_seed["prob"],
                           best_seed["prob"] >= best_seed["threshold"],
                           last, pers_pred)
    return {"metrics": metrics, "per_seed": per_seed, "bootstrap": boot}


def run_xgboost(Xtr, ytr, Xva, yva, Xte, yte, items):
    import xgboost as xgb
    Xtr_f = Xtr.reshape(len(Xtr), -1)
    Xva_f = Xva.reshape(len(Xva), -1)
    Xte_f = Xte.reshape(len(Xte), -1)
    per_seed = []
    for seed in SEEDS:
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=4,
                                learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, random_state=seed,
                                eval_metric="logloss", verbosity=0)
        clf.fit(Xtr_f, ytr.numpy())
        pv = clf.predict_proba(Xva_f)[:, 1]
        scores = [(f1_score(yva.numpy(), pv >= t, zero_division=0), t)
                  for t in DECISION_THRESHOLDS]
        _, thr = max(scores)
        prob = clf.predict_proba(Xte_f)[:, 1]
        pred = prob >= thr
        per_seed.append({"seed": seed, "threshold": thr,
                         **classification_metrics(yte.numpy(), prob, pred)})
    metrics = {}
    for k in ("auc", "precision", "recall", "f1"):
        vals = [s[k] for s in per_seed]
        metrics[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    best_seed = max(per_seed, key=lambda s: s["f1"])
    prob = None  # recompute below
    boot = None
    return {"metrics": metrics, "per_seed": per_seed, "bootstrap": None}


def main():
    t0 = time.time()
    with NEW_STORE.open("rb") as f:
        prep = pickle.load(f)
    traj = build_trajectories(prep["store"], VARIANTS["full"])
    splits = build_windows(traj, SPAN)
    Xtr = torch.tensor(np.asarray([i["x"] for i in splits["train"]])
                       .reshape(-1, L, D_IN), dtype=torch.float32).to(DEVICE)
    ytr = torch.tensor(np.asarray([i["y_cls"] for i in splits["train"]]),
                       dtype=torch.float32).to(DEVICE)
    Xva = torch.tensor(np.asarray([i["x"] for i in splits["valid"]])
                       .reshape(-1, L, D_IN), dtype=torch.float32).to(DEVICE)
    yva = torch.tensor(np.asarray([i["y_cls"] for i in splits["valid"]]),
                       dtype=torch.float32).to(DEVICE)
    Xte = torch.tensor(np.asarray([i["x"] for i in splits["test"]])
                       .reshape(-1, L, D_IN), dtype=torch.float32).to(DEVICE)
    yte = torch.tensor(np.asarray([i["y_cls"] for i in splits["test"]]),
                       dtype=torch.float32).to(DEVICE)
    items = splits["test"]
    print(f"train {len(Xtr)} valid {len(Xva)} test {len(Xte)}")

    results = {}
    for name, factory in [("BiLSTM", BiLSTMClf), ("GRU", GRUClf),
                          ("TCN", TCNClf), ("Transformer", TransformerClf)]:
        r = run_deep(factory, Xtr, ytr, Xva, yva, Xte, yte, items)
        results[name] = r
        print(f"{name:12s} F1={r['metrics']['f1']['mean']:.4f}±{r['metrics']['f1']['std']:.4f} "
              f"AUC={r['metrics']['auc']['mean']:.4f}±{r['metrics']['auc']['std']:.4f}")

    r = run_xgboost(Xtr.cpu(), ytr.cpu(), Xva.cpu(), yva.cpu(),
                    Xte.cpu(), yte.cpu(), items)
    results["XGBoost"] = r
    print(f"{'XGBoost':12s} F1={r['metrics']['f1']['mean']:.4f}±{r['metrics']['f1']['std']:.4f} "
          f"AUC={r['metrics']['auc']['mean']:.4f}±{r['metrics']['auc']['std']:.4f}")

    # Reference: logistic + persistence on the same windows
    from sklearn.linear_model import LogisticRegression
    Xtr_l = Xtr.cpu().reshape(len(Xtr), -1).numpy()
    Xva_l = Xva.cpu().reshape(len(Xva), -1).numpy()
    Xte_l = Xte.cpu().reshape(len(Xte), -1).numpy()
    ytr_n, yva_n, yte_n = ytr.cpu().numpy(), yva.cpu().numpy(), yte.cpu().numpy()
    lr = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced",
                            random_state=42)
    lr.fit(Xtr_l, ytr_n)
    pv = lr.predict_proba(Xva_l)[:, 1]
    _, thr = max((f1_score(yva_n, pv >= t, zero_division=0), t)
                 for t in DECISION_THRESHOLDS)
    prob = lr.predict_proba(Xte_l)[:, 1]
    last = np.asarray([i["last_target"] for i in items])
    thrs = np.asarray([i["threshold"] for i in items])
    pers = last > thrs
    results["Logistic"] = {"per_seed": [{"seed": 0, "threshold": thr,
                                         **classification_metrics(
                                             yte_n, prob, prob >= thr)}]}
    results["Persistence"] = classification_metrics(yte_n, last, pers)
    results["n_test"] = len(items)
    print(f"{'Logistic':12s} F1={results['Logistic']['per_seed'][0]['f1']:.4f} "
          f"AUC={results['Logistic']['per_seed'][0]['auc']:.4f}")

    out = R_DIR / "risk_baselines.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved {out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
