#!/usr/bin/env python3
"""
Extended experiment: try multiple architectures to beat BiLSTM.
Also generates paper-quality figures.
"""
import json, os, glob, pickle, warnings, time, sys
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
warnings.filterwarnings("ignore")

# Config
L, H = 12, 6
N_SYNTH, SYNTH_BINS = 60, 200
HIDDEN, DROPOUT = 64, 0.2
EPOCHS, LR, PATIENCE = 300, 1e-3, 40
TRAIN_SPLIT = 0.75
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULT_DIR = "experiment_results"
os.makedirs(RESULT_DIR, exist_ok=True)

BIN_HOURS = 12; TOP_N = 15; TOP_K = 0.15; WA,WE,WS = 0.5,0.3,0.2

def notify(msg):
    os.system(f'/home/violina/projects/notice/notice "{msg}" 2>/dev/null &')

# ═══ Data Generation ═══
def gen_traj(n_bins=SYNTH_BINS, seed=None):
    rng = np.random.RandomState(seed); n = n_bins
    t = np.arange(n)
    base = 0.3 + rng.uniform(0, 0.15)
    n_ev = rng.randint(1, 4)
    trend = np.zeros(n)
    for _ in range(n_ev):
        es, ed = rng.randint(10, n - 30), rng.randint(5, 15)
        ep = rng.uniform(0.15, 0.35)
        te = np.arange(ed * 2)
        sig = 1.0 / (1.0 + np.exp(-(te - ed / 2) / (ed / 8)))
        sig = (sig - sig[0]) / sig.max() * ep
        sig = sig * np.exp(-(te - ed) / (ed * 2)) if ed > 0 else sig
        idx = min(es + len(sig), n)
        trend[es:idx] += sig[:idx - es]
    season = 0.02 * np.sin(2 * np.pi * t / 14.0)
    white = rng.normal(0, 0.02, n)
    pink = np.zeros(n); pink[0] = white[0]
    for i in range(1, n): pink[i] = 0.6 * pink[i - 1] + 0.4 * white[i]
    y = np.clip(base + trend + season + pink, 0, 1)
    esc = (trend > 0.1).astype(int)
    return y, esc

def make_windows(n_topics=N_SYNTH, n_bins=SYNTH_BINS):
    all_X, all_y, all_esc = [], [], []
    for i in range(n_topics):
        vals, esc = gen_traj(n_bins, seed=i)
        for j in range(len(vals) - L - H + 1):
            all_X.append(vals[j:j+L]); all_y.append(vals[j+L:j+L+H])
            all_esc.append(esc[j+L:j+L+H].max())
    X = torch.tensor(np.array(all_X), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(all_y), dtype=torch.float32)
    return X, y, np.array(all_esc)

# ═══ Models ═══
class BiLSTM(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__(); self.h = h
        self.lstm = nn.LSTM(1, h, 2, batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.proj = nn.Linear(h*2, H)
    def forward(self, x):
        o, _ = self.lstm(x); return self.proj(torch.cat([o[:,-1,:self.h], o[:,0,self.h:]], dim=-1))

class DeepBiLSTM(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__(); self.h = h
        self.lstm = nn.LSTM(1, h, 3, batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.proj = nn.Linear(h*2, H)
    def forward(self, x):
        o, _ = self.lstm(x); return self.proj(torch.cat([o[:,-1,:self.h], o[:,0,self.h:]], dim=-1))

class AttnLSTM(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__(); self.h = h
        self.lstm = nn.LSTM(1, h, 2, batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.attn = nn.MultiheadAttention(h*2, 4, batch_first=True)
        self.proj = nn.Linear(h*2, H)
    def forward(self, x):
        o, _ = self.lstm(x); a, _ = self.attn(o, o, o)
        return self.proj(torch.cat([a[:,-1,:self.h], a[:,0,self.h:]], dim=-1))

class CNNLSTM(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__(); self.h = h
        self.conv = nn.Sequential(
            nn.Conv1d(1, h//2, 3, padding=1), nn.ReLU(),
            nn.Conv1d(h//2, h//2, 5, padding=2), nn.ReLU(),
        )
        self.lstm = nn.LSTM(h//2, h, 2, batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.proj = nn.Linear(h*2, H)
    def forward(self, x):
        c = self.conv(x.transpose(1,2)).transpose(1,2)
        o, _ = self.lstm(c); return self.proj(torch.cat([o[:,-1,:self.h], o[:,0,self.h:]], dim=-1))

class TCN(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__()
        self.tcn = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(1 if i==0 else h, h, 3, padding=2**(i+1), dilation=2**(i+1)),
                nn.ReLU(), nn.Dropout(DROPOUT),
            ) for i in range(4)
        ])
        self.proj = nn.Linear(h, H)
    def forward(self, x):
        o = self.tcn(x.transpose(1,2)).mean(-1); return self.proj(o)

class SmallTransformer(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__()
        self.input_proj = nn.Linear(1, h)
        encoder_layer = nn.TransformerEncoderLayer(h, 4, h*2, dropout=DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, 3)
        self.proj = nn.Linear(h, H)
    def forward(self, x):
        x = self.input_proj(x)
        # Positional encoding
        pos = torch.arange(x.size(1), device=x.device).float().unsqueeze(0).unsqueeze(-1)
        x = x + pos * 0.02
        o = self.transformer(x).mean(1)
        return self.proj(o)

# ═══ Training ═══
def train_model(model_factory, X, y, label, device=DEVICE):
    n = len(X); n_tr = int(n*TRAIN_SPLIT)
    perm = np.random.RandomState(42).permutation(n)
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    Xtr, ytr = X[tr_idx].to(device), y[tr_idx].to(device)
    Xte, yte = X[te_idx].to(device), y[te_idx].to(device)
    yte_np = yte.cpu().numpy()
    train_ds = TensorDataset(Xtr, ytr)
    test_ds = TensorDataset(Xte, yte)
    tl = DataLoader(train_ds, 64, True)
    vl = DataLoader(test_ds, 128)

    model = model_factory().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    huber = nn.HuberLoss(delta=0.5)
    best_vl, best_st, patience_c = float("inf"), None, 0
    tlosses, vlosses = [], []

    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tl: opt.zero_grad(); huber(model(xb), yb).backward(); opt.step()
        model.eval()
        tr_l = sum(huber(model(xb), yb).item() for xb, yb in tl) / len(tl)
        v_l = sum(huber(model(xb), yb).item() for xb, yb in vl) / len(vl)
        tlosses.append(tr_l); vlosses.append(v_l)
        if v_l < best_vl: best_vl = v_l; best_st = {k:v.cpu().clone() for k,v in model.state_dict().items()}; patience_c = 0
        else: patience_c += 1
        if patience_c >= PATIENCE: break

    model.load_state_dict(best_st); model.eval()
    with torch.no_grad(): yp = model(Xte.to(device)).cpu().numpy()

    mae = float(np.mean(np.abs(yp - yte_np)))
    rmse = float(np.sqrt(np.mean((yp - yte_np)**2)))
    ss_r = np.sum((yte_np-yp)**2); ss_t = np.sum((yte_np-yte_np.mean())**2); r2 = 1-ss_r/(ss_t+1e-8)
    pe = (yp.max(1)>=0.65).astype(int); te = (yte_np.max(1)>=0.65).astype(int)
    tp, fp, fn = (pe&te).sum(), (pe&(1-te)).sum(), ((1-pe)&te).sum()
    p, r = tp/(tp+fp) if (tp+fp) else 0, tp/(tp+fn) if (tp+fn) else 0
    escf1 = 2*p*r/(p+r) if(p+r) else 0

    print(f"  {label:20s} R²={r2:.4f} MAE={mae:.4f} Esc-F1={escf1:.3f} epochs={ep+1}")
    return {"label": label, "mae": mae, "rmse": rmse, "r2": r2, "esc_f1": escf1,
            "train_losses": tlosses, "val_losses": vlosses, "y_true": yte_np, "y_pred": yp,
            "best_epoch": ep+1-patience_c}

# ═══ Persistence & AR ═══
def train_baselines(X, y):
    n = len(X); n_tr = int(n*TRAIN_SPLIT)
    perm = np.random.RandomState(42).permutation(n)
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    Xtr, ytr = X[tr_idx], y[tr_idx]; Xte, yte = X[te_idx], y[te_idx]
    yte_np = yte.numpy()

    # Persistence
    yp = np.tile(Xte[:,-1,0].numpy().reshape(-1,1),(1,H))
    k = min(L, 6)
    Xar_tr = Xtr[:,-k:,0].numpy(); Xar_te = Xte[:,-k:,0].numpy()
    Xtr_flat = Xtr[:,:,0].numpy(); Xte_flat = Xte[:,:,0].numpy()

    # AR(k): linear autoregression
    from sklearn.linear_model import LinearRegression
    arp = []; lr_models = []
    for hh in range(H):
        lr = LinearRegression(); lr.fit(Xar_tr, ytr[:,hh].numpy()); arp.append(lr.predict(Xar_te))
        lr_models.append(lr)
    y_ar = np.stack(arp, 1)

    # SVR: sklearn SVR (used in Mu2023IPSO as classical ML baseline)
    from sklearn.svm import SVR
    svr_preds = []
    for hh in range(H):
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.01)
        svr.fit(Xtr_flat, ytr[:,hh].numpy())
        svr_preds.append(svr.predict(Xte_flat))
    y_svr = np.stack(svr_preds, 1)

    # XGBoost: gradient boosting (strong non-neural baseline)
    try:
        from xgboost import XGBRegressor
        xgb_preds = []
        for hh in range(H):
            xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, verbosity=0)
            xgb.fit(Xtr_flat, ytr[:,hh].numpy())
            xgb_preds.append(xgb.predict(Xte_flat))
        y_xgb = np.stack(xgb_preds, 1)
        xgb_ok = True
    except ImportError:
        y_xgb = None; xgb_ok = False

    def m(y_pred):
        mae = float(np.mean(np.abs(y_pred-yte_np)))
        rmse = float(np.sqrt(np.mean((y_pred-yte_np)**2)))
        ss_r = np.sum((yte_np-y_pred)**2); ss_t = np.sum((yte_np-yte_np.mean())**2); r2 = 1-ss_r/(ss_t+1e-8)
        pe = (y_pred.max(1)>=0.65).astype(int); te = (yte_np.max(1)>=0.65).astype(int)
        tp,fp,fn=(pe&te).sum(),(pe&(1-te)).sum(),((1-pe)&te).sum()
        p,r = tp/(tp+fp) if(tp+fp)else 0, tp/(tp+fn) if(tp+fn)else 0
        return {"mae":mae,"rmse":rmse,"r2":r2,"esc_f1":2*p*r/(p+r) if(p+r)else 0}
    out = {"Persistence": {"label":"Persistence", **m(yp), "y_pred":yp},
           "AR(k)": {"label":"AR(k)", **m(y_ar), "y_pred":y_ar},
           "SVR [Mu2023]": {"label":"SVR [Mu2023]", **m(y_svr), "y_pred":y_svr}}
    if xgb_ok:
        out["XGBoost"] = {"label":"XGBoost", **m(y_xgb), "y_pred":y_xgb}
    return out

# ═══ Main ═══
if __name__ == "__main__":
    t0 = time.time()
    print(f"Device: {DEVICE} | L={L} H={H} | {N_SYNTH} topics × {SYNTH_BINS} bins\n")

    X, y, esc = make_windows()
    print(f"Dataset: {len(X)} windows, {esc.mean():.1%} escalation\n")

    results = {}

    # Baselines
    bl = train_baselines(X, y)
    for n, r in bl.items(): results[n] = r; print(f"  {n:12s} R²={r['r2']:.4f} MAE={r['mae']:.4f}")

    # Neural models
    models = [
        ("BiLSTM", BiLSTM),
        ("Deep BiLSTM (3L)", DeepBiLSTM),
        ("Attn-BiLSTM", AttnLSTM),
        ("CNN-BiLSTM", CNNLSTM),
        ("TCN", TCN),
        ("Transformer", SmallTransformer),
    ]

    for name, factory in models:
        results[name] = train_model(factory, X, y, name)

    # ── Print comparison ──
    print(f"\n{'═'*70}")
    print(f"{'Model':22s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s} {'Esc-F1':>8s} {'ΔR²':>8s}")
    print(f"{'─'*70}")

    # Sort by R²
    sorted_models = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    best_r2 = sorted_models[0][1]['r2']
    for name, r in sorted_models:
        dr = r['r2'] - best_r2
        flag = " ★" if dr == 0 else ""
        print(f"{name:22s} {r['mae']:8.4f} {r['rmse']:8.4f} {r['r2']:8.4f} {r['esc_f1']:8.4f} {dr:+8.4f}{flag}")

    # ── Save results ──
    with open(f"{RESULT_DIR}/extended_results.pkl", "wb") as f:
        pickle.dump({k:{kk:vv for kk,vv in v.items() if kk not in ('y_true','y_pred','train_losses','val_losses')}
                     for k,v in results.items()}, f)

    # Save raw predictions for plotting
    with open(f"{RESULT_DIR}/plot_data.pkl", "wb") as f:
        pickle.dump(results, f)

    elapsed = (time.time()-t0)/60
    print(f"\nDone in {elapsed:.1f} min. Best: {sorted_models[0][0]} R²={best_r2:.4f}")

    # Notify
    best_name = sorted_models[0][0]
    notify(f"Extended experiment done! Best: {best_name} R²={best_r2:.4f} ({elapsed:.1f}min)")
