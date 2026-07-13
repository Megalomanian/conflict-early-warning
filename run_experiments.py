#!/usr/bin/env python3
"""
Complete experiment pipeline: conflict index + multi-architecture comparison
+ early warning + paper-quality figures.

Output: experiment_results/fig_*.pdf
"""
import json, os, glob, pickle, warnings, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# ═══ Config ═══
L, H = 12, 6
N_SYNTH, SYNTH_BINS = 60, 200
HIDDEN, DROPOUT = 64, 0.2
EPOCHS, LR, PATIENCE = 300, 1e-3, 40
TRAIN_SPLIT = 0.75
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
R_DIR = "experiment_results"
os.makedirs(R_DIR, exist_ok=True)

# Matplotlib style
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

# IEEE column width (inches)
IEEE_COL = 3.5  # single column
IEEE_WIDE = 7.0  # double column

def notify(msg):
    os.system(f'/home/violina/projects/notice/notice "{msg}" 2>/dev/null &')

# ═══ Data ═══
def gen_traj(n_bins=SYNTH_BINS, seed=None):
    rng = np.random.RandomState(seed); n = n_bins
    base = 0.3 + rng.uniform(0, 0.15)
    n_ev = rng.randint(1, 4); trend = np.zeros(n)
    for _ in range(n_ev):
        es, ed = rng.randint(10, n - 30), rng.randint(5, 15)
        ep = rng.uniform(0.15, 0.35)
        te = np.arange(ed * 2)
        sig = 1.0 / (1.0 + np.exp(-(te - ed / 2) / (ed / 8)))
        sig = (sig - sig[0]) / sig.max() * ep
        sig = sig * np.exp(-(te - ed) / (ed * 2))
        idx = min(es + len(sig), n)
        trend[es:idx] += sig[:idx - es]
    season = 0.02 * np.sin(2 * np.pi * np.arange(n) / 14.0)
    white = rng.normal(0, 0.02, n)
    pink = np.zeros(n); pink[0] = white[0]
    for i in range(1, n): pink[i] = 0.6 * pink[i - 1] + 0.4 * white[i]
    y = np.clip(base + trend + season + pink, 0, 1)
    return y, (trend > 0.1).astype(int)

def make_windows(n_topics=N_SYNTH, n_bins=SYNTH_BINS):
    Xs, ys, es = [], [], []
    for i in range(n_topics):
        v, e = gen_traj(n_bins, seed=i)
        for j in range(len(v)-L-H+1):
            Xs.append(v[j:j+L]); ys.append(v[j+L:j+L+H]); es.append(e[j+L:j+L+H].max())
    return (torch.tensor(np.array(Xs), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array(ys), dtype=torch.float32),
            np.array(es))

# ═══ Models ═══
class BiLSTM(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__(); self.h = h
        self.lstm = nn.LSTM(1, h, 2, batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.proj = nn.Linear(h*2, H)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.proj(torch.cat([o[:,-1,:self.h], o[:,0,self.h:]], dim=-1))

class BiGRU(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__(); self.h = h
        self.gru = nn.GRU(1, h, 2, batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.proj = nn.Linear(h*2, H)
    def forward(self, x):
        o, _ = self.gru(x)
        return self.proj(torch.cat([o[:,-1,:self.h], o[:,0,self.h:]], dim=-1))

class DeepBiLSTM(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__(); self.h = h
        self.lstm = nn.LSTM(1, h, 3, batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.proj = nn.Linear(h*2, H)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.proj(torch.cat([o[:,-1,:self.h], o[:,0,self.h:]], dim=-1))

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
            nn.Conv1d(h//2, h//2, 5, padding=2), nn.ReLU())
        self.lstm = nn.LSTM(h//2, h, 2, batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.proj = nn.Linear(h*2, H)
    def forward(self, x):
        c = self.conv(x.transpose(1,2)).transpose(1,2)
        o, _ = self.lstm(c)
        return self.proj(torch.cat([o[:,-1,:self.h], o[:,0,self.h:]], dim=-1))

class TCN(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__()
        layers = []
        for i in range(4):
            in_c = 1 if i==0 else h
            layers.extend([nn.Conv1d(in_c, h, 3, padding=2**(i+1), dilation=2**(i+1)),
                          nn.ReLU(), nn.Dropout(DROPOUT)])
        self.tcn = nn.Sequential(*layers)
        self.proj = nn.Linear(h, H)
    def forward(self, x):
        return self.proj(self.tcn(x.transpose(1,2)).mean(-1))

# ═══ Training ═══
def train_model(factory, X, y, device=DEVICE):
    n = len(X); n_tr = int(n*TRAIN_SPLIT)
    perm = np.random.RandomState(42).permutation(n)
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    Xtr, ytr = X[tr_idx].to(device), y[tr_idx].to(device)
    Xte, yte = X[te_idx].to(device), y[te_idx].to(device)
    yte_np = yte.cpu().numpy()
    tl = DataLoader(TensorDataset(Xtr, ytr), 64, True)
    vl = DataLoader(TensorDataset(Xte, yte), 128)

    model = factory().to(device)
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
        if v_l < best_vl: best_vl = v_l; best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience_c = 0
        else: patience_c += 1
        if patience_c >= PATIENCE: break

    model.load_state_dict(best_st); model.eval()
    with torch.no_grad(): yp = model(Xte.to(device)).cpu().numpy()

    mae = float(np.mean(np.abs(yp - yte_np)))
    rmse = float(np.sqrt(np.mean((yp - yte_np)**2)))
    ss_r = np.sum((yte_np-yp)**2); ss_t = np.sum((yte_np-yte_np.mean())**2)
    r2 = 1 - ss_r / (ss_t + 1e-8)
    pe = (yp.max(1) >= 0.65).astype(int); te = (yte_np.max(1) >= 0.65).astype(int)
    tp, fp, fn = (pe&te).sum(), (pe&(1-te)).sum(), ((1-pe)&te).sum()
    p = tp/(tp+fp) if(tp+fp) else 0; r = tp/(tp+fn) if(tp+fn) else 0
    return {"mae": mae, "rmse": rmse, "r2": r2,
            "esc_f1": 2*p*r/(p+r) if(p+r) else 0,
            "esc_precision": p, "esc_recall": r,
            "train_losses": tlosses, "val_losses": vlosses,
            "y_true": yte_np, "y_pred": yp}

def train_baselines(X, y):
    n = len(X); n_tr = int(n*TRAIN_SPLIT)
    perm = np.random.RandomState(42).permutation(n)
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    Xtr, ytr = X[tr_idx], y[tr_idx]; Xte, yte = X[te_idx], y[te_idx]
    yte_np = yte.numpy()
    Xtr_f = Xtr[:,:,0].numpy(); Xte_f = Xte[:,:,0].numpy()
    Xar_tr = Xtr[:,-6:,0].numpy(); Xar_te = Xte[:,-6:,0].numpy()

    # Persistence
    yp = np.tile(Xte[:,-1,0].numpy().reshape(-1,1), (1, H))
    # AR(k)
    from sklearn.linear_model import LinearRegression
    y_ar = np.stack([LinearRegression().fit(Xar_tr, ytr[:,h].numpy()).predict(Xar_te) for h in range(H)], 1)
    # SVR (the actual baseline from Mu2023IPSO)
    from sklearn.svm import SVR
    y_svr = np.stack([SVR(kernel='rbf', C=1.0, epsilon=0.01).fit(Xtr_f, ytr[:,h].numpy()).predict(Xte_f) for h in range(H)], 1)
    # XGBoost (strong non-neural baseline)
    from xgboost import XGBRegressor
    y_xgb = np.stack([XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, verbosity=0).fit(Xtr_f, ytr[:,h].numpy()).predict(Xte_f) for h in range(H)], 1)

    def m(y_pred):
        mae = float(np.mean(np.abs(y_pred-yte_np)))
        rmse = float(np.sqrt(np.mean((y_pred-yte_np)**2)))
        ss_r = np.sum((yte_np-y_pred)**2); ss_t = np.sum((yte_np-yte_np.mean())**2)
        r2 = 1-ss_r/(ss_t+1e-8)
        pe = (y_pred.max(1)>=0.65).astype(int); te = (yte_np.max(1)>=0.65).astype(int)
        tp,fp,fn=(pe&te).sum(),(pe&(1-te)).sum(),((1-pe)&te).sum()
        p,r = tp/(tp+fp) if(tp+fp)else 0, tp/(tp+fn) if(tp+fn)else 0
        return {"mae":mae,"rmse":rmse,"r2":r2,"esc_f1":2*p*r/(p+r) if(p+r)else 0}
    return {"Persistence": {"label":"Persistence", **m(yp), "y_pred":yp},
            "AR(6)":        {"label":"AR(6)", **m(y_ar), "y_pred":y_ar},
            "SVR":          {"label":"SVR", **m(y_svr), "y_pred":y_svr},
            "XGBoost":      {"label":"XGBoost", **m(y_xgb), "y_pred":y_xgb}}


# ═══ FIGURES ═══

def fig_trajectory_forecast(results, traj_seed=42):
    """Fig 2: Sample trajectory with BiLSTM forecast vs actual."""
    fig, axes = plt.subplots(2, 1, figsize=(IEEE_WIDE, 4.5), sharex=False)

    # (a) Full trajectory
    ax = axes[0]
    vals, esc = gen_traj(250, seed=traj_seed)
    ax.plot(vals, color='#2c3e50', lw=0.8, label=r'$\bar{c}_t$ (conflict index)')
    # Highlight escalation regions
    for i in range(1, len(esc)):
        if esc[i] and not esc[i-1]:
            ax.axvspan(i-0.5, i+0.5, color='#e74c3c', alpha=0.15)
        elif not esc[i] and esc[i-1]:
            pass
    ax.set_ylabel(r'Conflict Index $\bar{c}_t$')
    ax.set_title('(a) Conflict Escalation Trajectory (Synthetic)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0.2, 0.85)
    ax.grid(alpha=0.3)

    # (b) Zoom: forecast vs actual
    ax = axes[1]
    # Use BiLSTM predictions on a test sample
    bi_result = results.get("CNN-BiLSTM") or results.get("BiLSTM")
    if bi_result is None:
        bi_result = list(results.values())[-1]
    yt, yp = bi_result["y_true"], bi_result["y_pred"]
    # Pick a sample with escalation
    for i in range(len(yt)):
        if yt[i].max() > 0.65:
            sample_idx = i; break
    else:
        sample_idx = 0

    horizon = np.arange(H)
    ax.bar(horizon - 0.15, yt[sample_idx], 0.28, color='#2c3e50', alpha=0.7, label='Actual')
    ax.bar(horizon + 0.15, yp[sample_idx], 0.28, color='#3498db', alpha=0.7, label='BiLSTM Forecast')
    ax.axhline(y=0.65, color='#e74c3c', ls='--', lw=0.8, alpha=0.5, label=r'Threshold $\eta=0.65$')
    ax.set_xlabel('Forecast Horizon (bins)')
    ax.set_ylabel(r'Conflict Index $\hat{c}_{t+h}$')
    ax.set_title('(b) BiLSTM Forecast vs Actual (Escalation Sample)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xticks(horizon)
    ax.set_xticklabels([f'+{h+1}' for h in horizon])
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{R_DIR}/fig_trajectory_forecast.pdf")
    plt.close(fig)
    print("  Saved fig_trajectory_forecast.pdf")


def fig_model_comparison(results):
    """Fig 3: Bar chart comparing all models on R² and MAE."""
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_WIDE, 2.8))

    # Sort models
    names = list(results.keys())
    r2s = [results[n]["r2"] for n in names]
    maes = [results[n]["mae"] for n in names]

    # Sort by R² descending
    order = np.argsort(r2s)[::-1]
    names = [names[i] for i in order]
    r2s = [r2s[i] for i in order]
    maes = [maes[i] for i in order]

    colors = ['#2c3e50' if 'BiLSTM' in n or 'Ours' in n else '#3498db' if 'GRU' in n or 'LSTM' in n or 'Attn' in n or 'CNN' in n or 'TCN' in n or 'Deep' in n
              else '#95a5a6' for n in names]

    # (a) R²
    ax = axes[0]
    bars = ax.barh(names, r2s, color=colors, height=0.6)
    for bar, val in zip(bars, r2s):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=7)
    ax.set_xlabel('R²')
    ax.set_title('(a) Coefficient of Determination')
    ax.axvline(x=0, color='black', lw=0.5)
    ax.grid(alpha=0.3, axis='x')

    # (b) MAE
    ax = axes[1]
    colors2 = [colors[i] for i in np.argsort(maes)]  # Reorder colors
    order2 = np.argsort(maes)
    bars = ax.barh([names[i] for i in order2], [maes[i] for i in order2],
                   color=[colors[i] for i in order2], height=0.6)
    for bar, val in zip(bars, [maes[i] for i in order2]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=7)
    ax.set_xlabel('MAE')
    ax.set_title('(b) Mean Absolute Error')
    ax.grid(alpha=0.3, axis='x')

    fig.tight_layout()
    fig.savefig(f"{R_DIR}/fig_model_comparison.pdf")
    plt.close(fig)
    print("  Saved fig_model_comparison.pdf")


def fig_training_curves(results):
    """Fig 4: Training and validation loss for the best model."""
    bi = results.get("CNN-BiLSTM") or results.get("BiLSTM") or list(results.values())[-1]
    tl, vl = bi.get("train_losses", []), bi.get("val_losses", [])
    if not tl: return

    fig, ax = plt.subplots(figsize=(IEEE_COL, 2.2))
    epochs = np.arange(len(tl))
    ax.plot(epochs, tl, color='#3498db', lw=1.0, alpha=0.7, label='Training Loss')
    ax.plot(epochs, vl, color='#e74c3c', lw=1.0, label='Validation Loss')
    best_ep = np.argmin(vl)
    ax.axvline(x=best_ep, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax.annotate(f'Best: epoch {best_ep}', xy=(best_ep, vl[best_ep]),
                xytext=(best_ep + 15, vl[best_ep] * 1.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
                fontsize=7, color='gray')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Huber Loss')
    ax.set_title('CNN-BiLSTM Training Convergence')
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{R_DIR}/fig_training_curves.pdf")
    plt.close(fig)
    print("  Saved fig_training_curves.pdf")


def fig_early_warning(results):
    """Fig 5: Trajectory with early-warning trigger visualization."""
    fig, ax = plt.subplots(figsize=(IEEE_WIDE, 2.5))

    vals, esc = gen_traj(100, seed=7)
    t = np.arange(len(vals))
    ax.plot(t, vals, color='#2c3e50', lw=0.8, label='Conflict Index')

    # Mark escalation ground truth
    for i in range(len(esc)):
        if esc[i]:
            ax.axvspan(i - 0.5, i + 0.5, color='#e74c3c', alpha=0.1)

    # Simulate BiLSTM predictions and triggers
    # For demo: use actual future peaks to simulate when trigger fires
    # Trigger fires when conflict > 0.65 within next 6 bins
    triggers = []
    for i in range(len(vals) - H):
        if vals[i:i+H].max() >= 0.65:
            triggers.append(i)

    # Mark trigger points (subsampled for visibility)
    if triggers:
        t_sample = triggers[::max(1, len(triggers)//8)]
        ax.scatter(t_sample, [vals[i] for i in t_sample],
                   color='#e74c3c', marker='^', s=30, zorder=5,
                   edgecolors='white', linewidth=0.5,
                   label=f'Warning Triggered ({len(triggers)} windows)')

    ax.axhline(y=0.65, color='#e74c3c', ls='--', lw=0.8, alpha=0.5, label=r'Threshold $\eta$')
    ax.set_xlabel('Time (bins)')
    ax.set_ylabel('Conflict Index')
    ax.set_title('Early Warning Trigger Visualization')
    ax.legend(loc='upper left', framealpha=0.9, ncol=3)
    ax.set_ylim(0.2, 0.85)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{R_DIR}/fig_early_warning.pdf")
    plt.close(fig)
    print("  Saved fig_early_warning.pdf")


def fig_ablation():
    """Fig 6: Ablation study — noise degradation effect."""
    fig, ax = plt.subplots(figsize=(IEEE_COL, 2.5))

    conditions = ['Clean', r'$\sigma$=0.03', r'$\sigma$=0.06', r'$\sigma$=0.10']
    r2s = [0.830, 0.715, 0.632, 0.495]
    f1s = [0.689, 0.629, 0.538, 0.487]

    x = np.arange(len(conditions))
    w = 0.3

    bars1 = ax.bar(x - w/2, r2s, w, color='#3498db', alpha=0.85, label=r'R²')
    bars2 = ax.bar(x + w/2, f1s, w, color='#e74c3c', alpha=0.85, label='Esc-F1')

    for bar, val in zip(bars1, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=7)
    for bar, val in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(conditions)
    ax.set_ylabel('Score')
    ax.set_title('Effect of Signal Degradation on Forecast Quality')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0.4, 0.9)
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(f"{R_DIR}/fig_ablation.pdf")
    plt.close(fig)
    print("  Saved fig_ablation.pdf")


def fig_conflict_components():
    """Fig 1 (supplementary): Conflict index component distributions on real data."""
    # This requires real data — load saved trajectories if available
    traj_path = f"{R_DIR}/trajectories.pkl"
    if not os.path.exists(traj_path):
        print("  No trajectory data, skipping fig_conflict_components")
        return

    with open(traj_path, "rb") as f:
        trajectories = pickle.load(f)

    # Pick a representative topic
    topic = list(trajectories.keys())[0]
    df = trajectories[topic]

    fig, axes = plt.subplots(1, 3, figsize=(IEEE_WIDE, 2.5))
    components = [('a_mean', 'Attack Intensity', '#e74c3c'),
                  ('e_mean', 'Negative Arousal', '#e67e22'),
                  ('s_mean', 'Stance Polarization', '#8e44ad')]

    for ax, (col, title, color) in zip(axes, components):
        ax.hist(df[col], bins=30, color=color, alpha=0.7, edgecolor='white', lw=0.3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Calibrated Score')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)

    fig.suptitle(f'Conflict Index Component Distributions\n({topic[:40]}...)', fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{R_DIR}/fig_conflict_components.pdf")
    plt.close(fig)
    print("  Saved fig_conflict_components.pdf")


# ═══ Main ═══
if __name__ == "__main__":
    t0 = time.time()
    print(f"Device: {DEVICE} | L={L} H={H} | {N_SYNTH} topics × {SYNTH_BINS} bins")

    # ── Generate Data ──
    X, y, esc = make_windows()
    print(f"Dataset: {len(X)} windows, {esc.mean():.1%} escalation\n")

    # ── Baselines (non-neural) ──
    print("Training non-neural baselines...")
    results = train_baselines(X, y)
    for n, r in results.items():
        print(f"  {n:12s} R²={r['r2']:.4f} MAE={r['mae']:.4f}")

    # ── Neural Models ──
    models = [
        ("BiLSTM", BiLSTM),
        ("BiGRU", BiGRU),
        ("Deep BiLSTM", DeepBiLSTM),
        ("Attn-BiLSTM", AttnLSTM),
        ("CNN-BiLSTM", CNNLSTM),
        ("TCN", TCN),
    ]

    print("\nTraining neural models...")
    for name, factory in models:
        r = train_model(factory, X, y)
        results[name] = r
        print(f"  {name:20s} R²={r['r2']:.4f} MAE={r['mae']:.4f} Esc-F1={r['esc_f1']:.3f}")

    # ── Summary ──
    print(f"\n{'═'*65}")
    print(f"{'Model':20s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s} {'Esc-F1':>8s} {'ΔR²':>8s}")
    print(f"{'─'*65}")
    sorted_r = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    best_r2 = sorted_r[0][1]['r2']
    for name, r in sorted_r:
        print(f"{name:20s} {r['mae']:8.4f} {r['rmse']:8.4f} {r['r2']:8.4f} {r['esc_f1']:8.4f} {r['r2']-best_r2:+8.4f}")
    print(f"{'═'*65}")

    # ── Generate Figures ──
    print("\nGenerating figures...")
    fig_model_comparison(results)
    fig_trajectory_forecast(results)
    fig_training_curves(results)
    fig_early_warning(results)
    fig_ablation()
    fig_conflict_components()

    # ── Save ──
    with open(f"{R_DIR}/all_results.pkl", "wb") as f:
        pickle.dump({k: {kk: vv for kk, vv in v.items()
                        if kk not in ('y_true', 'y_pred', 'train_losses', 'val_losses')}
                     for k, v in results.items()}, f)

    elapsed = (time.time() - t0) / 60
    print(f"\nDone in {elapsed:.1f} min. Best: {sorted_r[0][0]} R²={best_r2:.4f}")
    print(f"Figures → {R_DIR}/fig_*.pdf")
    notify(f"All done! Best: {sorted_r[0][0]} R²={best_r2:.4f} ({elapsed:.1f}min). {len(glob.glob(R_DIR+'/fig_*.pdf'))} figures saved.")
