#!/usr/bin/env python3
"""Trigger rule evaluation under strict temporal split (V2 protocol).
Evaluates 4 trigger rules on CNN-BiLSTM forecasts, outputting TABLE IV numbers."""
import numpy as np, warnings, time, pickle, os
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

R_DIR = "experiment_results_v2"
os.makedirs(R_DIR, exist_ok=True)

L, H = 12, 6; HIDDEN, DROPOUT = 64, 0.2
EPOCHS, LR, PATIENCE = 150, 1e-3, 25
N_SYNTH, SYNTH_BINS = 60, 200
TEMPORAL_SPLIT = 0.75
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ETA, GAMMA = 0.65, 0.10  # trigger thresholds

# ═══ CNN-BiLSTM ═══
class CNNBiLSTM(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__(); self.h = h
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU())
        self.lstm = nn.LSTM(32, h, 2, batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.proj = nn.Linear(h*2, H)
    def forward(self, x):
        c = self.conv(x.transpose(1,2)).transpose(1,2)
        o, _ = self.lstm(c)
        return self.proj(torch.cat([o[:,-1,:self.h], o[:,0,self.h:]], dim=-1))

# ═══ Synthetic Data ═══
def gen_traj(n_bins=SYNTH_BINS, seed=None):
    rng = np.random.RandomState(seed); n = n_bins
    base = 0.3 + rng.uniform(0, 0.15); trend = np.zeros(n)
    for _ in range(rng.randint(1,4)):
        es, ed = rng.randint(10, n-30), rng.randint(5,15)
        ep = rng.uniform(0.15, 0.35)
        te = np.arange(ed*2)
        sig = 1.0/(1.0+np.exp(-(te-ed/2)/(ed/8)))
        sig = (sig-sig[0])/sig.max()*ep*sig*np.exp(-(te-ed)/(ed*2))
        idx = min(es+len(sig), n); trend[es:idx] += sig[:idx-es]
    season = 0.02*np.sin(2*np.pi*np.arange(n)/14.0)
    white = rng.normal(0,0.02,n); pink = np.zeros(n); pink[0] = white[0]
    for i in range(1,n): pink[i] = 0.6*pink[i-1]+0.4*white[i]
    traj = np.clip(base+trend+season+pink,0,1)
    return traj, (trend>0.1).astype(int)

def make_windows(trajs, L=L, H=H):
    Xs, ys, es = [], [], []
    for traj, esc in trajs:
        for i in range(len(traj)-L-H+1):
            Xs.append(traj[i:i+L]); ys.append(traj[i+L:i+L+H])
            es.append(esc[i+L:i+L+H].max())
    X = torch.tensor(np.array(Xs), dtype=torch.float32).unsqueeze(-1)
    return X, torch.tensor(np.array(ys), dtype=torch.float32), np.array(es)

def temporal_split(X, y, esc, frac=TEMPORAL_SPLIT):
    n = int(len(X)*frac)
    return X[:n], y[:n], esc[:n], X[n:], y[n:], esc[n:]

# ═══ Train ═══
print(f"Device: {DEVICE} | L={L} H={H}")
np.random.seed(42); torch.manual_seed(42)
trajs = [gen_traj(SYNTH_BINS, i*100+42) for i in range(N_SYNTH)]
X, y, esc = make_windows(trajs)
Xtr, ytr, esc_tr, Xte, yte, esc_te = temporal_split(X, y, esc)
print(f"Train: {len(Xtr)} windows, Test: {len(Xte)} windows")

Xtr_d, ytr_d = Xtr.to(DEVICE), ytr.to(DEVICE)
Xte_d, yte_d = Xte.to(DEVICE), yte.to(DEVICE)
tl = DataLoader(TensorDataset(Xtr_d, ytr_d), 64, True)
vl = DataLoader(TensorDataset(Xte_d, yte_d), 128)

model = CNNBiLSTM().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
huber = nn.HuberLoss(delta=0.5)
best_vl, best_st, patience_c = float("inf"), None, 0
for ep in range(EPOCHS):
    model.train()
    for xb, yb in tl: opt.zero_grad(); huber(model(xb), yb).backward(); opt.step()
    model.eval()
    v_l = sum(huber(model(xb), yb).item() for xb, yb in vl)/len(vl)
    if v_l < best_vl: best_vl=v_l; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; patience_c=0
    else: patience_c+=1
    if patience_c>=PATIENCE: break
model.load_state_dict(best_st); model.eval()
with torch.no_grad(): y_pred = model(Xte_d).cpu().numpy()
print(f"Trained: best epoch {max(0,ep+1-patience_c)}")

# ═══ Trigger Evaluation ═══
yte_np = yte.numpy()
Xte_np = Xte.squeeze(-1).numpy()
n_test = len(Xte)

def eval_triggers(rule_name, trigger_fn):
    """Evaluate a trigger rule: returns precision, recall, f1, alert_rate."""
    alerts = np.zeros(n_test, dtype=bool)
    for i in range(n_test):
        hist = Xte_np[i, -1]  # last observed value
        pred = y_pred[i]
        alerts[i] = trigger_fn(hist, pred)
    tp = (alerts & esc_te).sum()
    fp = (alerts & ~esc_te).sum()
    fn = (~alerts & esc_te).sum()
    p = tp/(tp+fp) if (tp+fp) else 0
    r = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    ar = alerts.mean()
    return {"precision":p, "recall":r, "f1":f1, "alert_rate":ar, "tp":int(tp), "fp":int(fp), "fn":int(fn)}

# Rule 1: Threshold — peak in horizon >= eta
r1 = eval_triggers("Threshold (Rule 1)", lambda h, p: p.max() >= ETA)

# Rule 2: Growth — predicted increase >= gamma
r2 = eval_triggers("Growth (Rule 2)", lambda h, p: (p[-1] - h) >= GAMMA)

# Rule 3: Threshold + Growth
r3 = eval_triggers("Thr + Growth", lambda h, p: (p.max() >= ETA) or ((p[-1] - h) >= GAMMA))

# Rule 4: Composite — weighted score >= tau
TAU = 0.55
r4 = eval_triggers("Composite (Rule 5)", 
    lambda h, p: (0.5*p.max() + 0.3*(p[-1]-h) + 0.2*(p.max()>=ETA)) >= TAU)

# ═══ Print & Save ═══
results = [r1, r2, r3, r4]
print(f"\n{'═'*70}")
print(f"{'Trigger Rule':25s} {'Precision':>8s} {'Recall':>8s} {'F1':>8s} {'Alert Rate':>10s}")
print(f"{'─'*70}")
for r in results:
    print(f"{r.get('_name',''):25s} {r['precision']:8.4f} {r['recall']:8.4f} {r['f1']:8.4f} {r['alert_rate']:9.1%}")
print(f"{'═'*70}")

# Save with names
for r, name in zip(results, ["Threshold (Rule 1)", "Growth (Rule 2)", "Thr + Growth", "Composite (Rule 5)"]):
    r["_name"] = name

with open(f"{R_DIR}/trigger_eval.pkl", "wb") as f:
    pickle.dump({"results": results, "config": {"L":L,"H":H,"eta":ETA,"gamma":GAMMA,"tau":TAU}}, f)
print(f"\nSaved to {R_DIR}/trigger_eval.pkl")
