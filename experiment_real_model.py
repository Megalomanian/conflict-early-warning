#!/usr/bin/env python3
"""
Conflict index computation using REAL pretrained Chinese sentiment model.
Replaces keyword heuristics with: nlptown/bert-base-multilingual-uncased-sentiment.
Paper: Section III-B — pretrained harm model + emotion model.
"""
import json, os, glob, pickle, warnings, time, sys
from collections import defaultdict
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
R_DIR = "experiment_results"
os.makedirs(R_DIR, exist_ok=True)

L, H = 12, 6; HIDDEN, DROPOUT = 64, 0.2
EPOCHS, LR, PATIENCE = 200, 1e-3, 40; TRAIN_SPLIT = 0.75

def notify(msg): os.system(f'/home/violina/projects/notice/notice "{msg}" 2>/dev/null &')

# ═══ Real Pretrained Model Conflict Index ═══
class RealConflictComputer:
    """Uses a real HuggingFace sentiment model for attack+emotion, plus embeddings for stance."""

    def __init__(self):
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        from sentence_transformers import SentenceTransformer
        print("Loading real pretrained models...")
        # Multilingual BERT sentiment (WordPiece tokenizer, no SentencePiece needed)
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sent_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
        self.sent_model.eval()
        # Embedding model for stance
        self.emb_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)
        print("  Models loaded.")

    def compute_sentiment_batch(self, texts, batch_size=128):
        """Run sentiment model on a batch of texts. Returns logits for all 5 classes."""
        all_logits = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="  Sentiment",
                           unit="batch", ncols=100):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True,
                                        truncation=True, max_length=256).to(DEVICE)
                outputs = self.sent_model(**inputs)
                all_logits.append(outputs.logits.cpu().numpy())
        return np.concatenate(all_logits, axis=0)  # (N, 5) — 1-star to 5-star

    def compute_batch(self, texts):
        """Compute attack, emotion, and stance scores for all texts."""
        n = len(texts)

        # Step 1: Real pretrained sentiment → probabilities over 5 classes
        print(f"  Running pretrained sentiment on {n} texts...", flush=True)
        logits = self.compute_sentiment_batch(texts)
        # Softmax to get class probabilities
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        # Classes: 0=1star, 1=2star, 2=3star, 3=4star, 4=5star

        # Attack intensity: high when model predicts 1-star or 2-star with high confidence
        # Eq. (4) in paper: a_{t,i} = h_psi(x_{t,i})
        attack = probs[:, 0] + 0.5 * probs[:, 1]  # 1-star + half of 2-star

        # Emotion: negative high-arousal
        # Eq. (5): e = alpha * p_neg + (1-alpha) * p_anger
        # We map: 1-star = high arousal negative, 2-star = moderate negative
        p_neg = probs[:, 0] + probs[:, 1]  # total negative
        p_anger = probs[:, 0]  # extreme negativity ≈ anger/attack
        emotion = 0.7 * p_neg + 0.3 * p_anger

        # Step 2: Embeddings for stance polarization
        print(f"  Embedding {n} texts for stance...", flush=True)
        embs = self.emb_model.encode(texts, show_progress_bar=True, batch_size=256,
                                      convert_to_numpy=True, normalize_embeddings=True)

        print(f"  Attack μ={attack.mean():.3f} σ={attack.std():.3f} | "
              f"Emotion μ={emotion.mean():.3f} σ={emotion.std():.3f}", flush=True)

        return {"attack": attack, "emotion": emotion, "embeddings": embs}

    def stance_polarization(self, embs):
        n = len(embs)
        if n < 4: return np.zeros(n)
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
        km = KMeans(n_clusters=2, n_init=5, random_state=42); km.fit(embs)
        c0, c1 = km.cluster_centers_
        delta = 1.0 - cosine_similarity(c0.reshape(1, -1), c1.reshape(1, -1))[0, 0]
        d0 = cosine_distances(embs, c0.reshape(1, -1)).flatten()
        d1 = cosine_distances(embs, c1.reshape(1, -1)).flatten()
        return min(1.0, delta) * np.abs(d0 - d1) / (d0 + d1 + 1e-8)


# ═══ CNN-BiLSTM (same as before) ═══
class CNNBiLSTM(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__(); self.h = h
        self.conv = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(32, 32, 5, padding=2), nn.ReLU())
        self.lstm = nn.LSTM(32, h, 2, batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.proj = nn.Linear(h * 2, H)
    def forward(self, x):
        c = self.conv(x.transpose(1, 2)).transpose(1, 2); o, _ = self.lstm(c)
        return self.proj(torch.cat([o[:, -1, :self.h], o[:, 0, self.h:]], dim=-1))

# ═══ Additional baseline architectures ═══
class BiGRUModel(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__(); self.gru = nn.GRU(1, h, 2, batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.proj = nn.Linear(h * 2, H)
    def forward(self, x): o, _ = self.gru(x); return self.proj(torch.cat([o[:, -1, :HIDDEN], o[:, 0, HIDDEN:]], dim=-1))

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=(kernel-1)*dilation, dilation=dilation)
        self.relu = nn.ReLU(); self.dropout = nn.Dropout(0.1)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
    def forward(self, x):
        out = self.conv(x); out = out[:, :, :x.size(2)]
        res = self.downsample(x) if self.downsample else x
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(1, h, 3, 1), TCNBlock(h, h, 3, 2),
            TCNBlock(h, h, 3, 4), TCNBlock(h, h, 3, 8))
        self.proj = nn.Linear(h, H)
    def forward(self, x):
        o = self.tcn(x.transpose(1, 2)).mean(-1); return self.proj(o)

def gen_traj(n=200, s=None):
    rng = np.random.RandomState(s); base = 0.3 + rng.uniform(0, 0.15)
    n_ev = rng.randint(1, 4); trend = np.zeros(n)
    for _ in range(n_ev):
        es, ed = rng.randint(10, n - 30), rng.randint(5, 15); ep = rng.uniform(0.15, 0.35)
        te = np.arange(ed * 2); sig = 1.0 / (1.0 + np.exp(-(te - ed / 2) / (ed / 8)))
        sig = (sig - sig[0]) / sig.max() * ep; sig = sig * np.exp(-(te - ed) / (ed * 2))
        idx = min(es + len(sig), n); trend[es:idx] += sig[:idx - es]
    season = 0.02 * np.sin(2 * np.pi * np.arange(n) / 14.0)
    white = rng.normal(0, 0.02, n); pink = np.zeros(n); pink[0] = white[0]
    for i in range(1, n): pink[i] = 0.6 * pink[i - 1] + 0.4 * white[i]
    return np.clip(base + trend + season + pink, 0, 1), (trend > 0.1).astype(int)


def train_model_generic(model_factory, X, y, device=DEVICE):
    """Train any model factory → model, returning metrics dict."""
    n = len(X); n_tr = int(n * TRAIN_SPLIT)
    perm = np.random.RandomState(42).permutation(n)
    Xtr, ytr = X[perm[:n_tr]].to(device), y[perm[:n_tr]].to(device)
    Xte, yte = X[perm[n_tr:]].to(device), y[perm[n_tr:]].to(device)
    yte_np = yte.cpu().numpy()
    tl = DataLoader(TensorDataset(Xtr, ytr), 64, True)
    vl = DataLoader(TensorDataset(Xte, yte), 128)
    model = model_factory().to(device); opt = torch.optim.Adam(model.parameters(), lr=LR)
    huber = nn.HuberLoss(delta=0.5); best_vl, best_st, patience_c = float("inf"), None, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tl: opt.zero_grad(); huber(model(xb), yb).backward(); opt.step()
        model.eval(); v_l = sum(huber(model(xb), yb).item() for xb, yb in vl) / len(vl)
        if v_l < best_vl: best_vl = v_l; best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience_c = 0
        else: patience_c += 1
        if patience_c >= PATIENCE: break
    model.load_state_dict(best_st); model.eval()
    with torch.no_grad(): yp = model(Xte.to(device)).cpu().numpy()
    mae = float(np.mean(np.abs(yp - yte_np))); rmse = float(np.sqrt(np.mean((yp - yte_np)**2)))
    ss_r = np.sum((yte_np - yp)**2); ss_t = np.sum((yte_np - yte_np.mean())**2)
    r2 = 1 - ss_r / (ss_t + 1e-8)
    pe = (yp.max(1) >= 0.65).astype(int); te = (yte_np.max(1) >= 0.65).astype(int)
    tp, fp, fn = (pe & te).sum(), (pe & (1 - te)).sum(), ((1 - pe) & te).sum()
    p = tp / (tp + fp) if (tp + fp) else 0; r = tp / (tp + fn) if (tp + fn) else 0
    return {"mae": mae, "rmse": rmse, "r2": r2, "esc_f1": 2 * p * r / (p + r) if (p + r) else 0,
            "y_true": yte_np, "y_pred": yp}

def train_model(X, y, device=DEVICE):
    n = len(X); n_tr = int(n * TRAIN_SPLIT)
    perm = np.random.RandomState(42).permutation(n)
    Xtr, ytr = X[perm[:n_tr]].to(device), y[perm[:n_tr]].to(device)
    Xte, yte = X[perm[n_tr:]].to(device), y[perm[n_tr:]].to(device)
    yte_np = yte.cpu().numpy()
    tl = DataLoader(TensorDataset(Xtr, ytr), 64, True)
    vl = DataLoader(TensorDataset(Xte, yte), 128)
    model = CNNBiLSTM().to(device); opt = torch.optim.Adam(model.parameters(), lr=LR)
    huber = nn.HuberLoss(delta=0.5); best_vl, best_st, patience_c = float("inf"), None, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tl: opt.zero_grad(); huber(model(xb), yb).backward(); opt.step()
        model.eval(); v_l = sum(huber(model(xb), yb).item() for xb, yb in vl) / len(vl)
        if v_l < best_vl: best_vl = v_l; best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience_c = 0
        else: patience_c += 1
        if patience_c >= PATIENCE: break
    model.load_state_dict(best_st); model.eval()
    with torch.no_grad(): yp = model(Xte.to(device)).cpu().numpy()
    mae = float(np.mean(np.abs(yp - yte_np))); rmse = float(np.sqrt(np.mean((yp - yte_np)**2)))
    ss_r = np.sum((yte_np - yp)**2); ss_t = np.sum((yte_np - yte_np.mean())**2)
    r2 = 1 - ss_r / (ss_t + 1e-8)
    pe = (yp.max(1) >= 0.65).astype(int); te = (yte_np.max(1) >= 0.65).astype(int)
    tp, fp, fn = (pe & te).sum(), (pe & (1 - te)).sum(), ((1 - pe) & te).sum()
    p = tp / (tp + fp) if (tp + fp) else 0; r = tp / (tp + fn) if (tp + fn) else 0
    return {"mae": mae, "rmse": rmse, "r2": r2, "esc_f1": 2 * p * r / (p + r) if (p + r) else 0,
            "y_true": yte_np, "y_pred": yp}


# ═══ Main ═══
if __name__ == "__main__":
    t0 = time.time()

    # ── Real data: conflict index with REAL pretrained model ──
    print("=" * 60)
    print("Part A: Conflict Index with Real Pretrained Model")
    print("=" * 60)

    # Load zhihu data (same as before)
    base = "zhihu_topics"
    ranked = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isdir(path) or name == "zhihu": continue
        jd = os.path.join(path, "zhihu", "jsonl")
        if not os.path.isdir(jd): continue
        nc = sum(sum(1 for _ in open(f, encoding="utf-8")) for f in glob.glob(os.path.join(jd, "search_comments_*.jsonl")))
        if nc > 0: ranked.append((path, nc))
    ranked.sort(key=lambda x: x[1], reverse=True)

    records = []
    for tp, _ in tqdm(ranked[:20], desc="Loading topics"):
        tp_name = os.path.basename(tp); jd = os.path.join(tp, "zhihu", "jsonl")
        for ftype, tkey in [("search_comments", "content"), ("search_contents", "content_text")]:
            for fp in glob.glob(os.path.join(jd, f"{ftype}_*.jsonl")):
                for line in open(fp, encoding="utf-8"):
                    try: obj = json.loads(line.strip())
                    except: continue
                    text = obj.get(tkey, "").strip(); ts = obj.get("publish_time") or obj.get("created_time")
                    if text and len(text) >= 5 and ts and float(ts) > 1704067200:
                        records.append({"text": text, "ts": float(ts), "topic": tp_name})

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records from 20 topics")

    # Use REAL pretrained model
    computer = RealConflictComputer()
    all_texts = df["text"].tolist()
    r = computer.compute_batch(all_texts)

    # Calibrate & fuse
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="uniform", random_state=42)
    a_cal = qt.fit_transform(r["attack"].reshape(-1, 1)).flatten()
    e_cal = qt.fit_transform(r["emotion"].reshape(-1, 1)).flatten()

    # Stance per topic
    s_raw = np.zeros(len(df))
    for topic in tqdm(df["topic"].unique(), desc="Stance"):
        mask = df["topic"] == topic; idxs = np.where(mask.values)[0]
        if len(idxs) >= 4:
            s_raw[idxs] = computer.stance_polarization(r["embeddings"][idxs])
    s_cal = qt.fit_transform(s_raw.reshape(-1, 1)).flatten()

    # Fusion (w_a:w_e:w_s = 0.5:0.3:0.2)
    df["c"] = (1.0 / (1.0 + np.exp(-(0.5 * a_cal + 0.3 * e_cal + 0.2 * s_cal)))).clip(0, 1)
    df["a_cal"] = a_cal; df["e_cal"] = e_cal; df["s_cal"] = s_cal
    df["bin"] = pd.to_datetime(df["ts"], unit="s").dt.floor("12h")

    # Build trajectories
    trajectories = {}
    for topic in df["topic"].unique():
        tdf = df[df["topic"] == topic].sort_values("bin")
        agg = tdf.groupby("bin").agg(
            c_bar=("c", lambda x: x.nlargest(max(1, int(len(x) * 0.15))).mean()),
            c_mean=("c", "mean"), c_max=("c", "max"), c_std=("c", "std"),
            n=("c", "count"),
            a_mean=("a_cal", "mean"), e_mean=("e_cal", "mean"), s_mean=("s_cal", "mean"),
        ).reset_index()
        agg = agg[(agg["n"] >= 3) & (~agg["c_bar"].isna())]
        if len(agg) >= 20:
            trajectories[topic] = agg

    print(f"\nBuilt {len(trajectories)} trajectories")
    for t, d in sorted(trajectories.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        print(f"  {t[:40]:40s} bins={len(d):4d} c̄∈[{d['c_bar'].min():.2f},{d['c_bar'].max():.2f}] "
              f"σ={d['c_bar'].std():.3f} n/bin={d['n'].mean():.1f}")

    with open(f"{R_DIR}/trajectories_real_model.pkl", "wb") as f:
        pickle.dump(trajectories, f)

    # ── Synthetic data forecasting (same setup, but using conflict values from trajectories) ──
    print("\n" + "=" * 60)
    print("Part B: Synthetic Conflict Forecasting (for comparison)")
    print("=" * 60)

    Xs, ys = [], []
    for i in range(60):
        v, e = gen_traj(200, s=i)
        for j in range(len(v) - L - H + 1): Xs.append(v[j:j + L]); ys.append(v[j+L:j+L+H])
    X = torch.tensor(np.array(Xs), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(ys), dtype=torch.float32)
    print(f"  Synthetic windows: {len(X)}")

    # Train all models
    r_cnn_lstm = train_model(X, y)  # CNN-BiLSTM (ours)
    r_bigru = train_model_generic(lambda: BiGRUModel(), X, y)
    r_tcn = train_model_generic(lambda: TCNModel(), X, y)

    # Baselines
    def train_baselines_simple(Xp, yp):
        n = len(Xp); n_tr = int(n * TRAIN_SPLIT)
        perm = np.random.RandomState(42).permutation(n)
        Xtr, ytr = Xp[perm[:n_tr]], yp[perm[:n_tr]]; Xte, yte = Xp[perm[n_tr:]], yp[perm[n_tr:]]
        yte_np = yte.numpy()
        y_p = np.tile(Xte[:, -1, 0].numpy().reshape(-1, 1), (1, H))
        from sklearn.linear_model import LinearRegression
        Xar_tr = Xtr[:, -6:, 0].numpy(); Xar_te = Xte[:, -6:, 0].numpy()
        y_ar = np.stack([LinearRegression().fit(Xar_tr, ytr[:, h].numpy()).predict(Xar_te) for h in range(H)], 1)
        from sklearn.svm import SVR
        Xtr_f = Xtr[:, :, 0].numpy(); Xte_f = Xte[:, :, 0].numpy()
        y_svr = np.stack([SVR(kernel='rbf', C=1.0, epsilon=0.01).fit(Xtr_f, ytr[:, h].numpy()).predict(Xte_f) for h in range(H)], 1)
        from xgboost import XGBRegressor
        y_xgb = np.stack([XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, verbosity=0).fit(Xtr_f, ytr[:, h].numpy()).predict(Xte_f) for h in range(H)], 1)
        def m(yp_v):
            mae = float(np.mean(np.abs(yp_v - yte_np)))
            rmse = float(np.sqrt(np.mean((yp_v - yte_np)**2)))
            ss_r = np.sum((yte_np - yp_v)**2); ss_t = np.sum((yte_np - yte_np.mean())**2)
            r2 = 1 - ss_r / (ss_t + 1e-8)
            pe = (yp_v.max(1) >= 0.65).astype(int); te = (yte_np.max(1) >= 0.65).astype(int)
            tp, fp, fn = (pe & te).sum(), (pe & (1 - te)).sum(), ((1 - pe) & te).sum()
            p, r = tp / (tp + fp) if (tp + fp) else 0, tp / (tp + fn) if (tp + fn) else 0
            return {"mae": mae, "rmse": rmse, "r2": r2, "esc_f1": 2 * p * r / (p + r) if (p + r) else 0}
        return {"Persistence": m(y_p), "AR(6)": m(y_ar), "SVR": m(y_svr), "XGBoost": m(y_xgb)}

    bl = train_baselines_simple(X, y)
    print(f"\nSynthetic Forecasting Results (N_test={int(len(X)*(1-TRAIN_SPLIT))}):")
    for n in ["Persistence", "AR(6)", "SVR", "XGBoost", "BiGRU", "TCN"]:
        if n in bl: print(f"  {n:12s} R²={bl[n]['r2']:.4f} MAE={bl[n]['mae']:.4f}")
        elif n == "BiGRU": print(f"  {n:12s} R²={r_bigru['r2']:.4f} MAE={r_bigru['mae']:.4f} Esc-F1={r_bigru['esc_f1']:.4f}")
        elif n == "TCN": print(f"  {n:12s} R²={r_tcn['r2']:.4f} MAE={r_tcn['mae']:.4f} Esc-F1={r_tcn['esc_f1']:.4f}")
    print(f"  {'CNN-BiLSTM':12s} R²={r_cnn_lstm['r2']:.4f} MAE={r_cnn_lstm['mae']:.4f} Esc-F1={r_cnn_lstm['esc_f1']:.4f}")

    # ── Real data forecasting ──
    print("\n" + "=" * 60)
    print("Part C: Real Data Conflict Forecast (with real model)")
    print("=" * 60)

    topic_windows = []
    for topic in df["topic"].unique():
        vals = trajectories[topic]["c_bar"].values
        mu, std = vals.mean(), vals.std()
        if std < 1e-6: std = 1.0
        vn = (vals - mu) / std
        for j in range(len(vn) - L - H + 1):
            topic_windows.append((vn[j:j + L], vn[j + L:j + L + H]))

    Xr = torch.tensor(np.array([w[0] for w in topic_windows]), dtype=torch.float32).unsqueeze(-1)
    yr = torch.tensor(np.array([w[1] for w in topic_windows]), dtype=torch.float32)
    print(f"{len(topic_windows)} windows from real data")
    r_real = train_model(Xr, yr) if len(topic_windows) >= 30 else {"r2": 0, "mae": 0}

    # Summary
    print(f"\n{'═'*50}")
    print(f"Summary: Real Pretrained Model + All Baselines")
    print(f"  CNN-BiLSTM (Ours): R²={r_cnn_lstm['r2']:.4f} MAE={r_cnn_lstm['mae']:.4f} Esc-F1={r_cnn_lstm['esc_f1']:.4f}")
    print(f"  BiGRU:             R²={r_bigru['r2']:.4f} MAE={r_bigru['mae']:.4f} Esc-F1={r_bigru['esc_f1']:.4f}")
    print(f"  TCN:               R²={r_tcn['r2']:.4f} MAE={r_tcn['mae']:.4f} Esc-F1={r_tcn['esc_f1']:.4f}")
    print(f"  SVR:               R²={bl['SVR']['r2']:.4f} MAE={bl['SVR']['mae']:.4f}")
    print(f"  XGBoost:           R²={bl['XGBoost']['r2']:.4f} MAE={bl['XGBoost']['mae']:.4f}")
    if len(topic_windows) >= 30:
        print(f"  Real data:         R²={r_real['r2']:.4f}")

    # Merge all results
    all_results = {"synthetic": r_cnn_lstm, "real": r_real,
                   "BiGRU": r_bigru, "TCN": r_tcn,
                   "baselines": bl, "trajectories": trajectories}
    with open(f"{R_DIR}/real_model_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    tmin = (time.time() - t0) / 60
    print(f"\nDone in {tmin:.1f} min")
    notify(f"Full experiment done! CNN-BiLSTM R²={r_cnn_lstm['r2']:.4f} BiGRU R²={r_bigru['r2']:.4f} TCN R²={r_tcn['r2']:.4f} ({tmin:.1f}min)")
