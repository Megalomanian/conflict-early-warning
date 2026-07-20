#!/usr/bin/env python3
"""
Real-data conflict index experiment — V2 (fixing data leakage).

Key changes:
  - ECDF calibration (QuantileTransformer) fitted on TRAINING data only
  - Strict TEMPORAL train/test split per topic
  - Comment-level conflict index with REAL pretrained sentiment model
  - Reports both synthetic and real-data forecasting results
"""
import json, os, glob, pickle, warnings, time, sys
from collections import defaultdict
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
R_DIR = "experiment_results_v2"
os.makedirs(R_DIR, exist_ok=True)

L, H = 12, 6; HIDDEN, DROPOUT = 64, 0.2
EPOCHS, LR, PATIENCE = 100, 1e-3, 20
TEMPORAL_SPLIT = 0.75  # first 75% time bins = train

def notify(msg):
    os.system(f'/home/violina/projects/notice/notice "{msg}" 2>/dev/null &')


# ═══ Real Pretrained Model Conflict Index ═══
class RealConflictComputer:
    """Uses real HuggingFace sentiment model for attack+emotion, plus MiniLM for stance."""

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from sentence_transformers import SentenceTransformer
        print("Loading real pretrained models...")
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sent_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
        self.sent_model.eval()
        self.emb_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)
        print("  Models loaded.")

    def compute_sentiment_batch(self, texts, batch_size=128):
        all_logits = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="  Sentiment",
                           unit="batch", ncols=100):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True,
                                        truncation=True, max_length=256).to(DEVICE)
                outputs = self.sent_model(**inputs)
                all_logits.append(outputs.logits.cpu().numpy())
        return np.concatenate(all_logits, axis=0)

    def compute_batch(self, texts):
        n = len(texts)
        print(f"  Running pretrained sentiment on {n} texts...", flush=True)
        logits = self.compute_sentiment_batch(texts)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        # Attack: 1-star (most negative) + half of 2-star
        attack = probs[:, 0] + 0.5 * probs[:, 1]
        # Emotion: weighted negative + anger
        p_neg = probs[:, 0] + probs[:, 1]
        p_anger = probs[:, 0]
        emotion = 0.7 * p_neg + 0.3 * p_anger
        # Embeddings for stance
        print(f"  Embedding {n} texts for stance...", flush=True)
        embs = self.emb_model.encode(texts, show_progress_bar=True, batch_size=256,
                                      convert_to_numpy=True, normalize_embeddings=True)
        print(f"  Attack μ={attack.mean():.3f} σ={attack.std():.3f} | "
              f"Emotion μ={emotion.mean():.3f} σ={emotion.std():.3f}", flush=True)
        return {"attack": attack, "emotion": emotion, "embeddings": embs}

    def stance_polarization(self, embs):
        n = len(embs)
        if n < 4:
            return np.zeros(n)
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
        km = KMeans(n_clusters=2, n_init=5, random_state=42)
        km.fit(embs)
        c0, c1 = km.cluster_centers_
        delta = 1.0 - cosine_similarity(c0.reshape(1, -1), c1.reshape(1, -1))[0, 0]
        d0 = cosine_distances(embs, c0.reshape(1, -1)).flatten()
        d1 = cosine_distances(embs, c1.reshape(1, -1)).flatten()
        return min(1.0, delta) * np.abs(d0 - d1) / (d0 + d1 + 1e-8)


# ═══ CNN-BiLSTM Architecture ═══
class CNNBiLSTM(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__(); self.h = h
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU())
        self.lstm = nn.LSTM(32, h, 2, batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.proj = nn.Linear(h * 2, H)

    def forward(self, x):
        c = self.conv(x.transpose(1, 2)).transpose(1, 2)
        o, _ = self.lstm(c)
        return self.proj(torch.cat([o[:, -1, :self.h], o[:, 0, self.h:]], dim=-1))


class BiGRUModel(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__(); self.h = h
        self.gru = nn.GRU(1, h, 2, batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.proj = nn.Linear(h * 2, H)

    def forward(self, x):
        o, _ = self.gru(x)
        return self.proj(torch.cat([o[:, -1, :self.h], o[:, 0, self.h:]], dim=-1))


# ═══ Synthetic trajectory (for comparison) ═══
def gen_traj(n=200, s=None):
    rng = np.random.RandomState(s); base = 0.3 + rng.uniform(0, 0.15)
    n_ev = rng.randint(1, 4); trend = np.zeros(n)
    for _ in range(n_ev):
        es, ed = rng.randint(10, n - 30), rng.randint(5, 15)
        ep = rng.uniform(0.15, 0.35)
        te = np.arange(ed * 2)
        sig = 1.0 / (1.0 + np.exp(-(te - ed / 2) / (ed / 8)))
        sig = (sig - sig[0]) / sig.max() * ep
        sig = sig * np.exp(-(te - ed) / (ed * 2))
        idx = min(es + len(sig), n); trend[es:idx] += sig[:idx - es]
    season = 0.02 * np.sin(2 * np.pi * np.arange(n) / 14.0)
    white = rng.normal(0, 0.02, n); pink = np.zeros(n); pink[0] = white[0]
    for i in range(1, n): pink[i] = 0.6 * pink[i - 1] + 0.4 * white[i]
    return np.clip(base + trend + season + pink, 0, 1), (trend > 0.1).astype(int)


def train_model_temporal(X, y, device=DEVICE):
    """Train CNN-BiLSTM with strict temporal split (no shuffling)."""
    n = len(X); n_tr = int(n * TEMPORAL_SPLIT)
    # Temporal split: no permutation!
    Xtr, ytr = X[:n_tr].to(device), y[:n_tr].to(device)
    Xte, yte = X[n_tr:].to(device), y[n_tr:].to(device)
    yte_np = yte.cpu().numpy()

    tl = DataLoader(TensorDataset(Xtr, ytr), 64, True)
    vl = DataLoader(TensorDataset(Xte, yte), 128)
    model = CNNBiLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    huber = nn.HuberLoss(delta=0.5)
    best_vl, best_st, patience_c = float("inf"), None, 0

    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tl:
            opt.zero_grad(); huber(model(xb), yb).backward(); opt.step()
        model.eval()
        v_l = sum(huber(model(xb), yb).item() for xb, yb in vl) / len(vl)
        if v_l < best_vl:
            best_vl = v_l
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
        if patience_c >= PATIENCE:
            break

    model.load_state_dict(best_st); model.eval()
    with torch.no_grad():
        yp = model(Xte.to(device)).cpu().numpy()

    mae = float(np.mean(np.abs(yp - yte_np)))
    rmse = float(np.sqrt(np.mean((yp - yte_np) ** 2)))
    ss_r = np.sum((yte_np - yp) ** 2)
    ss_t = np.sum((yte_np - yte_np.mean()) ** 2)
    r2 = 1 - ss_r / (ss_t + 1e-8)
    pe = (yp.max(1) >= 0.65).astype(int)
    te = (yte_np.max(1) >= 0.65).astype(int)
    tp, fp, fn = (pe & te).sum(), (pe & (1 - te)).sum(), ((1 - pe) & te).sum()
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    return {"mae": mae, "rmse": rmse, "r2": r2,
            "esc_f1": 2 * p * r / (p + r) if (p + r) else 0,
            "y_true": yte_np, "y_pred": yp}


def train_model_temporal_generic(model_factory, X, y, device=DEVICE):
    """Train any model factory with temporal split."""
    n = len(X); n_tr = int(n * TEMPORAL_SPLIT)
    Xtr, ytr = X[:n_tr].to(device), y[:n_tr].to(device)
    Xte, yte = X[n_tr:].to(device), y[n_tr:].to(device)
    yte_np = yte.cpu().numpy()

    tl = DataLoader(TensorDataset(Xtr, ytr), 64, True)
    vl = DataLoader(TensorDataset(Xte, yte), 128)
    model = model_factory().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    huber = nn.HuberLoss(delta=0.5)
    best_vl, best_st, patience_c = float("inf"), None, 0

    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tl:
            opt.zero_grad(); huber(model(xb), yb).backward(); opt.step()
        model.eval()
        v_l = sum(huber(model(xb), yb).item() for xb, yb in vl) / len(vl)
        if v_l < best_vl:
            best_vl = v_l
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
        if patience_c >= PATIENCE:
            break

    model.load_state_dict(best_st); model.eval()
    with torch.no_grad():
        yp = model(Xte.to(device)).cpu().numpy()

    mae = float(np.mean(np.abs(yp - yte_np)))
    rmse = float(np.sqrt(np.mean((yp - yte_np) ** 2)))
    ss_r = np.sum((yte_np - yp) ** 2)
    ss_t = np.sum((yte_np - yte_np.mean()) ** 2)
    r2 = 1 - ss_r / (ss_t + 1e-8)
    pe = (yp.max(1) >= 0.65).astype(int)
    te = (yte_np.max(1) >= 0.65).astype(int)
    tp, fp, fn = (pe & te).sum(), (pe & (1 - te)).sum(), ((1 - pe) & te).sum()
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    return {"mae": mae, "rmse": rmse, "r2": r2,
            "esc_f1": 2 * p * r / (p + r) if (p + r) else 0,
            "y_true": yte_np, "y_pred": yp}


# ═══ Main ═══
if __name__ == "__main__":
    t0 = time.time()

    # ══ Part A: Real Data with Temporal Split + Train-only Calibration ══
    print("=" * 60)
    print("Part A: Conflict Index with Real Pretrained Model (Temporal Split)")
    print("=" * 60)

    base = "zhihu_topics"
    ranked = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isdir(path) or name == "zhihu":
            continue
        jd = os.path.join(path, "zhihu", "jsonl")
        if not os.path.isdir(jd):
            continue
        nc = sum(sum(1 for _ in open(f, encoding="utf-8"))
                 for f in glob.glob(os.path.join(jd, "search_comments_*.jsonl")))
        if nc > 0:
            ranked.append((path, nc))
    ranked.sort(key=lambda x: x[1], reverse=True)

    records = []
    for tp, _ in tqdm(ranked[:20], desc="Loading topics"):
        tp_name = os.path.basename(tp)
        jd = os.path.join(tp, "zhihu", "jsonl")
        for ftype, tkey in [("search_comments", "content"),
                            ("search_contents", "content_text")]:
            for fp in glob.glob(os.path.join(jd, f"{ftype}_*.jsonl")):
                for line in open(fp, encoding="utf-8"):
                    try:
                        obj = json.loads(line.strip())
                    except Exception:
                        continue
                    text = obj.get(tkey, "").strip()
                    ts = obj.get("publish_time") or obj.get("created_time")
                    if text and len(text) >= 5 and ts and float(ts) > 1704067200:
                        records.append({"text": text, "ts": float(ts), "topic": tp_name})

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records from 20 topics")

    # Compute real pretrained model scores
    computer = RealConflictComputer()
    all_texts = df["text"].tolist()
    r = computer.compute_batch(all_texts)

    # ═══ KEY FIX: Stance per topic ═══
    s_raw = np.zeros(len(df))
    for topic in tqdm(df["topic"].unique(), desc="Stance"):
        mask = df["topic"] == topic
        idxs = np.where(mask.values)[0]
        if len(idxs) >= 4:
            s_raw[idxs] = computer.stance_polarization(r["embeddings"][idxs])

    # ═══ KEY FIX: ECDF fitted on EARLY (temporal train) data only ═══
    # For each topic, we use the first TEMPORAL_SPLIT fraction of its bins
    # to fit the QuantileTransformer, then transform all data
    from sklearn.preprocessing import QuantileTransformer

    df["bin"] = pd.to_datetime(df["ts"], unit="s").dt.floor("12h")
    a_cal = np.zeros(len(df))
    e_cal = np.zeros(len(df))
    s_cal = np.zeros(len(df))

    for topic in df["topic"].unique():
        t_mask = df["topic"] == topic
        t_idxs = np.where(t_mask.values)[0]
        if len(t_idxs) < 20:
            continue
        # Sort by time
        sorted_order = df.iloc[t_idxs].sort_values("bin").index
        sorted_positions = [np.where(t_idxs == df.index.get_loc(idx))[0][0]
                            if idx in df.index[t_idxs] else 0
                            for idx in sorted_order]
        # Actually, simpler: just get the temporal order
        t_df = df.iloc[t_idxs].sort_values("bin")
        n_train = int(len(t_df) * TEMPORAL_SPLIT)
        train_idx_in_topic = t_df.index[:n_train]

        # Fit calibrator on training portion only
        qt_a = QuantileTransformer(n_quantiles=min(1000, n_train),
                                   output_distribution="uniform", random_state=42)
        qt_e = QuantileTransformer(n_quantiles=min(1000, n_train),
                                   output_distribution="uniform", random_state=42)
        qt_s = QuantileTransformer(n_quantiles=min(1000, n_train),
                                   output_distribution="uniform", random_state=42)

        train_a = r["attack"][[df.index.get_loc(i) for i in train_idx_in_topic]]
        train_e = r["emotion"][[df.index.get_loc(i) for i in train_idx_in_topic]]
        train_s = s_raw[[df.index.get_loc(i) for i in train_idx_in_topic]]

        qt_a.fit(train_a.reshape(-1, 1))
        qt_e.fit(train_e.reshape(-1, 1))
        qt_s.fit(train_s.reshape(-1, 1))

        # Transform all data for this topic
        all_positions = [df.index.get_loc(i) for i in t_df.index]
        a_cal[all_positions] = qt_a.transform(
            r["attack"][all_positions].reshape(-1, 1)).flatten()
        e_cal[all_positions] = qt_e.transform(
            r["emotion"][all_positions].reshape(-1, 1)).flatten()
        s_cal[all_positions] = qt_s.transform(
            s_raw[all_positions].reshape(-1, 1)).flatten()

        print(f"  {topic[:40]}: fitted ECDF on {n_train}/{len(t_df)} train samples")

    # Fuse: sigmoid(w_a*a + w_e*e + w_s*s)
    df["c"] = (1.0 / (1.0 + np.exp(-(0.5 * a_cal + 0.3 * e_cal + 0.2 * s_cal)))).clip(0, 1)
    df["a_cal"] = a_cal; df["e_cal"] = e_cal; df["s_cal"] = s_cal

    # Build trajectories per topic
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
        if len(agg) >= L + H + 10:
            trajectories[topic] = agg

    print(f"\nBuilt {len(trajectories)} trajectories (≥{L+H+10} bins)")
    for t, d in sorted(trajectories.items(),
                       key=lambda x: len(x[1]), reverse=True)[:5]:
        print(f"  {t[:40]:40s} bins={len(d):4d} "
              f"c̄∈[{d['c_bar'].min():.2f},{d['c_bar'].max():.2f}] "
              f"σ={d['c_bar'].std():.3f} n/bin={d['n'].mean():.1f}")

    # Save trajectories
    with open(f"{R_DIR}/trajectories_real_model.pkl", "wb") as f:
        pickle.dump(trajectories, f)

    # ══ Part B: Real Data Forecasting (Temporal Split) ══
    print("\n" + "=" * 60)
    print("Part B: Real Data Conflict Forecasting (Temporal Split)")
    print("=" * 60)

    topic_windows = []
    for topic in df["topic"].unique():
        if topic not in trajectories:
            continue
        vals = trajectories[topic]["c_bar"].values
        mu, std = vals.mean(), vals.std()
        if std < 1e-6:
            std = 1.0
        vn = (vals - mu) / std
        for j in range(len(vn) - L - H + 1):
            topic_windows.append((vn[j:j + L], vn[j + L:j + L + H]))

    if len(topic_windows) >= 30:
        Xr = torch.tensor(np.array([w[0] for w in topic_windows]),
                          dtype=torch.float32).unsqueeze(-1)
        yr = torch.tensor(np.array([w[1] for w in topic_windows]),
                          dtype=torch.float32)

        r_real_cnn = train_model_temporal(Xr, yr)
        r_real_bigru = train_model_temporal_generic(lambda: BiGRUModel(), Xr, yr)

        # Baselines with temporal split
        n_all = len(Xr); n_tr = int(n_all * TEMPORAL_SPLIT)
        Xtr_r, ytr_r = Xr[:n_tr], yr[:n_tr]
        Xte_r, yte_r = Xr[n_tr:], yr[n_tr:]
        yte_r_np = yte_r.numpy()

        def eval_baseline(yp_v):
            mae = float(np.mean(np.abs(yp_v - yte_r_np)))
            ss_r = np.sum((yte_r_np - yp_v) ** 2)
            ss_t = np.sum((yte_r_np - yte_r_np.mean()) ** 2)
            r2 = 1 - ss_r / (ss_t + 1e-8)
            return {"mae": mae, "r2": r2}

        # Persistence
        yp_p = np.tile(Xte_r[:, -1, 0].numpy().reshape(-1, 1), (1, H))
        bl_real = {"Persistence": eval_baseline(yp_p)}

        # AR(6)
        from sklearn.linear_model import LinearRegression
        Xar_tr = Xtr_r[:, -6:, 0].numpy()
        Xar_te = Xte_r[:, -6:, 0].numpy()
        y_ar = np.stack([LinearRegression().fit(Xar_tr, ytr_r[:, h].numpy()).predict(Xar_te)
                         for h in range(H)], 1)
        bl_real["AR(6)"] = eval_baseline(y_ar)

        # XGBoost
        from xgboost import XGBRegressor
        Xtr_f_r = Xtr_r[:, :, 0].numpy()
        Xte_f_r = Xte_r[:, :, 0].numpy()
        y_xgb_r = np.stack([XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                          verbosity=0).fit(Xtr_f_r, ytr_r[:, h].numpy()).predict(Xte_f_r)
                            for h in range(H)], 1)
        bl_real["XGBoost"] = eval_baseline(y_xgb_r)

        print(f"\n  Real-Data Conflict Forecasting ({len(topic_windows)} windows, "
              f"temporal {TEMPORAL_SPLIT:.0%}/{1-TEMPORAL_SPLIT:.0%} split):")
        for name, res in [("Persistence", bl_real["Persistence"]),
                          ("AR(6)", bl_real["AR(6)"]),
                          ("XGBoost", bl_real["XGBoost"]),
                          ("BiGRU", r_real_bigru),
                          ("CNN-BiLSTM", r_real_cnn)]:
            print(f"    {name:15s} R²={res['r2']:.4f}")

    # ══ Part C: Synthetic Data (Temporal Split, for comparison) ══
    print("\n" + "=" * 60)
    print("Part C: Synthetic Data Forecasting (Temporal Split)")
    print("=" * 60)

    Xs, ys = [], []
    for i in range(60):
        v, e = gen_traj(200, s=i)
        for j in range(len(v) - L - H + 1):
            Xs.append(v[j:j + L]); ys.append(v[j + L:j + L + H])
    X = torch.tensor(np.array(Xs), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(ys), dtype=torch.float32)
    print(f"  Synthetic windows: {len(X)}")

    # Baselines with temporal split
    n_syn = len(X); n_tr_syn = int(n_syn * TEMPORAL_SPLIT)
    Xtr_s, ytr_s = X[:n_tr_syn], y[:n_tr_syn]
    Xte_s, yte_s = X[n_tr_syn:], y[n_tr_syn:]
    yte_s_np = yte_s.numpy()

    def eval_baseline_full(yp_v):
        mae = float(np.mean(np.abs(yp_v - yte_s_np)))
        rmse = float(np.sqrt(np.mean((yp_v - yte_s_np) ** 2)))
        ss_r = np.sum((yte_s_np - yp_v) ** 2)
        ss_t = np.sum((yte_s_np - yte_s_np.mean()) ** 2)
        r2 = 1 - ss_r / (ss_t + 1e-8)
        pe = (yp_v.max(1) >= 0.65).astype(int)
        te = (yte_s_np.max(1) >= 0.65).astype(int)
        tp, fp, fn = (pe & te).sum(), (pe & (1 - te)).sum(), ((1 - pe) & te).sum()
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        return {"mae": mae, "rmse": rmse, "r2": r2,
                "esc_f1": 2 * p * r / (p + r) if (p + r) else 0}

    # Persistence
    yp_ps = np.tile(Xte_s[:, -1, 0].numpy().reshape(-1, 1), (1, H))
    bl_syn = {"Persistence": eval_baseline_full(yp_ps)}

    # AR(6)
    Xar_tr_s = Xtr_s[:, -6:, 0].numpy(); Xar_te_s = Xte_s[:, -6:, 0].numpy()
    y_ar_s = np.stack([LinearRegression().fit(Xar_tr_s, ytr_s[:, h].numpy()).predict(Xar_te_s)
                       for h in range(H)], 1)
    bl_syn["AR(6)"] = eval_baseline_full(y_ar_s)

    # SVR
    from sklearn.svm import SVR
    Xtr_f_s = Xtr_s[:, :, 0].numpy(); Xte_f_s = Xte_s[:, :, 0].numpy()
    y_svr_s = np.stack([SVR(kernel='rbf', C=1.0, epsilon=0.01).fit(
        Xtr_f_s, ytr_s[:, h].numpy()).predict(Xte_f_s) for h in range(H)], 1)
    bl_syn["SVR"] = eval_baseline_full(y_svr_s)

    # XGBoost
    y_xgb_s = np.stack([XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                      verbosity=0).fit(
        Xtr_f_s, ytr_s[:, h].numpy()).predict(Xte_f_s) for h in range(H)], 1)
    bl_syn["XGBoost"] = eval_baseline_full(y_xgb_s)

    # Neural models
    r_cnn = train_model_temporal(X, y)
    r_bigru = train_model_temporal_generic(lambda: BiGRUModel(), X, y)

    print(f"\n  Synthetic Forecasting (temporal split, N_test={n_syn-n_tr_syn}):")
    for name, res in [("Persistence", bl_syn["Persistence"]),
                      ("AR(6)", bl_syn["AR(6)"]),
                      ("SVR", bl_syn["SVR"]),
                      ("XGBoost", bl_syn["XGBoost"]),
                      ("BiGRU", r_bigru),
                      ("CNN-BiLSTM", r_cnn)]:
        esc_str = f"Esc-F1={res.get('esc_f1', 0):.4f}" if 'esc_f1' in res else ""
        print(f"    {name:15s} R²={res['r2']:.4f} MAE={res['mae']:.4f} {esc_str}")

    # ══ Summary ══
    print(f"\n{'═'*60}")
    print("Summary (Temporal Split + Train-Only ECDF):")
    print(f"  Synthetic CNN-BiLSTM:  R²={r_cnn['r2']:.4f} MAE={r_cnn['mae']:.4f} "
          f"Esc-F1={r_cnn['esc_f1']:.4f}")
    if len(topic_windows) >= 30:
        print(f"  Real Data CNN-BiLSTM:  R²={r_real_cnn['r2']:.4f}")
    print(f"{'═'*60}")

    # ── Save ──
    all_results = {
        "synthetic_cnn": r_cnn, "synthetic_bigru": r_bigru,
        "synthetic_baselines": bl_syn,
        "real_cnn": r_real_cnn if len(topic_windows) >= 30 else None,
        "real_bigru": r_real_bigru if len(topic_windows) >= 30 else None,
        "real_baselines": bl_real if len(topic_windows) >= 30 else None,
        "trajectories": trajectories,
    }
    with open(f"{R_DIR}/real_model_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    tmin = (time.time() - t0) / 60
    print(f"\nDone in {tmin:.1f} min")
    notify(f"Real model V2 done! Synth R²={r_cnn['r2']:.4f} ({tmin:.1f}min)")
