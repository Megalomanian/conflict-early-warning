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
LOCAL_FILES_ONLY = (os.environ.get("HF_HUB_OFFLINE") == "1" or
                    os.environ.get("TRANSFORMERS_OFFLINE") == "1")
R_DIR = "experiment_results_v2"
os.makedirs(R_DIR, exist_ok=True)

L, H = 12, 6; HIDDEN, DROPOUT = 64, 0.2
EPOCHS, LR, PATIENCE = 100, 1e-3, 20
TEMPORAL_SPLIT = 0.75  # first 75% time bins = train
RISK_TRAIN_END = 0.60  # preprocessing boundary for 60/15/25 risk experiment
SMOOTH_SPAN = 6  # causal EWMA over 6 x 12h bins (3 days)
MIN_FUTURE_OBS = 2

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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=LOCAL_FILES_ONLY)
        self.sent_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, local_files_only=LOCAL_FILES_ONLY).to(DEVICE)
        self.sent_model.eval()
        emb_kwargs = {"device": DEVICE}
        if LOCAL_FILES_ONLY:
            emb_kwargs["local_files_only"] = True
        self.emb_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2", **emb_kwargs)
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

    def stance_polarization(self, train_embs, all_embs):
        """Fit stance camps on training embeddings and score any time period."""
        if len(train_embs) < 4:
            return np.zeros(len(all_embs))
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
        km = KMeans(n_clusters=2, n_init=5, random_state=42)
        km.fit(train_embs)
        c0, c1 = km.cluster_centers_
        delta = 1.0 - cosine_similarity(c0.reshape(1, -1), c1.reshape(1, -1))[0, 0]
        d0 = cosine_distances(all_embs, c0.reshape(1, -1)).flatten()
        d1 = cosine_distances(all_embs, c1.reshape(1, -1)).flatten()
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


def build_topic_temporal_windows(trajectories, target="c_bar", smooth_span=1):
    """Build leakage-free train/test windows with a temporal split per topic.

    Normalization is fitted on each topic's training bins only. Windows whose
    targets cross the split boundary are discarded; test inputs may use the
    immediately preceding training history, as in online forecasting.
    """
    train_windows, test_windows = [], []
    for trajectory in trajectories.values():
        trajectory = trajectory.copy()
        trajectory["bin"] = pd.to_datetime(trajectory["bin"])
        trajectory = trajectory.set_index("bin").sort_index()
        full_index = pd.date_range(
            trajectory.index.min(), trajectory.index.max(), freq="12h")
        raw_values = trajectory[target].astype(float).reindex(full_index)
        observed = raw_values.notna().astype(int).to_numpy()
        split_bin = int(len(full_index) * TEMPORAL_SPLIT)
        train_observed = raw_values.iloc[:split_bin].dropna()
        if len(train_observed) < L + H:
            continue

        lower, upper = train_observed.quantile([0.01, 0.99])
        values = raw_values.clip(lower, upper).fillna(train_observed.median())
        if smooth_span > 1:
            values = values.ewm(span=smooth_span, adjust=False).mean()
        values = values.to_numpy()
        train_values = values[:split_bin]
        mu = train_values.mean()
        std = train_values.std()
        if std < 1e-6:
            std = 1.0
        normalized = (values - mu) / std

        for start in range(len(normalized) - L - H + 1):
            target_start = start + L
            target_end = target_start + H
            if observed[target_start:target_end].sum() < MIN_FUTURE_OBS:
                continue
            window = (normalized[start:target_start],
                      normalized[target_start:target_end])
            if target_end <= split_bin:
                train_windows.append(window)
            elif target_start >= split_bin:
                test_windows.append(window)

    def to_tensors(windows):
        X = torch.tensor(np.asarray([w[0] for w in windows]),
                         dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(np.asarray([w[1] for w in windows]),
                         dtype=torch.float32)
        return X, y

    return (*to_tensors(train_windows), *to_tensors(test_windows))


def evaluate_ridge(X_train, y_train, X_test, y_test, alpha=10.0):
    """Evaluate a regularized linear baseline on pre-split windows."""
    from sklearn.linear_model import Ridge

    train_features = X_train[:, :, 0].numpy()
    test_features = X_test[:, :, 0].numpy()
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()
    predictions = np.stack([
        Ridge(alpha=alpha).fit(train_features, y_train_np[:, h]).predict(test_features)
        for h in range(H)
    ], axis=1)
    mae = float(np.mean(np.abs(predictions - y_test_np)))
    rmse = float(np.sqrt(np.mean((predictions - y_test_np) ** 2)))
    ss_res = np.sum((y_test_np - predictions) ** 2)
    ss_tot = np.sum((y_test_np - y_test_np.mean()) ** 2)
    return {"mae": mae, "rmse": rmse,
            "r2": float(1 - ss_res / (ss_tot + 1e-8)),
            "y_true": y_test_np, "y_pred": predictions}


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

    # ═══ Leakage-free stance and ECDF calibration ═══
    # Use the exact first-60%-of-regular-grid boundary later used by the risk
    # experiment.  Test comments never influence K-Means centers or ECDFs.
    from sklearn.preprocessing import QuantileTransformer

    df["bin"] = pd.to_datetime(df["ts"], unit="s").dt.floor("12h")
    s_raw = np.zeros(len(df))
    a_cal = np.zeros(len(df))
    e_cal = np.zeros(len(df))
    s_cal = np.zeros(len(df))
    preprocessing_audit = {}

    for topic in df["topic"].unique():
        t_df = df[df["topic"] == topic].sort_values("bin")
        if len(t_df) < 20:
            continue

        # Match the downstream trajectory endpoints: only bins with >=3 raw
        # records survive aggregation, but gaps remain on the regular grid.
        retained_bins = t_df.groupby("bin").size()
        retained_bins = retained_bins[retained_bins >= 3].index
        if len(retained_bins) < L + H + 10:
            continue
        full_index = pd.date_range(retained_bins.min(), retained_bins.max(), freq="12h")
        train_end = int(len(full_index) * RISK_TRAIN_END)
        if train_end <= 0 or train_end >= len(full_index):
            continue
        train_cutoff = full_index[train_end]
        train_df = t_df[t_df["bin"] < train_cutoff]
        if len(train_df) < 4:
            continue

        train_positions = np.asarray([df.index.get_loc(i) for i in train_df.index])
        all_positions = np.asarray([df.index.get_loc(i) for i in t_df.index])

        # Fit semantic camps on training comments only, then score all periods
        # against the fixed training-period centers.
        s_raw[all_positions] = computer.stance_polarization(
            r["embeddings"][train_positions], r["embeddings"][all_positions])

        # Fit calibrator on training portion only
        n_train = len(train_positions)
        qt_a = QuantileTransformer(n_quantiles=min(1000, n_train),
                                   output_distribution="uniform", random_state=42)
        qt_e = QuantileTransformer(n_quantiles=min(1000, n_train),
                                   output_distribution="uniform", random_state=42)
        qt_s = QuantileTransformer(n_quantiles=min(1000, n_train),
                                   output_distribution="uniform", random_state=42)

        train_a = r["attack"][train_positions]
        train_e = r["emotion"][train_positions]
        train_s = s_raw[train_positions]

        qt_a.fit(train_a.reshape(-1, 1))
        qt_e.fit(train_e.reshape(-1, 1))
        qt_s.fit(train_s.reshape(-1, 1))

        # Transform all data for this topic
        a_cal[all_positions] = qt_a.transform(
            r["attack"][all_positions].reshape(-1, 1)).flatten()
        e_cal[all_positions] = qt_e.transform(
            r["emotion"][all_positions].reshape(-1, 1)).flatten()
        s_cal[all_positions] = qt_s.transform(
            s_raw[all_positions].reshape(-1, 1)).flatten()

        preprocessing_audit[topic] = {
            "regular_grid_start": str(full_index.min()),
            "train_cutoff_exclusive": str(train_cutoff),
            "regular_grid_end": str(full_index.max()),
            "n_train_comments": int(n_train),
            "n_all_comments": int(len(t_df)),
            "n_test_comments_used_for_fit": 0,
            "stance_fit_scope": "training_comments_only",
            "ecdf_fit_scope": "training_comments_only",
        }
        print(f"  {topic[:40]}: train-only stance/ECDF on "
              f"{n_train}/{len(t_df)} comments, cutoff={train_cutoff}")

    # Fuse: sigmoid(w_a*a + w_e*e + w_s*s)
    df["c"] = (1.0 / (1.0 + np.exp(-(0.5 * a_cal + 0.3 * e_cal + 0.2 * s_cal)))).clip(0, 1)
    df["a_cal"] = a_cal; df["e_cal"] = e_cal; df["s_cal"] = s_cal

    # Build trajectories per topic
    trajectories = {}
    for topic in df["topic"].unique():
        if topic not in preprocessing_audit:
            continue
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
    with open(f"{R_DIR}/preprocessing_audit.json", "w", encoding="utf-8") as f:
        json.dump(preprocessing_audit, f, ensure_ascii=False, indent=2)

    # ══ Part B: Real Data Forecasting (Temporal Split) ══
    print("\n" + "=" * 60)
    print("Part B: Real Data Conflict Forecasting (Temporal Split)")
    print("=" * 60)

    from optimize_real_signal_v2 import run_complete_evaluation

    cleaning_output = run_complete_evaluation(trajectories)
    selected_span = cleaning_output["selected_span"]
    selected_feature_group = cleaning_output["selected_feature_group"]
    real_cleaning_result = cleaning_output["selected_result"]
    test_result = real_cleaning_result["test"]
    print(f"\n  Selected EWMA span on validation F1: {selected_span}")
    print(f"  Selected feature group: {selected_feature_group}")
    print(f"    Logistic:    AUC={test_result['logistic']['auc']:.4f} "
          f"Recall={test_result['logistic']['recall']:.4f} "
          f"F1={test_result['logistic']['f1']:.4f}")
    print(f"    Persistence: AUC={test_result['persistence']['auc']:.4f} "
          f"Recall={test_result['persistence']['recall']:.4f} "
          f"F1={test_result['persistence']['f1']:.4f}")

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
    from sklearn.linear_model import LinearRegression
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
    from xgboost import XGBRegressor
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
    print(f"  Real cleaned risk state: F1={test_result['logistic']['f1']:.4f} "
          f"(persistence={test_result['persistence']['f1']:.4f})")
    print(f"{'═'*60}")

    # ── Save ──
    all_results = {
        "synthetic_cnn": r_cnn, "synthetic_bigru": r_bigru,
        "synthetic_baselines": bl_syn,
        "real_cleaning": real_cleaning_result,
        "real_cleaning_sensitivity": cleaning_output["sensitivity"],
        "trajectories": trajectories,
    }
    with open(f"{R_DIR}/real_model_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    tmin = (time.time() - t0) / 60
    print(f"\nDone in {tmin:.1f} min")
    notify(f"Real model V2 done: cleaned F1={test_result['logistic']['f1']:.4f}, "
           f"persistence={test_result['persistence']['f1']:.4f} ({tmin:.1f}min)")
