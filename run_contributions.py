#!/usr/bin/env python3
"""
Contribution validation experiments:
  C1: Component ablation (A vs E vs S vs full fusion)
  C2: Real-data conflict index forecasting (find topics with ac1>0.15)
  C3: Lead time analysis for early warning triggers

Generates: fig_component_ablation.pdf, fig_lead_time.pdf, fig_real_conflict_forecast.pdf
Also saves all data for tables.
"""
import json, os, glob, pickle, warnings, time, sys
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Style
plt.rcParams.update({"font.family": "serif", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})
IEEE_COL, IEEE_WIDE = 3.5, 7.0

L, H = 12, 6; N_SYNTH, SYNTH_BINS = 60, 200
HIDDEN, DROPOUT = 64, 0.2; EPOCHS, LR, PATIENCE = 200, 1e-3, 40
TRAIN_SPLIT = 0.75
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
R_DIR = "experiment_results"; os.makedirs(R_DIR, exist_ok=True)

def notify(msg): os.system(f'/home/violina/projects/notice/notice "{msg}" 2>/dev/null &')

# ═══ Synthetic trajectory generator ═══
def gen_traj(n_bins=SYNTH_BINS, seed=None):
    rng = np.random.RandomState(seed); n = n_bins
    base = 0.3 + rng.uniform(0, 0.15); n_ev = rng.randint(1, 4); trend = np.zeros(n)
    for _ in range(n_ev):
        es, ed = rng.randint(10, n-30), rng.randint(5, 15); ep = rng.uniform(0.15, 0.35)
        te = np.arange(ed*2)
        sig = 1.0/(1.0+np.exp(-(te-ed/2)/(ed/8)))
        sig = (sig-sig[0])/sig.max()*ep
        sig = sig*np.exp(-(te-ed)/(ed*2))
        idx = min(es+len(sig), n); trend[es:idx] += sig[:idx-es]
    season = 0.02*np.sin(2*np.pi*np.arange(n)/14.0)
    white = rng.normal(0, 0.02, n)
    pink = np.zeros(n); pink[0] = white[0]
    for i in range(1, n): pink[i] = 0.6*pink[i-1]+0.4*white[i]
    y = np.clip(base+trend+season+pink, 0, 1)
    return y, (trend>0.1).astype(int)

def make_synthetic_windows(component_weights=None):
    """Generate synthetic trajectories with optional component weighting.
    component_weights: dict with 'a','e','s' keys (default all 1.0)"""
    if component_weights is None:
        component_weights = {'a': 1.0, 'e': 1.0, 's': 1.0}
    Xs, ys, es = [], [], []
    wa, we, ws = component_weights['a'], component_weights['e'], component_weights['s']
    tw = wa + we + ws
    for i in range(N_SYNTH):
        # Generate three component signals
        y_a, _ = gen_traj(SYNTH_BINS, seed=i*3)
        y_e, _ = gen_traj(SYNTH_BINS, seed=i*3+1)
        y_s, _ = gen_traj(SYNTH_BINS, seed=i*3+2)
        # Weighted fusion
        vals = (wa*y_a + we*y_e + ws*y_s) / tw
        esc = (vals.max()*0.75 + np.percentile(vals, 80))/2 > 0.65
        for j in range(len(vals)-L-H+1):
            Xs.append(vals[j:j+L]); ys.append(vals[j+L:j+L+H])
            es.append(1 if vals[j+L:j+L+H].max()>=0.65 else 0)
    return (torch.tensor(np.array(Xs), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array(ys), dtype=torch.float32), np.array(es))

# ═══ CNN-BiLSTM model ═══
class CNNBiLSTM(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__(); self.h = h
        self.conv = nn.Sequential(nn.Conv1d(1, h//2, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(h//2, h//2, 5, padding=2), nn.ReLU())
        self.lstm = nn.LSTM(h//2, h, 2, batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.proj = nn.Linear(h*2, H)
    def forward(self, x):
        c = self.conv(x.transpose(1,2)).transpose(1,2)
        o, _ = self.lstm(c)
        return self.proj(torch.cat([o[:,-1,:self.h], o[:,0,self.h:]], dim=-1))

def train_model(X, y, device=DEVICE):
    n = len(X); n_tr = int(n*TRAIN_SPLIT)
    perm = np.random.RandomState(42).permutation(n)
    Xtr, ytr = X[perm[:n_tr]].to(device), y[perm[:n_tr]].to(device)
    Xte, yte = X[perm[n_tr:]].to(device), y[perm[n_tr:]].to(device)
    tl = DataLoader(TensorDataset(Xtr, ytr), 64, True)
    vl = DataLoader(TensorDataset(Xte, yte), 128)
    model = CNNBiLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    huber = nn.HuberLoss(delta=0.5)
    best_vl, best_st, patience_c = float("inf"), None, 0
    tlosses, vlosses = [], []
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tl: opt.zero_grad(); huber(model(xb), yb).backward(); opt.step()
        model.eval()
        tr_l = sum(huber(model(xb), yb).item() for xb, yb in tl)/len(tl)
        v_l = sum(huber(model(xb), yb).item() for xb, yb in vl)/len(vl)
        tlosses.append(tr_l); vlosses.append(v_l)
        if v_l < best_vl: best_vl=v_l; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; patience_c=0
        else: patience_c+=1
        if patience_c>=PATIENCE: break
    model.load_state_dict(best_st); model.eval()
    with torch.no_grad(): yp = model(Xte.to(device)).cpu().numpy()
    yte_np = yte.cpu().numpy()
    mae = float(np.mean(np.abs(yp-yte_np)))
    rmse = float(np.sqrt(np.mean((yp-yte_np)**2)))
    ss_r = np.sum((yte_np-yp)**2); ss_t = np.sum((yte_np-yte_np.mean())**2); r2=1-ss_r/(ss_t+1e-8)
    pe = (yp.max(1)>=0.65).astype(int); te = (yte_np.max(1)>=0.65).astype(int)
    tp,fp,fn=(pe&te).sum(),(pe&(1-te)).sum(),((1-pe)&te).sum()
    p = tp/(tp+fp) if(tp+fp) else 0; r = tp/(tp+fn) if(tp+fn) else 0
    return {"mae":mae,"rmse":rmse,"r2":r2,"esc_f1":2*p*r/(p+r) if(p+r)else 0,
            "train_losses":tlosses,"val_losses":vlosses,"y_true":yte_np,"y_pred":yp,"best_epoch":len(tlosses)-patience_c}

def train_baselines_simple(X, y):
    n=len(X); n_tr=int(n*TRAIN_SPLIT)
    perm=np.random.RandomState(42).permutation(n)
    Xtr,ytr=X[perm[:n_tr]],y[perm[:n_tr]]; Xte,yte=X[perm[n_tr:]],y[perm[n_tr:]]
    yte_np=yte.numpy(); Xtr_f=Xtr[:,:,0].numpy(); Xte_f=Xte[:,:,0].numpy()
    # Persistence
    yp = np.tile(Xte[:,-1,0].numpy().reshape(-1,1),(1,H))
    # AR(6)
    from sklearn.linear_model import LinearRegression
    Xar_tr=Xtr[:,-6:,0].numpy(); Xar_te=Xte[:,-6:,0].numpy()
    y_ar=np.stack([LinearRegression().fit(Xar_tr,ytr[:,h].numpy()).predict(Xar_te) for h in range(H)],1)
    # XGBoost
    from xgboost import XGBRegressor
    y_xgb=np.stack([XGBRegressor(n_estimators=100,max_depth=4,learning_rate=0.1,verbosity=0).fit(Xtr_f,ytr[:,h].numpy()).predict(Xte_f) for h in range(H)],1)
    def m(yp_v):
        mae=float(np.mean(np.abs(yp_v-yte_np))); rmse=float(np.sqrt(np.mean((yp_v-yte_np)**2)))
        ss_r=np.sum((yte_np-yp_v)**2); ss_t=np.sum((yte_np-yte_np.mean())**2); r2=1-ss_r/(ss_t+1e-8)
        pe=(yp_v.max(1)>=0.65).astype(int); te=(yte_np.max(1)>=0.65).astype(int)
        tp,fp,fn=(pe&te).sum(),(pe&(1-te)).sum(),((1-pe)&te).sum()
        p=tp/(tp+fp) if(tp+fp)else 0; r=tp/(tp+fn) if(tp+fn)else 0
        return {"mae":mae,"rmse":rmse,"r2":r2,"esc_f1":2*p*r/(p+r) if(p+r)else 0}
    return {"Persistence":m(yp),"AR(6)":m(y_ar),"XGBoost":m(y_xgb)}


# ════════════════════════════════════════════════════════
# C1: COMPONENT ABLATION
# ════════════════════════════════════════════════════════

def run_c1_component_ablation():
    """C1: Real-data component ablation. Remove each component, recompute c_bar,
    measure autocorrelation and LSTM forecast quality."""
    print("\n"+"="*60)
    print("C1: Component Ablation on Real Data")
    print("="*60)

    # Build base trajectory (same as C2 will)
    base = "zhihu_topics"

    # Load data
    ranked = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isdir(path) or name == "zhihu": continue
        jd = os.path.join(path, "zhihu", "jsonl")
        if not os.path.isdir(jd): continue
        nc = sum(sum(1 for _ in open(f, encoding="utf-8"))
                 for f in glob.glob(os.path.join(jd, "search_comments_*.jsonl")))
        if nc > 0: ranked.append((path, nc))
    ranked.sort(key=lambda x: x[1], reverse=True)

    records = []
    for tp, _ in tqdm(ranked[:20], desc="Loading"):
        tp_name = os.path.basename(tp); jd = os.path.join(tp, "zhihu", "jsonl")
        for ftype, tkey in [("search_comments","content"),("search_contents","content_text")]:
            for fp in glob.glob(os.path.join(jd, f"{ftype}_*.jsonl")):
                for line in open(fp, encoding="utf-8"):
                    try: obj=json.loads(line.strip())
                    except: continue
                    text=obj.get(tkey,"").strip(); ts=obj.get("publish_time") or obj.get("created_time")
                    if text and len(text)>=5 and ts and float(ts)>1704067200:
                        records.append({"text":text,"ts":float(ts),"topic":tp_name})

    import pandas as pd
    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["ts"], unit="s")
    df["bin"] = df["datetime"].dt.floor("12h")
    print(f"Loaded {len(df)} records")

    # Compute raw scores (attack_raw, emotion_raw, stance_raw per comment)
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
    from sklearn.preprocessing import QuantileTransformer

    ATTACK_KW = {"人身攻击","辱骂","威胁","垃圾","去死","废物","傻逼","脑残","恶心","无耻","滚","有病","疯子","不要脸","死了","滚蛋"}
    ANGER_KW = {"气愤","愤怒","离谱","不可理喻","令人发指","荒唐","太过分","无法忍受","气死","怒了","受不了","恶心死了","糊弄","欺负","压榨","剥削","不公平","歧视","抗议"}
    NEG_KW = {"太差","反对","不同意","糟糕","不靠谱","有问题","不合理","不好","差评","错了","不对","不应该","不行","拒绝","失败","失望","不安","担心","焦虑","害怕","恐惧"}

    emb_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)
    all_texts = df["text"].tolist()
    print(f"  Embedding {len(all_texts)} texts...")
    embs = emb_model.encode(all_texts, show_progress_bar=True, batch_size=512, convert_to_numpy=True, normalize_embeddings=True)

    # Weak classifiers
    s_n = min(len(all_texts), 2000); s_idx = np.random.RandomState(42).choice(len(all_texts), s_n, replace=False)
    def label_attack(t): return int(any(k in t.lower() for k in ATTACK_KW))
    def label_emotion(t): return int(any(k in t.lower() for k in ANGER_KW) or (any(k in t.lower() for k in NEG_KW) and any(k in t.lower() for k in ATTACK_KW)))
    al = np.array([label_attack(all_texts[i]) for i in s_idx])
    el = np.array([label_emotion(all_texts[i]) for i in s_idx])
    atk_clf = LogisticRegression(max_iter=1000, C=0.1).fit(embs[s_idx], al)
    emo_clf = LogisticRegression(max_iter=1000, C=0.1).fit(embs[s_idx], el)
    a_raw = atk_clf.predict_proba(embs)[:,1]
    e_raw = emo_clf.predict_proba(embs)[:,1]

    # Stance per topic+bin
    s_raw = np.zeros(len(df))
    for topic in tqdm(df["topic"].unique(), desc="  Stance"):
        mask = df["topic"]==topic; idxs = np.where(mask.values)[0]
        if len(idxs) < 4: continue
        topic_embs = embs[idxs]
        km = KMeans(n_clusters=2, n_init=5, random_state=42); km.fit(topic_embs)
        c0, c1 = km.cluster_centers_
        delta = 1.0 - cosine_similarity(c0.reshape(1,-1), c1.reshape(1,-1))[0,0]
        d0 = cosine_distances(topic_embs, c0.reshape(1,-1)).flatten()
        d1 = cosine_distances(topic_embs, c1.reshape(1,-1)).flatten()
        s_raw[idxs] = min(1.0, delta) * np.abs(d0-d1) / (d0+d1+1e-8)

    # ECDF calibrate each component
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="uniform", random_state=42)
    a_cal = qt.fit_transform(a_raw.reshape(-1,1)).flatten()
    e_cal = qt.fit_transform(e_raw.reshape(-1,1)).flatten()
    s_cal = qt.fit_transform(s_raw.reshape(-1,1)).flatten()

    # Per-comment c values for different weight configs
    configs = {
        "Full (A+E+S)": (0.5, 0.3, 0.2),
        "- Attack":      (0.0, 0.6, 0.4),
        "- Emotion":     (0.625, 0.0, 0.375),
        "- Stance":      (0.625, 0.375, 0.0),
        "Attack only":   (1.0, 0.0, 0.0),
    }

    results_c1 = {}
    for name, (wa, we, ws) in tqdm(configs.items(), desc="C1 ablation"):
        tw = wa + we + ws
        if tw > 0: wa, we, ws = wa/tw, we/tw, ws/tw
        c_vals = (1.0/(1.0+np.exp(-(wa*a_cal + we*e_cal + ws*s_cal)))).clip(0,1)
        df["c"] = c_vals

        # Build trajectory per topic, find best ones
        topic_windows = []
        for topic in df["topic"].unique():
            tdf = df[df["topic"]==topic].sort_values("bin")
            agg = tdf.groupby("bin").agg(
                c_bar=("c", lambda x: x.nlargest(max(1,int(len(x)*0.15))).mean()),
                n=("c","count")).reset_index()
            agg = agg[(agg["n"]>=3) & (~agg["c_bar"].isna())]
            if len(agg) < L+H+10: continue
            vals = agg["c_bar"].values
            ac1 = np.corrcoef(vals[:-1], vals[1:])[0,1] if len(vals)>2 else 0
            if ac1 > 0.05:  # even weak autocorrelation
                mu, std = vals.mean(), vals.std()
                if std < 1e-6: std=1.0
                vn = (vals-mu)/std
                for i in range(len(vn)-L-H+1):
                    topic_windows.append((vn[i:i+L], vn[i+L:i+L+H]))

        if len(topic_windows) < 30:
            print(f"  {name:20s}: only {len(topic_windows)} windows, skipping")
            results_c1[name] = {"r2": -999, "mae": 999, "esc_f1": 0, "n_windows": len(topic_windows)}
            continue

        Xr = torch.tensor(np.array([w[0] for w in topic_windows]), dtype=torch.float32).unsqueeze(-1)
        yr = torch.tensor(np.array([w[1] for w in topic_windows]), dtype=torch.float32)
        r = train_model(Xr, yr)
        results_c1[name] = {"r2": r["r2"], "mae": r["mae"], "esc_f1": r["esc_f1"],
                            "n_windows": len(topic_windows)}
        print(f"  {name:20s} n={len(topic_windows):4d} R²={r['r2']:.4f}")

    # ── Save ──
    with open(f"{R_DIR}/c1_ablation.pkl", "wb") as f:
        pickle.dump(results_c1, f)

    # ── Figure ──
    valid = {k:v for k,v in results_c1.items() if v["r2"] > -999}
    if valid:
        fig, ax = plt.subplots(figsize=(IEEE_COL, 2.5))
        names = list(valid.keys())
        r2s = [valid[n]["r2"] for n in names]
        order = np.argsort(r2s)
        colors = ['#e74c3c' if 'Full' in names[i] else '#3498db' for i in order]
        bars = ax.barh([names[i] for i in order], [r2s[i] for i in order], color=colors, height=0.5)
        for bar, val in zip(bars, [r2s[i] for i in order]):
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=8)
        ax.set_xlabel('R²'); ax.set_title('Component Ablation on Real Data')
        ax.grid(alpha=0.3, axis='x')
        fig.tight_layout()
        fig.savefig(f"{R_DIR}/fig_component_ablation.pdf")
        plt.close(fig)
        print("  Saved fig_component_ablation.pdf\n")

    return results_c1


# ════════════════════════════════════════════════════════
# C2: REAL-DATA CONFLICT INDEX FORECASTING
# ════════════════════════════════════════════════════════

def run_c2_real_conflict_forecast():
    print("="*60)
    print("C2: Real-Data Conflict Index Forecasting")
    print("="*60)

    # Load real data with conflict index (from previous experiment code)
    base = "zhihu_topics"
    # Re-implement data loading inline (no dependency on experiment.py)
    import pandas as pd

    print("Loading real data...")
    # Find top 20 topics
    ranked = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isdir(path) or name == "zhihu": continue
        jd = os.path.join(path, "zhihu", "jsonl")
        if not os.path.isdir(jd): continue
        nc = sum(sum(1 for _ in open(f, encoding="utf-8"))
                 for f in glob.glob(os.path.join(jd, "search_comments_*.jsonl")))
        if nc > 0: ranked.append((path, nc))
    ranked.sort(key=lambda x: x[1], reverse=True)

    records = []
    for tp, _ in tqdm(ranked[:20], desc="Loading"):
        tp_name = os.path.basename(tp); jd = os.path.join(tp, "zhihu", "jsonl")
        for ftype, tkey in [("search_comments","content"),("search_contents","content_text")]:
            for fp in glob.glob(os.path.join(jd, f"{ftype}_*.jsonl")):
                for line in open(fp, encoding="utf-8"):
                    try: obj=json.loads(line.strip())
                    except: continue
                    text=obj.get(tkey,"").strip(); ts=obj.get("publish_time") or obj.get("created_time")
                    if text and len(text)>=5 and ts and float(ts)>1704067200:
                        records.append({"text":text,"ts":float(ts),"topic":tp_name})

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["ts"], unit="s")
    print(f"Loaded {len(df)} recent records from 20 topics")

    # Compute conflict index using embedding-based weak classifiers
    print("Computing conflict index...")
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

    ATTACK_KW = {"人身攻击","辱骂","威胁","垃圾","去死","废物","傻逼","脑残","恶心","无耻","滚","有病","疯子","不要脸","死了","滚蛋"}
    ANGER_KW = {"气愤","愤怒","离谱","不可理喻","令人发指","荒唐","太过分","无法忍受","气死","怒了","受不了","恶心死了","糊弄","欺负","压榨","剥削","不公平","歧视","抗议"}
    NEG_KW = {"太差","反对","不同意","糟糕","不靠谱","有问题","不合理","不好","差评","错了","不对","不应该","不行","拒绝","失败","失望","不安","担心","焦虑","害怕","恐惧"}

    emb_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)
    all_texts = df["text"].tolist()
    print(f"  Embedding {len(all_texts)} texts...")
    embs = emb_model.encode(all_texts, show_progress_bar=True, batch_size=512, convert_to_numpy=True, normalize_embeddings=True)

    # Weak classifiers on heuristic subset
    s_n = min(len(all_texts), 2000)
    s_idx = np.random.RandomState(42).choice(len(all_texts), s_n, replace=False)
    def label_attack(t): return int(any(k in t.lower() for k in ATTACK_KW))
    def label_emotion(t): return int(any(k in t.lower() for k in ANGER_KW) or (any(k in t.lower() for k in NEG_KW) and any(k in t.lower() for k in ATTACK_KW)))
    al = np.array([label_attack(all_texts[i]) for i in s_idx])
    el = np.array([label_emotion(all_texts[i]) for i in s_idx])
    atk_clf = LogisticRegression(max_iter=1000, C=0.1).fit(embs[s_idx], al)
    emo_clf = LogisticRegression(max_iter=1000, C=0.1).fit(embs[s_idx], el)
    a_cal = atk_clf.predict_proba(embs)[:,1]
    e_cal = emo_clf.predict_proba(embs)[:,1]

    # Stance: within-bin clustering
    print("  Computing stance polarization...")
    s_raw = np.zeros(len(df))
    for topic in tqdm(df["topic"].unique(), desc="  Stance"):
        mask = df["topic"]==topic; idxs = np.where(mask.values)[0]
        if len(idxs) < 4: continue
        topic_embs = embs[idxs]
        km = KMeans(n_clusters=2, n_init=5, random_state=42); km.fit(topic_embs)
        c0, c1 = km.cluster_centers_
        delta = 1.0 - cosine_similarity(c0.reshape(1,-1), c1.reshape(1,-1))[0,0]
        d0 = cosine_distances(topic_embs, c0.reshape(1,-1)).flatten()
        d1 = cosine_distances(topic_embs, c1.reshape(1,-1)).flatten()
        s_raw[idxs] = min(1.0, delta) * np.abs(d0-d1) / (d0+d1+1e-8)

    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="uniform", random_state=42)
    a_cal = qt.fit_transform(a_cal.reshape(-1,1)).flatten()
    e_cal = qt.fit_transform(e_cal.reshape(-1,1)).flatten()
    s_cal = qt.fit_transform(s_raw.reshape(-1,1)).flatten()
    c = (1.0/(1.0+np.exp(-(0.5*a_cal+0.3*e_cal+0.2*s_cal)))).clip(0,1)

    df["c"]=c; df["bin"]=df["datetime"].dt.floor("12h")

    # Build per-topic trajectories and find ones with ac1>0.15
    usable_topics = []
    for topic in df["topic"].unique():
        tdf = df[df["topic"]==topic].sort_values("bin")
        agg = tdf.groupby("bin").agg(
            c_bar=("c", lambda x: x.nlargest(max(1,int(len(x)*0.15))).mean()),
            n=("c","count")
        ).reset_index()
        agg = agg[(agg["n"]>=3) & (~agg["c_bar"].isna())]
        if len(agg) < L+H+10: continue
        vals = agg["c_bar"].values
        ac1 = np.corrcoef(vals[:-1], vals[1:])[0,1] if len(vals)>2 else 0
        if ac1 > 0.1:
            usable_topics.append((topic, agg["c_bar"].values, ac1))

    print(f"Found {len(usable_topics)} topics with ac1>0.1")
    for t, v, a in usable_topics[:5]:
        print(f"  {t[:40]:40s} n={len(v):3d} ac1={a:+.3f}")

    if not usable_topics:
        print("  No topics with sufficient temporal structure! Using global aggregate instead.")
        # Fall back: aggregate all topics to global conflict level
        global_agg = df.groupby("bin").agg(
            c_bar=("c", lambda x: x.nlargest(max(1,int(len(x)*0.15))).mean()),
            n=("c","count")
        ).reset_index()
        global_agg = global_agg[(global_agg["n"]>=5) & (~global_agg["c_bar"].isna())]
        vals = global_agg["c_bar"].values
        ac1 = np.corrcoef(vals[:-1], vals[1:])[0,1]
        print(f"  Global aggregate: n={len(vals)} ac1={ac1:+.3f}")
        usable_topics = [("Global", vals, ac1)]

    # Train on all usable topics pooled
    all_X, all_y = [], []
    for tname, vals, ac1 in usable_topics:
        # Normalize per-topic
        mu, std = vals.mean(), vals.std()
        if std < 1e-6: std = 1.0
        vn = (vals-mu)/std
        for i in range(len(vn)-L-H+1):
            all_X.append(vn[i:i+L]); all_y.append(vn[i+L:i+L+H])
    if len(all_X) < 50:
        print(f"  Too few windows ({len(all_X)}), skipping C2")
        return None

    Xr = torch.tensor(np.array(all_X), dtype=torch.float32).unsqueeze(-1)
    yr = torch.tensor(np.array(all_y), dtype=torch.float32)
    print(f"  {len(all_X)} windows for real-data conflict forecast")

    r_c2 = train_model(Xr, yr)
    bl_c2 = train_baselines_simple(Xr, yr)

    print(f"\n  Real-Data Conflict Index Forecasting:")
    for name in ["Persistence","AR(6)","XGBoost","CNN-BiLSTM"]:
        d = bl_c2.get(name) or r_c2
        if name == "CNN-BiLSTM": d = r_c2
        print(f"    {name:15s} R²={d['r2']:.4f} MAE={d['mae']:.4f} Esc-F1={d['esc_f1']:.3f}")

    # ── Figure ──
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_WIDE, 2.8))
    # (a) Bar comparison
    ax = axes[0]
    model_names = ["Persistence","AR(6)","XGBoost","CNN-BiLSTM"]
    r2_vals = [bl_c2[n]["r2"] for n in model_names[:3]] + [r_c2["r2"]]
    cols = ['#95a5a6','#95a5a6','#95a5a6','#e74c3c']
    bars = ax.bar(model_names, r2_vals, color=cols, width=0.5)
    for bar, v in zip(bars, r2_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f'{v:.3f}', ha='center', fontsize=8)
    ax.set_ylabel('R²'); ax.set_title('(a) Conflict-Index Forecast (Real Data)')
    ax.axhline(y=0, color='black', lw=0.5)
    ax.grid(alpha=0.3, axis='y')

    # (b) Sample trajectory
    ax = axes[1]
    best_topic = usable_topics[0]
    vals = best_topic[1]
    ax.plot(vals, color='#2c3e50', lw=0.8)
    ax.axhline(y=np.median(vals), color='#e74c3c', ls='--', lw=0.8, alpha=0.5, label='Median')
    ax.set_xlabel('Time (bins)'); ax.set_ylabel('Conflict Index'); ax.set_title(f'(b) Sample: {best_topic[0][:20]}... (ac1={best_topic[2]:+.2f})')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle('Real-Data Conflict Index Forecasting', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{R_DIR}/fig_real_conflict_forecast.pdf")
    plt.close(fig)
    print("  Saved fig_real_conflict_forecast.pdf")

    # Save
    with open(f"{R_DIR}/c2_real_conflict.pkl", "wb") as f:
        pickle.dump({"results": r_c2, "baselines": bl_c2, "topics": usable_topics}, f)

    return r_c2, bl_c2


# ════════════════════════════════════════════════════════
# C3: LEAD TIME ANALYSIS
# ════════════════════════════════════════════════════════

def run_c3_lead_time():
    print("\n"+"="*60)
    print("C3: Lead Time Analysis for Early Warning")
    print("="*60)

    # Generate synthetic data with known escalation timing
    X, y, _ = make_synthetic_windows({'a':1.0,'e':1.0,'s':1.0})
    n=len(X); n_tr=int(n*TRAIN_SPLIT)
    perm=np.random.RandomState(42).permutation(n)
    Xtr,ytr=X[perm[:n_tr]].to(DEVICE),y[perm[:n_tr]].to(DEVICE)
    Xte,yte=X[perm[n_tr:]].to(DEVICE),y[perm[n_tr:]].to(DEVICE)
    yte_np=yte.cpu().numpy()

    # Train model
    model=CNNBiLSTM().to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LR); huber=nn.HuberLoss(delta=0.5)
    tl=DataLoader(TensorDataset(Xtr,ytr),64,True); vl=DataLoader(TensorDataset(Xte,yte),128)
    best_vl,best_st,patience_c=float("inf"),None,0
    for ep in range(EPOCHS):
        model.train()
        for xb,yb in tl: opt.zero_grad(); huber(model(xb),yb).backward(); opt.step()
        model.eval()
        v_l=sum(huber(model(xb),yb).item() for xb,yb in vl)/len(vl)
        if v_l<best_vl: best_vl=v_l; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; patience_c=0
        else: patience_c+=1
        if patience_c>=PATIENCE: break
    model.load_state_dict(best_st); model.eval()
    with torch.no_grad(): y_pred=model(Xte.to(DEVICE)).cpu().numpy()

    # Compute lead time: for each window where GT escalation occurs,
    # how many bins BEFORE the actual peak does the prediction exceed threshold?
    eta = 0.65
    lead_times = []
    correct_leads = []  # only for TP predictions
    missed_leads = []   # for FN predictions

    for i in range(len(yte_np)):
        gt_peak_idx = np.argmax(yte_np[i])
        gt_peak_val = yte_np[i, gt_peak_idx]
        pred_peak_val = y_pred[i].max()

        if gt_peak_val >= eta:  # GT has escalation
            if pred_peak_val >= eta:  # TP: predicted correctly
                pred_peak_idx = np.argmax(y_pred[i])
                lead = gt_peak_idx - pred_peak_idx  # positive = early, negative = late
                lead_times.append(lead)
                correct_leads.append(lead)
            else:  # FN: missed
                lead_times.append(-1)  # marker for miss
                missed_leads.append(-1)

    # Generate individual trajectory lead time analysis
    # For finer analysis, generate a single long trajectory
    long_vals, long_esc = gen_traj(300, seed=99)
    # Slide through and compute pred vs actual peak timing
    horizon_leads = {h: [] for h in range(H)}

    print(f"  Test windows: {len(yte_np)}")
    print(f"  Escalation prevalence: {(yte_np.max(1)>=eta).mean():.1%}")
    print(f"  TP predictions: {len(correct_leads)}")
    print(f"  FN (missed): {len(missed_leads)}")
    print(f"  Lead time: mean={np.mean(correct_leads):.1f}±{np.std(correct_leads):.1f} bins "
          f"(positive=early, negative=late)")

    # ── Figure ──
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_WIDE, 2.8))

    # (a) Lead time histogram
    ax = axes[0]
    if correct_leads:
        ax.hist(correct_leads, bins=8, color='#3498db', alpha=0.8, edgecolor='white', lw=0.5)
        ax.axvline(x=np.mean(correct_leads), color='#e74c3c', ls='--', lw=1.0, label=f'Mean={np.mean(correct_leads):.1f} bins')
        ax.axvline(x=0, color='black', lw=0.5)
        ax.set_xlabel('Lead Time (bins, + = early)'); ax.set_ylabel('Count')
        ax.set_title('(a) Lead Time Distribution (TP predictions)')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis='y')

    # (b) Example: trajectory with warning markers
    ax = axes[1]
    demo_vals, demo_esc = gen_traj(100, seed=77)
    ax.plot(demo_vals, color='#2c3e50', lw=0.8, label='Conflict Index')
    # Mark escalation regions
    for i in range(len(demo_esc)):
        if demo_esc[i]: ax.axvspan(i-0.5,i+0.5,color='#e74c3c',alpha=0.1)
    # Simulate where triggers would fire
    triggers = []
    for i in range(len(demo_vals)-H):
        if demo_vals[i:i+H].max() >= 0.65:
            triggers.append(i)
    if triggers:
        t_sample = triggers[::max(1,len(triggers)//6)]
        ax.scatter(t_sample, [demo_vals[t] for t in t_sample],
                   color='#e74c3c', marker='^', s=30, zorder=5, edgecolors='white', lw=0.5,
                   label=f'Warning ({len(triggers)} triggers)')
    ax.axhline(y=0.65, color='#e74c3c', ls='--', lw=0.8, alpha=0.5, label=r'$\eta=0.65$')
    ax.set_xlabel('Time (bins)'); ax.set_ylabel('Conflict Index')
    ax.set_title('(b) Early Warning Trigger Demo')
    ax.legend(loc='upper left', fontsize=7, ncol=2); ax.grid(alpha=0.3)

    fig.suptitle('Early Warning Lead Time Analysis', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{R_DIR}/fig_lead_time.pdf")
    plt.close(fig)
    print("  Saved fig_lead_time.pdf")

    # Save
    with open(f"{R_DIR}/c3_lead_time.pkl", "wb") as f:
        pickle.dump({"lead_times": lead_times, "correct_leads": correct_leads,
                     "mean_lead": np.mean(correct_leads) if correct_leads else 0,
                     "std_lead": np.std(correct_leads) if correct_leads else 0}, f)

    return lead_times, correct_leads


# ═══ MAIN ═══
if __name__ == "__main__":
    t0 = time.time()

    c1 = run_c1_component_ablation()
    notify(f"C1 done: component ablation ({time.time()-t0:.0f}s)")

    c2 = run_c2_real_conflict_forecast()
    notify(f"C2 done: real-data conflict forecast ({time.time()-t0:.0f}s)")

    c3 = run_c3_lead_time()
    notify(f"C3 done: lead time analysis ({time.time()-t0:.0f}s)")

    elapsed = (time.time()-t0)/60
    print(f"\n{'='*60}")
    print(f"All done in {elapsed:.1f} min")
    print(f"Figures: {R_DIR}/fig_component_ablation.pdf")
    print(f"         {R_DIR}/fig_real_conflict_forecast.pdf")
    print(f"         {R_DIR}/fig_lead_time.pdf")
    notify(f"All 3 experiments done! ({elapsed:.1f}min)")
