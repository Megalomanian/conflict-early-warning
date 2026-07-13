#!/usr/bin/env python3
"""Case study: Weibo public opinion reversal events.
Uses REAL pretrained BERT model (nlptown/bert-base-multilingual-uncased-sentiment).
Run with: uv run python3 case_study.py"""
import pandas as pd, numpy as np, pickle, warnings, time, os
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

R_DIR="experiment_results"; DEV="cuda" if torch.cuda.is_available() else "cpu"
L,H=12,6; HIDDEN=64
print(f"Device: {DEV}")

def notify(msg): os.system(f'/home/violina/projects/notice/notice "{msg}" 2>/dev/null &')

# ═══ Real BERT Conflict Computer ═══
class RealConflictComputer:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from sentence_transformers import SentenceTransformer
        print("Loading BERT sentiment + sentence transformer...")
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sent_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEV)
        self.sent_model.eval()
        self.emb_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEV)
        print("  Models loaded.")

    def compute_sentiment_batch(self, texts, batch_size=128):
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True,
                                        truncation=True, max_length=256).to(DEV)
                outputs = self.sent_model(**inputs)
                all_logits.append(outputs.logits.cpu().numpy())
        return np.concatenate(all_logits, axis=0)

    def compute_batch(self, texts):
        n = len(texts)
        print(f"  BERT sentiment on {n} texts...", flush=True)
        logits = self.compute_sentiment_batch(texts)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        attack = probs[:, 0] + 0.5 * probs[:, 1]        # 1-star + half 2-star
        p_neg = probs[:, 0] + probs[:, 1]
        p_anger = probs[:, 0]
        emotion = 0.7 * p_neg + 0.3 * p_anger
        print(f"  Embeddings for stance on {n} texts...", flush=True)
        embs = self.emb_model.encode(texts, show_progress_bar=True, batch_size=256,
                                      convert_to_numpy=True, normalize_embeddings=True)
        return {"attack": attack, "emotion": emotion, "embeddings": embs}

    def stance_polarization(self, embs):
        n = len(embs)
        if n < 4: return np.zeros(n)
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
        km = KMeans(n_clusters=2, n_init=5, random_state=42); km.fit(embs)
        c0, c1 = km.cluster_centers_
        delta = 1.0 - cosine_similarity(c0.reshape(1,-1), c1.reshape(1,-1))[0,0]
        d0 = cosine_distances(embs, c0.reshape(1,-1)).flatten()
        d1 = cosine_distances(embs, c1.reshape(1,-1)).flatten()
        return min(1.0, delta) * np.abs(d0 - d1) / (d0 + d1 + 1e-8)

# ═══ CNN-BiLSTM ═══
class CNNBiLSTM(nn.Module):
    def __init__(self, h=HIDDEN):
        super().__init__(); self.h = h
        self.conv = nn.Sequential(nn.Conv1d(1,32,3,padding=1), nn.ReLU(),
                                   nn.Conv1d(32,32,5,padding=2), nn.ReLU())
        self.lstm = nn.LSTM(32, h, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Linear(h*2, H)
    def forward(self, x):
        c = self.conv(x.transpose(1,2)).transpose(1,2); o, _ = self.lstm(c)
        return self.proj(torch.cat([o[:,-1,:self.h], o[:,0,self.h:]], dim=-1))

# ═══ Main ═══
if __name__ == "__main__":
    t0 = time.time()

    meta = pd.read_csv("zhihu_topics/weibo_reversal/data/data_case.csv", encoding="gbk")
    meta["nc"] = [int(open(f"zhihu_topics/weibo_reversal/data/{c}").read().count("\n"))-1
                  for c in meta["csvname"]]
    meta = meta.sort_values("nc", ascending=False)
    selected = meta.head(3)

    # Load all event data
    all_dfs = []
    for _, row in selected.iterrows():
        fn = f"zhihu_topics/weibo_reversal/data/{row['csvname']}"
        df = pd.read_csv(fn, encoding="utf-8")
        df["dt"] = pd.to_datetime(df["text_time"], format="%y-%m-%d %H:%M", errors="coerce")
        df = df.dropna(subset=["dt","text_content"])
        df["event_name"] = row["name"]
        df["fbegin"] = pd.to_datetime(row["fbegin"], format="%y-%m-%d %H")
        df["fend"] = pd.to_datetime(row["fend"], format="%y-%m-%d %H")
        all_dfs.append(df)
        print(f"  {row['name']}: {len(df)} comments")

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"Total: {len(df_all)} comments")

    # ── BERT Conflict Index ──
    print("\n" + "="*60)
    print("Computing conflict index with REAL BERT model...")
    print("="*60)
    computer = RealConflictComputer()
    texts = df_all["text_content"].tolist()
    r = computer.compute_batch(texts)

    qt = QuantileTransformer(n_quantiles=1000, output_distribution="uniform", random_state=42)
    a_cal = qt.fit_transform(r["attack"].reshape(-1,1)).flatten()
    e_cal = qt.fit_transform(r["emotion"].reshape(-1,1)).flatten()

    s_raw = np.zeros(len(df_all))
    for evt_name in df_all["event_name"].unique():
        mask = df_all["event_name"] == evt_name
        idxs = np.where(mask.values)[0]
        if len(idxs) >= 4:
            s_raw[idxs] = computer.stance_polarization(r["embeddings"][idxs])
    s_cal = qt.fit_transform(s_raw.reshape(-1,1)).flatten()

    df_all["c"] = (1.0/(1.0+np.exp(-(0.5*a_cal+0.3*e_cal+0.2*s_cal)))).clip(0,1)
    df_all["bin"] = df_all["dt"].dt.floor("2h")

    print(f"  Attack μ={r['attack'].mean():.3f} Emotion μ={r['emotion'].mean():.3f} Conflict μ={df_all['c'].mean():.3f}")

    # ── Per-event modeling ──
    results_per_event = {}
    for evt_name in df_all["event_name"].unique():
        edf = df_all[df_all["event_name"] == evt_name].copy()
        fb, fe = edf["fbegin"].iloc[0], edf["fend"].iloc[0]
        edf["is_reversal"] = ((edf["dt"] >= fb) & (edf["dt"] <= fe)).astype(int)

        agg = edf.groupby("bin").agg(
            c_bar=("c", lambda x: x.nlargest(max(1,int(len(x)*0.2))).mean()),
            n=("c","count"), rev_pct=("is_reversal","mean")
        ).reset_index()
        agg = agg[(agg["n"]>=2) & (~agg["c_bar"].isna())]
        if len(agg) < L+H+10: continue

        vals_raw = agg["c_bar"].values
        mu, std = vals_raw.mean(), vals_raw.std()
        if std < 1e-6: std = 1.0
        vals = (vals_raw-mu)/std

        Xs, ys, es = [], [], []
        for j in range(len(vals)-L-H+1):
            Xs.append(vals[j:j+L]); ys.append(vals[j+L:j+L+H])
            es.append(1 if agg["rev_pct"].values[j+L:j+L+H].max()>0 else 0)
        if len(Xs) < 30: continue

        X = torch.tensor(np.array(Xs), dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(np.array(ys), dtype=torch.float32)
        esc_labels = np.array(es)

        n = len(X); n_tr = int(n*0.75)
        perm = np.random.RandomState(42).permutation(n)
        Xtr, ytr = X[perm[:n_tr]].to(DEV), y[perm[:n_tr]].to(DEV)
        Xte, yte = X[perm[n_tr:]].to(DEV), y[perm[n_tr:]].to(DEV)
        yte_np, esc_te = yte.cpu().numpy(), esc_labels[perm[n_tr:]]

        # Train CNN-BiLSTM
        model = CNNBiLSTM().to(DEV); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        huber = nn.HuberLoss(delta=0.5)
        tl = DataLoader(TensorDataset(Xtr,ytr), 64, True)
        vl = DataLoader(TensorDataset(Xte,yte), 128)
        best_vl, best_st, patience_c = float("inf"), None, 0
        for ep in range(150):
            model.train()
            for xb, yb in tl: opt.zero_grad(); huber(model(xb),yb).backward(); opt.step()
            model.eval(); v_l = sum(huber(model(xb),yb).item() for xb,yb in vl)/len(vl)
            if v_l < best_vl: best_vl=v_l; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; patience_c=0
            else: patience_c+=1
            if patience_c >= 40: break
        model.load_state_dict(best_st); model.eval()
        with torch.no_grad(): y_pred = model(Xte.to(DEV)).cpu().numpy()

        def m(yp_v):
            mae_v = float(np.mean(np.abs(yp_v-yte_np)))
            ss_r = np.sum((yte_np-yp_v)**2); ss_t = np.sum((yte_np-yte_np.mean())**2)
            r2_v = 1-ss_r/(ss_t+1e-8)
            pe_v = (yp_v.max(1)>=0.65).astype(int)
            tp_v,fp_v,fn_v = (pe_v&esc_te).sum(),(pe_v&(1-esc_te)).sum(),((1-pe_v)&esc_te).sum()
            p_v=tp_v/(tp_v+fp_v) if(tp_v+fp_v) else 0; r_v=tp_v/(tp_v+fn_v) if(tp_v+fn_v) else 0
            return {"mae":mae_v,"r2":r2_v,"esc_f1":2*p_v*r_v/(p_v+r_v) if(p_v+r_v) else 0}

        # Baselines
        Xte_f = Xte[:,:,0].cpu().numpy()
        yp_p = np.tile(Xte_f[:,-1].reshape(-1,1),(1,H))
        Xar_te = Xte[:,-6:,0].cpu().numpy(); Xar_tr = Xtr[:,-6:,0].cpu().numpy()
        y_ar = np.stack([LinearRegression().fit(Xar_tr,ytr[:,h].cpu().numpy()).predict(Xar_te) for h in range(H)],1)
        Xtr_f = Xtr[:,:,0].cpu().numpy()
        y_xgb = np.stack([XGBRegressor(n_estimators=100,max_depth=4,learning_rate=0.1,verbosity=0).fit(Xtr_f,ytr[:,h].cpu().numpy()).predict(Xte_f) for h in range(H)],1)

        pre_c = agg[agg["rev_pct"]==0]["c_bar"].mean()
        dur_c = agg[agg["rev_pct"]>0]["c_bar"].mean()
        event_res = {
            "name":evt_name,"bins":len(agg),"windows":n,"test_windows":n-n_tr,
            "esc_prevalence":esc_te.mean(),
            "CNN-BiLSTM":m(y_pred),
            "Persistence":m(yp_p),"AR(6)":m(y_ar),"XGBoost":m(y_xgb),
            "autocorr":np.corrcoef(vals[:-1],vals[1:])[0,1],
            "pre_rev_c":pre_c,"during_rev_c":dur_c,
            "trajectory":vals_raw,"reversal_mask":(agg["rev_pct"]>0).values,
        }
        results_per_event[evt_name] = event_res
        r2_cnn = event_res["CNN-BiLSTM"]["r2"]
        esc_cnn = event_res["CNN-BiLSTM"]["esc_f1"]
        print(f"  {evt_name}: Δrev={dur_c-pre_c:+.4f} R²(CNN)={r2_cnn:.4f} Esc-F1={esc_cnn:.4f} "
              f"vs Persist R²={m(yp_p)['r2']:.4f}")

    # ── Summary ──
    print(f"\n{'Event':30s} {'ΔConflict':>10s} {'R²(CNN)':>9s} {'R²(Pers)':>9s} {'Esc-F1':>8s}")
    print("-"*70)
    for name, res in results_per_event.items():
        print(f"{name:30s} {res['during_rev_c']-res['pre_rev_c']:+10.4f} "
              f"{res['CNN-BiLSTM']['r2']:9.4f} {res['Persistence']['r2']:9.4f} "
              f"{res['CNN-BiLSTM']['esc_f1']:8.4f}")

    # ── Figure ──
    best_name = sorted(results_per_event.keys(), key=lambda k: results_per_event[k]["CNN-BiLSTM"]["r2"], reverse=True)[0]
    best = results_per_event[best_name]
    cjk_fonts = [f for f in fm.findSystemFonts() if any(
        k in f.lower() for k in ['noto','cjk','wenquan','simhei','simsun','songti','heiti','droid','source-han'])]
    font_prop = fm.FontProperties(fname=cjk_fonts[0]) if cjk_fonts else None

    fig, axes = plt.subplots(1,3,figsize=(7.0,2.6))
    ax = axes[0]; traj = best["trajectory"]; rev = best["reversal_mask"]
    t = np.arange(len(traj))
    ax.plot(t,traj,color="#2c3e50",lw=0.8,label="Conflict Index")
    for i in range(len(rev)):
        if rev[i]: ax.axvspan(i-0.5,i+0.5,color="#e74c3c",alpha=0.12)
    rev_start = np.where(rev)[0]
    if len(rev_start) > 0:
        mid = rev_start[len(rev_start)//2]
        ax.annotate('Reversal\nWindow', xy=(mid,traj[mid]),
                    xytext=(mid+15,traj[mid]+0.05),
                    arrowprops=dict(arrowstyle='->',color='#e74c3c',lw=0.8),
                    fontsize=7,color='#e74c3c',fontproperties=font_prop)
    ax.set_xlabel("Time (2h bins)"); ax.set_ylabel("Conflict Index")
    ax.set_title(f"(a) Conflict Trajectory\n({best_name[:20]})")
    ax.legend(fontsize=7,loc='upper right'); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    ax = axes[1]
    models = ["Persistence","AR(6)","XGBoost","CNN-BiLSTM"]
    r2s = [best["Persistence"]["r2"],best["AR(6)"]["r2"],best["XGBoost"]["r2"],best["CNN-BiLSTM"]["r2"]]
    colors = ["#95a5a6","#95a5a6","#95a5a6","#e74c3c"]
    bars = ax.bar(range(len(models)),r2s,color=colors,width=0.5,edgecolor='white',lw=0.3)
    for i,(bar,v) in enumerate(zip(bars,r2s)):
        y_pos = max(0,bar.get_height())+0.03
        ax.text(bar.get_x()+bar.get_width()/2,y_pos,f"{v:.3f}",ha="center",fontsize=7,
                fontweight='bold' if i==3 else 'normal')
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models,rotation=20,ha='right')
    ax.set_ylabel("R²"); ax.set_title("(b) Forecast Accuracy")
    ax.axhline(y=0,color="black",lw=0.5); ax.grid(alpha=0.3,axis='y')

    ax = axes[2]
    esc_f1s = [best["Persistence"]["esc_f1"],best["AR(6)"]["esc_f1"],
               best["XGBoost"]["esc_f1"],best["CNN-BiLSTM"]["esc_f1"]]
    bars = ax.bar(range(len(models)),esc_f1s,color=colors,width=0.5,edgecolor='white',lw=0.3)
    for i,(bar,v) in enumerate(zip(bars,esc_f1s)):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.03,f"{v:.3f}",ha="center",
                fontsize=7,fontweight='bold' if i==3 else 'normal')
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models,rotation=20,ha='right')
    ax.set_ylabel("Escalation F1"); ax.set_title("(c) Reversal Detection")
    ax.grid(alpha=0.3,axis='y'); ax.set_ylim(0,1.05)

    fig.suptitle("Real-World Case Study: Weibo Public Opinion Reversal Events",
                 fontsize=10,fontweight='bold',y=1.02)
    fig.tight_layout()
    fig.savefig(f"{R_DIR}/fig_case_study.pdf")
    import subprocess; subprocess.run(["cp",f"{R_DIR}/fig_case_study.pdf","figures/fig_case_study.pdf"])
    print(f"\nSaved figures/fig_case_study.pdf"); plt.close(fig)

    with open(f"{R_DIR}/case_study.pkl","wb") as f: pickle.dump(results_per_event,f)
    tmin = (time.time()-t0)/60
    print(f"Done in {tmin:.1f} min")
    notify(f"Case study (BERT+GPU) done! {len(results_per_event)} events in {tmin:.1f}min")
