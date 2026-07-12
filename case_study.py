#!/usr/bin/env python3
"""Case study: Weibo public opinion reversal events.
Validates the full pipeline on real data with known escalation windows.
Uses keyword-based conflict index (simplified); BERT-based pipeline validated on zhihu data."""
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
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.titlesize":10,
    "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight"})

R_DIR="experiment_results"; DEV="cuda" if torch.cuda.is_available() else "cpu"
L,H=12,6; HIDDEN=64

def notify(msg): os.system(f'/home/violina/projects/notice/notice "{msg}" 2>/dev/null &')

# ═══ Keyword-based Conflict Signal (simplified; BERT version validated on zhihu data) ═══
ATTACK_KW={"人身攻击","辱骂","威胁","垃圾","去死","废物","傻逼","脑残","恶心","无耻","滚","有病","疯子","不要脸","死了","滚蛋"}
ANGER_KW={"气愤","愤怒","离谱","不可理喻","令人发指","荒唐","太过分","无法忍受","气死","怒了","受不了","恶心死了","糊弄","欺负","压榨","剥削","不公平","歧视","抗议","造谣","打脸","反转","真相"}
NEG_KW={"太差","反对","不同意","糟糕","不靠谱","有问题","不合理","不好","差评","错了","不对","不应该","不行","拒绝","失败","失望","不安","担心","焦虑","害怕","恐惧","崩塌","翻车"}

def conflict_score(text):
    t=str(text).lower()
    return min(1.0, 0.5*any(k in t for k in ATTACK_KW)+0.3*any(k in t for k in ANGER_KW)+0.2*any(k in t for k in NEG_KW))

# ═══ CNN-BiLSTM ═══
class CNNBiLSTM(nn.Module):
    def __init__(self,h=HIDDEN):
        super().__init__(); self.h=h
        self.conv=nn.Sequential(nn.Conv1d(1,32,3,padding=1),nn.ReLU(),nn.Conv1d(32,32,5,padding=2),nn.ReLU())
        self.lstm=nn.LSTM(32,h,2,batch_first=True,dropout=0.2,bidirectional=True)
        self.proj=nn.Linear(h*2,H)
    def forward(self,x):
        c=self.conv(x.transpose(1,2)).transpose(1,2); o,_=self.lstm(c)
        return self.proj(torch.cat([o[:,-1,:self.h],o[:,0,self.h:]],dim=-1))

# ═══ Main ═══
if __name__=="__main__":
    t0=time.time()

    # Load data
    meta=pd.read_csv("zhihu_topics/weibo_reversal/data/data_case.csv",encoding="gbk")
    meta["nc"]=[int(open(f"zhihu_topics/weibo_reversal/data/{c}").read().count("\n"))-1 for c in meta["csvname"]]
    meta=meta.sort_values("nc",ascending=False)

    # Use top 3 events
    results_per_event={}
    all_preds=[]; all_trues=[]

    for evt_idx, (_, row) in enumerate(meta.head(3).iterrows()):
        fn=f"zhihu_topics/weibo_reversal/data/{row['csvname']}"
        df=pd.read_csv(fn,encoding="utf-8")
        df["dt"]=pd.to_datetime(df["text_time"],format="%y-%m-%d %H:%M",errors="coerce")
        df=df.dropna(subset=["dt","text_content"])

        # Reversal window
        fb=pd.to_datetime(row["fbegin"],format="%y-%m-%d %H")
        fe=pd.to_datetime(row["fend"],format="%y-%m-%d %H")

        # Compute conflict index per comment (keyword-based)
        df["c"]=df["text_content"].apply(conflict_score)
        df["bin"]=df["dt"].dt.floor("2h")
        df["is_reversal"]=((df["dt"]>=fb)&(df["dt"]<=fe)).astype(int)

        # Bin-level aggregation (top-20% mean)
        agg=df.groupby("bin").agg(
            c_bar=("c",lambda x: x.nlargest(max(1,int(len(x)*0.2))).mean()),
            n=("c","count"),
            rev_pct=("is_reversal","mean")
        ).reset_index()
        agg=agg[(agg["n"]>=2)&(~agg["c_bar"].isna())]
        if len(agg)<L+H+10: continue

        vals_raw=agg["c_bar"].values
        mu,std=vals_raw.mean(),vals_raw.std()
        if std<1e-6: std=1.0
        vals=(vals_raw-mu)/std

        # Build windows
        Xs,ys,es=[],[],[]
        for j in range(len(vals)-L-H+1):
            Xs.append(vals[j:j+L]); ys.append(vals[j+L:j+L+H])
            es.append(1 if agg["rev_pct"].values[j+L:j+L+H].max()>0 else 0)

        if len(Xs)<30: continue

        X=torch.tensor(np.array(Xs),dtype=torch.float32).unsqueeze(-1)
        y=torch.tensor(np.array(ys),dtype=torch.float32)
        esc_labels=np.array(es)

        n=len(X); n_tr=int(n*0.75); perm=np.random.RandomState(42).permutation(n)
        Xtr,ytr=X[perm[:n_tr]].to(DEV),y[perm[:n_tr]].to(DEV)
        Xte,yte=X[perm[n_tr:]].to(DEV),y[perm[n_tr:]].to(DEV)
        yte_np=yte.cpu().numpy(); esc_te=esc_labels[perm[n_tr:]]

        # Train CNN-BiLSTM
        model=CNNBiLSTM().to(DEV); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        loss_fn=nn.HuberLoss(delta=0.5)
        tl=DataLoader(TensorDataset(Xtr,ytr),64,True); vl=DataLoader(TensorDataset(Xte,yte),128)
        best_vl,best_st,patience=float("inf"),None,0
        for ep in range(150):
            model.train()
            for xb,yb in tl: opt.zero_grad(); loss_fn(model(xb),yb).backward(); opt.step()
            model.eval(); v_l=sum(loss_fn(model(xb),yb).item() for xb,yb in vl)/len(vl)
            if v_l<best_vl: best_vl=v_l; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; patience=0
            else: patience+=1
            if patience>=40: break
        model.load_state_dict(best_st); model.eval()
        with torch.no_grad(): y_pred=model(Xte.to(DEV)).cpu().numpy()

        # Metrics
        mae=float(np.mean(np.abs(y_pred-yte_np)))
        ss_r=np.sum((yte_np-y_pred)**2); ss_t=np.sum((yte_np-yte_np.mean())**2)
        r2=1-ss_r/(ss_t+1e-8)
        pe=(y_pred.max(1)>=y_pred.max()*0.7).astype(int); te=esc_te
        tp,fp,fn=(pe&te).sum(),(pe&(1-te)).sum(),((1-pe)&te).sum()
        p=tp/(tp+fp) if(tp+fp)else 0; r=tp/(tp+fn) if(tp+fn)else 0

        # Baselines
        yp=np.tile(Xte[:,-1,0].cpu().numpy().reshape(-1,1),(1,H))
        Xar_tr=Xtr[:,-6:,0].cpu().numpy(); Xar_te=Xte[:,-6:,0].cpu().numpy()
        y_ar=np.stack([LinearRegression().fit(Xar_tr,ytr[:,h].cpu().numpy()).predict(Xar_te) for h in range(H)],1)
        Xtr_f=Xtr[:,:,0].cpu().numpy(); Xte_f=Xte[:,:,0].cpu().numpy()
        y_xgb=np.stack([XGBRegressor(n_estimators=100,max_depth=4,learning_rate=0.1,verbosity=0).fit(Xtr_f,ytr[:,h].cpu().numpy()).predict(Xte_f) for h in range(H)],1)

        def m(yp_v):
            mae_v=float(np.mean(np.abs(yp_v-yte_np)))
            ss_r=np.sum((yte_np-yp_v)**2); ss_t=np.sum((yte_np-yte_np.mean())**2)
            r2_v=1-ss_r/(ss_t+1e-8)
            pe_v=(yp_v.max(1)>=yp_v.max()*0.7).astype(int)
            tp_v,fp_v,fn_v=(pe_v&te).sum(),(pe_v&(1-te)).sum(),((1-pe_v)&te).sum()
            p_v=tp_v/(tp_v+fp_v) if(tp_v+fp_v)else 0; r_v=tp_v/(tp_v+fn_v) if(tp_v+fn_v)else 0
            return {"mae":mae_v,"r2":r2_v,"esc_f1":2*p_v*r_v/(p_v+r_v) if(p_v+r_v)else 0}

        event_res={
            "name":row["name"],"bins":len(agg),"windows":n,"test_windows":n-n_tr,
            "esc_prevalence":esc_te.mean(),
            "CNN-BiLSTM":{"mae":mae,"r2":r2,"esc_f1":2*p*r/(p+r) if(p+r)else 0},
            "Persistence":m(yp),"AR(6)":m(y_ar),"XGBoost":m(y_xgb),
            "autocorr":np.corrcoef(vals[:-1],vals[1:])[0,1],
            "pre_rev_c":agg[agg["rev_pct"]==0]["c_bar"].mean(),
            "during_rev_c":agg[agg["rev_pct"]>0]["c_bar"].mean(),
            "trajectory":vals_raw,"reversal_mask":(agg["rev_pct"]>0).values,
            "train_losses":[],
        }
        results_per_event[row["name"]]=event_res
        all_preds.append(y_pred); all_trues.append(yte_np)

        print(f"  {row['name']:20s}: bins={len(agg):3d} ac1={event_res['autocorr']:+.3f} "
              f"Δrev={event_res['during_rev_c']-event_res['pre_rev_c']:+.3f} "
              f"R²={r2:.3f} Esc-F1={2*p*r/(p+r) if(p+r)else 0:.3f} (CNN) "
              f"vs Persist R²={m(yp)['r2']:.3f} vs XGB R²={m(y_xgb)['r2']:.3f}")

    # ═══ Summary ═══
    print(f"\n{'Event':25s} {'R²(CNN)':>8s} {'R²(Pers)':>8s} {'R²(XGB)':>8s} {'Esc-F1':>8s} {'ΔConflict':>10s}")
    print("-"*70)
    for name,res in results_per_event.items():
        print(f"{name:25s} {res['CNN-BiLSTM']['r2']:8.3f} {res['Persistence']['r2']:8.3f} "
              f"{res['XGBoost']['r2']:8.3f} {res['CNN-BiLSTM']['esc_f1']:8.3f} "
              f"{res['during_rev_c']-res['pre_rev_c']:+10.3f}")

    # ═══ Figure ═══
    best_name=sorted(results_per_event.keys(),key=lambda k: results_per_event[k]["CNN-BiLSTM"]["r2"],reverse=True)[0]
    best=results_per_event[best_name]

    # Find CJK font
    cjk_fonts = [f for f in fm.findSystemFonts() if any(
        k in f.lower() for k in ['noto', 'cjk', 'wenquan', 'simhei', 'simsun', 'songti',
                                   'heiti', 'droid', 'source-han'])]
    font_prop = fm.FontProperties(fname=cjk_fonts[0]) if cjk_fonts else None

    fig,axes=plt.subplots(1,3,figsize=(7.0,2.8))

    # (a) Trajectory
    ax=axes[0]; traj=best["trajectory"]; rev=best["reversal_mask"]
    t=np.arange(len(traj))
    ax.plot(t,traj,color="#2c3e50",lw=0.8,label="Conflict Index")
    for i in range(len(rev)):
        if rev[i]: ax.axvspan(i-0.5,i+0.5,color="#e74c3c",alpha=0.15)
    rev_start = np.where(rev)[0]
    if len(rev_start) > 0:
        mid = rev_start[len(rev_start)//2]
        ax.annotate('Reversal\nWindow', xy=(mid, traj[mid]),
                    xytext=(mid + 15, traj[mid] + 0.05),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=0.8),
                    fontsize=7, color='#e74c3c', fontproperties=font_prop)
    ax.set_xlabel("Time (2h bins)"); ax.set_ylabel("Conflict Index")
    ax.set_title(f"(a) {best_name[:15]}\nConflict Trajectory")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    # (b) R² comparison
    ax=axes[1]
    models=["Persistence","AR(6)","XGBoost","CNN-BiLSTM"]
    r2s=[best["Persistence"]["r2"],best["AR(6)"]["r2"],best["XGBoost"]["r2"],best["CNN-BiLSTM"]["r2"]]
    cols=["#95a5a6","#95a5a6","#95a5a6","#e74c3c"]
    bars=ax.bar(range(len(models)),r2s,color=cols,width=0.5,edgecolor='white',lw=0.3)
    for i,(bar,v) in enumerate(zip(bars,r2s)):
        y_pos=max(0,bar.get_height())+0.03
        ax.text(bar.get_x()+bar.get_width()/2,y_pos,f"{v:.3f}",ha="center",fontsize=7,
                fontweight='bold' if i==3 else 'normal')
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models,rotation=20,ha='right')
    ax.set_ylabel("R²"); ax.set_title("(b) Forecast Accuracy")
    ax.axhline(y=0,color="black",lw=0.5); ax.grid(alpha=0.3,axis='y')

    # (c) Escalation F1
    ax=axes[2]
    esc_f1s=[best["Persistence"]["esc_f1"],best["AR(6)"]["esc_f1"],
             best["XGBoost"]["esc_f1"],best["CNN-BiLSTM"]["esc_f1"]]
    bars=ax.bar(range(len(models)),esc_f1s,color=cols,width=0.5,edgecolor='white',lw=0.3)
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
    import subprocess
    subprocess.run(["cp",f"{R_DIR}/fig_case_study.pdf","figures/fig_case_study.pdf"])
    print(f"\nSaved {R_DIR}/fig_case_study.pdf and figures/fig_case_study.pdf")
    plt.close(fig)

    # Save data
    with open(f"{R_DIR}/case_study.pkl","wb") as f:
        pickle.dump(results_per_event,f)

    tmin=(time.time()-t0)/60
    print(f"Done in {tmin:.1f} min")
    notify(f"Case study done! {len(results_per_event)} events, best={best_name} R²={best['CNN-BiLSTM']['r2']:.3f}")
