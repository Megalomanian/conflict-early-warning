"""C3: Lead time analysis for early warning."""
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle, warnings; warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family":"serif","font.size":9,"figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight"})

L,H=12,6; DEV="cuda" if torch.cuda.is_available() else "cpu"; R="experiment_results"

class CNNBiLSTM(nn.Module):
    def __init__(self,h=64):
        super().__init__(); self.h=h
        self.conv=nn.Sequential(nn.Conv1d(1,32,3,padding=1),nn.ReLU(),nn.Conv1d(32,32,5,padding=2),nn.ReLU())
        self.lstm=nn.LSTM(32,h,2,batch_first=True,dropout=0.2,bidirectional=True)
        self.proj=nn.Linear(h*2,H)
    def forward(self,x):
        c=self.conv(x.transpose(1,2)).transpose(1,2); o,_=self.lstm(c)
        return self.proj(torch.cat([o[:,-1,:self.h],o[:,0,self.h:]],dim=-1))

def gen(n=200,s=None):
    r=np.random.RandomState(s); base=0.3+r.uniform(0,0.15); ne=r.randint(1,4); tr=np.zeros(n)
    for _ in range(ne):
        es,ed=r.randint(10,n-30),r.randint(5,15); ep=r.uniform(0.15,0.35)
        te=np.arange(ed*2); sig=1.0/(1.0+np.exp(-(te-ed/2)/(ed/8)))
        sig=(sig-sig[0])/sig.max()*ep; sig=sig*np.exp(-(te-ed)/(ed*2))
        idx=min(es+len(sig),n); tr[es:idx]+=sig[:idx-es]
    sn=0.02*np.sin(2*np.pi*np.arange(n)/14.0); w=r.normal(0,0.02,n)
    pk=np.zeros(n); pk[0]=w[0]
    for i in range(1,n): pk[i]=0.6*pk[i-1]+0.4*w[i]
    return np.clip(base+tr+sn+pk,0,1),(tr>0.1).astype(int)

print("Generating data...")
Xl,yl=[],[]
for i in range(40):
    v,e=gen(200,s=i)
    for j in range(len(v)-L-H+1): Xl.append(v[j:j+L]); yl.append(v[j+L:j+L+H])
X=torch.tensor(np.array(Xl,dtype=np.float32)).unsqueeze(-1)
y=torch.tensor(np.array(yl,dtype=np.float32))
n=len(X); n_tr=int(n*0.75)
perm=np.random.RandomState(42).permutation(n)
Xtr,ytr=X[perm[:n_tr]].to(DEV),y[perm[:n_tr]].to(DEV)
Xte,yte=X[perm[n_tr:]].to(DEV),y[perm[n_tr:]].to(DEV)
yte_np=yte.cpu().numpy()
print(f"Data: {n} windows, train={n_tr}, test={n-n_tr}")

print("Training CNN-BiLSTM...")
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
print(f"Trained: {len(best_st)} params, best_val={best_vl:.4f}, epochs={ep+1}")

# Lead time analysis
ETA=0.65; leads=[]
for i in range(len(yte_np)):
    gt_peak=np.argmax(yte_np[i]); gt_val=yte_np[i,gt_peak]
    pred_peak=np.argmax(y_pred[i]); pred_val=y_pred[i].max()
    if gt_val>=ETA and pred_val>=ETA:
        leads.append(gt_peak-pred_peak)  # + = early warning

print(f"\nLead Time Results:")
print(f"  Escalation prevalence: {(yte_np.max(1)>=ETA).mean():.1%}")
print(f"  TP predictions with lead time: {len(leads)}")
if leads:
    print(f"  Mean lead time: {np.mean(leads):+.1f} +- {np.std(leads):.1f} bins (+ = early)")
    print(f"  Early warning rate (lead>0): {sum(1 for l in leads if l>0)}/{len(leads)} = {sum(1 for l in leads if l>0)/len(leads):.0%}")

# Figure
fig,axes=plt.subplots(1,2,figsize=(7.0,2.8))
ax=axes[0]
if leads:
    ax.hist(leads,bins=10,color='#3498db',alpha=0.8,edgecolor='white',lw=0.5)
    ax.axvline(x=np.mean(leads),color='#e74c3c',ls='--',lw=1.0,label=f'Mean={np.mean(leads):+.1f} bins')
    ax.axvline(x=0,color='black',lw=0.5)
    ax.set_xlabel('Lead Time (bins, + = early)'); ax.set_ylabel('Count')
    ax.set_title(f'(a) Lead Time Distribution (N={len(leads)})')
    ax.legend(fontsize=7); ax.grid(alpha=0.3,axis='y')

ax=axes[1]; demo_v,demo_e=gen(100,s=77)
P_WINDOW=H
triggers=[i for i in range(len(demo_v)-P_WINDOW) if demo_v[i:i+P_WINDOW].max()>=0.65]
ax.plot(demo_v,color='#2c3e50',lw=0.8,label='Conflict Index')
for i in range(len(demo_e)):
    if demo_e[i]: ax.axvspan(i-0.5,i+0.5,color='#e74c3c',alpha=0.1)
if triggers:
    ts=triggers[::max(1,len(triggers)//6)]
    ax.scatter(ts,[demo_v[t] for t in ts],color='#e74c3c',marker='^',s=40,zorder=5,edgecolors='white',lw=0.5,label=f'Warning ({len(triggers)})')
ax.axhline(y=0.65,color='#e74c3c',ls='--',lw=0.8,alpha=0.5,label=r'Threshold')
ax.set_xlabel('Time (bins)'); ax.set_ylabel('Conflict Index')
ax.set_title('(b) Warning Trigger Visualization')
ax.legend(loc='upper left',fontsize=7,ncol=2); ax.grid(alpha=0.3)

fig.suptitle('Early Warning Lead Time',fontsize=11,fontweight='bold')
fig.tight_layout(); fig.savefig(f"{R}/fig_lead_time.pdf"); plt.close(fig)
print(f"Saved {R}/fig_lead_time.pdf")

with open(f"{R}/c3_lead_time.pkl","wb") as f:
    pickle.dump({"leads":leads,"mean":np.mean(leads) if leads else 0,"std":np.std(leads) if leads else 0},f)
print("Done!")
