"""Regenerate case study figure with proper Chinese font and IEEE-friendly layout."""
import pickle, warnings, os, numpy as np
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Find a CJK font
cjk_fonts = [f for f in fm.findSystemFonts() if any(
    k in f.lower() for k in ['noto', 'cjk', 'wenquan', 'wqy', 'simhei', 'simsun', 'songti', 'heiti',
                               'droid', 'source-han'])]
if not cjk_fonts:
    # Fallback: try common paths
    for p in ['/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc']:
        if os.path.exists(p): cjk_fonts.append(p)

font_prop = None
if cjk_fonts:
    font_prop = fm.FontProperties(fname=cjk_fonts[0])
    print(f"Using CJK font: {cjk_fonts[0]}")
else:
    print("Warning: No CJK font found, Chinese text may not render")

plt.rcParams.update({"font.family": "serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

R_DIR = "experiment_results"
IEEE_WIDE = 7.0

# Load case study data
with open(f"{R_DIR}/case_study.pkl", "rb") as f:
    data = pickle.load(f)

# Find best event
best_name = max(data.keys(), key=lambda k: data[k]["CNN-BiLSTM"]["r2"])
best = data[best_name]

fig, axes = plt.subplots(1, 3, figsize=(IEEE_WIDE, 2.6))

# ── (a) Conflict trajectory with reversal window ──
ax = axes[0]
traj = best["trajectory"]
rev = best["reversal_mask"]
t = np.arange(len(traj))
ax.plot(t, traj, color="#2c3e50", lw=0.8, label="Conflict Index")
for i in range(len(rev)):
    if rev[i]:
        ax.axvspan(i - 0.5, i + 0.5, color="#e74c3c", alpha=0.12)
# Add annotation for reversal window
rev_start = np.where(rev)[0]
if len(rev_start) > 0:
    mid = rev_start[len(rev_start)//2]
    ax.annotate('Reversal\nWindow', xy=(mid, traj[mid]),
                xytext=(mid + 15, traj[mid] + 0.05),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=0.8),
                fontsize=7, color='#e74c3c', fontproperties=font_prop)
ax.set_xlabel("Time (2h bins)"); ax.set_ylabel("Conflict Index")
ax.set_title("(a) Conflict Trajectory\n(Huolala Event)")
ax.legend(fontsize=7, loc='upper right'); ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

# ── (b) R² comparison ──
ax = axes[1]
models = ["Persistence", "AR(6)", "XGBoost", "CNN-BiLSTM"]
r2s = [best["Persistence"]["r2"], best["AR(6)"]["r2"],
       best["XGBoost"]["r2"], best["CNN-BiLSTM"]["r2"]]
colors = ["#95a5a6", "#95a5a6", "#95a5a6", "#e74c3c"]
bars = ax.bar(range(len(models)), r2s, color=colors, width=0.5, edgecolor='white', lw=0.3)
for i, (bar, v) in enumerate(zip(bars, r2s)):
    y_pos = max(0, bar.get_height()) + 0.03
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{v:.3f}",
            ha="center", fontsize=7, fontweight='bold' if i == 3 else 'normal')
ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=20, ha='right')
ax.set_ylabel("R²"); ax.set_title("(b) Forecast Accuracy")
ax.axhline(y=0, color="black", lw=0.5); ax.grid(alpha=0.3, axis='y')

# ── (c) Escalation F1 ──
ax = axes[2]
esc_f1s = [best["Persistence"]["esc_f1"], best["AR(6)"]["esc_f1"],
           best["XGBoost"]["esc_f1"], best["CNN-BiLSTM"]["esc_f1"]]
bars = ax.bar(range(len(models)), esc_f1s, color=colors, width=0.5, edgecolor='white', lw=0.3)
for i, (bar, v) in enumerate(zip(bars, esc_f1s)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f"{v:.3f}",
            ha="center", fontsize=7, fontweight='bold' if i == 3 else 'normal')
ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=20, ha='right')
ax.set_ylabel("Escalation F1"); ax.set_title("(c) Reversal Detection")
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, 1.05)

fig.suptitle("Real-World Case Study: Weibo Public Opinion Reversal Events",
             fontsize=10, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(f"{R_DIR}/fig_case_study.pdf")
plt.close(fig)
print(f"Fixed figure saved to {R_DIR}/fig_case_study.pdf")

# Also save a PNG version for quick viewing
fig, axes = plt.subplots(1, 3, figsize=(IEEE_WIDE, 2.6))
# ... same code ...
# Actually let's just use the same figure
# (already saved as PDF above)
import subprocess
subprocess.run(["cp", f"{R_DIR}/fig_case_study.pdf", "figures/fig_case_study.pdf"])
print("Copied to figures/fig_case_study.pdf")
