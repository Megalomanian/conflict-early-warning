#!/bin/bash
# Auto pipeline: wait for the large XNLI model download -> run all v2
# experiments -> summarize -> notify via PushPlus (notify_server.sh).
cd ~/LSTM_paper
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
mkdir -p logs

# Wait until the XNLI model cache is complete (>1.4 GB) or timeout 30 min
MODEL_DIR=~/.cache/huggingface/hub/models--joeddav--xlm-roberta-large-xnli
for i in $(seq 1 90); do
  sz=$(du -sb "$MODEL_DIR" 2>/dev/null | awk '{print $1}')
  if [ -n "$sz" ] && [ "$sz" -gt 1400000000 ]; then
    echo "model ready: $sz bytes (after $((i*20))s)" | tee -a logs/run_all_v2.log
    break
  fi
  sleep 20
done

for step in exp_real_components_v2.py exp_weibo_v2.py exp_heldout.py exp_stance_audit.py; do
  echo "===== $step started $(date) =====" | tee -a logs/run_all_v2.log
  python "$step" >> logs/run_all_v2.log 2>&1 \
    && echo "===== $step OK $(date) =====" | tee -a logs/run_all_v2.log \
    || echo "!!!!! $step FAILED $(date) !!!!!" | tee -a logs/run_all_v2.log
done
echo ALLV2_DONE > logs/allv2_done.flag

MSG=$(python - <<'EOF'
import json
parts = []
try:
    d = json.load(open("experiment_results_v2/signal_cleaning_v2_results.json"))
    t, p = d["test"]["logistic"], d["test"]["persistence"]
    parts.append("知乎v2 F1=%.3f AUC=%.3f(持久F1=%.3f)" % (t["f1"], t["auc"], p["f1"]))
except Exception:
    parts.append("知乎v2 失败")
try:
    w = json.load(open("experiment_results_v2/weibo_v2_results.json"))
    t, p = w["test"]["logistic"], w["test"]["persistence"]
    parts.append("微博v2 F1=%.3f AUC=%.3f(持久F1=%.3f)" % (t["f1"], t["auc"], p["f1"]))
except Exception:
    parts.append("微博v2 失败")
try:
    h = json.load(open("experiment_results_v2/heldout_results.json"))
    parts.append("主题留出F1=%.3f" % h["zhihu_topic_heldout"]["logistic_f1"]["mean"])
except Exception:
    parts.append("heldout 失败")
try:
    s = json.load(open("experiment_results_v2/stance_audit.json"))
    parts.append("立场审计ARI=%.2f" % s["summary"]["stability_ari"]["mean"])
except Exception:
    parts.append("立场审计 失败")
print("; ".join(parts))
EOF
)
./notify_server.sh "LSTM_paper v2 实验全部完成: $MSG"
echo "NOTIFIED: $MSG" >> logs/run_all_v2.log
