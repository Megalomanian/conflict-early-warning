#!/bin/bash
# Run the v2 component/validation experiments sequentially.
cd ~/LSTM_paper
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
mkdir -p logs

for step in \
  "exp_detox_v2.py" \
  "exp_real_components_v2.py" \
  "exp_weibo_v2.py" \
  "exp_heldout.py" \
  "exp_stance_audit.py"; do
  echo "===== $step started $(date) =====" | tee -a logs/run_all_v2.log
  python "$step" >> logs/run_all_v2.log 2>&1 \
    && echo "===== $step OK $(date) =====" | tee -a logs/run_all_v2.log \
    || echo "!!!!! $step FAILED $(date) !!!!!" | tee -a logs/run_all_v2.log
done
echo "ALL V2 EXPERIMENTS DONE $(date)" | tee -a logs/run_all_v2.log
