#!/bin/bash
# Run all automatic experiments sequentially with logging.
cd ~/LSTM_paper
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
mkdir -p logs

for step in \
  "exp_detox_validation.py" \
  "exp_component_ablation.py" \
  "exp_weibo_generalization.py" \
  "exp_competitor_split_analysis.py"; do
  echo "===== $step started $(date) =====" | tee -a logs/run_all.log
  python "$step" >> logs/run_all.log 2>&1 \
    && echo "===== $step OK $(date) =====" | tee -a logs/run_all.log \
    || echo "!!!!! $step FAILED $(date) !!!!!" | tee -a logs/run_all.log
done
echo "ALL EXPERIMENTS DONE $(date)" | tee -a logs/run_all.log
