#!/bin/bash
# Server environment setup: venv, CUDA-compatible torch, HF mirror, Detox data.
# Uses domestic mirrors: TUNA PyPI, hf-mirror.com for HF models.
set -e
export HF_ENDPOINT=https://hf-mirror.com
PIP_INDEX="https://mirror.sjtu.edu.cn/pypi/web/simple"
cd ~/LSTM_paper

echo "==> python $(python3 --version)"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -U pip wheel -i "$PIP_INDEX"
echo "==> installing torch (cu118, driver 525 compatible)..."
python -c "import torch" 2>/dev/null || \
  pip install -q torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
echo "==> installing ML packages via TUNA mirror..."
pip install -q -i "$PIP_INDEX" transformers==4.44.2 sentence-transformers==2.7.0 \
  scikit-learn==1.4.2 pandas==2.2.2 numpy==1.26.4 xgboost==2.0.3 tqdm scipy

python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO-GPU")
EOF

echo "==> pre-downloading models via HF mirror..."
python - <<'EOF'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
AutoModelForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment")
SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
try:
    AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-xlm-roberta-base-sentiment")
    print("twitter model OK")
except Exception as exc:
    print("twitter model SKIP:", exc)
EOF

echo "==> downloading Wikipedia Detox labels (wiki_toxic mirror)..."
mkdir -p wikipedia_detox
python - <<'EOF'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import hf_hub_download
for name in ("train.csv", "test.csv"):
    out = os.path.join("wikipedia_detox", f"wiki_toxic_{name}")
    if os.path.exists(out):
        continue
    hf_hub_download(repo_id="OxAISH-AL-LLM/wiki_toxic", filename=name,
                    repo_type="dataset", local_dir="wikipedia_detox/.hub_detox")
    os.rename(os.path.join("wikipedia_detox/.hub_detox", name), out)
    print("downloaded", out)
EOF
ls -la wikipedia_detox/ | grep -E "wiki_toxic|tsv" || true
echo "==> setup done"
