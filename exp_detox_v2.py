#!/usr/bin/env python3
"""Construct validity of the NEW attack model on Wikipedia Detox.

Scores the multilingual toxicity model (unitary/multilingual-toxic-xlm-roberta)
on the same 191,634 human-annotated Detox comments and reports Spearman/Pearson
and AUC, directly comparable to the review-sentiment proxy (rho 0.254,
AUC 0.744).
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from exp_detox_validation import load_detox  # noqa: E402
from exp_real_components_v2 import toxicity_scores  # noqa: E402

R_DIR = Path("experiment_results_v2")


def main():
    t0 = time.time()
    df = load_detox()
    print(f"Loaded {len(df)} labeled comments")
    texts = df["text"].fillna("").astype(str).tolist()
    attack = toxicity_scores(texts)
    y = df["toxic"].to_numpy(dtype=float)
    rho, rho_p = spearmanr(attack, y)
    r_p, p_p = pearsonr(attack, y)
    auc = roc_auc_score(y, attack)
    results = {
        "model": "unitary/multilingual-toxic-xlm-roberta",
        "n": int(len(df)),
        "positive_rate": float(y.mean()),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "pearson_r": float(r_p),
        "pearson_p": float(p_p),
        "auc": float(auc),
        "attack_mean_label1": float(attack[y > 0.5].mean()),
        "attack_mean_label0": float(attack[y <= 0.5].mean()),
        "comparison_review_sentiment_proxy": {
            "spearman_rho": 0.254, "auc": 0.744},
    }
    out = R_DIR / "detox_toxicity_validation.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {out} in {(time.time() - t0) / 60:.1f} min")
    print(f"toxicity model: Spearman={rho:.3f} AUC={auc:.3f} "
          f"(mean scores: label1={results['attack_mean_label1']:.3f} "
          f"label0={results['attack_mean_label0']:.3f})")


if __name__ == "__main__":
    main()
