#!/usr/bin/env python3
"""Audit: do the two stance-polarization clusters correspond to two STANCES
rather than two subtopics?

For each Zhihu topic we reproduce the pipeline's stance protocol
(per-topic KMeans with n_clusters=2, n_init=5, random_state=42, fitted on
training-period comments only) and then diagnose the two clusters:
  1. size balance
  2. TF-IDF top keywords per cluster (jieba segmentation)
  3. stance-lexicon hit rates per cluster (pro-support vs pro-opposition words)
  4. clustering stability (pairwise ARI across 5 KMeans seeds)
  5. 5 representative comments per cluster (closest to each centroid)
"""
import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

from exp_component_ablation import load_records  # noqa: E402

R_DIR = Path("experiment_results_v2")
L, H = 12, 6
RISK_TRAIN_END = 0.60

PRO_STANCE_WORDS = [
    "支持", "赞同", "同意", "力挺", "声援", "点赞", "合理", "应该", "希望",
    "呼吁", "建议", "理解", "认可", "赞成", "表扬", "感谢", "致敬", "加油",
]
ANTI_STANCE_WORDS = [
    "反对", "抵制", "谴责", "痛斥", "不满", "愤怒", "恶心", "垃圾", "滚",
    "举报", "抗议", "严惩", "道歉", "开除", "违法", "不公平", "质疑", "驳回",
    "失望", "愤怒", "可怕", "过分", "离谱",
]


def stance_lexicon_hits(texts, words):
    hits = np.zeros(len(texts))
    for w in words:
        hits += np.asarray([w in t for t in texts], dtype=float)
    return hits > 0


def main():
    import jieba
    from experiment_real_model_v2 import RealConflictComputer

    df = load_records()
    print(f"Loaded {len(df)} records")
    texts = df["text"].tolist()
    computer = RealConflictComputer()
    print("Embedding all comments (MiniLM)...", flush=True)
    embs = computer.emb_model.encode(texts, batch_size=256,
                                     convert_to_numpy=True,
                                     normalize_embeddings=True)
    print(f"Embeddings: {embs.shape}")

    df["bin"] = pd.to_datetime(df["ts"], unit="s").dt.floor("12h")
    report = {}
    for topic in df["topic"].unique():
        t_df = df[df["topic"] == topic].sort_values("bin")
        if len(t_df) < 20:
            continue
        retained_bins = t_df.groupby("bin").size()
        retained_bins = retained_bins[retained_bins >= 3].index
        if len(retained_bins) < L + H + 10:
            continue
        full_index = pd.date_range(retained_bins.min(), retained_bins.max(),
                                   freq="12h")
        train_end = int(len(full_index) * RISK_TRAIN_END)
        if train_end <= 0 or train_end >= len(full_index):
            continue
        train_cutoff = full_index[train_end]
        train_df = t_df[t_df["bin"] < train_cutoff]
        if len(train_df) < 4:
            continue
        train_positions = np.asarray([df.index.get_loc(i) for i in train_df.index])
        all_positions = np.asarray([df.index.get_loc(i) for i in t_df.index])

        km = KMeans(n_clusters=2, n_init=5, random_state=42)
        km.fit(embs[train_positions])
        labels = km.predict(embs[all_positions])
        n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
        idx0 = all_positions[labels == 0]
        idx1 = all_positions[labels == 1]
        texts0 = [texts[i] for i in idx0]
        texts1 = [texts[i] for i in idx1]

        # Stability across seeds (fit on the same train positions)
        aris = []
        for seed in (0, 1, 7, 42, 123):
            km_s = KMeans(n_clusters=2, n_init=5, random_state=seed)
            km_s.fit(embs[train_positions])
            labels_s = km_s.predict(embs[all_positions])
            aris.append(adjusted_rand_score(labels, labels_s))
        stability = float(np.mean(aris))

        # TF-IDF top terms per cluster
        vectorizer = TfidfVectorizer(
            analyzer=lambda x: jieba.lcut(x), max_features=3000,
            token_pattern=None)
        corpus = texts0 + texts1
        tfidf = vectorizer.fit_transform(corpus)
        terms = vectorizer.get_feature_names_out()
        top0 = tfidf[:len(texts0)].toarray().mean(axis=0)
        top1 = tfidf[len(texts0):].toarray().mean(axis=0)
        top_terms_0 = [str(terms[i]) for i in np.argsort(top0)[-15:][::-1]]
        top_terms_1 = [str(terms[i]) for i in np.argsort(top1)[-15:][::-1]]

        # Stance lexicon hit rates
        pro0 = float(stance_lexicon_hits(texts0, PRO_STANCE_WORDS).mean())
        pro1 = float(stance_lexicon_hits(texts1, PRO_STANCE_WORDS).mean())
        anti0 = float(stance_lexicon_hits(texts0, ANTI_STANCE_WORDS).mean())
        anti1 = float(stance_lexicon_hits(texts1, ANTI_STANCE_WORDS).mean())

        # Representative comments (closest to each centroid)
        d0 = np.linalg.norm(embs[idx0] - km.cluster_centers_[0], axis=1)
        d1 = np.linalg.norm(embs[idx1] - km.cluster_centers_[1], axis=1)
        reps0 = [texts[i][:120] for i in idx0[np.argsort(d0)[:5]]]
        reps1 = [texts[i][:120] for i in idx1[np.argsort(d1)[:5]]]

        report[topic] = {
            "n": int(len(all_positions)), "n0": n0, "n1": n1,
            "balance": round(min(n0, n1) / max(n0, n1), 3),
            "stability_ari": round(stability, 3),
            "pro_lexicon_hit": {"c0": round(pro0, 3), "c1": round(pro1, 3)},
            "anti_lexicon_hit": {"c0": round(anti0, 3), "c1": round(anti1, 3)},
            "top_terms_c0": top_terms_0,
            "top_terms_c1": top_terms_1,
            "example_c0": reps0,
            "example_c1": reps1,
        }
        print(f"  {topic[:24]:24s} n={len(all_positions):5d} "
              f"bal={report[topic]['balance']:.2f} "
              f"ARI={stability:.2f} pro=({pro0:.2f},{pro1:.2f}) "
              f"anti=({anti0:.2f},{anti1:.2f})")

    # Aggregate judgment
    bal = [r["balance"] for r in report.values()]
    stab = [r["stability_ari"] for r in report.values()]
    pro_diffs = [abs(r["pro_lexicon_hit"]["c0"] - r["pro_lexicon_hit"]["c1"])
                 for r in report.values()]
    anti_diffs = [abs(r["anti_lexicon_hit"]["c0"] - r["anti_lexicon_hit"]["c1"])
                  for r in report.values()]
    summary = {
        "n_topics": len(report),
        "balance": {"mean": round(float(np.mean(bal)), 3),
                    "min": round(float(np.min(bal)), 3)},
        "stability_ari": {"mean": round(float(np.mean(stab)), 3),
                          "min": round(float(np.min(stab)), 3)},
        "pro_lexicon_abs_diff": {"mean": round(float(np.mean(pro_diffs)), 3),
                                 "max": round(float(np.max(pro_diffs)), 3)},
        "anti_lexicon_abs_diff": {"mean": round(float(np.mean(anti_diffs)), 3),
                                  "max": round(float(np.max(anti_diffs)), 3)},
        "interpretation": (
            "High ARI + large lexicon diffs + balanced sizes support the "
            "two-stance reading; low ARI or entity-dominated keywords with "
            "no lexicon difference indicate subtopic splitting. "
            "See per-topic examples in this report."),
    }
    out = {"summary": summary, "per_topic": report}
    path = R_DIR / "stance_audit.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {path}")
    print("SUMMARY:", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
