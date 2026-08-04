#!/usr/bin/env python3
"""Weibo cross-platform transfer with the TRUE attack/anger models.

Same protocol as exp_weibo_generalization.py (2h bins, MIN_BINS=72, 60/15/25)
but attack/emotion come from the new models:
  - attack: unitary/multilingual-toxic-xlm-roberta
  - emotion: zero-shot NLI (joeddav/xlm-roberta-large-xnli) over
    ["negative", "anger"], e = 0.7*p_neg + 0.3*p_anger
Stance (s_cal) is reused from the cached weibo_comment_level.pkl (row-aligned
on the identical load_weibo() ordering).
"""
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")

from exp_real_components_v2 import toxicity_scores, zero_shot_emotion  # noqa: E402
from exp_weibo_generalization import (BIN_FREQ, CACHE_STORE, DATA_DIR,  # noqa: E402
                                      MIN_BINS, build_weibo_trajectories,
                                      load_weibo)
from optimize_real_signal_v2 import run_complete_evaluation  # noqa: E402

L, H = 12, 6
RISK_TRAIN_END = 0.60
R_DIR = Path("experiment_results_v2")
R_DIR.mkdir(exist_ok=True)


def main():
    t0 = time.time()
    df = load_weibo()
    print(f"Loaded {len(df)} Weibo comments across "
          f"{df['event_name'].nunique()} events")

    texts = df["text_content"].fillna("").astype(str).tolist()
    print("Scoring toxicity (new attack model)...", flush=True)
    a_raw = toxicity_scores(texts)
    print(f"  toxicity mean={a_raw.mean():.3f}")
    print("Scoring zero-shot emotion (negative/anger)...", flush=True)
    p_neg, p_anger = zero_shot_emotion(texts)
    e_raw = 0.7 * p_neg + 0.3 * p_anger
    print(f"  p_neg={p_neg.mean():.3f} p_anger={p_anger.mean():.3f}")

    # Reuse cached stance (row-aligned on the same load_weibo ordering)
    with CACHE_STORE.open("rb") as f:
        cached = pickle.load(f)
    s_cal_old = cached["store"]["s_cal"].to_numpy()
    if len(s_cal_old) != len(df):
        raise SystemExit("Cached store rows do not align with the loaded data")

    df["bin"] = df["dt"].dt.floor(BIN_FREQ)
    a_cal = np.zeros(len(df))
    e_cal = np.zeros(len(df))
    s_cal = np.zeros(len(df))
    audit = {}
    for event in df["event_name"].unique():
        t_df = df[df["event_name"] == event].sort_values("bin")
        if len(t_df) < 20:
            continue
        retained_bins = t_df.groupby("bin").size()
        retained_bins = retained_bins[retained_bins >= 3].index
        if len(retained_bins) < max(L + H + 10, MIN_BINS):
            continue
        full_index = pd.date_range(retained_bins.min(), retained_bins.max(),
                                   freq=BIN_FREQ)
        train_end = int(len(full_index) * RISK_TRAIN_END)
        if train_end <= 0 or train_end >= len(full_index):
            continue
        train_cutoff = full_index[train_end]
        train_df = t_df[t_df["bin"] < train_cutoff]
        if len(train_df) < 4:
            continue
        train_positions = np.asarray([df.index.get_loc(i) for i in train_df.index])
        all_positions = np.asarray([df.index.get_loc(i) for i in t_df.index])
        n_train = len(train_positions)
        qt_a = QuantileTransformer(n_quantiles=min(1000, n_train),
                                   output_distribution="uniform", random_state=42)
        qt_e = QuantileTransformer(n_quantiles=min(1000, n_train),
                                   output_distribution="uniform", random_state=42)
        qt_a.fit(a_raw[train_positions].reshape(-1, 1))
        qt_e.fit(e_raw[train_positions].reshape(-1, 1))
        a_cal[all_positions] = qt_a.transform(
            a_raw[all_positions].reshape(-1, 1)).flatten()
        e_cal[all_positions] = qt_e.transform(
            e_raw[all_positions].reshape(-1, 1)).flatten()
        s_cal[all_positions] = s_cal_old[all_positions]
        audit[event] = {"n_train": int(n_train), "n_all": int(len(t_df)),
                        "n_bins": int(len(retained_bins))}

    a, e, s = 0.5, 0.3, 0.2
    c = (1.0 / (1.0 + np.exp(-(a * a_cal + e * e_cal + s * s_cal)))).clip(0, 1)
    df["c"] = c
    df["a_cal"], df["e_cal"], df["s_cal"] = a_cal, e_cal, s_cal

    trajectories = {}
    for event in df["event_name"].unique():
        if event not in audit:
            continue
        tdf = df[df["event_name"] == event].sort_values("bin")
        agg = tdf.groupby("bin").agg(
            c_bar=("c", lambda x: x.nlargest(max(1, int(len(x) * 0.15))).mean()),
            c_mean=("c", "mean"), c_max=("c", "max"), c_std=("c", "std"),
            n=("c", "count"),
            a_mean=("a_cal", "mean"), e_mean=("e_cal", "mean"),
            s_mean=("s_cal", "mean"),
        ).reset_index()
        agg = agg[(agg["n"] >= 3) & (~agg["c_bar"].isna())]
        if len(agg) >= max(L + H + 10, MIN_BINS):
            trajectories[event] = agg

    print(f"\nBuilt {len(trajectories)} Weibo trajectories "
          f"(audited events: {len(audit)})")
    output = run_complete_evaluation(trajectories, freq=BIN_FREQ)
    best = output["selected_result"]
    results = {
        "platform": "weibo_reversal_v2",
        "n_events_raw": int(df["event_name"].nunique()),
        "n_comments": int(len(df)),
        "n_trajectories": len(trajectories),
        "models": {"attack": "unitary/multilingual-toxic-xlm-roberta",
                   "emotion": "joeddav/xlm-roberta-large-xnli"},
        "selected_span": output["selected_span"],
        "selected_feature_group": output["selected_feature_group"],
        "test": {k: v for k, v in best["test"].items() if k != "per_topic"},
        "topic_summary": output["topic_summary"],
    }
    out = R_DIR / "weibo_v2_results.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out} in {(time.time() - t0) / 60:.1f} min")
    print(f"WEIBO v2 test: logistic F1={best['test']['logistic']['f1']:.4f} "
          f"AUC={best['test']['logistic']['auc']:.4f} | "
          f"persistence F1={best['test']['persistence']['f1']:.4f} "
          f"AUC={best['test']['persistence']['auc']:.4f}")


if __name__ == "__main__":
    main()
