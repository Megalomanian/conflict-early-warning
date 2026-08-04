#!/usr/bin/env python3
"""Cross-platform generalization: Weibo reversal events.

Transfers the exact Zhihu pipeline (causal cleaning + risk-state classification)
to the 27 Weibo public-opinion reversal events:
  - same conflict-index construction (sentiment attack/emotion + stance)
  - same train-only ECDF calibration and stance centers (first 60% of the
    regular 12h grid per event)
  - same comment-level fusion, bin aggregation, EWMA cleaning and
    risk-state classification (next-6-bin exceedance of the training-period
    80th percentile), evaluated on the held-out test segment.

Reports test F1/AUC vs persistence with topic-level bootstrap, alongside the
Zhihu reference numbers for comparison.
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

from experiment_real_model_v2 import RealConflictComputer  # noqa: E402
from optimize_real_signal_v2 import run_complete_evaluation  # noqa: E402

L, H = 12, 6
RISK_TRAIN_END = 0.60
# Weibo reversal events last 2-3 weeks; on a 12h grid that yields only
# 28-45 bins and 1-3 test windows (single-class splits -> AUC undefined).
# The original case_study.py also used 2h bins for Weibo. We keep EVERYTHING
# else identical (L/H, EWMA spans, 80th-percentile threshold, 60/15/25 split).
BIN_FREQ = "2h"
MIN_BINS = 72  # 6 days at 2h; guarantees a non-degenerate test segment
R_DIR = Path("experiment_results_v2")
R_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("zhihu_topics/weibo_reversal/data")
CACHE_STORE = R_DIR / "weibo_comment_level.pkl"
CACHE_TRAJ = R_DIR / "weibo_trajectories.pkl"


def load_weibo():
    meta = pd.read_csv(DATA_DIR / "data_case.csv", encoding="gbk")
    all_dfs = []
    for _, row in meta.iterrows():
        fn = DATA_DIR / row["csvname"]
        if not fn.exists():
            continue
        df = pd.read_csv(fn, encoding="utf-8")
        df["dt"] = pd.to_datetime(df["text_time"], format="%y-%m-%d %H:%M",
                                  errors="coerce")
        df = df.dropna(subset=["dt", "text_content"])
        df["event_name"] = str(row["name"])
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def build_weibo_trajectories(df, computer, r):
    """Same per-event leakage-free preprocessing as the Zhihu pipeline."""
    df = df.copy()
    df["bin"] = df["dt"].dt.floor(BIN_FREQ)
    s_raw = np.zeros(len(df))
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

        s_raw[all_positions] = computer.stance_polarization(
            r["embeddings"][train_positions], r["embeddings"][all_positions])

        n_train = len(train_positions)
        qt_a = QuantileTransformer(n_quantiles=min(1000, n_train),
                                   output_distribution="uniform", random_state=42)
        qt_e = QuantileTransformer(n_quantiles=min(1000, n_train),
                                   output_distribution="uniform", random_state=42)
        qt_s = QuantileTransformer(n_quantiles=min(1000, n_train),
                                   output_distribution="uniform", random_state=42)
        qt_a.fit(r["attack"][train_positions].reshape(-1, 1))
        qt_e.fit(r["emotion"][train_positions].reshape(-1, 1))
        qt_s.fit(s_raw[train_positions].reshape(-1, 1))

        a_cal[all_positions] = qt_a.transform(
            r["attack"][all_positions].reshape(-1, 1)).flatten()
        e_cal[all_positions] = qt_e.transform(
            r["emotion"][all_positions].reshape(-1, 1)).flatten()
        s_cal[all_positions] = qt_s.transform(
            s_raw[all_positions].reshape(-1, 1)).flatten()
        audit[event] = {"n_train": int(n_train), "n_all": int(len(t_df)),
                        "n_bins": int(len(retained_bins))}

    a, e, s = 0.5, 0.3, 0.2
    c = (1.0 / (1.0 + np.exp(-(a * a_cal + e * e_cal + s * s_cal)))).clip(0, 1)
    df["c"] = c
    df["a_cal"], df["e_cal"], df["s_cal"] = a_cal, e_cal, s_cal
    store = df[["event_name", "bin", "a_cal", "e_cal", "s_cal"]]

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
    return trajectories, audit, store


def get_trajectories():
    """Return (trajectories, audit), computing + caching if needed."""
    if CACHE_TRAJ.exists():
        with CACHE_TRAJ.open("rb") as f:
            cached = pickle.load(f)
        print(f"Loaded cached Weibo trajectories "
              f"({len(cached['trajectories'])} events)")
        return cached["trajectories"], cached["audit"]

    t0 = time.time()
    df = load_weibo()
    print(f"Loaded {len(df)} Weibo comments across "
          f"{df['event_name'].nunique()} events")

    computer = RealConflictComputer()
    r = computer.compute_batch(df["text_content"].fillna("").astype(str).tolist())

    trajectories, audit, store = build_weibo_trajectories(df, computer, r)
    with CACHE_STORE.open("wb") as f:
        pickle.dump({"store": store, "audit": audit}, f)
    with CACHE_TRAJ.open("wb") as f:
        pickle.dump({"trajectories": trajectories, "audit": audit}, f)
    print(f"Computed + cached in {(time.time() - t0) / 60:.1f} min")
    return trajectories, audit


def main():
    t0 = time.time()
    trajectories, audit = get_trajectories()
    counts_df = load_weibo()  # cheap CSV read, only for counts
    print(f"\n{len(trajectories)} Weibo trajectories "
          f"(audited events: {len(audit)})")
    for evt, info in list(audit.items())[:12]:
        print(f"  {str(evt)[:36]:36s} train={info['n_train']:6d} "
              f"all={info['n_all']:7d} bins={info['n_bins']:3d}")

    output = run_complete_evaluation(trajectories, freq=BIN_FREQ)
    best = output["selected_result"]

    zhihu_ref = None
    ref_path = R_DIR / "signal_cleaning_results.json"
    if ref_path.exists():
        with ref_path.open() as f:
            zhihu_ref = json.load(f)["selected_result"]["test"]

    results = {
        "platform": "weibo_reversal",
        "n_events_raw": int(counts_df["event_name"].nunique()),
        "n_comments": int(len(counts_df)),
        "n_trajectories": len(trajectories),
        "audit": audit,
        "selected_span": output["selected_span"],
        "selected_feature_group": output["selected_feature_group"],
        "selected_c": output["selected_c"],
        "selected_decision_threshold": output["selected_decision_threshold"],
        "test": {k: v for k, v in best["test"].items() if k != "per_topic"},
        "valid": {k: v for k, v in best["valid"].items() if k != "per_topic"},
        "topic_summary": output["topic_summary"],
        "zhihu_test_reference": zhihu_ref,
    }
    out = R_DIR / "weibo_generalization_results.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with (R_DIR / "weibo_generalization_results.pkl").open("wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved {out} in {(time.time() - t0) / 60:.1f} min")
    print(f"WEIBO  test: logistic F1={best['test']['logistic']['f1']:.4f} "
          f"AUC={best['test']['logistic']['auc']:.4f} | "
          f"persistence F1={best['test']['persistence']['f1']:.4f} "
          f"AUC={best['test']['persistence']['auc']:.4f}")
    if zhihu_ref:
        print(f"ZHIHU  test: logistic F1={zhihu_ref['logistic']['f1']:.4f} "
              f"AUC={zhihu_ref['logistic']['auc']:.4f} | "
              f"persistence F1={zhihu_ref['persistence']['f1']:.4f} "
              f"AUC={zhihu_ref['persistence']['auc']:.4f}")


if __name__ == "__main__":
    main()
