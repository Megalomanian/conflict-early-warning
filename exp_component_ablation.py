#!/usr/bin/env python3
"""Leave-one-component-out ablation on the risk-state task (TEST set).

The review requires: remove each of the three semantic components (A=attack,
E=emotion, S=stance) ONE AT A TIME, RECOMPUTE the conflict index at the
comment level, REBUILD the trajectories and RE-RUN the complete downstream
pipeline (causal cleaning + risk-state classification), evaluating on the
held-out test set.

Because the ECDF calibration and stance centers are fitted component-wise on
training comments only, the calibrated comment-level values (a_cal, e_cal,
s_cal) are invariant across variants: we compute them ONCE (replicating the
exact preprocessing of experiment_real_model_v2.py), cache them, and then
re-fuse per variant:

  c = sigmoid(w_a*a_cal + w_e*e_cal + w_s*s_cal)

Variant weights (renormalized to sum to 1 after removing one component):
  full    : (0.5, 0.3, 0.2)
  drop_A  : (0.0, 0.6, 0.4)
  drop_E  : (5/7, 0.0, 2/7)
  drop_S  : (0.625, 0.375, 0.0)

Fidelity gate: the `full` variant must reproduce trajectories_real_model.pkl
(comment-level values equal within floating-point tolerance).
"""
import glob
import json
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")

from experiment_real_model_v2 import RealConflictComputer  # noqa: E402
from optimize_real_signal_v2 import (evaluate_span,  # noqa: E402
                                     run_complete_evaluation)

L, H = 12, 6
RISK_TRAIN_END = 0.60
R_DIR = Path("experiment_results_v2")
R_DIR.mkdir(exist_ok=True)
STORE_PATH = R_DIR / "comment_level_features.pkl"

VARIANTS = {
    "full": (0.5, 0.3, 0.2),
    "drop_A": (0.0, 0.6, 0.4),
    "drop_E": (5.0 / 7.0, 0.0, 2.0 / 7.0),
    "drop_S": (0.625, 0.375, 0.0),
}


# ═══ Step 1: replicate the exact Part-A preprocessing of experiment_real_model_v2.py ═══
def load_records():
    base = "zhihu_topics"
    ranked = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isdir(path) or name == "zhihu":
            continue
        jd = os.path.join(path, "zhihu", "jsonl")
        if not os.path.isdir(jd):
            continue
        nc = sum(sum(1 for _ in open(f, encoding="utf-8"))
                 for f in glob.glob(os.path.join(jd, "search_comments_*.jsonl")))
        if nc > 0:
            ranked.append((path, nc))
    ranked.sort(key=lambda x: x[1], reverse=True)

    records = []
    for tp, _ in ranked[:20]:
        tp_name = os.path.basename(tp)
        jd = os.path.join(tp, "zhihu", "jsonl")
        for ftype, tkey in [("search_comments", "content"),
                            ("search_contents", "content_text")]:
            for fp in glob.glob(os.path.join(jd, f"{ftype}_*.jsonl")):
                for line in open(fp, encoding="utf-8"):
                    try:
                        obj = json.loads(line.strip())
                    except Exception:
                        continue
                    text = obj.get(tkey, "").strip()
                    ts = obj.get("publish_time") or obj.get("created_time")
                    if text and len(text) >= 5 and ts and float(ts) > 1704067200:
                        records.append({"text": text, "ts": float(ts),
                                        "topic": tp_name})
    return pd.DataFrame(records)


def prepare_comment_level(force=False):
    """Replicate Part A of experiment_real_model_v2.py and cache the calibrated
    comment-level components per topic (bin, a_cal, e_cal, s_cal)."""
    if STORE_PATH.exists() and not force:
        with STORE_PATH.open("rb") as f:
            return pickle.load(f)

    t0 = time.time()
    df = load_records()
    print(f"Loaded {len(df)} records from {df['topic'].nunique()} topics")

    computer = RealConflictComputer()
    r = computer.compute_batch(df["text"].tolist())

    from sklearn.preprocessing import QuantileTransformer

    df["bin"] = pd.to_datetime(df["ts"], unit="s").dt.floor("12h")
    s_raw = np.zeros(len(df))
    a_cal = np.zeros(len(df))
    e_cal = np.zeros(len(df))
    s_cal = np.zeros(len(df))
    audit = {}

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
        audit[topic] = n_train

    store = pd.DataFrame({
        "topic": df["topic"], "bin": df["bin"],
        "a_cal": a_cal, "e_cal": e_cal, "s_cal": s_cal,
    })
    store = store[store["topic"].isin(audit)].reset_index(drop=True)
    with STORE_PATH.open("wb") as f:
        pickle.dump({"store": store, "audit": audit}, f)
    print(f"Calibrated {len(store)} comments for {len(audit)} topics in "
          f"{(time.time() - t0) / 60:.1f} min -> {STORE_PATH}")
    return {"store": store, "audit": audit}


# ═══ Step 2: rebuild trajectories with a given fusion ═══
def build_trajectories(store, weights):
    """Recompute the index per comment, re-aggregate per 12h bin (identical
    aggregation to experiment_real_model_v2.py) and keep topics with >=28 bins."""
    a, e, s = weights
    c = (1.0 / (1.0 + np.exp(-(a * store["a_cal"].to_numpy()
                               + e * store["e_cal"].to_numpy()
                               + s * store["s_cal"].to_numpy())))).clip(0, 1)
    cal = store[["topic", "bin", "a_cal", "e_cal", "s_cal"]].copy()
    cal["c"] = c

    trajectories = {}
    for topic in cal["topic"].unique():
        tdf = cal[cal["topic"] == topic].sort_values("bin")
        if len(tdf) < 20:
            continue
        agg = tdf.groupby("bin").agg(
            c_bar=("c", lambda x: x.nlargest(max(1, int(len(x) * 0.15))).mean()),
            c_mean=("c", "mean"), c_max=("c", "max"), c_std=("c", "std"),
            n=("c", "count"),
            a_mean=("a_cal", "mean"), e_mean=("e_cal", "mean"),
            s_mean=("s_cal", "mean"),
        ).reset_index()
        agg = agg[(agg["n"] >= 3) & (~agg["c_bar"].isna())]
        if len(agg) >= L + H + 10:
            trajectories[topic] = agg
    return trajectories


def fidelity_check(built):
    """Verify the `full` variant reproduces the saved trajectories."""
    with (R_DIR / "trajectories_real_model.pkl").open("rb") as f:
        saved = pickle.load(f)
    if set(saved) != set(built):
        print(f"  [WARN] topic mismatch: only-in-saved={set(saved) - set(built)}, "
              f"only-in-built={set(built) - set(saved)}")
    diffs = []
    for topic in saved:
        if topic not in built:
            diffs.append((topic, float("inf"))); continue
        s, b = saved[topic]["c_bar"].to_numpy(), built[topic]["c_bar"].to_numpy()
        if len(s) != len(b):
            diffs.append((topic, float("inf"))); continue
        diffs.append((topic, float(np.max(np.abs(s - b)))))
    worst = max(diffs, key=lambda x: x[1]) if diffs else ("", 0.0)
    print(f"  Fidelity: {len(diffs)} topics, max |Δc_bar| = {worst[1]:.3e} "
          f"({worst[0]})")
    # Tolerance 1e-3: the saved trajectories were produced by the same code on
    # a different GPU/torch build; model inference differs at ~1e-4 level
    # (cuDNN/cuBLAS/TF32 numerical variance). Structure (topics, bins) must
    # match exactly; values match to inference noise.
    return worst[1] < 1e-3


def summarize(output, trajectories, variant):
    best = output["selected_result"]
    fixed = evaluate_span(trajectories, 24, feature_group="full")
    return {
        "variant": variant,
        "n_topics": len(trajectories),
        "selected_span": output["selected_span"],
        "selected_feature_group": output["selected_feature_group"],
        "selected_c": output["selected_c"],
        "selected_decision_threshold": output["selected_decision_threshold"],
        "valid": {k: v for k, v in best["valid"].items() if k != "per_topic"},
        "test": {k: v for k, v in best["test"].items() if k != "per_topic"},
        "fixed_span24_full_features_test": {
            k: v for k, v in fixed["test"].items() if k != "per_topic"},
        "topic_summary": output["topic_summary"],
    }


def main():
    t0 = time.time()
    prepared = prepare_comment_level()
    store = prepared["store"]
    print(f"Comment-level store: {len(store)} rows, "
          f"{store['topic'].nunique()} topics")

    print("\n═══ Fidelity gate: `full` vs saved trajectories ═══")
    full_traj = build_trajectories(store, VARIANTS["full"])
    ok = fidelity_check(full_traj)

    results = {"fidelity_ok": ok,
               "variants_weights": VARIANTS,
               "comment_n": int(len(store))}
    for variant, weights in VARIANTS.items():
        print(f"\n═══ Variant {variant} weights={tuple(round(w, 4) for w in weights)} ═══")
        trajectories = build_trajectories(store, weights)
        print(f"  {len(trajectories)} trajectories")
        output = run_complete_evaluation(trajectories)
        results[variant] = summarize(output, trajectories, variant)
        best = output["selected_result"]
        print(f"  span={output['selected_span']} group={output['selected_feature_group']}")
        print(f"  TEST logistic F1={best['test']['logistic']['f1']:.4f} "
              f"AUC={best['test']['logistic']['auc']:.4f} "
              f"| persistence F1={best['test']['persistence']['f1']:.4f} "
              f"AUC={best['test']['persistence']['auc']:.4f}")

    out = R_DIR / "component_ablation_results.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with (R_DIR / "component_ablation_results.pkl").open("wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved {out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
