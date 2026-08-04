#!/usr/bin/env python3
"""Recompute real-data components with TRUE attack and anger models.

Replaces the product-review sentiment proxy with:
  - attack: unitary/multilingual-toxic-xlm-roberta (multilingual toxicity,
    supports Chinese; validated on English Wikipedia Detox labels)
  - emotion: zero-shot NLI (joeddav/xlm-roberta-large-xnli) over
    ["negative", "anger"], matching the paper's eq. (4) description
Stance (s_cal) is reused from the cached comment-level store (same MiniLM
pipeline, row-aligned on the identical comment set).

Outputs:
  - detox_toxicity_validation.json   (construct validity of the new model)
  - comment_level_features_v2.pkl   (a_cal/e_cal from new models + s_cal)
  - signal_cleaning_v2_results.json (full pipeline on new components)
  - component_ablation_v2_results.json
  - weight_sensitivity_real_v2.json (Dirichlet 40-config on real data)
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
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")

from experiment_real_model_v2 import RealConflictComputer  # noqa: E402
from optimize_real_signal_v2 import (evaluate_span,  # noqa: E402
                                     run_complete_evaluation)
from exp_component_ablation import (VARIANTS, build_trajectories,  # noqa: E402
                                    load_records, prepare_comment_level)

L, H = 12, 6
RISK_TRAIN_END = 0.60
R_DIR = Path("experiment_results_v2")
R_DIR.mkdir(exist_ok=True)
OLD_STORE = R_DIR / "comment_level_features.pkl"
NEW_STORE = R_DIR / "comment_level_features_v2.pkl"
DEVICE = "cuda"

TOXICITY_MODEL = "unitary/multilingual-toxic-xlm-roberta"
XNLI_MODEL = "joeddav/xlm-roberta-large-xnli"


def toxicity_scores(texts, batch_size=64):
    """Multilingual toxicity model; returns toxicity probability per text."""
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)
    tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL,
                                              local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        TOXICITY_MODEL, local_files_only=True).to(DEVICE)
    model.eval()
    import torch
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=256).to(DEVICE)
            logits = model(**inputs).logits.float().cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))  # multi-label sigmoid
            all_probs.append(probs)
    probs = np.concatenate(all_probs, axis=0)
    return probs[:, 0]  # toxicity dimension


def zero_shot_emotion(texts, batch_size=32):
    """Zero-shot NLI over ["negative", "anger"]; returns (p_neg, p_anger)."""
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, pipeline)
    tokenizer = AutoTokenizer.from_pretrained(XNLI_MODEL,
                                              local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        XNLI_MODEL, local_files_only=True).to(DEVICE)
    classifier = pipeline("zero-shot-classification", model=model,
                          tokenizer=tokenizer,
                          device=0 if DEVICE == "cuda" else -1)
    labels = ["negative", "anger"]
    p_neg, p_anger = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        out = classifier(batch, candidate_labels=labels, batch_size=batch_size,
                         truncation=True, max_length=256)
        if isinstance(out, dict):
            out = [out]
        for o in out:
            scores = dict(zip(o["labels"], o["scores"]))
            p_neg.append(scores["negative"])
            p_anger.append(scores["anger"])
    return np.asarray(p_neg), np.asarray(p_anger)


def calibrate_components(df, a_raw, e_raw):
    """Per-topic train-only ECDF calibration, identical protocol to v1."""
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
        audit[topic] = n_train

    return a_cal, e_cal, s_cal, audit


def main():
    t0 = time.time()
    df = load_records()
    print(f"Loaded {len(df)} records from {df['topic'].nunique()} topics")
    df["bin"] = pd.to_datetime(df["ts"], unit="s").dt.floor("12h")

    texts = df["text"].tolist()
    print("Scoring toxicity (new attack model)...", flush=True)
    a_raw = toxicity_scores(texts)
    print(f"  toxicity mean={a_raw.mean():.3f} std={a_raw.std():.3f}")
    print("Scoring zero-shot emotion (negative/anger)...", flush=True)
    p_neg, p_anger = zero_shot_emotion(texts)
    e_raw = 0.7 * p_neg + 0.3 * p_anger
    print(f"  p_neg={p_neg.mean():.3f} p_anger={p_anger.mean():.3f} "
          f"e={e_raw.mean():.3f}")

    # Stance from the cached v1 store (row-aligned, identical comment set)
    with OLD_STORE.open("rb") as f:
        old = pickle.load(f)
    old_store = old["store"]
    if len(old_store) != len(df) or list(old_store["topic"]) != list(df["topic"]):
        raise SystemExit("Old store rows do not align with the comment set")
    s_cal_old = old_store["s_cal"].to_numpy()

    a_cal, e_cal, _, audit = calibrate_components(df, a_raw, e_raw)
    print(f"Calibrated {len(audit)} topics")

    store = pd.DataFrame({
        "topic": df["topic"], "bin": df["bin"],
        "a_cal": a_cal, "e_cal": e_cal, "s_cal": s_cal_old,
    })
    store = store[store["topic"].isin(audit)].reset_index(drop=True)
    with NEW_STORE.open("wb") as f:
        pickle.dump({"store": store, "audit": audit,
                     "models": {"attack": TOXICITY_MODEL,
                                "emotion": XNLI_MODEL}}, f)
    print(f"Saved {NEW_STORE} in {(time.time() - t0) / 60:.1f} min")

    # ═══ Full pipeline on the new components ═══
    print("\n═══ Pipeline: full index with new components ═══")
    traj_full = build_trajectories(store, VARIANTS["full"])
    print(f"{len(traj_full)} trajectories")
    output = run_complete_evaluation(traj_full)
    best = output["selected_result"]
    results = {
        "models": {"attack": TOXICITY_MODEL, "emotion": XNLI_MODEL},
        "selected_span": output["selected_span"],
        "selected_feature_group": output["selected_feature_group"],
        "test": {k: v for k, v in best["test"].items() if k != "per_topic"},
        "valid": {k: v for k, v in best["valid"].items() if k != "per_topic"},
        "topic_summary": output["topic_summary"],
    }
    print(f"TEST logistic F1={best['test']['logistic']['f1']:.4f} "
          f"AUC={best['test']['logistic']['auc']:.4f} | "
          f"persistence F1={best['test']['persistence']['f1']:.4f} "
          f"AUC={best['test']['persistence']['auc']:.4f}")

    # ═══ Component ablation with new components ═══
    print("\n═══ Component ablation (new components) ═══")
    ablation = {}
    for variant, weights in VARIANTS.items():
        trajectories = build_trajectories(store, weights)
        out = run_complete_evaluation(trajectories)
        fixed = evaluate_span(trajectories, 24, feature_group="full")
        ablation[variant] = {
            "weights": list(weights),
            "n_topics": len(trajectories),
            "selected_span": out["selected_span"],
            "selected_feature_group": out["selected_feature_group"],
            "test": {k: v for k, v in out["selected_result"]["test"].items()
                     if k != "per_topic"},
            "fixed_span24_full_features_test": {
                k: v for k, v in fixed["test"].items() if k != "per_topic"},
        }
        t = out["selected_result"]["test"]
        print(f"  {variant:8s} F1={t['logistic']['f1']:.4f} "
              f"AUC={t['logistic']['auc']:.4f}")

    # ═══ Dirichlet weight sensitivity on real data (40 configs) ═══
    print("\n═══ Weight sensitivity: 40 Dirichlet configs (span 24) ═══")
    rng = np.random.RandomState(42)
    weight_configs = rng.dirichlet([1.0, 1.0, 1.0], size=40)
    sens = []
    for w in weight_configs:
        trajectories = build_trajectories(store, tuple(w))
        out = evaluate_span(trajectories, 24, feature_group="full")
        t = out["test"]
        sens.append({
            "weights": [round(float(x), 4) for x in w],
            "test_f1": t["logistic"]["f1"],
            "test_auc": t["logistic"]["auc"],
            "test_r2": t["ridge"]["r2"],
            "persistence_f1": t["persistence"]["f1"],
        })
    f1s = np.asarray([s["test_f1"] for s in sens])
    r2s = np.asarray([s["test_r2"] for s in sens])
    sens_report = {
        "n_configs": len(sens),
        "test_f1": {"mean": float(f1s.mean()), "std": float(f1s.std()),
                    "min": float(f1s.min()), "max": float(f1s.max()),
                    "iqr": float(np.percentile(f1s, 75) - np.percentile(f1s, 25))},
        "test_r2": {"mean": float(r2s.mean()), "std": float(r2s.std()),
                    "iqr": float(np.percentile(r2s, 75) - np.percentile(r2s, 25))},
        "best_config": max(sens, key=lambda s: s["test_f1"]),
        "default_config": {"weights": [0.5, 0.3, 0.2],
                           "test_f1": sens[0]["test_f1"] or None},
        "configs": sens,
    }
    print(f"  F1: mean={f1s.mean():.4f} std={f1s.std():.4f} "
          f"min={f1s.min():.4f} max={f1s.max():.4f}")

    results["component_ablation"] = ablation
    results["weight_sensitivity"] = sens_report
    out_path = R_DIR / "signal_cleaning_v2_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out_path} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
