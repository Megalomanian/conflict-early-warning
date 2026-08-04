#!/usr/bin/env python3
"""Topic-held-out (Zhihu) and event-held-out (Weibo) generalization.

The main pipeline evaluates on time-held-out windows pooled from all topics.
Here we evaluate the SAME classifier protocol under topic/event-held-out
splits: the logistic model is trained on windows from a subset of
topics/events and tested on windows from topics/events never seen in
training, with regularization and decision threshold selected on a
disjoint held-out set of topics/events (5-fold rotation).

Metrics and persistence baseline are identical to evaluate_span.
"""
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from exp_component_ablation import VARIANTS, build_trajectories  # noqa: E402
from exp_real_components_v2 import NEW_STORE  # noqa: E402
from optimize_real_signal_v2 import (arrays, build_windows,  # noqa: E402
                                     classification_metrics,
                                     fit_selected_classifier,
                                     per_topic_metrics, select_features,
                                     topic_bootstrap)
from exp_weibo_generalization import BIN_FREQ  # noqa: E402

R_DIR = Path("experiment_results_v2")
N_FOLDS = 5
SPAN = 24
WEIBO_SPAN = 36  # span selected by the main Weibo run
FEATURE_GROUP = "full"


def heldout_folds(topic_items, n_folds=N_FOLDS, seed=0):
    """Yield (train, valid, test) item lists per fold, topics disjoint."""
    topics = sorted(topic_items.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(topics)
    per_fold = len(topics) // n_folds
    folds = []
    for f in range(n_folds):
        test_topics = topics[f * per_fold:(f + 1) * per_fold]
        rest = [t for t in topics if t not in test_topics]
        valid_topics = rest[: max(1, len(rest) // 5)]
        train_topics = [t for t in rest if t not in valid_topics]
        def gather(ts):
            return [item for t in ts for item in topic_items[t]]
        folds.append((train_topics, valid_topics, test_topics,
                      gather(train_topics), gather(valid_topics),
                      gather(test_topics)))
    return folds


def evaluate_fold(splits):
    classifier, c_value, decision_threshold = fit_selected_classifier(
        splits, FEATURE_GROUP)
    items = splits["test"]
    X_all, y, _ = arrays(items)
    X = select_features(X_all, FEATURE_GROUP)
    probability = classifier.predict_proba(X)[:, 1]
    prediction = probability >= decision_threshold
    last_target = np.asarray([item["last_target"] for item in items])
    thresholds = np.asarray([item["threshold"] for item in items])
    persistence_prediction = last_target > thresholds
    metrics = {
        "n": len(items),
        "prevalence": float(y.mean()),
        "logistic": classification_metrics(y, probability, prediction),
        "persistence": classification_metrics(y, last_target,
                                              persistence_prediction),
        "per_topic": per_topic_metrics(
            items, y, probability, prediction, last_target,
            persistence_prediction),
        "bootstrap": topic_bootstrap(items, y, probability, prediction,
                                     last_target, persistence_prediction),
    }
    return metrics


def summarize(name, fold_results, trajectories, freq):
    out = {"name": name, "n_units": len(trajectories),
           "n_folds": len(fold_results), "freq": freq,
           "feature_group": FEATURE_GROUP}
    for key in ("auc", "f1", "recall", "precision"):
        vals = [r["logistic"][key] for r in fold_results]
        out[f"logistic_{key}"] = {"mean": float(np.mean(vals)),
                                  "std": float(np.std(vals)),
                                  "per_fold": [float(v) for v in vals]}
    for key in ("auc", "f1", "recall", "precision"):
        vals = [r["persistence"][key] for r in fold_results]
        out[f"persistence_{key}"] = {"mean": float(np.mean(vals)),
                                     "std": float(np.std(vals))}
    boot_diffs = [r["bootstrap"]["f1"]["mean_difference"] for r in fold_results]
    out["bootstrap_f1_diff"] = {"mean": float(np.mean(boot_diffs)),
                                "std": float(np.std(boot_diffs)),
                                "per_fold": [float(v) for v in boot_diffs]}
    out["n_test_total"] = int(sum(r["n"] for r in fold_results))
    return out


def run_zhihu():
    with NEW_STORE.open("rb") as f:
        prepared = pickle.load(f)
    trajectories = build_trajectories(prepared["store"], VARIANTS["full"])
    print(f"Zhihu trajectories: {len(trajectories)}")
    all_windows = build_windows(trajectories, SPAN)
    topic_items = {}
    for split_name in ("train", "valid", "test"):
        for item in all_windows[split_name]:
            topic_items.setdefault(item["topic"], []).append(item)
    folds = heldout_folds(topic_items)
    fold_results = []
    for f, (tr_t, va_t, te_t, tr, va, te) in enumerate(folds):
        splits = {"train": tr, "valid": va, "test": te}
        res = evaluate_fold(splits)
        fold_results.append(res)
        print(f"  fold {f}: test n={res['n']} (topics: {te_t}) "
              f"logF1={res['logistic']['f1']:.4f} persF1={res['persistence']['f1']:.4f}")
    return summarize("zhihu_topic_heldout", fold_results, trajectories, "12h")


def run_weibo():
    from exp_weibo_v2 import R_DIR as _R
    with (R_DIR / "weibo_trajectories.pkl").open("rb") as f:
        cached = pickle.load(f)
    trajectories = cached["trajectories"]
    print(f"Weibo trajectories: {len(trajectories)}")
    all_windows = build_windows(trajectories, WEIBO_SPAN, freq=BIN_FREQ)
    topic_items = {}
    for split_name in ("train", "valid", "test"):
        for item in all_windows[split_name]:
            topic_items.setdefault(item["topic"], []).append(item)
    folds = heldout_folds(topic_items)
    fold_results = []
    for f, (tr_t, va_t, te_t, tr, va, te) in enumerate(folds):
        splits = {"train": tr, "valid": va, "test": te}
        res = evaluate_fold(splits)
        fold_results.append(res)
        print(f"  fold {f}: test n={res['n']} "
              f"logF1={res['logistic']['f1']:.4f} persF1={res['persistence']['f1']:.4f}")
    return summarize("weibo_event_heldout", fold_results, trajectories, BIN_FREQ)


def main():
    zhihu = run_zhihu()
    weibo = run_weibo()
    out = {"zhihu_topic_heldout": zhihu, "weibo_event_heldout": weibo}
    path = R_DIR / "heldout_results.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {path}")
    for name, res in out.items():
        print(f"{name}: logF1={res['logistic_f1']['mean']:.4f}±{res['logistic_f1']['std']:.4f} "
              f"| persF1={res['persistence_f1']['mean']:.4f} | "
              f"logAUC={res['logistic_auc']['mean']:.4f} "
              f"| ΔF1(boot)={res['bootstrap_f1_diff']['mean']:+.4f}")


if __name__ == "__main__":
    main()
