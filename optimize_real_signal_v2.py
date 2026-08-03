#!/usr/bin/env python3
"""Leakage-resistant cleaning and forecasting for the real Zhihu trajectories.

The script regularizes the 12-hour time axis, fills missing bins with a
training-only topic median, applies causal EWMA stabilization, and predicts
whether the next six bins exceed the topic's training-period 80th percentile.
Hyperparameters are selected on a 60--75% validation segment; the final 25%
is used once for testing.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score)


# ═══ Configuration ═══
L, H = 12, 6
TRAIN_END, VALID_END = 0.60, 0.75
QUANTILE = 0.80
MIN_FUTURE_OBS = 2
SPAN_CANDIDATES = (12, 18, 24, 30, 36, 48)
C_CANDIDATES = (0.01, 0.03, 0.1, 0.3, 1.0)
DECISION_THRESHOLDS = tuple(np.arange(0.30, 0.71, 0.05))
QUANTILE_CANDIDATES = (0.70, 0.80, 0.90)
FEATURE_COLUMNS = ("c_bar", "c_mean", "c_max", "n",
                   "a_mean", "e_mean", "s_mean")
FEATURE_GROUPS = {
    "state_only": (7,),
    "state_activity": (3, 7, 8),
    "state_semantic": (0, 1, 2, 4, 5, 6, 7, 8),
    "full": tuple(range(9)),
}
RESULT_DIR = Path("experiment_results_v2")


# ═══ Dataset Construction ═══
def build_windows(trajectories, span, quantile=QUANTILE):
    splits = {"train": [], "valid": [], "test": []}

    for topic, trajectory in trajectories.items():
        frame = trajectory.copy()
        frame["bin"] = pd.to_datetime(frame["bin"])
        frame = frame.set_index("bin").sort_index()
        full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="12h")
        frame = frame[list(FEATURE_COLUMNS)].reindex(full_index)
        observed = frame["c_bar"].notna().astype(float)
        train_end = int(len(frame) * TRAIN_END)
        valid_end = int(len(frame) * VALID_END)

        # All cleaning parameters are learned from the training period only.
        for column in FEATURE_COLUMNS:
            train_values = frame[column].iloc[:train_end].dropna()
            if train_values.empty:
                break
            lower, upper = train_values.quantile([0.01, 0.99])
            frame[column] = frame[column].clip(lower, upper).fillna(
                train_values.median())
        else:
            frame["n"] = np.log1p(frame["n"])
            raw_target = frame["c_bar"]
            target = raw_target.ewm(span=span, adjust=False).mean()
            threshold = target.iloc[:train_end].quantile(quantile)
            frame["target"] = target
            frame["observed"] = observed

            target_mean = target.iloc[:train_end].mean()
            target_std = target.iloc[:train_end].std() or 1.0
            threshold_z = (threshold - target_mean) / target_std
            for column in frame.columns:
                mean = frame[column].iloc[:train_end].mean()
                std = frame[column].iloc[:train_end].std() or 1.0
                frame[column] = (frame[column] - mean) / std

            values = frame.to_numpy()
            target_col = frame.columns.get_loc("target")
            observed_values = observed.to_numpy()
            for start in range(len(frame) - L - H + 1):
                target_start = start + L
                target_end = target_start + H
                if observed_values[target_start:target_end].sum() < MIN_FUTURE_OBS:
                    continue
                future_raw = target.iloc[target_start:target_end]
                item = {
                    "topic": topic,
                    "x": values[start:target_start].reshape(-1),
                    "y_reg": values[target_start:target_end, target_col],
                    "y_cls": int(future_raw.max() > threshold),
                    "last_target": values[target_start - 1, target_col],
                    "threshold": threshold_z,
                }
                if target_end <= train_end:
                    splits["train"].append(item)
                elif target_start >= train_end and target_end <= valid_end:
                    splits["valid"].append(item)
                elif target_start >= valid_end:
                    splits["test"].append(item)

    return splits


def arrays(items):
    X = np.asarray([item["x"] for item in items])
    y_cls = np.asarray([item["y_cls"] for item in items])
    y_reg = np.asarray([item["y_reg"] for item in items])
    return X, y_cls, y_reg


def select_features(X, feature_group):
    """Select time-major feature columns from the flattened L x 9 input."""
    columns = FEATURE_GROUPS[feature_group]
    return X.reshape(len(X), L, -1)[:, :, columns].reshape(len(X), -1)


# ═══ Evaluation ═══
def classification_metrics(y_true, probability, prediction):
    return {
        "auc": float(roc_auc_score(y_true, probability)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }


def topic_bootstrap(items, y_true, probability, prediction,
                    persistence_score, persistence_prediction, n_boot=1000):
    """Bootstrap paired metric differences by resampling whole topics."""
    topics = np.asarray([item["topic"] for item in items])
    unique_topics = np.unique(topics)
    rng = np.random.RandomState(42)
    differences = {"f1": [], "recall": [], "auc": []}
    for _ in range(n_boot):
        sampled = rng.choice(unique_topics, len(unique_topics), replace=True)
        indices = np.concatenate([np.where(topics == topic)[0] for topic in sampled])
        y_sample = y_true[indices]
        if len(np.unique(y_sample)) < 2:
            continue
        differences["f1"].append(
            f1_score(y_sample, prediction[indices], zero_division=0) -
            f1_score(y_sample, persistence_prediction[indices], zero_division=0))
        differences["recall"].append(
            recall_score(y_sample, prediction[indices], zero_division=0) -
            recall_score(y_sample, persistence_prediction[indices], zero_division=0))
        differences["auc"].append(
            roc_auc_score(y_sample, probability[indices]) -
            roc_auc_score(y_sample, persistence_score[indices]))
    return {
        metric: {
            "mean_difference": float(np.mean(values)),
            "ci95": [float(value) for value in np.percentile(values, [2.5, 97.5])],
        }
        for metric, values in differences.items()
    }


def fit_selected_classifier(splits, feature_group="full"):
    """Select regularization and decision threshold using validation F1 only."""
    X_train, y_train, _ = arrays(splits["train"])
    X_valid, y_valid, _ = arrays(splits["valid"])
    X_train = select_features(X_train, feature_group)
    X_valid = select_features(X_valid, feature_group)
    candidates = []
    for c_value in C_CANDIDATES:
        classifier = LogisticRegression(
            C=c_value, max_iter=2000, class_weight="balanced", random_state=42)
        classifier.fit(X_train, y_train)
        probability = classifier.predict_proba(X_valid)[:, 1]
        for decision_threshold in DECISION_THRESHOLDS:
            score = f1_score(
                y_valid, probability >= decision_threshold, zero_division=0)
            candidates.append((score, c_value, decision_threshold, classifier))
    _, c_value, decision_threshold, classifier = max(
        candidates, key=lambda candidate: (candidate[0], -candidate[1]))
    return classifier, float(c_value), float(decision_threshold)


def per_topic_metrics(items, y, probability, prediction,
                      persistence_score, persistence_prediction):
    topics = np.asarray([item["topic"] for item in items])
    output = {}
    for topic in np.unique(topics):
        idx = np.where(topics == topic)[0]
        topic_y = y[idx]
        result = {
            "n": int(len(idx)),
            "prevalence": float(topic_y.mean()),
            "logistic_f1": float(f1_score(
                topic_y, prediction[idx], zero_division=0)),
            "persistence_f1": float(f1_score(
                topic_y, persistence_prediction[idx], zero_division=0)),
            "logistic_recall": float(recall_score(
                topic_y, prediction[idx], zero_division=0)),
            "persistence_recall": float(recall_score(
                topic_y, persistence_prediction[idx], zero_division=0)),
        }
        if len(np.unique(topic_y)) == 2:
            result["logistic_auc"] = float(roc_auc_score(topic_y, probability[idx]))
            result["persistence_auc"] = float(
                roc_auc_score(topic_y, persistence_score[idx]))
        output[topic] = result
    return output


def evaluate_span(trajectories, span, bootstrap=False, quantile=QUANTILE,
                  feature_group="full", tune_classifier=True,
                  include_test=True):
    splits = build_windows(trajectories, span, quantile=quantile)
    X_train_all, y_train, y_reg_train = arrays(splits["train"])
    if tune_classifier:
        classifier, c_value, decision_threshold = fit_selected_classifier(
            splits, feature_group)
    else:
        c_value, decision_threshold = 0.1, 0.5
        classifier = LogisticRegression(
            C=c_value, max_iter=2000, class_weight="balanced", random_state=42)
        classifier.fit(select_features(X_train_all, feature_group), y_train)

    ridge_models = [
        Ridge(alpha=10.0).fit(X_train_all, y_reg_train[:, horizon])
        for horizon in range(H)
    ]
    results = {
        "span": span,
        "quantile": quantile,
        "feature_group": feature_group,
        "selected_c": c_value,
        "selected_decision_threshold": decision_threshold,
        "n_train": len(splits["train"]),
    }
    evaluated_splits = ("valid", "test") if include_test else ("valid",)
    for split_name in evaluated_splits:
        items = splits[split_name]
        X_all, y, y_reg = arrays(items)
        X = select_features(X_all, feature_group)
        probability = classifier.predict_proba(X)[:, 1]
        prediction = probability >= decision_threshold
        last_target = np.asarray([item["last_target"] for item in items])
        thresholds = np.asarray([item["threshold"] for item in items])
        persistence_prediction = last_target > thresholds
        results[split_name] = {
            "n": len(items),
            "prevalence": float(y.mean()),
            "logistic": classification_metrics(y, probability, prediction),
            "persistence": classification_metrics(
                y, last_target, persistence_prediction),
        }
        if split_name == "test":
            results[split_name]["per_topic"] = per_topic_metrics(
                items, y, probability, prediction,
                last_target, persistence_prediction)
            if bootstrap:
                results[split_name]["topic_bootstrap_logistic_minus_persistence"] = (
                    topic_bootstrap(items, y, probability, prediction,
                                    last_target, persistence_prediction))

        regression_prediction = np.stack(
            [model.predict(X_all) for model in ridge_models], axis=1)
        persistence_regression = np.repeat(last_target[:, None], H, axis=1)
        results[split_name]["ridge"] = {
            "r2": float(r2_score(y_reg.ravel(), regression_prediction.ravel())),
            "mae": float(mean_absolute_error(
                y_reg.ravel(), regression_prediction.ravel())),
        }
        results[split_name]["persistence_regression"] = {
            "r2": float(r2_score(y_reg.ravel(), persistence_regression.ravel())),
            "mae": float(mean_absolute_error(
                y_reg.ravel(), persistence_regression.ravel())),
        }
    return results


def select_span(trajectories, quantile=QUANTILE, feature_group="full"):
    sensitivity = [
        evaluate_span(trajectories, span, quantile=quantile,
                      feature_group=feature_group, include_test=False)
        for span in SPAN_CANDIDATES
    ]
    selected = max(
        sensitivity, key=lambda result: result["valid"]["logistic"]["f1"])
    return selected["span"], sensitivity


def run_complete_evaluation(trajectories):
    """Run validation-only model selection and final held-out evaluation."""
    selected_span, sensitivity = select_span(trajectories)
    feature_ablation = {}
    for feature_group in FEATURE_GROUPS:
        feature_ablation[feature_group] = evaluate_span(
            trajectories, selected_span, quantile=QUANTILE,
            feature_group=feature_group, include_test=False)

    # Prefer the smallest feature set within 0.005 validation F1 of the best.
    best_validation_f1 = max(
        result["valid"]["logistic"]["f1"]
        for result in feature_ablation.values())
    eligible_groups = [
        group for group, result in feature_ablation.items()
        if result["valid"]["logistic"]["f1"] >= best_validation_f1 - 0.005
    ]
    selected_feature_group = min(
        eligible_groups, key=lambda group: len(FEATURE_GROUPS[group]))
    best = evaluate_span(
        trajectories, selected_span, feature_group=selected_feature_group,
        bootstrap=True)

    threshold_sensitivity = {
        str(quantile): evaluate_span(
            trajectories, selected_span, quantile=quantile,
            feature_group=selected_feature_group)
        for quantile in QUANTILE_CANDIDATES
    }

    per_topic = best["test"]["per_topic"]
    f1_improved_topics = sum(
        result["logistic_f1"] > result["persistence_f1"]
        for result in per_topic.values())
    recall_improved_topics = sum(
        result["logistic_recall"] > result["persistence_recall"]
        for result in per_topic.values())
    output = {
        "selection_rule": (
            "maximum validation F1 for span/C/decision threshold; then smallest "
            "feature set within 0.005 validation F1 of the best ablation"),
        "selected_span": best["span"],
        "selected_feature_group": selected_feature_group,
        "selected_c": best["selected_c"],
        "selected_decision_threshold": best["selected_decision_threshold"],
        "test_was_not_used_for_selection": True,
        "sensitivity": sensitivity,
        "threshold_sensitivity": threshold_sensitivity,
        "feature_ablation": feature_ablation,
        "topic_summary": {
            "n_topics": len(per_topic),
            "f1_improved_topics": f1_improved_topics,
            "recall_improved_topics": recall_improved_topics,
        },
        "selected_result": best,
    }
    return output


def main():
    trajectory_path = RESULT_DIR / "trajectories_real_model.pkl"
    with trajectory_path.open("rb") as handle:
        trajectories = pickle.load(handle)

    output = run_complete_evaluation(trajectories)
    best = output["selected_result"]
    selected_feature_group = output["selected_feature_group"]

    RESULT_DIR.mkdir(exist_ok=True)
    with (RESULT_DIR / "signal_cleaning_results.pkl").open("wb") as handle:
        pickle.dump(output, handle)
    with (RESULT_DIR / "signal_cleaning_results.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    print(json.dumps({
        "selected_span": best["span"],
        "selected_feature_group": selected_feature_group,
        "selected_c": best["selected_c"],
        "selected_decision_threshold": best["selected_decision_threshold"],
        "topic_summary": output["topic_summary"],
        "validation": best["valid"],
        "test": {key: value for key, value in best["test"].items()
                 if key != "per_topic"},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
