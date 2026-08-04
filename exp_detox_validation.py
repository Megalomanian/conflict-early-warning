#!/usr/bin/env python3
"""Wikipedia Detox construct-validity check (English side, fully automatic).

Scores the SAME multilingual sentiment model used by the conflict index
(nlptown/bert-base-multilingual-uncased-sentiment) on the Wikipedia Detox
personal-attacks labels and reports how well the attack component
(p(1-star) + 0.5*p(2-star)) tracks human annotations:

  - Spearman / Pearson correlation with the human label
  - AUC of the attack component as a binary personal-attack predictor

A second multilingual sentiment model (cardiffnlp/twitter-xlm-roberta-base-sentiment)
is included as a robustness reference when it can be downloaded.
"""
import json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
R_DIR = Path("experiment_results_v2")
R_DIR.mkdir(exist_ok=True)
MAX_N_CPU = 12000  # stratified subsample when running without GPU

LABEL_COLUMNS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")


def load_detox():
    """Load human-labeled Wikipedia Detox data (wiki_toxic CSV mirror of the
    personal-attacks annotations, or the original TSVs if present)."""
    frames = []
    for p in ("wikipedia_detox/wiki_toxic_train.csv",
              "wikipedia_detox/wiki_toxic_test.csv"):
        if Path(p).exists():
            frames.append(pd.read_csv(p))
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.rename(columns={"comment_text": "text"})
        if "label" in df.columns and "toxic" not in df.columns:
            df = df.rename(columns={"label": "toxic"})
        return df

    # Original Detox TSV layout fallback
    for p in ("wikipedia_detox/attack_annotated_comments.tsv",
              "wikipedia_detox/attack_annotations.tsv"):
        if Path(p).exists() and "403" not in Path(p).read_text(
                encoding="utf-8", errors="ignore")[:200]:
            ann = pd.read_csv("wikipedia_detox/attack_annotations.tsv",
                              sep="\t", low_memory=False)
            com = pd.read_csv("wikipedia_detox/attack_annotated_comments.tsv",
                              sep="\t", low_memory=False)
            agg = ann.groupby("rev_id")["attack"].mean().reset_index()
            df = com.merge(agg, on="rev_id")
            df["text"] = df["comment"]
            df["toxic"] = (df["attack"] >= 0.5).astype(int)
            return df

    raise SystemExit(
        "No Detox data found under wikipedia_detox/ — run the server setup "
        "script to download wiki_toxic train/test CSVs first.")


def sentiment_probs(texts, model_name, batch_size=128):
    """Return softmax probabilities over star classes, same code path as the
    conflict-index pipeline (max_length=256)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=256).to(DEVICE)
            logits = model(**inputs).logits.cpu().numpy()
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def attack_from_stars(probs):
    """Attack proxy used in the conflict index: 1-star + half of 2-star."""
    return probs[:, 0] + 0.5 * probs[:, 1]


def evaluate(df, attack, label_col):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score
    y = df[label_col].to_numpy(dtype=float)
    valid = ~np.isnan(y)
    y, attack = y[valid], attack[valid]
    result = {"n": int(len(y)), "positive_rate": float(y.mean())}
    if len(np.unique(y)) < 2 or len(y) < 20:
        return result
    rho, rho_p = spearmanr(attack, y)
    r_pearson, r_p = pearsonr(attack, y)
    result.update({
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "pearson_r": float(r_pearson),
        "pearson_p": float(r_p),
        "auc": float(roc_auc_score(y, attack)),
        "attack_mean_label1": float(attack[y > 0.5].mean()),
        "attack_mean_label0": float(attack[y <= 0.5].mean()),
    })
    return result


def main():
    t0 = time.time()
    df = load_detox()
    print(f"Loaded {len(df)} labeled comments (cols: {list(df.columns)[:8]}...)")

    texts = df["text"].fillna("").astype(str).tolist()
    if DEVICE == "cpu" and len(texts) > MAX_N_CPU:
        rng = np.random.RandomState(0)
        pos = df["toxic"].fillna(0).to_numpy().astype(int)
        n_pos = int(MAX_N_CPU * max(pos.mean(), 0.05))
        idx_pos = rng.choice(np.where(pos == 1)[0], min(n_pos, (pos == 1).sum()), replace=False)
        idx_neg = rng.choice(np.where(pos == 0)[0], MAX_N_CPU - len(idx_pos), replace=False)
        keep = np.sort(np.concatenate([idx_pos, idx_neg]))
        df, texts = df.iloc[keep].reset_index(drop=True), [texts[i] for i in keep]
        print(f"CPU mode: stratified subsample of {len(df)} comments")

    results = {"device": DEVICE, "n_comments": len(df), "source": "wiki_toxic (Wikipedia Detox)"}

    models = ["nlptown/bert-base-multilingual-uncased-sentiment",
              "cardiffnlp/twitter-xlm-roberta-base-sentiment"]
    for model_name in models:
        try:
            print(f"\nScoring {model_name} on {len(texts)} texts ...", flush=True)
            probs = sentiment_probs(texts, model_name)
        except Exception as exc:
            print(f"  SKIP {model_name}: {exc}", flush=True)
            continue
        attack = attack_from_stars(probs)
        results[model_name] = {"attack_mean": float(attack.mean()),
                               "attack_std": float(attack.std())}
        for label in LABEL_COLUMNS:
            if label in df.columns:
                results[model_name][label] = evaluate(df, attack, label)

    out_path = R_DIR / "detox_validation.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {out_path} in {(time.time() - t0) / 60:.1f} min")
    for model_name in list(results):
        if model_name.startswith(("nlptown", "cardiffnlp")):
            row = results[model_name].get("toxic", {})
            print(f"  {model_name.split('/')[-1]:35s} "
                  f"toxic n={row.get('n')} spearman={row.get('spearman_rho', float('nan')):.3f} "
                  f"auc={row.get('auc', float('nan')):.3f}")


if __name__ == "__main__":
    main()
