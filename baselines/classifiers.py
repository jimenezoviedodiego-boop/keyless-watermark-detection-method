"""Generic classifiers (LightGBM, MLP) on two representations of each answer, development CV only.

Question: does a generic classifier find the watermark from the text alone, without being told
about SynthID's structure (4-token context + next token)? Representations:
  bow   - the answer text as a TF-IDF bag of words (fit on the training folds only); no
          tokenizer, contexts or n-gram structure
  4+1   - binary hashed (4-token context, next token) events of the generator's tokens (2^22
          buckets), the events condz counts
Features seen in fewer than 5 training texts are dropped per fold.

The classifier (LightGBM, or an MLP with one hidden layer and early stopping on 10% of the
training texts) is trained on the training folds and scores the held-out fold, with the
prompt-grouped folds of one CV repeat (fold_r0 by default). With --tune, for each outer fold
the next fold is held out as validation, every setting in GRIDS is trained on the remaining
three folds and scored on it, and the setting with the best validation AUROC is refit on all
four training folds to score the outer fold. The sealed test split is not read.

Usage (from the repository root):
    python baselines/classifiers.py --dataset llama --model lgbm --tune
    python baselines/classifiers.py --dataset qwen --model mlp            # fixed settings
Writes runs/<dataset>/<model>_<tuned|fixed>_r<repeat>.json.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.pairs import DATASETS, groups, load_dev, make_data, report, run_dir  # noqa: E402

BUCKETS = 2**22
P1, P2 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xC2B2AE3D27D4EB4F)
FEATURES = {"bow": None, "4+1": 5}  # None = TF-IDF on the text; 5 = token 5-grams (4 context + 1)
PARAMS = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=10,
              colsample_bytree=0.3, subsample=0.8, subsample_freq=1, reg_lambda=1.0, verbose=-1, n_jobs=4)
MLP = dict(hidden_layer_sizes=(256,), alpha=1e-4, batch_size=64, learning_rate_init=1e-3, max_iter=100,
           early_stopping=True, validation_fraction=0.1, n_iter_no_change=5)
GRIDS = {
    "lgbm": [dict(num_leaves=nl, learning_rate=lr, min_child_samples=mc)
             for nl, lr, mc in itertools.product((15, 63), (0.03, 0.1), (5, 20))],
    "mlp": [dict(hidden_layer_sizes=(h,), alpha=a) for h, a in itertools.product((128, 256), (1e-4, 1e-2))],
}


def classifier(model, seed, cfg=None):
    if model == "lgbm":
        return lgb.LGBMClassifier(**{**PARAMS, **(cfg or {})}, random_state=seed)
    return MLPClassifier(**{**MLP, **(cfg or {})}, random_state=seed)


def ngram_matrix(ids, n):
    """Binary matrix (texts x hashed token n-grams)."""
    rows, cols = [], []
    for i, x in enumerate(ids):
        x = np.asarray(x, dtype=np.uint64)
        h = np.full(len(x) - n + 1, np.uint64(n))
        with np.errstate(over="ignore"):
            for j in range(n):
                h = h * P1 + x[j:len(x) - n + 1 + j] + P2
        f = np.unique(h % np.uint64(BUCKETS)).astype(np.int64)
        rows.append(np.full(len(f), i))
        cols.append(f)
    r, c = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(ids), BUCKETS))


def fit_score(model, cfg, seed, tr, te, texts, X, y):
    """Fit on texts in tr (features built from tr only), return scores for texts in te."""
    if X is None:
        vec = TfidfVectorizer(min_df=5)
        Xtr, Xte = vec.fit_transform(texts[tr]), vec.transform(texts[te])
    else:
        keep = np.flatnonzero(np.asarray((X[tr] > 0).sum(0)).ravel() >= 5)
        Xtr, Xte = X[tr][:, keep], X[te][:, keep]
    m = classifier(model, seed, cfg).fit(Xtr.astype(np.float32), y[tr])
    return m.predict_proba(Xte.astype(np.float32))[:, 1], Xtr.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--model", choices=("lgbm", "mlp"), required=True)
    ap.add_argument("--tune", action="store_true", help="select hyperparameters on a validation fold")
    ap.add_argument("--repeat", type=int, default=0)
    args = ap.parse_args()
    d = load_dev(args.dataset, text=True)
    gi = groups(d)
    D = make_data(d, f"fold_r{args.repeat}")  # labels, folds and the paired layout (wm texts, then twins)
    ids = list(d["watermarked.token_ids"]) + list(d["unwatermarked.token_ids"])
    texts = np.array(list(d["watermarked.text"]) + list(d["unwatermarked.text"]), dtype=object)
    path = run_dir(args.dataset) / f"{args.model}_{'tuned' if args.tune else 'fixed'}_r{args.repeat}.json"
    res = json.loads(path.read_text()) if path.exists() else []
    label = "LightGBM" if args.model == "lgbm" else "MLP"
    for name, n in FEATURES.items():
        if any(r.get("features") == name for r in res):
            continue
        t0 = time.time()
        X = None if n is None else ngram_matrix(ids, n)
        s, chosen = np.zeros(D.T), []
        for f in range(5):
            tr, te = D.fold != f, D.fold == f
            cfg = None
            if args.tune:
                g = (f + 1) % 5
                itr, val = tr & (D.fold != g), D.fold == g
                aucs = [roc_auc_score(D.y[val], fit_score(args.model, c, f, itr, val, texts, X, D.y)[0])
                        for c in GRIDS[args.model]]
                cfg = GRIDS[args.model][int(np.argmax(aucs))]
                chosen.append({"cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
                               "val_auc": round(float(max(aucs)), 4)})
            s[te], nf = fit_score(args.model, cfg, f, tr, te, texts, X, D.y)
            print(f"  {name} fold {f}: {nf} features" + (f", chosen {chosen[-1]}" if args.tune else ""), flush=True)
        r = report(f"{label}, {name}{', tuned' if args.tune else ''} (repeat {args.repeat})", s, D, gi)
        r.update(features=name, minutes=(time.time() - t0) / 60, chosen=chosen)
        res.append(r)
        path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
