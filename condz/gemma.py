"""condz on the public Gemma SynthID-Text pairs (Google DeepMind, human_eval.jsonl; key unknown).

3,000 pairs of Gemma-7B-IT answers (about 250 tokens), one pair per question. The texts are
tokenized with the Gemma tokenizer (unsloth/gemma-2-2b-it, an ungated copy). There is no
sealed test split: 5-fold cross-validation grouped by pair, nested selection over the same
48 settings as for Qwen and Llama. CIs resample pairs (= questions).

Usage (from the repository root; needs `transformers` for the tokenizer):
    python condz/gemma.py
Downloads data/gemma/human_eval.jsonl on first use; writes runs/gemma/results.json.
"""

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from condz.cv import grid_scorers  # noqa: E402
from core.ngram import FOLDS, SEED, Data, nested_select, wins  # noqa: E402
from core.pairs import ROOT, run_dir  # noqa: E402

URL = "https://raw.githubusercontent.com/google-deepmind/synthid-text/main/data/human_eval.jsonl"
TOKENIZER = "unsloth/gemma-2-2b-it"


def load_gemma():
    """Token ids of the n watermarked answers followed by their n unwatermarked twins."""
    raw = ROOT / "data" / "gemma" / "human_eval.jsonl"
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, raw)
    rows = [json.loads(l) for l in raw.read_text(encoding="utf-8").split("\n") if l.strip()]
    cache = run_dir("gemma") / "token_ids.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    texts = [r["watermarked_model_response"] for r in rows] + [r["unwatermarked_model_response"] for r in rows]
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    cache.write_bytes(pickle.dumps(ids))
    return ids


def main():
    ids = load_gemma()
    n = len(ids) // 2
    y, pair = np.r_[np.ones(n), np.zeros(n)], np.r_[np.arange(n), np.arange(n)]
    fold = np.zeros(2 * n, dtype=int)
    for f, (_, te) in enumerate(GroupKFold(FOLDS).split(ids, y, pair)):
        fold[te] = f
    D = Data(ids, fold)
    s, chosen = nested_select(D, grid_scorers(D), "auc")
    print(f"chosen per fold: {chosen}", flush=True)

    w = wins(s, n)
    rng = np.random.default_rng(SEED)
    draws = [rng.integers(0, n, n) for _ in range(2000)]
    paired = [w[k].mean() for k in draws]
    aucs = [roc_auc_score(np.r_[np.ones(n), np.zeros(n)], np.r_[s[k], s[n + k]]) for k in draws]
    res = {"pairs": n, "chosen": chosen,
           "auc": float(roc_auc_score(D.y, s)), "auc_ci95": [float(np.percentile(aucs, q)) for q in (2.5, 97.5)],
           "paired_accuracy": float(w.mean()), "paired_ci95": [float(np.percentile(paired, q)) for q in (2.5, 97.5)]}
    print(f"Gemma, condz nested ({n} pairs): AUROC {res['auc']:.3f} [{res['auc_ci95'][0]:.3f},{res['auc_ci95'][1]:.3f}]"
          f"  paired {res['paired_accuracy']:.3f} [{res['paired_ci95'][0]:.3f},{res['paired_ci95'][1]:.3f}]", flush=True)
    (run_dir("gemma") / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
