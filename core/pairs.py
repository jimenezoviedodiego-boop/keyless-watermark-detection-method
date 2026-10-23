"""Loading the Qwen / Llama pair datasets and the shared evaluation helpers.

A dataset folder (data/qwen, data/llama) holds consolidated.parquet (one row per pair:
watermarked and unwatermarked answer with their exact token ids) and splits.csv (sealed
test split, and 5 folds for three CV repeats on the development split; all pairs of a
prompt stay in one fold). See data_prep/build_pairs_dataset.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core.ngram import SEED, Data, wins

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("qwen", "llama")
TEST_FOLD = 5  # outside the CV folds 0-4, so test texts are never counted
COLS = ["source_file", "record_index", "watermarked.token_ids", "unwatermarked.token_ids"]


def data_dir(name):
    return ROOT / "data" / name


def run_dir(name):
    """Where a dataset's outputs (scores, results json, logs) are written."""
    p = ROOT / "runs" / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def load(name, splits=("dev",), text=False):
    cols = COLS + (["watermarked.text", "unwatermarked.text"] if text else [])
    w = pd.read_parquet(data_dir(name) / "consolidated.parquet", columns=cols)
    s = pd.read_csv(data_dir(name) / "splits.csv")
    d = w.merge(s, on=["source_file", "record_index"], validate="one_to_one")
    return d[d.split.isin(splits)].reset_index(drop=True)


def load_dev(name, text=False):
    """Development pairs only; the sealed test split is not read."""
    return load(name, ("dev",), text)


def make_data(d, fold_col):
    """Data object for the pairs in d, with the fold of each pair taken from d[fold_col]."""
    ids = list(d["watermarked.token_ids"]) + list(d["unwatermarked.token_ids"])
    return Data(ids, np.tile(d[fold_col].to_numpy(), 2))


def groups(d):
    """Prompt index of each pair, for bootstrap CIs over prompts."""
    return pd.factorize(d["group_id"])[0]


def cluster_boot(x, gi, B=2000):
    """Mean of x with a 95% CI from resampling prompts."""
    G = gi.max() + 1
    gs, gc = np.bincount(gi, weights=x, minlength=G), np.bincount(gi, minlength=G)
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(B):
        k = np.bincount(rng.integers(0, G, G), minlength=G)
        bs.append((k * gs).sum() / (k * gc).sum())
    return float(x.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def report(name, s, D, gi, base=None):
    """AUROC and paired accuracy (prompt-bootstrap CI); with base, the paired-accuracy
    difference to another scorer on the same pairs."""
    w = wins(s, D.n)
    acc, ci = cluster_boot(w, gi)
    r = {"method": name, "auc": float(roc_auc_score(D.y, s)), "paired_accuracy": acc, "paired_ci95": ci}
    line = f"  {name:<44} AUC={r['auc']:.3f} paired={acc:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
    if base is not None:
        dl, dci = cluster_boot(w - wins(base, D.n), gi)
        r.update(delta=dl, delta_ci95=dci)
        line += f"  delta {dl:+.3f} [{dci[0]:+.3f},{dci[1]:+.3f}]"
    print(line, flush=True)
    return r
