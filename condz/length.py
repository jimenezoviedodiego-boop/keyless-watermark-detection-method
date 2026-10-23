"""Training and scoring length: nested condz when the training texts and the scored texts
are truncated independently to their first N tokens (N in 100, 250, 500, full).

Truncating training text excludes positions >= N from the counts; truncating scored text
restricts the scored events to positions < N. The first occurrence of a context before N is
the same in the full and the truncated text, so context dedup is unaffected. Development
pairs only, folds of one CV repeat (fold_r0 by default), prompt-bootstrap CIs.

Usage (from the repository root):
    python condz/length.py --dataset qwen
    python condz/length.py --dataset llama --windows 100 250      # a subset of the grid
Resumable: cells already in runs/<dataset>/window_grid_results_r<repeat>.json are skipped.
"""

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from condz.cv import grid_scorers  # noqa: E402
from core.ngram import nested_select  # noqa: E402
from core.pairs import DATASETS, groups, load_dev, make_data, report, run_dir  # noqa: E402

FULL = 10**9


def name(n):
    return "full" if n >= FULL else str(n)


def truncated_data(d, repeat, train_len, test_len):
    D = make_data(d, f"fold_r{repeat}")
    D.pfold = np.where(D.pos < train_len, D.pfold, -1)  # positions >= train_len never counted
    build = D.level

    def level(offsets, dedup="ctx"):
        new = (tuple(offsets), dedup) not in D.levels
        L = build(offsets, dedup)
        if new:
            L["sel"] = L["sel"] & (D.pos < test_len)  # only score the first test_len tokens
        return L

    D.level = level
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--windows", nargs="+", default=["100", "250", "500", "full"],
                    help="token windows to combine (train x score); 'full' = no truncation")
    args = ap.parse_args()
    windows = [FULL if w == "full" else int(w) for w in args.windows]
    d = load_dev(args.dataset)
    gi = groups(d)
    path = run_dir(args.dataset) / f"window_grid_results_r{args.repeat}.json"
    res = json.loads(path.read_text()) if path.exists() else []
    done = {(r["train_len"], r["test_len"]) for r in res}
    for tr, te in itertools.product(windows, windows):
        if (name(tr), name(te)) in done:
            continue
        t0 = time.time()
        D = truncated_data(d, args.repeat, tr, te)
        s, chosen = nested_select(D, grid_scorers(D), "auc")
        r = report(f"train {name(tr)} / score {name(te)}  (nested)", s, D, gi)
        r.update(train_len=name(tr), test_len=name(te), chosen=chosen)
        res.append(r)
        print(f"  chosen: {chosen}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        path.write_text(json.dumps(res, indent=1))
        del D


if __name__ == "__main__":
    main()
