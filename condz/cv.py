"""condz on the development pairs: fixed settings for context lengths k = 2..6, and the
detector of the paper, nested selection over 48 settings. Three CV repeats (folds fold_r0..2
of splits.csv); CIs resample prompts. The sealed test split is not read.

Usage (from the repository root):
    python condz/cv.py --dataset qwen
    python condz/cv.py --dataset llama --repeats 0
Writes runs/<dataset>/cv_results.json and out-of-fold scores oof_condz_nested_r<repeat>.npy.
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
from core.ngram import cfg_scorer, ctx, nested_select, oof  # noqa: E402
from core.pairs import DATASETS, groups, load_dev, make_data, report, run_dir  # noqa: E402

GRID = list(itertools.product((3, 4, 5), (0.5, 1), (1, 4), (1, 2), ("sqrt", "len")))  # k, c, alpha, min count, agg
FIXED = dict(stat="condz", alpha=1, mc=2, agg="sqrt", c=1)


def grid_scorers(D, grid=GRID):
    return {f"k{k}_c{c}_a{a}_mc{mc}_{agg}": cfg_scorer(D, dict(offsets=ctx(k), stat="condz", c=c, alpha=a,
                                                              mc=mc, agg=agg))
            for k, c, a, mc, agg in grid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--repeats", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    out = run_dir(args.dataset)
    d = load_dev(args.dataset)
    gi = groups(d)
    results = {}
    for rep in args.repeats:
        t0 = time.time()
        print(f"\n=== {args.dataset}, repeat {rep} (folds fold_r{rep}) ===", flush=True)
        D = make_data(d, f"fold_r{rep}")
        R = []
        for k in (2, 3, 4, 5, 6):
            R.append(report(f"condz k={k}, fixed settings", oof(D, dict(FIXED, offsets=ctx(k))), D, gi))
            if k in (2, 6):  # the nested grid only needs k = 3..5
                D.levels.pop((ctx(k), "ctx"), None)
        s, chosen = nested_select(D, grid_scorers(D), "auc")
        r = report("condz, nested selection (48 settings)", s, D, gi)
        r["chosen"] = chosen
        R.append(r)
        print(f"  chosen per fold: {chosen}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        np.save(out / f"oof_condz_nested_r{rep}.npy", s)
        results[rep] = R
        (out / "cv_results.json").write_text(json.dumps(results, indent=1))
        del D
    if len(results) > 1:
        print("\n=== mean over repeats ===")
        reps = list(results.values())
        for j, r in enumerate(reps[0]):
            print(f"  {r['method']:<44} AUC={np.mean([x[j]['auc'] for x in reps]):.3f} "
                  f"paired={np.mean([x[j]['paired_accuracy'] for x in reps]):.3f}")


if __name__ == "__main__":
    main()
