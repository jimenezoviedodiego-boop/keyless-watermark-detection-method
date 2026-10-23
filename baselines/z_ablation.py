"""Ablation: condz without context conditioning, development CV only.

condz scores an event (k-token context, next token) by the z-score of its watermarked count a
against its expectation given the watermarked share of its context. The unconditioned
variant drops the context: (a - b) / sqrt(a + b + alpha). Everything else is as for condz:
the same folds, nested selection by inner AUROC, and condz's grid without the parameter c
(which only enters the context share):
    k in (3, 4, 5), alpha in (1, 4), min count in (1, 2), aggregation in (sqrt, len)
If condz/cv.py has been run for the same repeat, the paired-accuracy difference condz - z
is reported with a prompt-bootstrap CI.

Usage (from the repository root):
    python baselines/z_ablation.py --dataset qwen
Writes runs/<dataset>/z_results.json and oof_z_nested_r<repeat>.npy.
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
from core.ngram import cfg_scorer, ctx, nested_select, wins  # noqa: E402
from core.pairs import DATASETS, cluster_boot, groups, load_dev, make_data, report, run_dir  # noqa: E402

Z_GRID = list(itertools.product((3, 4, 5), (1, 4), (1, 2), ("sqrt", "len")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--repeats", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    out = run_dir(args.dataset)
    path = out / "z_results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    d = load_dev(args.dataset)
    gi = groups(d)
    for rep in args.repeats:
        if str(rep) in results:
            continue
        t0 = time.time()
        print(f"\n=== {args.dataset}, repeat {rep} (folds fold_r{rep}) ===", flush=True)
        D = make_data(d, f"fold_r{rep}")
        grid = {f"k{k}_a{a}_mc{mc}_{agg}": cfg_scorer(D, dict(offsets=ctx(k), stat="z", alpha=a, mc=mc, agg=agg))
                for k, a, mc, agg in Z_GRID}
        z, chosen = nested_select(D, grid, "auc")
        np.save(out / f"oof_z_nested_r{rep}.npy", z)
        r = report("z (no conditioning), nested selection", z, D, gi)
        r["chosen"] = chosen
        entry = {"z": r}
        cz = out / f"oof_condz_nested_r{rep}.npy"
        if cz.exists():
            condz = np.load(cz)
            entry["condz"] = report("condz, nested selection", condz, D, gi)
            dl, ci = cluster_boot(wins(condz, D.n) - wins(z, D.n), gi)
            entry["condz_minus_z"] = {"delta": dl, "delta_ci95": ci}
            print(f"  condz minus z: delta {dl:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]", flush=True)
        print(f"  z chosen per fold: {chosen}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        results[str(rep)] = entry
        path.write_text(json.dumps(results, indent=1))
        del D
    if len(results) > 1:
        reps = list(results.values())
        print(f"\n=== mean over repeats ===\n  z: AUC={np.mean([x['z']['auc'] for x in reps]):.3f} "
              f"paired={np.mean([x['z']['paired_accuracy'] for x in reps]):.3f}")
        if all("condz_minus_z" in x for x in reps):
            print(f"  condz minus z: {np.mean([x['condz_minus_z']['delta'] for x in reps]):+.3f}")


if __name__ == "__main__":
    main()
