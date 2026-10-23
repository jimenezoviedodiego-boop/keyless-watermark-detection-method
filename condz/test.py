"""One-time evaluation of condz on the sealed test pairs (600 per dataset).

1. Settings: 5-fold CV (fold_r0) over the 48-setting grid on the development pairs only;
   the setting with the best development AUROC is chosen.
2. Counts: learned from all development pairs. Test texts are never counted.
3. Every test text is scored once. Reported:
   - paired accuracy, 95% CI from resampling test prompts;
   - AUROC, 95% CI from resampling prompts;
   - single-text detection: for each target false-positive rate eps, the threshold is the
     (1 - eps) quantile of the out-of-fold scores of unwatermarked development texts; the
     realised TPR and FPR on the test texts are reported with prompt-bootstrap CIs.

Usage (from the repository root):
    python condz/test.py --dataset qwen
Writes runs/<dataset>/test_results.json.
"""

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from condz.cv import GRID  # noqa: E402
from core.ngram import SEED, ctx, level_scores, oof  # noqa: E402
from core.pairs import DATASETS, TEST_FOLD, cluster_boot, load, make_data, run_dir  # noqa: E402

TARGETS = (0.01, 0.05, 0.10)
B = 1000


def boot_groups(gi):
    """Bootstrap draws of prompts -> per-draw multiplicity of each prompt index."""
    rng = np.random.default_rng(SEED)
    G = gi.max() + 1
    return [np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(B)]


def ratio_ci(num, den, gi, draws):
    """sum(num) / sum(den) over texts with a prompt-bootstrap CI."""
    G = gi.max() + 1
    ns, ds = np.bincount(gi, weights=num, minlength=G), np.bincount(gi, weights=den, minlength=G)
    bs = [(k * ns).sum() / (k * ds).sum() for k in draws]
    return float(num.sum() / den.sum()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def auc_ci(s, y, gi, draws):
    by_g = [np.flatnonzero(gi == g) for g in range(gi.max() + 1)]
    bs = []
    for k in draws:
        idx = np.concatenate([np.repeat(by_g[g], m) for g, m in enumerate(k) if m])
        bs.append(roc_auc_score(y[idx], s[idx]))
    return float(roc_auc_score(y, s)), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    args = ap.parse_args()
    d = load(args.dataset, ("dev", "test"))
    test = (d.split == "test").to_numpy()
    d["cvfold"] = np.where(test, TEST_FOLD, d.fold_r0)
    D = make_data(d, "cvfold")
    dev_t, test_t = D.fold != TEST_FOLD, D.fold == TEST_FOLD
    print(f"{args.dataset}: dev pairs {int((~test).sum())}, test pairs {int(test.sum())}", flush=True)

    best, best_auc, best_s = None, -1, None
    for k, c, a, mc, agg in GRID:
        cfg = dict(offsets=ctx(k), stat="condz", c=c, alpha=a, mc=mc, agg=agg)
        s = oof(D, cfg)  # 5-fold CV on development texts only
        v = roc_auc_score(D.y[dev_t], s[dev_t])
        if v > best_auc:
            best, best_auc, best_s = cfg, v, s
    settings = dict(k=len(best["offsets"]), c=best["c"], alpha=best["alpha"], min_count=best["mc"], agg=best["agg"])
    print(f"selected on development CV: {settings} (dev AUROC {best_auc:.3f})", flush=True)

    L = D.level(best["offsets"])
    s, _ = level_scores(D, L, (), test_t, "condz", best["alpha"], best["mc"], best["agg"], best["c"])
    res = {"selected": dict(settings, dev_auc=best_auc)}

    i = np.flatnonzero(test)  # test pairs; text i is watermarked, text D.n + i its twin
    w = (s[i] > s[D.n + i]).astype(float) + 0.5 * (s[i] == s[D.n + i])
    res["paired_accuracy"], res["paired_ci95"] = cluster_boot(w, pd.factorize(d.loc[test, "group_id"])[0])

    # prompt indices over the whole dataset; prompts outside the test split carry no texts
    gi = np.tile(pd.factorize(d["group_id"])[0], 2)[test_t]
    draws = boot_groups(gi)
    y, st = D.y[test_t], s[test_t]
    res["auc"], res["auc_ci95"] = auc_ci(st, y, gi, draws)
    print(f"\n=== sealed test ({int(test_t.sum())} texts, {int(test.sum())} pairs) ===", flush=True)
    print(f"  paired accuracy {res['paired_accuracy']:.3f} [{res['paired_ci95'][0]:.3f},{res['paired_ci95'][1]:.3f}]"
          f"   AUROC {res['auc']:.3f} [{res['auc_ci95'][0]:.3f},{res['auc_ci95'][1]:.3f}]", flush=True)

    neg_dev = best_s[dev_t][D.y[dev_t] == 0]
    pos, neg = (y == 1).astype(float), (y == 0).astype(float)
    res["thresholds"] = []
    for t in TARGETS:
        tau = float(np.quantile(neg_dev, 1 - t))
        flag = (st > tau).astype(float)
        tpr, tci = ratio_ci(flag * pos, pos, gi, draws)
        fpr, fci = ratio_ci(flag * neg, neg, gi, draws)
        res["thresholds"].append(dict(target_fpr=t, tau=tau, tpr=tpr, tpr_ci95=tci, fpr=fpr, fpr_ci95=fci))
        print(f"  target FPR {t:.0%}: TPR {tpr:.3f} [{tci[0]:.3f},{tci[1]:.3f}]   "
              f"FPR {fpr:.3f} [{fci[0]:.3f},{fci[1]:.3f}]", flush=True)
    (run_dir(args.dataset) / "test_results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
