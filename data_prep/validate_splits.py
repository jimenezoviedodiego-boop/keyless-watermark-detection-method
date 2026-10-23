"""Split integrity checks: prompts never straddle dev/test or folds, no duplicated texts.

Usage (from the repository root):
    python data_prep/validate_splits.py --dataset qwen
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.pairs import DATASETS, data_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    args = ap.parse_args()
    cols = ["source_file", "record_index", "prompt_id", "prompt", "watermarked.text", "unwatermarked.text"]
    w = pd.read_parquet(data_dir(args.dataset) / "consolidated.parquet", columns=cols)
    s = pd.read_csv(data_dir(args.dataset) / "splits.csv")
    d = w.merge(s, on=["source_file", "record_index"], validate="one_to_one")
    fails = 0

    def ok(cond, msg):
        nonlocal fails
        fails += not cond
        print(("PASS " if cond else "FAIL ") + msg)

    ok(len(d) == len(w) == len(s), f"{len(d)} pairs, splits join 1:1")
    ok((d.groupby("prompt_id").group_id.nunique() == 1).all()
       and (d.groupby("group_id").prompt_id.nunique() == 1).all(), "group_id == prompt_id (one-to-one)")
    ok((d.groupby("prompt_id").split.nunique() == 1).all(), "every prompt entirely in dev or entirely in test")
    dev = d[d.split == "dev"]
    for r in range(3):
        ok((dev.groupby("prompt_id")[f"fold_r{r}"].nunique() == 1).all(), f"repeat {r}: every prompt in one fold")
        print(f"     repeat {r} fold sizes (pairs):", dev[f"fold_r{r}"].value_counts().sort_index().to_dict())
    ok((d[d.split == "test"][["fold_r0", "fold_r1", "fold_r2"]] == -1).all().all(), "test pairs have no fold")
    norm = lambda t: " ".join(str(t).lower().split())  # noqa: E731
    d["pnorm"] = d.prompt.map(norm)
    ok((d.groupby("pnorm").prompt_id.nunique() == 1).all(), "no identical prompt text under two prompt ids")
    ok((d.groupby("pnorm").split.nunique() == 1).all(), "no prompt text shared between dev and test")
    txt = pd.concat([d[["prompt_id"]].assign(h=d["watermarked.text"].map(norm)),
                     d[["prompt_id"]].assign(h=d["unwatermarked.text"].map(norm))])
    ok(txt.h.duplicated().sum() == 0, f"no duplicated answer texts ({int(txt.h.duplicated().sum())} duplicates)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
