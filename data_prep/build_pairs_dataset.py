"""Build consolidated.parquet + splits.csv from a folder of pair JSON files, in the layout the
experiments read (see core/pairs.py).

Input:  <dataset>/raw/pairs/*.json, one pair (watermarked + unwatermarked answer) per file.
Output: <dataset>/consolidated.parquet  one row per pair, nested fields flattened with dots
        <dataset>/splits.csv            split (dev/test) and CV folds:
          - all pairs of one prompt stay together (group_id = prompt_id),
          - length_bin = quartile of the prompt's mean answer length (both twins, all replicates),
          - per length bin, 20% of prompts go to a sealed test split,
          - dev prompts get folds 0-4 for three CV repeats (fold_r0..fold_r2), balanced per bin.
        <dataset>/splits.meta.json      how the splits were made

Invalid pairs (too short, max tokens) are kept and marked by pair_valid_long.

The Llama files in data/llama were built with this script (seed below). The Qwen files in
data/qwen use the same scheme but were split by an earlier version of this procedure, so
rerunning the script on the Qwen pairs gives a different, equally valid split; the shipped
splits.csv is the one used in the paper.

Usage (from the repository root):
    python data_prep/build_pairs_dataset.py path/to/dataset     # folder containing raw/pairs/*.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20240917
TEST_FRAC = 0.2
FOLDS, REPEATS = 5, 3


def load_pairs(pair_dir):
    rows = []
    for f in sorted(pair_dir.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        for i, r in enumerate(rec if isinstance(rec, list) else [rec]):
            flat = {"source_file": f.name, "record_index": i}
            for k, v in r.items():
                if k in ("watermarked", "unwatermarked"):
                    flat.update({f"{k}.{kk}": vv for kk, vv in v.items()})
                elif isinstance(v, dict):
                    flat[k] = json.dumps(v, sort_keys=True)
                else:
                    flat[k] = v
            rows.append(flat)
    return pd.DataFrame(rows)


def make_splits(d):
    s = d[["source_file", "record_index", "pair_id", "prompt_id"]].rename(columns={"prompt_id": "group_id"})
    s["n_tokens_unwatermarked"] = d["unwatermarked.token_ids"].map(len)
    s["n_tokens_watermarked"] = d["watermarked.token_ids"].map(len)
    s["pair_texts_identical"] = d["watermarked.text"] == d["unwatermarked.text"]
    plen = ((s.n_tokens_unwatermarked + s.n_tokens_watermarked) / 2).groupby(s.group_id).mean()
    pbin = pd.qcut(plen.rank(method="first"), 4, labels=False)
    s["length_bin"] = s.group_id.map(pbin).astype(int)

    rng = np.random.default_rng(SEED)
    split, folds = {}, {r: {} for r in range(REPEATS)}
    for b in range(4):
        prompts = np.array(sorted(pbin.index[pbin == b]))
        rng.shuffle(prompts)
        n_test = int(round(TEST_FRAC * len(prompts)))
        split.update({p: "test" for p in prompts[:n_test]})
        dev = prompts[n_test:]
        split.update({p: "dev" for p in dev})
        for r in range(REPEATS):
            perm = rng.permutation(dev)
            off = rng.integers(FOLDS)  # rotate so leftover prompts don't always land in fold 0
            folds[r].update({p: (i + off) % FOLDS for i, p in enumerate(perm)})
    s["split"] = s.group_id.map(split)
    for r in range(REPEATS):
        s[f"fold_r{r}"] = s.group_id.map(folds[r]).fillna(-1).astype(int)
    cols = ["source_file", "record_index", "pair_id", "group_id", "length_bin", "n_tokens_unwatermarked",
            "n_tokens_watermarked", "pair_texts_identical", "split"] + [f"fold_r{r}" for r in range(REPEATS)]
    return s[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path, help="folder containing raw/pairs/*.json")
    ap.add_argument("--force", action="store_true", help="overwrite an existing splits.csv")
    args = ap.parse_args()
    out_split = args.dataset / "splits.csv"
    if out_split.exists() and not args.force:
        raise SystemExit(f"{out_split} exists; splits are kept fixed once made (use --force to rebuild)")

    d = load_pairs(args.dataset / "raw" / "pairs")
    assert d.pair_id.is_unique, "duplicate pair ids"
    d.to_parquet(args.dataset / "consolidated.parquet", index=False)
    s = make_splits(d)
    s.to_csv(out_split, index=False)
    meta = {"seed": SEED, "test_frac": TEST_FRAC, "folds": FOLDS, "repeats": REPEATS, "group": "prompt_id",
            "stratify": "quartile of prompt mean answer length (tokens, both twins)",
            "pairs": len(s), "prompts": int(s.group_id.nunique()),
            "split_pairs": s.split.value_counts().to_dict(),
            "valid_long_pairs": int(d.pair_valid_long.sum()) if "pair_valid_long" in d else None}
    (args.dataset / "splits.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print(s[s.split == "dev"].groupby(["fold_r0", "length_bin"]).size().unstack())


if __name__ == "__main__":
    main()
