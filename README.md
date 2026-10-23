# condz: Learning to Detect SynthID-Text Without the Key

Code and data for the paper *condz: Learning to Detect SynthID-Text Without the Key*
(Furgała, Iardella, Jimenez Oviedo, Maistro, Parate; Eindhoven University of Technology).

SynthID-Text biases each token choice with keyed pseudorandom values of the four preceding
tokens and the candidate token. Because the key is fixed, the bias for a given
(4-token context, next token) event is the same in every watermarked text, so it can be
learned by counting. **condz** (conditional z-score) counts each event in watermarked and
unwatermarked training texts, scores it by how far its watermarked count exceeds what its
context alone predicts, and averages these scores over a text. It needs labelled examples
and the generator's tokenizer, but not the key, the model weights, the logits, or any query
to the generator at detection time.

## Results

Paired accuracy: the share of pairs (same prompt, watermark on and off) in which the
watermarked answer scores higher; chance is 0.5. 95% confidence intervals resample prompts.

| Dataset | Evaluation | AUROC | Paired accuracy |
|---|---|---|---|
| Gemma-7B-IT (Google's key) | 5-fold CV, 3,000 pairs | 0.568 | 0.573 (0.556–0.590) |
| Qwen2.5-1.5B-Instruct (own key) | 5-fold CV, 2,400 dev pairs, mean of 3 repeats | 0.925 | 0.931 |
| Qwen2.5-1.5B-Instruct | sealed test, 600 pairs | 0.934 (0.920–0.946) | 0.937 (0.917–0.955) |
| Llama-3.1-8B-Instruct (own key) | 5-fold CV, 2,400 dev pairs, mean of 3 repeats | 0.851 | 0.858 |
| Llama-3.1-8B-Instruct | sealed test, 600 pairs | 0.872 (0.853–0.891) | 0.892 (0.863–0.920) |

Single-text detection on the sealed test sets, thresholds set on development data for a target
false-positive rate (FPR):

| Target FPR | Qwen TPR | Qwen FPR | Llama TPR | Llama FPR |
|---|---|---|---|---|
| 1% | 0.387 | 0.022 | 0.250 | 0.008 |
| 5% | 0.735 | 0.070 | 0.483 | 0.055 |
| 10% | 0.842 | 0.127 | 0.622 | 0.102 |

Baselines and ablation (development CV, first repeat, paired accuracy):

| Input | Detector | Qwen | Llama |
|---|---|---|---|
| Bag of words (TF-IDF) | LightGBM, tuned | 0.546 | 0.508 |
| Bag of words (TF-IDF) | MLP, tuned | 0.514 | 0.522 |
| 4+1 events | LightGBM, tuned | 0.792 | 0.716 |
| 4+1 events | MLP, tuned | 0.773 | 0.715 |
| 4+1 events | z-score without conditioning | 0.918 | 0.823 |
| 4+1 events | **condz** | **0.929** | **0.858** |

The length experiment (training and scored texts truncated to 100 / 250 / 500 tokens) and its
figure are produced by `condz/length.py` and `condz/plot_length.py`; the figure is in
[results/fig_length_auroc.pdf](results/fig_length_auroc.pdf).

## Layout

```
core/
  ngram.py            event counting, condz and z event scores, out-of-fold scoring, nested selection
  pairs.py            loading the Qwen / Llama datasets, prompt-bootstrap CIs, reporting
condz/
  cv.py               development CV: condz with fixed settings for k = 2..6, and nested selection
  test.py             one-time sealed-test evaluation: paired accuracy, AUROC, thresholds (TPR / FPR)
  length.py           training x scoring length grid (100 / 250 / 500 / full tokens)
  plot_length.py      figure of the length experiment
  gemma.py            condz on the public Gemma pairs (5-fold CV)
baselines/
  classifiers.py      LightGBM and MLP on bag-of-words vs 4+1 event features (optionally tuned)
  z_ablation.py       condz without context conditioning
data_prep/
  build_pairs_dataset.py   raw pair JSON -> consolidated.parquet + splits.csv
  validate_splits.py       split integrity checks
data/
  qwen/, llama/       consolidated.parquet (pairs with exact token ids), splits.csv
  gemma/              downloaded on first use from google-deepmind/synthid-text
results/              figure used in the paper
runs/                 created by the scripts: scores, logs and result json (not in git)
```

## Data

Both own datasets use the same 1,000 prompts from the training split of
[hassanjbara/LONG](https://huggingface.co/datasets/hassanjbara/LONG), three watermarked /
unwatermarked pairs per prompt (3,000 pairs), generated with the SynthID-Text implementation
in Hugging Face Transformers (n-gram length 5, i.e. 4 context tokens, 30 keys, sampling table
of 65,536 entries, context history 1,024), temperature 1.0, top-k 50, and 768 to 2,048 new
tokens. The same seed is set before the watermarked and the unwatermarked answer of a batch,
so the two answers of a pair share their sampling randomness. Each dataset has its own key;
**the keys are not part of this repository.** `data/llama/generation_config.json` records the
Llama generation settings.

`splits.csv` assigns 200 prompts (600 pairs) to a sealed test split and the other 800 prompts
to five folds for three CV repeats, with all pairs of a prompt in the same split and fold.
The test splits were scored once, by `condz/test.py`.

Each row of `consolidated.parquet` holds one pair: `watermarked.text`,
`watermarked.token_ids`, `unwatermarked.text`, `unwatermarked.token_ids`, the prompt and
quality flags (`pair_valid_long` is false for answers under 600 words or cut at 2,048
tokens; they are kept in all experiments).

## Reproducing

Python 3.11+ and the packages in `requirements.txt`. Run everything from the repository root.
Times are for a 4-core laptop CPU.

```bash
python data_prep/validate_splits.py --dataset qwen       # seconds

python condz/cv.py --dataset qwen                         # 3 repeats, ~15-25 min
python condz/test.py --dataset qwen                       # sealed test, ~10-15 min
python condz/length.py --dataset qwen                     # 16 cells, ~1.5 h
python condz/plot_length.py                               # needs length.py for qwen and llama

python baselines/z_ablation.py --dataset qwen             # 3 repeats, ~10 min
python baselines/classifiers.py --dataset qwen --model lgbm --tune   # ~40 min
python baselines/classifiers.py --dataset qwen --model mlp --tune    # ~1.5 h

python condz/gemma.py                                     # downloads data + tokenizer, ~10 min
```

Replace `qwen` by `llama` for the Llama dataset. Results are written to `runs/<dataset>/`.

## Scope and limitations

The detector assumes labelled watermarked and unwatermarked texts from the same model under
the same key, and the generator's tokenizer. Robustness to edits, human-written text and
other watermarking schemes were not evaluated. Learning which tokens a key favours is also the
first step of watermark stealing, so this approach could help an attacker as well as an
auditor.
