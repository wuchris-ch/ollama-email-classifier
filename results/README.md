# Results

Two JSON files land here after running the pipeline end-to-end:

| File | Produced by |
|---|---|
| `eval_baseline.json` | `python -m src.eval --split test --out eval_baseline` |
| `eval_after_gepa.json` | `python -m src.eval --split test --classifier results/classifier_gepa.json --out eval_after_gepa` |
| `classifier_gepa.json` | `python -m src.optimize` |

The headline number quoted in the top-level `README.md` must match
`eval_after_gepa.json#accuracy`. If you re-run the pipeline, update the
README table in the same commit.

## What each metric means

- **accuracy** — exact-match against the synthetic ground truth (6-way).
- **macro_f1** — class-balanced F1, robust to any class imbalance.
- **judge_agreement** — fraction of predictions the GPT-4o-mini judge calls
  correct without seeing the ground truth. Useful as a no-labels signal.
- **judge_vs_truth_alignment** — how often the judge agrees with the ground
  truth on whether a prediction is correct. Sanity-checks the judge itself.
- **per_class** — sklearn classification report per label.
- **confusion_matrix** — `{labels, rows_are_truth, matrix}`; rows are ground
  truth, columns are predictions, both in the canonical label order.
- **samples** — up to 10 `{subject, truth, predicted}` rows, with every
  misclassification listed before any correct prediction, so the most
  informative examples survive truncation.

Runs invoked with `--no-judge` set `judge_model`, `judge_agreement`, and
`judge_vs_truth_alignment` to `null`; everything else is identical.

Every result records `classifier_lm` — the exact LiteLLM model string the
predictions came from (`ollama_chat/...` for local runs, e.g.
`deepseek/deepseek-v4-pro` for remote ones) — so a number can never be
silently attributed to the wrong backend.

## Reading the GEPA delta

GEPA reflectively rewrites the classifier's chain-of-thought instructions.
A meaningful run shows accuracy and macro_f1 climbing on `eval_after_gepa`
relative to `eval_baseline`, with the largest gains on the most-confused
classes (typically `transactional` vs `promo`, and `spam` vs `transactional`).
