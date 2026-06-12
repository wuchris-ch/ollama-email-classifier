# Ollama Email Classifier

Local-inference email classifier on **Ollama (Qwen 2.5 7B)** with a **DSPy**
program, **LLM-as-judge** evaluation, and **GEPA** prompt optimization. All
inference happens on-device; only the judge model (used at eval time) is
remote.

The repo is small on purpose: one classifier, one judge, one optimizer, one
synthetic corpus, and a results table you can reproduce end-to-end.

## Architecture

```
                    ┌─────────────────────────┐
   email ──▶ DSPy ──▶ Qwen 2.5 7B (Ollama)    │ ──▶ label
            CoT     │   on-device, free       │
                    └─────────────────────────┘
                              │
                              ▼  predicted label only (no ground truth)
                    ┌─────────────────────────┐
                    │ LLM-as-judge            │
                    │   GPT-4o-mini, OpenAI   │ ──▶ is_correct + reasoning
                    └─────────────────────────┘

   GEPA loop:
   train.jsonl ─▶ DSPy program ─▶ metric+feedback ─▶ reflective rewrite
                       ▲                                       │
                       └───────────────────────────────────────┘
```

## Results

Latest committed run: classifier and judge on **DeepSeek v4 Pro**
(`CLASSIFIER_LM=deepseek/deepseek-v4-pro`), GEPA budget 400. The local
Qwen-on-Ollama numbers will replace these the next time the GPU box is up;
re-run `python -m src.eval` to refresh.

| Metric | Baseline (zero-shot) | After GEPA |
|---|---|---|
| accuracy | 0.9804 | 0.9804 |
| macro F1 | 0.9801 | 0.9801 |
| judge agreement | 0.9608 | 0.9608 |
| judge vs ground-truth alignment | 0.9804 | 0.9804 |

Test set: 204 emails, 6 classes, balanced. See `data/README.md`.

Two honest caveats about this run:

- **GEPA showed no delta** because a frontier reasoning model is already at
  ceiling on this corpus: every remaining miss is the `finance` "fraud
  alert" template, which legitimately reads as phishing without sender
  metadata. GEPA's headroom argument is for the local 7B model, where
  zero-shot leaves real room.
- **The judge was the same model as the classifier** in this run
  (self-judging inflates `judge_agreement`); with the default config the
  judge is an independent model.

Each eval run also writes a full **confusion matrix** (rows = truth,
columns = predicted) and a **sample of predictions** (misclassifications
first) into the same `results/*.json`, and prints the matrix to the
terminal. See `results/README.md` for the schema.

## Quickstart

```bash
# 1. install deps
pip install -e ".[dev]"

# 2. start Ollama and pull the model (one time)
ollama pull qwen2.5:7b-instruct

# 3. set OPENAI_API_KEY for the judge
cp .env.example .env  # then edit

# 4. baseline eval
python -m src.eval --split test --out eval_baseline

# 5. optimize with GEPA on the train split
python -m src.optimize --budget 60

# 6. eval the optimized program
python -m src.eval --split test \
  --classifier results/classifier_gepa.json \
  --out eval_after_gepa

# 7. classify a single email
python -m src.cli --subject "Order #99213 shipped" \
                  --body "Your package is on the way. Tracking link inside."
# {"subject": "Order #99213 shipped", "label": "transactional"}

# 8. or classify a whole JSONL file (batch / automation mode)
python -m src.cli --jsonl data/test.jsonl > labeled.jsonl
```

No OpenAI key? Run the eval without the judge — ground-truth accuracy,
macro F1, and the confusion matrix only need Ollama:

```bash
python -m src.eval --split test --no-judge --out eval_baseline
```

No GPU at all? Point the classifier at any remote LiteLLM-compatible API
instead of Ollama (this is how the committed DeepSeek run was produced):

```bash
# in .env
CLASSIFIER_LM=deepseek/deepseek-v4-pro
DEEPSEEK_API_KEY=...
JUDGE_MODEL=deepseek/deepseek-v4-pro
```

Useful flags: `--limit N` evaluates only the first N emails (quick smoke
run), `--classifier <path>` loads a GEPA-optimized program for both
`src.eval` and `src.cli`.

## Label set

`personal` · `work` · `finance` · `promo` · `transactional` · `spam`

Definitions and edge cases live in `src/classifier.py:EmailClassification`
(the docstring is the prompt, by design).

## Why these choices

- **Local Qwen 7B** keeps inference free, private, and on-device. The whole
  point of the project: classification at scale without per-email API spend
  and without leaking inbox content to a third party.
- **GPT-4o-mini as judge** is the cheapest credible judge model for a small
  test set ($1 in API spend covers many full eval runs).
- **Synthetic corpus** is reproducible and PII-free. It under-represents
  real-world noise; that limitation is documented in `data/README.md`.
- **GEPA over MIPRO/Bootstrap** because the resume claims GEPA, and GEPA's
  reflective evolution is more interpretable than search-based optimizers
  for a six-way classifier with mixed-prompt failures.

## Layout

```
src/         classifier, judge, eval, GEPA loop, CLI
data/        synthesize.py + train/dev/test jsonl (committed)
results/     eval JSON + saved optimized program
tests/       smoke tests that run without Ollama or OpenAI
```

## Tests

```bash
pytest -q
```

The default suite mocks every LM call, so it runs (and CI can run) without
Ollama, a GPU, or any API key. It covers the dataset contract, label
normalization, the full eval pipeline (metrics, confusion matrix, sample
outputs, judge-error handling), the CLI in single and batch mode, the GEPA
feedback metric, and byte-for-byte reproducibility of the committed JSONL
from `data/synthesize.py`.

Live integration tests against a real Ollama instance are opt-in:

```bash
RUN_OLLAMA_TESTS=1 pytest tests/test_integration_ollama.py -v
```

They respect `OLLAMA_BASE_URL` / `OLLAMA_MODEL`, so they work against a
remote Ollama server too.

## Limitations

- **Synthetic data only.** The corpus is template-generated (60 base
  templates with placeholder fills). It has no HTML, signatures, threading,
  attachments, or multi-language content, and accuracy on it will overstate
  accuracy on a real inbox. See `data/README.md` for the full discussion.
- **Six fixed labels.** Real inboxes need an "other/unsure" escape hatch;
  here, unparseable or out-of-vocabulary model outputs are coerced to
  `spam`, which is the conservative choice for a filter but inflates the
  spam row of the confusion matrix.
- **Judge is not ground truth.** `judge_agreement` is a no-labels proxy;
  `judge_vs_truth_alignment` is reported precisely so you can see how far
  the proxy drifts.
- **Single local model, wired by env var.** No model-swapping abstraction,
  no Docker, no web UI, no inbox integration — those are out of scope for
  this repo by design (see `CLAUDE.md`).
- **GEPA budget matters.** The default `--budget 30` is a smoke-test
  setting; meaningful optimization runs need 60+ metric calls and a real
  reflection model.

## Is this real? (resume proof)

This repo backs a resume bullet, so here is the claim-by-claim map:

| Claim | Where it lives |
|---|---|
| Local inference on Ollama (Qwen 2.5 7B) | `src/classifier.py:configure_ollama_lm` — env-var wiring, no cloud fallback |
| DSPy program | `src/classifier.py:EmailClassifier` (ChainOfThought over a typed signature) |
| LLM-as-judge evaluation | `src/judge.py` + `judge_all` in `src/eval.py`; judge never sees ground truth |
| GEPA prompt optimization | `src/optimize.py` — `dspy.GEPA` with a feedback-bearing metric |
| Measured results | `results/eval_baseline.json` / `results/eval_after_gepa.json`, committed; the README table must match the latter |
| Reproducible end-to-end | `data/synthesize.py` is stdlib-only and deterministic; a test asserts the committed JSONL matches regeneration byte-for-byte |
| Tested without a GPU | `pytest -q` passes with no Ollama and no API key |

## License

MIT.
