# Ollama Email Classifier — project rules

## What this repo is
A small, public-ready proof of a local-inference email classifier:
Qwen 2.5 7B on Ollama, DSPy program, LLM-as-judge evaluation,
GEPA prompt optimization. The repo is the artifact that backs a
specific resume bullet, so the README, results table, and
reproduction steps are part of the deliverable.

## Order to read
1. `README.md` — public face, claimed numbers, quickstart
2. `data/README.md` — dataset provenance and label taxonomy
3. `src/classifier.py` — DSPy program shape
4. `src/eval.py`, `src/optimize.py` — the eval and GEPA loops

## Conventions
- Python only. No JS/TS, no Docker in v1.
- Stdlib in `data/synthesize.py` so the dataset is reproducible without API keys.
- No abstractions for "swappable models"; the Ollama model is wired in by env var only.
- Results files (`results/*.json`) are committed. The README's headline
  number must match the latest `results/eval_after_gepa.json`.

## What this repo will NOT grow into
- Gmail OAuth / inbox automation (would belong in a separate repo).
- A web UI.
- Multiple models orchestrated together.
- A general-purpose classification framework.

If a change would push the repo past those boundaries, stop and
flag it instead of building it.
