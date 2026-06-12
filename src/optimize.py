"""GEPA prompt optimization on the email classifier.

GEPA (Genetic-Pareto) reflectively evolves prompts using a feedback metric.
This script wraps DSPy's GEPA optimizer and saves the optimized program.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import dspy

from src.classifier import EmailClassifier, configure_judge_lm, configure_ollama_lm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_examples(path: Path) -> list[dspy.Example]:
    raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [
        dspy.Example(subject=r["subject"], body=r["body"], label=r["label"])
        .with_inputs("subject", "body")
        for r in raw
    ]


def metric_with_feedback(example: dspy.Example, pred: dspy.Prediction, *_args, **_kwargs):
    """Boolean correctness + a short textual reason GEPA can reflect on."""
    correct = str(getattr(pred, "label", "")).strip().lower() == example.label
    feedback = (
        f"Correct (predicted {pred.label})."
        if correct
        else f"Incorrect: predicted '{pred.label}', expected '{example.label}'. "
             f"Subject: {example.subject!r}. Reconsider how this label differs from the prediction."
    )
    return dspy.Prediction(score=float(correct), feedback=feedback)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trainset", default=str(DATA / "train.jsonl"))
    p.add_argument("--valset", default=str(DATA / "dev.jsonl"))
    p.add_argument("--budget", type=int, default=30, help="Max metric calls; raise for stronger results.")
    p.add_argument("--val-limit", type=int, default=None,
                   help="Use only the first N validation examples; smaller valset = more GEPA iterations per budget.")
    p.add_argument("--out", default=str(ROOT / "results" / "classifier_gepa.json"))
    args = p.parse_args()

    configure_ollama_lm()
    trainset = load_examples(Path(args.trainset))
    valset = load_examples(Path(args.valset))
    if args.val_limit:
        valset = valset[: args.val_limit]

    program = EmailClassifier()

    reflection_lm = configure_judge_lm()
    optimizer = dspy.GEPA(
        metric=metric_with_feedback,
        max_metric_calls=args.budget,
        reflection_lm=reflection_lm,
        num_threads=int(os.getenv("EVAL_WORKERS", "8")),
    )

    print(f"running GEPA: budget={args.budget}, train={len(trainset)}, val={len(valset)}")
    optimized = optimizer.compile(program, trainset=trainset, valset=valset)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    optimized.save(args.out)
    print(f"saved optimized program to {args.out}")
    print("\nnext: python -m src.eval --split test --classifier", args.out, "--out eval_after_gepa")


if __name__ == "__main__":
    main()
