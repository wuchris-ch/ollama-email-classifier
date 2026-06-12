"""End-to-end evaluation: ground-truth metrics + LLM-as-judge agreement."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import dspy
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.classifier import (
    LABELS,
    EmailClassifier,
    classifier_lm_name,
    configure_judge_lm,
    configure_ollama_lm,
)
from src.judge import Judge

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def workers() -> int:
    return int(os.getenv("EVAL_WORKERS", "8"))


def load_jsonl(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def predict_all(classifier: EmailClassifier, examples: list[dict[str, str]]) -> list[str]:
    def one(ex: dict[str, str]) -> str:
        try:
            out = classifier(subject=ex["subject"], body=ex["body"])
            label = str(out.label).strip().lower()
        except Exception as e:
            print(f"  classifier error on '{ex['subject'][:40]}': {e}")
            label = "spam"
        return label if label in LABELS else "spam"

    with ThreadPoolExecutor(max_workers=workers()) as pool:
        return list(pool.map(one, examples))


def judge_all(judge: Judge, examples: list[dict[str, str]], preds: list[str]) -> list[bool]:
    # dspy.context is thread-local, so swap the global LM for the judge phase
    # (prediction is already done) and restore it afterwards.
    judge_lm = configure_judge_lm()
    prev_lm = dspy.settings.lm
    dspy.configure(lm=judge_lm)

    def one(pair: tuple[dict[str, str], str]) -> bool:
        ex, pred = pair
        try:
            out = judge(subject=ex["subject"], body=ex["body"], predicted_label=pred)
            return bool(out.is_correct)
        except Exception as e:
            print(f"  judge error on '{ex['subject'][:40]}': {e}")
            return False

    try:
        with ThreadPoolExecutor(max_workers=workers()) as pool:
            return list(pool.map(one, zip(examples, preds)))
    finally:
        dspy.configure(lm=prev_lm)


def sample_outputs(
    examples: list[dict[str, str]], preds: list[str], k: int = 10
) -> list[dict[str, str]]:
    """A reviewable slice of predictions: every distinct mistake pattern first, then hits."""
    misses = [
        {"subject": ex["subject"], "truth": ex["label"], "predicted": p}
        for ex, p in zip(examples, preds)
        if p != ex["label"]
    ]
    hits = [
        {"subject": ex["subject"], "truth": ex["label"], "predicted": p}
        for ex, p in zip(examples, preds)
        if p == ex["label"]
    ]
    return (misses + hits)[:k]


def evaluate(
    split: str, classifier_path: str | None, out_name: str, run_judge: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    examples = load_jsonl(DATA / f"{split}.jsonl")
    if limit:
        examples = examples[:limit]
    truths = [ex["label"] for ex in examples]

    configure_ollama_lm()
    classifier = EmailClassifier()
    if classifier_path:
        classifier.load(classifier_path)

    print(f"classifying {len(examples)} {split} emails on {classifier_lm_name()}...")
    preds = predict_all(classifier, examples)

    accuracy = sum(p == t for p, t in zip(preds, truths)) / len(truths)
    macro_f1 = f1_score(truths, preds, labels=list(LABELS), average="macro", zero_division=0)
    report = classification_report(truths, preds, labels=list(LABELS), output_dict=True, zero_division=0)
    cm = confusion_matrix(truths, preds, labels=list(LABELS)).tolist()

    judge_agreement = judge_vs_truth = None
    if run_judge:
        print("running judge...")
        judge = Judge()
        verdicts = judge_all(judge, examples, preds)
        judge_agreement = round(sum(verdicts) / len(verdicts), 4)
        judge_vs_truth = round(
            sum(int(v) == int(p == t) for v, p, t in zip(verdicts, preds, truths)) / len(verdicts), 4
        )

    result: dict[str, Any] = {
        "split": split,
        "n": len(examples),
        "classifier_lm": classifier_lm_name(),
        "judge_model": os.getenv("JUDGE_MODEL", "gpt-4o-mini") if run_judge else None,
        "classifier_path": classifier_path,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(macro_f1), 4),
        "judge_agreement": judge_agreement,
        "judge_vs_truth_alignment": judge_vs_truth,
        "per_class": {label: report[label] for label in LABELS if label in report},
        "confusion_matrix": {"labels": list(LABELS), "rows_are_truth": True, "matrix": cm},
        "samples": sample_outputs(examples, preds),
    }

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"{out_name}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")
    print(f"accuracy={result['accuracy']}  macro_f1={result['macro_f1']}  judge_agreement={result['judge_agreement']}")
    print("\nconfusion matrix (rows=truth, cols=predicted):")
    print(format_confusion_matrix(list(LABELS), cm))
    return result


def format_confusion_matrix(labels: list[str], matrix: list[list[int]]) -> str:
    width = max(len(label) for label in labels) + 2
    header = " " * width + "".join(label[:width - 1].rjust(width) for label in labels)
    rows = [
        label.ljust(width) + "".join(str(c).rjust(width) for c in row)
        for label, row in zip(labels, matrix)
    ]
    return "\n".join([header, *rows])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--classifier", default=None, help="Path to a saved DSPy program JSON (post-GEPA).")
    p.add_argument("--out", default="eval_baseline")
    p.add_argument("--no-judge", action="store_true",
                   help="Skip the LLM-as-judge pass; runs without an OPENAI_API_KEY.")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples.")
    args = p.parse_args()
    evaluate(split=args.split, classifier_path=args.classifier, out_name=args.out,
             run_judge=not args.no_judge, limit=args.limit)


if __name__ == "__main__":
    main()
