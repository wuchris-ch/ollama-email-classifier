"""Single-email classification CLI.

Usage:
  python -m src.cli --subject "Order #123 shipped" --body "Your package is on the way."
  cat email.eml | python -m src.cli --stdin
  python -m src.cli --jsonl data/test.jsonl > labeled.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys

from src.classifier import LABELS, EmailClassifier, configure_ollama_lm


def classify_one(classifier: EmailClassifier, subject: str, body: str) -> dict[str, str]:
    pred = classifier(subject=subject, body=body)
    label = str(pred.label).strip().lower()
    if label not in LABELS:
        label = "spam"
    return {"subject": subject, "label": label}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="")
    p.add_argument("--body", default="")
    p.add_argument("--classifier", default=None, help="Path to a saved DSPy program (post-GEPA).")
    p.add_argument("--stdin", action="store_true", help="Read body from stdin; subject from --subject.")
    p.add_argument("--jsonl", default=None,
                   help="Batch mode: classify every {subject, body} row in a JSONL file, emit JSONL to stdout.")
    args = p.parse_args()

    body = sys.stdin.read() if args.stdin else args.body
    if not (args.subject or body or args.jsonl):
        p.error("provide --subject and/or --body, --stdin, or --jsonl")

    configure_ollama_lm()
    classifier = EmailClassifier()
    if args.classifier:
        classifier.load(args.classifier)

    if args.jsonl:
        with open(args.jsonl) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                print(json.dumps(classify_one(classifier, row.get("subject", ""), row.get("body", ""))))
        return

    print(json.dumps(classify_one(classifier, args.subject, body)))


if __name__ == "__main__":
    main()
