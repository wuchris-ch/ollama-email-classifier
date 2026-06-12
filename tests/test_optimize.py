"""GEPA metric and example-loading tests — no LM calls."""
from __future__ import annotations

import json
from pathlib import Path

from src.optimize import load_examples, metric_with_feedback


class FakePred:
    def __init__(self, label: str):
        self.label = label


class FakeExample:
    def __init__(self, subject: str, label: str):
        self.subject = subject
        self.label = label


def test_metric_correct():
    out = metric_with_feedback(FakeExample("hi", "personal"), FakePred("Personal "))
    assert out.score == 1.0
    assert "Correct" in out.feedback


def test_metric_incorrect_includes_actionable_feedback():
    out = metric_with_feedback(FakeExample("50% off!", "promo"), FakePred("transactional"))
    assert out.score == 0.0
    assert "expected 'promo'" in out.feedback
    assert "50% off!" in out.feedback


def test_load_examples_sets_inputs(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"subject": "s", "body": "b", "label": "work"}) + "\n")
    examples = load_examples(path)
    assert len(examples) == 1
    assert set(examples[0].inputs().keys()) == {"subject", "body"}
    assert examples[0].label == "work"
