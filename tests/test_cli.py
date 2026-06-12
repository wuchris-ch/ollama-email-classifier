"""CLI tests with the classifier mocked — no Ollama required."""
from __future__ import annotations

import json
import sys
from unittest.mock import patch

import src.cli as cli


class FakePred:
    def __init__(self, label: str):
        self.label = label


class FakeClassifier:
    def load(self, path: str) -> None:
        pass

    def __call__(self, subject: str, body: str) -> FakePred:
        return FakePred("transactional" if "order" in subject.lower() else "Personal ")


def run_cli(argv, capsys):
    with patch.object(cli, "configure_ollama_lm"), \
         patch.object(cli, "EmailClassifier", FakeClassifier), \
         patch.object(sys, "argv", ["cli", *argv]):
        cli.main()
    return capsys.readouterr().out


def test_single_email(capsys):
    out = run_cli(["--subject", "Order #99 shipped", "--body", "on the way"], capsys)
    assert json.loads(out) == {"subject": "Order #99 shipped", "label": "transactional"}


def test_label_is_normalized(capsys):
    out = run_cli(["--subject", "hey", "--body", "coffee?"], capsys)
    assert json.loads(out)["label"] == "personal"


def test_batch_jsonl(tmp_path, capsys):
    path = tmp_path / "in.jsonl"
    path.write_text(
        json.dumps({"subject": "order #1", "body": "x"}) + "\n"
        + json.dumps({"subject": "hi", "body": "y"}) + "\n"
    )
    out = run_cli(["--jsonl", str(path)], capsys)
    labels = [json.loads(line)["label"] for line in out.strip().splitlines()]
    assert labels == ["transactional", "personal"]


def test_requires_some_input(capsys):
    import pytest

    with patch.object(sys, "argv", ["cli"]):
        with pytest.raises(SystemExit):
            cli.main()


def test_unknown_label_falls_back_to_spam():
    class WeirdClassifier:
        def __call__(self, subject: str, body: str):
            return FakePred("not-a-label")

    result = cli.classify_one(WeirdClassifier(), "s", "b")
    assert result["label"] == "spam"
