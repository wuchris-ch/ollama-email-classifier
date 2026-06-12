"""Eval pipeline tests with the LM fully mocked — no Ollama, no OpenAI."""
from __future__ import annotations

import json
from unittest.mock import patch

import src.eval as eval_mod
from src.classifier import LABELS
from src.eval import format_confusion_matrix, sample_outputs


class FakePred:
    def __init__(self, label: str):
        self.label = label


class FakeClassifier:
    """Predicts the true label except for one planted mistake: promo -> transactional."""

    def __init__(self) -> None:
        self.miss_planted = False

    def load(self, path: str) -> None:  # pragma: no cover - interface parity
        pass

    def __call__(self, subject: str, body: str) -> FakePred:
        if "PLANTED-MISS" in subject:
            return FakePred("transactional")
        return FakePred(subject.split("|")[0])


def fake_dataset(tmp_path):
    rows = [{"subject": f"{label}|email {i}", "body": "b", "label": label}
            for label in LABELS for i in range(3)]
    rows.append({"subject": "promo|PLANTED-MISS", "body": "b", "label": "promo"})
    (tmp_path / "test.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    return rows


def test_evaluate_end_to_end_mocked(tmp_path, monkeypatch):
    rows = fake_dataset(tmp_path)
    monkeypatch.setattr(eval_mod, "DATA", tmp_path)
    monkeypatch.setattr(eval_mod, "RESULTS", tmp_path / "results")

    with patch.object(eval_mod, "configure_ollama_lm"), \
         patch.object(eval_mod, "EmailClassifier", FakeClassifier):
        result = eval_mod.evaluate("test", classifier_path=None,
                                   out_name="eval_test", run_judge=False)

    n = len(rows)
    assert result["n"] == n
    assert result["accuracy"] == round((n - 1) / n, 4)
    assert 0 < result["macro_f1"] < 1
    assert result["judge_agreement"] is None
    assert result["judge_model"] is None

    cm = result["confusion_matrix"]
    assert cm["labels"] == list(LABELS)
    promo_row = cm["matrix"][LABELS.index("promo")]
    assert promo_row[LABELS.index("transactional")] == 1  # the planted miss
    assert sum(sum(r) for r in cm["matrix"]) == n

    # samples lead with the mistake
    assert result["samples"][0]["truth"] == "promo"
    assert result["samples"][0]["predicted"] == "transactional"

    written = json.loads((tmp_path / "results" / "eval_test.json").read_text())
    assert written["accuracy"] == result["accuracy"]


def test_evaluate_with_mocked_judge(tmp_path, monkeypatch):
    fake_dataset(tmp_path)
    monkeypatch.setattr(eval_mod, "DATA", tmp_path)
    monkeypatch.setattr(eval_mod, "RESULTS", tmp_path / "results")

    # judge agrees with the classifier on everything
    with patch.object(eval_mod, "configure_ollama_lm"), \
         patch.object(eval_mod, "EmailClassifier", FakeClassifier), \
         patch.object(eval_mod, "judge_all", lambda judge, exs, preds: [True] * len(preds)):
        result = eval_mod.evaluate("test", classifier_path=None,
                                   out_name="eval_judged", run_judge=True)

    assert result["judge_agreement"] == 1.0
    # judge said "correct" on the one true miss, so alignment dips below 1
    assert result["judge_vs_truth_alignment"] == result["accuracy"]


def test_evaluate_limit(tmp_path, monkeypatch):
    fake_dataset(tmp_path)
    monkeypatch.setattr(eval_mod, "DATA", tmp_path)
    monkeypatch.setattr(eval_mod, "RESULTS", tmp_path / "results")

    with patch.object(eval_mod, "configure_ollama_lm"), \
         patch.object(eval_mod, "EmailClassifier", FakeClassifier):
        result = eval_mod.evaluate("test", classifier_path=None,
                                   out_name="eval_lim", run_judge=False, limit=5)
    assert result["n"] == 5


def test_sample_outputs_misses_first():
    examples = [
        {"subject": "a", "body": "", "label": "work"},
        {"subject": "b", "body": "", "label": "spam"},
        {"subject": "c", "body": "", "label": "promo"},
    ]
    preds = ["work", "promo", "promo"]  # one miss in the middle
    samples = sample_outputs(examples, preds, k=2)
    assert samples[0] == {"subject": "b", "truth": "spam", "predicted": "promo"}
    assert len(samples) == 2


def test_format_confusion_matrix_shape():
    text = format_confusion_matrix(["a", "bb"], [[1, 2], [3, 4]])
    lines = text.splitlines()
    assert len(lines) == 3
    assert lines[1].startswith("a")
    assert lines[2].startswith("bb")


def test_judge_all_counts_errors_as_incorrect(monkeypatch):
    class ExplodingJudge:
        def __call__(self, **kw):
            raise RuntimeError("api down")

    monkeypatch.setattr(eval_mod, "configure_judge_lm", lambda: None)
    verdicts = eval_mod.judge_all(ExplodingJudge(), [{"subject": "s", "body": "b"}], ["work"])
    assert verdicts == [False]


def test_judge_all_restores_previous_lm(monkeypatch):
    import dspy

    sentinel = object()
    monkeypatch.setattr(eval_mod, "configure_judge_lm", lambda: sentinel)
    dspy.configure(lm=None)

    class OkJudge:
        def __call__(self, **kw):
            assert dspy.settings.lm is sentinel
            return type("P", (), {"is_correct": True})()

    verdicts = eval_mod.judge_all(OkJudge(), [{"subject": "s", "body": "b"}], ["work"])
    assert verdicts == [True]
    assert dspy.settings.lm is None
