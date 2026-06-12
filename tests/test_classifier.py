"""Smoke tests that run without Ollama or OpenAI by mocking the LM."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

DATA = Path(__file__).resolve().parents[1] / "data"


def test_dataset_files_exist_and_are_well_formed():
    for split in ("train", "dev", "test"):
        path = DATA / f"{split}.jsonl"
        assert path.exists(), f"missing {path}"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert rows, f"{split}.jsonl is empty"
        for r in rows:
            assert set(r) == {"subject", "body", "label"}
            assert r["label"] in {"personal", "work", "finance", "promo", "transactional", "spam"}


def test_classifier_imports_and_signature_shape():
    from src.classifier import LABELS, EmailClassification

    assert LABELS == ("personal", "work", "finance", "promo", "transactional", "spam")
    fields = EmailClassification.model_fields
    assert "subject" in fields and "body" in fields and "label" in fields


def test_predict_all_normalizes_unknown_labels():
    from src.classifier import EmailClassifier
    from src.eval import predict_all

    class FakePred:
        def __init__(self, label):
            self.label = label

    fake = EmailClassifier()
    with patch.object(fake, "classify", side_effect=lambda **kw: FakePred("PERSONAL")):
        preds = predict_all(fake, [{"subject": "hi", "body": "yo"}])
    assert preds == ["personal"]

    with patch.object(fake, "classify", side_effect=lambda **kw: FakePred("nonsense")):
        preds = predict_all(fake, [{"subject": "hi", "body": "yo"}])
    assert preds == ["spam"]
