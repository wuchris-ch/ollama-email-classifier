"""Optional live integration test against a running Ollama instance.

Off by default so CI and contributors never need a GPU. Enable with:

    RUN_OLLAMA_TESTS=1 pytest tests/test_integration_ollama.py -v

Respects OLLAMA_BASE_URL / OLLAMA_MODEL from the environment or .env.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_OLLAMA_TESTS"),
    reason="live Ollama test; set RUN_OLLAMA_TESTS=1 to enable",
)


def test_classifies_an_obvious_email():
    from src.classifier import LABELS, EmailClassifier, configure_ollama_lm

    configure_ollama_lm()
    pred = EmailClassifier()(
        subject="Your verification code is 482910",
        body="Your two-factor code is 482910. It expires in 10 minutes. Don't share this code.",
    )
    label = str(pred.label).strip().lower()
    assert label in LABELS
    assert label == "transactional"


def test_classifies_obvious_spam():
    from src.classifier import EmailClassifier, configure_ollama_lm

    configure_ollama_lm()
    pred = EmailClassifier()(
        subject="URGENT: account suspended",
        body="Your account has been suspended. Click here and confirm your card number and PIN within 24 hours.",
    )
    assert str(pred.label).strip().lower() == "spam"
