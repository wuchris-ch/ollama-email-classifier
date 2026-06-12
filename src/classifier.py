"""DSPy email classifier wired to a local Ollama backend."""
from __future__ import annotations

import os
from typing import Literal

import dspy
from dotenv import load_dotenv

load_dotenv()

Label = Literal["personal", "work", "finance", "promo", "transactional", "spam"]
LABELS: tuple[str, ...] = ("personal", "work", "finance", "promo", "transactional", "spam")


class EmailClassification(dspy.Signature):
    """Classify an email into exactly one of: personal, work, finance, promo, transactional, spam.

    personal: friends, family, social plans, casual.
    work: internal team comms, code reviews, planning, 1:1s.
    finance: bank statements, balance alerts, tax, investments.
    promo: marketing, sales, newsletters, abandoned-cart.
    transactional: order confirmations, shipping, receipts, 2FA, password resets.
    spam: phishing, scams, advance-fee fraud, fake invoices.
    """

    subject: str = dspy.InputField()
    body: str = dspy.InputField()
    label: Label = dspy.OutputField()


class EmailClassifier(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classify = dspy.ChainOfThought(EmailClassification)

    def forward(self, subject: str, body: str) -> dspy.Prediction:
        return self.classify(subject=subject, body=body)


def classifier_lm_name() -> str:
    """LiteLLM model string for the classifier: local Ollama unless CLASSIFIER_LM overrides."""
    override = os.getenv("CLASSIFIER_LM")
    if override:
        return override
    return f"ollama_chat/{os.getenv('OLLAMA_MODEL', 'qwen2.5:7b-instruct')}"


def configure_ollama_lm() -> dspy.LM:
    """Wire DSPy to the classifier LM.

    Defaults to the local Ollama instance (OLLAMA_BASE_URL / OLLAMA_MODEL).
    Set CLASSIFIER_LM to a full LiteLLM model string
    (e.g. deepseek/deepseek-v4-pro) to run on a remote API instead —
    reasoning models need the larger max_tokens budget.
    """
    if os.getenv("CLASSIFIER_LM"):
        lm = dspy.LM(classifier_lm_name(), temperature=0.0, max_tokens=8000)
    else:
        lm = dspy.LM(
            classifier_lm_name(),
            api_base=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            api_key="",
            temperature=0.0,
            max_tokens=512,
        )
    dspy.configure(lm=lm)
    return lm


def configure_judge_lm() -> dspy.LM:
    """Build the judge LM. JUDGE_MODEL may be a bare OpenAI model name
    (gpt-4o-mini) or a full LiteLLM string (deepseek/deepseek-v4-pro)."""
    model = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
    if "/" not in model:
        model = f"openai/{model}"
    return dspy.LM(model, temperature=0.0, max_tokens=8000)
