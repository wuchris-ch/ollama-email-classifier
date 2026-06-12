"""LLM-as-judge: independently scores classifier predictions without ground truth."""
from __future__ import annotations

import dspy


class JudgeEmailLabel(dspy.Signature):
    """Independently judge whether the predicted label fits the email.

    The judge does NOT see the ground-truth label. It reasons from the
    email content alone and decides if the predicted label is the most
    appropriate of the six options.

    Allowed labels: personal, work, finance, promo, transactional, spam.
    Be strict: borderline cases that should be a different label are 'incorrect'.
    """

    subject: str = dspy.InputField()
    body: str = dspy.InputField()
    predicted_label: str = dspy.InputField()
    is_correct: bool = dspy.OutputField()
    reasoning: str = dspy.OutputField()


class Judge(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.judge = dspy.ChainOfThought(JudgeEmailLabel)

    def forward(self, subject: str, body: str, predicted_label: str) -> dspy.Prediction:
        return self.judge(subject=subject, body=body, predicted_label=predicted_label)
