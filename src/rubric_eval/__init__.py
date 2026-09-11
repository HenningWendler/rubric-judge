"""rubric-eval — judge LLM answers against weighted reference criteria."""

from rubric_eval.evaluation import evaluate_case
from rubric_eval.judge import Judge, JudgeConfig, OpenAIJudge, Verdict
from rubric_eval.metrics import case_score
from rubric_eval.models import (
    PRESENCE_THRESHOLD,
    SCALE_MAX,
    Criterion,
    CriterionResult,
    EvaluateRequest,
    EvaluationResult,
)

__all__ = [
    "PRESENCE_THRESHOLD",
    "SCALE_MAX",
    "Criterion",
    "CriterionResult",
    "EvaluateRequest",
    "EvaluationResult",
    "Judge",
    "JudgeConfig",
    "OpenAIJudge",
    "Verdict",
    "case_score",
    "evaluate_case",
]
