"""rubric-eval — judge LLM answers against weighted reference criteria."""

from rubric_eval.evaluation import evaluate_batch, evaluate_case
from rubric_eval.judge import Judge, JudgeConfig, OpenAIJudge, Verdict
from rubric_eval.metrics import case_score, run_metrics
from rubric_eval.models import (
    PRESENCE_THRESHOLD,
    SCALE_MAX,
    WEAKEST_CASES_REPORTED,
    Batch,
    BatchResult,
    Case,
    CaseResult,
    Criterion,
    CriterionResult,
    RunMetrics,
)

__all__ = [
    "PRESENCE_THRESHOLD",
    "SCALE_MAX",
    "WEAKEST_CASES_REPORTED",
    "Batch",
    "BatchResult",
    "Case",
    "CaseResult",
    "Criterion",
    "CriterionResult",
    "Judge",
    "JudgeConfig",
    "OpenAIJudge",
    "RunMetrics",
    "Verdict",
    "case_score",
    "evaluate_batch",
    "evaluate_case",
    "run_metrics",
]
