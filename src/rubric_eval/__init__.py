"""rubric-eval — judge LLM answers against weighted reference criteria."""

from rubric_eval.comparison import RunsNotComparableError, compare_runs
from rubric_eval.evaluation import evaluate_batch, evaluate_case
from rubric_eval.judge import Judge, JudgeConfig, OpenAIJudge, Verdict
from rubric_eval.metrics import case_score, run_metrics
from rubric_eval.models import (
    PRESENCE_THRESHOLD,
    SCALE_MAX,
    SCORE_EQUALITY_TOLERANCE,
    WEAKEST_CASES_REPORTED,
    Batch,
    BatchResult,
    Case,
    CaseComparisonResult,
    CaseResult,
    ChangeMagnitude,
    ChangeStatus,
    ChangeSummary,
    ComparisonResult,
    Criterion,
    CriterionComparisonResult,
    CriterionResult,
    RunMetrics,
    RunMetricsDelta,
    RunPair,
)

__all__ = [
    "PRESENCE_THRESHOLD",
    "SCALE_MAX",
    "SCORE_EQUALITY_TOLERANCE",
    "WEAKEST_CASES_REPORTED",
    "Batch",
    "BatchResult",
    "Case",
    "CaseComparisonResult",
    "CaseResult",
    "ChangeMagnitude",
    "ChangeStatus",
    "ChangeSummary",
    "ComparisonResult",
    "Criterion",
    "CriterionComparisonResult",
    "CriterionResult",
    "Judge",
    "JudgeConfig",
    "OpenAIJudge",
    "RunMetrics",
    "RunMetricsDelta",
    "RunPair",
    "RunsNotComparableError",
    "Verdict",
    "case_score",
    "compare_runs",
    "evaluate_batch",
    "evaluate_case",
    "run_metrics",
]
