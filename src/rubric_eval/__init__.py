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
    CaseComparison,
    CaseResult,
    ChangeMagnitude,
    ChangeStatus,
    ChangeSummary,
    Comparison,
    ComparisonResult,
    Criterion,
    CriterionComparison,
    CriterionResult,
    RunMetrics,
    RunMetricsDelta,
)

__all__ = [
    "PRESENCE_THRESHOLD",
    "SCALE_MAX",
    "SCORE_EQUALITY_TOLERANCE",
    "WEAKEST_CASES_REPORTED",
    "Batch",
    "BatchResult",
    "Case",
    "CaseComparison",
    "CaseResult",
    "ChangeMagnitude",
    "ChangeStatus",
    "ChangeSummary",
    "Comparison",
    "ComparisonResult",
    "Criterion",
    "CriterionComparison",
    "CriterionResult",
    "Judge",
    "JudgeConfig",
    "OpenAIJudge",
    "RunMetrics",
    "RunMetricsDelta",
    "RunsNotComparableError",
    "Verdict",
    "case_score",
    "compare_runs",
    "evaluate_batch",
    "evaluate_case",
    "run_metrics",
]
