"""Evaluates one case: judge every criterion, contain the failures, fold them into one score.

The layer between the judge and the transport, and the only place that speaks both
languages: a `Judge` answers with a `Verdict` or raises, a caller wants a complete
`EvaluationResult`. Nothing here knows about HTTP, so a CLI or a notebook uses the
exact same entry point as `POST /evaluate`.
"""

import asyncio

from rubric_eval.judge import Judge
from rubric_eval.metrics import case_score
from rubric_eval.models import (
    Criterion,
    CriterionResult,
    EvaluateRequest,
    EvaluationResult,
)

#: Errors that mean *this program* is wrong, not that the judge's endpoint is having a bad
#: day: a typo, a bad call, asyncio used incorrectly. They are re-raised instead of contained,
#: because scoring a criterion 0 over a bug produces a plausible-looking number that nobody
#: can tell from a real result — far worse than a 500. Everything else, including whatever a
#: third-party `Judge` raises for an outage, still costs one criterion.
_BUGS_NOT_OUTAGES = (TypeError, AttributeError, NameError, ImportError, RuntimeError)


async def evaluate_case(judge: Judge, request: EvaluateRequest) -> EvaluationResult:
    """Judge the whole rubric concurrently — one task per criterion, all of them independent.

    One task per criterion regardless of the rubric's size; tasks are cheap. How many of them
    reach the endpoint at the same time is the judge's decision, not this layer's (see `Judge`).
    """
    results = await asyncio.gather(
        *(_judge_criterion(judge, request, criterion) for criterion in request.criteria)
    )
    return EvaluationResult(score=case_score(results), criteria=results)


async def _judge_criterion(
    judge: Judge, request: EvaluateRequest, criterion: Criterion
) -> CriterionResult:
    """Turn one `Verdict` — or one dead judge — into one `CriterionResult`."""
    try:
        verdict = await judge.score(request.question, request.answer, criterion)
    except _BUGS_NOT_OUTAGES:
        raise
    except Exception as error:  # noqa: BLE001 — a dead judge must not kill the whole case
        return CriterionResult.unjudged(criterion, _describe(error))
    return CriterionResult.judged(criterion, verdict.score, verdict.reasoning)


def _describe(error: Exception) -> str:
    """`str(TimeoutError())` is the empty string, and a timeout is the likeliest judge failure
    of all — then the class name is the only cause there is to report."""
    return str(error) or type(error).__name__
