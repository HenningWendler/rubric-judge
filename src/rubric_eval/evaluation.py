"""Scoring cases: judge every criterion, contain the failures, fold them into scores.

The layer between the judge and the transport, and the only place that speaks both
languages: a `Judge` answers with a `Verdict` or raises, a caller wants a complete
`CaseResult`. Nothing here knows about HTTP, so a CLI or a notebook uses the exact same
entry points as `POST /evaluate` and `POST /evaluate/batch`.

Two public functions, one fan-out: `evaluate_case` scores a single answer,
`evaluate_batch` scores many of them and adds the run metrics on top.
"""

import asyncio

from rubric_eval.judge import Judge
from rubric_eval.metrics import case_score, run_metrics
from rubric_eval.models import (
    Batch,
    BatchResult,
    Case,
    CaseResult,
    Criterion,
    CriterionResult,
)

#: Errors that mean *this program* is wrong, not that the judge's endpoint is having a bad
#: day: a typo, a bad call, asyncio used incorrectly. They are re-raised instead of contained,
#: because scoring a criterion 0 over a bug produces a plausible-looking number that nobody
#: can tell from a real result — far worse than a 500. Everything else, including whatever a
#: third-party `Judge` raises for an outage, still costs one criterion.
_BUGS_NOT_OUTAGES = (TypeError, AttributeError, NameError, ImportError, RuntimeError)


async def evaluate_case(judge: Judge, case: Case) -> CaseResult:
    """Score one answer against its rubric — one judge call per criterion, all in parallel.

    Args:
        judge: Scores one criterion at a time. It also owns the concurrency limit: this
            function starts one task per criterion whatever the rubric's size, because only
            the implementation knows what its backend tolerates. Build one judge and share
            it — a judge per call would multiply its budget by the number of callers.
        case: Question, answer under test and rubric. Pydantic has already guaranteed at
            least one criterion, unique criterion ids and positive finite weights, so
            nothing here re-checks them.

    Returns:
        A complete `CaseResult`, always — never a partial one. `criterion_results` is in
        rubric order, so it can be zipped with `case.criteria`; `score` is the weighted fold
        over it, in [0, 1]. A criterion the judge could not answer for is still in the list,
        marked `failed=True` with `score=0.0` and the cause in its `reasoning`.

    Raises:
        Whatever the judge raises from `_BUGS_NOT_OUTAGES` — a broken program must not be
        reported as a plausible score. `asyncio.CancelledError` propagates as well, so a
        disconnected client aborts the work instead of billing a full evaluation for it.
        Endpoint failures are *not* raised; see `_judge_criterion`.

    Example:
        result = await evaluate_case(judge, Case(
            id=1,
            question="How do I report sick leave?",
            answer="Email hr@example.com before 10:00.",
            criteria=[Criterion(id=1, content="Report by email before 10:00", weight=3)],
        ))
        result.score                             # 1.0
        result.criterion_results[0].reasoning    # "The answer instructs the reader to ..."
    """
    results = await asyncio.gather(
        *(_judge_criterion(judge, case, criterion) for criterion in case.criteria)
    )
    return CaseResult(case_id=case.id, score=case_score(results), criterion_results=results)


async def evaluate_batch(judge: Judge, batch: Batch) -> BatchResult:
    """Score a whole catalog of answers and aggregate the case scores into run metrics.

    A fan-out over `evaluate_case` plus `run_metrics`, and deliberately nothing else: one
    entry of `BatchResult.case_results` is exactly what `evaluate_case` returns for that
    case, so the single and the batch path cannot drift apart.

    Concurrency — no second throttle is applied here on purpose. The cases fan out *and*
    every case fans out over its criteria, so the coroutines multiply (50 cases x 10
    criteria is 500 of them), but coroutines are not connections: the judge's
    `max_concurrent` still caps how many calls are actually in flight. That budget belongs
    to the judge instance, so the same cap holds across the cases of one batch, across
    concurrent batches and across parallel HTTP requests — as long as one judge serves them
    all. Tune it on the judge, never by shrinking the batch.

    Args:
        judge: As for `evaluate_case`. The same instance serves every case of the batch,
            which is what makes the shared limit above work.
        batch: At least one case, with unique case ids (Pydantic has checked both).

    Returns:
        A `BatchResult`: `case_results` in request order, and `metrics` aggregated over
        them. A judge outage never aborts the run — it lowers the scores and is counted in
        `RunMetrics.failed_criteria_count`, which is the field to read before the average.

    Raises:
        As `evaluate_case`. A programming error in any single case aborts the whole batch:
        one silently wrong case makes every number computed alongside it untrustworthy.

    Example:
        run = await evaluate_batch(judge, Batch(cases=[case_a, case_b]))
        run.metrics.average_score           # 0.5
        run.metrics.cases_with_score_zero   # [2]  — the answers to read first
        run.case_results[0].score           # 1.0  — every single result is still there
    """
    results = await asyncio.gather(*(evaluate_case(judge, case) for case in batch.cases))
    return BatchResult(metrics=run_metrics(results), case_results=results)


async def _judge_criterion(judge: Judge, case: Case, criterion: Criterion) -> CriterionResult:
    """Turn one `Verdict` — or one dead judge — into one `CriterionResult`.

    This is where the failure policy lives: an endpoint that refuses, times out, runs out of
    quota or never produces a parseable reply costs exactly one criterion, which then counts
    as 0 and keeps its weight. A bug re-raises instead (see `_BUGS_NOT_OUTAGES`).
    """
    try:
        verdict = await judge.score(case.question, case.answer, criterion)
    except _BUGS_NOT_OUTAGES:
        raise
    except Exception as error:  # noqa: BLE001 — a dead judge must not kill the whole case
        return CriterionResult.unjudged(criterion, _describe(error))
    return CriterionResult.judged(criterion, verdict.score, verdict.reasoning)


def _describe(error: Exception) -> str:
    """The cause to report in `CriterionResult.reasoning`, never an empty string.

    `str(TimeoutError())` *is* the empty string, and a timeout is the likeliest judge failure
    of all — then the class name is the only cause there is to report.
    """
    return str(error) or type(error).__name__
