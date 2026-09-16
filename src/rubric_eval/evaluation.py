"""Scoring cases: judge every criterion, contain the failures, fold them into scores.

The layer between the judge and the transport, and the only place that speaks both
languages: a `Judge` answers with a `Verdict` or raises, a caller wants a complete
`CaseResult`. Nothing here knows about HTTP, so a CLI or a notebook uses the exact same
entry points as `POST /evaluate` and `POST /evaluate/batch`.

Three public functions, one fan-out: `evaluate_case` scores a single answer,
`evaluate_batch` scores many of them and adds the run metrics on top, and
`filter_cases_by_labels` picks the subset of a catalog to hand either of them.
"""

import asyncio

from rubric_eval.judge import Judge
from rubric_eval.metrics import case_score, label_metrics, run_metrics
from rubric_eval.models import (
    Batch,
    BatchResult,
    Case,
    CaseResult,
    Criterion,
    CriterionResult,
    matches_label_selection,
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
            it — a judge per call would multiply its budget by the number of callers. Its
            `scale` decides how the raw scores are read, and is stored on the result.
        case: Question, answer under test and rubric. Pydantic has already guaranteed at
            least one criterion, unique criterion ids and positive finite weights, so
            nothing here re-checks them.

    Returns:
        A complete `CaseResult`, always — never a partial one. `criterion_results` is in
        rubric order, so it can be zipped with `case.criteria`; `score` is the weighted fold
        over it, in [0, 1]. A criterion the judge could not answer for is still in the list,
        marked `failed=True` with `score=0.0` and the cause in its `reasoning`. The case's
        `labels` are copied over untouched — they did not reach the judge and cannot have
        moved the score.

    Raises:
        Whatever the judge raises from `_BUGS_NOT_OUTAGES` — a broken program must not be
        reported as a plausible score. `asyncio.CancelledError` propagates as well, so a
        disconnected client aborts the work instead of billing a full evaluation for it.
        AttributeError for a judge that declares no `scale`, and ValueError for one that
        grades above its own — both are bugs in the judge, not outages of its endpoint.
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
    return CaseResult(
        case_id=case.id,
        score=case_score(results, judge.scale),
        scale=judge.scale,
        criterion_results=results,
        labels=case.labels,
    )


async def evaluate_batch(judge: Judge, batch: Batch) -> BatchResult:
    """Score a whole catalog of answers and aggregate the case scores into run metrics.

    A fan-out over `evaluate_case` plus `run_metrics` and `label_metrics`, and deliberately
    nothing else: one entry of `BatchResult.case_results` is exactly what `evaluate_case`
    returns for that case, so the single and the batch path cannot drift apart.

    Which cases run is the batch's own business: `evaluate_batch` judges
    `batch.selected_cases`, so handing it a whole catalog and a `label_filter` runs the
    subset and records what picked it. An empty selection runs everything.

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
        batch: At least one case, with unique case ids (Pydantic has checked both). Its
            `label_filter` decides which of them run — also already checked, so a selection
            matching no case never reaches here.

    Returns:
        A `BatchResult`: one `case_results` entry per **selected** case in request order —
        fewer than `batch.cases` when a `label_filter` narrowed the run — `metrics` over them,
        `label_metrics` the same aggregate once per label the cases carry, and `label_filter`
        echoed from the batch. A judge outage never aborts the run — it lowers the scores and
        is counted in `RunMetrics.failed_criteria_count`, which is the field to read before
        the average.

    Raises:
        As `evaluate_case`. A programming error in any single case aborts the whole batch:
        one silently wrong case makes every number computed alongside it untrustworthy.

    Example:
        run = await evaluate_batch(judge, Batch(cases=[case_a, case_b]))
        run.metrics.average_score           # 0.5
        run.metrics.cases_with_score_zero   # [2]  — the answers to read first
        run.label_metrics[0].label          # "table"
        run.case_results[0].score           # 1.0  — every single result is still there
    """
    results = await asyncio.gather(
        *(evaluate_case(judge, case) for case in batch.selected_cases)
    )
    return BatchResult(
        metrics=run_metrics(results),
        label_metrics=label_metrics(results),
        label_filter=batch.label_filter,
        case_results=results,
    )


def filter_cases_by_labels(cases: list[Case], selection: list[list[str]]) -> list[Case]:
    """Pick the cases a `LabelSelection` covers — any group, every label of it.

    The same rule `Batch.selected_cases` runs by and the per-label metrics bucket by, so "the
    run selected by `table`" and "the `table` bucket of the full run" are the same cases.

    You rarely need this to *run* a subset — hand `Batch` the catalog and the selection and it
    does exactly this. Reach for it to see what a selection would pick before spending a judge
    call on it, or to select by something a `Batch` never sees.

    A filter, and it behaves like one: no match is an empty list, not an exception, so it
    composes. `Batch` is what refuses to *run* an empty selection.

    Args:
        cases: The catalog to select from; returned in its own order, never reordered.
        selection: Groups of required labels, read as an OR of ANDs. Empty selects every
            case.

    Returns:
        The matching cases, empty when none match.

    Example:
        filter_cases_by_labels(catalog, [["table", "split_infos"], ["agentic"]])
    """
    if any(isinstance(group, str) for group in selection):
        raise TypeError(
            "a selection is a list of label *groups*, not a list of labels: pass "
            f"{[list(selection)]} to require all of them, or "
            f"{[[label] for label in selection]} to require any of them"
        )
    return [case for case in cases if matches_label_selection(case.labels, selection)]


async def _judge_criterion(judge: Judge, case: Case, criterion: Criterion) -> CriterionResult:
    """Turn one `Verdict` — or one dead judge — into one `CriterionResult`.

    This is where the failure policy lives: an endpoint that refuses, times out, runs out of
    quota or never produces a parseable reply costs exactly one criterion, which then counts
    as 0 and keeps its weight. A bug re-raises instead (see `_BUGS_NOT_OUTAGES`).

    The scale is read up front, outside the containment, so a judge that declares none fails
    as the program error it is rather than as an outage that happens to score 0.
    """
    scale = judge.scale
    try:
        verdict = await judge.score(case.question, case.answer, criterion)
    except _BUGS_NOT_OUTAGES:
        raise
    except Exception as error:  # noqa: BLE001 — a dead judge must not kill the whole case
        return CriterionResult.unjudged(criterion, _describe(error))
    return CriterionResult.judged(criterion, verdict.score, verdict.reasoning, scale)


def _describe(error: Exception) -> str:
    """The cause to report in `CriterionResult.reasoning`, never an empty string.

    `str(TimeoutError())` *is* the empty string, and a timeout is the likeliest judge failure
    of all — then the class name is the only cause there is to report.
    """
    return str(error) or type(error).__name__
