"""Scoring cases: judge every criterion and fold the results into scores.

The layer between the judge and the transport, and the only place that speaks both
languages: a `Judge` answers with a `Verdict` or raises, a caller wants a complete
`CaseResult`. Nothing here knows about HTTP, so a CLI or a notebook uses the exact same
entry points as `POST /evaluate` and `POST /evaluate/run`.

A judge that cannot answer invalidates everything it was judging: there is no result with a
hole in it, because a criterion scored 0 for want of a grade is indistinguishable from one
the answer really missed.

Three public functions, one fan-out: `evaluate_case` scores a single answer,
`evaluate_run` scores many of them and adds the run metrics on top, and
`filter_cases_by_labels` picks the subset of a catalog to hand either of them.
"""

import asyncio

from rubric_eval.judge import Judge
from rubric_eval.metrics import case_score, label_metrics, run_metrics
from rubric_eval.models import (
    Case,
    CaseResult,
    Criterion,
    CriterionResult,
    Run,
    RunResult,
    matches_label_filter,
    read_label_filter,
)


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
        A complete `CaseResult` or nothing at all — never a partial one. `criterion_results`
        is in rubric order, so it can be zipped with `case.criteria`; `score` is the weighted
        fold over it, in [0, 1]. The case's `labels` are copied over untouched — they did not
        reach the judge and cannot have moved the score.

    Raises:
        JudgeUnavailableError: The judge could not answer for one of the criteria. The case
            has no score then and does not get one: a fabricated 0 would be a plausible
            number nobody can tell from a real result.
        Exception: Whatever else the judge raises — a broken program must not be reported as
            a plausible score either. `AttributeError` for a judge that declares no `scale`,
            and `ValueError` for one that grades above its own, are the two this layer
            provokes itself. `asyncio.CancelledError` propagates as well, so a disconnected
            client aborts the work instead of billing a full evaluation for it.

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
    criterion_results = await asyncio.gather(
        *(_judge_criterion(judge, case, criterion) for criterion in case.criteria)
    )
    return CaseResult(
        case_id=case.id,
        score=case_score(criterion_results, judge.scale),
        scale=judge.scale,
        criterion_results=criterion_results,
        labels=case.labels,
    )


async def evaluate_run(judge: Judge, run: Run) -> RunResult:
    """Score a whole catalog of answers and aggregate the case scores into run metrics.

    A fan-out over `evaluate_case` plus `run_metrics` and `label_metrics`, and deliberately
    nothing else: one entry of `RunResult.case_results` is exactly what `evaluate_case`
    returns for that case, so the single and the run path cannot drift apart.

    Which cases run is the `Run`'s own business: `evaluate_run` judges
    `run.selected_cases`, so handing it a whole catalog and a `label_filter` runs the
    subset and records what picked it. An empty label filter runs everything.

    Concurrency — no second throttle is applied here on purpose. The cases fan out *and*
    every case fans out over its criteria, so the coroutines multiply (50 cases x 10
    criteria is 500 of them), but coroutines are not connections: the judge's
    `max_concurrent` still caps how many calls are actually in flight. That budget belongs
    to the judge instance, so the same cap holds across the cases of one run, across
    concurrent runs and across parallel HTTP requests — as long as one judge serves them
    all. Tune it on the judge, never by shrinking the run.

    Args:
        judge: As for `evaluate_case`. The same instance serves every case of the run,
            which is what makes the shared limit above work.
        run: At least one case, with unique case ids (Pydantic has checked both). Its
            `label_filter` decides which of them run — also already checked, so a filter
            matching no case never reaches here.

    Returns:
        A `RunResult`: one `case_results` entry per **selected** case in request order —
        fewer than `run.cases` when a `label_filter` narrowed the run — `metrics` over them,
        `label_metrics` the same aggregate once per label the cases carry, and
        `applied_label_filter` recording what picked them. Every selected case is in it, or
        the run raised instead.

    Raises:
        JudgeUnavailableError: As `evaluate_case`. One criterion the judge could not answer
            for ends the whole run: the metrics are an average over the cases, so one case
            missing a grade makes every number computed alongside it untrustworthy.
        Exception: As `evaluate_case`, and for the same reason — a programming error in any
            single case aborts the run rather than being averaged into it.

    Example:
        run_result = await evaluate_run(judge, Run(cases=[case_a, case_b]))
        run_result.metrics.average_score           # 0.5
        run_result.metrics.cases_with_score_zero   # [2]  — the answers to read first
        run_result.label_metrics[0].label          # "table"
        run_result.case_results[0].score           # 1.0  — every result is still there
    """
    case_results = await asyncio.gather(
        *(evaluate_case(judge, case) for case in run.selected_cases)
    )
    return RunResult(
        metrics=run_metrics(case_results),
        label_metrics=label_metrics(case_results),
        applied_label_filter=run.label_filter,
        case_results=case_results,
    )


def filter_cases_by_labels(cases: list[Case], label_filter: list[list[str]]) -> list[Case]:
    """Pick the cases a `LabelFilter` covers — any group, every label of it.

    The same rule `Run.selected_cases` runs by and the per-label metrics bucket by, so "the
    run selected by `table`" and "the `table` bucket of the full run" are the same cases.

    You rarely need this to *run* a subset — hand `Run` the catalog and the label filter
    and it does exactly this. Reach for it to see what a label filter would pick before
    spending a judge call on it, or to select by something a `Run` never sees.

    A filter, and it behaves like one: no match is an empty list, not an exception, so it
    composes. `Run` is what refuses to *run* an empty label filter.

    The label filter is read as the `LabelFilter` a `Run` would read it as, so a preview and
    the run it previews can never pick different cases.

    Args:
        cases: The catalog to select from; returned in its own order, never reordered.
        label_filter: Groups of required labels, read as an OR of ANDs. Empty selects every
            case.

    Returns:
        The matching cases, empty when none match.

    Raises:
        TypeError: For a flat `["table"]`, which would otherwise compare *characters* and
            quietly return the wrong cases — the message names both readings you may have
            meant.
        pydantic.ValidationError: For a label filter a `Run` would refuse too — a blank
            label, or the same group twice.

    Example:
        filter_cases_by_labels(catalog, [["table", "split_infos"], ["agentic"]])
    """
    if any(isinstance(group, str) for group in label_filter):
        raise TypeError(
            "a label filter is a list of label *groups*, not a list of labels: pass "
            f"{[list(label_filter)]} to require all of them, or "
            f"{[[label] for label in label_filter]} to require any of them"
        )
    label_filter = read_label_filter(label_filter)
    return [case for case in cases if matches_label_filter(case.labels, label_filter)]


async def _judge_criterion(judge: Judge, case: Case, criterion: Criterion) -> CriterionResult:
    """Turn one `Verdict` into one `CriterionResult`, on the scale its judge declares.

    Nothing is caught here: the judge decides what its failures mean by which exception it
    raises, and both answers — the run is invalid, or the program is broken — travel up.
    """
    verdict = await judge.score(case.question, case.answer, criterion)
    return CriterionResult.judged(criterion, verdict.score, verdict.reasoning, judge.scale)
