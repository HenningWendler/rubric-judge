"""Holding two finished runs against each other: did the change help, where, and what did it cost.

Pure computation over two `BatchResult` documents — no judge, no network, no cost. Runs
saved to disk months apart compare exactly like runs produced a second ago.

One entry point, `compare_runs`, and one hard rule underneath it: two runs are comparable
only if they cover the same cases with the same rubric and the same weights. Different
weights make the case scores non-commensurable, and a delta between them would look like a
result while meaning nothing.
"""

import statistics
from operator import attrgetter

from rubric_eval.models import (
    BatchResult,
    CaseComparison,
    CaseResult,
    ChangeMagnitude,
    ChangeStatus,
    ChangeSummary,
    Comparison,
    ComparisonResult,
    RunMetrics,
    RunMetricsDelta,
)


class RunsNotComparableError(ValueError):
    """The two runs do not describe the same catalog, so their scores do not subtract.

    A `ValueError`, because that is what it is and what a caller would catch anyway — but a
    named one, so the HTTP layer can tell a refused comparison apart from a `ValidationError`
    or a `StatisticsError`. Both of those are `ValueError` subclasses too, and reporting one
    of them as "your runs are not comparable" would dress a bug up as the caller's mistake.
    """


def compare_runs(comparison: Comparison) -> ComparisonResult:
    """Compare two finished runs of the same catalog at three grains: run, case, criterion.

    Every delta is `candidate - baseline`, so a positive number always means the candidate
    did better — with the two counting fields of `RunMetricsDelta` as the documented
    exception, where fewer is better.

    Takes a `Comparison` rather than two arguments because both sides have the same type:
    a swapped pair would be undetectable and would silently invert the whole document.

    Args:
        comparison: The `baseline` run to compare against and the `candidate` run under
            test. Both must cover the same case ids, the same criterion ids per case and the
            same weights — see Raises. Nothing else is required of them: results loaded back
            from stored JSON compare exactly like results just computed.

    Returns:
        A `ComparisonResult`. `metrics_delta` says whether the run got better, `summary` how
        that is distributed over the cases, and `case_comparisons` — ordered by `case_id` —
        which criterion is responsible. Read `metrics_delta.failed_criteria_count_delta`
        first: anything but 0 means the two runs suffered different amounts of judge outage,
        and every other number is then partly an artefact of that.

    Raises:
        RunsNotComparableError: A `ValueError`. The runs do not describe the same catalog.
            The message names every difference found — cases present on only one side,
            criteria that differ within a shared case, and weights that changed — rather
            than only the first, so one fix can address all of them.

    Example:
        result = compare_runs(Comparison(baseline=last_weeks_run, candidate=todays_run))
        result.metrics_delta.average_score_delta   # +0.084
        result.summary.worsened_case_ids           # [5] — what the win cost
        result.summary.improvement.largest         # +0.31
    """
    _reject_incomparable_runs(comparison.baseline, comparison.candidate)
    case_comparisons = _compare_cases(comparison.baseline, comparison.candidate)
    return ComparisonResult(
        metrics_delta=_metrics_delta(comparison.baseline.metrics, comparison.candidate.metrics),
        summary=_summarize(case_comparisons),
        case_comparisons=case_comparisons,
    )


def _compare_cases(baseline: BatchResult, candidate: BatchResult) -> list[CaseComparison]:
    """One comparison per case, ordered by id — neither run's storage order is canonical,
    so sorting by id gives a document that does not depend on either."""
    baseline_by_id = _case_results_by_id(baseline)
    candidate_by_id = _case_results_by_id(candidate)
    return [
        CaseComparison.between(baseline_by_id[case_id], candidate_by_id[case_id])
        for case_id in sorted(candidate_by_id)
    ]


def _case_results_by_id(run: BatchResult) -> dict[int, CaseResult]:
    """Cases are matched by id, never by position: two runs of the same catalog may well be
    stored in different orders, and zipping those would compare unrelated answers."""
    return {result.case_id: result for result in run.case_results}


def _metrics_delta(baseline: RunMetrics, candidate: RunMetrics) -> RunMetricsDelta:
    """Candidate minus baseline, field by field. `total_cases` is absent because the
    comparability check has already guaranteed it is equal, and the id lists are absent
    because a set of ids does not subtract."""
    return RunMetricsDelta(
        average_score_delta=candidate.average_score - baseline.average_score,
        median_score_delta=candidate.median_score - baseline.median_score,
        variance_delta=candidate.variance - baseline.variance,
        standard_deviation_delta=candidate.standard_deviation - baseline.standard_deviation,
        average_criterion_score_delta=(
            candidate.average_criterion_score - baseline.average_criterion_score
        ),
        criteria_fulfillment_rate_delta=(
            candidate.criteria_fulfillment_rate - baseline.criteria_fulfillment_rate
        ),
        cases_with_score_zero_count_delta=(
            candidate.cases_with_score_zero_count - baseline.cases_with_score_zero_count
        ),
        failed_criteria_count_delta=(
            candidate.failed_criteria_count - baseline.failed_criteria_count
        ),
    )


def _summarize(case_comparisons: list[CaseComparison]) -> ChangeSummary:
    """The case movements folded into the distribution they form: who moved which way, in
    order of how much, and how large those moves were on each side."""
    improved = _ranked_by_movement(case_comparisons, ChangeStatus.IMPROVED)
    worsened = _ranked_by_movement(case_comparisons, ChangeStatus.WORSENED)
    stable = _with_status(case_comparisons, ChangeStatus.STABLE)
    return ChangeSummary(
        improved_case_ids=[case.case_id for case in improved],
        stable_case_ids=[case.case_id for case in stable],
        worsened_case_ids=[case.case_id for case in worsened],
        improvement=_magnitude_of(improved),
        worsening=_magnitude_of(worsened),
    )


def _with_status(
    case_comparisons: list[CaseComparison], status: ChangeStatus
) -> list[CaseComparison]:
    return [case for case in case_comparisons if case.status is status]


def _ranked_by_movement(
    case_comparisons: list[CaseComparison], status: ChangeStatus
) -> list[CaseComparison]:
    """One side of the comparison, biggest move first — for improvements the largest positive
    delta, for regressions the most negative one. Ordering the complete list this way is what
    makes a separate "top five" field unnecessary: the top five are its first five."""
    moved = _with_status(case_comparisons, status)
    return sorted(moved, key=attrgetter("score_delta"), reverse=status is ChangeStatus.IMPROVED)


def _magnitude_of(moved_cases: list[CaseComparison]) -> ChangeMagnitude:
    """How large the moves on one side were. All zero for an empty side: `statistics.mean`
    raises on an empty list, and a run where nothing got worse has no worsening to report.

    Reads `largest` off the front of the list because both sides arrive ordered biggest-move
    first, so the extreme is the first entry either way.
    """
    if not moved_cases:
        return ChangeMagnitude(largest=0.0, mean=0.0, median=0.0)
    deltas = [case.score_delta for case in moved_cases]
    return ChangeMagnitude(
        largest=deltas[0],
        mean=statistics.mean(deltas),
        median=statistics.median(deltas),
    )


def _reject_incomparable_runs(baseline: BatchResult, candidate: BatchResult) -> None:
    """Refuse before computing anything, naming every difference at once — fixing them one
    error message at a time would mean one full re-run per difference."""
    if differences := _differences_between(baseline, candidate):
        raise RunsNotComparableError(
            "the runs are not comparable: " + "; ".join(differences)
        )


def _differences_between(baseline: BatchResult, candidate: BatchResult) -> list[str]:
    """Everything that stops these two runs from being compared, in reading order: first the
    cases that are missing on one side, then the rubric changes inside the shared ones."""
    baseline_by_id = _case_results_by_id(baseline)
    candidate_by_id = _case_results_by_id(candidate)
    differences = _only_on_one_side("cases", set(baseline_by_id), set(candidate_by_id))
    for case_id in sorted(set(baseline_by_id) & set(candidate_by_id)):
        differences += _rubric_differences(
            case_id, baseline_by_id[case_id], candidate_by_id[case_id]
        )
    return differences


def _only_on_one_side(subject: str, baseline_ids: set[int], candidate_ids: set[int]) -> list[str]:
    """Ids one run carries and the other does not — the same sentence for missing cases and
    for missing criteria, so the two cannot drift into two phrasings. Both directions are
    reported, because a catalog that grew and one that shrank need different fixes."""
    return [
        f"{subject} only in the {run}: {sorted(only_here)}"
        for run, only_here in (
            ("baseline", baseline_ids - candidate_ids),
            ("candidate", candidate_ids - baseline_ids),
        )
        if only_here
    ]


def _rubric_differences(case_id: int, baseline: CaseResult, candidate: CaseResult) -> list[str]:
    """The two rubric rules for one shared case: the same criteria, and the same weights on
    them. A changed criterion set makes the two case scores answer different questions; a
    changed weight makes them non-commensurable.
    """
    baseline_weights = _weights_by_criterion_id(baseline)
    candidate_weights = _weights_by_criterion_id(candidate)
    return _only_on_one_side(
        f"case {case_id}: criteria", set(baseline_weights), set(candidate_weights)
    ) + _weight_differences(case_id, baseline_weights, candidate_weights)


def _weight_differences(
    case_id: int, baseline_weights: dict[int, float], candidate_weights: dict[int, float]
) -> list[str]:
    """Weights are copied verbatim from the rubric and never computed, so two runs of the
    same catalog carry bit-identical ones — which is why this compares exactly rather than
    with a tolerance. A weight that merely *nearly* matches means the rubric was edited, and
    the weights are the denominator every case score is normalized by.
    """
    return [
        f"case {case_id}, criterion {criterion_id}: weight "
        f"{baseline_weights[criterion_id]} vs {candidate_weights[criterion_id]}"
        for criterion_id in sorted(set(baseline_weights) & set(candidate_weights))
        if baseline_weights[criterion_id] != candidate_weights[criterion_id]
    ]


def _weights_by_criterion_id(result: CaseResult) -> dict[int, float]:
    return {verdict.criterion_id: verdict.weight for verdict in result.criterion_results}
