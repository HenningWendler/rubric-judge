"""Holding two finished runs against each other: did the change help, where, and what did it cost.

Pure computation over two `RunResult` documents — no judge, no network, no cost. Runs
saved to disk months apart compare exactly like runs produced a second ago.

One entry point, `compare_runs`, and one hard rule underneath it: two runs are comparable
only if they were judged on the same scale and cover the same cases with the same rubric, the
same weights and the same labels. Different weights make the case scores non-commensurable, a
different scale makes the raw criterion scores so, changed labels make the per-label buckets
hold different cases on each side, and a delta between any of them would look like a result
while meaning nothing.
"""

import statistics
from operator import attrgetter

from rubric_eval.models import (
    CaseComparisonResult,
    CaseResult,
    ChangeMagnitude,
    ChangeStatus,
    ChangeSummary,
    LabelMetrics,
    LabelMetricsDelta,
    RunComparison,
    RunComparisonResult,
    RunMetrics,
    RunMetricsDelta,
    RunResult,
    reworded_grades_clause,
)


class RunsNotComparableError(ValueError):
    """The two runs do not describe the same catalog, so their scores do not subtract.

    A `ValueError`, because that is what it is and what a caller would catch anyway — but a
    named one, so the HTTP layer can tell a refused comparison apart from a `ValidationError`
    or a `StatisticsError`. Both of those are `ValueError` subclasses too, and reporting one
    of them as "your runs are not comparable" would dress a bug up as the caller's mistake.
    """


def compare_runs(run_comparison: RunComparison) -> RunComparisonResult:
    """Compare two finished runs of the same catalog at three grains: run, case, criterion.

    Every delta is `candidate - baseline`, so a positive number always means the candidate
    did better — with the two counting fields of `RunMetricsDelta` as the documented
    exception, where fewer is better.

    Takes a `RunComparison` rather than two arguments because both sides have the same type:
    a swapped pair would be undetectable and would silently invert the whole document.

    Args:
        run_comparison: The `baseline` run to compare against and the `candidate` run under
            test. Both must have been judged on the same scale and must cover the same case
            ids, the same criterion ids per case, the same weights and the same labels per
            case — see Raises. Nothing else is required of them: results loaded back from
            stored JSON compare exactly like results just computed.

    Returns:
        A `RunComparisonResult`. `metrics_delta` says whether the run got better, `summary` how
        that is distributed over the cases, `label_metrics_deltas` which *kind* of case moved,
        and `case_comparison_results` — ordered by `case_id` — which criterion is responsible.
        Both runs are complete by construction: a run whose judge failed anywhere was never
        handed back, so no delta here is an artefact of an outage on one side.

        `applied_label_filter` is deliberately *not* compared: two runs covering the same case
        ids are comparable however each of them was selected, and two different filters can
        legitimately arrive at the same cases.

    Raises:
        RunsNotComparableError: A `ValueError`. The runs do not describe the same catalog,
            or were not judged on the same scale. The message names every difference found —
            a differing scale, cases present on only one side, criteria that differ within a
            shared case, weights that changed, and labels that changed — rather than only the
            first, so one fix can address all of them.

    Example:
        result = compare_runs(RunComparison(baseline=last_weeks_run, candidate=todays_run))
        result.metrics_delta.average_score_delta   # +0.084
        result.label_metrics_deltas[0].label       # "agentic_search"
        result.summary.worsened_case_ids           # [5] — what the win cost
        result.summary.improvement.largest         # +0.31
    """
    baseline, candidate = run_comparison.baseline, run_comparison.candidate
    _reject_incomparable_runs(baseline, candidate)
    case_comparison_results = _compare_cases(baseline, candidate)
    return RunComparisonResult(
        metrics_delta=_metrics_delta(baseline.metrics, candidate.metrics),
        summary=_summarize(case_comparison_results),
        label_metrics_deltas=_label_metrics_deltas(baseline, candidate),
        case_comparison_results=case_comparison_results,
    )


def _compare_cases(baseline: RunResult, candidate: RunResult) -> list[CaseComparisonResult]:
    """One comparison per case, ordered by id — neither run's storage order is canonical,
    so sorting by id gives a document that does not depend on either."""
    baseline_by_id = _case_results_by_id(baseline)
    candidate_by_id = _case_results_by_id(candidate)
    return [
        CaseComparisonResult.between(baseline_by_id[case_id], candidate_by_id[case_id])
        for case_id in sorted(candidate_by_id)
    ]


def _case_results_by_id(run: RunResult) -> dict[int, CaseResult]:
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
    )


def _label_metrics_deltas(
    baseline: RunResult, candidate: RunResult
) -> list[LabelMetricsDelta]:
    """One delta per label, in the alphabetical order both breakdowns already carry.

    Pairing by label needs no intersection: the comparability check has guaranteed that every
    case carries the same labels in both runs, and `RunResult` has guaranteed that each
    breakdown covers exactly the labels its cases carry — so the two label sets are equal, and
    zipping them pairs like with like.
    """
    return [
        LabelMetricsDelta(
            label=baseline_bucket.label,
            metrics_delta=_metrics_delta(baseline_bucket.metrics, candidate_bucket.metrics),
        )
        for baseline_bucket, candidate_bucket in zip(
            _buckets_by_label(baseline), _buckets_by_label(candidate), strict=True
        )
    ]


def _buckets_by_label(run: RunResult) -> list[LabelMetrics]:
    """Sorted here as well as at the source, so the zip above pairs by label rather than by
    trusting the storage order of a run that was read back from JSON."""
    return sorted(run.label_metrics, key=attrgetter("label"))


def _summarize(case_comparison_results: list[CaseComparisonResult]) -> ChangeSummary:
    """The case movements folded into the distribution they form: who moved which way, in
    order of how much, and how large those moves were on each side."""
    improved = _ranked_by_movement(case_comparison_results, ChangeStatus.IMPROVED)
    worsened = _ranked_by_movement(case_comparison_results, ChangeStatus.WORSENED)
    stable = _with_status(case_comparison_results, ChangeStatus.STABLE)
    return ChangeSummary(
        improved_case_ids=[case.case_id for case in improved],
        stable_case_ids=[case.case_id for case in stable],
        worsened_case_ids=[case.case_id for case in worsened],
        improvement=_magnitude_of(improved),
        worsening=_magnitude_of(worsened),
    )


def _with_status(
    case_comparison_results: list[CaseComparisonResult], status: ChangeStatus
) -> list[CaseComparisonResult]:
    """One place to pick a side out, so the three lists of a summary are cut the same way."""
    return [case for case in case_comparison_results if case.status is status]


def _ranked_by_movement(
    case_comparison_results: list[CaseComparisonResult], status: ChangeStatus
) -> list[CaseComparisonResult]:
    """One side of the comparison, biggest move first — for improvements the largest positive
    delta, for regressions the most negative one. Ordering the complete list this way is what
    makes a separate "top five" field unnecessary: the top five are its first five."""
    moved = _with_status(case_comparison_results, status)
    return sorted(moved, key=attrgetter("score_delta"), reverse=status is ChangeStatus.IMPROVED)


def _magnitude_of(moved_cases: list[CaseComparisonResult]) -> ChangeMagnitude:
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


def _reject_incomparable_runs(baseline: RunResult, candidate: RunResult) -> None:
    """Refuse before computing anything, naming every difference at once — fixing them one
    error message at a time would mean one full re-run per difference."""
    if differences := _differences_between(baseline, candidate):
        raise RunsNotComparableError(
            "the runs are not comparable: " + "; ".join(differences)
        )


def _differences_between(baseline: RunResult, candidate: RunResult) -> list[str]:
    """Everything that stops these two runs from being compared, in reading order: first the
    scale, which invalidates everything under it, then the cases that are missing on one side,
    then the rubric and label changes inside the shared ones."""
    baseline_by_id = _case_results_by_id(baseline)
    candidate_by_id = _case_results_by_id(candidate)
    differences = _scale_differences(baseline, candidate)
    differences += _only_on_one_side("cases", set(baseline_by_id), set(candidate_by_id))
    for case_id in sorted(set(baseline_by_id) & set(candidate_by_id)):
        differences += _rubric_differences(
            case_id, baseline_by_id[case_id], candidate_by_id[case_id]
        )
        differences += _label_differences(
            case_id, baseline_by_id[case_id], candidate_by_id[case_id]
        )
    return differences


def _label_differences(case_id: int, baseline: CaseResult, candidate: CaseResult) -> list[str]:
    """A case re-labelled between the two runs puts different cases in the two per-label
    buckets of the same name, so every delta in `label_metrics_deltas` would silently compare
    two different populations. Compared as sets: labels are read as a set everywhere, so a
    reordered list is the same labelling and must not be reported as a change.
    """
    if set(baseline.labels) == set(candidate.labels):
        return []
    return [
        f"case {case_id}: labels {sorted(baseline.labels)} vs {sorted(candidate.labels)}"
    ]


def _scale_differences(baseline: RunResult, candidate: RunResult) -> list[str]:
    """Reported first, because it is the difference that makes every other number meaningless:
    a 2 out of 2 and a 2 out of 10 are not the same verdict, so subtracting them would turn a
    change of judge into a collapse of the system under test."""
    if baseline.scale == candidate.scale:
        return []
    return [
        f"the runs were judged on different scales: baseline {baseline.scale}, "
        f"candidate {candidate.scale}{reworded_grades_clause([baseline.scale, candidate.scale])}"
    ]


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
    """Keyed by id because the two runs are checked criterion by criterion, not position by
    position — the criteria of a stored result come in whatever order it was saved in."""
    return {verdict.criterion_id: verdict.weight for verdict in result.criterion_results}
