"""Scoring: case scores, the run metrics over them, and the same metrics per label.

One case folded into one number, a whole run folded into its metrics, and that run sliced
once per label.

Every number here is folded from criterion results the judge really gave: a run that lost
one to an outage is invalidated before it gets this far, so no average is quietly depressed
by a criterion nobody graded.
"""

import math
import statistics
from operator import attrgetter

from rubric_judge.models import (
    WEAKEST_CASES_REPORTED,
    CaseResult,
    CriterionResult,
    LabelMetrics,
    RunMetrics,
    Scale,
    carries_every_label,
    case_count_per_label,
    one_scale_of,
    scores_must_fit,
)


def case_score(criterion_results: list[CriterionResult], scale: Scale) -> float:
    """The weighted share of the reachable points, normalized to [0, 1].

        score = sum(wi * si / scale.maximum) / sum(wi)

    Dividing by the scale is what makes the result scale-free: half marks everywhere is 0.5
    whether the judge counted in halves or in fifths. Two runs are therefore comparable at
    case grain even though their raw grades are not — and that is the whole reason this
    number is normalized rather than reported raw.

    Args:
        criterion_results: The results of one case, all of them. Only `weight` and `score`
            are read, so results loaded back from a stored run score identically.
        scale: The scale they were given on — `CaseResult.scale`, or `judge.scale` while the
            case result is still being built. A result does not carry it: one case is judged
            by one judge on one scale, so one copy per case is the honest place for it.

    Returns:
        A float in [0, 1]. Exactly 1.0 when every criterion reached `scale.maximum`, and
        exactly 0.0 when none scored anything — callers do compare against those bounds.
        Only the *ratios* of the weights matter: 3 and 1 score like 30 and 10.

    Raises:
        ValueError: The list is empty — an empty rubric has no meaningful score, and
            returning 0.0 for it would be indistinguishable from a completely missed answer —
            or a result is graded above `scale.maximum`, which no share of the reachable
            points can be made of.

    Example:
        Weights 3, 2, 1 scored 2, 1, 0 reach 3.0 + 1.0 + 0.0 of 6 possible points:

        criterion_results = [
            CriterionResult.judged(Criterion(id=1, content="a", weight=3), 2.0, None, DEFAULT_SCALE),
            CriterionResult.judged(Criterion(id=2, content="b", weight=2), 1.0, None, DEFAULT_SCALE),
            CriterionResult.judged(Criterion(id=3, content="c", weight=1), 0.0, None, DEFAULT_SCALE),
        ]
        case_score(criterion_results, DEFAULT_SCALE)   # 0.6666666666666666
    """
    if not criterion_results:
        raise ValueError("a case score needs at least one criterion")
    scores_must_fit(criterion_results, scale)
    weights = _weights_scaled_to_at_most_one(criterion_results)
    reached_points = sum(
        weight * criterion_result.score / scale.maximum
        for weight, criterion_result in zip(weights, criterion_results)
    )
    reachable_points = sum(weights)
    return reached_points / reachable_points


def _weights_scaled_to_at_most_one(
    criterion_results: list[CriterionResult],
) -> list[float]:
    """Bring a case's weights into a range whose sums cannot overflow.

    Args:
        criterion_results: The results of one case, at least one, each carrying a positive
            finite weight — `CriterionResult` enforces both, so nothing here re-checks them.

    Returns:
        One weight per result, in the given order, each divided by the largest of them: the
        largest comes back as exactly 1.0 and every other lies in (0, 1]. Only weight
        *ratios* decide a case score, so this leaves every score untouched — but summed raw,
        two weights of 1e308 overflow to `inf` and `inf / inf` would make the case score
        `nan`.
    """
    largest_weight = max(
        criterion_result.weight for criterion_result in criterion_results
    )
    return [
        criterion_result.weight / largest_weight for criterion_result in criterion_results
    ]


def run_metrics(case_results: list[CaseResult]) -> RunMetrics:
    """Aggregate the case results of one run into the numbers that run is judged by.

    Reads as its parts: a distribution over the case scores, two views on the criteria
    underneath them, and the shortlist of cases worth reading next.

    Public on purpose: `evaluate_run` calls it, but so can you — on case results loaded
    back from a stored run, for instance.

    Args:
        case_results: One `CaseResult` per case of the run. Order is irrelevant; every case
            counts exactly once, whatever the size of its rubric, so that one case with 20
            criteria cannot outvote nineteen cases with one.

    Returns:
        A `RunMetrics`. Every field is documented on the model; the most easily misread is
        `average_criterion_score` — unweighted and across case boundaries, which is a
        different question from `average_score`.

    Raises:
        ValueError: The list is empty — a run of no cases has no distribution to describe —
            or the cases were not all judged on the same scale, which would make
            `average_criterion_score` an average of grades in different units. `RunResult`
            refuses such a run too, but this function is documented as callable on its own
            and must not hand back a number nobody can interpret.

    Example:
        metrics = run_metrics(run_result.case_results)
        metrics.average_score            # 0.5
        metrics.average_criterion_score  # 1.0 — on the run's raw 0..2 scale, not on 0..1
        metrics.cases_with_score_zero    # [1]
    """
    if not case_results:
        raise ValueError("run metrics need at least one case")
    one_scale_of(case_result.scale for case_result in case_results)
    scores = [case_result.score for case_result in case_results]
    variance = _variance(scores)
    return RunMetrics(
        total_cases=len(case_results),
        average_score=statistics.mean(scores),
        median_score=statistics.median(scores),
        variance=variance,
        standard_deviation=math.sqrt(variance),
        average_criterion_score=statistics.mean(_every_criterion_score(case_results)),
        criteria_fulfillment_rate=statistics.mean(_fulfillment_rate_per_case(case_results)),
        cases_with_score_zero=_case_ids_scoring_zero(case_results),
        weakest_cases_above_zero=_weakest_case_ids_above_zero(case_results),
    )


def label_metrics(case_results: list[CaseResult]) -> list[LabelMetrics]:
    """Slice a run once per label: the same metrics over only the cases carrying each one.

    The breakdown that says *which kind* of case a bad average is made of — a run averaging
    0.54 whose `agentic_search` bucket averages 0.013 has one problem, not a general one.

    A case counts in every bucket it carries a label for, so the buckets overlap and their
    case counts add up to more than the run. Cases with no labels land in no bucket at all;
    they are in the run-wide `run_metrics` and nowhere else.

    Public for the same reason `run_metrics` is: `evaluate_run` calls it, but so can you,
    on case results loaded back from a stored run — including one saved before you started
    labelling, once you have added the labels to its results.

    Args:
        case_results: The case results of one run, in any order. Only `labels` decides which
            bucket a case lands in; everything else is handed to `run_metrics` unchanged.

    Returns:
        One `LabelMetrics` per label occurring anywhere in them, in alphabetical label
        order so the document does not depend on the run's storage order. Empty when no case
        carries a label — which is the honest answer, not a missing breakdown.

    Raises:
        ValueError: The cases were not all judged on the same scale, raised by `run_metrics`
            for the first bucket that mixes two. No bucket can be empty, so the "no cases"
            refusal of `run_metrics` is unreachable from here.

    Example:
        buckets = label_metrics(run_result.case_results)
        buckets[0].label                   # "table"
        buckets[0].metrics.average_score   # 0.0
        buckets[0].metrics.total_cases     # 1 — one case carries that label
    """
    return [
        LabelMetrics(label=label, metrics=run_metrics(_cases_carrying(label, case_results)))
        for label in _labels_to_bucket_by(case_results)
    ]


def _labels_to_bucket_by(case_results: list[CaseResult]) -> list[str]:
    """Which buckets there are to build, in alphabetical order.

    Counted so a label shared by forty cases still opens exactly one bucket, sorted so the
    breakdown reads the same whatever order the run was stored in.
    """
    return sorted(case_count_per_label(case_result.labels for case_result in case_results))


def _cases_carrying(label: str, case_results: list[CaseResult]) -> list[CaseResult]:
    """One bucket's cases — "carries this label", never "is exactly this label".

    The question a bucket answers is how the cases *involving* tables do, and a case tagged
    both `table` and `images` is one of them. Through the same predicate
    `filter_cases_by_labels` uses, so a bucket and the filter of the same name can never
    select different cases.
    """
    return [
        case_result
        for case_result in case_results
        if carries_every_label(case_result.labels, [label])
    ]


def _variance(scores: list[float]) -> float:
    """A single case has no spread; `statistics.variance` would raise on it instead.

    The only place that rule lives: `standard_deviation` is the square root of what this
    returns, so the two can never disagree about a one-case run.
    """
    return statistics.variance(scores) if len(scores) > 1 else 0.0


def _every_criterion_score(case_results: list[CaseResult]) -> list[float]:
    """All criteria of the run in one flat list — case boundaries and weights ignored."""
    return [
        criterion_result.score
        for case_result in case_results
        for criterion_result in case_result.criterion_results
    ]


def _fulfillment_rate_per_case(case_results: list[CaseResult]) -> list[float]:
    """One rate per case, never one rate over all criteria.

    Averaging the cases afterwards is what keeps a case with a 20-criteria rubric from
    outweighing nineteen short ones. No division by zero to guard here —
    `CaseResult.criterion_results` rejects an empty list.
    """
    return [
        sum(
            criterion_result.is_present
            for criterion_result in case_result.criterion_results
        )
        / len(case_result.criterion_results)
        for case_result in case_results
    ]


def _case_ids_scoring_zero(case_results: list[CaseResult]) -> list[int]:
    """The total misses, named apart from the near misses.

    The two usually have different causes, and these are the answers worth reading first.
    """
    return [
        case_result.case_id for case_result in case_results if case_result.score == 0.0
    ]


def _weakest_case_ids_above_zero(case_results: list[CaseResult]) -> list[int]:
    """The weakest cases that still scored something, weakest first.

    Cases at exactly 0 are reported separately, so this list does not fill up with them and
    hide the near misses.
    """
    above_zero = sorted(
        (case_result for case_result in case_results if case_result.score > 0),
        key=attrgetter("score"),
    )
    return [case_result.case_id for case_result in above_zero[:WEAKEST_CASES_REPORTED]]
