"""Scoring: one case folded into one number, and a whole run folded into its metrics.

Failed criteria count as 0 and stay in the denominator — a judge outage must lower the
score visibly, not silently shrink the rubric.
"""

import statistics

from rubric_eval.models import (
    SCALE_MAX,
    WEAKEST_CASES_REPORTED,
    CaseResult,
    CriterionResult,
    RunMetrics,
)


def case_score(results: list[CriterionResult]) -> float:
    """The weighted share of the reachable points, normalized to [0, 1].

        score = sum(wi * si / SCALE_MAX) / sum(wi)

    Args:
        results: The verdicts of one case, failed ones included. Only `weight` and `score`
            are read, so results loaded back from a stored run score identically.

    Returns:
        A float in [0, 1]. Exactly 1.0 when every criterion scored `SCALE_MAX`, and exactly
        0.0 when none scored anything — callers do compare against those bounds.
        Only the *ratios* of the weights matter: 3 and 1 score like 30 and 10.

    Raises:
        ValueError: The list is empty. An empty rubric has no meaningful score, and
            returning 0.0 for it would be indistinguishable from a completely missed answer.

    Example:
        Weights 3, 2, 1 scored 2, 1, 0 reach 3.0 + 1.0 + 0.0 of 6 possible points:

        case_score(results)   # 0.667
    """
    if not results:
        raise ValueError("a case score needs at least one criterion")
    weights = _weights_scaled_to_at_most_one(results)
    reached_points = sum(
        weight * result.score / SCALE_MAX for weight, result in zip(weights, results)
    )
    return reached_points / sum(weights)


def _weights_scaled_to_at_most_one(results: list[CriterionResult]) -> list[float]:
    """Only weight *ratios* matter, so dividing every weight by the largest one leaves every
    score untouched — but it keeps both sums inside the float range. Summed raw, two weights
    of 1e308 overflow to `inf`, and `inf / inf` would make the whole case score `nan`.
    """
    largest_weight = max(result.weight for result in results)
    return [result.weight / largest_weight for result in results]


def run_metrics(results: list[CaseResult]) -> RunMetrics:
    """Aggregate the case results of one run into the numbers that run is judged by.

    Reads as its parts: a distribution over the case scores, two views on the criteria
    underneath them, and the shortlist of cases worth reading next.

    Public on purpose: `evaluate_batch` calls it, but so can you — on case results loaded
    back from a stored run, for instance.

    Args:
        results: One `CaseResult` per case of the run. Order is irrelevant; every case
            counts exactly once, whatever the size of its rubric, so that one case with 20
            criteria cannot outvote nineteen cases with one.

    Returns:
        A `RunMetrics`. Every field is documented on the model; the two most easily misread
        are `average_criterion_score` (unweighted, across case boundaries — a different
        question from `average_score`) and `failed_criteria_count` (read it *before* the
        average: above 0 the run was depressed by judge outages, not only by the answers).

    Raises:
        ValueError: The list is empty. A run of no cases has no distribution to describe.

    Example:
        metrics = run_metrics([CaseResult(**row) for row in json.load(file)])
        metrics.average_score          # 0.5
        metrics.failed_criteria_count  # 0 — read this before trusting the average
    """
    if not results:
        raise ValueError("run metrics need at least one case")
    scores = [result.score for result in results]
    return RunMetrics(
        total_cases=len(results),
        average_score=statistics.mean(scores),
        median_score=statistics.median(scores),
        variance=_variance(scores),
        standard_deviation=_standard_deviation(scores),
        average_criterion_score=statistics.mean(_every_criterion_score(results)),
        criteria_fulfillment_rate=statistics.mean(_fulfillment_rate_per_case(results)),
        cases_with_score_zero=_case_ids_scoring_zero(results),
        weakest_cases_above_zero=_weakest_case_ids_above_zero(results),
        failed_criteria_count=_failed_criteria_count(results),
    )


def _variance(scores: list[float]) -> float:
    """A single case has no spread; `statistics.variance` would raise on it instead."""
    return statistics.variance(scores) if len(scores) > 1 else 0.0


def _standard_deviation(scores: list[float]) -> float:
    return statistics.stdev(scores) if len(scores) > 1 else 0.0


def _every_criterion_score(results: list[CaseResult]) -> list[float]:
    """All criteria of the run in one flat list — case boundaries and weights ignored."""
    return [criterion.score for result in results for criterion in result.criterion_results]


def _fulfillment_rate_per_case(results: list[CaseResult]) -> list[float]:
    """One rate per case, never one rate over all criteria: averaging the cases afterwards is
    what keeps a case with a 20-criteria rubric from outweighing nineteen short ones.

    No division by zero to guard here — `Case.criteria` rejects an empty rubric.
    """
    return [
        sum(criterion.is_present for criterion in result.criterion_results)
        / len(result.criterion_results)
        for result in results
    ]


def _case_ids_scoring_zero(results: list[CaseResult]) -> list[int]:
    return [result.case_id for result in results if result.score == 0.0]


def _weakest_case_ids_above_zero(results: list[CaseResult]) -> list[int]:
    """The weakest cases that still scored something, weakest first. Cases at exactly 0 are
    reported separately, so this list does not fill up with them and hide the near misses."""
    above_zero = sorted((result for result in results if result.score > 0), key=_score_of)
    return [result.case_id for result in above_zero[:WEAKEST_CASES_REPORTED]]


def _score_of(result: CaseResult) -> float:
    return result.score


def _failed_criteria_count(results: list[CaseResult]) -> int:
    return sum(criterion.failed for result in results for criterion in result.criterion_results)
