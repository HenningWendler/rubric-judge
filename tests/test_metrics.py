import itertools
import math
import statistics

import pytest

from rubric_eval.metrics import case_score, run_metrics
from rubric_eval.models import (
    DEFAULT_SCALE,
    WEAKEST_CASES_REPORTED,
    CaseResult,
    Criterion,
    CriterionResult,
)


_NEXT_CRITERION_ID = itertools.count(1)


def _result(weight: float, score: float) -> CriterionResult:
    """A verdict written as weight and score alone — the only two things the formulas read.
    The id is handed out fresh each call, because a `CaseResult` rejects a repeated one."""
    criterion = Criterion(id=next(_NEXT_CRITERION_ID), content="x", weight=weight)
    return CriterionResult.judged(criterion, score, None, DEFAULT_SCALE)


def test_weights_the_scores_and_normalizes_to_one():
    # (3*2/2 + 1*0/2) / 4
    assert case_score([_result(3, 2), _result(1, 0)], DEFAULT_SCALE) == pytest.approx(0.75)


def test_a_partial_score_counts_half():
    assert case_score([_result(1, 1)], DEFAULT_SCALE) == pytest.approx(0.5)


def test_presence_is_derived_from_the_score():
    assert _result(1, 1).is_present is True
    assert _result(1, 0).is_present is False


def test_an_empty_rubric_is_a_clear_error_not_a_division_by_zero():
    with pytest.raises(ValueError, match="at least one criterion"):
        case_score([], DEFAULT_SCALE)


def test_a_rubric_missed_completely_scores_zero_rather_than_erroring():
    """An answer that covers nothing is a 0, not an exception — and a real one, because a run
    that lost a verdict to an outage never reaches the formulas at all."""
    assert case_score([_result(3, 0), _result(1, 0)], DEFAULT_SCALE) == 0.0


def test_a_perfect_rubric_scores_exactly_one():
    """The upper bound is exact, not 0.9999… — callers compare against 1.0."""
    assert case_score([_result(3, 2), _result(1, 2)], DEFAULT_SCALE) == 1.0


def test_the_score_never_leaves_the_unit_interval():
    """Mixed weights and scores stay inside [0, 1], the range `CaseResult.score` promises."""
    score = case_score([_result(0.001, 0), _result(1000, 1), _result(7, 2)], DEFAULT_SCALE)
    assert 0.0 <= score <= 1.0


def test_scaling_every_weight_leaves_the_score_unchanged():
    """`Criterion.weight` promises that only the ratios matter — 3 and 1 must score like
    3e307 and 1e307. Extreme weights are what expose an overflow in the intermediate sums."""
    baseline = case_score([_result(3, 2), _result(1, 0)], DEFAULT_SCALE)
    extreme = case_score([_result(3e307, 2), _result(1e307, 0)], DEFAULT_SCALE)
    assert extreme == pytest.approx(baseline)


def test_huge_weights_do_not_overflow_the_weight_sum():
    """Two weights whose sum exceeds the float range must not silently become `nan`:
    a `nan` score is serialized as `null` and breaks the declared result schema."""
    score = case_score([_result(1e308, 2), _result(1e308, 0)], DEFAULT_SCALE)
    assert score == pytest.approx(0.5)


# --- run metrics: one run folded into the numbers a run is judged by ---------------------


def _case(case_id: int, *verdicts: CriterionResult) -> CaseResult:
    criterion_results = list(verdicts)
    return CaseResult(
        case_id=case_id,
        score=case_score(criterion_results, DEFAULT_SCALE),
        criterion_results=criterion_results,
    )


THREE_CASES = [
    _case(1, _result(3, 2), _result(1, 0)),  # 0.75
    _case(2, _result(1, 1)),  # 0.5
    _case(3, _result(1, 0)),  # 0.0
]


def test_averages_the_case_scores_not_the_criteria():
    """A case counts once, whatever the size of its rubric — otherwise one case with twenty
    criteria would outvote nineteen cases with one."""
    metrics = run_metrics(THREE_CASES)

    assert metrics.total_cases == 3
    assert metrics.average_score == pytest.approx((0.75 + 0.5 + 0.0) / 3)
    assert metrics.median_score == pytest.approx(0.5)


def test_a_single_case_has_no_spread_rather_than_erroring():
    """`statistics.variance` raises on one value; a run of one case is a legitimate run."""
    metrics = run_metrics([_case(1, _result(1, 2))])

    assert metrics.variance == 0.0
    assert metrics.standard_deviation == 0.0


def test_the_spread_of_several_cases_is_reported():
    metrics = run_metrics(THREE_CASES)

    assert metrics.variance == pytest.approx(statistics.variance([0.75, 0.5, 0.0]))
    assert metrics.standard_deviation == pytest.approx(statistics.stdev([0.75, 0.5, 0.0]))


def test_the_average_criterion_score_ignores_weights_and_case_boundaries():
    """Deliberately a different question than `average_score`: how the judge rates an average
    single statement, on the raw 0-2 scale, not how good the average answer is."""
    metrics = run_metrics(THREE_CASES)

    assert metrics.average_criterion_score == pytest.approx((2 + 0 + 1 + 0) / 4)


def test_the_fulfillment_rate_is_averaged_per_case_not_over_all_criteria():
    """The one that would silently be wrong if the criteria were flattened first: a case with
    a long rubric must not dominate the rate. Flattened this would be 10/11 ≈ 0.91."""
    long_rubric = _case(1, *[_result(1, 2) for _ in range(10)])
    one_miss = _case(2, _result(1, 0))

    assert run_metrics([long_rubric, one_miss]).criteria_fulfillment_rate == pytest.approx(0.5)


def test_cases_that_missed_the_rubric_completely_are_named_by_id():
    metrics = run_metrics(THREE_CASES)

    assert metrics.cases_with_score_zero == [3]
    assert metrics.cases_with_score_zero_count == 1


def test_the_weakest_cases_are_listed_weakest_first_and_exclude_the_zeros():
    """Zero-score cases have their own list; mixing them in would fill the shortlist with
    total misses and hide the near misses, which usually have a different cause."""
    metrics = run_metrics(THREE_CASES)

    assert metrics.weakest_cases_above_zero == [2, 1]


def test_the_weakest_case_list_is_capped():
    """A shortlist to read next, not a full ranking — the complete data is in `cases`."""
    many = [_case(number, _result(1, 1)) for number in range(20)]

    assert len(run_metrics(many).weakest_cases_above_zero) == WEAKEST_CASES_REPORTED


def test_an_empty_run_is_a_clear_error_not_a_division_by_zero():
    with pytest.raises(ValueError, match="at least one case"):
        run_metrics([])


def test_a_case_result_always_carries_at_least_one_verdict():
    """`Case.criteria` rejects an empty rubric, so "one verdict per criterion" means at least
    one verdict — and the fulfillment rate divides by exactly that count. Without the rule on
    the *result* type, a run loaded back from disk reaches `run_metrics` as a division by
    zero instead of a clean rejection."""
    with pytest.raises(ValueError):
        CaseResult(case_id=1, score=0.0, criterion_results=[])


def test_a_run_loaded_back_from_stored_rows_is_aggregated_like_a_fresh_one():
    """The documented use of `run_metrics`: case results read from a file, not from a judge.
    They arrive as plain dicts, so the model is the only thing standing between a malformed
    row and the formulas."""
    rows = [
        {"case_id": 1, "score": 0.75, "criterion_results": [
            {"criterion_id": 1, "weight": 3.0, "score": 2.0, "is_present": True}]},
        {"case_id": 2, "score": 0.0, "criterion_results": [
            {"criterion_id": 1, "weight": 1.0, "score": 0.0, "is_present": False}]},
    ]

    metrics = run_metrics([CaseResult(**row) for row in rows])

    assert metrics.average_score == pytest.approx(0.375)
    assert metrics.cases_with_score_zero == [2]


def test_the_readme_worked_example_scores_two_thirds():
    """Weights 3/2/1 scored 2/1/0 reach 3.0 + 1.0 + 0.0 of 6 points — the example the README
    and the `case_score` docstring both spell out, and the one a reader reproduces first."""
    readme_example = [_result(3, 2), _result(2, 1), _result(1, 0)]
    assert case_score(readme_example, DEFAULT_SCALE) == pytest.approx(2 / 3)


def test_presence_starts_exactly_at_the_threshold():
    """`is_present` is documented as `score >= scale.presence_threshold`, so the threshold
    itself counts as present. A `>` would move the cut silently and only for averaged scores."""
    assert _result(1, DEFAULT_SCALE.presence_threshold).is_present is True
    assert _result(1, DEFAULT_SCALE.presence_threshold / 2).is_present is False


def test_the_standard_deviation_is_the_square_root_of_the_variance():
    """The README defines one as the root of the other. `run_metrics` derives it that way, so
    this guards against anyone computing the two from the scores independently again."""
    metrics = run_metrics(THREE_CASES)

    assert metrics.standard_deviation == pytest.approx(math.sqrt(metrics.variance))


def test_the_metrics_do_not_depend_on_the_order_of_the_case_results():
    """Documented as order-irrelevant: every case counts once, whatever position it arrives
    in. Only the id lists are allowed to be ordered, and they are ordered by score."""
    forward = run_metrics(THREE_CASES)
    backward = run_metrics(list(reversed(THREE_CASES)))

    assert forward == backward


def test_the_weakest_cases_are_the_weakest_of_the_run_not_the_first_five_found():
    """The cap is a shortlist of the worst, not of the earliest — capping before sorting
    would name whichever cases happened to arrive first."""
    strongest_first = [_case(number, _result(1, 2)) for number in range(1, 4)]
    weakest = [_case(number, _result(1, 0.2)) for number in range(90, 95)]

    metrics = run_metrics(strongest_first + weakest)

    assert metrics.weakest_cases_above_zero == [90, 91, 92, 93, 94]


def test_the_readme_run_example_reports_exactly_the_documented_numbers():
    """Every number in the README is claimed to be reproducible verbatim. This is the run of
    the documented two-case run — one perfect answer, one total miss."""
    metrics = run_metrics([_case(1, _result(3, 2)), _case(2, _result(1, 0))])

    assert metrics.model_dump() == {
        "total_cases": 2,
        "average_score": 0.5,
        "median_score": 0.5,
        "variance": 0.5,
        "standard_deviation": 0.7071067811865476,
        "average_criterion_score": 1.0,
        "criteria_fulfillment_rate": 0.5,
        "cases_with_score_zero": [2],
        "cases_with_score_zero_count": 1,
        "weakest_cases_above_zero": [1],
    }
