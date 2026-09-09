import pytest

from rubric_eval.metrics import case_score
from rubric_eval.models import Criterion, CriterionResult


def _result(weight: float, score: float) -> CriterionResult:
    return CriterionResult.judged(Criterion(id=1, content="x", weight=weight), score, None)


def test_weights_the_scores_and_normalizes_to_one():
    # (3*2/2 + 1*0/2) / 4
    assert case_score([_result(3, 2), _result(1, 0)]) == pytest.approx(0.75)


def test_a_partial_score_counts_half():
    assert case_score([_result(1, 1)]) == pytest.approx(0.5)


def test_an_unjudged_criterion_scores_zero_but_keeps_its_weight():
    unjudged = CriterionResult.unjudged(Criterion(id=2, content="x", weight=1), "judge down")
    assert case_score([_result(1, 2), unjudged]) == pytest.approx(0.5)


def test_presence_is_derived_from_the_score():
    assert _result(1, 1).is_present is True
    assert _result(1, 0).is_present is False


def test_an_empty_rubric_is_a_clear_error_not_a_division_by_zero():
    with pytest.raises(ValueError, match="at least one criterion"):
        case_score([])


def test_all_criteria_failing_scores_zero_rather_than_erroring():
    """A total judge outage is a 0, not an exception — the caller still gets a result."""
    dead = [
        CriterionResult.unjudged(Criterion(id=1, content="x", weight=3), "judge down"),
        CriterionResult.unjudged(Criterion(id=2, content="y", weight=1), "judge down"),
    ]
    assert case_score(dead) == 0.0


def test_a_perfect_rubric_scores_exactly_one():
    """The upper bound is exact, not 0.9999… — callers compare against 1.0."""
    assert case_score([_result(3, 2), _result(1, 2)]) == 1.0


def test_the_score_never_leaves_the_unit_interval():
    """Mixed weights and scores stay inside [0, 1], the range `EvaluationResult.score` promises."""
    score = case_score([_result(0.001, 0), _result(1000, 1), _result(7, 2)])
    assert 0.0 <= score <= 1.0


def test_scaling_every_weight_leaves_the_score_unchanged():
    """`Criterion.weight` promises that only the ratios matter — 3 and 1 must score like
    3e307 and 1e307. Extreme weights are what expose an overflow in the intermediate sums."""
    baseline = case_score([_result(3, 2), _result(1, 0)])
    assert case_score([_result(3e307, 2), _result(1e307, 0)]) == pytest.approx(baseline)


def test_huge_weights_do_not_overflow_the_weight_sum():
    """Two weights whose sum exceeds the float range must not silently become `nan`:
    a `nan` score is serialized as `null` and breaks the declared result schema."""
    score = case_score([_result(1e308, 2), _result(1e308, 0)])
    assert score == pytest.approx(0.5)
