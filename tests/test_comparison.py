"""Comparing two finished runs: the deltas, the classification, and what is refused.

Every run here is built by `run_of` from plain judge scores, so a test reads as scores in
and deltas out with no judge and no event loop in between.
"""

import pytest
from pydantic import ValidationError

from conftest import run_of
from rubric_eval import ChangeStatus, Comparison, RunsNotComparableError, compare_runs
from rubric_eval.metrics import case_score, run_metrics
from rubric_eval.models import BatchResult, CaseResult, Criterion, CriterionResult


def compared(baseline: BatchResult, candidate: BatchResult):
    """The result of comparing two runs — the whole file's one line of setup."""
    return compare_runs(Comparison(baseline=baseline, candidate=candidate))


class TestDirection:
    """Every delta is candidate minus baseline, so positive always means "the candidate won"."""

    def test_a_better_candidate_gives_positive_deltas(self):
        result = compared(run_of({1: 0}), run_of({1: 2}))
        assert result.metrics_delta.average_score_delta == 1.0
        assert result.case_comparisons[0].score_delta == 1.0
        assert result.case_comparisons[0].criterion_comparisons[0].score_delta == 2.0

    def test_a_worse_candidate_gives_negative_deltas(self):
        result = compared(run_of({1: 2}), run_of({1: 0}))
        assert result.metrics_delta.average_score_delta == -1.0
        assert result.case_comparisons[0].score_delta == -1.0

    def test_swapping_the_sides_flips_every_sign(self):
        better, worse = run_of({1: 2}), run_of({1: 1})
        assert compared(worse, better).metrics_delta.average_score_delta == -(
            compared(better, worse).metrics_delta.average_score_delta
        )

    def test_fewer_zero_scored_cases_is_a_negative_delta(self):
        """The one place where negative is the good direction, which is why it is documented."""
        result = compared(run_of({1: 0}, {2: 0}), run_of({1: 2}, {2: 0}))
        assert result.metrics_delta.cases_with_score_zero_count_delta == -1


class TestStatus:
    """improved / stable / worsened, derived from the raw score delta at both grains."""

    def test_a_criterion_rising_from_partial_to_full_counts_as_improved(self):
        """1 to 2 never crosses the presence threshold, yet it moves the case score — deriving
        the status from `is_present` instead would report this as stable."""
        result = compared(run_of({1: 1}), run_of({1: 2}))
        criterion = result.case_comparisons[0].criterion_comparisons[0]
        assert criterion.status is ChangeStatus.IMPROVED
        assert criterion.baseline_score == 1.0
        assert criterion.candidate_score == 2.0

    def test_identical_runs_are_stable_everywhere(self):
        result = compared(run_of({1: 1, 2: 2}), run_of({1: 1, 2: 2}))
        assert result.summary.stable_case_ids == [1]
        assert result.summary.improved_case_ids == []
        assert result.summary.worsened_case_ids == []
        assert all(
            criterion.status is ChangeStatus.STABLE
            for criterion in result.case_comparisons[0].criterion_comparisons
        )

    def test_float_noise_below_the_tolerance_is_stable_not_a_regression(self):
        """Two runs that summed the same weights in a different order can land a few ulps
        apart. Exact equality would file that as a worsened case and put it on the
        regressions list, where a reader would go looking for a cause that does not exist."""
        baseline = run_of({1: 2, 2: 1, 3: 1})
        candidate = run_of({1: 2, 2: 1, 3: 1})
        nudged = candidate.case_results[0].model_copy(
            update={"score": candidate.case_results[0].score + 1e-15}
        )
        candidate = candidate.model_copy(update={"case_results": [nudged]})
        assert compared(baseline, candidate).case_comparisons[0].status is ChangeStatus.STABLE

    def test_a_case_can_be_stable_while_its_criteria_move_against_each_other(self):
        """The reason the criterion grain exists: equal weights cancelling out leave the case
        score untouched and the answer materially different."""
        result = compared(run_of({1: 2, 2: 0}), run_of({1: 0, 2: 2}))
        case = result.case_comparisons[0]
        assert case.status is ChangeStatus.STABLE
        assert [criterion.status for criterion in case.criterion_comparisons] == [
            ChangeStatus.WORSENED,
            ChangeStatus.IMPROVED,
        ]


class TestSummary:
    """Who moved which way, in which order, and by how much."""

    def test_cases_are_split_over_the_three_lists_exactly_once(self):
        result = compared(
            run_of({1: 0}, {2: 2}, {3: 1}),
            run_of({1: 2}, {2: 0}, {3: 1}),
        )
        summary = result.summary
        assert summary.improved_case_ids == [1]
        assert summary.worsened_case_ids == [2]
        assert summary.stable_case_ids == [3]
        assert summary.improvement_rate + summary.stability_rate + summary.worsening_rate == 1.0

    def test_counts_follow_the_lists(self):
        result = compared(run_of({1: 0}, {2: 0}), run_of({1: 2}, {2: 2}))
        assert result.summary.improved_case_count == 2
        assert result.summary.stable_case_count == 0
        assert result.summary.worsened_case_count == 0

    def test_improvements_are_ordered_biggest_first_so_the_top_k_is_a_slice(self):
        result = compared(
            run_of({1: 0}, {2: 0}, {3: 0}),
            run_of({1: 1}, {2: 2}, {3: 1}),
        )
        assert result.summary.improved_case_ids[0] == 2
        assert result.summary.improvement.largest == 1.0

    def test_regressions_are_ordered_biggest_first_too(self):
        result = compared(
            run_of({1: 2}, {2: 2}, {3: 2}),
            run_of({1: 1}, {2: 0}, {3: 1}),
        )
        assert result.summary.worsened_case_ids[0] == 2
        assert result.summary.worsening.largest == -1.0

    def test_a_magnitude_is_all_zero_when_nothing_moved_that_way(self):
        """`statistics.mean` raises on an empty list, and a run where nothing got worse has
        no worsening to report — both have to come out as a plain zero."""
        result = compared(run_of({1: 0}), run_of({1: 2}))
        assert result.summary.worsening.largest == 0.0
        assert result.summary.worsening.mean == 0.0
        assert result.summary.worsening.median == 0.0

    def test_mean_and_median_describe_only_their_own_side(self):
        result = compared(
            run_of({1: 0}, {2: 0}, {3: 2}),
            run_of({1: 2}, {2: 1}, {3: 0}),
        )
        assert result.summary.improvement.mean == pytest.approx(0.75)
        assert result.summary.improvement.median == pytest.approx(0.75)
        assert result.summary.worsening.mean == pytest.approx(-1.0)


class TestOrderIndependence:
    """Runs are matched by id, never by position — two stored runs need not share an order."""

    def test_cases_stored_in_different_orders_still_pair_up(self):
        baseline = run_of({1: 0}, {2: 2})
        candidate = run_of({1: 2}, {2: 2})
        reversed_candidate = candidate.model_copy(
            update={"case_results": list(reversed(candidate.case_results))}
        )
        result = compared(baseline, reversed_candidate)
        assert result.summary.improved_case_ids == [1]
        assert result.summary.stable_case_ids == [2]

    def test_criteria_stored_in_different_orders_still_pair_up(self):
        baseline = run_of({1: 0, 2: 2})
        candidate = run_of({1: 2, 2: 2})
        flipped = candidate.case_results[0].model_copy(
            update={
                "criterion_results": list(
                    reversed(candidate.case_results[0].criterion_results)
                )
            }
        )
        candidate = candidate.model_copy(update={"case_results": [flipped]})
        criteria = compared(baseline, candidate).case_comparisons[0].criterion_comparisons
        assert [criterion.criterion_id for criterion in criteria] == [1, 2]
        assert [criterion.score_delta for criterion in criteria] == [2.0, 0.0]

    def test_case_comparisons_come_out_in_id_order(self):
        result = compared(run_of({1: 1}, {2: 1}, {3: 1}), run_of({1: 1}, {2: 1}, {3: 1}))
        assert [case.case_id for case in result.case_comparisons] == [1, 2, 3]


class TestRefusal:
    """What is not comparable, and how loudly."""

    def test_a_case_missing_from_the_candidate_is_refused_and_named(self):
        with pytest.raises(RunsNotComparableError, match=r"cases only in the baseline: \[2\]"):
            compared(run_of({1: 1}, {2: 1}), run_of({1: 1}))

    def test_a_case_added_to_the_candidate_is_refused_and_named(self):
        with pytest.raises(RunsNotComparableError, match=r"cases only in the candidate: \[2\]"):
            compared(run_of({1: 1}), run_of({1: 1}, {2: 1}))

    def test_a_changed_rubric_inside_a_shared_case_is_refused(self):
        expected = r"case 1: criteria only in the candidate: \[9\]"
        with pytest.raises(RunsNotComparableError, match=expected):
            compared(run_of({1: 1}), run_of({1: 1, 9: 1}))

    def test_a_changed_weight_is_refused_because_the_scores_do_not_subtract(self):
        """The weights are the denominator each case score is normalized by. 0.5 under one set
        of weights and 0.5 under another are not the same 0.5, so their difference is not 0."""
        baseline = run_of({1: 2, 2: 0})
        candidate = _reweighted(run_of({1: 2, 2: 0}), {1: 1.0, 2: 3.0})
        expected = r"case 1, criterion 2: weight 1.0 vs 3.0"
        with pytest.raises(RunsNotComparableError, match=expected):
            compared(baseline, candidate)

    def test_every_difference_is_reported_at_once(self):
        """One message per re-run: fixing them one error at a time would mean re-judging the
        whole catalog for each."""
        with pytest.raises(RunsNotComparableError) as refusal:
            compared(run_of({1: 1}, {2: 1}), run_of({1: 1, 9: 1}))
        message = str(refusal.value)
        assert "cases only in the baseline: [2]" in message
        assert "case 1: criteria only in the candidate: [9]" in message

    def test_a_weight_that_differs_only_in_the_last_digit_is_still_refused(self):
        """Weights are copied verbatim from the rubric, never computed, so two runs of the
        same catalog carry bit-identical ones. "Nearly equal" therefore means the rubric was
        edited between the runs — which is exactly what must not be compared away."""
        baseline = run_of({1: 2})
        candidate = _reweighted(run_of({1: 2}), {1: 1.0 + 1e-12})
        with pytest.raises(RunsNotComparableError, match="weight 1.0 vs 1.000000000001"):
            compared(baseline, candidate)

    def test_the_refusal_is_a_value_error_so_existing_handlers_still_catch_it(self):
        """Named for the HTTP layer's sake — but a library caller that wrote
        `except ValueError` around it must keep working."""
        assert issubclass(RunsNotComparableError, ValueError)

    def test_a_bug_is_not_dressed_up_as_an_incomparable_pair(self):
        """`ValidationError` and `StatisticsError` are `ValueError` subclasses too. Catching
        the bare `ValueError` at the HTTP boundary would report either of them as "your runs
        are not comparable" — a 422 blaming the caller for our bug."""
        assert not issubclass(ValidationError, RunsNotComparableError)

    def test_a_run_without_cases_cannot_exist_to_be_compared(self):
        """`run_metrics` refuses to describe a distribution over no cases, so an empty run was
        never producible. Without the constraint on `BatchResult` the comparison of two of
        them builds fine and then divides by zero computing the rates."""
        with pytest.raises(ValidationError, match="at least 1 item"):
            BatchResult(metrics=run_of({1: 1}).metrics, case_results=[])

    def test_nothing_is_computed_before_the_refusal(self):
        """The check runs first on purpose — a half-built document is worse than none."""
        with pytest.raises(RunsNotComparableError, match="the runs are not comparable"):
            compared(run_of({1: 1}), run_of({2: 1}))


class TestJudgeOutages:
    """A failed criterion is scored 0 today, so it shows up as an ordinary regression. The
    run-level delta is what tells a reader the movement is an artefact."""

    def test_a_run_with_more_outages_is_flagged_at_the_run_level(self):
        baseline = run_of({1: 2})
        candidate = _with_failed_criterion(run_of({1: 0}), criterion_id=1)
        result = compared(baseline, candidate)
        assert result.metrics_delta.failed_criteria_count_delta == 1
        assert result.summary.worsened_case_ids == [1]


def _reweighted(run: BatchResult, weights: dict[int, float]) -> BatchResult:
    """A run whose criteria carry different weights — the one thing `run_of` keeps constant."""
    case_results = [
        _rebuilt(
            result,
            [
                verdict.model_copy(update={"weight": weights[verdict.criterion_id]})
                for verdict in result.criterion_results
            ],
        )
        for result in run.case_results
    ]
    return BatchResult(metrics=run_metrics(case_results), case_results=case_results)


def _with_failed_criterion(run: BatchResult, criterion_id: int) -> BatchResult:
    """A run in which the judge never answered for one criterion, as `evaluate_case` records it."""
    case_results = [
        _rebuilt(
            result,
            [
                CriterionResult.unjudged(
                    Criterion(id=verdict.criterion_id, content="x", weight=verdict.weight), "down"
                )
                if verdict.criterion_id == criterion_id
                else verdict
                for verdict in result.criterion_results
            ],
        )
        for result in run.case_results
    ]
    return BatchResult(metrics=run_metrics(case_results), case_results=case_results)


def _rebuilt(result: CaseResult, criterion_results: list[CriterionResult]) -> CaseResult:
    """A case result re-scored from changed verdicts, so its `score` never lies about them."""
    return CaseResult(
        case_id=result.case_id,
        score=case_score(criterion_results),
        criterion_results=criterion_results,
    )
