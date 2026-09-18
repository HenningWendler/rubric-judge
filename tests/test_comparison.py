"""Comparing two finished runs: the deltas, the classification, and what is refused.

Every run here is built by `run_of` from plain judge scores, so a test reads as scores in
and deltas out with no judge and no event loop in between.
"""

import json

import pytest
from pydantic import ValidationError

from tests.conftest import run_of
from rubric_eval import (
    SCORE_EQUALITY_TOLERANCE,
    ChangeStatus,
    RunComparison,
    RunMetricsDelta,
    RunsNotComparableError,
    compare_runs,
)
from rubric_eval.comparison import _metric_difference
from rubric_eval.metrics import case_score, run_metrics
from rubric_eval.models import DEFAULT_SCALE, CaseResult, CriterionResult, RunResult


def compared(baseline: RunResult, candidate: RunResult):
    """The result of comparing two runs — the whole file's one line of setup."""
    return compare_runs(RunComparison(baseline=baseline, candidate=candidate))


UNEVEN_BASELINE = run_of({1: 0, 2: 0}, {3: 0, 4: 1}, {5: 0, 6: 2})
UNEVEN_CANDIDATE = run_of({1: 1, 2: 1}, {3: 2, 4: 2}, {5: 2, 6: 2})
"""Three cases scoring 0.0 / 0.25 / 0.5 against 0.5 / 1.0 / 1.0. Every run metric differs
between the two, and the six float deltas they produce are pairwise different and equal to
no raw metric of either run — so a delta reading the wrong pair cannot come out right by
accident, the way it could between two runs of uniform scores."""


class TestDirection:
    """Every delta is candidate minus baseline, so positive always means "the candidate won"."""

    def test_a_better_candidate_gives_positive_deltas(self):
        result = compared(run_of({1: 0}), run_of({1: 2}))
        assert result.metrics_delta.average_score_delta == 1.0
        assert result.case_comparison_results[0].score_delta == 1.0
        assert result.case_comparison_results[0].criterion_comparison_results[0].score_delta == 2.0

    def test_a_worse_candidate_gives_negative_deltas(self):
        result = compared(run_of({1: 2}), run_of({1: 0}))
        assert result.metrics_delta.average_score_delta == -1.0
        assert result.case_comparison_results[0].score_delta == -1.0

    def test_swapping_the_sides_flips_the_whole_document_not_only_the_average(self):
        """`RunComparison` exists because a swapped pair would be undetectable. What such a swap
        would cost is every number in the result, so every number has to invert: each delta
        changes sign, the movers trade lists, and each magnitude mirrors the other side."""
        forward = compared(UNEVEN_BASELINE, UNEVEN_CANDIDATE)
        backward = compared(UNEVEN_CANDIDATE, UNEVEN_BASELINE)

        for field in RunMetricsDelta.model_fields:
            assert getattr(forward.metrics_delta, field) == -getattr(backward.metrics_delta, field)
        assert forward.summary.improved_case_ids == backward.summary.worsened_case_ids
        assert forward.summary.improvement.largest == -backward.summary.worsening.largest
        assert forward.summary.improvement.mean == -backward.summary.worsening.mean
        assert forward.summary.improvement.median == -backward.summary.worsening.median

    def test_every_metric_delta_subtracts_its_own_pair_of_metrics(self):
        """One delta per `RunMetrics` field that subtracts, each reading the *same* metric on
        both sides. These six values are pairwise different and match neither run's raw
        metrics, so a delta wired to the wrong metric — or to the candidate alone — cannot
        come out right by accident."""
        delta = compared(UNEVEN_BASELINE, UNEVEN_CANDIDATE).metrics_delta

        assert delta.average_score_delta == pytest.approx(0.5833333333333334)
        assert delta.median_score_delta == pytest.approx(0.75)
        assert delta.variance_delta == pytest.approx(0.02083333333333333)
        assert delta.standard_deviation_delta == pytest.approx(0.038675134594812866)
        assert delta.average_criterion_score_delta == pytest.approx(1.1666666666666667)
        assert delta.criteria_fulfillment_rate_delta == pytest.approx(0.6666666666666667)
        assert delta.cases_with_score_zero_count_delta == -1

    def test_a_metric_that_cannot_be_subtracted_is_refused_not_skipped(self):
        """The deltas are derived from the fields `RunMetricsDelta` declares, so a delta
        naming a metric that does not subtract — an id list, say — has to stop the comparison.
        Skipped instead, it would leave the field at whatever a missing value defaults to and
        report "this metric did not move"."""
        metrics = UNEVEN_BASELINE.metrics

        with pytest.raises(TypeError, match="cases_with_score_zero does not subtract"):
            _metric_difference("cases_with_score_zero_delta", metrics, metrics)

    def test_a_run_compared_with_itself_moves_nothing(self):
        """The fixed point of a comparison, and the cheapest check that every delta really
        subtracts: hand in the same document twice and no field may report a movement."""
        result = compared(UNEVEN_BASELINE, UNEVEN_BASELINE)

        assert all(
            getattr(result.metrics_delta, field) == 0 for field in RunMetricsDelta.model_fields
        )
        assert result.summary.stable_case_ids == [1, 2, 3]
        assert result.summary.stability_rate == 1.0

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
        criterion = result.case_comparison_results[0].criterion_comparison_results[0]
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
            for criterion in result.case_comparison_results[0].criterion_comparison_results
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
        case = compared(baseline, candidate).case_comparison_results[0]
        assert case.status is ChangeStatus.STABLE

    def test_a_delta_of_exactly_the_tolerance_is_stable_and_twice_it_is_not(self):
        """"Stable" is documented as equal *within* `SCORE_EQUALITY_TOLERANCE` and "improved"
        as more than it, so the tolerance itself falls on the stable side. A baseline of 0.0
        subtracts exactly, which puts the assertion on the boundary rather than on float
        noise near it."""
        assert _case_status_for_score(SCORE_EQUALITY_TOLERANCE) is ChangeStatus.STABLE
        assert _case_status_for_score(2 * SCORE_EQUALITY_TOLERANCE) is ChangeStatus.IMPROVED
        assert _case_status_for_score(-SCORE_EQUALITY_TOLERANCE) is ChangeStatus.STABLE
        assert _case_status_for_score(-2 * SCORE_EQUALITY_TOLERANCE) is ChangeStatus.WORSENED

    def test_a_case_can_be_stable_while_its_criteria_move_against_each_other(self):
        """The reason the criterion grain exists: equal weights cancelling out leave the case
        score untouched and the answer materially different."""
        result = compared(run_of({1: 2, 2: 0}), run_of({1: 0, 2: 2}))
        case = result.case_comparison_results[0]
        assert case.status is ChangeStatus.STABLE
        assert [criterion.status for criterion in case.criterion_comparison_results] == [
            ChangeStatus.WORSENED,
            ChangeStatus.IMPROVED,
        ]


class TestSummary:
    """Who moved which way, in which order, and by how much."""

    def test_cases_are_split_over_the_three_lists_exactly_once(self):
        """Every case lands in exactly one list, so the three counts add up to the run. The
        six-case split is deliberate: the three *rates* are three separate divisions and sum
        to 0.9999999999999999 here, so only the counts can carry an exactness claim."""
        summary = compared(
            run_of({1: 0}, {2: 2}, {3: 1}, {4: 1}, {5: 1}, {6: 1}),
            run_of({1: 2}, {2: 0}, {3: 1}, {4: 1}, {5: 1}, {6: 1}),
        ).summary

        assert summary.improved_case_ids == [1]
        assert summary.worsened_case_ids == [2]
        assert summary.stable_case_ids == [3, 4, 5, 6]
        assert (
            summary.improved_case_count
            + summary.stable_case_count
            + summary.worsened_case_count
        ) == 6
        assert (
            summary.improvement_rate + summary.stability_rate + summary.worsening_rate
        ) == pytest.approx(1.0)

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

    def test_a_magnitude_is_null_when_nothing_moved_that_way(self):
        """A run where nothing got worse has no worsening to report, and 0.0 would report one
        of exactly zero: a JSON consumer reading the magnitude alone cannot tell those apart,
        because only the emptiness of the id list next to it says which it is."""
        result = compared(run_of({1: 0}), run_of({1: 2}))

        assert result.summary.worsened_case_ids == []
        assert result.summary.worsening.largest is None
        assert result.summary.worsening.mean is None
        assert result.summary.worsening.median is None

    def test_a_magnitude_of_null_survives_the_json_a_stored_comparison_is_read_from(self):
        """The `null`s are the published answer, not an in-memory nicety: a reader of the
        stored document has to see "no data" where a zero used to stand."""
        result = compared(run_of({1: 0}), run_of({1: 2}))

        assert json.loads(result.model_dump_json())["summary"]["worsening"] == {
            "largest": None,
            "mean": None,
            "median": None,
        }

    def test_a_magnitude_median_sits_between_the_two_middle_moves(self):
        """Four improvements of 0.25, 0.5, 1.0 and 1.0: the median of an even-sized side lies
        between the two middle ones and is therefore a move no case actually made, while the
        mean is a third value again. Reporting either in place of the other, or in place of
        `largest`, would go unnoticed on a side whose moves are all the same size."""
        summary = compared(
            run_of({1: 0, 2: 0}, {3: 0, 4: 0}, {5: 0, 6: 0}, {7: 0, 8: 0}),
            run_of({1: 1, 2: 0}, {3: 1, 4: 1}, {5: 2, 6: 2}, {7: 2, 8: 2}),
        ).summary

        assert summary.improved_case_ids == [3, 4, 2, 1]
        assert summary.improvement.largest == 1.0
        assert summary.improvement.mean == pytest.approx(0.6875)
        assert summary.improvement.median == pytest.approx(0.75)

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
        case = compared(baseline, candidate).case_comparison_results[0]
        criteria = case.criterion_comparison_results
        assert [criterion.criterion_id for criterion in criteria] == [1, 2]
        assert [criterion.score_delta for criterion in criteria] == [2.0, 0.0]

    def test_criteria_are_sorted_by_id_even_when_neither_run_stored_them_that_way(self):
        """Ordering by id is what makes the document independent of both storage orders. Here
        neither run is in id order and the two disagree with each other, so a comparison that
        trusted either order — or sorted only one side — would pair unrelated results and
        report deltas for criteria that never moved."""
        case = compared(
            run_of({300: 0, 7: 1, 50: 2}),
            run_of({7: 2, 300: 1, 50: 0}),
        ).case_comparison_results[0]

        assert [criterion.criterion_id for criterion in case.criterion_comparison_results] == [
            7,
            50,
            300,
        ]
        assert [criterion.score_delta for criterion in case.criterion_comparison_results] == [
            1.0,
            -2.0,
            1.0,
        ]

    def test_case_comparison_results_come_out_in_id_order(self):
        result = compared(run_of({1: 1}, {2: 1}, {3: 1}), run_of({1: 1}, {2: 1}, {3: 1}))
        assert [case.case_id for case in result.case_comparison_results] == [1, 2, 3]


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

    def test_a_changed_weight_is_named_beside_the_other_kinds_of_difference(self):
        """The three rules are checked by two different code paths — ids by set difference,
        weights by value — and all of them have to reach the one message. A pair that breaks
        every rule at once is the case where reporting only the first would cost three full
        re-runs of the catalog to discover the other two."""
        baseline = run_of({1: 2, 2: 1}, {3: 1})
        candidate = _reweighted(run_of({1: 2, 2: 1, 9: 1}), {1: 1.0, 2: 5.0, 9: 1.0})

        with pytest.raises(RunsNotComparableError) as refusal:
            compared(baseline, candidate)

        message = str(refusal.value)
        assert "cases only in the baseline: [2]" in message
        assert "case 1: criteria only in the candidate: [9]" in message
        assert "case 1, criterion 2: weight 1.0 vs 5.0" in message

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
        never producible. Without the constraint on `RunResult` the comparison of two of
        them builds fine and then divides by zero computing the rates."""
        with pytest.raises(ValidationError, match="at least 1 item"):
            RunResult(metrics=run_of({1: 1}).metrics, case_results=[])

    def test_a_run_naming_the_same_case_twice_cannot_exist_to_be_compared(self):
        """`Run.cases` rejects repeated ids, so `evaluate_run` never produces such a run —
        but it is postable, and cases are paired by id. Without the constraint the second of
        a repeated pair silently replaces the first, and the comparison reports deltas over
        fewer cases than its own `metrics` block describes."""
        run = run_of({1: 1}, {2: 1})
        twice = [run.case_results[0], run.case_results[1].model_copy(update={"case_id": 1})]
        with pytest.raises(ValidationError, match=r"case ids must be unique, repeated: \[1\]"):
            RunResult(metrics=run.metrics, case_results=twice)

    def test_a_case_naming_the_same_criterion_twice_cannot_exist_to_be_compared(self):
        """Same hole one level down: results are paired by criterion id, so a repeated one
        would drop a result out of the comparison while still counting in the case score."""
        criterion_results = run_of({1: 1}).case_results[0].criterion_results
        with pytest.raises(ValidationError, match=r"criterion ids must be unique, repeated: \[1\]"):
            CaseResult(
                case_id=1,
                score=0.5,
                scale=DEFAULT_SCALE,
                criterion_results=criterion_results * 2,
            )

    def test_nothing_is_computed_before_the_refusal(self):
        """The check runs first on purpose — a half-built document is worse than none."""
        with pytest.raises(RunsNotComparableError, match="the runs are not comparable"):
            compared(run_of({1: 1}), run_of({2: 1}))


def _case_status_for_score(candidate_score: float) -> ChangeStatus:
    """A one-case comparison whose baseline scores an exact 0.0, so the delta is the
    candidate score itself and the tolerance can be asserted on without float noise."""
    baseline = run_of({1: 0})
    candidate = baseline.model_copy(
        update={
            "case_results": [
                baseline.case_results[0].model_copy(update={"score": candidate_score})
            ]
        }
    )
    return compared(baseline, candidate).case_comparison_results[0].status


def _reweighted(run: RunResult, weights: dict[int, float]) -> RunResult:
    """A run whose criteria carry different weights — the one thing `run_of` keeps constant."""
    case_results = [
        _rebuilt(
            result,
            [
                criterion_result.model_copy(
                    update={"weight": weights[criterion_result.criterion_id]}
                )
                for criterion_result in result.criterion_results
            ],
        )
        for result in run.case_results
    ]
    return RunResult(metrics=run_metrics(case_results), case_results=case_results)


def _rebuilt(result: CaseResult, criterion_results: list[CriterionResult]) -> CaseResult:
    """A case result re-scored from changed criterion results, so its `score` never lies."""
    return CaseResult(
        case_id=result.case_id,
        score=case_score(criterion_results, result.scale),
        scale=result.scale,
        criterion_results=criterion_results,
    )
