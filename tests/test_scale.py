"""The grading scale as a judge-owned setting: what a valid scale is, what it does to a
criterion result, and what it stops you from doing with two runs that disagree about it.

The scale story cuts across every module — model, judge, evaluation, comparison — so it is
told in one file rather than in five fragments nobody reads together.
"""

import json
from typing import cast

import pytest
from pydantic import ValidationError

from tests.conftest import CASE, FakeJudge, run_of
from rubric_judge import (
    DEFAULT_SCALE,
    Case,
    CaseResult,
    Criterion,
    CriterionResult,
    Judge,
    JudgeUnavailableError,
    RunComparison,
    RunResult,
    RunsNotComparableError,
    Scale,
    JudgeReply,
    compare_runs,
    evaluate_case,
)
from rubric_judge.metrics import case_score, run_metrics

TEN_POINT = Scale(maximum=10, presence_threshold=5)
"""A scale that shares nothing with the default: a different maximum *and* a threshold that
is not half a point, so a test passing on it cannot be passing by coincidence."""


def _criterion(criterion_id: int = 1, weight: float = 1) -> Criterion:
    return Criterion(id=criterion_id, content="x", weight=weight)


def _case_result_of(
    *criterion_results: CriterionResult, scale: Scale = DEFAULT_SCALE
) -> CaseResult:
    """A case around the given criterion results — the object that owns the scale, and
    therefore the one that has to refuse a result disagreeing with it. Its own `score` is
    irrelevant here and left at 0.0."""
    return CaseResult(
        case_id=1, score=0.0, scale=scale, criterion_results=list(criterion_results)
    )


# --------------------------------------------------------------------------- the scale itself


def test_the_default_scale_is_the_one_the_bundled_prompt_describes():
    assert (DEFAULT_SCALE.maximum, DEFAULT_SCALE.presence_threshold) == (2, 0.5)


def test_scales_compare_by_value_so_a_rebuilt_default_is_the_default():
    rebuilt = Scale(
        maximum=2,
        presence_threshold=0.5,
        level_descriptions=dict(DEFAULT_SCALE.level_descriptions),
    )
    assert rebuilt == DEFAULT_SCALE


def test_the_same_numbers_without_the_wording_are_not_the_default_scale():
    """What the judge is told about a grade is part of the scale, so a bare 0..2 is a
    different one — and a judge built on it is not handed the bundled prompt by accident."""
    assert Scale(maximum=2, presence_threshold=0.5) != DEFAULT_SCALE


def test_a_scale_is_frozen_because_a_judge_and_its_results_share_one():
    with pytest.raises(ValidationError):
        DEFAULT_SCALE.maximum = 5


def test_a_finished_result_keeps_its_wording_when_the_shared_default_is_written_into():
    """`frozen` covers the fields, not the dict behind `level_descriptions`, and every judge
    on the default shares one `DEFAULT_SCALE`. A stored run has to keep saying what its grades
    meant, so the scale is copied into the result rather than referenced out of the constant."""
    result = CaseResult(
        case_id=1,
        score=1.0,
        scale=DEFAULT_SCALE,  # named, as `evaluate_case` names `judge.scale` — not defaulted
        criterion_results=[CriterionResult.judged(_criterion(), 2, None, DEFAULT_SCALE)],
    )
    original = DEFAULT_SCALE.level_descriptions[2]
    try:
        DEFAULT_SCALE.level_descriptions[2] = "Anything at all."
        assert result.scale.level_descriptions[2] == original
    finally:
        DEFAULT_SCALE.level_descriptions[2] = original


def test_scales_that_differ_only_in_their_threshold_are_different_scales():
    """`0..2, covered from 2` and `0..2, covered from 0.5` disagree about what "covered"
    means, so `criteria_fulfillment_rate` does not subtract between them either."""
    assert Scale(maximum=2, presence_threshold=2) != DEFAULT_SCALE


@pytest.mark.parametrize("maximum", [0, -1])
def test_rejects_a_scale_with_nothing_to_reach(maximum):
    with pytest.raises(ValidationError):
        Scale(maximum=maximum, presence_threshold=0.5)


def test_rejects_a_fractional_maximum_because_the_scale_is_integral():
    with pytest.raises(ValidationError):
        Scale.model_validate({"maximum": 2.5, "presence_threshold": 0.5})


def test_rejects_a_threshold_of_zero_that_would_call_a_total_miss_covered():
    with pytest.raises(ValidationError):
        Scale(maximum=2, presence_threshold=0)


def test_rejects_a_threshold_no_score_could_reach():
    with pytest.raises(ValidationError, match="presence threshold"):
        Scale(maximum=2, presence_threshold=2.5)


def test_allows_a_threshold_at_the_maximum_where_only_full_coverage_counts():
    strict = Scale(maximum=2, presence_threshold=2)
    assert CriterionResult.judged(_criterion(), 1, None, strict).is_present is False
    assert CriterionResult.judged(_criterion(), 2, None, strict).is_present is True


def test_the_grades_of_a_scale_run_from_zero_to_its_maximum_inclusive():
    """`grades` is the one place "inclusive" is spelled out, so the validator, the prompt and
    the retry hints cannot disagree about whether the top grade is on the scale."""
    assert list(DEFAULT_SCALE.grades) == [0, 1, 2]
    assert list(Scale(maximum=1, presence_threshold=1).grades) == [0, 1]
    assert list(TEN_POINT.grades) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_a_scale_is_not_stored_on_the_criterion_result_but_on_the_case_above_it():
    """One judge, one scale, one case — so a 10-criterion case stores it once, not ten times."""
    assert "scale" not in CriterionResult.judged(_criterion(), 2, None, DEFAULT_SCALE).model_dump()
    assert _case_result_of(CriterionResult.judged(_criterion(), 2, None, DEFAULT_SCALE)).scale


@pytest.mark.parametrize("threshold", [float("nan"), float("inf")])
def test_rejects_a_threshold_that_is_not_a_real_number(threshold):
    with pytest.raises(ValidationError):
        Scale(maximum=2, presence_threshold=threshold)


# --------------------------------------------------- what a criterion result does with it


def test_presence_follows_the_scales_own_threshold_not_half_a_point():
    assert CriterionResult.judged(_criterion(), 4, None, TEN_POINT).is_present is False
    assert CriterionResult.judged(_criterion(), 5, None, TEN_POINT).is_present is True


def test_presence_cannot_contradict_the_score_it_is_defined_by():
    """A stored run claiming a covered zero is refused, not quietly believed — the invariant
    the field documents is enforced by the case, which is the object that knows the scale."""
    with pytest.raises(ValidationError, match="is_present=True"):
        _case_result_of(CriterionResult(criterion_id=1, weight=1, score=0.0, is_present=True))


def test_presence_is_refused_the_other_way_round_too():
    """A covered score reported as absent would lower `criteria_fulfillment_rate` instead."""
    with pytest.raises(ValidationError, match="is_present=False"):
        _case_result_of(CriterionResult(criterion_id=1, weight=1, score=2.0, is_present=False))


def test_presence_is_part_of_the_serialized_criterion_result():
    assert CriterionResult.judged(_criterion(), 2, None, DEFAULT_SCALE).model_dump()["is_present"]


def test_rejects_a_case_whose_criterion_result_is_graded_above_its_scale():
    with pytest.raises(ValidationError, match="above the scale"):
        _case_result_of(CriterionResult(criterion_id=1, weight=1, score=3.0, is_present=True))


def test_accepts_a_score_that_would_be_off_the_default_scale_but_fits_the_cases_own():
    case_result = CaseResult(
        case_id=1,
        score=0.9,
        scale=TEN_POINT,
        criterion_results=[CriterionResult.judged(_criterion(), 9, None, TEN_POINT)],
    )
    assert case_result.criterion_results[0].score == 9.0


def test_a_case_that_names_no_scale_is_refused_rather_than_read_on_the_default():
    """A default here would decide the unit of every score under it: a stored 0..10 case that
    lost the field would validate as a 0..2 one, and `/compare` would subtract it from a real
    0..2 run and answer 200. The document has to say what its grades were out of."""
    with pytest.raises(ValidationError, match="scale"):
        CaseResult.model_validate(
            {
                "case_id": 1,
                "score": 1.0,
                "criterion_results": [
                    {"criterion_id": 1, "weight": 1.0, "score": 2.0, "is_present": True}
                ],
            }
        )


def test_a_zero_is_never_present_on_any_scale():
    """Every presence threshold is above 0, so the bottom grade is uncovered whatever the
    scale — on the bundled 0..2 as well as on a 0..10."""
    assert CriterionResult.judged(_criterion(), 0, None, DEFAULT_SCALE).is_present is False
    assert CriterionResult.judged(_criterion(), 0, None, TEN_POINT).is_present is False


# ------------------------------------------------------------------- one scale per case and run


def test_rejects_a_run_whose_cases_disagree_about_the_scale():
    default_run = run_of({1: 2})
    on_another_scale = CaseResult(
        case_id=2,
        score=0.7,
        scale=TEN_POINT,
        criterion_results=[CriterionResult.judged(_criterion(2), 7, None, TEN_POINT)],
    )
    with pytest.raises(ValidationError, match="one scale"):
        RunResult(
            metrics=default_run.metrics,
            case_results=[*default_run.case_results, on_another_scale],
        )


def test_a_run_reports_the_scale_of_its_cases():
    assert run_of({1: 7}, scale=TEN_POINT).scale == TEN_POINT


def test_a_run_reads_its_scale_off_its_cases_instead_of_storing_it_twice():
    """`RunResult.scale` is a Python accessor, not a field: the cases already carry it, and
    a second copy on the run is a second thing that can disagree with them."""
    run = run_of({1: 7}, scale=TEN_POINT)
    assert "scale" not in run.model_dump()
    assert "scale" in run.model_dump()["case_results"][0]
    assert run.scale == TEN_POINT


def test_the_coverage_rate_of_a_run_follows_the_scales_threshold_not_its_grades():
    """`is_present` is the one metric input the threshold decides, so two runs of identical
    grades report the same raw average and a different fulfillment rate."""
    lenient = Scale(maximum=10, presence_threshold=1)
    strict_run = run_of({1: 4, 2: 6}, {21: 2, 22: 10}, scale=TEN_POINT)
    lenient_run = run_of({1: 4, 2: 6}, {21: 2, 22: 10}, scale=lenient)

    strict_metrics, lenient_metrics = strict_run.metrics, lenient_run.metrics
    assert strict_metrics.average_criterion_score == lenient_metrics.average_criterion_score
    assert strict_metrics.average_score == lenient_metrics.average_score
    assert strict_metrics.criteria_fulfillment_rate == 0.5
    assert lenient_metrics.criteria_fulfillment_rate == 1.0


def test_rejects_a_run_whose_average_criterion_score_is_off_its_own_scale():
    """The bound moved from a constant to the run's own scale, so it still has to hold."""
    run = run_of({1: 2})
    with pytest.raises(ValidationError, match="average criterion score"):
        RunResult(
            metrics=run.metrics.model_copy(update={"average_criterion_score": 4.0}),
            case_results=run.case_results,
        )


# ------------------------------------------------------------------------- scoring against it


def test_a_perfect_case_scores_one_on_any_scale():
    criterion_results = [CriterionResult.judged(_criterion(1, 3), 10, None, TEN_POINT)]
    assert case_score(criterion_results, TEN_POINT) == 1.0


def test_the_fold_refuses_a_grade_the_scale_cannot_carry():
    """`case_score` divides by the maximum, so it cannot make a share of the reachable points
    out of a grade above it — and it is the first thing `evaluate_case` calls, which is what
    makes the complaint name the criterion instead of the case score it would have produced."""
    with pytest.raises(ValueError, match=r"criteria \[1\] scored above the scale"):
        case_score([CriterionResult.judged(_criterion(1), 10, None, TEN_POINT)], DEFAULT_SCALE)


def test_the_fold_normalizes_by_the_scale_the_case_names():
    criterion_results = [
        CriterionResult.judged(_criterion(1, 3), 10, None, TEN_POINT),
        CriterionResult.judged(_criterion(2, 1), 0, None, TEN_POINT),
    ]
    assert case_score(criterion_results, TEN_POINT) == pytest.approx(0.75)  # 3*10/10 / 4


def test_the_same_shape_of_answer_scores_the_same_on_two_scales():
    """Half marks everywhere is 0.5 whether the judge counted in halves or in fifths — which
    is the whole reason the case score is normalized rather than reported raw."""
    on_default = [CriterionResult.judged(_criterion(1), 1, None, DEFAULT_SCALE)]
    on_ten = [CriterionResult.judged(_criterion(1), 5, None, TEN_POINT)]
    assert case_score(on_default, DEFAULT_SCALE) == case_score(on_ten, TEN_POINT) == 0.5


# --------------------------------------------------------------------- through an evaluation


async def test_a_custom_scale_reaches_the_results_it_produced():
    judge = FakeJudge({1: 10, 2: 0}, scale=TEN_POINT)
    result = await evaluate_case(judge, Case(**CASE))

    assert result.scale == TEN_POINT
    graded = [criterion_result.score for criterion_result in result.criterion_results]
    assert graded == [10.0, 0.0]
    assert result.score == pytest.approx(0.75)  # weights 3 and 1


async def test_an_outage_on_a_custom_scale_invalidates_the_case_like_any_other():
    """The scale changes what a grade means, never what an outage costs."""
    judge = FakeJudge({1: 10, 2: JudgeUnavailableError("endpoint down")}, scale=TEN_POINT)

    with pytest.raises(JudgeUnavailableError, match="endpoint down"):
        await evaluate_case(judge, Case(**CASE))


async def test_a_judge_that_declares_no_scale_is_a_broken_program_not_an_outage():
    """Both end the run, but only one of them is the endpoint's fault: a missing attribute
    arrives as the `AttributeError` it is, so the HTTP layer answers 500 rather than 503."""

    class ScalelessJudge:
        async def score(self, question: str, answer: str, criterion: Criterion) -> JudgeReply:
            return JudgeReply(score=2, reasoning="")

    # Cast because it is not a `Judge` and that is the whole test: what reaches
    # `evaluate_case` at runtime is whatever a caller handed it.
    with pytest.raises(AttributeError):
        await evaluate_case(cast(Judge, ScalelessJudge()), Case(**CASE))


async def test_a_judge_scoring_above_the_scale_it_declared_is_refused():
    """Checked against the judge's *own* scale, not against a constant: 11 is a fine grade on
    plenty of scales and a bug on this one, and nothing downstream could tell a case score
    above 1.0 apart from a real result."""
    with pytest.raises(ValueError, match=r"criteria \[1\] scored above the scale 0\.\.10"):
        await evaluate_case(FakeJudge({1: 11, 2: 0}, scale=TEN_POINT), Case(**CASE))


# ------------------------------------------------------------------------------- comparison


def test_refuses_to_compare_runs_judged_on_different_scales():
    with pytest.raises(RunsNotComparableError, match="different scales"):
        compare_runs(
            RunComparison(baseline=run_of({1: 2}), candidate=run_of({1: 7}, scale=TEN_POINT))
        )


def test_the_refusal_names_both_scales():
    with pytest.raises(RunsNotComparableError, match=r"0\.\.2.*0\.\.10"):
        compare_runs(
            RunComparison(baseline=run_of({1: 2}), candidate=run_of({1: 7}, scale=TEN_POINT))
        )


def test_two_runs_on_the_same_custom_scale_compare_normally():
    comparison = compare_runs(
        RunComparison(
            baseline=run_of({1: 3}, scale=TEN_POINT),
            candidate=run_of({1: 8}, scale=TEN_POINT),
        )
    )
    criterion = comparison.case_comparison_results[0].criterion_comparison_results[0]
    assert criterion.score_delta == 5.0
    assert comparison.metrics_delta.average_score_delta == pytest.approx(0.5)


def test_a_run_of_cases_on_different_scales_has_no_metrics_to_report():
    """`run_metrics` is documented as callable on stored case results, so it cannot rely on
    `RunResult` having refused the mix first."""
    on_default = run_of({1: 2}).case_results[0]
    on_ten = CaseResult(
        case_id=2,
        score=0.7,
        scale=TEN_POINT,
        criterion_results=[CriterionResult.judged(_criterion(2), 7, None, TEN_POINT)],
    )
    with pytest.raises(ValueError, match="one scale"):
        run_metrics([on_default, on_ten])


def test_two_scales_sharing_a_maximum_are_still_two_scales():
    strict = Scale(maximum=2, presence_threshold=2)
    with pytest.raises(RunsNotComparableError, match="different scales"):
        compare_runs(
            RunComparison(baseline=run_of({1: 1}), candidate=run_of({1: 1}, scale=strict))
        )


def test_a_run_whose_case_names_no_scale_is_refused_at_load():
    """`RunResult` reads its scale off its cases, so a case that names none takes the whole
    run's unit with it. Refused where the document is read, not where a delta is computed —
    by then the number already looks like a result."""
    with pytest.raises(ValidationError, match="scale"):
        RunResult.model_validate(
            {
                "metrics": run_of({1: 2}).metrics.model_dump(),
                "case_results": [
                    {
                        "case_id": 1,
                        "score": 1.0,
                        "criterion_results": [
                            {"criterion_id": 1, "weight": 1.0, "score": 2.0, "is_present": True}
                        ],
                    }
                ],
            }
        )


def test_a_case_that_names_a_null_scale_is_refused_like_one_that_names_none():
    """An explicit `null` is the shape a serializer that drops empty values writes, and the
    shape a hand-edited document ends up with. It has to be the same refusal as an absent
    field, or "required and never defaulted" holds for one spelling of missing and not the
    other."""
    stored = json.loads(run_of({1: 2}).model_dump_json())
    stored["case_results"][0]["scale"] = None

    with pytest.raises(ValidationError, match="scale"):
        RunResult.model_validate(stored)


def test_a_run_read_back_from_a_file_is_held_to_the_scale_rule_too():
    """`model_validate_json` is the documented way to load a stored run off disk, and it is
    the path a run reaches `/compare` by. Pydantic validates JSON on a separate code path from
    Python objects, so the rule is worth asserting on the one a reader actually uses."""
    stored = json.loads(run_of({1: 2}).model_dump_json())
    del stored["case_results"][0]["scale"]

    with pytest.raises(ValidationError, match="scale"):
        RunResult.model_validate_json(json.dumps(stored))


def test_a_custom_scale_survives_a_json_round_trip():
    """The scale is part of the published result, so a run written to disk on a ten-point
    scale is still a ten-point run when it is read back."""
    run = run_of({1: 7}, scale=TEN_POINT)
    reloaded = RunResult.model_validate_json(run.model_dump_json())
    assert reloaded.scale == TEN_POINT
    assert reloaded.case_results[0].scale == TEN_POINT
    assert reloaded.case_results[0].criterion_results[0].is_present is True
    assert reloaded == run


# --------------------------------------------------------------- what the levels mean


def test_the_default_scale_describes_all_three_of_its_levels():
    assert sorted(DEFAULT_SCALE.level_descriptions) == [0, 1, 2]
    assert DEFAULT_SCALE.level_descriptions[2].startswith("Fully covered.")


def test_a_scale_may_describe_no_levels_at_all():
    """Arithmetic only, for a judge that brings its own prompt — the scale still has to say
    what its grades are worth, just not what they mean in words."""
    assert Scale(maximum=5, presence_threshold=3).level_descriptions == {}


def test_rejects_descriptions_that_leave_a_grade_unexplained():
    with pytest.raises(ValidationError, match="describe every grade"):
        Scale(maximum=2, presence_threshold=0.5, level_descriptions={2: "Yes.", 0: "No."})


def test_rejects_a_description_for_a_grade_the_scale_does_not_have():
    with pytest.raises(ValidationError, match="describe every grade"):
        Scale(
            maximum=1,
            presence_threshold=1,
            level_descriptions={2: "Yes.", 1: "Partly.", 0: "No."},
        )


def test_rejects_a_blank_description_that_would_leave_the_prompt_with_a_bare_number():
    with pytest.raises(ValidationError):
        Scale(maximum=1, presence_threshold=1, level_descriptions={1: "Yes.", 0: "   "})


def test_rewording_a_level_makes_it_a_different_scale():
    """Descriptions are part of the scale's value, so they are part of what `/compare` holds
    two runs against: telling the judge something else about a 1 changes the grades it gives."""
    reworded = DEFAULT_SCALE.model_copy(
        update={"level_descriptions": DEFAULT_SCALE.level_descriptions | {1: "Halfway there."}}
    )
    assert reworded != DEFAULT_SCALE
    with pytest.raises(RunsNotComparableError, match="different scales"):
        compare_runs(
            RunComparison(baseline=run_of({1: 1}), candidate=run_of({1: 1}, scale=reworded))
        )


def test_the_refusal_says_which_grade_was_reworded_when_the_scales_print_alike():
    """Both sides print `0..2 (covered from 0.5)`, so naming them twice would read as a
    contradiction rather than as a cause."""
    reworded = DEFAULT_SCALE.model_copy(
        update={"level_descriptions": DEFAULT_SCALE.level_descriptions | {1: "Halfway there."}}
    )
    with pytest.raises(RunsNotComparableError, match=r"the wording of \[1\] differs"):
        compare_runs(
            RunComparison(baseline=run_of({1: 1}), candidate=run_of({1: 1}, scale=reworded))
        )


def test_a_run_says_which_grade_was_reworded_too_when_its_cases_print_alike():
    """The same explanation at the run grain: `one_scale_of` fires first and on more paths —
    `RunResult`, `run_metrics` on stored results — so it is the message most callers hit."""
    reworded = DEFAULT_SCALE.model_copy(
        update={"level_descriptions": DEFAULT_SCALE.level_descriptions | {1: "Halfway there."}}
    )
    on_reworded = run_of({2: 1}, scale=reworded).case_results[0]
    with pytest.raises(ValueError, match=r"the wording of \[1\] differs"):
        run_metrics([run_of({1: 1}).case_results[0], on_reworded])


def test_a_run_of_scales_that_print_differently_needs_no_such_clause():
    """The two names already explain it — the clause is only there for scales that print
    alike, at either grain."""
    on_ten_point = run_of({2: 7}, scale=TEN_POINT).case_results[0]
    both_named = r"0\.\.10 \(covered from 5\.0\), 0\.\.2 \(covered from 0\.5\)$"
    with pytest.raises(ValueError, match=both_named):
        run_metrics([run_of({1: 1}).case_results[0], on_ten_point])


def test_a_refusal_over_different_maximums_stays_short():
    """The two names already explain it, so listing eleven reworded grades would be noise."""
    ten_point = Scale(maximum=10, presence_threshold=5)
    with pytest.raises(RunsNotComparableError, match=r"candidate 0\.\.10 \(covered from 5\.0\)$"):
        compare_runs(
            RunComparison(baseline=run_of({1: 1}), candidate=run_of({1: 1}, scale=ten_point))
        )


def test_descriptions_survive_a_json_round_trip_with_their_grades_as_numbers():
    """JSON object keys are strings, so the grades have to come back as ints or every lookup
    by grade would miss."""
    reloaded = Scale.model_validate_json(DEFAULT_SCALE.model_dump_json())
    assert reloaded == DEFAULT_SCALE
    assert reloaded.level_descriptions[2] == DEFAULT_SCALE.level_descriptions[2]
