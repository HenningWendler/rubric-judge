"""The evaluation layer on its own: fan-out, failure policy and the weighted fold — no HTTP.

The failure policy is the theme of half of this file: a judge that cannot answer invalidates
everything it was judging, and a bug does too. Nothing here ever comes back with a hole in it.
"""

import asyncio

import pytest
from pydantic import ValidationError

from tests.conftest import CASE, RUN, RUN_SCORES, FakeJudge

from rubric_eval import (
    DEFAULT_SCALE,
    Case,
    JudgeReply,
    JudgeUnavailableError,
    Run,
    evaluate_case,
    evaluate_run,
)

THE_CASE = Case(**CASE)
THE_RUN = Run(**RUN)


async def test_weighted_score_and_reasoning():
    result = await evaluate_case(FakeJudge({1: 2, 2: 0}), THE_CASE)

    assert result.score == pytest.approx(0.75)  # (3*2/2 + 1*0/2) / 4
    assert [criterion.score for criterion in result.criterion_results] == [2.0, 0.0]
    assert [criterion.is_present for criterion in result.criterion_results] == [True, False]
    assert result.criterion_results[0].reasoning == "reasoning for 1"


async def test_one_unanswered_criterion_invalidates_the_whole_case():
    """The policy in one test. The other criterion was graded and the case would fold to a
    perfectly plausible 0.75 around the gap — which is exactly why there is no result: a 0
    nobody judged cannot be told apart from an answer that really missed the criterion."""
    judge = FakeJudge({1: 2, 2: JudgeUnavailableError("judge down")})

    with pytest.raises(JudgeUnavailableError, match="judge down"):
        await evaluate_case(judge, THE_CASE)


async def test_criterion_results_keep_the_order_of_the_rubric():
    """`asyncio.gather` preserves argument order — results can be zipped with the criteria."""
    result = await evaluate_case(FakeJudge({1: 0, 2: 2}), THE_CASE)

    assert [criterion.criterion_id for criterion in result.criterion_results] == [1, 2]


async def test_a_total_outage_produces_no_result_at_all():
    """The extreme case reads the same way: a run against a dead endpoint has to be
    recognisable as one, and a case scoring 0.0 everywhere is what a genuinely terrible
    system looks like."""
    outage = JudgeUnavailableError("endpoint down")

    with pytest.raises(JudgeUnavailableError, match="endpoint down"):
        await evaluate_case(FakeJudge({1: outage, 2: outage}), THE_CASE)


async def test_an_empty_answer_is_judged_rather_than_rejected():
    """A system under test that returned nothing is a valid case scoring 0 — not a 422.
    The judge is still asked, because "nothing" can only be graded against the rubric."""
    empty = Case(**{**CASE, "answer": ""})

    result = await evaluate_case(FakeJudge({1: 0, 2: 0}), empty)

    assert result.score == 0.0


async def test_duplicate_criterion_ids_are_rejected():
    """`CriterionResult.criterion_id` is documented as the way to match results to the rubric
    without relying on list order. Two criteria sharing an id make that impossible — and a
    caller merging by id would drop or double-count a result."""
    with pytest.raises(ValueError):
        Case.model_validate(
            {
                "id": 1,
                "question": "q",
                "answer": "a",
                "criteria": [
                    {"id": 7, "content": "first", "weight": 1},
                    {"id": 7, "content": "second", "weight": 1},
                ],
            }
        )


async def test_a_large_rubric_is_judged_completely_and_in_order():
    """The fan-out has to survive a rubric far bigger than the example case, with every
    result still zippable against the criteria it came from."""
    many = Case.model_validate(
        {
            "id": 1,
            "question": "q",
            "answer": "a",
            "criteria": [
                {"id": i, "content": f"criterion {i}", "weight": 1} for i in range(200)
            ],
        }
    )

    result = await evaluate_case(FakeJudge({i: 2 for i in range(200)}), many)

    assert [criterion.criterion_id for criterion in result.criterion_results] == list(range(200))
    assert result.score == 1.0


async def test_cancellation_aborts_the_case():
    """A cancelled request (client disconnect, shutdown) must abort the work rather than
    finish paying for a result nobody is waiting for."""

    class CancellingJudge:
        scale = DEFAULT_SCALE

        async def score(self, question, answer, criterion):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await evaluate_case(CancellingJudge(), THE_CASE)


async def test_a_bug_reaches_the_caller_as_itself():
    """A broken program and a dead endpoint both end the case, but they must not arrive as
    the same thing: a `RuntimeError` — asyncio misuse, for instance — stays a `RuntimeError`,
    so the HTTP layer answers 500 where an outage gets a 503."""
    with pytest.raises(RuntimeError, match="Semaphore is bound to a different event loop"):
        await evaluate_case(
            FakeJudge({1: RuntimeError("Semaphore is bound to a different event loop"), 2: 2}),
            THE_CASE,
        )


async def test_a_custom_judge_reports_its_outage_in_the_shared_vocabulary():
    """`Judge` is a Protocol, and `JudgeUnavailableError` is the one word an implementation
    has to speak: it is what says "my endpoint could not answer" rather than "I am broken",
    and it is what the HTTP layer answers 503 to. Subclassing keeps the implementation's own
    type for anyone catching it."""

    class QuotaExceeded(JudgeUnavailableError):
        pass

    with pytest.raises(QuotaExceeded, match="no credit left"):
        await evaluate_case(FakeJudge({1: QuotaExceeded("no credit left"), 2: 2}), THE_CASE)


async def test_a_grade_off_the_scale_is_refused_rather_than_folded_into_the_score():
    """`Judge` is a Protocol, so a custom implementation can answer 5 where the scale ends at
    2. That is a bug in the judge, not an outage, and it is refused rather than folded into a
    case score above 1.0 — a number no reader downstream could tell from a real one. The
    complaint names the criterion and the scale, not the case score it would have produced."""

    class OffScaleJudge:
        scale = DEFAULT_SCALE

        async def score(self, question, answer, criterion) -> JudgeReply:
            return JudgeReply(score=5, reasoning="way past the top of the scale")

    with pytest.raises(ValueError, match=r"criteria \[1, 2\] scored above the scale 0\.\.2"):
        await evaluate_case(OffScaleJudge(), THE_CASE)


# --- run: many cases in one call ----------------------------------------------------------


async def test_a_run_returns_every_case_result_in_request_order():
    result = await evaluate_run(FakeJudge(RUN_SCORES), THE_RUN)

    assert [case.case_id for case in result.case_results] == [1, 2, 3]
    assert [case.score for case in result.case_results] == pytest.approx([0.75, 0.5, 0.0])


async def test_a_run_folds_its_case_scores_into_run_metrics():
    """The point of the run over calling `evaluate_case` in a loop: one aggregate."""
    result = await evaluate_run(FakeJudge(RUN_SCORES), THE_RUN)

    assert result.metrics.total_cases == 3
    assert result.metrics.average_score == pytest.approx((0.75 + 0.5 + 0.0) / 3)
    assert result.metrics.cases_with_score_zero == [3]


async def test_a_run_entry_is_the_very_same_result_as_a_single_evaluation():
    """Both paths return a `CaseResult`, so there is nothing left that could disagree. The
    one type is what guarantees it — this test only keeps the two entry points honest."""
    judge = FakeJudge(RUN_SCORES)

    in_run = (await evaluate_run(judge, THE_RUN)).case_results[0]
    alone = await evaluate_case(judge, THE_CASE)

    assert in_run == alone


async def test_one_unanswered_criterion_invalidates_the_whole_run():
    """The same policy one grain up, and the reason it has to reach that far: the run metrics
    average the cases against each other, so a single fabricated 0 moves every number in the
    document — including the ones about cases the judge answered for perfectly well."""
    judge = FakeJudge({**RUN_SCORES, 21: JudgeUnavailableError("endpoint down")})

    with pytest.raises(JudgeUnavailableError, match="endpoint down"):
        await evaluate_run(judge, THE_RUN)


async def test_a_custom_outage_type_invalidates_a_whole_run_too():
    """`JudgeUnavailableError` is documented as subclassable, so the policy has to be keyed on
    the *type* and not on an exact match. Asserted at the run grain as well, because that is
    where a `except JudgeUnavailableError` that missed a subclass would silently degrade into
    the "bug in the program" path and report a 500 for an outage."""

    class QuotaExceeded(JudgeUnavailableError):
        pass

    judge = FakeJudge({**RUN_SCORES, 21: QuotaExceeded("no credit left")})

    with pytest.raises(QuotaExceeded, match="no credit left"):
        await evaluate_run(judge, THE_RUN)


async def test_a_bug_in_one_case_aborts_the_whole_run_as_itself():
    """A bug ends the run like an outage does, and stays distinguishable from one."""
    judge = FakeJudge({**RUN_SCORES, 21: RuntimeError("bound to a different event loop")})

    with pytest.raises(RuntimeError):
        await evaluate_run(judge, THE_RUN)


async def test_duplicate_case_ids_are_rejected():
    """Run metrics name cases by id — `cases_with_score_zero` would be ambiguous otherwise."""
    with pytest.raises(ValueError, match="case ids must be unique"):
        Run.model_validate(
            {"cases": [{**RUN["cases"][0], "id": 5}, {**RUN["cases"][1], "id": 5}]}
        )


async def test_an_empty_run_is_rejected_before_any_judge_call():
    """An empty run has no meaningful metrics, and a 422 costs nothing."""
    with pytest.raises(ValueError):
        Run(cases=[])


class _JudgeByAnswer:
    """Scores by the answer rather than by the criterion id — the shared-rubric case needs it,
    because there the same criterion id appears in both cases."""

    scale = DEFAULT_SCALE

    async def score(self, question, answer, criterion):
        return JudgeReply(score=2 if "HR tool" in answer else 0, reasoning=answer)


async def test_criterion_ids_may_repeat_across_the_cases_of_a_run():
    """Criterion ids are documented as unique *within* a case. Two cases written from the
    same rubric template share them, and their results still belong to their own case."""
    shared_rubric = [{"id": 1, "content": "Submit it in the HR tool", "weight": 1}]
    run = Run.model_validate(
        {
            "cases": [
                {"id": 1, "question": "q", "answer": "In the HR tool.", "criteria": shared_rubric},
                {"id": 2, "question": "q", "answer": "No idea.", "criteria": shared_rubric},
            ]
        }
    )

    result = await evaluate_run(_JudgeByAnswer(), run)

    assert [case.case_id for case in result.case_results] == [1, 2]
    assert [case.score for case in result.case_results] == [1.0, 0.0]
    assert result.metrics.cases_with_score_zero == [2]


async def test_an_empty_question_is_accepted_because_it_is_never_scored():
    """The question is context for the judge, not something the rubric holds against the
    answer — so a case without one is valid rather than a 422."""
    result = await evaluate_case(FakeJudge({1: 2, 2: 2}), Case(**{**CASE, "question": ""}))

    assert result.score == 1.0
