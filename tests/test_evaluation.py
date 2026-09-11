"""The evaluation layer on its own: fan-out, failure policy and the weighted fold — no HTTP."""

import asyncio

import pytest

from conftest import BATCH, BATCH_VERDICTS, CASE, FakeJudge

from rubric_eval import Batch, Case, Verdict, evaluate_batch, evaluate_case

THE_CASE = Case(**CASE)
THE_BATCH = Batch(**BATCH)


async def test_weighted_score_and_reasoning():
    result = await evaluate_case(FakeJudge({1: 2, 2: 0}), THE_CASE)

    assert result.score == pytest.approx(0.75)  # (3*2/2 + 1*0/2) / 4
    assert [criterion.score for criterion in result.criterion_results] == [2.0, 0.0]
    assert [criterion.is_present for criterion in result.criterion_results] == [True, False]
    assert result.criterion_results[0].reasoning == "reasoning for 1"
    assert result.criterion_results[0].failed is False


async def test_a_failed_criterion_scores_zero_and_stays_in_the_denominator():
    result = await evaluate_case(FakeJudge({1: 2, 2: ValueError("judge down")}), THE_CASE)

    assert result.score == pytest.approx(0.75)  # same score: a 0 and a failed 0 weigh alike
    assert result.criterion_results[1].failed is True
    assert "judge down" in result.criterion_results[1].reasoning


async def test_verdicts_keep_the_order_of_the_rubric():
    """`asyncio.gather` preserves argument order — results can be zipped with the criteria."""
    result = await evaluate_case(FakeJudge({1: 0, 2: 2}), THE_CASE)

    assert [criterion.criterion_id for criterion in result.criterion_results] == [1, 2]


async def test_every_criterion_failing_still_yields_a_complete_result():
    """A total outage must not raise: the caller gets a 0.0 and sees why, per criterion."""
    outage = FakeJudge({1: ConnectionError("endpoint down"), 2: ConnectionError("endpoint down")})

    result = await evaluate_case(outage, THE_CASE)

    assert result.score == 0.0
    assert [criterion.failed for criterion in result.criterion_results] == [True, True]


async def test_a_failure_without_a_message_still_names_its_cause():
    """`str(TimeoutError())` is the empty string, so the plain message alone loses the reason
    entirely — and a timeout is the most likely judge failure of all."""
    result = await evaluate_case(FakeJudge({1: 2, 2: TimeoutError()}), THE_CASE)

    assert result.criterion_results[1].failed is True
    assert "TimeoutError" in result.criterion_results[1].reasoning


async def test_an_empty_answer_is_judged_rather_than_rejected():
    """A system under test that returned nothing is a valid case scoring 0 — not a 422.
    The judge is still asked, because "nothing" can only be graded against the rubric."""
    empty = Case(**{**CASE, "answer": ""})

    result = await evaluate_case(FakeJudge({1: 0, 2: 0}), empty)

    assert result.score == 0.0


async def test_duplicate_criterion_ids_are_rejected():
    """`CriterionResult.criterion_id` is documented as the way to match results to the rubric
    without relying on list order. Two criteria sharing an id make that impossible — and a
    caller merging by id would drop or double-count a verdict."""
    with pytest.raises(ValueError):
        Case(
            id=1,
            question="q",
            answer="a",
            criteria=[
                {"id": 7, "content": "first", "weight": 1},
                {"id": 7, "content": "second", "weight": 1},
            ],
        )


async def test_a_large_rubric_is_judged_completely_and_in_order():
    """The fan-out has to survive a rubric far bigger than the example case, with every
    verdict still zippable against the criteria it came from."""
    many = Case(
        id=1,
        question="q",
        answer="a",
        criteria=[{"id": i, "content": f"criterion {i}", "weight": 1} for i in range(200)],
    )

    result = await evaluate_case(FakeJudge({i: 2 for i in range(200)}), many)

    assert [criterion.criterion_id for criterion in result.criterion_results] == list(range(200))
    assert result.score == 1.0


async def test_cancellation_is_not_swallowed_by_the_failure_containment():
    """A cancelled request (client disconnect, shutdown) must abort, not come back as a
    plausible-looking 0.0 with every criterion marked failed."""

    class CancellingJudge:
        async def score(self, question, answer, criterion):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await evaluate_case(CancellingJudge(), THE_CASE)


async def test_a_bug_is_re_raised_instead_of_being_scored_as_a_failed_criterion():
    """The containment exists for a judge that cannot answer, not for a broken program. A
    `RuntimeError` — asyncio misuse, for instance — must reach the caller: scoring it 0 would
    hand back a number indistinguishable from a real result."""
    with pytest.raises(RuntimeError, match="Semaphore is bound to a different event loop"):
        await evaluate_case(
            FakeJudge({1: RuntimeError("Semaphore is bound to a different event loop"), 2: 2}),
            THE_CASE,
        )


async def test_a_dead_endpoint_is_still_only_one_criterion():
    """The other side of that line: everything an endpoint can do to you stays contained."""
    result = await evaluate_case(FakeJudge({1: ConnectionError("refused"), 2: 2}), THE_CASE)

    assert result.criterion_results[0].failed is True
    assert result.criterion_results[1].score == 2.0


async def test_a_custom_judges_own_error_type_is_still_contained():
    """`Judge` is a Protocol: an implementation may raise whatever it likes for an outage,
    and only the bug-shaped built-ins are excluded from the containment."""

    class QuotaExceeded(Exception):
        pass

    result = await evaluate_case(FakeJudge({1: QuotaExceeded("no credit left"), 2: 2}), THE_CASE)

    assert result.criterion_results[0].failed is True
    assert "no credit left" in result.criterion_results[0].reasoning


# --- batch: many cases in one call ----------------------------------------------------------

THE_BATCH = Batch(**BATCH)


async def test_a_batch_returns_every_case_result_in_request_order():
    result = await evaluate_batch(FakeJudge(BATCH_VERDICTS), THE_BATCH)

    assert [case.case_id for case in result.case_results] == [1, 2, 3]
    assert [case.score for case in result.case_results] == pytest.approx([0.75, 0.5, 0.0])


async def test_a_batch_folds_its_case_scores_into_run_metrics():
    """The point of the batch over calling `evaluate_case` in a loop: one aggregate."""
    result = await evaluate_batch(FakeJudge(BATCH_VERDICTS), THE_BATCH)

    assert result.metrics.total_cases == 3
    assert result.metrics.average_score == pytest.approx((0.75 + 0.5 + 0.0) / 3)
    assert result.metrics.cases_with_score_zero == [3]


async def test_a_batch_entry_is_the_very_same_result_as_a_single_evaluation():
    """Both paths return a `CaseResult`, so there is nothing left that could disagree. The
    one type is what guarantees it — this test only keeps the two entry points honest."""
    judge = FakeJudge(BATCH_VERDICTS)

    batched = (await evaluate_batch(judge, THE_BATCH)).case_results[0]
    alone = await evaluate_case(judge, THE_CASE)

    assert batched == alone


async def test_a_dead_judge_costs_one_criterion_of_one_case_not_the_batch():
    """Same containment as for a single case, now visible in the aggregate: the run finishes,
    and `failed_criteria_count` says the average was depressed by an outage."""
    judge = FakeJudge({**BATCH_VERDICTS, 21: ConnectionError("endpoint down")})

    result = await evaluate_batch(judge, THE_BATCH)

    assert result.metrics.failed_criteria_count == 1
    assert result.case_results[1].criterion_results[0].failed is True
    assert result.case_results[0].score == pytest.approx(0.75)  # the other cases are untouched


async def test_a_bug_in_one_case_still_aborts_the_whole_batch():
    """The other side of the line: a broken program must not be reported as a run with a
    plausible-looking average — every case of it would be suspect."""
    judge = FakeJudge({**BATCH_VERDICTS, 21: RuntimeError("bound to a different event loop")})

    with pytest.raises(RuntimeError):
        await evaluate_batch(judge, THE_BATCH)


async def test_duplicate_case_ids_are_rejected():
    """Run metrics name cases by id — `cases_with_score_zero` would be ambiguous otherwise."""
    with pytest.raises(ValueError, match="case ids must be unique"):
        Batch(
            cases=[{**BATCH["cases"][0], "id": 5}, {**BATCH["cases"][1], "id": 5}]
        )


async def test_an_empty_batch_is_rejected_before_any_judge_call():
    """An empty run has no meaningful metrics, and a 422 costs nothing."""
    with pytest.raises(ValueError):
        Batch(cases=[])


class _JudgeByAnswer:
    """Scores by the answer rather than by the criterion id — the shared-rubric case needs it,
    because there the same criterion id appears in both cases."""

    async def score(self, question, answer, criterion):
        return Verdict(score=2 if "HR tool" in answer else 0, reasoning=answer)


async def test_criterion_ids_may_repeat_across_the_cases_of_a_batch():
    """Criterion ids are documented as unique *within* a case. Two cases written from the
    same rubric template share them, and their verdicts still belong to their own case."""
    shared_rubric = [{"id": 1, "content": "Submit it in the HR tool", "weight": 1}]
    batch = Batch(cases=[
        {"id": 1, "question": "q", "answer": "In the HR tool.", "criteria": shared_rubric},
        {"id": 2, "question": "q", "answer": "No idea.", "criteria": shared_rubric},
    ])

    result = await evaluate_batch(_JudgeByAnswer(), batch)

    assert [case.case_id for case in result.case_results] == [1, 2]
    assert [case.score for case in result.case_results] == [1.0, 0.0]
    assert result.metrics.cases_with_score_zero == [2]


async def test_an_empty_question_is_accepted_because_it_is_never_scored():
    """The question is context for the judge, not something the rubric holds against the
    answer — so a case without one is valid rather than a 422."""
    result = await evaluate_case(FakeJudge({1: 2, 2: 2}), Case(**{**CASE, "question": ""}))

    assert result.score == 1.0
