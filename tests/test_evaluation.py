"""The evaluation layer on its own: fan-out, failure policy and the weighted fold — no HTTP."""

import asyncio

import pytest

from conftest import CASE, FakeJudge

from rubric_eval import EvaluateRequest, evaluate_case

REQUEST = EvaluateRequest(**CASE)


async def test_weighted_score_and_reasoning():
    result = await evaluate_case(FakeJudge({1: 2, 2: 0}), REQUEST)

    assert result.score == pytest.approx(0.75)  # (3*2/2 + 1*0/2) / 4
    assert [criterion.score for criterion in result.criteria] == [2.0, 0.0]
    assert [criterion.is_present for criterion in result.criteria] == [True, False]
    assert result.criteria[0].reasoning == "reasoning for 1"
    assert result.criteria[0].failed is False


async def test_a_failed_criterion_scores_zero_and_stays_in_the_denominator():
    result = await evaluate_case(FakeJudge({1: 2, 2: ValueError("judge down")}), REQUEST)

    assert result.score == pytest.approx(0.75)  # same score: a 0 and a failed 0 weigh alike
    assert result.criteria[1].failed is True
    assert "judge down" in result.criteria[1].reasoning


async def test_verdicts_keep_the_order_of_the_rubric():
    """`asyncio.gather` preserves argument order — results can be zipped with the criteria."""
    result = await evaluate_case(FakeJudge({1: 0, 2: 2}), REQUEST)

    assert [criterion.criterion_id for criterion in result.criteria] == [1, 2]


async def test_every_criterion_failing_still_yields_a_complete_result():
    """A total outage must not raise: the caller gets a 0.0 and sees why, per criterion."""
    outage = FakeJudge({1: ConnectionError("endpoint down"), 2: ConnectionError("endpoint down")})

    result = await evaluate_case(outage, REQUEST)

    assert result.score == 0.0
    assert [criterion.failed for criterion in result.criteria] == [True, True]


async def test_a_failure_without_a_message_still_names_its_cause():
    """`str(TimeoutError())` is the empty string, so the plain message alone loses the reason
    entirely — and a timeout is the most likely judge failure of all."""
    result = await evaluate_case(FakeJudge({1: 2, 2: TimeoutError()}), REQUEST)

    assert result.criteria[1].failed is True
    assert "TimeoutError" in result.criteria[1].reasoning


async def test_an_empty_answer_is_judged_rather_than_rejected():
    """A system under test that returned nothing is a valid case scoring 0 — not a 422.
    The judge is still asked, because "nothing" can only be graded against the rubric."""
    empty = EvaluateRequest(**{**CASE, "answer": ""})

    result = await evaluate_case(FakeJudge({1: 0, 2: 0}), empty)

    assert result.score == 0.0


async def test_duplicate_criterion_ids_are_rejected():
    """`CriterionResult.criterion_id` is documented as the way to match results to the rubric
    without relying on list order. Two criteria sharing an id make that impossible — and a
    caller merging by id would drop or double-count a verdict."""
    with pytest.raises(ValueError):
        EvaluateRequest(
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
    many = EvaluateRequest(
        question="q",
        answer="a",
        criteria=[{"id": i, "content": f"criterion {i}", "weight": 1} for i in range(200)],
    )

    result = await evaluate_case(FakeJudge({i: 2 for i in range(200)}), many)

    assert [criterion.criterion_id for criterion in result.criteria] == list(range(200))
    assert result.score == 1.0


async def test_cancellation_is_not_swallowed_by_the_failure_containment():
    """A cancelled request (client disconnect, shutdown) must abort, not come back as a
    plausible-looking 0.0 with every criterion marked failed."""

    class CancellingJudge:
        async def score(self, question, answer, criterion):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await evaluate_case(CancellingJudge(), REQUEST)


async def test_a_bug_is_re_raised_instead_of_being_scored_as_a_failed_criterion():
    """The containment exists for a judge that cannot answer, not for a broken program. A
    `RuntimeError` — asyncio misuse, for instance — must reach the caller: scoring it 0 would
    hand back a number indistinguishable from a real result."""
    with pytest.raises(RuntimeError, match="Semaphore is bound to a different event loop"):
        await evaluate_case(
            FakeJudge({1: RuntimeError("Semaphore is bound to a different event loop"), 2: 2}),
            REQUEST,
        )


async def test_a_dead_endpoint_is_still_only_one_criterion():
    """The other side of that line: everything an endpoint can do to you stays contained."""
    result = await evaluate_case(FakeJudge({1: ConnectionError("refused"), 2: 2}), REQUEST)

    assert result.criteria[0].failed is True
    assert result.criteria[1].score == 2.0


async def test_a_custom_judges_own_error_type_is_still_contained():
    """`Judge` is a Protocol: an implementation may raise whatever it likes for an outage,
    and only the bug-shaped built-ins are excluded from the containment."""

    class QuotaExceeded(Exception):
        pass

    result = await evaluate_case(FakeJudge({1: QuotaExceeded("no credit left"), 2: 2}), REQUEST)

    assert result.criteria[0].failed is True
    assert "no credit left" in result.criteria[0].reasoning
