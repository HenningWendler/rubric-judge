import asyncio

import pytest

from rubric_eval import Batch, Case, evaluate_batch, evaluate_case
from rubric_eval.judge import JudgeConfig, OpenAIJudge, parse_verdict
from rubric_eval.prompt import JUDGE_EN
from rubric_eval.models import DEFAULT_SCALE, Criterion, Scale


def test_parses_reasoning_and_score():
    verdict = parse_verdict('The answer names the address.\n{"score": 2}')
    assert verdict.score == 2
    assert verdict.reasoning == "The answer names the address."


def test_takes_the_last_json_object_because_the_judge_reasons_first():
    verdict = parse_verdict('Not {"score": 0} but rather this.\n{"score": 1}')
    assert verdict.score == 1


def test_survives_code_fences_and_extra_keys():
    verdict = parse_verdict('Reasoning.\n```json\n{"score": 0, "confidence": 0.9}\n```')
    assert verdict.score == 0


def test_rejects_a_reply_without_json():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_verdict("I think it is fully covered.")


def test_rejects_a_score_off_the_scale():
    with pytest.raises(ValueError, match="not on the scale"):
        parse_verdict('Reasoning.\n{"score": 3}')


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.in_flight = 0
        self.peak_in_flight = 0

    async def create(self, **kwargs):
        self.calls.append(kwargs["messages"])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return await self._answer_while_counting(reply)

    async def _answer_while_counting(self, reply):
        """Yields to the event loop while the call is "in flight", so concurrent callers
        really do overlap and `peak_in_flight` measures the throttle instead of luck."""
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        return _completion(reply)


def _completion(content):
    class Response:
        choices = [type("Choice", (), {"message": type("Message", (), {"content": content})})]

    return Response


def _judge(replies, **overrides):
    config = JudgeConfig(model="m", endpoint="http://x/v1", api_key="k", **overrides)
    judge = OpenAIJudge(config)
    fake = FakeCompletions(replies)
    judge.client.chat.completions = fake
    return judge, fake


CRITERION = Criterion(id=1, content="Send an email", weight=1)


async def test_scores_a_criterion():
    judge, fake = _judge(['Covered literally.\n{"score": 2}'])
    verdict = await judge.score("How?", "Send an email.", CRITERION)
    assert verdict.score == 2
    assert len(fake.calls) == 1


async def test_retries_with_the_concrete_error_appended():
    judge, fake = _judge(["no json at all", 'Now properly.\n{"score": 1}'])
    verdict = await judge.score("How?", "Vaguely.", CRITERION)
    assert verdict.score == 1
    assert len(fake.calls) == 2
    assert fake.calls[1][-2] == {"role": "assistant", "content": "no json at all"}
    assert "no JSON object" in fake.calls[1][-1]["content"]


async def test_gives_up_after_max_attempts():
    judge, fake = _judge(["nope"] * 3, max_attempts=3)
    with pytest.raises(ValueError, match="no valid answer in 3 attempts"):
        await judge.score("How?", "Vaguely.", CRITERION)
    assert len(fake.calls) == 3


def test_rejects_a_fractional_score_off_the_integral_scale():
    with pytest.raises(ValueError, match="not on the scale"):
        parse_verdict('Reasoning.\n{"score": 1.5}')


def test_names_every_missing_environment_variable_at_once(monkeypatch):
    """One start per missing variable is misery, so the complaint lists all of them.
    The order they are listed in is incidental and deliberately not asserted."""
    required = (
        "RUBRIC_EVAL_JUDGE_ENDPOINT",
        "RUBRIC_EVAL_JUDGE_API_KEY",
        "RUBRIC_EVAL_JUDGE_MODEL",
    )
    for variable in required:
        monkeypatch.delenv(variable, raising=False)

    with pytest.raises(RuntimeError) as complaint:
        JudgeConfig.from_env()

    unnamed = [variable for variable in required if variable not in str(complaint.value)]
    assert unnamed == []


def test_unset_optional_variables_keep_the_field_defaults(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_ENDPOINT", "http://x/v1")
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_API_KEY", "k")
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_MODEL", "m")
    monkeypatch.delenv("RUBRIC_EVAL_JUDGE_TEMPERATURE", raising=False)
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_MAX_ATTEMPTS", "5")
    config = JudgeConfig.from_env()
    assert config.temperature == 0.0
    assert config.max_attempts == 5


# --- parser: replies that are almost right -------------------------------------------------


def test_rejects_a_negative_score():
    with pytest.raises(ValueError, match="not on the scale"):
        parse_verdict('Reasoning.\n{"score": -1}')


def test_accepts_a_float_that_lands_exactly_on_the_scale():
    """Models like to write 2.0 — that is the same grade, not a finer one."""
    assert parse_verdict('Reasoning.\n{"score": 2.0}').score == 2


def test_complains_concretely_when_the_score_object_is_no_valid_json():
    """A trailing comma looks like a score object to the regex but is not JSON. The judge
    has to hear *that*, not the generic "no JSON at all" complaint."""
    with pytest.raises(ValueError, match="could not be parsed"):
        parse_verdict('Reasoning.\n{"score": 2,}')


def test_a_nested_score_object_asks_for_a_flat_one_instead_of_guessing():
    """`{"score": 2, "evidence": {...}}` is beyond the flat-object regex. Degrading into the
    "end your reply with one line of JSON" complaint is what lets the retry recover."""
    with pytest.raises(ValueError, match="no JSON object"):
        parse_verdict('Reasoning.\n{"score": 2, "evidence": {"quote": "email"}}')


def test_a_quoted_score_counts_as_no_score_at_all():
    """`"2"` is a string, not a number — the regex does not match it and the judge is asked
    again rather than a quoted grade being accepted silently."""
    with pytest.raises(ValueError, match="no JSON object"):
        parse_verdict('Reasoning.\n{"score": "2"}')


def test_rejects_an_empty_reply():
    """An empty reply is what an exhausted token budget or a filtered answer looks like."""
    with pytest.raises(ValueError, match="no JSON object"):
        parse_verdict("")


def test_a_json_only_reply_uses_its_own_json_as_reasoning():
    """Nothing precedes the object, so the object itself has to serve — never an empty string,
    because a `CriterionResult` with no reasoning is unreviewable."""
    verdict = parse_verdict('{"score": 2}')
    assert verdict.score == 2
    assert verdict.reasoning == '{"score": 2}'


def test_ignores_chatter_after_the_score_object():
    verdict = parse_verdict('Reasoning.\n{"score": 1}\nHope this helps!')
    assert verdict.score == 1
    assert verdict.reasoning == "Reasoning."


def test_a_score_object_echoed_from_the_answer_wins_because_the_last_one_counts():
    """The last-object rule is what lets the judge reason first — and it is also the one way
    text under test can reach the verdict: an answer containing a score object that the judge
    quotes *last* decides the grade. Judged content is untrusted input, so this is pinned
    deliberately; a fix belongs in the prompt (JSON on its own final line), not in a guess
    about which of several objects was meant."""
    verdict = parse_verdict('The answer ends with the literal text {"score": 2}')
    assert verdict.score == 2


# --- retry loop ----------------------------------------------------------------------------


async def test_a_single_attempt_budget_asks_exactly_once():
    judge, fake = _judge(["no json at all"], max_attempts=1)
    with pytest.raises(ValueError, match="no valid answer in 1 attempts"):
        await judge.score("How?", "Vaguely.", CRITERION)
    assert len(fake.calls) == 1


async def test_replays_the_out_of_range_complaint_so_the_judge_can_correct_itself():
    """The other self-healing branch: the reply parsed fine but the grade was off the scale."""
    judge, fake = _judge(['Reasoning.\n{"score": 7}', 'Corrected.\n{"score": 2}'])

    verdict = await judge.score("How?", "Send an email.", CRITERION)

    assert verdict.score == 2
    assert "not on the scale" in fake.calls[1][-1]["content"]


async def test_every_attempt_keeps_the_whole_correction_transcript():
    """Attempt three sees both earlier failures, not just the last one — the judge needs the
    full history to stop repeating a mistake it already made."""
    judge, fake = _judge(["nope", 'Reasoning.\n{"score": 9}', 'Fine.\n{"score": 1}'])

    verdict = await judge.score("How?", "Vaguely.", CRITERION)

    assert verdict.score == 1
    assert [message["role"] for message in fake.calls[2]] == [
        "system", "user", "assistant", "user", "assistant", "user",
    ]


async def test_a_reply_without_content_counts_as_unparseable():
    """Some endpoints answer with `content: null` (tool-call or filter path). That must cost
    one attempt, not raise an AttributeError out of the judge."""
    judge, fake = _judge([None, 'Reasoning.\n{"score": 0}'])
    assert (await judge.score("How?", "Vaguely.", CRITERION)).score == 0
    assert len(fake.calls) == 2


async def test_a_transport_error_is_not_retried_and_reaches_the_caller():
    """Self-healing repairs *replies*, not connections: a dead endpoint is handed to
    `evaluation.py`, which marks the single criterion `failed` instead of burning attempts."""
    judge, fake = _judge([ConnectionError("endpoint unreachable")], max_attempts=3)

    with pytest.raises(ConnectionError, match="endpoint unreachable"):
        await judge.score("How?", "Vaguely.", CRITERION)

    assert len(fake.calls) == 1


async def test_the_judge_is_asked_with_the_configured_sampling_settings():
    """Reproducibility is a promise of the config, so it has to reach the wire."""
    judge = OpenAIJudge(
        JudgeConfig(model="m", endpoint="http://x/v1", api_key="k", temperature=0.0, max_tokens=64)
    )
    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        return _completion('Reasoning.\n{"score": 2}')

    judge.client.chat.completions = type("C", (), {"create": staticmethod(create)})
    await judge.score("How?", "Send an email.", CRITERION)

    assert sent["temperature"] == 0.0
    assert sent["max_completion_tokens"] == 64
    assert sent["model"] == "m"


# --- configuration bounds ------------------------------------------------------------------


def test_an_attempt_budget_below_one_is_rejected():
    """`max_attempts=0` would never ask the judge at all and report "no valid answer in 0
    attempts" — a rubric silently scoring 0 without a single LLM call. It comes straight from
    RUBRIC_EVAL_JUDGE_MAX_ATTEMPTS, so it has to be caught at the boundary."""
    with pytest.raises(ValueError):
        JudgeConfig(model="m", endpoint="http://x/v1", api_key="k", max_attempts=0)


def test_a_token_budget_below_one_is_rejected():
    with pytest.raises(ValueError):
        JudgeConfig(model="m", endpoint="http://x/v1", api_key="k", max_tokens=0)


def test_a_negative_temperature_is_rejected():
    with pytest.raises(ValueError):
        JudgeConfig(model="m", endpoint="http://x/v1", api_key="k", temperature=-1.0)


def test_an_empty_environment_variable_counts_as_missing(monkeypatch):
    """`export RUBRIC_EVAL_JUDGE_API_KEY=` is a typo, not a configuration — an empty key would
    otherwise reach the endpoint and fail there with an unrelated 401."""
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_ENDPOINT", "http://x/v1")
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_API_KEY", "")
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_MODEL", "m")

    with pytest.raises(RuntimeError, match="RUBRIC_EVAL_JUDGE_API_KEY"):
        JudgeConfig.from_env()


def test_a_non_numeric_environment_value_names_the_offending_setting(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_ENDPOINT", "http://x/v1")
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_API_KEY", "k")
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_MODEL", "m")
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_TEMPERATURE", "warm")

    with pytest.raises(ValueError, match="temperature"):
        JudgeConfig.from_env()


# --- throttling -----------------------------------------------------------------------------


async def test_never_puts_more_calls_in_flight_than_configured():
    """The endpoint's rate limit is the reason the judge is throttled at all: ten criteria
    asked at once must not become ten simultaneous connections."""
    judge, fake = _judge(['Covered.\n{"score": 2}'] * 10, max_concurrent=2)

    await asyncio.gather(*(judge.score("How?", "Yes.", CRITERION) for _ in range(10)))

    assert fake.peak_in_flight == 2
    assert len(fake.calls) == 10


async def test_throttling_does_not_serialize_the_calls():
    """A limit of 4 has to mean four at a time, not one after another — otherwise a rubric
    would be judged as slowly as a for-loop."""
    judge, fake = _judge(['Covered.\n{"score": 2}'] * 8, max_concurrent=4)

    await asyncio.gather(*(judge.score("How?", "Yes.", CRITERION) for _ in range(8)))

    assert fake.peak_in_flight == 4


async def test_a_single_slot_still_lets_a_criterion_retry():
    """The slot is released before the reply is parsed. Held across the retry loop instead,
    a judge limited to one call would wait for a slot it is holding itself — a deadlock."""
    judge, fake = _judge(["no json at all", 'Now properly.\n{"score": 1}'], max_concurrent=1)

    verdict = await asyncio.wait_for(judge.score("How?", "Vaguely.", CRITERION), timeout=5)

    assert verdict.score == 1
    assert fake.peak_in_flight == 1


async def test_a_slot_is_released_even_when_the_call_fails():
    """A transport error must not leak its slot, or a flaky endpoint starves the judge."""
    judge, fake = _judge(
        [ConnectionError("down"), 'Covered.\n{"score": 2}'], max_concurrent=1
    )

    with pytest.raises(ConnectionError):
        await judge.score("How?", "Yes.", CRITERION)
    verdict = await asyncio.wait_for(judge.score("How?", "Yes.", CRITERION), timeout=5)

    assert verdict.score == 2


def test_a_concurrency_limit_below_one_is_rejected():
    """`max_concurrent=0` is a semaphore no call can ever pass: it would hang, not fail."""
    with pytest.raises(ValueError):
        JudgeConfig(model="m", endpoint="http://x/v1", api_key="k", max_concurrent=0)


def test_the_concurrency_limit_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_ENDPOINT", "http://x/v1")
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_API_KEY", "k")
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_MODEL", "m")
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_MAX_CONCURRENT", "16")

    assert JudgeConfig.from_env().max_concurrent == 16


def test_the_concurrency_limit_defaults_to_eight():
    assert JudgeConfig(model="m", endpoint="http://x/v1", api_key="k").max_concurrent == 8


def test_a_judge_can_be_reused_from_a_second_event_loop():
    """`api.get_judge` caches one judge for the whole process, so it outlives any single
    event loop — two `asyncio.run()` calls, a notebook cell run twice, a test suite with two
    clients. A semaphore bound to the first loop refuses to be awaited from the next one, and
    it binds only on the *queueing* path, so the breakage would start exactly when a rubric
    outgrows `max_concurrent`. Not an async test on purpose: it needs its own two loops.
    """
    judge, fake = _judge(['Covered.\n{"score": 2}'] * 6, max_concurrent=2)

    async def score_three_at_once():
        return await asyncio.gather(*(judge.score("How?", "Yes.", CRITERION) for _ in range(3)))

    verdicts = asyncio.run(score_three_at_once()) + asyncio.run(score_three_at_once())

    assert [verdict.score for verdict in verdicts] == [2] * 6
    assert fake.peak_in_flight == 2  # the second loop is throttled too, not just unbroken


def test_the_throttle_of_a_finished_loop_is_not_kept_forever():
    """One entry per running loop, not per loop that ever ran. Reaching into the private dict
    on purpose: retained memory has no public surface to assert on, and the tempting fix — a
    `WeakKeyDictionary` — silently does nothing here, because a semaphore holds a strong
    reference to the loop it bound itself to and would keep its own key alive.
    """
    judge, _ = _judge(['Covered.\n{"score": 2}'] * 9, max_concurrent=2)

    async def score_three_at_once():
        return await asyncio.gather(*(judge.score("How?", "Yes.", CRITERION) for _ in range(3)))

    for _ in range(3):
        asyncio.run(score_three_at_once())

    assert len(judge._slots_per_loop) == 1


async def test_the_limit_holds_across_several_cases_judged_at_once():
    """The guarantee a library caller gets: one judge, one event loop, any number of cases in
    parallel — the peak is the configured limit, not the limit *per case*.

    Which is why a judge is meant to be built once and shared (`api.get_judge` caches one for
    the HTTP path): the budget belongs to the judge instance, so a judge per case would hand
    each case its own eight slots.
    """
    judge, fake = _judge(['Covered.\n{"score": 2}'] * 30, max_concurrent=8)
    five_cases = [
        Case(
            id=first,
            question="How do I report sick leave?",
            answer="Email hr@example.com before 10:00.",
            criteria=[
                {"id": number, "content": f"criterion {number}", "weight": 1}
                for number in range(first * 6, first * 6 + 6)
            ],
        )
        for first in range(5)
    ]

    await asyncio.gather(*(evaluate_case(judge, case) for case in five_cases))

    assert len(fake.calls) == 30
    assert fake.peak_in_flight == 8


def _batch_of(case_count: int, criteria_per_case: int) -> Batch:
    return Batch(
        cases=[
            {
                "id": case_number,
                "question": "How do I report sick leave?",
                "answer": "Email hr@example.com before 10:00.",
                "criteria": [
                    {"id": number, "content": f"criterion {number}", "weight": 1}
                    for number in range(criteria_per_case)
                ],
            }
            for case_number in range(case_count)
        ]
    )


async def test_a_batch_does_not_multiply_the_concurrency_limit_by_its_cases():
    """The guardrail the batch could plausibly break: `evaluate_batch` fans out over the cases
    *and* every case fans out over its criteria, so a naive implementation would put
    cases x criteria calls in flight. The budget belongs to the judge, so it stays the same
    eight whether 40 criteria come from one case or from five.
    """
    judge, fake = _judge(['Covered.\n{"score": 2}'] * 40, max_concurrent=8)

    result = await evaluate_batch(judge, _batch_of(case_count=5, criteria_per_case=8))

    assert len(fake.calls) == 40
    assert fake.peak_in_flight == 8
    assert result.metrics.average_score == 1.0


async def test_two_batches_running_at_once_share_the_budget():
    """Two HTTP requests, one shared judge: the second batch must queue on the same slots,
    not open its own eight connections."""
    judge, fake = _judge(['Covered.\n{"score": 2}'] * 40, max_concurrent=8)
    two_batches = [_batch_of(case_count=2, criteria_per_case=10) for _ in range(2)]

    await asyncio.gather(*(evaluate_batch(judge, batch) for batch in two_batches))

    assert len(fake.calls) == 40
    assert fake.peak_in_flight == 8


async def test_a_batch_is_judged_concurrently_rather_than_case_after_case():
    """A limit of 8 with 4 criteria per case only pays off if the cases overlap — awaiting
    them one after another would cap the peak at 4 and make a run as slow as a for-loop."""
    judge, fake = _judge(['Covered.\n{"score": 2}'] * 16, max_concurrent=8)

    await evaluate_batch(judge, _batch_of(case_count=4, criteria_per_case=4))

    assert fake.peak_in_flight == 8


# --- what the judge is and is not told ------------------------------------------------------


async def test_the_criterion_weight_never_reaches_the_judge():
    """Documented as deliberate: a judge that knew how much a criterion counts could let that
    importance leak into the score. Only `content` is sent, never the weight."""
    judge, fake = _judge(['Covered.\n{"score": 2}'])
    weighted = Criterion(id=1, content="Send an email", weight=7)

    await judge.score("How?", "Send an email.", weighted)

    sent = " ".join(message["content"] for message in fake.calls[0])
    assert "Send an email" in sent
    assert "7" not in sent


async def test_a_custom_prompt_replaces_the_system_message():
    """`OpenAIJudge(config, prompt=...)` is the documented way to bring your own wording. It
    replaces the system message only — the per-criterion user message stays the bundled one."""
    config = JudgeConfig(model="m", endpoint="http://x/v1", api_key="k")
    judge = OpenAIJudge(config, prompt="Judge in Klingon.")
    judge.client.chat.completions = FakeCompletions(['Covered.\n{"score": 2}'])

    await judge.score("How?", "Send an email.", CRITERION)

    system, user = judge.client.chat.completions.calls[0]
    assert system == {"role": "system", "content": "Judge in Klingon."}
    assert "Send an email" in user["content"]


def test_an_empty_optional_environment_variable_falls_back_to_the_default(monkeypatch):
    """An empty value counts as missing for the required variables, so it has to mean the
    same for the optional ones — an exported but unset `MAX_CONCURRENT=` must not be read as
    a limit of zero, which would be rejected and take the process down at startup."""
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_ENDPOINT", "http://x/v1")
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_API_KEY", "k")
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_MODEL", "m")
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_MAX_CONCURRENT", "")
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_TEMPERATURE", "")

    config = JudgeConfig.from_env()

    assert config.max_concurrent == 8
    assert config.temperature == 0.0


# --- the scale the judge declares ----------------------------------------------------------


TEN_POINT = Scale(maximum=10, presence_threshold=5)


def test_the_parser_accepts_what_the_given_scale_allows():
    assert parse_verdict('Reasoning.\n{"score": 7}', TEN_POINT).score == 7


def test_the_parser_still_refuses_what_that_scale_does_not_reach():
    with pytest.raises(ValueError, match="not on the scale"):
        parse_verdict('Reasoning.\n{"score": 11}', TEN_POINT)


def test_a_bigger_scale_is_still_integral():
    """More levels is a finer grid, not a continuous one — the judge picks a level."""
    with pytest.raises(ValueError, match="not on the scale"):
        parse_verdict('Reasoning.\n{"score": 7.5}', TEN_POINT)


def test_the_complaints_quote_the_scale_that_was_asked_for():
    """A judge told "answer with 0, 1 or 2" while working on a ten-point scale would correct
    itself into a wrong grade, so the hint is derived from the scale, never from a constant."""
    with pytest.raises(ValueError, match=r"0, 1, 2, 3, 4, 5, 6, 7, 8, 9 or 10"):
        parse_verdict('Reasoning.\n{"score": 11}', TEN_POINT)
    with pytest.raises(ValueError, match=r"0, 1, 2, 3, 4, 5, 6, 7, 8, 9 or 10"):
        parse_verdict("no json here", TEN_POINT)


def test_the_malformed_json_complaint_offers_the_scales_own_grades_too():
    """The third complaint is the one a judge sees when it wrote a score object at all, so it
    is the likeliest to be corrected into a grade — on the scale it was asked for, not 0-2."""
    with pytest.raises(ValueError, match=r"0, 1, 2, 3, 4, 5, 6, 7, 8, 9 or 10"):
        parse_verdict('Reasoning.\n{"score": 7,}', TEN_POINT)


def test_a_binary_scale_is_offered_as_two_choices_not_as_a_list_of_one():
    with pytest.raises(ValueError, match=r"using 0 or 1"):
        parse_verdict("no json here", Scale(maximum=1, presence_threshold=1))


def test_the_parser_reads_the_default_scale_when_none_is_named():
    """The bundled prompt describes exactly that scale, so it is the only safe default."""
    assert parse_verdict('Reasoning.\n{"score": 2}').score == 2
    with pytest.raises(ValueError, match="not on the scale"):
        parse_verdict('Reasoning.\n{"score": 3}')


def test_a_judge_on_the_default_scale_needs_no_prompt_of_its_own():
    judge, _ = _judge([])
    assert judge.scale == DEFAULT_SCALE


def test_a_scale_that_describes_no_levels_needs_a_prompt_of_its_own():
    """Nothing to instruct the model with: the bundled prompt spells 0-2 out in prose and in
    its examples, and keeping it while the parser checks against 0..10 would fail every
    reply, one wasted call at a time."""
    config = JudgeConfig(model="m", endpoint="http://x/v1", api_key="k")
    with pytest.raises(ValueError, match="describes no levels"):
        OpenAIJudge(config, scale=TEN_POINT)


def test_a_scale_that_describes_its_levels_needs_no_prompt_at_all():
    """The whole point of the descriptions: a custom scale is five lines, not a rewritten
    prompt."""
    config = JudgeConfig(model="m", endpoint="http://x/v1", api_key="k")
    described = Scale(
        maximum=10,
        presence_threshold=5,
        level_descriptions={grade: f"Level {grade} of ten." for grade in range(11)},
    )
    judge = OpenAIJudge(config, scale=described)

    assert judge.scale == described
    assert "Use this 0-10 scale:" in judge.system_prompt
    assert "Example:" not in judge.system_prompt


def test_a_custom_scale_with_a_custom_prompt_is_the_supported_way():
    config = JudgeConfig(model="m", endpoint="http://x/v1", api_key="k")
    judge = OpenAIJudge(config, prompt="Grade 0 to 10.", scale=TEN_POINT)
    assert judge.scale == TEN_POINT
    assert judge.system_prompt == "Grade 0 to 10."


def test_a_rebuilt_default_scale_still_gets_the_bundled_prompt_with_its_examples():
    """Scales compare by value, so an equal scale built by hand is the default one — and the
    worked examples, which belong to that scale alone, come with it."""
    config = JudgeConfig(model="m", endpoint="http://x/v1", api_key="k")
    rebuilt = Scale(
        maximum=2,
        presence_threshold=0.5,
        level_descriptions=dict(DEFAULT_SCALE.level_descriptions),
    )
    assert OpenAIJudge(config, scale=rebuilt).system_prompt == JUDGE_EN


def test_a_custom_prompt_on_the_default_scale_stays_allowed():
    config = JudgeConfig(model="m", endpoint="http://x/v1", api_key="k")
    assert OpenAIJudge(config, prompt="Grade it.").system_prompt == "Grade it."


async def test_the_judge_scores_and_self_heals_on_its_own_scale():
    config = JudgeConfig(model="m", endpoint="http://x/v1", api_key="k")
    judge = OpenAIJudge(config, prompt="Grade 0 to 10.", scale=TEN_POINT)
    fake = FakeCompletions(['Reasoning.\n{"score": 12}', 'Corrected.\n{"score": 8}'])
    judge.client.chat.completions = fake

    verdict = await judge.score("How?", "Send an email.", CRITERION)

    assert verdict.score == 8
    assert "0, 1, 2, 3, 4, 5, 6, 7, 8, 9 or 10" in fake.calls[1][-1]["content"]
