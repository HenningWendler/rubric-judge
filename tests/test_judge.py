import asyncio
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from rubric_eval import Case, Run, evaluate_case, evaluate_run
from rubric_eval.judge import (
    JudgeConfig,
    JudgeUnavailableError,
    OpenAIJudge,
    UnusableReplyError,
    parse_judge_reply,
)
from rubric_eval.prompt import JUDGE_EN
from rubric_eval.models import DEFAULT_SCALE, Criterion, Scale

_REQUEST = httpx.Request("POST", "http://x/v1/chat/completions")
"""The request every faked transport failure claims to have been raised for — the openai
exceptions carry one, and none of the code under test reads it."""


def rate_limited(message: str = "rate limited") -> RateLimitError:
    """A real `openai.RateLimitError`, because the judge decides what to retry by the SDK's
    own exception types and a stand-in would prove nothing about that."""
    return RateLimitError(message, response=httpx.Response(429, request=_REQUEST), body=None)


def server_error(message: str = "bad gateway") -> InternalServerError:
    """The other retryable status family, built the same way."""
    return InternalServerError(message, response=httpx.Response(502, request=_REQUEST), body=None)


def test_parses_reasoning_and_score():
    judge_reply = parse_judge_reply('The answer names the address.\n{"score": 2}', DEFAULT_SCALE)
    assert judge_reply.score == 2
    assert judge_reply.reasoning == "The answer names the address."


def test_takes_the_last_json_object_because_the_judge_reasons_first():
    judge_reply = parse_judge_reply(
        'Not {"score": 0} but rather this.\n{"score": 1}', DEFAULT_SCALE
    )
    assert judge_reply.score == 1


def test_survives_code_fences_and_extra_keys():
    judge_reply = parse_judge_reply(
        'Reasoning.\n```json\n{"score": 0, "confidence": 0.9}\n```', DEFAULT_SCALE
    )
    assert judge_reply.score == 0


def test_rejects_a_reply_without_json():
    with pytest.raises(UnusableReplyError, match="no JSON object"):
        parse_judge_reply("I think it is fully covered.", DEFAULT_SCALE)


def test_rejects_a_score_off_the_scale():
    with pytest.raises(ValueError, match="not on the scale"):
        parse_judge_reply('Reasoning.\n{"score": 3}', DEFAULT_SCALE)


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


def _completion(content, finish_reason="stop"):
    """A chat completion shaped like the SDK's, down to the `finish_reason` the judge quotes
    when an endpoint answers with no content at all."""

    class Response:
        choices = [
            type(
                "Choice",
                (),
                {
                    "message": type("Message", (), {"content": content}),
                    "finish_reason": finish_reason,
                },
            )
        ]

    return Response


@pytest.fixture(autouse=True)
def instant_backoff(monkeypatch):
    """No real waiting between retries. Reaching for the private constant on purpose: the
    backoff has no public surface, and a suite that really slept would only be slower."""
    monkeypatch.setattr("rubric_eval.judge._FIRST_BACKOFF_SECONDS", 0)


def _client_answering(completions):
    """The stand-in `OpenAIJudge(client=...)` is given: the judge only ever reaches for
    `client.chat.completions.create`, so that is the whole client a test has to supply — no
    SDK object to build and no socket to open behind it."""
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _judge(replies, **overrides):
    config = JudgeConfig(model="m", endpoint="http://x/v1", api_key="k", **overrides)
    fake = FakeCompletions(replies)
    return OpenAIJudge(config, client=_client_answering(fake)), fake


def _judge_answering(response, **overrides):
    """A judge whose endpoint always hands back one prepared response object — for the
    answers that are not reply text at all: no content, or not even a choice to read it
    from."""

    async def create(**kwargs):
        return response

    config = JudgeConfig(model="m", endpoint="http://x/v1", api_key="k", **overrides)
    answering = type("C", (), {"create": staticmethod(create)})
    return OpenAIJudge(config, client=_client_answering(answering))


CRITERION = Criterion(id=1, content="Send an email", weight=1)


async def test_scores_a_criterion():
    judge, fake = _judge(['Covered literally.\n{"score": 2}'])
    judge_reply = await judge.score("How?", "Send an email.", CRITERION)
    assert judge_reply.score == 2
    assert len(fake.calls) == 1


async def test_retries_with_the_concrete_error_appended():
    judge, fake = _judge(["no json at all", 'Now properly.\n{"score": 1}'])
    judge_reply = await judge.score("How?", "Vaguely.", CRITERION)
    assert judge_reply.score == 1
    assert len(fake.calls) == 2
    assert fake.calls[1][-2] == {"role": "assistant", "content": "no json at all"}
    assert "no JSON object" in fake.calls[1][-1]["content"]


async def test_gives_up_after_max_attempts():
    """Three unusable replies are a judge that cannot answer, and the run dies with it —
    `JudgeUnavailableError` rather than a plain `ValueError`, so the HTTP layer can tell this
    apart from a bug and answer 503."""
    judge, fake = _judge(["nope"] * 3, max_attempts=3)
    with pytest.raises(JudgeUnavailableError, match="no usable answer in 3 attempts"):
        await judge.score("How?", "Vaguely.", CRITERION)
    assert len(fake.calls) == 3


def test_rejects_a_fractional_score_off_the_integral_scale():
    with pytest.raises(ValueError, match="not on the scale"):
        parse_judge_reply('Reasoning.\n{"score": 1.5}', DEFAULT_SCALE)


REQUIRED_ENVIRONMENT = {
    "RUBRIC_EVAL_JUDGE_ENDPOINT": "http://x/v1",
    "RUBRIC_EVAL_JUDGE_API_KEY": "k",
    "RUBRIC_EVAL_JUDGE_MODEL": "m",
}
"""The three variables a config cannot be built without. Every policy test starts from these
and breaks or adds exactly the one variable it is about — `from_mapping` takes the whole
environment as an argument, so none of them touches the process's own."""


def test_names_every_missing_environment_variable_at_once():
    """One start per missing variable is misery, so the complaint lists all of them.
    The order they are listed in is incidental and deliberately not asserted."""
    with pytest.raises(RuntimeError) as complaint:
        JudgeConfig.from_mapping({})

    unnamed = [name for name in REQUIRED_ENVIRONMENT if name not in str(complaint.value)]
    assert unnamed == []


def test_unset_optional_variables_keep_the_field_defaults():
    config = JudgeConfig.from_mapping(
        {**REQUIRED_ENVIRONMENT, "RUBRIC_EVAL_JUDGE_MAX_ATTEMPTS": "5"}
    )
    assert config.temperature == 0.0
    assert config.max_attempts == 5


# --- parser: replies that are almost right -------------------------------------------------


def test_rejects_a_negative_score():
    with pytest.raises(ValueError, match="not on the scale"):
        parse_judge_reply('Reasoning.\n{"score": -1}', DEFAULT_SCALE)


def test_accepts_a_float_that_lands_exactly_on_the_scale():
    """Models like to write 2.0 — that is the same grade, not a finer one."""
    assert parse_judge_reply('Reasoning.\n{"score": 2.0}', DEFAULT_SCALE).score == 2


def test_complains_concretely_when_the_score_object_is_no_valid_json():
    """A trailing comma looks like a score object to the regex but is not JSON. The judge
    has to hear *that*, not the generic "no JSON at all" complaint."""
    with pytest.raises(ValueError, match="could not be parsed"):
        parse_judge_reply('Reasoning.\n{"score": 2,}', DEFAULT_SCALE)


def test_a_nested_score_object_asks_for_a_flat_one_instead_of_guessing():
    """`{"score": 2, "evidence": {...}}` is beyond the flat-object regex. Degrading into the
    "end your reply with one line of JSON" complaint is what lets the retry recover."""
    with pytest.raises(ValueError, match="no JSON object"):
        parse_judge_reply(
            'Reasoning.\n{"score": 2, "evidence": {"quote": "email"}}', DEFAULT_SCALE
        )


def test_a_quoted_score_counts_as_no_score_at_all():
    """`"2"` is a string, not a number — the regex does not match it and the judge is asked
    again rather than a quoted grade being accepted silently."""
    with pytest.raises(ValueError, match="no JSON object"):
        parse_judge_reply('Reasoning.\n{"score": "2"}', DEFAULT_SCALE)


def test_rejects_an_empty_reply():
    """An empty reply is what an exhausted token budget or a filtered answer looks like."""
    with pytest.raises(ValueError, match="no JSON object"):
        parse_judge_reply("", DEFAULT_SCALE)


def test_a_json_only_reply_uses_its_own_json_as_reasoning():
    """Nothing precedes the object, so the object itself has to serve — never an empty string,
    because a `CriterionResult` with no reasoning is unreviewable."""
    judge_reply = parse_judge_reply('{"score": 2}', DEFAULT_SCALE)
    assert judge_reply.score == 2
    assert judge_reply.reasoning == '{"score": 2}'


def test_ignores_chatter_after_the_score_object():
    judge_reply = parse_judge_reply('Reasoning.\n{"score": 1}\nHope this helps!', DEFAULT_SCALE)
    assert judge_reply.score == 1
    assert judge_reply.reasoning == "Reasoning."


def test_a_score_object_echoed_from_the_answer_wins_because_the_last_one_counts():
    """The last-object rule is what lets the judge reason first — and it is also the one way
    text under test can reach the grade: an answer containing a score object that the judge
    quotes *last* decides the grade. Judged content is untrusted input, so this is pinned
    deliberately; a fix belongs in the prompt (JSON on its own final line), not in a guess
    about which of several objects was meant."""
    judge_reply = parse_judge_reply(
        'The answer ends with the literal text {"score": 2}', DEFAULT_SCALE
    )
    assert judge_reply.score == 2


# --- retry loop ----------------------------------------------------------------------------


async def test_a_single_attempt_budget_asks_exactly_once():
    judge, fake = _judge(["no json at all"], max_attempts=1)
    with pytest.raises(JudgeUnavailableError, match="no usable answer in 1 attempts"):
        await judge.score("How?", "Vaguely.", CRITERION)
    assert len(fake.calls) == 1


async def test_replays_the_out_of_range_complaint_so_the_judge_can_correct_itself():
    """The other self-healing branch: the reply parsed fine but the grade was off the scale."""
    judge, fake = _judge(['Reasoning.\n{"score": 7}', 'Corrected.\n{"score": 2}'])

    judge_reply = await judge.score("How?", "Send an email.", CRITERION)

    assert judge_reply.score == 2
    assert "not on the scale" in fake.calls[1][-1]["content"]


async def test_every_attempt_keeps_the_whole_correction_transcript():
    """Attempt three sees both earlier failures, not just the last one — the judge needs the
    full history to stop repeating a mistake it already made."""
    judge, fake = _judge(["nope", 'Reasoning.\n{"score": 9}', 'Fine.\n{"score": 1}'])

    judge_reply = await judge.score("How?", "Vaguely.", CRITERION)

    assert judge_reply.score == 1
    assert [message["role"] for message in fake.calls[2]] == [
        "system", "user", "assistant", "user", "assistant", "user",
    ]


async def test_a_reply_without_content_names_the_finish_reason_instead_of_blaming_the_model():
    """Some endpoints answer with `content: null` — a truncated reply, a content filter, a
    tool-call path. Read as an empty string it fails to parse, and the judge is then told
    "your reply contained no JSON object": a complaint about something the model never wrote,
    and a retry that cannot possibly heal it. The `finish_reason` is what says which of the
    three it was, so it is in the message."""
    judge, fake = _judge([None, 'Reasoning.\n{"score": 0}'])

    with pytest.raises(JudgeUnavailableError, match="no content, finish_reason 'stop'"):
        await judge.score("How?", "Vaguely.", CRITERION)

    assert len(fake.calls) == 1  # not reprompted: no wording of the question would fix it


async def test_a_truncated_reply_says_so():
    """The likeliest cause of an empty reply is a token budget too small for the argument
    *and* the closing JSON — and `finish_reason` is the only thing that says so."""
    judge = _judge_answering(_completion(None, finish_reason="length"))

    with pytest.raises(JudgeUnavailableError, match="finish_reason 'length'"):
        await judge.score("How?", "Vaguely.", CRITERION)


async def test_a_reply_without_a_single_choice_is_the_endpoints_fault_not_a_bug():
    """An empty `choices` list used to be an `IndexError` — indistinguishable from a
    programming error, and reported as one. It is an endpoint answering with nothing."""
    judge = _judge_answering(type("Response", (), {"choices": []}))

    with pytest.raises(JudgeUnavailableError, match="without a single choice"):
        await judge.score("How?", "Vaguely.", CRITERION)


async def test_a_rate_limit_is_waited_out_rather_than_thrown_away():
    """Under the invalidate-the-run policy a single 429 would otherwise cost a whole
    catalog's worth of judging. There is nothing to correct in the conversation, so the
    criterion is simply asked again."""
    judge, fake = _judge([rate_limited(), 'Covered.\n{"score": 2}'], max_attempts=3)

    judge_reply = await judge.score("How?", "Send an email.", CRITERION)

    assert judge_reply.score == 2
    assert len(fake.calls) == 2
    assert fake.calls[1] == fake.calls[0]  # asked again, not corrected


async def test_a_timeout_and_a_broken_gateway_are_retried_too():
    """The three retryable shapes of "the endpoint is having a bad day", by the SDK's own
    types: unreachable, rate limited, 5xx."""
    judge, fake = _judge(
        [APITimeoutError(request=_REQUEST), server_error(), 'Covered.\n{"score": 2}'],
        max_attempts=3,
    )

    assert (await judge.score("How?", "Yes.", CRITERION)).score == 2
    assert len(fake.calls) == 3


def test_the_wait_after_a_transport_failure_doubles_and_stops_at_the_last_attempt(monkeypatch):
    """Backing off is the whole point of retrying a rate limit: asking again immediately is
    what got throttled in the first place. Read off the private schedule rather than from the
    clock — a test that really waited would be slow and still prove nothing exactly."""
    monkeypatch.setattr("rubric_eval.judge._FIRST_BACKOFF_SECONDS", 0.5)
    judge, _ = _judge([], max_attempts=3)

    assert [judge._backoff_seconds(failed) for failed in range(3)] == [0.5, 1.0, 0.0]


async def test_an_endpoint_that_stays_down_invalidates_the_run():
    """The attempts are spent and the judge has no usable reply: that is
    `JudgeUnavailableError`, with the transport failure chained so the cause stays readable."""
    judge, fake = _judge([rate_limited("slow down")] * 3, max_attempts=3)

    with pytest.raises(JudgeUnavailableError, match="slow down") as given_up:
        await judge.score("How?", "Vaguely.", CRITERION)

    assert len(fake.calls) == 3
    assert isinstance(given_up.value.__cause__, RateLimitError)


async def test_a_failure_without_a_message_still_names_its_cause():
    """`str(APIConnectionError(...))` can be empty, and a connection that never came up is
    the likeliest judge failure of all — then the class name is the only cause to report."""
    judge, _ = _judge([_ConnectionErrorWithoutMessage(request=_REQUEST)], max_attempts=1)

    with pytest.raises(JudgeUnavailableError, match="_ConnectionErrorWithoutMessage"):
        await judge.score("How?", "Vaguely.", CRITERION)


class _ConnectionErrorWithoutMessage(APIConnectionError):
    """An SDK transport failure whose `str()` is empty — the SDK's own always has a message,
    a custom `http_client` raising through it need not."""

    def __init__(self, *, request):
        super().__init__(message="", request=request)


async def test_an_unretryable_endpoint_error_is_not_asked_again():
    """A rejected key, an unknown model, a malformed request: repeating those only delays the
    report. They travel up as themselves, and the HTTP layer answers 500 — a configuration
    fault is not "try again later"."""

    class AuthenticationFailed(Exception):
        pass

    judge, fake = _judge([AuthenticationFailed("invalid api key")], max_attempts=3)

    with pytest.raises(AuthenticationFailed, match="invalid api key"):
        await judge.score("How?", "Vaguely.", CRITERION)

    assert len(fake.calls) == 1


async def test_the_judge_is_asked_with_the_configured_sampling_settings():
    """Reproducibility is a promise of the config, so it has to reach the wire."""
    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        return _completion('Reasoning.\n{"score": 2}')

    judge = OpenAIJudge(
        JudgeConfig(
            model="m", endpoint="http://x/v1", api_key="k", temperature=0.0, max_tokens=64
        ),
        client=_client_answering(type("C", (), {"create": staticmethod(create)})),
    )
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


def test_an_empty_required_environment_variable_is_refused():
    """`export RUBRIC_EVAL_JUDGE_API_KEY=` is a typo, not a configuration — an empty key would
    otherwise reach the endpoint and fail there with an unrelated 401."""
    with pytest.raises(RuntimeError, match="RUBRIC_EVAL_JUDGE_API_KEY"):
        JudgeConfig.from_mapping({**REQUIRED_ENVIRONMENT, "RUBRIC_EVAL_JUDGE_API_KEY": ""})


def test_a_non_numeric_environment_value_names_the_offending_setting():
    with pytest.raises(ValueError, match="temperature"):
        JudgeConfig.from_mapping(
            {**REQUIRED_ENVIRONMENT, "RUBRIC_EVAL_JUDGE_TEMPERATURE": "warm"}
        )


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

    judge_reply = await asyncio.wait_for(judge.score("How?", "Vaguely.", CRITERION), timeout=5)

    assert judge_reply.score == 1
    assert fake.peak_in_flight == 1


async def test_a_slot_is_released_even_when_the_call_fails():
    """A transport failure must not leak its slot, or a flaky endpoint starves the judge —
    which with one slot and a retry would be a deadlock against itself."""
    judge, fake = _judge(
        [rate_limited(), 'Covered.\n{"score": 2}'], max_concurrent=1, max_attempts=2
    )

    judge_reply = await asyncio.wait_for(judge.score("How?", "Yes.", CRITERION), timeout=5)

    assert judge_reply.score == 2
    assert fake.peak_in_flight == 1


def test_a_concurrency_limit_below_one_is_rejected():
    """`max_concurrent=0` is a semaphore no call can ever pass: it would hang, not fail."""
    with pytest.raises(ValueError):
        JudgeConfig(model="m", endpoint="http://x/v1", api_key="k", max_concurrent=0)


def test_the_concurrency_limit_is_read_from_the_environment(monkeypatch):
    """The one test that goes through `from_env` and the real process environment: what the
    rules are is `from_mapping`'s business and is tested on plain dicts, but that the wrapper
    actually hands it `os.environ` is only visible here."""
    exported = {**REQUIRED_ENVIRONMENT, "RUBRIC_EVAL_JUDGE_MAX_CONCURRENT": "16"}
    for variable, value in exported.items():
        monkeypatch.setenv(variable, value)

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

    judge_replies = asyncio.run(score_three_at_once()) + asyncio.run(score_three_at_once())

    assert [reply.score for reply in judge_replies] == [2] * 6
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


def _run_of(case_count: int, criteria_per_case: int) -> Run:
    return Run(
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


async def test_a_run_does_not_multiply_the_concurrency_limit_by_its_cases():
    """The guardrail the run could plausibly break: `evaluate_run` fans out over the cases
    *and* every case fans out over its criteria, so a naive implementation would put
    cases x criteria calls in flight. The budget belongs to the judge, so it stays the same
    eight whether 40 criteria come from one case or from five.
    """
    judge, fake = _judge(['Covered.\n{"score": 2}'] * 40, max_concurrent=8)

    result = await evaluate_run(judge, _run_of(case_count=5, criteria_per_case=8))

    assert len(fake.calls) == 40
    assert fake.peak_in_flight == 8
    assert result.metrics.average_score == 1.0


async def test_two_runs_running_at_once_share_the_budget():
    """Two HTTP requests, one shared judge: the second run must queue on the same slots,
    not open its own eight connections."""
    judge, fake = _judge(['Covered.\n{"score": 2}'] * 40, max_concurrent=8)
    two_runs = [_run_of(case_count=2, criteria_per_case=10) for _ in range(2)]

    await asyncio.gather(*(evaluate_run(judge, run) for run in two_runs))

    assert len(fake.calls) == 40
    assert fake.peak_in_flight == 8


async def test_a_run_is_judged_concurrently_rather_than_case_after_case():
    """A limit of 8 with 4 criteria per case only pays off if the cases overlap — awaiting
    them one after another would cap the peak at 4 and make a run as slow as a for-loop."""
    judge, fake = _judge(['Covered.\n{"score": 2}'] * 16, max_concurrent=8)

    await evaluate_run(judge, _run_of(case_count=4, criteria_per_case=4))

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
    """`OpenAIJudge(config, system_prompt=...)` is the documented way to bring your own wording.
    replaces the system message only — the per-criterion user message stays the bundled one."""
    config = JudgeConfig(model="m", endpoint="http://x/v1", api_key="k")
    fake = FakeCompletions(['Covered.\n{"score": 2}'])
    judge = OpenAIJudge(config, system_prompt="Judge in Klingon.", client=_client_answering(fake))

    await judge.score("How?", "Send an email.", CRITERION)

    system, user = fake.calls[0]
    assert system == {"role": "system", "content": "Judge in Klingon."}
    assert "Send an email" in user["content"]


def test_an_empty_optional_environment_variable_is_refused_like_an_empty_required_one():
    """`MAX_CONCURRENT=` is a half-finished export, exactly as `API_KEY=` is. One condition
    gets one policy: falling back to the field default here would swallow the same typo the
    required variables are refused for, and the process would start on settings nobody
    chose. Both offending variables are named, not only the first."""
    with pytest.raises(RuntimeError) as complaint:
        JudgeConfig.from_mapping(
            {
                **REQUIRED_ENVIRONMENT,
                "RUBRIC_EVAL_JUDGE_MAX_CONCURRENT": "",
                "RUBRIC_EVAL_JUDGE_TEMPERATURE": "",
            }
        )

    assert "RUBRIC_EVAL_JUDGE_MAX_CONCURRENT" in str(complaint.value)
    assert "RUBRIC_EVAL_JUDGE_TEMPERATURE" in str(complaint.value)


# --- the scale the judge declares ----------------------------------------------------------


TEN_POINT = Scale(maximum=10, presence_threshold=5)


def test_the_parser_accepts_what_the_given_scale_allows():
    assert parse_judge_reply('Reasoning.\n{"score": 7}', TEN_POINT).score == 7


def test_the_parser_still_refuses_what_that_scale_does_not_reach():
    with pytest.raises(ValueError, match="not on the scale"):
        parse_judge_reply('Reasoning.\n{"score": 11}', TEN_POINT)


def test_a_bigger_scale_is_still_integral():
    """More levels is a finer grid, not a continuous one — the judge picks a level."""
    with pytest.raises(ValueError, match="not on the scale"):
        parse_judge_reply('Reasoning.\n{"score": 7.5}', TEN_POINT)


def test_the_complaints_quote_the_scale_that_was_asked_for():
    """A judge told "answer with 0, 1 or 2" while working on a ten-point scale would correct
    itself into a wrong grade, so the hint is derived from the scale, never from a constant."""
    with pytest.raises(ValueError, match=r"0, 1, 2, 3, 4, 5, 6, 7, 8, 9 or 10"):
        parse_judge_reply('Reasoning.\n{"score": 11}', TEN_POINT)
    with pytest.raises(ValueError, match=r"0, 1, 2, 3, 4, 5, 6, 7, 8, 9 or 10"):
        parse_judge_reply("no json here", TEN_POINT)


def test_the_malformed_json_complaint_offers_the_scales_own_grades_too():
    """The third complaint is the one a judge sees when it wrote a score object at all, so it
    is the likeliest to be corrected into a grade — on the scale it was asked for, not 0-2."""
    with pytest.raises(ValueError, match=r"0, 1, 2, 3, 4, 5, 6, 7, 8, 9 or 10"):
        parse_judge_reply('Reasoning.\n{"score": 7,}', TEN_POINT)


def test_a_binary_scale_is_offered_as_two_choices_not_as_a_list_of_one():
    with pytest.raises(ValueError, match=r"using 0 or 1"):
        parse_judge_reply("no json here", Scale(maximum=1, presence_threshold=1))


def test_the_parser_insists_on_being_told_which_scale_to_check_against():
    """A default scale here would validate a ten-point judge's replies against 0..2 and then
    correct the model into answering "0, 1 or 2" — every criterion of every case failing, one
    paid call at a time. The caller names the scale or gets no parse."""
    with pytest.raises(TypeError, match="scale"):
        parse_judge_reply('Reasoning.\n{"score": 2}')  # type: ignore[call-arg]

    assert parse_judge_reply('Reasoning.\n{"score": 2}', DEFAULT_SCALE).score == 2
    with pytest.raises(ValueError, match="not on the scale"):
        parse_judge_reply('Reasoning.\n{"score": 3}', DEFAULT_SCALE)


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
    judge = OpenAIJudge(config, system_prompt="Grade 0 to 10.", scale=TEN_POINT)
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
    assert OpenAIJudge(config, system_prompt="Grade it.").system_prompt == "Grade it."


async def test_the_judge_scores_and_self_heals_on_its_own_scale():
    config = JudgeConfig(model="m", endpoint="http://x/v1", api_key="k")
    fake = FakeCompletions(['Reasoning.\n{"score": 12}', 'Corrected.\n{"score": 8}'])
    judge = OpenAIJudge(
        config, system_prompt="Grade 0 to 10.", scale=TEN_POINT, client=_client_answering(fake)
    )

    judge_reply = await judge.score("How?", "Send an email.", CRITERION)

    assert judge_reply.score == 8
    assert "0, 1, 2, 3, 4, 5, 6, 7, 8, 9 or 10" in fake.calls[1][-1]["content"]
