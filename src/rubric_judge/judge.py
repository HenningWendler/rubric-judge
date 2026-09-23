"""The LLM-as-a-judge: one call per (case x criterion), with self-healing retries.

Any OpenAI-compatible endpoint works (OpenAI, vLLM, Azure, Ollama, Groq, ...) —
that is the whole point of talking to `base_url` through the official SDK.
"""

import asyncio
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from typing import Protocol

from openai import (
    NOT_GIVEN,
    APIConnectionError,
    AsyncOpenAI,
    AuthenticationError,
    InternalServerError,
    NotFoundError,
    NotGiven,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import Field, field_validator

from rubric_judge.models import DEFAULT_SCALE, Criterion, DocumentedModel, Scale
from rubric_judge.prompt import (
    JUDGE_EN,
    criterion_prompt,
    judge_prompt,
    malformed_json_hint,
    no_json_hint,
    out_of_range_hint,
)

ChatMessage = ChatCompletionMessageParam
"""One chat message the way the OpenAI SDK wants it: {"role": ..., "content": ...}. The SDK's
own union of message types rather than a plain dict, so a message built here is checked
against what `chat.completions.create` accepts instead of only looking like it."""

_SCORE_OBJECT = re.compile(r'\{[^{}]*?"score"\s*:\s*(-?\d+(?:\.\d+)?)[^{}]*?\}')
"""Where a grade hides in a reply. The judge reasons first and closes with the JSON object,
so the *last* match in a reply is the decision and everything before it is the argument."""

_RETRYABLE_TRANSPORT_FAILURES = (APIConnectionError, RateLimitError, InternalServerError)
"""Endpoint failures worth waiting out: unreachable, rate limited, or broken on the far side.
A rejected key, an unknown model or a malformed request are absent on purpose — asking again
changes nothing about any of them, it only delays the report."""

_SERVICE_WIDE_REJECTIONS = (AuthenticationError, PermissionDeniedError, NotFoundError)
"""Endpoint answers that condemn every call this judge will ever make, not just one: the key
is refused, the key may not use the model, or the model or the URL does not exist. A `400` is
absent on purpose, because one oversized case can earn it while the next case goes through."""

_FIRST_BACKOFF_SECONDS = 0.5
"""How long to wait after the first transport failure; doubled after each further one."""

_HEALTH_CHECK_CONVERSATION: list[ChatMessage] = [{"role": "user", "content": "Reply with OK."}]
"""What `OpenAIJudge.check` sends. The reply is never read, so the prompt only has to be
short and harmless."""

_HEALTH_CHECK_TOKEN_BUDGET = 16
"""Reply tokens a health check may cost. Small, because the reply is never read, but not 1:
the check has to be a request the endpoint accepts like any other."""

_HEALTH_CHECK_TIMEOUT_SECONDS = 60.0
"""How long one health check waits for an answer before it counts as an outage. A 16-token
reply comes back within seconds, while the SDK's own default of 600 seconds would let one
hanging endpoint hold a check for ten minutes. Judge calls keep that default, because a
reasoning model can legitimately think for longer."""

MINIMUM_HEALTH_INTERVAL_SECONDS = 60
"""The shortest `JudgeConfig.health_interval_seconds` there is, apart from 0 for "never". Every
check is a paid call, so an interval of a few seconds would bill one each time."""

JUDGE_VARIABLE_PREFIX = "RUBRIC_JUDGE_"
"""What every environment variable of this project starts with. Any name carrying it, apart
from `ACCESS_TOKEN_VARIABLE`, is read as meant for the judge: refused by
`JudgeConfig.from_mapping` when no field reads it, and enough for the HTTP service to expect a
judge at all."""

ACCESS_TOKEN_VARIABLE = "RUBRIC_JUDGE_ACCESS_TOKEN"
"""The one `RUBRIC_JUDGE_*` variable that configures the HTTP service instead of the judge: the
token its callers must send. It is known here so that `JudgeConfig.from_mapping` does not refuse
it as a typo, and it does not count as a judge being intended, so a service that only compares
can still require it."""

_VARIABLE_PER_FIELD = {
    "endpoint": "RUBRIC_JUDGE_ENDPOINT",
    "api_key": "RUBRIC_JUDGE_API_KEY",
    "model": "RUBRIC_JUDGE_MODEL",
    "temperature": "RUBRIC_JUDGE_TEMPERATURE",
    "max_tokens": "RUBRIC_JUDGE_MAX_TOKENS",
    "max_attempts": "RUBRIC_JUDGE_MAX_ATTEMPTS",
    "max_concurrent": "RUBRIC_JUDGE_MAX_CONCURRENT",
    "health_interval_seconds": "RUBRIC_JUDGE_HEALTH_INTERVAL",
    "health_check_retries": "RUBRIC_JUDGE_HEALTH_RETRIES",
    "health_check_first_pause_seconds": "RUBRIC_JUDGE_HEALTH_FIRST_PAUSE",
}
"""The one list of `JudgeConfig` fields the environment sets, and the variable each is read from."""

_REQUIRED_FIELDS = ("endpoint", "api_key", "model")
"""The fields a judge cannot be built without. A tuple, so the refusal names them in this order."""


class JudgeUnavailableError(Exception):
    """The judge produced no usable reply, so the run it was part of is invalid.

    The whole failure policy in one type: a criterion nobody graded has no score, and the
    plausible-looking 0 that would stand in for it is indistinguishable from a real result.
    Raising this aborts the case and the run — `POST /evaluate` answers `503`, and there is
    no partial document to mistake for a finished one.

    Raise it from a custom `Judge` for anything its endpoint does: refusing, timing out,
    running out of quota, replying with nothing usable. Everything else a judge raises is
    read as a bug in the program and surfaces as a `500`.

    Example:
        class MyJudge:
            scale = DEFAULT_SCALE

            async def score(self, answer, criterion, context=None) -> JudgeReply:
                raise JudgeUnavailableError("my quota is used up")
    """


class UnusableReplyError(ValueError):
    """One judge reply the parser refuses, worded as the complaint to send back to the model.

    A `ValueError`, because that is what `parse_judge_reply` has always raised for a broken
    reply. Its own type nonetheless: the retry loop repeats an attempt for *this* exception
    and for nothing else, so a `ValueError` escaping the parser by accident stays the bug it
    is instead of being replayed to the model three times and reported as a dead endpoint.

    Example:
        try:
            parse_judge_reply("I think it is fine.", DEFAULT_SCALE)
        except UnusableReplyError as complaint:
            str(complaint).startswith("Your reply contained no JSON object")   # True
    """


class JudgeReply(DocumentedModel):
    """One parsed and validated judge reply, for exactly one criterion.

    Named apart from `CriterionResult` because the two are different stages of the same
    criterion: this is what the model said, before any weight, presence or scale is attached
    to it. Only `evaluation._judge_criterion` turns one into the other.

    Example:
        judge_reply = parse_judge_reply(
            'The answer names the address. {"score": 2}', DEFAULT_SCALE
        )
        judge_reply.score       # 2
        judge_reply.reasoning   # "The answer names the address."
    """

    score: int
    """The grade the model named, checked to be one of `Scale.grades` before this object
    exists — hence a whole number, and typed as one. `CriterionResult.score` is a float
    instead, because it may one day be an average over repeated runs of the same criterion;
    `evaluation._judge_criterion` is the single place that widens the one into the other, and
    it does so in writing rather than leaving it to Pydantic's coercion."""

    reasoning: str
    """The judge's argument: everything it wrote before the closing JSON object."""


class Judge(Protocol):
    """Extension point: bring your own client, the core does not care.

    An implementation scores one criterion at a time and either returns a valid `JudgeReply`
    or raises — and a raising judge invalidates the whole run, so raise
    `JudgeUnavailableError` for what the endpoint did and anything else for a bug.

    It is also where throttling belongs: `evaluation.py` fans out over the whole rubric at
    once and deliberately does not limit that, because only the implementation knows what
    its backend tolerates. `OpenAIJudge` allows `JudgeConfig.max_concurrent` calls in
    flight; an own implementation that talks to a rate-limited service needs its own bound.

    Example:
        class AlwaysFullMarks:
            scale = DEFAULT_SCALE

            async def score(self, answer, criterion, context=None) -> JudgeReply:
                return JudgeReply(score=2, reasoning="every criterion is covered")

        case_result = await evaluate_case(AlwaysFullMarks(), case)
        case_result.score   # 1.0
    """

    scale: Scale
    """The grading scale this judge answers on, and the reason a custom judge is not tied to
    0..2: every grade it produces is stored with this scale, `case_score` normalizes by its
    maximum and `is_present` uses its threshold. A judge without it is a broken program, and
    the resulting `AttributeError` reaches the caller like any other bug."""

    async def score(
        self, answer: str, criterion: Criterion, context: str | None = None
    ) -> JudgeReply:
        """Decide how well one criterion is covered by one answer.

        Args:
            answer: The answer under test, exactly as the caller supplied it. May be empty:
                a system that returned nothing is a valid case that scores 0.
            criterion: The single requirement to judge. Only `content` is meant to reach the
                model; `weight` belongs to the scoring layer, and a judge that saw it could
                let importance leak into the score.
            context: What the answer was produced in response to, in the caller's own words,
                or `None` when it stands on its own. Background only — an implementation must
                not score it, and must judge the answer on its own terms when it is absent.

        Returns:
            A `JudgeReply` whose `score` is an integer in 0..`scale.maximum`. An implementation
            that returns more is refused when the `CriterionResult` is built, so a judge
            disagreeing with its own declared scale fails loudly instead of pushing the case
            score above 1.0.

        Raises:
            JudgeUnavailableError: The endpoint could not answer — refused, timed out, out of
                quota, or never replied with anything usable. Raising is the correct way to
                report that, and returning a made-up 0 is not; the run is invalidated rather
                than completed around the gap.
            Exception: Anything else an implementation raises is read as a bug in the
                program and reaches the caller unchanged.

        Example:
            judge_reply = await AlwaysFullMarks().score(   # the class docstring's judge
                "Email hr@example.com before 10:00.",
                Criterion(id=1, content="Report by email before 10:00", weight=3),
                "The question asked was: How do I report sick leave?",
            )
            judge_reply.score   # 2
        """
        ...


class JudgeConfig(DocumentedModel):
    """How to reach the judge model and how hard to try.

    Built from the environment in production, by hand in tests or when the settings come
    from somewhere else.

    Example:
        config = JudgeConfig(
            model="gpt-4o-mini",
            endpoint="https://api.openai.com/v1",
            api_key="sk-test",
        )
        config.max_attempts    # 3
        config.max_concurrent  # 8
    """

    model: str
    """Model name as this endpoint knows it, e.g. "gpt-4o-mini" or "qwen3:8b"."""

    endpoint: str
    """OpenAI-compatible base URL including the version path, e.g. "http://localhost:11434/v1"."""

    api_key: str
    """Key for that endpoint. Local servers usually accept any non-empty string."""

    temperature: float = Field(default=0.0, ge=0)
    """Sampling temperature. 0.0 keeps grades reproducible and is the right value while
    each criterion is judged once — judging one several times only says something about the
    model's certainty above 0, where the runs can actually differ."""

    max_tokens: int = Field(default=768, gt=0)
    """Token budget per judge call. Must fit the argument *and* the closing JSON object —
    a reply cut off before the JSON is unparseable and costs a retry."""

    max_attempts: int = Field(default=3, ge=1)
    """How often one criterion may be asked, the first try included, whether the endpoint
    failed or its reply did. An unusable reply is replayed to the model with the concrete
    complaint; a transport failure is waited out. Running out invalidates the run. This is
    the only retry budget there is — the SDK's own is switched off, so two of them cannot
    multiply behind your back. At least 1: a budget of 0 would never ask the judge at all."""

    max_concurrent: int = Field(default=8, ge=1)
    """How many judge calls may be in flight at once, across all cases this judge serves.
    The endpoint's rate limit is the whole reason: a rubric of 200 criteria would otherwise
    open 200 connections at the same moment and get itself throttled or banned."""

    health_interval_seconds: int = Field(default=0, ge=0)
    """How often a running service proves its judge still answers, in seconds, with one
    `OpenAIJudge.check`. 0 never does, and otherwise it is at least
    `MINIMUM_HEALTH_INTERVAL_SECONDS`. Every judge call that gets an answer counts as proof
    too, so a busy service never spends a check."""

    health_check_retries: int = Field(default=3, ge=0)
    """How often a failed periodic health check is repeated before the judge counts as
    unhealthy, for outages only. A rejected key, model or request is final at once. 0 turns
    the judge unhealthy on the first failed check."""

    health_check_first_pause_seconds: int = Field(default=300, ge=1)
    """How long a periodic health check waits before its first retry, doubled before each
    further one. The default schedule waits 5, 10 and 20 minutes, so a provider's outage has
    35 minutes to heal before any replica is reported unhealthy."""

    @field_validator("health_interval_seconds")
    @classmethod
    def _refuse_an_interval_shorter_than_a_minute(cls, interval_seconds: int) -> int:
        """Refuse a health interval that would bill a check every few seconds.

        Args:
            interval_seconds: The configured interval, already known to be 0 or more.

        Returns:
            `interval_seconds` unchanged, when it is 0 or at least
            `MINIMUM_HEALTH_INTERVAL_SECONDS`.

        Raises:
            ValueError: 1 to 59 seconds. Pydantic reports it as a `ValidationError` naming
                the field.
        """
        if 0 < interval_seconds < MINIMUM_HEALTH_INTERVAL_SECONDS:
            raise ValueError(
                f"must be 0 to disable the check, or at least "
                f"{MINIMUM_HEALTH_INTERVAL_SECONDS} seconds, not {interval_seconds}"
            )
        return interval_seconds

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        """Build the config from the process's `RUBRIC_JUDGE_*` environment variables.

        The one place in the library that reads `os.environ`; everything it then decides is
        `from_mapping`, which is where the rules and the wording of the complaints live.

        Returns:
            A validated `JudgeConfig`, exactly as `from_mapping(os.environ)` builds it.

        Raises:
            RuntimeError: A required variable is missing, or any variable is set to the empty
                string — see `from_mapping`.
            ValidationError: A numeric variable does not parse or is out of range.

        Example:
            JudgeConfig.from_env().model   # "gpt-4o-mini", with RUBRIC_JUDGE_MODEL set
        """
        return cls.from_mapping(os.environ)

    @classmethod
    def from_mapping(cls, environment: Mapping[str, str]) -> "JudgeConfig":
        """Build the config from a mapping of `RUBRIC_JUDGE_*` variables to their values.

        Reads `ENDPOINT`, `API_KEY`, `MODEL` (all required) plus `TEMPERATURE`,
        `MAX_TOKENS`, `MAX_ATTEMPTS`, `MAX_CONCURRENT`, `HEALTH_INTERVAL`, `HEALTH_RETRIES`
        and `HEALTH_FIRST_PAUSE`, each prefixed `RUBRIC_JUDGE_`. An optional variable that is
        absent is not passed on, so the field defaults above stay the single source of truth
        for it. `ACCESS_TOKEN` belongs to the HTTP service: it is accepted here and refused
        when empty like every other variable, but no field reads it.

        A variable set to the *empty* string is a half-finished configuration and is refused
        for every variable alike, required or optional — one condition cannot mean "your key
        is missing" on one line and "take the default" on the next.

        Whitespace around a value is dropped before anything else is decided. `docker run
        --env-file` passes a trailing space through verbatim where `uvicorn --env-file`
        strips it, and an endpoint URL ending in two spaces answers every call with a 404.
        A value of nothing but whitespace is therefore empty, and refused as such.

        A `RUBRIC_JUDGE_*` name outside the set above is refused as unknown. It is nearly
        always a typo, and `RUBRIC_JUDGE_HEALTH_INTERVALL` read as nothing would start a
        service that never checks its judge while its operator believes it does.

        Args:
            environment: Variable name to value, `os.environ` in production and a plain dict
                anywhere else. Names without the `RUBRIC_JUDGE_` prefix are ignored, so the
                whole process environment can be handed in. Values are the strings they are
                exported as, surrounding whitespace ignored; an empty one is refused rather
                than read as "unset".

        Returns:
            A validated `JudgeConfig`. Numeric variables are parsed and range-checked by
            Pydantic, so a typo cannot turn into a silently odd setting.

        Raises:
            RuntimeError: A required variable is missing, any variable is set to the empty
                string, or a `RUBRIC_JUDGE_*` name is unknown. The message names *all* of
                them at once and says which of the three each one is, because fixing
                configuration one error per restart is misery.
            ValidationError: A numeric variable does not parse or is out of range. The
                message names the offending setting.

        Example:
            JudgeConfig.from_mapping(
                {
                    "RUBRIC_JUDGE_ENDPOINT": "http://localhost:11434/v1",
                    "RUBRIC_JUDGE_API_KEY": "ollama",
                    "RUBRIC_JUDGE_MODEL": "qwen3:8b",
                }
            ).max_attempts   # 3, the field default
        """
        trimmed_value_per_variable = {
            name: value.strip()
            for name, value in environment.items()
            if name.startswith(JUDGE_VARIABLE_PREFIX)
        }
        unusable = (
            _missing_required_variables(trimmed_value_per_variable)
            + _empty_variables(trimmed_value_per_variable)
            + _unknown_variables(trimmed_value_per_variable)
        )
        if unusable:
            raise RuntimeError(f"Unusable environment variables: {', '.join(unusable)}")

        return cls.model_validate(
            {
                field: trimmed_value_per_variable[variable]
                for field, variable in _VARIABLE_PER_FIELD.items()
                if variable in trimmed_value_per_variable
            }
        )


def _missing_required_variables(value_per_variable: Mapping[str, str]) -> list[str]:
    """Name each required variable that is absent, since a judge cannot be built without it."""
    return [
        f"{_VARIABLE_PER_FIELD[field]} is missing"
        for field in _REQUIRED_FIELDS
        if _VARIABLE_PER_FIELD[field] not in value_per_variable
    ]


def _empty_variables(value_per_variable: Mapping[str, str]) -> list[str]:
    """Name each variable set to nothing, a half-finished line rather than a wish for a default."""
    return [f"{variable} is empty" for variable, value in value_per_variable.items() if value == ""]


def _unknown_variables(value_per_variable: Mapping[str, str]) -> list[str]:
    """Name each `RUBRIC_JUDGE_*` name no field reads, because it is nearly always a typo."""
    known_variables = {*_VARIABLE_PER_FIELD.values(), ACCESS_TOKEN_VARIABLE}
    return [
        f"{variable} is unknown"
        for variable in value_per_variable
        if variable not in known_variables
    ]


def parse_judge_reply(reply: str, scale: Scale) -> JudgeReply:
    r"""Pull the score and the argument out of one raw judge reply.

    The judge reasons first and closes with a JSON object, so the *last* `{"score": ...}`
    in the reply wins and everything before it is the reasoning.

    Args:
        reply: The model's message content, unmodified. Markdown fences, prose around the
            object and several score objects are all tolerated.
        scale: What counts as a valid grade, and what the complaints offer the judge instead
            of an invalid one. The judge's own `scale` and never a guess: assuming one would
            check a ten-point reply against 0..2 and then instruct the model to "answer with
            0, 1 or 2" — failing every criterion of every case, one paid call at a time.

    Returns:
        A `JudgeReply` with an integer score on `scale` and the text preceding the object as
        `reasoning` (the whole reply, if it wrote nothing but the object).

    Raises:
        UnusableReplyError: A `ValueError`. No score object, unparseable JSON, or a score off
            the scale. **The message is not for humans** — the retry loop sends it straight
            back to the model as the correction, so rewording one means changing the prompt.
            The wordings live in `prompt.py`.

    Example:
        parse_judge_reply('The answer names the address.\n{"score": 2}', DEFAULT_SCALE)
        # JudgeReply(score=2, reasoning="The answer names the address.")
    """
    score_object = _last_score_object(reply, scale)
    score = _score_in(score_object, scale)
    if not _is_on(scale, score):
        raise UnusableReplyError(out_of_range_hint(score, scale))
    return JudgeReply(score=int(score), reasoning=_reasoning_before(reply, score_object))


def _last_score_object(reply: str, scale: Scale) -> re.Match[str]:
    """Find the judge's decision, or complain to it in the words it will be shown.

    Args:
        reply: The model's message content, unmodified.
        scale: The scale the judge answers on, quoted in the complaint so a judge on a
            ten-point scale is never told to answer with 0, 1 or 2.

    Returns:
        The match for the *last* score object in the reply — the judge argues first, so
        everything before that match is its reasoning.

    Raises:
        UnusableReplyError: Nothing in the reply looks like a score object. The message is
            the correction the retry loop sends to the model verbatim, not a report for a
            human reading a log.
    """
    matches = list(_SCORE_OBJECT.finditer(reply))
    if not matches:
        raise UnusableReplyError(no_json_hint(scale))
    return matches[-1]


def _score_in(score_object: re.Match[str], scale: Scale) -> float:
    """Read the grade out of a matched score object, concretely enough to correct the judge.

    Args:
        score_object: A match from `_SCORE_OBJECT`; its whole text is parsed as JSON.
        scale: The scale the judge answers on, quoted in the complaint.

    Returns:
        The grade as a float, whatever the scale — whether it is *on* the scale is
        `_is_on`'s question, and separating the two is what lets an off-scale grade be
        quoted back to the judge.

    Raises:
        UnusableReplyError: The object is not valid JSON, carries no `score` key, or carries
            one that is not a number. All three are collapsed into one complaint because the
            judge can act on all three the same way: write the line again, correctly. The
            underlying error is quoted inside it, so the model is told what is broken.
    """
    try:
        return float(json.loads(score_object.group(0))["score"])
    except (ValueError, KeyError, TypeError) as error:  # JSONDecodeError is a ValueError
        raise UnusableReplyError(malformed_json_hint(error, scale)) from error


def _is_on(scale: Scale, score: float) -> bool:
    """Asks `Scale.grades` rather than the bounds.

    That way the parser cannot disagree with the prompt about whether the top grade counts.
    Every scale is integral: 1.5 is a judge ignoring the instruction, not a finer grade,
    because more levels means a finer grid to pick from, never a continuous one.
    """
    return score.is_integer() and int(score) in scale.grades


def _reasoning_before(reply: str, score_object: re.Match[str]) -> str:
    """The judge argues first; if it only emitted the JSON, that has to serve as reasoning."""
    return reply[: score_object.start()].strip() or reply.strip()


class JudgeHealth:
    """Whether a judge's endpoint has recently proven to work, and when to ask it again.

    The one clock every record and every question about staleness is measured on lives here,
    so the judge that records evidence and the monitor that asks about it can never disagree
    about what time it is. A test hands in a clock of its own and lets a day pass at once. A
    judge starts unproven: nothing is healthy before its endpoint has answered once.

    Example:
        health = JudgeHealth()
        health.is_healthy                                          # False
        health.record_proof()
        health.is_healthy                                          # True
        health.seconds_until_check_due(interval_seconds=3600)      # just under 3600.0
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        """Start unproven, with a check due at once.

        Args:
            clock: Returns the current moment in seconds and never goes backwards. The
                default is `time.monotonic`, which a wall-clock change cannot move.
        """
        self.clock = clock
        self.is_healthy = False
        """True from the last answer the endpoint gave until the next refusal."""

        self.last_evidence_at: float | None = None
        """When the endpoint last answered or last failed, on `clock`, or `None` before
        either. A failure restarts the wait as much as an answer does, so a check that failed
        is repeated one interval later and not in a tight loop."""

    def record_proof(self) -> None:
        """The endpoint answered, so the judge is healthy and the next check waits again.

        Example:
            judge.health.record_proof()
        """
        self.is_healthy = True
        self.last_evidence_at = self.clock()

    def record_failure(self) -> None:
        """The endpoint refused or never answered, so the judge is unhealthy until it answers.

        Example:
            judge.health.record_failure()
        """
        self.is_healthy = False
        self.last_evidence_at = self.clock()

    def seconds_until_check_due(self, interval_seconds: float) -> float:
        """How long a periodic check can still wait.

        Args:
            interval_seconds: How long evidence stays fresh, positive. The caller does not
                ask at all when checks are disabled.

        Returns:
            Seconds until the evidence is older than `interval_seconds`, and 0.0 once it is,
            or when there has been no evidence yet.

        Example:
            judge.health.record_failure()
            judge.health.seconds_until_check_due(interval_seconds=86400)   # about 86400.0
        """
        if self.last_evidence_at is None:
            return 0.0
        return max(0.0, self.last_evidence_at + interval_seconds - self.clock())


class OpenAIJudge:
    """A `Judge` backed by any OpenAI-compatible endpoint, with retries and a call limit.

    OpenAI, vLLM, Azure, Ollama, Groq, OpenRouter — whatever answers at the configured
    endpoint. An unusable reply is corrected rather than merely repeated, and the endpoint's
    rate limit is respected by holding `JudgeConfig.max_concurrent` calls in flight at most.

    Build it **once** and share it. The `max_concurrent` budget belongs to the instance, so
    one judge per case or per request would hand each of them its own full set of slots —
    exactly the throttle it was configured to have. `api.get_judge` caches one for the whole
    process for that reason.

    Example:
        judge = OpenAIJudge(JudgeConfig.from_env())   # once per process, then shared
        str(judge.scale)                              # "0..2 (covered from 0.5)"
    """

    def __init__(
        self,
        config: JudgeConfig,
        system_prompt: str | None = None,
        scale: Scale = DEFAULT_SCALE,
        client: AsyncOpenAI | None = None,
    ):
        """Wire a judge to an endpoint, a scale and the prompt that agrees with both.

        Args:
            config: Endpoint, credentials, model and the retry/throttle limits.
            system_prompt: Replaces the generated system prompt — the only one of the three
                prompts a judge sends that is yours to write; `criterion_prompt` is the user
                prompt and the retry complaints are derived from `scale` in `prompt.py`.
                Whatever you pass has to keep two promises or every reply fails to parse: the
                model argues first and closes with a single `{"score": <grade>}` object, and
                the prose scale it describes is `scale`. `None` generates one from `scale`.
            scale: The grading scale this judge answers on. When it describes its levels, the
                prompt is written from it by `prompt.judge_prompt` and `system_prompt` can be
                left out; when it does not, there is nothing to instruct the model with — see
                Raises.
            client: An already-built SDK client to talk through — one with a shared connection
                pool, an `AzureOpenAI`, or a test double. `None` builds one from `config`
                with the SDK's own retrying switched off, so `config.max_attempts` is the only
                retry budget there is. A client you pass keeps whatever `max_retries` it was
                built with, and that cannot be enforced from here: the two budgets then
                multiply, and the SDK's default of 2 turns 3 configured attempts into 9 calls.
                Build yours with `max_retries=0` unless you mean exactly that.

        Raises:
            ValueError: `scale` describes no levels and no `system_prompt` was given. Refused
                at construction rather than at the first reply: a model told 0-2 while its
                answers are checked against 0..10 fails every criterion of every case, one
                paid call at a time, and the run still comes back looking like a bad system.

        Example:
            config = JudgeConfig(
                model="gpt-4o-mini",
                endpoint="https://api.openai.com/v1",
                api_key="sk-test",
            )
            pass_fail = Scale(
                maximum=1,
                presence_threshold=1,
                level_descriptions={1: "Covered.", 0: "Not covered."},
            )
            judge = OpenAIJudge(config, scale=pass_fail)   # prompt written from the scale
            "1 = Covered." in judge.system_prompt          # True
        """
        self.config = config
        self.scale = scale
        self.system_prompt = system_prompt or _system_prompt_for(scale)
        self.client = client or _client_for(config)
        self.health = JudgeHealth()
        # One throttle per event loop, see `free_call_slots`.
        self._slots_per_loop: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}

    @property
    def free_call_slots(self) -> asyncio.Semaphore:
        """The throttle of the loop this call is running in, created on first use.

        One semaphore per loop, not one per judge: an `asyncio.Semaphore` binds itself to
        the event loop of the first caller that has to queue on it and refuses every other
        loop afterwards. A judge outlives loops — `api.get_judge` caches one for the whole
        process — so a single semaphore would serve the first loop and then raise "bound to
        a different event loop" in the next one, and only for rubrics big enough to queue.

        Returns:
            The `asyncio.Semaphore` guarding this judge's calls on the running loop, with
            `JudgeConfig.max_concurrent` permits. The same object for every call on that
            loop, which is what makes it a shared budget rather than a per-call one.

        Raises:
            RuntimeError: Read outside a running event loop. A throttle without a loop to
                throttle is nothing a caller could use.

        Example:
            async def one_throttle_per_loop(judge: OpenAIJudge) -> bool:
                return judge.free_call_slots is judge.free_call_slots

            asyncio.run(one_throttle_per_loop(OpenAIJudge(config)))   # True
        """
        loop = asyncio.get_running_loop()
        if loop not in self._slots_per_loop:
            self._forget_closed_loops()
            self._slots_per_loop[loop] = asyncio.Semaphore(self.config.max_concurrent)
        return self._slots_per_loop[loop]

    def _forget_closed_loops(self) -> None:
        """Keep one entry per *running* loop, not per loop that ever ran.

        A `weakref.WeakKeyDictionary` would not free anything here: a semaphore stores the
        loop it bound itself to, so the value keeps its own key alive. Sweeping on the rare
        event of a new loop appearing is what actually releases them.
        """
        self._slots_per_loop = {
            loop: slots for loop, slots in self._slots_per_loop.items() if not loop.is_closed()
        }

    async def score(
        self, answer: str, criterion: Criterion, context: str | None = None
    ) -> JudgeReply:
        """Ask the model about one criterion until it answers usably, or give the run up.

        Two kinds of failure share the one attempt budget, because each of them costs a call.
        An unusable reply is not blindly repeated: the model is shown its own output plus the
        concrete complaint, so attempt two answers a question rather than repeating one. A
        transport failure has nothing to correct and is waited out instead — a single rate
        limit must not throw away a whole catalog's worth of judging.

        Args:
            answer: The answer under test.
            criterion: The single requirement to judge — only its `content` is sent.
            context: What the answer was produced in response to, or `None`. Sent to the
                model as background and never scored; when it is `None` the user message
                carries no context block at all.

        Returns:
            A `JudgeReply` with a validated integer score and the model's argument for it.

        Raises:
            JudgeUnavailableError: No usable reply within `config.max_attempts`, naming
                the last cause and chaining it as `__cause__`. The run is invalid from here
                on — nothing above turns this into a score.
            openai.OpenAIError: An endpoint failure no retry can heal — a rejected key, an
                unknown model, a malformed request — raised on the first attempt, unchanged.

        Example:
            Against a stub endpoint, so the example costs nothing; a live one is reached by
            leaving `client` out and configuring it with `JudgeConfig.from_env()`.

            completion = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='It says so. {"score": 2}'),
                        finish_reason="stop",
                    )
                ]
            )

            async def always_the_same(**request):
                return completion

            judge = OpenAIJudge(
                JudgeConfig(model="stub", endpoint="http://localhost/v1", api_key="stub"),
                client=SimpleNamespace(
                    chat=SimpleNamespace(
                        completions=SimpleNamespace(create=always_the_same)
                    )
                ),
            )
            judge_reply = await judge.score(
                "Email hr@example.com before 10:00.",
                Criterion(id=1, content="Report by email before 10:00", weight=3),
                "The question asked was: How do I report sick leave?",
            )
            judge_reply.score       # 2
            judge_reply.reasoning   # "It says so."
        """
        conversation = self._opening_messages(answer, criterion, context)
        last_failure: Exception | None = None
        for attempt in range(self.config.max_attempts):
            try:
                reply = await self._ask(conversation)
            except _RETRYABLE_TRANSPORT_FAILURES as outage:
                last_failure = outage
                await asyncio.sleep(self._backoff_seconds(attempt))
                continue
            try:
                return parse_judge_reply(reply, self.scale)
            except UnusableReplyError as complaint:
                last_failure = complaint
                conversation = conversation + _correction(reply, complaint)
        raise JudgeUnavailableError(
            f"Judge gave no usable answer in {self.config.max_attempts} attempts: "
            f"{_describe(last_failure)}"
        ) from last_failure

    def _backoff_seconds(self, failed_attempt: int) -> float:
        """How long to wait before asking a failing endpoint again.

        Backing off gives an endpoint that is rate limiting or restarting time, instead of
        hammering it with the retry it just refused.

        Args:
            failed_attempt: Which attempt just failed, counting from 0 — the loop variable
                of `score`, not a count of failures.

        Returns:
            Seconds to sleep: `_FIRST_BACKOFF_SECONDS` doubled once per failure so far, and
            exactly 0.0 once the attempt budget is spent — the run is invalid either way,
            and waiting then only delays the bad news.
        """
        if failed_attempt + 1 >= self.config.max_attempts:
            return 0.0
        return _FIRST_BACKOFF_SECONDS * 2.0**failed_attempt

    def _opening_messages(
        self, answer: str, criterion: Criterion, context: str | None
    ) -> list[ChatMessage]:
        """The conversation every attempt starts from.

        A correction is appended to it rather than the whole prompt being rebuilt around a
        broken reply, which is what makes the second attempt a follow-up question.
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": criterion_prompt(answer, criterion.content, context)},
        ]

    async def check(self) -> None:
        """Prove that the endpoint accepts this judge's key, model and request shape.

        One tiny call with the same model and temperature `score` sends, so a service can
        refuse to start before a real request finds out. Any answer is proof, even an empty
        one, because only the endpoint's acceptance is under test. Outages are retried quickly
        with the backoff `score` uses, so a starting service fails fast and its platform's
        restart policy tries again. A rejection is not retried.

        Every failure marks `health` unhealthy, where a failed `score` does so only for a
        rejection. This request is fixed and small, so nothing about one case can be to blame.

        Raises:
            JudgeUnavailableError: The endpoint gave no answer within `config.max_attempts`
                attempts of `check_once`, naming the last cause.
            openai.OpenAIError: The endpoint rejected the call, for example a refused key
                (`401`), an unknown model (`404`) or a parameter the model does not take
                (`400`). Raised on the first attempt, unchanged.

        Example:
            judge = OpenAIJudge(JudgeConfig.from_env())
            await judge.check()   # returns, or raises before the first real request
            judge.health.is_healthy   # True
        """
        try:
            await self._check_until_answered()
        except (JudgeUnavailableError, OpenAIError):
            self.health.record_failure()
            raise

    async def check_once(self) -> None:
        """Send the health check exactly once, for a caller with a retry schedule of its own.

        The call waits at most `_HEALTH_CHECK_TIMEOUT_SECONDS` for an answer. An answer marks
        `health` healthy. An outage is only reported and never recorded, because the caller
        decides when a run of outages is long enough to call the judge unhealthy. A `401`,
        `403` or `404` is recorded as a failure here, as it is on every call.

        Raises:
            JudgeUnavailableError: The endpoint did not answer: unreachable, timed out, rate
                limited or failing on its side. Chained to the SDK's exception.
            openai.OpenAIError: The endpoint rejected the call, unchanged.

        Example:
            await judge.check_once()   # one call of at most 16 tokens and 60 seconds
        """
        try:
            await self._call_endpoint(
                _HEALTH_CHECK_CONVERSATION,
                _HEALTH_CHECK_TOKEN_BUDGET,
                timeout_seconds=_HEALTH_CHECK_TIMEOUT_SECONDS,
            )
        except _RETRYABLE_TRANSPORT_FAILURES as outage:
            raise JudgeUnavailableError(
                f"the judge's endpoint did not answer the health check: {_describe(outage)}"
            ) from outage

    async def _check_until_answered(self) -> None:
        """Repeat `check_once` through outages with the quick backoff `score` uses.

        Raises:
            JudgeUnavailableError: No answer within `config.max_attempts`.
            openai.OpenAIError: The endpoint rejected the call.
        """
        last_outage: JudgeUnavailableError | None = None
        for attempt in range(self.config.max_attempts):
            try:
                await self.check_once()
                return
            except JudgeUnavailableError as outage:
                last_outage = outage
                await asyncio.sleep(self._backoff_seconds(attempt))
        raise JudgeUnavailableError(
            f"Judge endpoint did not answer the health check in {self.config.max_attempts} "
            f"attempts: {_describe(last_outage.__cause__ if last_outage else None)}"
        ) from last_outage

    async def _ask(self, conversation: list[ChatMessage]) -> str:
        """One judge call for one criterion, and the text the model wrote."""
        response = await self._call_endpoint(conversation, self.config.max_tokens)
        return _reply_text(response)

    async def _call_endpoint(
        self,
        conversation: list[ChatMessage],
        token_budget: int,
        timeout_seconds: float | NotGiven = NOT_GIVEN,
    ) -> ChatCompletion:
        """One HTTP call, a slot held for exactly its duration, and what it says about health.

        The slot is taken around the call and not around the retry loop in `score()`: a
        criterion that is parsing a reply, or waiting to be asked again, must not keep a
        slot another criterion could use.

        Any answer is recorded as proof, even one without content, because the endpoint did
        accept the call. A rejection in `_SERVICE_WIDE_REJECTIONS` is recorded as a failure at
        once, since every later call would meet it too. An outage is not, because it is
        answered with a `503` and heals by itself.

        Args:
            conversation: The messages to send, system prompt first.
            token_budget: The most reply tokens the call may cost, positive.
            timeout_seconds: How long to wait for the answer. Left out, the SDK's own
                default applies.

        Returns:
            The chat completion exactly as the endpoint returned it.

        Raises:
            openai.OpenAIError: Whatever the SDK raised, unchanged.
        """
        try:
            async with self.free_call_slots:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=conversation,
                    temperature=self.config.temperature,
                    max_completion_tokens=token_budget,
                    timeout=timeout_seconds,
                )
        except _SERVICE_WIDE_REJECTIONS:
            self.health.record_failure()
            raise
        self.health.record_proof()
        return response


def _reply_text(response: ChatCompletion) -> str:
    """What the model actually wrote, or a loud failure when it wrote nothing.

    Args:
        response: One chat completion exactly as the endpoint returned it.

    Returns:
        The content of its first choice, never empty.

    Raises:
        JudgeUnavailableError: The response carries no choice at all, or one whose content is
            absent — a reply cut off by the token budget, removed by a content filter, or
            spent on a tool call. The `finish_reason` is named because it is the only thing
            that tells those apart. Not retried and never sent back to the model: reprompting
            cannot undo a truncation, and complaining about "no JSON object" would blame the
            model for what its endpoint did.
    """
    if not response.choices:
        raise JudgeUnavailableError("the judge's endpoint answered without a single choice")
    choice = response.choices[0]
    if not choice.message.content:
        raise JudgeUnavailableError(
            f"the judge's endpoint answered with no content, finish_reason "
            f"{choice.finish_reason!r}"
        )
    return choice.message.content


def _client_for(config: JudgeConfig) -> AsyncOpenAI:
    """The SDK client a judge talks through when it was given none of its own.

    `max_retries=0` is the point of building it here: the SDK retries twice by default, and
    in series with `score`'s own loop the two budgets multiply into nine calls where three
    were configured. `config.max_attempts` is meant to be the only retry budget there is.
    """
    return AsyncOpenAI(base_url=config.endpoint, api_key=config.api_key, max_retries=0)


def _system_prompt_for(scale: Scale) -> str:
    """The system prompt a judge gets when it brings none of its own.

    `JUDGE_EN` for the bundled scale rather than `judge_prompt(scale)`, because the bundled
    worked examples close on grades of 0, 1 and 2 and belong to that scale alone. Every other
    described scale gets the generated instructions without them.
    """
    return JUDGE_EN if scale == DEFAULT_SCALE else judge_prompt(scale)


def _describe(failure: BaseException | None) -> str:
    """The cause to name when the judge is given up on, never an empty string.

    `str(TimeoutError())` *is* the empty string, and a timeout is the likeliest judge failure
    of all — then the class name is the only cause there is to report.
    """
    return str(failure) or type(failure).__name__


def _correction(reply: str, error: ValueError) -> list[ChatMessage]:
    """Show the judge its own broken reply plus the concrete complaint.

    That pairing is the self-healing: the model is corrected, not merely asked again.
    """
    return [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": str(error)},
    ]
