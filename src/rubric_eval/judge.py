"""The LLM-as-a-judge: one call per (case x criterion), with self-healing retries.

Any OpenAI-compatible endpoint works (OpenAI, vLLM, Azure, Ollama, Groq, ...) —
that is the whole point of talking to `base_url` through the official SDK.
"""

import asyncio
import json
import os
import re
from typing import Protocol

from openai import AsyncOpenAI
from pydantic import Field

from rubric_eval.models import DEFAULT_SCALE, Criterion, DocumentedModel, Scale
from rubric_eval.prompt import (
    JUDGE_EN,
    criterion_prompt,
    judge_prompt,
    malformed_json_hint,
    no_json_hint,
    out_of_range_hint,
)

ChatMessage = dict[str, str]
"""One chat message the way the OpenAI SDK wants it: {"role": ..., "content": ...}."""

#: The judge reasons first and closes with the JSON object, so the *last* match wins.
_SCORE_OBJECT = re.compile(r'\{[^{}]*?"score"\s*:\s*(-?\d+(?:\.\d+)?)[^{}]*?\}')


class Verdict(DocumentedModel):
    """One parsed and validated judge reply, for exactly one criterion."""

    score: int
    """Score on the integral scale the judge works on — already checked to be on it."""

    reasoning: str
    """The judge's argument: everything it wrote before the closing JSON object."""


class Judge(Protocol):
    """Extension point: bring your own client, the core does not care.

    An implementation scores one criterion at a time and either returns a valid `Verdict`
    or raises — a raising judge costs one criterion (marked `failed`), never the case.

    It is also where throttling belongs: `evaluation.py` fans out over the whole rubric at
    once and deliberately does not limit that, because only the implementation knows what
    its backend tolerates. `OpenAIJudge` allows `JudgeConfig.max_concurrent` calls in
    flight; an own implementation that talks to a rate-limited service needs its own bound.
    """

    scale: Scale
    """The grading scale this judge answers on, and the reason a custom judge is not tied to
    0..2: every verdict it produces is stored with this scale, `case_score` normalizes by its
    maximum and `is_present` uses its threshold. A judge without it is a broken program —
    `evaluation.py` lets the resulting `AttributeError` through rather than scoring 0."""

    async def score(self, question: str, answer: str, criterion: Criterion) -> Verdict:
        """Decide how well one criterion is covered by one answer.

        Args:
            question: What was asked. Context only — an implementation must not score it.
            answer: The answer under test, exactly as the caller supplied it. May be empty:
                a system that returned nothing is a valid case that scores 0.
            criterion: The single requirement to judge. Only `content` is meant to reach the
                model; `weight` belongs to the scoring layer, and a judge that saw it could
                let importance leak into the score.

        Returns:
            A `Verdict` whose `score` is an integer in 0..`scale.maximum`. An implementation
            that returns more is refused when the `CriterionResult` is built, so a judge
            disagreeing with its own declared scale fails loudly instead of pushing the case
            score above 1.0.

        Raises:
            Anything, for an endpoint that cannot answer. `evaluation.py` turns it into one
            criterion marked `failed` — so raising is the correct way to report an outage,
            and returning a made-up 0 is not.
        """
        ...


class JudgeConfig(DocumentedModel):
    """How to reach the judge model and how hard to try.

    Built from the environment in production, by hand in tests or when the settings come
    from somewhere else.

    Example:
        JudgeConfig.from_env()
        JudgeConfig(model="gpt-4o-mini", endpoint="https://api.openai.com/v1", api_key="sk-...")
    """

    model: str
    """Model name as this endpoint knows it, e.g. "gpt-4o-mini" or "qwen3:8b"."""

    endpoint: str
    """OpenAI-compatible base URL including the version path, e.g. "http://localhost:11434/v1"."""

    api_key: str
    """Key for that endpoint. Local servers usually accept any non-empty string."""

    temperature: float = Field(default=0.0, ge=0)
    """Sampling temperature. 0.0 keeps verdicts reproducible and is the right value while
    each criterion is judged once — judging one several times only says something about the
    model's certainty above 0, where the runs can actually differ."""

    max_tokens: int = Field(default=768, gt=0)
    """Token budget per judge call. Must fit the argument *and* the closing JSON object —
    a reply cut off before the JSON is unparseable and costs a retry."""

    max_attempts: int = Field(default=3, ge=1)
    """How often one criterion may be asked, the first try included. Every further attempt
    replays the broken reply plus the concrete complaint. Running out marks it `failed`.
    At least 1: a budget of 0 would never ask the judge and score the whole rubric 0."""

    max_concurrent: int = Field(default=8, ge=1)
    """How many judge calls may be in flight at once, across all cases this judge serves.
    The endpoint's rate limit is the whole reason: a rubric of 200 criteria would otherwise
    open 200 connections at the same moment and get itself throttled or banned."""

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        """Build the config from `RUBRIC_EVAL_JUDGE_*` environment variables.

        Reads `ENDPOINT`, `API_KEY`, `MODEL` (all required) plus `TEMPERATURE`,
        `MAX_TOKENS`, `MAX_ATTEMPTS` and `MAX_CONCURRENT`, each prefixed
        `RUBRIC_EVAL_JUDGE_`. A variable that is not set is not passed on at all, so the
        field defaults above stay the single source of truth for the optional ones.

        Returns:
            A validated `JudgeConfig`. Numeric variables are parsed and range-checked by
            Pydantic, so a typo cannot turn into a silently odd setting.

        Raises:
            RuntimeError: One or more required variables are missing or empty. The message
                names *all* of them at once — fixing configuration one error per restart is
                misery.
            ValidationError: A numeric variable does not parse or is out of range. The
                message names the offending setting.
        """
        required = (
            "RUBRIC_EVAL_JUDGE_ENDPOINT",
            "RUBRIC_EVAL_JUDGE_API_KEY",
            "RUBRIC_EVAL_JUDGE_MODEL",
        )
        if missing := [variable for variable in required if not os.environ.get(variable)]:
            raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

        settings = {
            "endpoint": os.environ.get("RUBRIC_EVAL_JUDGE_ENDPOINT"),
            "api_key": os.environ.get("RUBRIC_EVAL_JUDGE_API_KEY"),
            "model": os.environ.get("RUBRIC_EVAL_JUDGE_MODEL"),
            "temperature": os.environ.get("RUBRIC_EVAL_JUDGE_TEMPERATURE"),
            "max_tokens": os.environ.get("RUBRIC_EVAL_JUDGE_MAX_TOKENS"),
            "max_attempts": os.environ.get("RUBRIC_EVAL_JUDGE_MAX_ATTEMPTS"),
            "max_concurrent": os.environ.get("RUBRIC_EVAL_JUDGE_MAX_CONCURRENT"),
        }
        return cls(**{field: value for field, value in settings.items() if value})


def parse_verdict(reply: str, scale: Scale = DEFAULT_SCALE) -> Verdict:
    """Pull the score and the argument out of one raw judge reply.

    The judge reasons first and closes with a JSON object, so the *last* `{"score": ...}`
    in the reply wins and everything before it is the reasoning.

    Args:
        reply: The model's message content, unmodified. Markdown fences, prose around the
            object and several score objects are all tolerated.
        scale: What counts as a valid grade, and what the complaints offer the judge instead
            of an invalid one. Defaults to the scale `prompt.JUDGE_EN` describes, which is
            the only one a reply can be assumed to be on when none is named.

    Returns:
        A `Verdict` with an integer score on `scale` and the text preceding the object as
        `reasoning` (the whole reply, if it wrote nothing but the object).

    Raises:
        ValueError: No score object, unparseable JSON, or a score off the scale. **The
            message is not for humans** — the retry loop sends it straight back to the model
            as the correction, so rewording one means changing the prompt. The wordings live
            in `prompt.py`.

    Example:
        parse_verdict('The answer names the address.\n{"score": 2}')
        # Verdict(score=2, reasoning="The answer names the address.")
    """
    score_object = _last_score_object(reply, scale)
    score = _score_in(score_object, scale)
    if not _is_on(scale, score):
        raise ValueError(out_of_range_hint(score, scale))
    return Verdict(score=int(score), reasoning=_reasoning_before(reply, score_object))


def _last_score_object(reply: str, scale: Scale) -> re.Match[str]:
    matches = list(_SCORE_OBJECT.finditer(reply))
    if not matches:
        raise ValueError(no_json_hint(scale))
    return matches[-1]


def _score_in(score_object: re.Match[str], scale: Scale) -> float:
    """Raise ValueError with a concrete hint if the object does not parse as valid JSON —
    the retry loop feeds that hint back to the judge instead of guessing at a broken reply."""
    try:
        return float(json.loads(score_object.group(0))["score"])
    except (ValueError, KeyError, TypeError) as error:  # JSONDecodeError is a ValueError
        raise ValueError(malformed_json_hint(error, scale)) from error


def _is_on(scale: Scale, score: float) -> bool:
    """A reply is on the scale when it named one of the scale's own grades — read from
    `Scale.grades`, so the parser cannot disagree with the prompt about whether the top grade
    counts. Every scale is integral: 1.5 is a judge ignoring the instruction, not a finer
    grade, because more levels means a finer grid to pick from, never a continuous one."""
    return score.is_integer() and int(score) in scale.grades


def _reasoning_before(reply: str, score_object: re.Match[str]) -> str:
    """The judge argues first; if it only emitted the JSON, that has to serve as reasoning."""
    return reply[: score_object.start()].strip() or reply.strip()


class OpenAIJudge:
    """A `Judge` backed by any OpenAI-compatible endpoint — OpenAI, vLLM, Azure, Ollama,
    Groq, OpenRouter — with self-healing retries and a concurrency limit.

    Build it **once** and share it. The `max_concurrent` budget belongs to the instance, so
    one judge per case or per request would hand each of them its own full set of slots —
    exactly the throttle it was configured to have. `api.get_judge` caches one for the whole
    process for that reason.

    Args:
        config: Endpoint, credentials, model and the retry/throttle limits.
        prompt: Replaces the system prompt. Whatever you pass has to keep two promises or
            every reply fails to parse: the model argues first and closes with a single
            `{"score": <grade>}` object, and the prose scale it describes is `scale`. The
            user prompt and the retry complaints are not covered by this — they are derived
            from `scale` in `prompt.py`.
        scale: The grading scale this judge answers on. When it describes its levels, the
            prompt is written from it by `prompt.judge_prompt` and `prompt` can be left out;
            when it does not, there is nothing to instruct the model with — see Raises.

    Raises:
        ValueError: `scale` describes no levels and no `prompt` was given. Refused at
            construction rather than at the first reply: a model told 0-2 while its answers
            are checked against 0..10 fails every criterion of every case, one paid call at
            a time, and the run still comes back looking like a bad system.

    Example:
        judge = OpenAIJudge(JudgeConfig.from_env())
        verdict = await judge.score("How do I report sick leave?", answer, criterion)

        ten_point = Scale(maximum=10, presence_threshold=5, level_descriptions={...})
        finer = OpenAIJudge(config, scale=ten_point)      # prompt written from the scale
    """

    def __init__(
        self, config: JudgeConfig, prompt: str | None = None, scale: Scale = DEFAULT_SCALE
    ):
        self.config = config
        self.scale = scale
        self.system_prompt = prompt or _prompt_for(scale)
        self.client = AsyncOpenAI(base_url=config.endpoint, api_key=config.api_key)
        self._slots_per_loop: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}
        #: One throttle per event loop, see `free_call_slots`.

    @property
    def free_call_slots(self) -> asyncio.Semaphore:
        """The throttle of the loop this call is running in, created on first use.

        One semaphore per loop, not one per judge: an `asyncio.Semaphore` binds itself to
        the event loop of the first caller that has to queue on it and refuses every other
        loop afterwards. A judge outlives loops — `api.get_judge` caches one for the whole
        process — so a single semaphore would serve the first loop and then raise "bound to
        a different event loop" in the next one, and only for rubrics big enough to queue.
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

    async def score(self, question: str, answer: str, criterion: Criterion) -> Verdict:
        """Ask the model about one criterion, correcting it until the reply parses.

        Not a blind retry: on an unusable reply the model is shown its own output plus the
        concrete complaint, so attempt two answers a question rather than repeating one.

        Args:
            question: Context for the model; never scored.
            answer: The answer under test.
            criterion: The single requirement to judge — only its `content` is sent.

        Returns:
            A `Verdict` with a validated integer score and the model's argument for it.

        Raises:
            ValueError: No usable reply within `config.max_attempts`, naming the last
                complaint. `evaluation.py` turns this into one `failed` criterion.
            Exception: Whatever the `openai` SDK raises for a transport, auth or quota
                problem, unchanged — it is contained one layer up, not here.
        """
        conversation = self._opening_messages(question, answer, criterion)
        last_error: ValueError | None = None
        for _ in range(self.config.max_attempts):
            reply = await self._ask(conversation)
            try:
                return parse_verdict(reply, self.scale)
            except ValueError as error:
                last_error = error
                conversation = conversation + _correction(reply, error)
        raise ValueError(
            f"Judge gave no valid answer in {self.config.max_attempts} attempts: {last_error}"
        )

    def _opening_messages(
        self, question: str, answer: str, criterion: Criterion
    ) -> list[ChatMessage]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": criterion_prompt(question, answer, criterion.content)},
        ]

    async def _ask(self, conversation: list[ChatMessage]) -> str:
        """One HTTP call, and a slot held for exactly its duration.

        The slot is taken around the call and not around the retry loop in `score()`: a
        criterion that is parsing a reply, or waiting to be asked again, must not keep a
        slot another criterion could use.
        """
        async with self.free_call_slots:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=conversation,
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens,
            )
        return response.choices[0].message.content or ""


def _prompt_for(scale: Scale) -> str:
    """The system prompt a judge gets when it brings none of its own.

    `JUDGE_EN` for the bundled scale rather than `judge_prompt(scale)`, because the bundled
    worked examples close on grades of 0, 1 and 2 and belong to that scale alone. Every other
    described scale gets the generated instructions without them.
    """
    return JUDGE_EN if scale == DEFAULT_SCALE else judge_prompt(scale)


def _correction(reply: str, error: ValueError) -> list[ChatMessage]:
    """Show the judge its own broken reply plus the concrete complaint.

    That pairing is the self-healing: the model is corrected, not merely asked again.
    """
    return [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": str(error)},
    ]
