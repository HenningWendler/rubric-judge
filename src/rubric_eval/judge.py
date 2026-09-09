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

from rubric_eval.models import SCALE_MAX, Criterion, DocumentedModel
from rubric_eval.prompt import (
    JUDGE_EN,
    NO_JSON_HINT,
    criterion_prompt,
    malformed_json_hint,
    out_of_range_hint,
)

ChatMessage = dict[str, str]
"""One chat message the way the OpenAI SDK wants it: {"role": ..., "content": ...}."""

#: The judge reasons first and closes with the JSON object, so the *last* match wins.
_SCORE_OBJECT = re.compile(r'\{[^{}]*?"score"\s*:\s*(-?\d+(?:\.\d+)?)[^{}]*?\}')


class Verdict(DocumentedModel):
    """One parsed and validated judge reply, for exactly one criterion."""

    score: int
    """Score on the integral 0..SCALE_MAX scale — already checked to be on it."""

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

    async def score(self, question: str, answer: str, criterion: Criterion) -> Verdict: ...


class JudgeConfig(DocumentedModel):
    """How to reach the judge model and how hard to try.

    Built by hand in tests, from the environment in production:

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
        """Names *all* missing variables at once — fixing config one error per start is misery.

        Variables that are not set are not passed on, so the field defaults above stay the
        single source of truth for them.
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


def parse_verdict(reply: str) -> Verdict:
    """Raise ValueError with the concrete cause — the message is fed back to the judge."""
    score_object = _last_score_object(reply)
    score = _score_in(score_object)
    if not _is_on_scale(score):
        raise ValueError(out_of_range_hint(score))
    return Verdict(score=int(score), reasoning=_reasoning_before(reply, score_object))


def _last_score_object(reply: str) -> re.Match[str]:
    matches = list(_SCORE_OBJECT.finditer(reply))
    if not matches:
        raise ValueError(NO_JSON_HINT)
    return matches[-1]


def _score_in(score_object: re.Match[str]) -> float:
    """Raise ValueError with a concrete hint if the object does not parse as valid JSON —
    the retry loop feeds that hint back to the judge instead of guessing at a broken reply."""
    try:
        return float(json.loads(score_object.group(0))["score"])
    except (ValueError, KeyError, TypeError) as error:  # JSONDecodeError is a ValueError
        raise ValueError(malformed_json_hint(error)) from error


def _is_on_scale(score: float) -> bool:
    """The scale is integral: 1.5 is a judge ignoring the instruction, not a finer grade."""
    return score.is_integer() and 0 <= score <= SCALE_MAX


def _reasoning_before(reply: str, score_object: re.Match[str]) -> str:
    """The judge argues first; if it only emitted the JSON, that has to serve as reasoning."""
    return reply[: score_object.start()].strip() or reply.strip()


class OpenAIJudge:
    """A `Judge` backed by any OpenAI-compatible endpoint, throttled to
    `JudgeConfig.max_concurrent` calls in flight.

    Because one instance serves every request of the process (`api.get_judge` is cached),
    that limit also holds across cases judged at the same time — ten parallel requests
    share the slots instead of being allowed their own set each.
    """

    def __init__(self, config: JudgeConfig, prompt: str | None = None):
        self.config = config
        self.system_prompt = prompt or JUDGE_EN
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
        conversation = self._opening_messages(question, answer, criterion)
        last_error: ValueError | None = None
        for _ in range(self.config.max_attempts):
            reply = await self._ask(conversation)
            try:
                return parse_verdict(reply)
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


def _correction(reply: str, error: ValueError) -> list[ChatMessage]:
    """Show the judge its own broken reply plus the concrete complaint. That is the self-healing."""
    return [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": str(error)},
    ]
