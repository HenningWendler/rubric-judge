"""Shared test doubles: one fake judge, one case, one run and one run builder, used by the
domain and the HTTP tests alike — so a domain test and an HTTP test never describe *almost*
the same input."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rubric_judge.api import app, get_judge
from rubric_judge.judge import Judge, JudgeReply
from rubric_judge.metrics import case_score, label_metrics, run_metrics
from rubric_judge.models import (
    DEFAULT_SCALE,
    CaseResult,
    Criterion,
    CriterionResult,
    RunResult,
    Scale,
)

CASE: dict[str, Any] = {
    "id": 1,
    "context": "The question asked was: How do I report sick leave?",
    "answer": "Email hr@example.com before 10:00.",
    "criteria": [
        {"id": 1, "content": "Email before 10:00", "weight": 3},
        {"id": 2, "content": "State the expected last day", "weight": 1},
    ],
}
"""Two criteria weighted 3 and 1 — as JSON for the HTTP tests, as `Case(**CASE)` for the
domain tests, so both describe literally the same case."""


RUN: dict[str, Any] = {
    "cases": [
        CASE,
        {
            "id": 2,
            "context": "The question asked was: How do I request vacation?",
            "answer": "Ask your team lead.",
            "criteria": [{"id": 21, "content": "Submit the request in the HR tool", "weight": 1}],
        },
        {
            "id": 3,
            "context": "The question asked was: Who approves overtime?",
            "answer": "Nobody really knows.",
            "criteria": [{"id": 31, "content": "The line manager approves it", "weight": 1}],
        },
    ]
}
"""Three cases whose criterion ids are unique across the whole run, so one `FakeJudge`
lookup table scores each criterion of each case separately. With `{1: 2, 2: 0, 21: 1, 31: 0}`
the cases score 0.75, 0.5 and 0.0 — one strong, one partial, one total miss, which is what
makes the run metrics say something."""

RUN_SCORES = {1: 2, 2: 0, 21: 1, 31: 0}
"""The lookup table producing exactly those three scores."""


class FakeJudge:
    """Scores from a lookup table: `{1: 2, 2: JudgeUnavailableError("down")}` scores criterion
    1 with a 2 and lets the judge fail on criterion 2. No LLM, no network, no retries.

    Carries a `scale` like every `Judge` does, so a test can hand the evaluation layer a
    judge that grades 0..10 without an endpoint that grades 0..10 existing anywhere."""

    def __init__(
        self, by_criterion: Mapping[int, int | Exception], scale: Scale = DEFAULT_SCALE
    ):
        self.by_criterion = by_criterion
        self.scale = scale

    async def score(
        self, answer: str, criterion: Criterion, context: str | None = None
    ) -> JudgeReply:
        outcome = self.by_criterion[criterion.id]
        if isinstance(outcome, Exception):
            raise outcome
        return JudgeReply(score=outcome, reasoning=f"reasoning for {criterion.id}")


JUDGE_ENVIRONMENT = {
    "RUBRIC_JUDGE_ENDPOINT": "http://stub/v1",
    "RUBRIC_JUDGE_API_KEY": "stub-key",
    "RUBRIC_JUDGE_MODEL": "stub-model",
}
"""Enough for the service to build its judge at startup. That judge is never called, so the
endpoint does not have to exist; a test that needs answers installs its own with
`use_judge()`."""


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """HTTP client against the real app; `use_judge()` swaps in a fake for one test."""
    with _client_of_started_app(monkeypatch) as client:
        yield client


@pytest.fixture
def status_reporting_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """HTTP client that reports a server fault as its status code, like a real client would,
    instead of re-raising the server-side error into the test."""
    with _client_of_started_app(monkeypatch, raise_server_exceptions=False) as client:
        yield client


@contextmanager
def _client_of_started_app(
    monkeypatch: pytest.MonkeyPatch, **client_options: Any
) -> Iterator[TestClient]:
    """The app started the way uvicorn starts it, configured from `JUDGE_ENVIRONMENT`, and
    reset afterwards so no judge or override leaks into the next test."""
    for variable, value in JUDGE_ENVIRONMENT.items():
        monkeypatch.setenv(variable, value)
    get_judge.cache_clear()
    with TestClient(app, **client_options) as client:
        yield client
    app.dependency_overrides.clear()
    get_judge.cache_clear()


def use_judge(judge: Judge) -> None:
    """Point the app's `get_judge` dependency at another judge for the duration of one test —
    a `FakeJudge`, or a real `OpenAIJudge` wired to a stub endpoint."""
    app.dependency_overrides[get_judge] = lambda: judge


def run_of(
    *scores_per_case: Mapping[int, float],
    scale: Scale = DEFAULT_SCALE,
    labels_by_case_id: Mapping[int, list[str]] | None = None,
) -> RunResult:
    """A finished `RunResult` built straight from judge scores, no judge and no async.

    One mapping per case: `run_of({1: 2, 2: 0}, {21: 1})` is a two-case run whose first case
    has criteria 1 and 2 scored 2 and 0. Case ids count from 1, weights are all 1, so two
    runs built this way are always comparable and every score is easy to predict by hand.
    `scale` grades the whole run on something other than the bundled 0..2, and
    `labels_by_case_id` tags individual cases — `{1: ["table"]}` labels the first one.

    The per-label breakdown is computed rather than passed in, because `RunResult` refuses
    one that does not match its cases and a test should not have to restate it.

    Comparison tests are about the *difference* between two runs, so building them through
    `evaluate_run` and a `FakeJudge` would only add an event loop between the test and the
    numbers it is asserting on.
    """
    labels_by_case_id = labels_by_case_id or {}
    case_results = [
        CaseResult(
            case_id=case_id,
            score=case_score(_criterion_results(scores, scale), scale),
            scale=scale,
            criterion_results=_criterion_results(scores, scale),
            labels=labels_by_case_id.get(case_id, []),
        )
        for case_id, scores in enumerate(scores_per_case, start=1)
    ]
    return RunResult(
        metrics=run_metrics(case_results),
        label_metrics=label_metrics(case_results),
        case_results=case_results,
    )


def _criterion_results(
    scores: Mapping[int, float], scale: Scale = DEFAULT_SCALE
) -> list[CriterionResult]:
    """One result per criterion id, all weighted 1 — weights are what `run_of` keeps boring
    so that a comparison test reads as scores in and deltas out."""
    return [
        CriterionResult.judged(
            Criterion(id=criterion_id, content="x", weight=1), score, None, scale
        )
        for criterion_id, score in scores.items()
    ]
