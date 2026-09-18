"""Shared test doubles: one fake judge, one case, one run and one run builder, used by the
domain and the HTTP tests alike — so a domain test and an HTTP test never describe *almost*
the same input."""

import pytest
from fastapi.testclient import TestClient

from rubric_eval.api import app, get_judge
from rubric_eval.judge import Verdict
from rubric_eval.metrics import case_score, label_metrics, run_metrics
from rubric_eval.models import (
    DEFAULT_SCALE,
    CaseResult,
    Criterion,
    CriterionResult,
    RunResult,
    Scale,
)

CASE = {
    "id": 1,
    "question": "How do I report sick leave?",
    "answer": "Email hr@example.com before 10:00.",
    "criteria": [
        {"id": 1, "content": "Email before 10:00", "weight": 3},
        {"id": 2, "content": "State the expected last day", "weight": 1},
    ],
}
"""Two criteria weighted 3 and 1 — as JSON for the HTTP tests, as `Case(**CASE)` for the
domain tests, so both describe literally the same case."""


RUN = {
    "cases": [
        CASE,
        {
            "id": 2,
            "question": "How do I request vacation?",
            "answer": "Ask your team lead.",
            "criteria": [{"id": 21, "content": "Submit the request in the HR tool", "weight": 1}],
        },
        {
            "id": 3,
            "question": "Who approves overtime?",
            "answer": "Nobody really knows.",
            "criteria": [{"id": 31, "content": "The line manager approves it", "weight": 1}],
        },
    ]
}
"""Three cases whose criterion ids are unique across the whole run, so one `FakeJudge`
lookup table scores each criterion of each case separately. With `{1: 2, 2: 0, 21: 1, 31: 0}`
the cases score 0.75, 0.5 and 0.0 — one strong, one partial, one total miss, which is what
makes the run metrics say something."""

RUN_VERDICTS = {1: 2, 2: 0, 21: 1, 31: 0}
"""The lookup table producing exactly those three scores."""


class FakeJudge:
    """Scores from a lookup table: `{1: 2, 2: JudgeUnavailableError("down")}` scores criterion
    1 with a 2 and lets the judge fail on criterion 2. No LLM, no network, no retries.

    Carries a `scale` like every `Judge` does, so a test can hand the evaluation layer a
    judge that grades 0..10 without an endpoint that grades 0..10 existing anywhere."""

    def __init__(self, by_criterion: dict[int, int | Exception], scale: Scale = DEFAULT_SCALE):
        self.by_criterion = by_criterion
        self.scale = scale

    async def score(self, question, answer, criterion) -> Verdict:
        outcome = self.by_criterion[criterion.id]
        if isinstance(outcome, Exception):
            raise outcome
        return Verdict(score=outcome, reasoning=f"reasoning for {criterion.id}")


@pytest.fixture
def client():
    """HTTP client against the real app; `use_judge()` swaps in a fake for one test."""
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
    get_judge.cache_clear()


@pytest.fixture
def unconfigured_client(monkeypatch):
    """HTTP client against an app whose judge cannot be built: no environment, no override.
    `raise_server_exceptions=False` makes the client behave like a real one and report the
    status code instead of re-raising the server-side error."""
    for variable in ("ENDPOINT", "API_KEY", "MODEL"):
        monkeypatch.delenv(f"RUBRIC_EVAL_JUDGE_{variable}", raising=False)
    get_judge.cache_clear()
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    get_judge.cache_clear()


def use_judge(judge: FakeJudge) -> None:
    """Point the app's `get_judge` dependency at a fake for the duration of one test."""
    app.dependency_overrides[get_judge] = lambda: judge


def run_of(
    *scores_per_case: dict[int, float],
    scale: Scale = DEFAULT_SCALE,
    labels_by_case_id: dict[int, list[str]] | None = None,
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
            score=case_score(_verdicts(scores, scale), scale),
            scale=scale,
            criterion_results=_verdicts(scores, scale),
            labels=labels_by_case_id.get(case_id, []),
        )
        for case_id, scores in enumerate(scores_per_case, start=1)
    ]
    return RunResult(
        metrics=run_metrics(case_results),
        label_metrics=label_metrics(case_results),
        case_results=case_results,
    )


def _verdicts(scores: dict[int, float], scale: Scale = DEFAULT_SCALE) -> list[CriterionResult]:
    """One verdict per criterion id, all weighted 1 — weights are what `run_of` keeps boring
    so that a comparison test reads as scores in and deltas out."""
    return [
        CriterionResult.judged(
            Criterion(id=criterion_id, content="x", weight=1), score, None, scale
        )
        for criterion_id, score in scores.items()
    ]
