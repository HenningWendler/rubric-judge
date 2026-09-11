"""Shared test doubles: one fake judge and one case, used by the domain and the HTTP tests."""

import pytest
from fastapi.testclient import TestClient

from rubric_eval.api import app, get_judge
from rubric_eval.judge import Verdict

CASE = {
    "question": "How do I report sick leave?",
    "answer": "Email hr@example.com before 10:00.",
    "criteria": [
        {"id": 1, "content": "Email before 10:00", "weight": 3},
        {"id": 2, "content": "State the expected last day", "weight": 1},
    ],
}
"""Two criteria weighted 3 and 1 — as JSON for the HTTP tests, as `EvaluateRequest(**CASE)`
for the domain tests, so both describe literally the same case."""


class FakeJudge:
    """Scores from a lookup table: `{1: 2, 2: ValueError("down")}` scores criterion 1 with a
    2 and lets the judge die on criterion 2. No LLM, no network, no retries."""

    def __init__(self, by_criterion: dict[int, int | Exception]):
        self.by_criterion = by_criterion

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
