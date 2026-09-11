"""The HTTP layer only: request validation, the judge wiring, and that the result is passed
through unchanged. What the numbers mean is tested in `test_evaluation.py`."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from conftest import BATCH, BATCH_VERDICTS, CASE, FakeJudge, use_judge

from rubric_eval.api import app, get_judge


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_evaluate_serves_the_evaluation_of_the_case(client):
    use_judge(FakeJudge({1: 2, 2: 0}))

    body = client.post("/evaluate", json=CASE).json()

    assert body["score"] == pytest.approx(0.75)
    assert [verdict["score"] for verdict in body["criterion_results"]] == [2.0, 0.0]


def test_criteria_must_not_be_empty(client):
    use_judge(FakeJudge({}))
    assert client.post("/evaluate", json={**CASE, "criteria": []}).status_code == 422


def test_weight_must_be_positive(client):
    use_judge(FakeJudge({}))
    bad = {**CASE, "criteria": [{"id": 1, "content": "x", "weight": 0}]}
    assert client.post("/evaluate", json=bad).status_code == 422


def test_criterion_content_must_not_be_empty(client):
    use_judge(FakeJudge({}))
    bad = {**CASE, "criteria": [{"id": 1, "content": "", "weight": 3}]}
    assert client.post("/evaluate", json=bad).status_code == 422


def test_criterion_content_must_not_be_only_whitespace(client):
    """`min_length=1` lets "   " through, and a blank criterion costs a real LLM call to
    produce a meaningless verdict."""
    use_judge(FakeJudge({}))
    bad = {**CASE, "criteria": [{"id": 1, "content": "   ", "weight": 3}]}
    assert client.post("/evaluate", json=bad).status_code == 422


def test_an_infinite_weight_is_rejected_instead_of_scoring_null(client):
    """JSON allows the bare literal `Infinity`, and Python's parser accepts it. It poisons the
    weighted fold into `nan`, which FastAPI serializes as `null` — a 200 response whose
    `score` violates the declared non-nullable float. Reject it at the boundary."""
    use_judge(FakeJudge({1: 2}))
    body = (
        '{"question": "q", "answer": "a",'
        ' "criteria": [{"id": 1, "content": "x", "weight": Infinity}]}'
    )
    json_body = {"content-type": "application/json"}

    response = client.post("/evaluate", content=body, headers=json_body)

    assert response.status_code == 422
    assert "finite" in response.text


def test_a_rejected_request_is_reported_without_echoing_the_value(client):
    """The 422 body has to stay serializable — and a validation error that cannot be rendered
    would reach the client as a 500 instead."""
    use_judge(FakeJudge({}))
    bad = {**CASE, "criteria": [{"id": 1, "content": "x", "weight": -2}]}

    detail = client.post("/evaluate", json=bad).json()["detail"]

    assert [item["loc"] for item in detail] == [["body", "criteria", 0, "weight"]]
    assert all("input" not in item for item in detail)


def test_the_result_carries_every_published_field(client):
    """The result shape is a published interface — it may grow, never shrink."""
    use_judge(FakeJudge({1: 2, 2: 1}))

    body = client.post("/evaluate", json=CASE).json()

    assert set(body) == {"case_id", "score", "criterion_results"}
    assert set(body["criterion_results"][0]) == {
        "criterion_id", "weight", "score", "is_present", "spread", "failed", "reasoning",
    }
    assert body["criterion_results"][0]["weight"] == 3
    assert body["criterion_results"][0]["spread"] == 0.0


def test_health_answers_even_when_the_judge_is_unconfigured(unconfigured_client):
    """Liveness must not depend on the judge, or a missing key takes the container down."""
    assert unconfigured_client.get("/health").status_code == 200


def test_evaluate_fails_loudly_when_the_judge_is_unconfigured(unconfigured_client):
    """No silent fallback and no fake score: a missing key is a server fault, not a 0.0."""
    assert unconfigured_client.post("/evaluate", json=CASE).status_code == 500


# --- POST /evaluate/batch -----------------------------------------------------------------


def test_evaluate_batch_serves_every_case_and_the_run_metrics(client):
    use_judge(FakeJudge(BATCH_VERDICTS))

    body = client.post("/evaluate/batch", json=BATCH).json()

    assert [case["case_id"] for case in body["case_results"]] == [1, 2, 3]
    assert body["metrics"]["average_score"] == pytest.approx((0.75 + 0.5 + 0.0) / 3)


def test_the_batch_result_carries_every_published_field(client):
    """The result shape is a published interface — it may grow, never shrink."""
    use_judge(FakeJudge(BATCH_VERDICTS))

    body = client.post("/evaluate/batch", json=BATCH).json()

    assert set(body) == {"metrics", "case_results"}
    assert set(body["metrics"]) == {
        "total_cases", "average_score", "median_score", "variance", "standard_deviation",
        "average_criterion_score", "criteria_fulfillment_rate", "cases_with_score_zero",
        "cases_with_score_zero_count", "weakest_cases_above_zero", "failed_criteria_count",
    }
    assert set(body["case_results"][0]) == {"case_id", "score", "criterion_results"}


def test_a_batch_entry_is_serialized_exactly_like_a_single_evaluation(client):
    """Both endpoints return a `CaseResult`, so the documents must be identical — not merely
    similar. That is what lets a caller treat the two results interchangeably."""
    use_judge(FakeJudge(BATCH_VERDICTS))

    batched = client.post("/evaluate/batch", json=BATCH).json()["case_results"][0]
    alone = client.post("/evaluate", json=CASE).json()

    assert batched == alone


def test_a_case_without_an_id_is_rejected(client):
    """`Case.id` is mandatory on both paths, so one result shape serves both: every result
    carries a meaningful `case_id` and nothing has to be reconciled later."""
    use_judge(FakeJudge({}))
    without_id = {key: value for key, value in CASE.items() if key != "id"}

    assert client.post("/evaluate", json=without_id).status_code == 422


def test_an_empty_batch_is_rejected(client):
    use_judge(FakeJudge({}))
    assert client.post("/evaluate/batch", json={"cases": []}).status_code == 422


def test_duplicate_case_ids_are_rejected(client):
    use_judge(FakeJudge({}))
    duplicated = {"cases": [{**BATCH["cases"][0], "id": 5}, {**BATCH["cases"][1], "id": 5}]}

    assert client.post("/evaluate/batch", json=duplicated).status_code == 422


def test_one_invalid_case_rejects_the_whole_batch_before_any_call(client):
    """Validation stays at the boundary: a batch is not partially judged and partially
    refused, so nothing is paid for a run whose result would be incomplete anyway."""
    use_judge(FakeJudge(BATCH_VERDICTS))
    broken = {"cases": [BATCH["cases"][0], {**BATCH["cases"][1], "criteria": []}]}

    response = client.post("/evaluate/batch", json=broken)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "cases", 1, "criteria"]


# --- end to end: the real SDK against a stub OpenAI-compatible endpoint ---------------------
#
# Every other judge test replaces `judge.client.chat.completions`, so nothing checks the one
# thing the whole tool rests on: that an unmodified `OpenAIJudge` talks to an OpenAI-compatible
# endpoint correctly. These go through the full chain — HTTP request, `JudgeConfig.from_env`,
# the openai SDK, a real socket, the parser, the weighted fold, HTTP response.

EMAIL_CRITERION = CASE["criteria"][0]["content"]
LAST_DAY_CRITERION = CASE["criteria"][1]["content"]


class StubJudgeEndpoint(BaseHTTPRequestHandler):
    """A minimal OpenAI-compatible endpoint: answers each chat completion from a script and
    records the request bodies it received."""

    replies: dict[str, list[str]] = {}
    """Criterion text -> its replies, in order. Keyed by criterion, not a single queue: the
    criteria are judged concurrently, so a queue would hand out replies in a racy order."""

    received: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers["content-length"])
        request = json.loads(self.rfile.read(length))
        type(self).received.append(request)
        self._respond(self._next_reply_for(request))

    def _next_reply_for(self, request: dict) -> str:
        asked = request["messages"][1]["content"]
        for criterion, replies in type(self).replies.items():
            if criterion in asked:
                return replies.pop(0)
        raise AssertionError(f"the stub was not scripted for: {asked}")

    def _respond(self, reply: str) -> None:
        payload = json.dumps(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 0,
                "model": "stub-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        """Silence the default stderr access log."""


@pytest.fixture
def stub_endpoint(monkeypatch):
    """Serves the stub on a free port and points the judge environment at it."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubJudgeEndpoint)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    StubJudgeEndpoint.replies, StubJudgeEndpoint.received = {}, []

    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_ENDPOINT", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_API_KEY", "stub-key")
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_MODEL", "stub-model")
    get_judge.cache_clear()

    yield StubJudgeEndpoint

    server.shutdown()
    get_judge.cache_clear()


def test_evaluate_end_to_end_against_an_openai_compatible_endpoint(client, stub_endpoint):
    stub_endpoint.replies = {
        EMAIL_CRITERION: ['Named literally.\n{"score": 2}'],
        LAST_DAY_CRITERION: ['Never stated.\n{"score": 0}'],
    }

    body = client.post("/evaluate", json=CASE).json()

    assert body["score"] == pytest.approx(0.75)  # (3*2/2 + 1*0/2) / 4
    assert body["criterion_results"][0]["reasoning"] == "Named literally."
    assert body["criterion_results"][1]["reasoning"] == "Never stated."


def test_the_judge_is_asked_exactly_as_configured(client, stub_endpoint):
    """What actually goes on the wire: the configured model and sampling settings, the system
    prompt, and one user message carrying question, answer and the single criterion."""
    stub_endpoint.replies = {EMAIL_CRITERION: ['Covered.\n{"score": 2}']}

    client.post("/evaluate", json={**CASE, "criteria": [CASE["criteria"][0]]})

    asked = stub_endpoint.received[0]
    assert asked["model"] == "stub-model"
    assert asked["temperature"] == 0.0
    assert asked["max_completion_tokens"] == 768
    assert [message["role"] for message in asked["messages"]] == ["system", "user"]
    assert "0-2 scale" in asked["messages"][0]["content"]
    assert CASE["question"] in asked["messages"][1]["content"]
    assert CASE["answer"] in asked["messages"][1]["content"]
    assert EMAIL_CRITERION in asked["messages"][1]["content"]
    assert LAST_DAY_CRITERION not in asked["messages"][1]["content"]


def test_a_broken_reply_is_healed_over_the_wire(client, stub_endpoint):
    """One criterion, two round trips: the complaint really does travel back to the model."""
    stub_endpoint.replies = {
        EMAIL_CRITERION: ["I would say it is fine.", 'On reflection.\n{"score": 1}']
    }

    body = client.post("/evaluate", json={**CASE, "criteria": [CASE["criteria"][0]]}).json()

    assert body["score"] == pytest.approx(0.5)
    assert body["criterion_results"][0]["failed"] is False
    assert len(stub_endpoint.received) == 2
    assert "no JSON object" in stub_endpoint.received[1]["messages"][-1]["content"]


def test_an_unhealable_endpoint_costs_only_its_own_criterion(client, stub_endpoint):
    """An unhealable judge costs one criterion, not the case: the other one is still scored
    and the case score drops visibly instead of the request failing."""
    stub_endpoint.replies = {
        EMAIL_CRITERION: ['Covered.\n{"score": 2}'],
        LAST_DAY_CRITERION: ["no json"] * 3,
    }

    body = client.post("/evaluate", json=CASE).json()

    scored, failed = body["criterion_results"]
    assert scored["score"] == 2.0
    assert failed["failed"] is True
    assert "no valid answer in 3 attempts" in failed["reasoning"]
    assert body["score"] == pytest.approx(0.75)  # (3*2/2 + 1*0/2) / 4


def test_one_judge_serves_the_whole_process_so_its_limit_is_shared(stub_endpoint):
    """`OpenAIJudge` carries the concurrency limit, so it only bounds anything if every
    request shares one instance — that is what caching `get_judge` is for."""
    assert get_judge() is get_judge()


def test_a_rubric_larger_than_the_concurrency_limit_is_scored_completely(
    client, stub_endpoint, monkeypatch
):
    """Twenty criteria through two slots: queueing must lose no verdict, mix up no reply and
    deadlock nowhere. The limit only bounds the connections, never the result."""
    monkeypatch.setenv("RUBRIC_EVAL_JUDGE_MAX_CONCURRENT", "2")
    get_judge.cache_clear()
    criteria = [
        {"id": number, "content": f"criterion {number:02d}", "weight": 1}
        for number in range(1, 21)
    ]
    stub_endpoint.replies = {
        criterion["content"]: [f'Covered by {criterion["content"]}.\n{{"score": 2}}']
        for criterion in criteria
    }

    body = client.post("/evaluate", json={**CASE, "criteria": criteria}).json()

    assert body["score"] == 1.0
    assert len(stub_endpoint.received) == 20
    assert [result["criterion_id"] for result in body["criterion_results"]] == list(range(1, 21))
    assert body["criterion_results"][7]["reasoning"] == "Covered by criterion 08."


def test_two_requests_in_two_event_loops_score_a_large_rubric_identically(stub_endpoint):
    """The regression that started this: one cached judge, two `TestClient` blocks — two
    event loops — and a rubric larger than the concurrency limit.

    With one semaphore for the judge's lifetime, the second round bound none of its queued
    criteria to a usable loop: twelve of twenty came back `failed`, each carrying
    "…is bound to a different event loop" as its judge reasoning, and the case score dropped
    from 1.0 to 0.4 while the response stayed `200 OK`. A silently wrong score is the worst
    possible outcome for an evaluator, so this asserts both rounds are identical.

    Deliberately not using the `client` fixture: the two clients have to be built here, in
    one test, sharing one cached judge — that is the whole scenario.
    """
    criteria = [
        {"id": number, "content": f"criterion {number:02d}", "weight": 1}
        for number in range(1, 21)
    ]
    stub_endpoint.replies = {
        criterion["content"]: [f'Covered.\n{{"score": 2}}'] * 2  # asked once per round
        for criterion in criteria
    }
    case = {**CASE, "criteria": criteria}

    rounds = []
    for _ in range(2):
        with TestClient(app) as client:
            rounds.append(client.post("/evaluate", json=case).json())

    assert [body["score"] for body in rounds] == [1.0, 1.0]
    verdicts = [verdict for body in rounds for verdict in body["criterion_results"]]
    assert [verdict["failed"] for verdict in verdicts] == [False] * 40
