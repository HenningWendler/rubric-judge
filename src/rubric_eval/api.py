"""Thin HTTP wrapper around the library: three endpoints and the judge wiring, no domain logic.

Stateless: no catalog, no run ids, no persistence. Everything that decides *what* a score
means lives in `evaluation.py` and below.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rubric_eval.evaluation import evaluate_batch, evaluate_case
from rubric_eval.judge import Judge, JudgeConfig, OpenAIJudge
from rubric_eval.models import Batch, BatchResult, Case, CaseResult

app = FastAPI(title="rubric-eval", version="0.1.0")


@lru_cache
def get_judge() -> Judge:
    """The one judge this process uses, built on the first request that needs it.

    Cached rather than built per request for two reasons: importing this module must not
    read the environment (or a test could never import it), and `max_concurrent` is a
    property of the *instance* — a judge per request would give each request its own full
    set of slots instead of sharing one budget.

    Returns:
        The process-wide `OpenAIJudge`, configured from the environment.

    Raises:
        RuntimeError: The judge cannot be configured; the message names every missing
            variable. Surfaces as a 500, which is correct — an unconfigured service is a
            server fault, not a bad request, and must never fall back to a fake score.
    """
    return OpenAIJudge(JudgeConfig.from_env())


@app.exception_handler(RequestValidationError)
async def report_invalid_request(_: Request, error: RequestValidationError) -> JSONResponse:
    """Report *what* was wrong, never the value that was sent.

    FastAPI's own handler echoes the offending input back, which cannot always be serialized:
    `Infinity` and `NaN` are literals Python's JSON parser accepts but its writer refuses, so
    rendering the 422 for a rejected `"weight": Infinity` would itself fail and turn a clean
    client error into a 500. Keeping only location, message and type also stops the API from
    mirroring arbitrary request content back to the caller.
    """
    reportable = [
        {"loc": item["loc"], "msg": item["msg"], "type": item["type"]} for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": reportable})


@app.get("/health", summary="Liveness and readiness probe")
async def health() -> dict[str, str]:
    """Readiness probe for container orchestration.

    Answers `{"status": "ok"}` even when the judge is unconfigured, so a missing API key
    never takes the container down — `POST /evaluate` is what fails then, loudly, with a
    500. Liveness must not depend on a third-party endpoint.
    """
    return {"status": "ok"}


@app.post("/evaluate", summary="Score one answer against its rubric")
async def evaluate(case: Case, judge: Annotated[Judge, Depends(get_judge)]) -> CaseResult:
    """Score one answer against its rubric — one LLM call per criterion, run in parallel.

    Send a `Case`: your `id`, the `question` that was asked, the `answer` your system
    produced, and the `criteria` a good answer has to satisfy, each with a positive weight.

    Returns a `CaseResult`: your id echoed as `case_id`, one `criterion_results` entry per
    criterion (score 0/1/2, whether it counts as present, and the judge's reasoning), and
    `score` — the weighted fold of all of them, normalized to 0..1.

    **422** if the body is invalid — an empty rubric, duplicate criterion ids, a blank
    `content`, or a weight that is not positive and finite. Validation happens before the
    first LLM call, so a rejected request costs nothing.

    **500** only if the judge is unconfigured or the program is broken. An endpoint that is
    merely *down* does not fail the request: each criterion it could not answer for comes
    back with `failed: true` and `score: 0`, which lowers the case score visibly.
    """
    return await evaluate_case(judge, case)


@app.post("/evaluate/batch", summary="Score a catalog of answers and aggregate the run")
async def evaluate_many(batch: Batch, judge: Annotated[Judge, Depends(get_judge)]) -> BatchResult:
    """Score a whole catalog of answers in one request and get metrics over the run.

    Send a `Batch`: a list of exactly the cases `POST /evaluate` takes, with unique ids.

    Returns a `BatchResult`: `case_results` holds one entry per case — each one **the same
    document** `POST /evaluate` returns for that case — and `metrics` aggregates them:
    average, median, spread, criteria fulfillment, which cases scored zero, the weakest
    cases above zero, and how many criteria the judge failed to answer.

    Synchronous: the response arrives when the last case is done. Sizing the request is
    therefore yours to do — the whole catalog is one HTTP timeout.

    **422** on the single-case rules, plus an empty `cases` or duplicate case ids. One
    invalid case rejects the whole batch: a run that is partly judged and partly refused
    would produce metrics nobody can compare.

    Concurrency is bounded by the judge, not by the batch: every case of this request shares
    one budget, and so does every other request in flight.
    """
    return await evaluate_batch(judge, batch)
