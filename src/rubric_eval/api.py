"""Thin HTTP wrapper around the library: four endpoints and the judge wiring, no domain logic.

Stateless: no catalog, no run ids, no persistence. Everything that decides *what* a score
means lives in `evaluation.py` and below.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rubric_eval import comparison, evaluation
from rubric_eval.judge import Judge, JudgeConfig, OpenAIJudge
from rubric_eval.models import (
    Batch,
    BatchResult,
    Case,
    CaseResult,
    ComparisonResult,
    RunPair,
)

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
async def evaluate_case(case: Case, judge: Annotated[Judge, Depends(get_judge)]) -> CaseResult:
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
    return await evaluation.evaluate_case(judge, case)


@app.post("/evaluate/batch", summary="Score a catalog of answers and aggregate the run")
async def evaluate_batch(batch: Batch, judge: Annotated[Judge, Depends(get_judge)]) -> BatchResult:
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
    return await evaluation.evaluate_batch(judge, batch)


@app.post("/compare", summary="Hold two finished runs against each other")
async def compare_runs(runs: RunPair) -> ComparisonResult:
    """Compare two runs of the same catalog — did your change help, where, and what did it cost.

    Send a `RunPair`: the `baseline` run to compare against and the `candidate` run under
    test, each one exactly the `BatchResult` document `POST /evaluate/batch` returned. Runs
    stored as JSON months apart compare just like runs produced a second ago.

    Returns a `ComparisonResult` at three grains. `metrics_delta` holds `candidate - baseline`
    for every run metric, so a positive number always means the candidate did better — except
    for the two counting fields, where fewer is better. `summary` says how that is distributed:
    which cases improved, stayed, or got worse, biggest movers first, and how large the moves
    were on each side. `case_comparison_results` goes down to the individual criterion.

    Read `metrics_delta.failed_criteria_count_delta` first. Anything but 0 means the two runs
    suffered different amounts of judge outage, and every other number is then partly an
    artefact of that rather than of the answers.

    **422** if a body is invalid, or if the two runs are not comparable — different case ids,
    different criteria within a case, or different weights. The message names every difference
    at once, so one fix can address all of them. Comparing runs with different weights is
    refused rather than approximated: the weights are the denominator each case score is
    normalized by, so scores computed under different ones do not subtract.

    No judge is involved: this endpoint is pure computation and answers correctly even when
    the service has no API key configured.
    """
    try:
        return comparison.compare_runs(runs)
    except comparison.RunsNotComparableError as incomparable:
        raise HTTPException(status_code=422, detail=str(incomparable)) from incomparable
