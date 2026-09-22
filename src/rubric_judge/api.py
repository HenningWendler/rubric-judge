"""Thin HTTP wrapper around the library: four endpoints and the judge wiring, no domain logic.

Stateless: no catalog, no run ids, no persistence. Everything that decides *what* a score
means lives in `evaluation.py` and below.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rubric_judge import comparison, evaluation
from rubric_judge.judge import Judge, JudgeConfig, JudgeUnavailableError, OpenAIJudge
from rubric_judge.models import (
    Case,
    CaseResult,
    Run,
    RunComparison,
    RunComparisonResult,
    RunResult,
)

app = FastAPI(title="rubric-judge", version="0.1.0")


@lru_cache
def get_judge() -> Judge:
    """The one judge this process uses, built on the first request that needs it.

    Cached rather than built per request for two reasons: importing this module must not
    read the environment (or a test could never import it), and `max_concurrent` is a
    property of the *instance* — a judge per request would give each request its own full
    set of slots instead of sharing one budget.

    Returns:
        The process-wide `OpenAIJudge`, configured from the environment and grading on
        `DEFAULT_SCALE`. The scale is not an environment setting on purpose: it carries a
        sentence per grade, and prose does not belong in a variable meant for a URL or a key.
        A service that has to grade differently builds its own judge and overrides this
        dependency.

    Raises:
        RuntimeError: The judge cannot be configured; the message names every missing
            variable. Surfaces as a 500, which is correct — an unconfigured service is a
            server fault, not a bad request, and must never fall back to a fake score. A
            judge that is configured but unreachable is a 503 instead, raised per request.

    Example:
        app.dependency_overrides[get_judge] = lambda: my_own_judge   # grade differently
        get_judge.cache_clear()   # after changing RUBRIC_JUDGE_* in this process
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

    The refused request itself is not read — nothing the caller sent goes back out.

    Args:
        error: The validation failure FastAPI raised. Only where, what and which kind is
            taken from each of its entries; the offending value is dropped.

    Returns:
        A 422 whose `detail` holds one `{"loc", "msg", "type"}` object per rejected field, in
        the order FastAPI reports them. Never empty — FastAPI raises this for at least one
        failure.

    Example:
        httpx.post("http://localhost:8000/evaluate", json={"id": 1}).json()["detail"][0]
        # {"loc": ["body", "answer"], "msg": "Field required", "type": "missing"}
    """
    reportable = [
        {"loc": item["loc"], "msg": item["msg"], "type": item["type"]} for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": reportable})


@app.exception_handler(JudgeUnavailableError)
async def report_unavailable_judge(_: Request, error: JudgeUnavailableError) -> JSONResponse:
    """Answer 503 for a run the judge could not finish, never a partial 200.

    A criterion nobody graded has no score, and the 0 that would stand in for it turns the
    whole run into a plausible number no reader can tell from a real result. So the run is
    dropped rather than patched, and the caller is told to try again when the judge is back.

    The unanswerable request itself is not read; nothing here is stored, so there is nothing
    to hand back but the cause.

    Args:
        error: What the judge gave up with, after every attempt its configuration allowed.

    Returns:
        A 503 whose `detail` is that failure's own message, which names the last cause. No
        partial result travels with it: there is nothing to hand back that a caller could
        mistake for a finished evaluation.

    Example:
        httpx.post("http://localhost:8000/evaluate", json=case).json()["detail"]
        # "Judge gave no usable answer in 3 attempts: the endpoint is down"
    """
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.get("/health", summary="Liveness and readiness probe")
async def health() -> dict[str, str]:
    """Readiness probe for container orchestration.

    Answers `{"status": "ok"}` even when the judge is unconfigured or its endpoint is down,
    so neither takes the container down — `POST /evaluate` is what fails then, loudly, with
    a 500 or a 503. Liveness must not depend on a third-party endpoint.

    Returns:
        `{"status": "ok"}`, always and with a 200. There is no other body and no other
        status: an answer at all is the whole signal.

    Example:
        httpx.get("http://localhost:8000/health").json()   # {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/evaluate", summary="Score one answer against its rubric")
async def evaluate_case(case: Case, judge: Annotated[Judge, Depends(get_judge)]) -> CaseResult:
    """Score one answer against its rubric — one LLM call per criterion, run in parallel.

    Send a `Case`: your `id`, the `answer` your system produced, and the `criteria` a good
    answer has to satisfy, each with a positive weight. If the answer was produced in
    response to something, put that in `context` and say what it is, for example
    `"The question asked was: ..."`; the judge reads it as background and never scores it.
    Leave `context` out for text that stands on its own, such as a summary or a report, and
    the answer is judged against the criteria alone.

    Returns a `CaseResult`: your id echoed as `case_id`, one `criterion_results` entry per
    criterion, and `score` — the weighted fold of all of them, normalized to 0..1.

    The case carries the `scale` its grades were given on — the maximum, the threshold
    `is_present` is cut at, and what each grade means in words — so a stored result still
    explains itself months later. By default that is 0..2: 2 fully covered, 1 partially,
    0 not covered.

    Each `criterion_results` entry holds the judge's **raw** grade on that scale, whether it
    counts as present, and the judge's reasoning. Read a grade against the scale; `score` is
    normalized precisely so that it can be read without one.

    Any `labels` you send come back untouched on the result. They are never shown to the
    judge and cannot move a score — they exist to slice a run (see `POST /evaluate/run`).
    A requirement the answer must actually meet belongs in `criteria`.

    **422** if the body is invalid — an empty rubric, duplicate criterion ids, a blank
    `content`, or a weight that is not positive and finite. Validation happens before the
    first LLM call, so a rejected request costs nothing.

    **503** if the judge could not answer for a criterion — its endpoint refused, timed out,
    ran out of quota, or never replied usably within `RUBRIC_JUDGE_MAX_ATTEMPTS`
    attempts. You get no result at all then, on purpose: a criterion nobody graded would have
    to be scored `0`, and a run with one invented `0` in it is a plausible number you could
    not tell from a real one. Retry the request once the judge is reachable again.

    **500** if the judge is unconfigured or the program is broken.

    Args:
        case: The case to score: `id`, `answer`, at least one `criteria` entry with a
            unique `id` and a positive finite `weight`, and optional `context` and `labels`.
            A blank criterion, a repeated criterion id, a weight of 0 and a blank `context`
            are each a 422.
        judge: The process-wide judge, injected rather than sent — no request can choose the
            model it is graded by.

    Returns:
        The `CaseResult` for that case: `case_id`, the weighted `score` in 0..1, the `scale`
        the grades are on, one `criterion_results` entry per criterion, and the `labels` you
        sent. A score of 0.0 means the answer missed every criterion, not that nothing was
        judged — a case nobody could grade is a 503 instead.

    Example:
        httpx.post("http://localhost:8000/evaluate", json={
            "id": 1,
            "context": "The question asked was: How do I report sick leave?",
            "answer": "Email hr@example.com before 10:00.",
            "criteria": [{"id": 1, "content": "Report by email before 10:00", "weight": 3}],
        }).json()["score"]   # 1.0

        httpx.post("http://localhost:8000/evaluate", json={
            "id": 2,
            "answer": "The Cologne office has an underground garage.",
            "criteria": [{"id": 1, "content": "Names an office with a garage", "weight": 1}],
        }).json()["score"]   # 1.0
    """
    return await evaluation.evaluate_case(judge, case)


@app.post("/evaluate/run", summary="Score a catalog of answers and aggregate the run")
async def evaluate_run(run: Run, judge: Annotated[Judge, Depends(get_judge)]) -> RunResult:
    """Score a whole catalog of answers in one request and get metrics over the run.

    Send a `Run`: a list of exactly the cases `POST /evaluate` takes, with unique ids.

    Returns a `RunResult`: `case_results` holds one entry per case — each one **the same
    document** `POST /evaluate` returns for that case — and `metrics` aggregates them:
    average, median, spread, criteria fulfillment, which cases scored zero and the weakest
    cases above zero.

    **Labels.** Tag your cases (`"labels": ["table"]`) and `label_metrics` reports that whole
    set of numbers again for each label on its own — the breakdown that tells a generally
    mediocre system apart from one that is fine except on tables. A case counts in every
    bucket it carries a label for, so the buckets overlap.

    **Running only part of a catalog.** Send the whole thing plus a `label_filter`, an **OR
    of ANDs**: `[["table", "split_infos"], ["agentic"]]` runs the cases carrying both `table`
    and `split_infos`, plus the cases carrying `agentic`. One group is a plain AND, several
    one-label groups are a plain OR, and an absent `label_filter` runs everything. It comes
    back as `applied_label_filter`, so a stored run still says which subset it is.

    Synchronous: the response arrives when the last selected case is done. Sizing the request
    is therefore yours to do — the whole catalog is one HTTP timeout, whether or not a
    `label_filter` narrows what is judged.

    **422** on the single-case rules, plus an empty `cases` or duplicate case ids. One
    invalid case rejects the whole run: a run that is partly judged and partly refused
    would produce metrics nobody can compare. Also **422** when `label_filter` matches no
    case at all — the message lists the labels your run does carry, with counts, because
    that is nearly always a typo.

    **503** if the judge could not answer for a single criterion of a single case — the whole
    run is dropped, not the one case. The metrics average the cases against each other, so a
    run with one fabricated `0` in it reports a number you could not tell from a real one.
    Nothing is stored here, so a retry costs only the judge calls.

    Concurrency is bounded by the judge, not by the run: every case of this request shares
    one budget, and so does every other request in flight.

    Args:
        run: The catalog to score: `cases`, at least one and with unique ids, each exactly
            the body `POST /evaluate` takes, plus an optional `label_filter`. An empty
            `cases`, a repeated case id and a `label_filter` matching nothing are each a 422.
        judge: The process-wide judge, injected rather than sent — every case of every
            request is graded by the same one.

    Returns:
        The `RunResult`: `metrics` over the selected cases, `label_metrics` once per label
        those cases carry (empty when none do), `applied_label_filter` recording what picked
        them, and one `case_results` entry per selected case in request order. Every selected
        case is in it, or the request answered 503 instead.

    Example:
        httpx.post("http://localhost:8000/evaluate/run", json={
            "cases": [
                {
                    "id": 1,
                    "context": "The question asked was: How do I report sick leave?",
                    "answer": "Email hr@example.com before 10:00.",
                    "criteria": [{"id": 1, "content": "Report by email", "weight": 3}],
                    "labels": ["table"],
                }
            ]
        }).json()["metrics"]["average_score"]   # 1.0
    """
    return await evaluation.evaluate_run(judge, run)


@app.post("/compare", summary="Hold two finished runs against each other")
async def compare_runs(run_comparison: RunComparison) -> RunComparisonResult:
    """Compare two runs of the same catalog — did your change help, where, and what did it cost.

    Send a `RunComparison`: the `baseline` run to compare against and the `candidate` run
    under test, each one exactly the `RunResult` document `POST /evaluate/run` returned. Runs
    stored as JSON months apart compare just like runs produced a second ago.

    Returns a `RunComparisonResult` at three grains. `metrics_delta` holds
    `candidate - baseline` for every run metric, so a positive number always means the candidate did better — except
    for the two counting fields, where fewer is better. `summary` says how that is distributed:
    which cases improved, stayed, or got worse, biggest movers first, and how large the moves
    were on each side — `largest`, `mean` and `median` are `null` for a side nothing moved to,
    so "nothing got worse" never reads as "everything got worse by 0.0".
    `case_comparison_results` goes down to the individual criterion.

    `label_metrics_deltas` repeats `metrics_delta` for each label the cases carry, which is
    what says whether an average that rose did so by fixing one kind of case or by lifting
    all of them. `applied_label_filter` is not compared: two runs covering the same case ids
    are comparable however each was selected.

    Every `case_results` entry has to name the `scale` it was judged on; a run that dropped
    the field is refused rather than read as the bundled `0–2`, because a 0–10 run silently
    reinterpreted that way would subtract cleanly from a real `0–2` one and answer `200`.

    **422** if a body is invalid, or if the two runs are not comparable — a different grading
    scale, different case ids, different criteria within a case, different weights, or a case
    whose labels changed between the runs (which would put different cases in the two buckets
    of the same name). The message names every difference at once, so one fix can address all
    of them. Comparing
    runs with different weights or scales is refused rather than approximated: the weights are
    the denominator each case score is normalized by and the scale is the unit every raw
    criterion score is in, so numbers computed under different ones do not subtract.

    No judge is involved: this endpoint is pure computation and answers correctly even when
    the service has no API key configured.

    Args:
        run_comparison: The two runs to hold against each other — `baseline` and `candidate`,
            each exactly the `RunResult` document `POST /evaluate/run` returned. Two runs
            that do not describe the same catalog, on the same scale, with the same weights
            and labels, are a 422.

    Returns:
        The `RunComparisonResult`: `metrics_delta`, `summary`, `label_metrics_deltas` (empty
        when neither run carries labels) and `case_comparison_results` ordered by `case_id`.
        Every delta is `candidate - baseline`, and a delta of 0.0 means the two runs really
        landed on the same number.

    Raises:
        HTTPException: 422, when the two runs are not comparable. Its `detail` names every
            difference found at once, so one fix can address all of them.

    Example:
        httpx.post(
            "http://localhost:8000/compare",
            json={"baseline": baseline_run, "candidate": candidate_run},
        ).json()["metrics_delta"]["average_score_delta"]   # 0.5
    """
    try:
        return comparison.compare_runs(run_comparison)
    except comparison.RunsNotComparableError as incomparable:
        raise HTTPException(status_code=422, detail=str(incomparable)) from incomparable
