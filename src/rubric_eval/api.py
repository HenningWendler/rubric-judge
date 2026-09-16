"""Thin HTTP wrapper around the library: four endpoints and the judge wiring, no domain logic.

Stateless: no catalog, no run ids, no persistence. Everything that decides *what* a score
means lives in `evaluation.py` and below.
"""

from collections import Counter
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
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
    Labels,
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
        The process-wide `OpenAIJudge`, configured from the environment and grading on
        `DEFAULT_SCALE`. The scale is not an environment setting on purpose: it carries a
        sentence per grade, and prose does not belong in a variable meant for a URL or a key.
        A service that has to grade differently builds its own judge and overrides this
        dependency.

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
    criterion, and `score` — the weighted fold of all of them, normalized to 0..1.

    The case carries the `scale` its grades were given on — the maximum, the threshold
    `is_present` is cut at, and what each grade means in words — so a stored result still
    explains itself months later. By default that is 0..2: 2 fully covered, 1 partially,
    0 not covered.

    Each `criterion_results` entry holds the judge's **raw** grade on that scale, whether it
    counts as present, and the judge's reasoning. Read a grade against the scale; `score` is
    normalized precisely so that it can be read without one.

    Any `labels` you send come back untouched on the result. They are never shown to the
    judge and cannot move a score — they exist to slice a batch (see `POST /evaluate/batch`).
    A requirement the answer must actually meet belongs in `criteria`.

    **422** if the body is invalid — an empty rubric, duplicate criterion ids, a blank
    `content`, or a weight that is not positive and finite. Validation happens before the
    first LLM call, so a rejected request costs nothing.

    **500** only if the judge is unconfigured or the program is broken. An endpoint that is
    merely *down* does not fail the request: each criterion it could not answer for comes
    back with `failed: true` and `score: 0`, which lowers the case score visibly.
    """
    return await evaluation.evaluate_case(judge, case)


@app.post("/evaluate/batch", summary="Score a catalog of answers and aggregate the run")
async def evaluate_batch(
    batch: Batch,
    judge: Annotated[Judge, Depends(get_judge)],
    labels: Annotated[Labels | None, Query()] = None,
) -> BatchResult:
    """Score a whole catalog of answers in one request and get metrics over the run.

    Send a `Batch`: a list of exactly the cases `POST /evaluate` takes, with unique ids.

    Returns a `BatchResult`: `case_results` holds one entry per case — each one **the same
    document** `POST /evaluate` returns for that case — and `metrics` aggregates them:
    average, median, spread, criteria fulfillment, which cases scored zero, the weakest
    cases above zero, and how many criteria the judge failed to answer.

    **Labels.** Tag your cases (`"labels": ["table"]`) and `label_metrics` reports that whole
    set of numbers again for each label on its own — the breakdown that tells a generally
    mediocre system apart from one that is fine except on tables. A case counts in every
    bucket it carries a label for, so the buckets overlap.

    Add `?labels=table&labels=images` to run only the cases carrying **all** of the named
    labels; repeat the parameter per label. The filter is echoed back as `label_filter`, so a
    stored run says which subset it is. Filtering here does not save you the upload — the
    whole batch travels either way, so for a big catalog prefer posting just the cases you
    want.

    Synchronous: the response arrives when the last case is done. Sizing the request is
    therefore yours to do — the whole catalog is one HTTP timeout.

    **422** on the single-case rules, plus an empty `cases` or duplicate case ids. One
    invalid case rejects the whole batch: a run that is partly judged and partly refused
    would produce metrics nobody can compare. Also **422** when `?labels=` matches no case —
    the message lists the labels your batch does carry, with counts — when a `label_filter`
    in the body contradicts the query parameter, and when a `label_filter` in the body names
    a label some case does not carry.

    Concurrency is bounded by the judge, not by the batch: every case of this request shares
    one budget, and so does every other request in flight.
    """
    return await evaluation.evaluate_batch(judge, _batch_selected_by(batch, labels or []))


def _batch_selected_by(batch: Batch, labels: list[str]) -> Batch:
    """The query parameter applied: the subset to actually run, recording what selected it.

    Refuses rather than picks a winner when the body already claims a different
    `label_filter` — a request carrying two disagreeing filters is a caller who has lost
    track of what they are sending, and silently honouring one would store a run whose
    recorded provenance contradicts the request that produced it.
    """
    if not labels:
        return batch
    if batch.label_filter and set(batch.label_filter) != set(labels):
        raise HTTPException(
            status_code=422,
            detail=(
                f"query labels {sorted(labels)} contradict the body's label_filter "
                f"{sorted(batch.label_filter)}; send one or the other"
            ),
        )
    cases = evaluation.filter_cases_by_labels(batch.cases, labels)
    if not cases:
        raise HTTPException(status_code=422, detail=_no_case_carries(labels, batch.cases))
    return Batch(cases=cases, label_filter=labels)


def _no_case_carries(labels: list[str], cases: list[Case]) -> str:
    """Name the labels the batch *does* carry, with counts. A filter that matches nothing is
    a typo far more often than an genuinely empty subset, and "table" is unguessable from
    "no cases matched" alone."""
    present = Counter(label for case in cases for label in case.labels)
    carried = ", ".join(f"{label} ({count})" for label, count in sorted(present.items()))
    return (
        f"no case carries all of {sorted(labels)}; "
        f"labels present in this batch: {carried or 'none'}"
    )


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

    `label_metrics_deltas` repeats `metrics_delta` for each label the cases carry, which is
    what says whether an average that rose did so by fixing one kind of case or by lifting
    all of them. `label_filter` is not compared: two runs covering the same case ids are
    comparable however each was selected.

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
    """
    try:
        return comparison.compare_runs(runs)
    except comparison.RunsNotComparableError as incomparable:
        raise HTTPException(status_code=422, detail=str(incomparable)) from incomparable
