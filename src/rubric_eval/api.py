"""Thin HTTP wrapper around the library: two endpoints and the judge wiring, no domain logic.

Stateless: no catalog, no run ids, no persistence. Everything that decides *what* a score
means lives in `evaluation.py` and below.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rubric_eval.evaluation import evaluate_case
from rubric_eval.judge import Judge, JudgeConfig, OpenAIJudge
from rubric_eval.models import EvaluateRequest, EvaluationResult

app = FastAPI(title="rubric-eval", version="0.1.0")


@lru_cache
def get_judge() -> Judge:
    """Built once on the first request, so importing this module never reads the environment.
    Tests replace it with `app.dependency_overrides[get_judge]`."""
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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/evaluate")
async def evaluate(
    request: EvaluateRequest, judge: Annotated[Judge, Depends(get_judge)]
) -> EvaluationResult:
    return await evaluate_case(judge, request)
