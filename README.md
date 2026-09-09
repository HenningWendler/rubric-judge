# rubric-eval

Judge the answers of an LLM application against weighted reference criteria.
You bring the question, the answer and the rubric — an LLM-as-a-judge returns a
score per criterion on a 0–2 scale, its reasoning, and one normalized score for
the whole case.

Pure evaluator: no RAG, no retrieval, no answer generation. Answers come in ready.

## Install

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

## Configure

Secrets live in the environment, never in a committed file.
Copy [.env.example](.env.example) to `.env` and fill it in:

| Variable | Meaning | Default | Valid |
|---|---|---|---|
| `RUBRIC_EVAL_JUDGE_ENDPOINT` | OpenAI-compatible base URL | required | non-empty |
| `RUBRIC_EVAL_JUDGE_API_KEY` | API key for that endpoint | required | non-empty |
| `RUBRIC_EVAL_JUDGE_MODEL` | judge model name | required | non-empty |
| `RUBRIC_EVAL_JUDGE_TEMPERATURE` | sampling temperature | `0.0` | `≥ 0` |
| `RUBRIC_EVAL_JUDGE_MAX_TOKENS` | max completion tokens per call | `768` | `≥ 1` |
| `RUBRIC_EVAL_JUDGE_MAX_ATTEMPTS` | attempts per criterion, **first try included** | `3` | `≥ 1` |
| `RUBRIC_EVAL_JUDGE_MAX_CONCURRENT` | judge calls in flight at once | `8` | `≥ 1` |

An empty value counts as missing. All missing or out-of-range variables are named in a
single error at the first request, so configuration is fixed in one pass instead of one
restart per mistake.

Any OpenAI-compatible endpoint works — OpenAI, vLLM, Azure, Ollama, Groq,
OpenRouter — because the official `openai` SDK talks to whatever `base_url` you give it.

## Run

```bash
.venv/bin/uvicorn rubric_eval.api:app --env-file .env --port 8000
```

Interactive API docs: [localhost:8000/docs](http://localhost:8000/docs).

### `POST /evaluate`

```json
{
  "question": "How do I report sick leave?",
  "answer": "Send an email to hr@example.com before 10:00 on your first day.",
  "criteria": [
    { "id": 1, "content": "Report by email before 10:00 on the first day", "weight": 3 },
    { "id": 2, "content": "State the expected last day of absence", "weight": 2 }
  ]
}
```

```json
{
  "score": 0.6,
  "criteria": [
    { "criterion_id": 1, "weight": 3.0, "score": 2.0, "is_present": true,
      "spread": 0.0, "failed": false, "reasoning": "The answer explicitly …" },
    { "criterion_id": 2, "weight": 2.0, "score": 0.0, "is_present": false,
      "spread": 0.0, "failed": false, "reasoning": "The expected last day is …" }
  ]
}
```

A request is rejected with `422` before any LLM call — and therefore before any cost — if

- `criteria` is empty, or two criteria share an `id` (results are matched by `id`),
- a `content` is empty or only whitespace,
- a `weight` is not a positive, finite number (`0`, `-2`, `Infinity`, `NaN`).

The error body names only where and what, never the value that was sent:

```json
{ "detail": [ { "loc": ["body", "criteria", 0, "weight"],
                "msg": "Input should be greater than 0", "type": "greater_than" } ] }
```

### `GET /health`

Readiness probe for container orchestration. Answers even when the judge is unconfigured,
so a missing key never takes the container down — `POST /evaluate` is what fails then,
loudly, with a `500`.

## How a case is scored

Per criterion the judge returns `s ∈ {0, 1, 2}`:

- **2** — fully covered, even if worded differently
- **1** — partially covered, essential parts missing but the idea is derivable
- **0** — not covered

The case score normalizes over the sum of the weights:

```
score = Σᵢ ( wᵢ · sᵢ / 2 ) / Σᵢ wᵢ   ∈ [0, 1]
```

Only the *ratios* of the weights matter — `3` and `1` score exactly like `30` and `10`.

`is_present` is derived from the score as `s ≥ 0.5`; the judge is never asked for it.
`spread` is the standard deviation across repeated runs of the same criterion and stays
`0.0` while each criterion is judged exactly once.

### Unparseable replies heal themselves

If the judge replies with no JSON, or with a score off the integral scale (`3`, `1.5`),
the concrete cause is fed back to it together with its own broken reply, and the call is
repeated up to `RUBRIC_EVAL_JUDGE_MAX_ATTEMPTS` times.

### A dead judge costs one criterion, never the case

A criterion the judge could not answer for is marked `failed: true`, counts as `s = 0`,
keeps its weight and stays in the denominator — an outage lowers the score visibly instead
of silently shrinking the rubric. The cause is reported in its `reasoning`.

Only endpoint failures are contained this way: refused connections, timeouts, quota errors,
unparseable replies. Errors that mean the program itself is wrong (`TypeError` and the
like, listed in `evaluation._BUGS_NOT_OUTAGES`) are re-raised and surface as a `500`,
because a criterion scored `0` over a bug is indistinguishable from a real result.

### Concurrency

The criteria of a case are judged concurrently, but at most
`RUBRIC_EVAL_JUDGE_MAX_CONCURRENT` calls are in flight — a rubric with 200 criteria costs
8 open connections, not 200. Raise it for a local vLLM or Ollama, lower it for a small
hosted tier.

The limit sits on the judge, not on the fan-out, so it also holds across concurrent
requests: one judge instance serves the whole process, and ten parallel `POST /evaluate`
calls share the same 8 slots instead of getting 8 each. A slot is held for one HTTP call
only — a criterion waiting to be retried does not occupy one.

## Code map

Six modules, each with one job. A request walks straight down through them:

| Step | File | Responsibility |
|---|---|---|
| 1 | [api.py](src/rubric_eval/api.py) | FastAPI endpoints: validate the body, inject the judge, hand back JSON. No domain logic |
| 2 | [evaluation.py](src/rubric_eval/evaluation.py) | `evaluate_case()` — one async task per criterion, failures contained, verdicts folded |
| 3 | [judge.py](src/rubric_eval/judge.py) | `Judge` protocol, OpenAI-compatible client, reply parsing, retries, throttle, env config |
| 4 | [prompt.py](src/rubric_eval/prompt.py) | every word the judge is told: system prompt, user prompt, retry complaints |
| — | [models.py](src/rubric_eval/models.py) | request/result shapes, the `0–2` scale, `CriterionResult.judged()` / `.unjudged()` |
| — | [metrics.py](src/rubric_eval/metrics.py) | `case_score()` — the weighted formula above, nothing else |

### Where does a new rule go?

The two result types draw the layer boundary and answer the question by themselves:

```python
Verdict:          score, reasoning                           # what the model replied
CriterionResult:  criterion_id, weight, score, is_present,    # what the system concluded
                  spread, failed, reasoning
```

`judge.py` speaks `Verdict` — prompts, parsing and retries are *its* business, and another
implementation may do all three differently. Everything that has to hold no matter who
judges — the `0–2` scale, the `is_present` threshold, the weighting, "a dead judge costs
one criterion" — lives outside it, or two judges would produce incomparable scores.

So nothing above `judge.py` knows what a chat completion is, nothing inside it knows what a
weight is, and `evaluation.py` is the only module that speaks both languages.

`prompt.py` splits wording from protocol one step further: it returns plain strings and
knows nothing about roles, message dicts or the SDK, so every sentence the model ever reads
is reviewable in one file by someone who does not read Python.

| In `prompt.py` | Sent as |
|---|---|
| `JUDGE_EN` | the system message |
| `criterion_prompt(question, answer, criterion)` | the user message |
| `NO_JSON_HINT`, `out_of_range_hint()`, `malformed_json_hint()` | the follow-up message on a retry |

The complaints are worded as instructions on purpose: the parser raises them as `ValueError`
messages and the retry loop hands that text straight back to the model, so the exception
message *is* the corrective prompt.

Model fields are documented with docstrings instead of comments:

```python
class Criterion(DocumentedModel):
    weight: float = Field(gt=0, allow_inf_nan=False)
    """How much this criterion counts next to the others. Only the ratios matter:
    weights 3 and 1 score exactly like 30 and 10."""
```

Pydantic's `use_attribute_docstrings` (set once on `DocumentedModel`) copies them into the
OpenAPI schema, so the same sentence serves IDE hover and the Swagger UI. One text, never two.

## Use as a library

The HTTP layer is optional — `evaluate_case()` is the same entry point `POST /evaluate`
uses, so a CLI or a notebook gets identical scores:

```python
from rubric_eval import Criterion, EvaluateRequest, JudgeConfig, OpenAIJudge, evaluate_case

judge = OpenAIJudge(JudgeConfig.from_env())
result = await evaluate_case(judge, EvaluateRequest(
    question="How do I report sick leave?",
    answer="Email hr@example.com before 10:00.",
    criteria=[Criterion(id=1, content="Report by email before 10:00", weight=3)],
))
print(result.score, result.criteria[0].reasoning)
```

One criterion at a time, without the fan-out or the failure handling:

```python
verdict = await judge.score(question, answer, Criterion(id=1, content="…", weight=3))
```

`Judge` is a Protocol — plug in your own client without touching the core:

```python
from pathlib import Path

judge = OpenAIJudge(JudgeConfig.from_env(), prompt=Path("my_prompt.txt").read_text())
```

Two things a custom judge is responsible for:

- **Throttling.** `evaluate_case()` hands out one task per criterion whatever the rubric's
  size, because only the implementation knows what its backend tolerates. `OpenAIJudge`
  bounds itself with `max_concurrent`; your judge needs its own bound.
- **Raising on failure.** Return a valid `Verdict` or raise — `evaluation.py` turns the
  exception into one `failed` criterion.

`prompt=...` replaces the system prompt only; the user prompt and the retry complaints are
edited in [prompt.py](src/rubric_eval/prompt.py). The bundled prompt is a Python module
rather than a `.txt` so it cannot go missing from a wheel or a container image; reading one
from disk stays the caller's decision.

## Test

```bash
.venv/bin/python -m pytest
```

No real LLM is ever called. The suite mocks at two levels:

- **`FakeJudge`** ([conftest.py](tests/conftest.py)) replaces the `Judge` protocol and scores
  from a lookup table — `{1: 2, 2: ValueError("down")}` scores criterion 1 with a `2` and lets
  the judge die on criterion 2. [test_evaluation.py](tests/test_evaluation.py) uses it for the
  scoring and failure policy, [test_api.py](tests/test_api.py) for validation and wiring.
- **`StubJudgeEndpoint`** ([test_api.py](tests/test_api.py)) is a stdlib HTTP server speaking
  the OpenAI chat-completions format on a free port, scripted per criterion. The end-to-end
  tests point `RUBRIC_EVAL_JUDGE_ENDPOINT` at it and drive the whole chain — HTTP request,
  `JudgeConfig.from_env()`, the real `openai` SDK, a real socket, reply parsing, the weighted
  fold. That is what proves the wire format and the self-healing retry, which a replaced
  `Judge` can never show.

[test_judge.py](tests/test_judge.py) covers the parser reply by reply (missing JSON, malformed
JSON, off-scale and fractional scores, an empty reply, `content: null`), the retry loop, and
the concurrency limit — including that a judge reused from a second event loop still scores a
rubric larger than its limit.

## Scope

Implemented: evaluating one (question, answer, rubric) case, over HTTP or as a library.

Not implemented: batch evaluation with run metrics, rubric catalog files, a CLI, and
self-consistency (judging each criterion several times and reporting the `spread`).

## License

MIT — see [LICENSE](LICENSE).
