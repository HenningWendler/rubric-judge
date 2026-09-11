# rubric-eval

Judge the answers of an LLM application against weighted reference criteria.

You know what a good answer to a question has to contain. Write that down as a **rubric** —
a list of weighted criteria — hand over the answer your system produced, and rubric-eval
asks an LLM judge, **one call per criterion**, whether each one is covered. Back comes a
score per criterion with the judge's own reasoning, one normalized score for the answer,
and — for a whole catalog of them — metrics over the entire run.

**It is a pure evaluator.** No RAG, no retrieval, no answer generation, no database, no
storage. Answers come in ready; results go out as JSON.

```python
import asyncio
from rubric_eval import Case, Criterion, JudgeConfig, OpenAIJudge, evaluate_case

judge = OpenAIJudge(JudgeConfig.from_env())

result = asyncio.run(evaluate_case(judge, Case(
    id=1,
    question="How do I report sick leave?",
    answer="Send an email to hr@example.com before 10:00 on your first day.",
    criteria=[
        Criterion(id=1, content="Report by email before 10:00 on the first day", weight=3),
        Criterion(id=2, content="State the expected last day of absence", weight=1),
    ],
)))

print(result.score)                       # 0.75  →  criterion 1 covered, criterion 2 missing
for verdict in result.criterion_results:
    print(verdict.criterion_id, verdict.score, verdict.reasoning)
# 1 2.0 The answer instructs the reader to email hr@example.com before 10:00 …
# 2 0.0 Neither the expected last day nor any duration is mentioned …
```

**Contents** — [Install](#install) · [Configure](#configure) · [Quickstart](#quickstart) ·
[Reference](#reference) ([inputs](#inputs) · [outputs](#outputs) ·
[functions](#functions) · [HTTP](#http-api) · [errors](#errors)) ·
[How scoring works](#how-scoring-works) · [Failure and load](#failure-and-load) ·
[Extending](#extending) · [Architecture](#architecture) · [Tests](#tests) · [Scope](#scope)

---

## Install

Python 3.11 or newer.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[test]'      # drop [test] if you do not want pytest
```

Runtime dependencies: `pydantic`, `openai`, `fastapi`, `uvicorn`.

## Configure

The judge is configured **only** through environment variables, so no secret ever lands in
a committed file. Copy [.env.example](.env.example) to `.env` and fill it in.

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `RUBRIC_EVAL_JUDGE_ENDPOINT` | URL | **required** | OpenAI-compatible base URL, version path included: `https://api.openai.com/v1`, `http://localhost:11434/v1` |
| `RUBRIC_EVAL_JUDGE_API_KEY` | string | **required** | Key for that endpoint. Local servers usually accept any non-empty string |
| `RUBRIC_EVAL_JUDGE_MODEL` | string | **required** | Model name as *that* endpoint knows it: `gpt-4o-mini`, `qwen3:8b`, … |
| `RUBRIC_EVAL_JUDGE_TEMPERATURE` | float ≥ 0 | `0.0` | Keep at `0.0`: verdicts stay reproducible while each criterion is judged once |
| `RUBRIC_EVAL_JUDGE_MAX_TOKENS` | int ≥ 1 | `768` | Budget per judge call. Must fit the reasoning **and** the closing JSON — a reply cut off before the JSON is unparseable and costs a retry |
| `RUBRIC_EVAL_JUDGE_MAX_ATTEMPTS` | int ≥ 1 | `3` | Tries per criterion, **the first one included**. Raise it for a small model that formats badly |
| `RUBRIC_EVAL_JUDGE_MAX_CONCURRENT` | int ≥ 1 | `8` | Judge calls in flight at once. Raise for a local vLLM or Ollama, lower for a small hosted tier |

An empty value counts as missing. **All** missing or invalid variables are reported in one
error, so configuration is fixed in a single pass instead of one restart per mistake:

```
RuntimeError: Missing environment variables: RUBRIC_EVAL_JUDGE_API_KEY, RUBRIC_EVAL_JUDGE_MODEL
```

Any OpenAI-compatible endpoint works — OpenAI, vLLM, Azure, Ollama, Groq, OpenRouter —
because the official `openai` SDK talks to whatever `base_url` you give it.

## Quickstart

### As a library

The library is **async**. From synchronous code, wrap the call in `asyncio.run()` as in the
example at the top; inside an existing event loop, `await` it directly.

Build the judge **once** and reuse it. The concurrency budget belongs to the judge instance,
so a judge per case would hand each case its own set of slots:

```python
judge = OpenAIJudge(JudgeConfig.from_env())     # once, at startup
```

**One answer** → a `CaseResult`:

```python
from rubric_eval import Case, Criterion, evaluate_case

result = await evaluate_case(judge, Case(
    id=1,
    question="How do I report sick leave?",
    answer="Email hr@example.com before 10:00.",
    criteria=[Criterion(id=1, content="Report by email before 10:00", weight=3)],
))

result.case_id                       # 1        — the id you gave the case
result.score                         # 1.0      — weighted, normalized to [0, 1]
result.criterion_results[0].score    # 2.0      — the raw judge score, 0 / 1 / 2
result.criterion_results[0].reasoning  # "The answer instructs the reader to …"
```

**A whole catalog** → a `BatchResult`: every `CaseResult` plus the aggregate over them.

```python
from rubric_eval import Batch, evaluate_batch

run = await evaluate_batch(judge, Batch(cases=[
    Case(id=1, question="How do I report sick leave?", answer="Email hr@example.com.",
         criteria=[Criterion(id=1, content="Report by email before 10:00", weight=3)]),
    Case(id=2, question="How do I request vacation?", answer="Ask your team lead.",
         criteria=[Criterion(id=1, content="Submit the request in the HR tool", weight=1)]),
]))

run.metrics.average_score          # 0.5   — mean over the cases
run.metrics.cases_with_score_zero  # [2]   — read these answers first
run.case_results[0].score          # 1.0   — the individual results are all still there
```

`run.case_results[0]` is exactly the `CaseResult` `evaluate_case()` would have returned for
that case on its own — the batch is a fan-out over it plus the aggregate, nothing more.

Every result is a Pydantic model, so `result.model_dump()` and `result.model_dump_json()`
give you plain data to write to disk.

### As an HTTP service

```bash
.venv/bin/uvicorn rubric_eval.api:app --env-file .env --port 8000
```

Interactive, generated API docs: [localhost:8000/docs](http://localhost:8000/docs).

```bash
curl -s localhost:8000/evaluate -H 'content-type: application/json' -d '{
  "id": 1,
  "question": "How do I report sick leave?",
  "answer": "Send an email to hr@example.com before 10:00 on your first day.",
  "criteria": [
    { "id": 1, "content": "Report by email before 10:00 on the first day", "weight": 3 },
    { "id": 2, "content": "State the expected last day of absence", "weight": 1 }
  ]
}'
```

The service is stateless: no catalog, no run ids, no persistence. It is the same library
behind a different door, so both paths always produce identical scores.

---

## Reference

### Inputs

Three input types, one per scale. All are Pydantic models — construct them in Python, or
post the same shape as JSON.

#### `Criterion` — one requirement a good answer has to satisfy

| Field | Type | Required | Rules |
|---|---|---|---|
| `id` | `int` | yes | Yours. Echoed back as `CriterionResult.criterion_id`. Must be unique **within its case** |
| `content` | `str` | yes | The requirement in plain language. Non-empty after stripping whitespace |
| `weight` | `float` | yes | `> 0`, finite. Only *ratios* matter: `3` and `1` score exactly like `30` and `10` |

Phrase `content` as **one checkable fact**. Two facts in one criterion cannot be scored
apart, and the judge will have to pick a compromise score:

```python
Criterion(id=1, content="Send an email to hr@example.com", weight=3)          # good
Criterion(id=2, content="Email HR before 10:00 and inform your team", weight=3)  # two facts
```

#### `Case` — one answer plus the rubric to hold it against

| Field | Type | Required | Rules |
|---|---|---|---|
| `id` | `int` | yes | Yours. Echoed back as `CaseResult.case_id`. Unique within its batch |
| `question` | `str` | yes | Context for the judge only — it is **never scored**. May be empty |
| `answer` | `str` | yes | The answer under test, judged exactly as it comes in. May be empty: a system that returned nothing is a valid case that scores `0` |
| `criteria` | `list[Criterion]` | yes | At least one. Criterion ids must be unique |

`id` is mandatory even for a single evaluation. That is what makes one result type serve
both paths — see [the vocabulary](#the-vocabulary).

#### `Batch` — many cases evaluated in one go

| Field | Type | Required | Rules |
|---|---|---|---|
| `cases` | `list[Case]` | yes | At least one. Case ids must be unique |

### Outputs

#### `CriterionResult` — the verdict for one criterion

| Field | Type | Range | Meaning |
|---|---|---|---|
| `criterion_id` | `int` | — | The `Criterion.id` this verdict belongs to |
| `weight` | `float` | `> 0` | Copy of `Criterion.weight`, so a result can be re-scored without the rubric at hand |
| `score` | `float` | `0.0 … 2.0` | `2` fully covered · `1` partially · `0` not covered. A float so averaging repeated runs cannot change the type |
| `is_present` | `bool` | — | `score >= 0.5`. Derived here, never asked of the judge — one question less for it to get wrong |
| `spread` | `float` | `>= 0` | Standard deviation across repeated runs of this criterion. Always `0.0` today: each criterion is judged exactly once |
| `failed` | `bool` | — | `true` when the judge produced no usable verdict even after all retries. The criterion still counts as `0` and keeps its weight |
| `reasoning` | `str \| None` | — | The judge's argument for the score — or, when `failed`, the error that prevented one |

#### `CaseResult` — one whole answer

| Field | Type | Range | Meaning |
|---|---|---|---|
| `case_id` | `int` | — | The `Case.id` this result belongs to |
| `score` | `float` | `0.0 … 1.0` | The weighted case score, see [How scoring works](#how-scoring-works). `1.0` means every criterion fully covered |
| `criterion_results` | `list[CriterionResult]` | — | One verdict per criterion, **in rubric order**, so it can be zipped with `Case.criteria` |

> Named `criterion_results`, not `criteria`: the list holds *verdicts*, one per criterion —
> `Case.criteria` is the rubric, and one name must not mean two things.

#### `RunMetrics` — the aggregate over a batch

Every case counts **once**, whatever the size of its rubric — otherwise one case with
twenty criteria would outvote nineteen cases with one.

| Field | Type | Range | What it answers |
|---|---|---|---|
| `total_cases` | `int` | `>= 1` | How many cases went into these numbers |
| `average_score` | `float` | `0 … 1` | Mean of the case scores — the single number a run is usually reported by |
| `median_score` | `float` | `0 … 1` | Middle case score. Far above the mean means a few catastrophic cases drag an otherwise solid run down |
| `variance` | `float` | `>= 0` | Sample variance of the case scores. `0.0` for a single case, which has no spread |
| `standard_deviation` | `float` | `>= 0` | Square root of it, in score units. Small = uniformly good or bad; large = it depends heavily on the question |
| `average_criterion_score` | `float` | `0 … 2` | How the judge rates an average *statement*, on the raw 0–2 scale, ignoring weights and case boundaries. A different question from `average_score` |
| `criteria_fulfillment_rate` | `float` | `0 … 1` | Mean share of criteria counting as `is_present`, averaged **per case first** so a long rubric cannot dominate |
| `cases_with_score_zero` | `list[int]` | — | Ids of answers that missed their rubric completely. Read these first |
| `cases_with_score_zero_count` | `int` | `>= 0` | Length of that list. Derived, so the two can never disagree |
| `weakest_cases_above_zero` | `list[int]` | ≤ 5 entries | The weakest cases that still scored *something*, weakest first. Kept apart from the zeros because a total miss and a partial answer usually have different causes |
| `failed_criteria_count` | `int` | `>= 0` | Criteria the judge never answered for. **Read this before the average**: anything above `0` means the run is depressed by outages, not only by the answers |

#### `BatchResult` — one whole run

| Field | Type | Meaning |
|---|---|---|
| `metrics` | `RunMetrics` | The aggregate over every case of the run |
| `case_results` | `list[CaseResult]` | One per case, in request order — so any suspicious number can be traced back |

### Functions

```python
async def evaluate_case(judge: Judge, case: Case) -> CaseResult
```
Judges one case. Fans out over the criteria concurrently, contains judge failures
(see [Failure and load](#failure-and-load)), folds the verdicts into one score.

```python
async def evaluate_batch(judge: Judge, batch: Batch) -> BatchResult
```
Judges every case concurrently and adds `RunMetrics`. A fan-out over `evaluate_case` and
nothing else, so the two paths cannot drift apart.

```python
def run_metrics(results: list[CaseResult]) -> RunMetrics
```
The aggregate on its own — use it when you already have case results (loaded from disk,
say) and only want the numbers. Raises `ValueError` on an empty list.

```python
def case_score(results: list[CriterionResult]) -> float
```
The weighted formula alone, `→ [0, 1]`. Raises `ValueError` on an empty list.

```python
class OpenAIJudge:
    def __init__(self, config: JudgeConfig, prompt: str | None = None)
    async def score(self, question: str, answer: str, criterion: Criterion) -> Verdict
```
The bundled judge. `prompt` replaces the system prompt; `score()` judges a single criterion
with no fan-out and no failure handling — handy for a quick experiment.

```python
class JudgeConfig:
    model: str; endpoint: str; api_key: str          # required
    temperature: float = 0.0; max_tokens: int = 768  # optional
    max_attempts: int = 3; max_concurrent: int = 8

    @classmethod
    def from_env(cls) -> JudgeConfig
```
`from_env()` reads the variables from [Configure](#configure). Build it by hand in tests or
when your settings come from elsewhere:

```python
JudgeConfig(model="qwen3:8b", endpoint="http://localhost:11434/v1", api_key="ollama")
```

Constants, if you need to compute against them: `SCALE_MAX = 2` (best score per criterion),
`PRESENCE_THRESHOLD = 0.5` (the `is_present` cut), `WEAKEST_CASES_REPORTED = 5`.

### HTTP API

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/evaluate` | a `Case` | a `CaseResult` |
| `POST` | `/evaluate/batch` | a `Batch` | a `BatchResult` |
| `GET` | `/health` | — | `{"status": "ok"}` |

The JSON shapes are exactly the models above. `POST /evaluate/batch`:

```json
{
  "cases": [
    { "id": 1, "question": "How do I report sick leave?", "answer": "Send an email …",
      "criteria": [ { "id": 1, "content": "Report by email before 10:00", "weight": 3 } ] },
    { "id": 2, "question": "How do I request vacation?", "answer": "Ask your team lead.",
      "criteria": [ { "id": 1, "content": "Submit the request in the HR tool", "weight": 1 } ] }
  ]
}
```

```json
{
  "metrics": {
    "total_cases": 2,
    "average_score": 0.5, "median_score": 0.5,
    "variance": 0.5, "standard_deviation": 0.7071067811865476,
    "average_criterion_score": 1.0,
    "criteria_fulfillment_rate": 0.5,
    "cases_with_score_zero": [2], "cases_with_score_zero_count": 1,
    "weakest_cases_above_zero": [1],
    "failed_criteria_count": 0
  },
  "case_results": [
    { "case_id": 1, "score": 1.0, "criterion_results": [
        { "criterion_id": 1, "weight": 3.0, "score": 2.0, "is_present": true,
          "spread": 0.0, "failed": false, "reasoning": "The answer instructs …" } ] },
    { "case_id": 2, "score": 0.0, "criterion_results": [ … ] }
  ]
}
```

A `case_results[i]` entry is **the same document** `POST /evaluate` returns for that case —
the same type, not a similar one — so the two endpoints cannot disagree.

`GET /health` answers even when the judge is unconfigured, so a missing key never takes the
container down. `POST /evaluate` is what fails then, loudly, with a `500`.

### Errors

Everything is validated **before** the first LLM call, so a malformed request costs nothing.

| Situation | Library | HTTP |
|---|---|---|
| `criteria` or `cases` empty; duplicate ids; `content` blank; `weight` `0`, negative, `Infinity` or `NaN`; missing field | `pydantic.ValidationError` | `422` |
| Judge not configured (missing env vars) | `RuntimeError` naming every missing variable | `500` |
| Judge endpoint refused / timed out / out of quota | **contained** — that criterion gets `failed: true`, `score: 0`, cause in `reasoning` | `200` |
| Judge reply unparseable after `MAX_ATTEMPTS` | **contained**, same way | `200` |
| A bug in the program (`TypeError`, `AttributeError`, `NameError`, `ImportError`, `RuntimeError`) | re-raised | `500` |
| Request cancelled (client disconnect, shutdown) | `asyncio.CancelledError` propagates | — |

The `422` body names only *where* and *what*, never the value that was sent — so it stays
serializable and the API never mirrors arbitrary request content back:

```json
{ "detail": [ { "loc": ["body", "criteria", 0, "weight"],
                "msg": "Input should be greater than 0", "type": "greater_than" } ] }
```

---

## How scoring works

### The 0–2 scale

The judge sees the question, the answer and **one** criterion, and returns one integer:

| Score | Meaning |
|---|---|
| `2` | Fully covered. Every essential part is recognizable, even if worded differently |
| `1` | Partially covered. Essential information is missing, but the idea is derivable |
| `0` | Not covered. Absent, or no recognizable connection to the criterion |

It writes its argument first and the score last, as a JSON object — which is why
`reasoning` costs nothing extra: the judge produces it anyway.

### One case

Each criterion's score is turned into a fraction of what it could have reached, weighted,
and normalized over the sum of the weights:

```
score = Σᵢ ( wᵢ · sᵢ / 2 ) / Σᵢ wᵢ        ∈ [0, 1]
```

Worked example — three criteria, weights 3 / 2 / 1, scored 2 / 1 / 0:

| Criterion | Weight `w` | Score `s` | `w · s / 2` |
|---|---|---|---|
| 1 | 3 | 2 | 3.0 |
| 2 | 2 | 1 | 1.0 |
| 3 | 1 | 0 | 0.0 |
| | **6** | | **4.0** |

`4.0 / 6 = 0.667`. Only the ratios matter, so weights `30 / 20 / 10` give the same `0.667`.

A criterion the judge never answered for stays in the denominator with `score = 0` — an
outage lowers the score **visibly** rather than silently shrinking the rubric.

### One run

`RunMetrics` aggregates the case scores; every field is defined in
[the reference table](#runmetrics-the-aggregate-over-a-batch). Two of them are easy to
misread, so in short:

- **`average_criterion_score` is not `average_score` on a different scale.** It ignores
  weights and case boundaries — "how well does the judge rate an average statement", not
  "how good is the average answer".
- **`criteria_fulfillment_rate` is averaged per case, then over the cases.** Flattening all
  criteria first would let one case with a 20-criterion rubric dominate nineteen short ones.

---

## Failure and load

### Unparseable replies heal themselves

If the judge replies with no JSON, with malformed JSON, or with a score off the integral
scale (`3`, `1.5`), the **concrete cause** is fed back to it together with its own broken
reply, and the call is repeated up to `RUBRIC_EVAL_JUDGE_MAX_ATTEMPTS` times. Not a blind
retry — the model is told what was wrong with what it wrote.

### A dead judge costs one criterion, never the case

A criterion the judge could not answer for is marked `failed: true`, counts as `score = 0`,
keeps its weight, and reports the cause in its `reasoning`. The case and the run still
finish. `RunMetrics.failed_criteria_count` is how you notice.

Only *endpoint* failures are contained this way — refused connections, timeouts, quota
errors, unparseable replies. Errors that mean the **program** is wrong are re-raised: a
criterion scored `0` because of a bug is indistinguishable from a real result, which is far
worse than a crash.

### Concurrency

The criteria of a case are judged concurrently, but at most
`RUBRIC_EVAL_JUDGE_MAX_CONCURRENT` calls are in flight at once — a rubric with 200 criteria
costs 8 open connections, not 200.

**The limit sits on the judge, not on the fan-out.** One judge instance serves the whole
process, so the budget is shared by everything using it:

| Situation | Coroutines | Connections in flight |
|---|---|---|
| one case, 200 criteria | 200 | 8 |
| 50 cases × 10 criteria in one batch | 500 | 8 |
| five concurrent batches | thousands | 8 |
| ten parallel `POST /evaluate` | — | 8 total, not 8 each |

A slot is held for one HTTP call only, so a criterion waiting to be retried does not occupy
one. And the run is still judged case-overlapping, not case after case.

A batch deliberately gets **no second throttle**. Only the judge implementation knows what
its backend tolerates, so that is the single place the limit lives: raise
`RUBRIC_EVAL_JUDGE_MAX_CONCURRENT`, not the batch size.

---

## Extending

### Your own prompt

```python
from pathlib import Path

judge = OpenAIJudge(JudgeConfig.from_env(), prompt=Path("my_prompt.txt").read_text())
```

`prompt=` replaces the **system prompt** only. Whatever you write has to keep two promises,
or the parser will reject every reply: the model must argue first and end with a single
`{"score": 0|1|2}` object, and the prose scale must stay 0–2.

The user prompt and the retry complaints live in [prompt.py](src/rubric_eval/prompt.py).
Every sentence the model ever reads is in that one file, as plain Python strings — a
reviewer who does not read Python can still audit the whole evaluation.

| In `prompt.py` | Sent as |
|---|---|
| `JUDGE_EN` | the system message |
| `criterion_prompt(question, answer, criterion)` | the user message |
| `NO_JSON_HINT`, `out_of_range_hint()`, `malformed_json_hint()` | the follow-up on a retry |

The complaints are worded as instructions on purpose: the parser raises them as `ValueError`
messages and the retry loop hands that text straight back to the model. The exception
message *is* the corrective prompt — rewording one means changing the prompt.

### Your own judge

`Judge` is a `Protocol`. Anything with this method plugs into `evaluate_case()` and
`evaluate_batch()` unchanged — a different SDK, a local model, a cached judge, a stub:

```python
from rubric_eval import Criterion, Verdict

class MyJudge:
    async def score(self, question: str, answer: str, criterion: Criterion) -> Verdict:
        ...
        return Verdict(score=2, reasoning="…")
```

Three responsibilities come with it:

- **Staying on the scale.** `Verdict.score` must be an integer in `0..SCALE_MAX`. The model
  type only checks that it is an `int` — the range is enforced by `OpenAIJudge`'s parser, so
  a custom judge that returns `5` produces a case score above `1.0` and nothing will stop it.
- **Throttling.** `evaluate_case()` hands out one task per criterion whatever the rubric's
  size, because only your implementation knows what your backend tolerates. `OpenAIJudge`
  bounds itself with `max_concurrent`; yours needs its own bound.
- **Raising on failure.** Return a valid `Verdict` or raise. Anything you raise that is not
  a programming error becomes one `failed` criterion.

---

## Architecture

Six modules, each with one job. A request walks straight down through them:

| Step | File | Responsibility |
|---|---|---|
| 1 | [api.py](src/rubric_eval/api.py) | FastAPI endpoints: validate the body, inject the judge, hand back JSON. No domain logic |
| 2 | [evaluation.py](src/rubric_eval/evaluation.py) | `evaluate_case()` and `evaluate_batch()` — fan out, contain failures, fold the verdicts |
| 3 | [judge.py](src/rubric_eval/judge.py) | `Judge` protocol, OpenAI-compatible client, reply parsing, retries, throttle, env config |
| 4 | [prompt.py](src/rubric_eval/prompt.py) | every word the judge is told |
| — | [models.py](src/rubric_eval/models.py) | the types below, the 0–2 scale, `CriterionResult.judged()` / `.unjudged()` |
| — | [metrics.py](src/rubric_eval/metrics.py) | `case_score()` and `run_metrics()` — the formulas, nothing else |

### The vocabulary

Three scales, each a pair of *what goes in* and *what comes back*:

| Scale | In | Out |
|---|---|---|
| one requirement | `Criterion` | `CriterionResult` |
| one answer | `Case` | `CaseResult` |
| a whole catalog | `Batch` | `BatchResult` |

Plus `RunMetrics`, which is `BatchResult.metrics` and nothing else, and `Verdict`, which
never leaves `judge.py`.

Two rules hold the naming together — worth knowing before adding a field:

- **A result never reuses the name of its input.** `CaseResult.criterion_results` holds
  verdicts, so it is not called `criteria`.
- **Exactly one type per scale.** A case evaluated alone and a case inside a batch are the
  same `Case`, and both come back as the same `CaseResult`. That is why `Case.id` is
  mandatory rather than optional: an id that is sometimes there would have meant two
  near-identical types, two result shapes, and a `null` to check for.

### Where does a new rule go?

The result types draw the layer boundary and answer the question by themselves:

```python
Verdict:          score, reasoning                            # what the model replied
CriterionResult:  criterion_id, weight, score, is_present,     # what the system concluded
                  spread, failed, reasoning
CaseResult:       case_id, score, criterion_results            # one whole case
RunMetrics:       the aggregate over many case results
```

`judge.py` speaks `Verdict` — prompts, parsing and retries are *its* business, and another
implementation may do all three differently. Everything that has to hold no matter who
judges — the 0–2 scale, the `is_present` threshold, the weighting, "a dead judge costs one
criterion" — lives outside it, or two judges would produce incomparable scores.

So nothing above `judge.py` knows what a chat completion is, nothing inside it knows what a
weight is, and `evaluation.py` is the only module that speaks both languages.

### How the code documents itself

Every public function states its inputs, its return value and what it raises, in Google
style — `help(evaluate_batch)` in a REPL is the same reference as this README. Sections
always appear in this order, the example last:

```
Args:
    judge: As for `evaluate_case`. The same instance serves every case of the batch …
Returns:
    A `BatchResult`: `case_results` in request order, and `metrics` aggregated over them …
Raises:
    As `evaluate_case`. A programming error in any single case aborts the whole batch …
Example:
    run = await evaluate_batch(judge, Batch(cases=[case_a, case_b]))
```

Private helpers deliberately do **not** get that treatment. They are two or three lines
with a name that says what they do; an `Args:` block there would be noise around the one
sentence that actually matters — *why* it exists.

FastAPI endpoint docstrings are user-facing too: they become the descriptions in the
Swagger UI at `/docs`, so they are written for whoever is calling the API, not for whoever
is maintaining it.

Model fields are documented with attribute docstrings rather than comments:

```python
class Criterion(DocumentedModel):
    weight: float = Field(gt=0, allow_inf_nan=False)
    """How much this criterion counts next to the others. Only the ratios matter:
    weights 3 and 1 score exactly like 30 and 10."""
```

Pydantic's `use_attribute_docstrings` (set once on `DocumentedModel`) copies them into the
OpenAPI schema, so the same sentence serves IDE hover, the Swagger UI at `/docs` and the
field tables in [Reference](#reference). One text, never three — they cannot drift apart.

---

## Tests

```bash
.venv/bin/python -m pytest
```

108 tests, no real LLM ever called. Mocked at two levels:

- **`FakeJudge`** ([conftest.py](tests/conftest.py)) replaces the `Judge` protocol and scores
  from a lookup table — `{1: 2, 2: ValueError("down")}` scores criterion 1 with a `2` and
  lets the judge die on criterion 2. `CASE` and `BATCH` live in the same file, so the domain
  tests and the HTTP tests describe literally the same input. `BATCH` is deliberately uneven
  (`0.75`, `0.5`, `0.0`) because uniform scores make most run metrics indistinguishable.
- **`StubJudgeEndpoint`** ([test_api.py](tests/test_api.py)) is a stdlib HTTP server speaking
  the OpenAI chat-completions format on a free port, scripted per criterion. The end-to-end
  tests point `RUBRIC_EVAL_JUDGE_ENDPOINT` at it and drive the whole chain — HTTP request,
  `JudgeConfig.from_env()`, the real `openai` SDK, a real socket, reply parsing, the weighted
  fold. That is what proves the wire format and the self-healing retry.

| File | Covers |
|---|---|
| [test_metrics.py](tests/test_metrics.py) | the scoring formula, float extremes, every run metric |
| [test_evaluation.py](tests/test_evaluation.py) | fan-out, ordering, failure policy, batch aggregation |
| [test_judge.py](tests/test_judge.py) | the parser reply by reply, the retry loop, the concurrency limit |
| [test_api.py](tests/test_api.py) | validation, wiring, serialization, and the end-to-end chain |

## Scope

**Implemented** — evaluating one case or a whole batch with run metrics, as a library or
over HTTP.

**Not implemented** — rubric catalog files, a CLI, comparing two runs against each other,
labels and per-label metrics, streaming progress for long batches, and self-consistency
(judging each criterion several times and reporting the `spread`).

## License

MIT — see [LICENSE](LICENSE).
