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
[Comparing two runs](#comparing-two-runs) · [Reference](#reference) ([inputs](#inputs) · [outputs](#outputs) ·
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

The **grading scale** is deliberately not an environment variable: it carries a sentence per
grade, and prose does not belong in a variable meant for a URL or a key. It is set in code, on
the judge — see [Another scale](#another-scale).

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
result.scale.maximum                 # 2        — what those raw scores are out of
```

The `scale` comes along because a result has to stay readable on its own: it says what the
grades are out of, from where one counts as covered, and what each grade means in words. The
judge owns it, and you can give it another one — see [Another scale](#another-scale).

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

### Comparing two runs

Two `BatchResult`s of the **same catalog** → one `ComparisonResult`: did your change help,
where, and what did it cost. No judge, no network, no cost — runs stored as JSON months
apart compare exactly like runs produced a second ago.

```python
from pathlib import Path
from rubric_eval import BatchResult, RunPair, compare_runs

baseline = BatchResult.model_validate_json(Path("run_before.json").read_text())
candidate = BatchResult.model_validate_json(Path("run_after.json").read_text())

result = compare_runs(RunPair(baseline=baseline, candidate=candidate))

result.metrics_delta.average_score_delta   # +0.125  — the run got better on average
result.metrics_delta.median_score_delta    # -0.125  — but the typical case did not
result.summary.improved_case_ids           # [1]     — biggest improvement first
result.summary.worsened_case_ids           # [3]     — what the win cost
result.summary.worsening.largest           # -0.5
```

Every delta is **candidate minus baseline**, so a positive number always means the candidate
did better. `compare_runs` takes a `RunPair` rather than two arguments on purpose: both
sides have the same type, so a swapped pair would be undetectable and would invert every
sign. Only what comes back carries `Result` in its name — `RunPair` is what you hand in.

Drill down when a number needs explaining — run, case, criterion:

```python
regressed = result.case_comparison_results[2]
regressed.case_id                                      # 3
regressed.baseline_score, regressed.candidate_score    # (1.0, 0.5)

dropped = regressed.criterion_comparison_results[0]
dropped.score_delta                                    # -1.0  — the judge dropped it 2 → 1
dropped.status                                         # ChangeStatus.WORSENED
```

Comparing runs of **different** catalogs is refused rather than approximated:

```python
compare_runs(RunPair(baseline=run_of_8_cases, candidate=run_of_7_cases))
# RunsNotComparableError: the runs are not comparable: cases only in the baseline: [6]
```

`RunsNotComparableError` subclasses `ValueError`, so `except ValueError` still catches it.
It has its own name so the HTTP layer can tell a genuinely incomparable pair from a bug —
`ValidationError` and `StatisticsError` are `ValueError`s too, and reporting one of those as
a `422` would blame the caller for our mistake.

Same case ids, same criterion ids per case, same weights — all three, or no comparison.
Weights are the denominator every case score is normalized by, so `0.5` scored under one
set of weights and `0.5` under another are not the same `0.5`, and their difference is not
`0`. The message names **every** difference at once, because fixing them one error at a
time would mean re-judging the whole catalog for each.

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

Four input types. All are Pydantic models — construct them in Python, or post the same
shape as JSON.

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

#### `RunPair` — two finished runs to hold against each other

| Field | Type | Required | Rules |
|---|---|---|---|
| `baseline` | `BatchResult` | yes | The run compared *against* — the state of things before your change |
| `candidate` | `BatchResult` | yes | The run *under test*. Must have been judged on the same `scale` and must cover the same case ids, the same criterion ids per case and the same weights as `baseline` |

A named pair rather than two arguments: both sides have the same type, so a swap would be
impossible to detect and would invert the sign of every number in the result.

### Outputs

#### `Scale` — the grading scale a case was judged on

Carried by every `CaseResult`, so a stored run can be read and re-scored without the judge
that produced it — and still says what its grades meant. Compared by value, descriptions
included; two runs are comparable only if their scales are equal.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `maximum` | `int` | `> 0` | Best score one criterion can reach. The scale is **integral**: `maximum = 2` offers exactly the grades `0`, `1`, `2` |
| `presence_threshold` | `float` | `> 0`, `<= maximum` | From which score `is_present` counts the criterion as covered. Above `0` because a `0` is by definition not covered |
| `level_descriptions` | `dict[int, str]` | empty, or one entry per grade | What each grade means, keyed by the grade. Either complete or absent — a prompt explaining four of ten levels is worse than one explaining none. A described scale **writes its own judge prompt**; an undescribed one is arithmetic only |

`DEFAULT_SCALE` is the `0–2` scale the bundled prompt describes, descriptions included, and
the one a stored result is read on when it names none.

#### `CriterionResult` — the verdict for one criterion

| Field | Type | Range | Meaning |
|---|---|---|---|
| `criterion_id` | `int` | — | The `Criterion.id` this verdict belongs to |
| `weight` | `float` | `> 0` | Copy of `Criterion.weight`, so a result can be re-scored without the rubric at hand |
| `score` | `float` | `0.0 … scale.maximum` | The judge's **raw** grade, not normalized. On the default scale: `2` fully covered · `1` partially · `0` not covered. A float so averaging repeated runs cannot change the type. Bounded by `CaseResult.scale`, which is the object that knows it |
| `is_present` | `bool` | — | `score >= scale.presence_threshold`, using the scale of the `CaseResult` above. Never asked of the judge, and a `CaseResult` **refuses** a verdict whose value here contradicts its own score |
| `spread` | `float` | `>= 0` | Standard deviation across repeated runs of this criterion. Always `0.0` today: each criterion is judged exactly once |
| `failed` | `bool` | — | `true` when the judge produced no usable verdict even after all retries. The criterion still counts as `0` and keeps its weight |
| `reasoning` | `str \| None` | — | The judge's argument for the score — or, when `failed`, the error that prevented one |

#### `CaseResult` — one whole answer

| Field | Type | Range | Meaning |
|---|---|---|---|
| `case_id` | `int` | — | The `Case.id` this result belongs to |
| `score` | `float` | `0.0 … 1.0` | The weighted case score, see [How scoring works](#how-scoring-works). `1.0` means every criterion fully covered |
| `scale` | `Scale` | — | What the grades below mean. Stored **once per case**, not per verdict: one case is judged by one judge on one scale, and `POST /evaluate` returns this document on its own, so this is the lowest level that always exists. Defaults to `DEFAULT_SCALE` when a stored result names none |
| `criterion_results` | `list[CriterionResult]` | ≥ 1 entry | One verdict per criterion, **in rubric order**, so it can be zipped with `Case.criteria`. Never empty — a rubric has at least one criterion, so a result has at least one verdict. Criterion ids must be unique: verdicts are paired by id when two runs are compared |

> Named `criterion_results`, not `criteria`: the list holds *verdicts*, one per criterion —
> `Case.criteria` is the rubric, and one name must not mean two things.

`BatchResult.scale` reads the one scale its cases were judged on — a Python accessor, not a
JSON field, since the cases already carry it. A run whose cases name more than one scale is
rejected, and so is a comparison of two runs whose scales differ.

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
| `average_criterion_score` | `float` | `0 … scale.maximum` | How the judge rates an average *statement*, on the run's **raw** scale (`1.4` of `2`, not `0.7`), ignoring weights and case boundaries. A different question from `average_score` |
| `criteria_fulfillment_rate` | `float` | `0 … 1` | Mean share of criteria counting as `is_present`, averaged **per case first** so a long rubric cannot dominate |
| `cases_with_score_zero` | `list[int]` | — | Ids of answers that missed their rubric completely. Read these first |
| `cases_with_score_zero_count` | `int` | `>= 0` | Length of that list. Derived, so the two can never disagree |
| `weakest_cases_above_zero` | `list[int]` | ≤ 5 entries | The weakest cases that still scored *something*, weakest first. Kept apart from the zeros because a total miss and a partial answer usually have different causes |
| `failed_criteria_count` | `int` | `>= 0` | Criteria the judge never answered for. **Read this before the average**: anything above `0` means the run is depressed by outages, not only by the answers |

#### `BatchResult` — one whole run

| Field | Type | Meaning |
|---|---|---|
| `metrics` | `RunMetrics` | The aggregate over every case of the run |
| `case_results` | `list[CaseResult]` | ≥ 1 entry. One per case, in request order — so any suspicious number can be traced back. Never empty: `run_metrics` refuses a run of no cases, so such a run was never producible. Case ids must be unique: cases are paired by id when two runs are compared |

#### `ChangeStatus` — which way a score moved

A string enum with three values, used at both the case and the criterion grain. It
serializes as the plain string, so JSON readers never see an object.

| Value | Meaning |
|---|---|
| `"improved"` | The candidate scored higher, by more than `SCORE_EQUALITY_TOLERANCE` |
| `"stable"` | The two scores are equal within `SCORE_EQUALITY_TOLERANCE` |
| `"worsened"` | The candidate scored lower, by more than `SCORE_EQUALITY_TOLERANCE` |

Derived from the **raw score**, not from `is_present`: a criterion that went from `1` to `2`
never crosses the presence threshold yet visibly moved the case score, so it is reported as
improved.

#### `CriterionComparisonResult` — one criterion across two runs

| Field | Type | Range | Meaning |
|---|---|---|---|
| `criterion_id` | `int` | — | The criterion both runs judged. Identical in both by construction |
| `weight` | `float` | `> 0` | Its weight, identical in both — compared **exactly**, since weights are copied from the rubric and never computed. Read a movement against it: a `2.0` swing on weight `1` beside nine criteria of weight `3` barely moves the case |
| `baseline_score` | `float` | `0.0 … 2.0` | What the baseline run's judge gave it |
| `candidate_score` | `float` | `0.0 … 2.0` | What the candidate run's judge gave it |
| `score_delta` | `float` | `-2.0 … 2.0` | `candidate_score - baseline_score` |
| `status` | `ChangeStatus` | — | That delta as a verdict |

#### `CaseComparisonResult` — one case across two runs

| Field | Type | Range | Meaning |
|---|---|---|---|
| `case_id` | `int` | — | The case both runs evaluated |
| `baseline_score` | `float` | `0.0 … 1.0` | Its weighted score in the baseline run |
| `candidate_score` | `float` | `0.0 … 1.0` | Its weighted score in the candidate run |
| `score_delta` | `float` | `-1.0 … 1.0` | `candidate_score - baseline_score` |
| `status` | `ChangeStatus` | — | That delta as a verdict |
| `criterion_comparison_results` | `list[CriterionComparisonResult]` | ≥ 1 entry | One per criterion, **ordered by `criterion_id`** — so the list reads the same whichever order either run happened to be stored in |

A case can be `"stable"` while its criteria moved hard in opposite directions. That is
exactly why the criterion grain exists.

#### `RunMetricsDelta` — the run-level difference

Candidate minus baseline, one field per `RunMetrics` field that can meaningfully be
subtracted. `total_cases` has none — a comparison of different case sets is refused — and
the two id lists have none, because a set of ids does not subtract; `ChangeSummary` reports
the movement of cases instead.

| Field | Type | Meaning |
|---|---|---|
| `average_score_delta` | `float` | Change in the mean case score. The headline number |
| `median_score_delta` | `float` | Change in the median. Read next to the mean: a mean that rose while the median fell means a few cases carried the win |
| `variance_delta` | `float` | Change in the sample variance of the case scores |
| `standard_deviation_delta` | `float` | Change in their spread. Negative means more uniform — an improvement or a regression depending on which way the mean went |
| `average_criterion_score_delta` | `float` | Change on the runs' shared raw scale, ignoring weights and case boundaries. Moves independently of `average_score_delta` |
| `criteria_fulfillment_rate_delta` | `float` | Change in the mean share of criteria counting as covered |
| `cases_with_score_zero_count_delta` | `int` | Change in how many answers missed completely. **Negative is the good direction here** |
| `failed_criteria_count_delta` | `int` | Change in judge outages. **Read this first**: anything but `0` and every other number above is partly an artefact of the outage rather than of the answers |

#### `ChangeMagnitude` — how large the moves on one side were

| Field | Type | Meaning |
|---|---|---|
| `largest` | `float` | The single biggest move on this side |
| `mean` | `float` | Mean of the moves — what a typical one was worth |
| `median` | `float` | Median of them. Far below the mean on the improvement side means one case carries the win |

All three carry the **sign of their side**, so a worsening's `largest` is the most negative
delta, not its absolute value. All three are `0.0` when nothing moved that way — a run where
nothing got worse has no worsening to report.

#### `ChangeSummary` — where the run moved, case by case

The counterpart to `RunMetricsDelta`. That one says the average rose by `0.125`; this one
says whether every case rose a little or one rose a lot while another collapsed.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `improved_case_ids` | `list[int]` | — | Cases the candidate scored higher on, **biggest improvement first**. Complete, not capped — the top three are its first three |
| `stable_case_ids` | `list[int]` | — | Cases whose score did not move beyond the tolerance, in id order |
| `worsened_case_ids` | `list[int]` | — | Cases the candidate scored lower on, **biggest regression first**. Read these when an average went up and you want to know what it cost |
| `improvement` | `ChangeMagnitude` | — | Size of the moves behind `improved_case_ids`, all positive |
| `worsening` | `ChangeMagnitude` | — | Size of the moves behind `worsened_case_ids`, all negative |
| `improved_case_count` | `int` | `>= 0` | Length of `improved_case_ids`. Derived, so list and count cannot disagree |
| `stable_case_count` | `int` | `>= 0` | Length of `stable_case_ids` |
| `worsened_case_count` | `int` | `>= 0` | Length of `worsened_case_ids` |
| `improvement_rate` | `float` | `0 … 1` | Share of cases that improved |
| `stability_rate` | `float` | `0 … 1` | Share that did not move |
| `worsening_rate` | `float` | `0 … 1` | Share that got worse. Every case lands in exactly one list, so the three **counts** always add up to the run; the three rates are three separate divisions and sum to `1.0` only to within float rounding |

#### `ComparisonResult` — one whole comparison

| Field | Type | Meaning |
|---|---|---|
| `metrics_delta` | `RunMetricsDelta` | Whether the run got better |
| `summary` | `ChangeSummary` | How that is distributed over the cases |
| `case_comparison_results` | `list[CaseComparisonResult]` | Which criterion is responsible. **Ordered by `case_id`** — both runs cover the same cases, so neither one's storage order is canonical |

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
say) and only want the numbers. Raises `ValueError` on an empty list, or on cases judged on
different scales: `average_criterion_score` averages raw grades, and grades in two units do
not average.

```python
def case_score(results: list[CriterionResult], scale: Scale) -> float
```
The weighted formula alone, `→ [0, 1]`. Dividing by `scale.maximum` is what makes the result
scale-free. Raises `ValueError` on an empty list, or when a verdict is graded above the scale.

```python
def judge_prompt(scale: Scale, examples: str = "") -> str
```
Writes a judge's system prompt from a scale that describes its levels — header, one line per
level, and the reply format down to the grades the model may answer with. `examples` is
appended verbatim. Raises `ValueError` for a scale with no `level_descriptions`.

```python
def compare_runs(run_pair: RunPair) -> ComparisonResult
```
Holds two finished runs against each other at three grains — run, case, criterion. Pure
computation: no judge, no network, no cost. Raises `RunsNotComparableError` — a `ValueError`
— naming **every** difference at once when the runs do not describe the same catalog.

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

Constants, if you need to compute against them: `DEFAULT_SCALE`
(`Scale(maximum=2, presence_threshold=0.5)`), `WEAKEST_CASES_REPORTED = 5`,
`SCORE_EQUALITY_TOLERANCE = 1e-9` (how close two scores must be to count as unchanged in a
comparison — far above the float noise two runs accumulate summing the same weights in a
different order, far below the smallest difference a rubric can actually produce).

### HTTP API

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/evaluate` | a `Case` | a `CaseResult` |
| `POST` | `/evaluate/batch` | a `Batch` | a `BatchResult` |
| `POST` | `/compare` | a `RunPair` | a `ComparisonResult` |
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
    { "case_id": 1, "score": 1.0,
      "scale": { "maximum": 2, "presence_threshold": 0.5,
                 "level_descriptions": { "2": "Fully covered. …", "1": "Partially covered. …",
                                         "0": "Not covered. …" } },
      "criterion_results": [
        { "criterion_id": 1, "weight": 3.0, "score": 2.0, "is_present": true,
          "spread": 0.0, "failed": false, "reasoning": "The answer instructs …" } ] },
    { "case_id": 2, "score": 0.0, "criterion_results": [ … ] }
  ]
}
```

A `case_results[i]` entry is **the same document** `POST /evaluate` returns for that case —
the same type, not a similar one — so the two endpoints cannot disagree.

`POST /compare` takes two of those `BatchResult` documents back verbatim, no reshaping:

```json
{ "baseline":  { "metrics": { … }, "case_results": [ … ] },
  "candidate": { "metrics": { … }, "case_results": [ … ] } }
```

For a three-case catalog where case 1 went `0.0 → 0.875`, case 2 held at `1.0` and case 3
fell `1.0 → 0.5` — the response, printed verbatim:

```json
{
  "metrics_delta": {
    "average_score_delta": 0.1250000000000001,
    "median_score_delta": -0.12499999999999989,
    "variance_delta": -0.265625, "standard_deviation_delta": -0.3171420192563591,
    "average_criterion_score_delta": 0.5,
    "criteria_fulfillment_rate_delta": 0.33333333333333337,
    "cases_with_score_zero_count_delta": -1,
    "failed_criteria_count_delta": 0
  },
  "summary": {
    "improved_case_ids": [1], "stable_case_ids": [2], "worsened_case_ids": [3],
    "improvement": { "largest": 0.8750000000000001, "mean": 0.8750000000000001,
                     "median": 0.8750000000000001 },
    "worsening":   { "largest": -0.5,  "mean": -0.5,  "median": -0.5 },
    "improved_case_count": 1, "stable_case_count": 1, "worsened_case_count": 1,
    "improvement_rate": 0.3333333333333333,
    "stability_rate": 0.3333333333333333,
    "worsening_rate": 0.3333333333333333
  },
  "case_comparison_results": [
    { "case_id": 1, "baseline_score": 0.0, "candidate_score": 0.8750000000000001,
      "score_delta": 0.8750000000000001, "status": "improved",
      "criterion_comparison_results": [
        { "criterion_id": 1, "weight": 3.0, "baseline_score": 0.0, "candidate_score": 2.0,
          "score_delta": 2.0, "status": "improved" },
        { "criterion_id": 2, "weight": 1.0, "baseline_score": 0.0, "candidate_score": 1.0,
          "score_delta": 1.0, "status": "improved" } ] },
    { "case_id": 2, "…": "stable" },
    { "case_id": 3, "…": "worsened" }
  ]
}
```

The mean rose while the median *fell* — one case carried the whole win, and
`worsened_case_ids` says which one paid for it. That pair of numbers is the reason both are
reported.

Those trailing digits are real, printed verbatim from a run: `0.8750000000000001` is what
summing weights `3` and `1` in that order actually produces. It is also why "stable" is a
tolerance and not an `==` — see `SCORE_EQUALITY_TOLERANCE` under
[Functions](#functions).

`GET /health` answers even when the judge is unconfigured, so a missing key never takes the
container down. `POST /evaluate` is what fails then, loudly, with a `500`. `POST /compare`
needs no judge at all and answers correctly with no API key configured.

### Errors

Everything is validated **before** the first LLM call, so a malformed request costs nothing.

| Situation | Library | HTTP |
|---|---|---|
| `criteria`, `cases`, `criterion_results` or `case_results` empty; duplicate ids in any of them; `content` blank; `weight` `0`, negative, `Infinity` or `NaN`; missing field | `pydantic.ValidationError` | `422` |
| A run posted to `/compare` whose numbers leave the ranges the [output tables](#outputs) give — a score above its own `scale.maximum` or off `0 … 1`, a non-positive weight, any `Infinity` or `NaN` | `pydantic.ValidationError` | `422` |
| A run whose cases name more than one `scale`; a verdict whose `is_present` contradicts its own score | `pydantic.ValidationError` | `422` |
| A `Scale` describing only some of its grades, or a grade it does not have | `pydantic.ValidationError` | `422` |
| Two runs not comparable: a different grading scale, different case ids, different criteria within a case, or different weights | `RunsNotComparableError` (a `ValueError`) naming **every** difference at once | `422` |
| A judge returning a score above the `scale` it declares | `ValueError` out of `evaluate_case()` naming the criterion — a bug in the judge, not an outage | `500` |
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

### The scale

The judge sees the question, the answer and **one** criterion, and returns one integer. Out
of the box that is `DEFAULT_SCALE`, `0–2`:

| Score | Meaning |
|---|---|
| `2` | Fully covered. Every essential part is recognizable, even if worded differently |
| `1` | Partially covered. Essential information is missing, but the idea is derivable |
| `0` | Not covered. Absent, or no recognizable connection to the criterion |

It writes its argument first and the score last, as a JSON object — which is why
`reasoning` costs nothing extra: the judge produces it anyway.

That table is not built in — it **is** `DEFAULT_SCALE.level_descriptions`, and the judge's
prompt is generated from it. The scale belongs to the **judge**, not to the package: a judge of
your own can grade `0–5` or `0–10` with meanings of its own (see
[Another scale](#another-scale)), and every `CaseResult` it produces stores that scale.

Raw grades are therefore always reported in the judge's own units, while `CaseResult.score`
and `RunMetrics.average_score` are normalized to `0 … 1` and stay comparable across scales.

### One case

Each criterion's score is turned into a fraction of what it could have reached, weighted,
and normalized over the sum of the weights:

```
score = Σᵢ ( wᵢ · sᵢ / max ) / Σᵢ wᵢ         ∈ [0, 1]
```

`max` is `CaseResult.scale.maximum` — dividing by it is what makes the case score
scale-free. Half marks everywhere is `0.5` whether the judge counted in halves or in fifths,
and that is the whole reason this number is normalized rather than reported raw.

Worked example — three criteria, weights 3 / 2 / 1, scored 2 / 1 / 0 on the default scale:

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
`{"score": <grade>}` object, and the prose scale it describes must be the judge's `scale` —
`0–2` unless you pass one.

You usually do not need this. A scale that describes its levels writes the prompt itself —
see [Another scale](#another-scale). Reach for `prompt=` when you want different *instructions*
(another language, a stricter examiner, your own worked examples), not merely another scale.

The user prompt and the retry complaints live in [prompt.py](src/rubric_eval/prompt.py).
Every sentence the model ever reads is in that one file, as plain Python strings — a
reviewer who does not read Python can still audit the whole evaluation.

| In `prompt.py` | Sent as |
|---|---|
| `judge_prompt(scale, examples)` | the system message, written from the scale |
| `JUDGE_EN` | what that returns for `DEFAULT_SCALE` plus `WORKED_EXAMPLES_EN` |
| `criterion_prompt(question, answer, criterion)` | the user message |
| `no_json_hint()`, `out_of_range_hint()`, `malformed_json_hint()` | the follow-up on a retry |

Every one of them takes the judge's `Scale` and lists its grades out, so a judge on a
ten-point scale is never corrected into answering `0`, `1` or `2`. Nothing writes a scale out
by hand any more: `JUDGE_EN` is generated, and
[tests/judge_prompt_en.txt](tests/judge_prompt_en.txt) holds the hand-written original that a
test pins it to, byte for byte — so a change to a level description shows up as a diff.

The complaints are worded as instructions on purpose: the parser raises them as `ValueError`
messages and the retry loop hands that text straight back to the model. The exception
message *is* the corrective prompt — rewording one means changing the prompt.

### Another scale

A judge is not tied to `0–2`. Describe the grades you want, and the judge writes its own
prompt from them — no prompt to rewrite, no text to keep in sync:

```python
from rubric_eval import JudgeConfig, OpenAIJudge, Scale

judge = OpenAIJudge(JudgeConfig.from_env(), scale=Scale(
    maximum=3,
    presence_threshold=2,
    level_descriptions={
        3: "Fully covered, with the specifics the criterion names.",
        2: "Covered in substance, but a detail is missing or imprecise.",
        1: "Touched on only. The reader could not act on what is there.",
        0: "Not covered. Absent, or no recognizable connection to the criterion.",
    },
))
```

That produces a system prompt whose scale block, header and reply format all come from the
scale, so the model can never be instructed on a scale the parser does not enforce:

```
Use this 0-3 scale:

3 = Fully covered, with the specifics the criterion names.
2 = Covered in substance, but a detail is missing or imprecise.
...
[Two or three sentences arguing which score the scale calls for.]
{"score": 0, 1, 2 or 3}
```

**Describe every grade or none.** A half-described scale is refused, because a prompt that
explains four of ten levels is worse than one that explains none:

```python
Scale(maximum=2, presence_threshold=0.5, level_descriptions={2: "Yes.", 0: "No."})
# ValidationError: level descriptions must describe every grade of the scale 0..2 …
```

A scale with **no** descriptions is arithmetic only — legal, but then you owe the judge a
prompt:

```python
OpenAIJudge(config, scale=Scale(maximum=10, presence_threshold=5))
# ValueError: the scale 0..10 (covered from 5.0) describes no levels, so no prompt can be
#             written from it: give it a level_descriptions entry per grade, or pass a
#             prompt of your own
```

**Worked examples are not generated.** The three bundled ones close on grades of `0`, `1` and
`2`, so they belong to `DEFAULT_SCALE` alone — a judge on another scale gets the instructions
without them. To add your own:

```python
from rubric_eval import judge_prompt

OpenAIJudge(config, prompt=judge_prompt(my_scale, my_examples), scale=my_scale)
```

What changes with the scale, and what does not:

| | Follows the scale | Stays the same |
|---|---|---|
| grades | `CriterionResult.score`, `RunMetrics.average_criterion_score`, every criterion-level delta | |
| coverage | `is_present`, via `presence_threshold` | |
| the prompt | the scale block, the header, the list of allowed grades, every retry complaint | the instructions and examples around them |
| normalized | | `CaseResult.score`, `average_score`, `median_score`, `criteria_fulfillment_rate` — all still `0 … 1` |

Two runs judged on different scales are **not comparable**: `compare_runs()` refuses them the
same way it refuses different weights, because a `2` out of `2` and a `2` out of `10` are not
the same verdict. **The descriptions count as part of the scale**, so rewording what a grade
means — even fixing a typo in it — makes new runs incomparable with old ones. That is on
purpose: telling the judge something else about a `1` changes the grades it gives, and a
comparison that ignored that would report a prompt edit as a change in your system.

### Your own judge

`Judge` is a `Protocol`. Anything with this attribute and this method plugs into
`evaluate_case()` and `evaluate_batch()` unchanged — a different SDK, a local model, a cached
judge, a stub:

```python
from rubric_eval import DEFAULT_SCALE, Criterion, Scale, Verdict

class MyJudge:
    scale: Scale = DEFAULT_SCALE

    async def score(self, question: str, answer: str, criterion: Criterion) -> Verdict:
        ...
        return Verdict(score=2, reasoning="…")
```

Three responsibilities come with it:

- **Declaring a scale, and staying on it.** `scale` is what `case_score()` normalizes by and
  what `is_present` cuts at, so a judge without it is a broken program: the `AttributeError`
  is re-raised rather than scored `0`. `Verdict.score` must be an integer in
  `0..scale.maximum` — `Verdict` itself only checks that it is an `int`, but the verdicts are
  held against the scale, so a judge that declares `0–2` and returns `5` raises
  `ValueError: criteria [1] scored above the scale 0..2 …` out of `evaluate_case()` rather than
  folding into a case score above `1.0`. It fails loudly on purpose: an out-of-scale verdict
  is a bug in the judge, and a bug must never come back as a plausible number (see
  [Failure and load](#failure-and-load)). It is not contained as an outage — only what
  `judge.score()` *raises* is.
- **Throttling.** `evaluate_case()` hands out one task per criterion whatever the rubric's
  size, because only your implementation knows what your backend tolerates. `OpenAIJudge`
  bounds itself with `max_concurrent`; yours needs its own bound.
- **Raising on failure.** Return a valid `Verdict` or raise. Anything you raise that is not
  a programming error becomes one `failed` criterion.

---

## Architecture

Seven modules, each with one job. A request walks straight down through them:

| Step | File | Responsibility |
|---|---|---|
| 1 | [api.py](src/rubric_eval/api.py) | FastAPI endpoints: validate the body, inject the judge, hand back JSON. No domain logic |
| 2 | [evaluation.py](src/rubric_eval/evaluation.py) | `evaluate_case()` and `evaluate_batch()` — fan out, contain failures, fold the verdicts |
| 3 | [judge.py](src/rubric_eval/judge.py) | `Judge` protocol, OpenAI-compatible client, reply parsing, retries, throttle, env config |
| 4 | [prompt.py](src/rubric_eval/prompt.py) | every word the judge is told, written from the scale |
| — | [models.py](src/rubric_eval/models.py) | the types below, `Scale` and `DEFAULT_SCALE`, `CriterionResult.judged()` / `.unjudged()` |
| — | [metrics.py](src/rubric_eval/metrics.py) | `case_score()` and `run_metrics()` — the formulas, nothing else |
| — | [comparison.py](src/rubric_eval/comparison.py) | `compare_runs()` — two finished runs into their differences. Reads no judge and no config |

### The vocabulary

Three grains, each a pair of *what goes in* and *what comes back*:

| Grain | In | Out |
|---|---|---|
| one requirement | `Criterion` | `CriterionResult` |
| one answer | `Case` | `CaseResult` |
| a whole catalog | `Batch` | `BatchResult` |
| two whole runs | `RunPair` | `ComparisonResult` |

Plus `RunMetrics`, which is `BatchResult.metrics` and nothing else, `Scale`, which every
verdict carries, and `Verdict`, which never leaves `judge.py`.

("Grain", not "scale" — `Scale` is the grading scale and one word must not mean two things.)

Comparing repeats the same three grains one level up, and the names say so: a
`CriterionComparisonResult` sits inside a `CaseComparisonResult` exactly as a
`CriterionResult` sits inside a `CaseResult`. **Only a produced type carries `Result`** —
`RunPair` is what you hand in, everything ending in `Result` is what comes back.

Two rules hold the naming together — worth knowing before adding a field:

- **A result never reuses the name of its input.** `CaseResult.criterion_results` holds
  verdicts, so it is not called `criteria`.
- **Exactly one type per grain.** A case evaluated alone and a case inside a batch are the
  same `Case`, and both come back as the same `CaseResult`. That is why `Case.id` is
  mandatory rather than optional: an id that is sometimes there would have meant two
  near-identical types, two result shapes, and a `null` to check for.

### Where does a new rule go?

The result types draw the layer boundary and answer the question by themselves:

```python
Verdict:          score, reasoning                            # what the model replied
CriterionResult:  criterion_id, weight, score, is_present,     # what the system concluded
                  spread, failed, reasoning
CaseResult:       case_id, score, scale, criterion_results     # one whole case
RunMetrics:       the aggregate over many case results
```

`judge.py` speaks `Verdict` — prompts, parsing and retries are *its* business, and another
implementation may do all three differently. Everything that has to hold no matter who
judges — how a grade becomes a share of the reachable points, the weighting, "a dead judge
costs one criterion" — lives outside it, or two judges would produce incomparable scores.

The grading scale is the one setting that belongs to the judge *and* has to be known outside
it: a judge owns it, every `CaseResult` carries a copy, so `metrics.py` and `comparison.py`
never have to ask which judge produced a run they are reading off disk. It is the reason
`prompt.py` can generate instead of store — the judge's words and the reader's units come
from the same object.

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

168 tests, no real LLM ever called. Mocked at two levels:

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
- **`run_of`** ([conftest.py](tests/conftest.py)) builds a finished `BatchResult` straight
  from judge scores — `run_of({1: 2, 2: 0}, {21: 1})` is a two-case run. Comparison tests
  are about the *difference* between two runs, so going through a judge and an event loop would
  only stand between the test and the numbers it asserts on.

| File | Covers |
|---|---|
| [test_metrics.py](tests/test_metrics.py) | the scoring formula, float extremes, every run metric |
| [test_evaluation.py](tests/test_evaluation.py) | fan-out, ordering, failure policy, batch aggregation |
| [test_judge.py](tests/test_judge.py) | the parser reply by reply, the retry loop, the concurrency limit |
| [test_comparison.py](tests/test_comparison.py) | deltas and their direction, the three statuses, ordering, and every refusal |
| [test_api.py](tests/test_api.py) | validation, wiring, serialization, and the end-to-end chain |

## Scope

**Implemented** — evaluating one case or a whole batch with run metrics, and comparing two
finished runs against each other, as a library or over HTTP.

**Not implemented** — rubric catalog files, a CLI, labels and per-label metrics, streaming
progress for long batches, and self-consistency (judging each criterion several times and
reporting the `spread`).

## License

MIT — see [LICENSE](LICENSE).
