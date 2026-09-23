# rubric-judge

Judge the answers of an LLM application against weighted reference criteria.

You changed a prompt, swapped a model or reworked your retrieval, and now you want to know
whether the assistant actually got better. Asking a model "is this a good answer" gives you
a number nobody can argue with or act on. rubric-judge asks something narrower instead.

You already know what a good answer has to contain. Write that down as a **rubric**, a
handful of weighted criteria, one checkable requirement each. Hand over the answer your
system produced, and an LLM judge is asked **once per criterion** whether that one
requirement is covered. Back comes a grade and the judge's own reasoning for every
criterion, so a score is never a verdict you have to take on faith. You can read why it
came out that way and disagree with a single line of it.

An answer and its rubric make a **case**, and a case is worth one normalized score between
0 and 1. When the answer was produced in response to something, put that in the case's
**context** and say what it is, for example `"The question asked was: ..."`. The judge reads
the context as background and never scores it. Leave it out and the answer is judged on its
own terms, which is what a summary, a report or a drafted email needs. A catalog of cases
makes a **run**, which is worth a set of metrics over all of them. Cases carry **labels**,
so one run can be sliced into the kinds of case you care about, and the two slices often
tell you something the average hides.
Two finished runs can be **compared**, which is how you find out whether a change helped,
and which single requirement it quietly broke.

It is a pure evaluator. There is no retrieval, no answer generation, no database and no
storage. Answers come in ready, results go out as JSON. It runs as a Python library or as
an HTTP service, and both produce identical scores because the service is the library
behind a different door.

## The smallest complete program

```python
import asyncio

from rubric_judge import Case, Criterion, JudgeConfig, OpenAIJudge, evaluate_case

case = Case(
    id=1,
    context="The question asked was: How do I report sick leave?",
    answer=(
        "Call your line manager as early as you can on the first day you are ill, "
        "at the latest before 9:00. If you cannot reach them, leave a voicemail and "
        "send a short message as well."
    ),
    criteria=[
        Criterion(id=11, content="Tells the employee to inform their line manager.", weight=3),
        Criterion(id=12, content="Names the deadline: before 9:00 on the first day of absence.", weight=2),
        Criterion(id=13, content="Says a doctor's note is needed from the fourth day of absence.", weight=1),
    ],
)

case_result = asyncio.run(evaluate_case(OpenAIJudge(JudgeConfig.from_env()), case))

print(f"case {case_result.case_id} scored {case_result.score:.2f}")
for criterion_result in case_result.criterion_results:
    covered = "covered" if criterion_result.is_present else "missing"
    print(f"  criterion {criterion_result.criterion_id}: {criterion_result.score} of 2  {covered}")
```

```
case 1 scored 0.83
  criterion 11: 2.0 of 2  covered
  criterion 12: 2.0 of 2  covered
  criterion 13: 0.0 of 2  missing
```

The answer names the right person and the right deadline but never mentions the doctor's
note, so it loses the weight-1 criterion and keeps 5 of the 6 reachable points. That is
0.83.

### The same program without a context

Not every answer is a reply to something. A summary, a report or a drafted email is judged
against its criteria and nothing else. Leave `context` out and the judge is shown the answer
and the criterion alone, with no `Context:` heading above them inviting it to wonder what is
missing.

```python
case = Case(
    id=2,
    answer=(
        "The Cologne office has an underground garage with 40 spots. Employees reserve "
        "a spot through the facility portal, at the latest on the day before. Visitors "
        "are registered at reception."
    ),
    criteria=[
        Criterion(id=21, content="Says what a parking spot costs per month.", weight=3),
        Criterion(id=22, content="Says how an employee reserves a spot.", weight=2),
        Criterion(id=23, content="Names the deadline for a reservation.", weight=1),
    ],
)
```

```
case 2 scored 0.50
  criterion 21: 0.0 of 2  missing
  criterion 22: 2.0 of 2  covered
  criterion 23: 2.0 of 2  covered
```

The passage says how to reserve and by when, but nothing about a monthly price, so it loses
the heaviest criterion and keeps 3 of the 6 points. Everything else is the same program: the
same judge, the same scale, the same result type.

### About the output in this manual

Every snippet in this manual was really run, and every block of output under one is what came
back, pasted unedited. They were judged by **`gpt-5.4-mini-2026-03-17`** on **2026-09-22** at
temperature `0.0`, on the scale that ships with the package.

An LLM judge is not a fixed function. Run the same snippet against the same model tomorrow
and the grades will usually be the same, the reasoning will be worded differently, and an
arguable criterion can land on the other side of the line. [Repeatability](#repeatability)
measures how far that goes and what it costs you. Read the numbers here as one honest
sample, not as an assertion about your run.

**Contents** &middot; [Install](#install) &middot; [Configure](#configure) &middot;
[As a library](#as-a-library) &middot; [Over HTTP](#over-http) &middot;
[Going further](#going-further) &middot; [How scoring works](#how-scoring-works) &middot;
[Failure and load](#failure-and-load) &middot;
[Reference](docs/REFERENCE.md) &middot; [Architecture](docs/ARCHITECTURE.md)

---

## Install

Python 3.11 or newer.

```bash
python3.12 -m venv .venv
.venv/bin/pip install --constraint constraints.txt -e .
```

The runtime dependencies are `pydantic`, `openai`, `fastapi` and `uvicorn`.
[constraints.txt](constraints.txt) pins them, and everything they pull in, to the versions the
test suite passed on. Leave the flag out and pip takes the newest versions `pyproject.toml`
allows, which is also what a project installing rubric-judge as a dependency gets. Add the test
extra if you want to run the suite,
`.venv/bin/pip install --constraint constraints.txt -e '.[test]'`.

There is no command line interface. The package installs no console script, and
`python -m rubric_judge` does nothing. The two entry points are the library and the HTTP
service.

### With Docker

The image serves the HTTP API from [Over HTTP](#over-http) and nothing else. It runs Python
3.12 and exactly the versions in `constraints.txt`, as one uvicorn process on port 8000 under
a user that is not root.

```bash
docker build --tag rubric-judge .
```

Docker builds for the machine it runs on, so an image built on an Apple silicon Mac runs only
on arm64. Name the platform of the server it is meant for.

```bash
docker build --platform linux/amd64 --tag rubric-judge .
```

The configuration is the variables from [Configure](#configure), passed in when the container
starts. None of them is baked into the image. `--env-file` reads the same `.env` the service
reads without Docker, and `--publish 8001:8000` puts the service on the port every curl
example in this manual uses.

```bash
docker run --rm --env-file .env --publish 8001:8000 rubric-judge
```

```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Docker reads the file literally. Write `KEY=value` without quotes, because a quoted value
reaches the service with its quotes. Whitespace around a value does no harm, because the service
drops it.

Before the log says `Application startup complete`, the service has sent its judge one small
call and got an answer, as [Configuring the service](#configuring-the-service) explains. A
container that came up has a judge that works.

The image asks `/health` every second while it starts and every 30 seconds after that, so
`docker ps` tells you whether the service is healthy. Docker itself only shows that verdict.
Restarting an unhealthy container or routing around it is your orchestrator's job.

```bash
docker ps --filter ancestor=rubric-judge --format '{{.Image}}  {{.Status}}'
```

```
rubric-judge  Up 5 seconds (healthy)
```

Every request in [Over HTTP](#over-http) works against the container unchanged. The first one
there, sent to this container on 2026-09-23 with the judge from
[About the output](#about-the-output-in-this-manual), scores what the library scored.

```bash
curl -sS -X POST http://localhost:8001/evaluate \
  -H 'Content-Type: application/json' \
  --data @- <<'JSON' | jq '{score, grades: [.criterion_results[] | {criterion_id, score, is_present}]}'
{
  "id": 1,
  "context": "The question asked was: How do I report sick leave?",
  "answer": "Call your line manager as early as you can on the first day you are ill, at the latest before 9:00. If you cannot reach them, leave a voicemail and send a short message as well.",
  "criteria": [
    {"id": 11, "content": "Tells the employee to inform their line manager.", "weight": 3},
    {"id": 12, "content": "Names the deadline: before 9:00 on the first day of absence.", "weight": 2},
    {"id": 13, "content": "Says a doctor's note is needed from the fourth day of absence.", "weight": 1}
  ],
  "labels": ["procedure"]
}
JSON
```

```json
{
  "score": 0.8333333333333333,
  "grades": [
    {
      "criterion_id": 11,
      "score": 2.0,
      "is_present": true
    },
    {
      "criterion_id": 12,
      "score": 2.0,
      "is_present": true
    },
    {
      "criterion_id": 13,
      "score": 0.0,
      "is_present": false
    }
  ]
}
```

A container whose judge does not work never comes up. It exits with code 3, and the traceback
it prints ends with the cause. Here a key the endpoint refuses, passed over the one in `.env`.

```bash
docker run --rm --env-file .env --env RUBRIC_JUDGE_API_KEY=sk-invalid rubric-judge
```

```
    raise self._make_status_error_from_response(err.response) from None
openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-invalid. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'code': 'invalid_api_key', 'param': None}, 'status': 401}

ERROR:    Application startup failed. Exiting.
```

A half-written configuration is refused the same way, with every missing variable named.

```bash
docker run --rm --env RUBRIC_JUDGE_MODEL=gpt-5.4-mini rubric-judge
```

```
    raise RuntimeError(f"Unusable environment variables: {', '.join(unusable)}")
RuntimeError: Unusable environment variables: RUBRIC_JUDGE_ENDPOINT is missing, RUBRIC_JUDGE_API_KEY is missing

ERROR:    Application startup failed. Exiting.
```

A container started with no `RUBRIC_JUDGE_*` variable at all comes up in compare-only mode,
described in [Configuring the service](#configuring-the-service).

```bash
docker run --rm --publish 8001:8000 rubric-judge
```

```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
No RUBRIC_JUDGE_* variable is set, so this service starts without a judge. POST /compare works, POST /evaluate and POST /evaluate/run answer 503.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Configure

The judge is configured through environment variables only, so no key ever lands in a
committed file. Copy [.env.example](.env.example) to `.env` and fill it in. The same
variables serve both entry points.

| Variable | Type | Required | Meaning |
|---|---|---|---|
| `RUBRIC_JUDGE_ENDPOINT` | URL | yes | OpenAI-compatible base URL with the version path, for example `https://api.openai.com/v1` or `http://localhost:11434/v1` |
| `RUBRIC_JUDGE_API_KEY` | string | yes | Key for that endpoint. Local servers usually accept any non-empty string |
| `RUBRIC_JUDGE_MODEL` | string | yes | Model name as that endpoint knows it, for example `gpt-5.4-mini` or `qwen3:8b` |
| `RUBRIC_JUDGE_TEMPERATURE` | float, 0 or more | no, `0.0` | Leave it at `0.0`. Each criterion is judged once, so anything higher buys variance you cannot see. See [Repeatability](#repeatability) |
| `RUBRIC_JUDGE_MAX_TOKENS` | int, 1 or more | no, `768` | Budget per judge call. It has to fit the reasoning and the closing JSON. A reply cut off before its JSON is unusable and costs a retry |
| `RUBRIC_JUDGE_MAX_ATTEMPTS` | int, 1 or more | no, `3` | Tries per criterion, counting the first. An unusable reply and a failed connection each cost one. Running out fails the whole run. Raise it for a small model that formats badly, or for a flaky endpoint |
| `RUBRIC_JUDGE_MAX_CONCURRENT` | int, 1 or more | no, `8` | Judge calls in flight at once. Raise it for a local vLLM or Ollama, lower it for a small hosted tier |
| `RUBRIC_JUDGE_HEALTH_INTERVAL` | int, `0` or 60 or more | no, `0` | Seconds a running service lets its judge go without an answered call before it lists the endpoint's models, which costs no tokens, for example `86400` for once a day. `0` never checks. See [A judge that stops working](#a-judge-that-stops-working) |
| `RUBRIC_JUDGE_HEALTH_RETRIES` | int, 0 or more | no, `3` | How often that periodic check is repeated after an outage before the service reports unhealthy. `0` reports it after the first failed check |
| `RUBRIC_JUDGE_HEALTH_FIRST_PAUSE` | int, 1 or more | no, `300` | Seconds before the first of those retries. Each further pause doubles, so the default waits 5, 10 and 20 minutes |
| `RUBRIC_JUDGE_ACCESS_TOKEN` | string | no, none | Read by the HTTP service only. Once set, every endpoint except `/health` needs `Authorization: Bearer <token>`. Unset, the service is open. See [Configuring the service](#configuring-the-service) |

Any OpenAI-compatible endpoint works, including OpenAI, Azure, vLLM, Ollama, Groq and
OpenRouter, because the official `openai` SDK talks to whatever `base_url` it is given.

A variable exported **empty** is refused, and that holds for the optional ones too. An
empty optional variable does not fall back to its default, only one that is absent
entirely does. Whitespace around a value is dropped first, so a value of nothing but spaces
counts as empty. Every missing or empty variable is reported at once, in a single error.

```
RuntimeError: Unusable environment variables: RUBRIC_JUDGE_API_KEY is missing,
RUBRIC_JUDGE_MODEL is missing, RUBRIC_JUDGE_MAX_TOKENS is empty
```

A misspelled name is refused too. Every variable starting with `RUBRIC_JUDGE_` has to be one
of the table above, because a typo read as nothing would quietly fall back to the default. Here
the periodic check would have stayed off while you believed it ran once a day.

```bash
RUBRIC_JUDGE_HEALTH_INTERVALL=86400 .venv/bin/python -c "from rubric_judge import JudgeConfig; JudgeConfig.from_env()"
```

```
RuntimeError: Unusable environment variables: RUBRIC_JUDGE_HEALTH_INTERVALL is unknown
```

A numeric variable that does not parse is a separate, later failure. It raises a
`pydantic.ValidationError` naming the setting, not the `RuntimeError` above.

The HTTP service also reads the absence of all of them. Set no `RUBRIC_JUDGE_*` variable at
all and it starts in compare-only mode, which [Configuring the service](#configuring-the-service)
describes. Any single one, an optional one included, means a judge was intended, and then the
three required ones have to be there too. `RUBRIC_JUDGE_ACCESS_TOKEN` is the one exception,
because it locks the service and says nothing about the judge.

The grading scale is deliberately not an environment variable. It belongs to the judge and
is set in code, which [Another scale](#another-scale) shows.

Nothing reads `.env` for you. The library reads the process environment, so load the file
however your project already does, for example with `python-dotenv`. The HTTP service is
the exception, because `uvicorn --env-file .env` and `docker run --env-file .env` both do it
for you.

---

## As a library

The four steps below build on each other. One answer, then a whole catalog, then that same
catalog sliced by label, then two catalogs compared. They all run against the same four
questions an employee might ask a company handbook assistant, each one carried in the case's
`context`.

<details>
<summary>The catalog these examples use, in full</summary>

Two answers exist for each case. The baseline is the assistant in production, the
candidate is a newer one under test. Only the answers differ between the two, which is what
makes them comparable. The labels split the catalog into `procedure`, meaning how to do
something step by step, and `policy`, meaning what you are entitled to and what the rules
are.

```python
from rubric_judge import Case, Criterion, Run

BASELINE_CASES = [
    Case(
        id=1,
        context="The question asked was: How do I report sick leave?",
        answer=(
            "Call your line manager as early as you can on the first day you are ill, "
            "at the latest before 9:00. If you cannot reach them, leave a voicemail and "
            "send a short message as well."
        ),
        criteria=[
            Criterion(id=11, content="Tells the employee to inform their line manager.", weight=3),
            Criterion(id=12, content="Names the deadline: before 9:00 on the first day of absence.", weight=2),
            Criterion(id=13, content="Says a doctor's note is needed from the fourth day of absence.", weight=1),
        ],
        labels=["procedure"],
    ),
    Case(
        id=2,
        context="The question asked was: How do I get the expenses for a business trip reimbursed?",
        answer=(
            "File the claim in the expense tool with your receipts attached. It has to "
            "be in within 30 days of the day the trip ended."
        ),
        criteria=[
            Criterion(id=21, content="Says the claim is filed in the expense tool.", weight=3),
            Criterion(id=22, content="Names the deadline: within 30 days of the end of the trip.", weight=1),
        ],
        labels=["procedure"],
    ),
    Case(
        id=3,
        context="The question asked was: How many vacation days do I get per year?",
        answer=(
            "You get a generous amount of paid vacation, and most colleagues take theirs "
            "in summer. Your remaining balance is shown in the HR tool."
        ),
        criteria=[
            Criterion(id=31, content="Gives the number of vacation days per year.", weight=3),
            Criterion(id=32, content="Says up to 5 unused days carry over into the next year.", weight=2),
            Criterion(id=33, content="Says carried-over days expire at the end of March.", weight=1),
        ],
        labels=["policy"],
    ),
    Case(
        id=4,
        context="The question asked was: Can I work from home?",
        answer=(
            "Yes, working from home is possible. Agree the days with your line manager "
            "beforehand."
        ),
        criteria=[
            Criterion(id=41, content="Gives the maximum number of home office days per week.", weight=2),
            Criterion(id=42, content="Says the days have to be agreed with the line manager.", weight=1),
        ],
        labels=["policy"],
    ),
]

CANDIDATE_ANSWER_PER_CASE_ID = {
    1: (
        "Call your line manager before 9:00 on the first day you are ill. From the "
        "fourth day of absence you also need a doctor's note for HR."
    ),
    2: (
        "Attach your receipts and file the claim in the expense tool. The deadline is "
        "30 days after the last day of the trip."
    ),
    3: (
        "You get 30 vacation days per year. Up to 5 days you do not use carry over into "
        "the next year."
    ),
    4: (
        "We are flexible about this. Plenty of teams work from home whenever it suits "
        "them, so just do what works for you."
    ),
}

baseline_run = Run(cases=BASELINE_CASES)
candidate_run = Run(
    cases=[
        case.model_copy(update={"answer": CANDIDATE_ANSWER_PER_CASE_ID[case.id]})
        for case in BASELINE_CASES
    ]
)
```

The answers are of mixed quality on purpose. Case 2 is essentially perfect, case 3 is
useless, cases 1 and 4 are partial. A catalog where everything scores well teaches you
nothing about your evaluator.

</details>

### Configuring the library

Build one judge and reuse it. `JudgeConfig.from_env()` reads the variables from
[Configure](#configure), and `OpenAIJudge` holds the connection pool and the concurrency
budget, so one instance per process is what you want.

```python
from rubric_judge import JudgeConfig, OpenAIJudge

judge = OpenAIJudge(JudgeConfig.from_env())
```

Call `asyncio.run()` once per program, on an `async def main()` that does all the
awaiting. An `OpenAIJudge` binds its HTTP client to the first event loop it runs in, and a
second `asyncio.run()` has closed that loop, so a judge reused across two of them dies with
`RuntimeError: Event loop is closed`. A fresh judge per `asyncio.run()` does work and is
still the wrong habit, because each loop then gets its own full set of `max_concurrent`
slots and the limit your endpoint was configured with is silently multiplied.

Everything is configurable in code as well, which is what tests and notebooks usually want.
`JudgeConfig` is an ordinary model, so you can build one without touching the environment,
and `JudgeConfig.from_mapping()` reads any mapping you hand it instead of `os.environ`.

```python
config = JudgeConfig(
    endpoint="http://localhost:11434/v1",
    api_key="ollama",
    model="qwen3:8b",
    max_concurrent=32,
)
```

### 1. One answer

`evaluate_case` judges every criterion of one case, concurrently, and returns a
`CaseResult`. The criterion results come back in rubric order, so they zip with the
criteria you passed in.

```python
print(f"context: {case.context}")
print(f"score:    {case_result.score:.4f}   scale: {case_result.scale}")
print()
for criterion, criterion_result in zip(case.criteria, case_result.criterion_results):
    print(f"[weight {criterion_result.weight:g}] {criterion.content}")
    print(f"    grade {criterion_result.score:g} of 2, is_present={criterion_result.is_present}")
    print(f"    reasoning: {criterion_result.reasoning}")
    print()
```

```
context: The question asked was: How do I report sick leave?
score:    0.8333   scale: 0..2 (covered from 0.5)

[weight 3] Tells the employee to inform their line manager.
    grade 2 of 2, is_present=True
    reasoning: The answer explicitly instructs the employee to call their line manager, which directly matches the criterion about informing the line manager. The additional timing and fallback instructions do not change that the core requirement is clearly present.

[weight 2] Names the deadline: before 9:00 on the first day of absence.
    grade 2 of 2, is_present=True
    reasoning: The answer explicitly states the deadline “at the latest before 9:00” and also anchors it to “the first day you are ill,” which matches the criterion closely. The required deadline is clearly named, with no essential part missing.

[weight 1] Says a doctor's note is needed from the fourth day of absence.
    grade 0 of 2, is_present=False
    reasoning: The answer explains how to notify the line manager about sick leave and gives a deadline, but it does not mention any doctor's note or when one is required. The specific requirement about needing a doctor's note from the fourth day is absent.
```

The reasoning costs nothing extra. The judge writes its argument first and its grade last,
so the text is produced either way and is simply kept.

### 2. A whole run

`evaluate_run` takes a `Run`, judges every case in it and adds metrics over all of them.

```python
import asyncio

from catalog import baseline_run
from rubric_judge import JudgeConfig, OpenAIJudge, evaluate_run

run_result = asyncio.run(evaluate_run(OpenAIJudge(JudgeConfig.from_env()), baseline_run))

for case_result in run_result.case_results:
    print(f"case {case_result.case_id} {case_result.labels}: {case_result.score:.4f}")
print()

metrics = run_result.metrics
print(f"total_cases                {metrics.total_cases}")
print(f"average_score              {metrics.average_score:.4f}")
print(f"median_score               {metrics.median_score:.4f}")
print(f"variance                   {metrics.variance:.4f}")
print(f"standard_deviation         {metrics.standard_deviation:.4f}")
print(f"average_criterion_score    {metrics.average_criterion_score:.4f}")
print(f"criteria_fulfillment_rate  {metrics.criteria_fulfillment_rate:.4f}")
print(f"cases_with_score_zero      {metrics.cases_with_score_zero}")
print(f"weakest_cases_above_zero   {metrics.weakest_cases_above_zero}")
```

```
case 1 ['procedure']: 0.8333
case 2 ['procedure']: 1.0000
case 3 ['policy']: 0.0000
case 4 ['policy']: 0.3333

total_cases                4
average_score              0.5417
median_score               0.5833
variance                   0.2106
standard_deviation         0.4590
average_criterion_score    1.0000
criteria_fulfillment_rate  0.5417
cases_with_score_zero      [3]
weakest_cases_above_zero   [4, 1, 2]
```

Read `cases_with_score_zero` first. Case 3 got nothing at all, which is a different kind of
problem from a case that scored badly. `weakest_cases_above_zero` is the shortlist after
that, weakest first, and it holds at most five entries. Case 2 is on it here only because
the run has four cases and there was room.

Every field of `RunMetrics` is defined in [the reference](docs/REFERENCE.md).

The full `RunResult` of this run is committed as
[examples/run_result_baseline.json](examples/run_result_baseline.json), so you can see the
whole document without running anything.

### 3. Slicing a run by label

Nothing is judged again here. The buckets are computed from the case results the run
already produced.

```python
from rubric_judge import filter_cases_by_labels, label_metrics

for bucket in label_metrics(run_result.case_results):
    print(
        f"{bucket.label:10} cases {bucket.metrics.total_cases}"
        f"  average {bucket.metrics.average_score:.4f}"
        f"  fulfillment {bucket.metrics.criteria_fulfillment_rate:.4f}"
        f"  zero {bucket.metrics.cases_with_score_zero}"
    )
print()

policy_cases = filter_cases_by_labels(baseline_run.cases, [["policy"]])
print("a run filtered to policy would judge cases", [one_case.id for one_case in policy_cases])
```

```
policy     cases 2  average 0.1667  fulfillment 0.2500  zero [3]
procedure  cases 2  average 0.9167  fulfillment 0.8333  zero []

a run filtered to policy would judge cases [3, 4]
```

This is what labels are for. The run average of 0.5417 describes no case in the run. The
assistant answers procedure questions almost perfectly and policy questions barely at all,
and a single headline number hides exactly that. `run_result.label_metrics` carries the
same buckets, so a stored run keeps them without the catalog.

A case may carry several labels and then counts in each of its buckets, so the bucket sizes
do not have to add up to the run.

To judge only part of a catalog in the first place, give the `Run` a label filter and
`evaluate_run` judges the subset.

```python
Run(cases=BASELINE_CASES, label_filter=[["policy"]])
```

The filter is an OR of ANDs. `[["policy"]]` picks everything labelled `policy`.
`[["policy", "urgent"]]` picks what carries both labels. `[["policy"], ["urgent"]]` picks
what carries either. An empty filter picks everything.

### 4. Comparing two runs

`compare_runs` takes two finished runs and needs no judge, because it only computes. Every
delta is `candidate - baseline`, so a positive number always means the candidate did
better.

```python
import asyncio

from catalog import baseline_run, candidate_run
from rubric_judge import (
    JudgeConfig,
    OpenAIJudge,
    RunComparison,
    RunComparisonResult,
    compare_runs,
    evaluate_run,
)


async def main() -> RunComparisonResult:
    judge = OpenAIJudge(JudgeConfig.from_env())
    baseline_result = await evaluate_run(judge, baseline_run)
    candidate_result = await evaluate_run(judge, candidate_run)
    return compare_runs(
        RunComparison(baseline=baseline_result, candidate=candidate_result)
    )


comparison_result = asyncio.run(main())

delta = comparison_result.metrics_delta
print(f"average_score_delta              {delta.average_score_delta:+.4f}")
print(f"median_score_delta               {delta.median_score_delta:+.4f}")
print(f"average_criterion_score_delta    {delta.average_criterion_score_delta:+.4f}")
print(f"criteria_fulfillment_rate_delta  {delta.criteria_fulfillment_rate_delta:+.4f}")
print()

summary = comparison_result.summary
print(f"improved {summary.improved_case_ids}   stable {summary.stable_case_ids}"
      f"   worsened {summary.worsened_case_ids}")
print(f"improvement.largest {summary.improvement.largest:+.4f}"
      f"   worsening.largest {summary.worsening.largest:+.4f}")
print()

for case_comparison in comparison_result.case_comparison_results:
    print(f"case {case_comparison.case_id}: {case_comparison.baseline_score:.4f}"
          f" -> {case_comparison.candidate_score:.4f}"
          f"  {case_comparison.status}")
```

```
average_score_delta              +0.1667
median_score_delta               +0.3333
average_criterion_score_delta    +0.4000
criteria_fulfillment_rate_delta  +0.1250

improved [3, 1]   stable [2]   worsened [4]
improvement.largest +0.8333   worsening.largest -0.3333

case 1: 0.8333 -> 1.0000  improved
case 2: 1.0000 -> 1.0000  stable
case 3: 0.0000 -> 0.8333  improved
case 4: 0.3333 -> 0.0000  worsened
```

`improved_case_ids` is ordered by how far each case moved, largest first, which is why it
reads `[3, 1]` rather than `[1, 3]`. `stable_case_ids` is in case order.

The average went up by 0.17 and the headline looks like a clean win. Case 4 got worse
though, so the comparison also reports the criterion grain, which says which single
requirement was lost.

```python
regressed = next(
    one for one in comparison_result.case_comparison_results if one.status == "worsened"
)
print(f"case {regressed.case_id}, criterion by criterion:")
for criterion_comparison in regressed.criterion_comparison_results:
    print(f"  criterion {criterion_comparison.criterion_id}"
          f" (weight {criterion_comparison.weight:g}):"
          f" {criterion_comparison.baseline_score:g} -> {criterion_comparison.candidate_score:g}"
          f"  {criterion_comparison.status}")
```

```
case 4, criterion by criterion:
  criterion 41 (weight 2): 0 -> 0  stable
  criterion 42 (weight 1): 2 -> 0  worsened
```

The new answer talks about flexibility and drops the manager agreement the old one
mentioned. That is the reason a comparison reports three grains instead of one number.

Two runs are comparable only when they cover the same case ids, with the same criterion
ids, the same weights, the same labels and the same scale. Only the answers may differ,
which is why `candidate_run` is built from the baseline cases with `model_copy`. Anything
else raises `RunsNotComparableError`.

One more trap worth knowing. Both runs above have exactly one case at zero, so
`cases_with_score_zero_count_delta` is `0` even though it is a different case on each side.
That field counts cases, it does not identify them.

---

## Over HTTP

The service is the library behind a different door. Same judge, same scoring, same result
types, and every score below is identical to the one the library section produced for the
same case. Nothing is stored. There are no run ids, no catalog on the server and no
persistence, so a result exists only in the response you are holding.

### Configuring the service

```bash
RUBRIC_JUDGE_TEMPERATURE=0.0 .venv/bin/uvicorn rubric_judge.api:app --env-file .env --port 8001
```

The variables are the ones in [Configure](#configure), and `--env-file` is uvicorn reading
`.env` into the process for you, which is the one place you do not have to load it
yourself. An exported variable wins over the file.

Port 8001 rather than 8000 only because 8000 was occupied on the machine these examples ran
on. Generated, interactive API docs are served at `/docs`. The container from
[With Docker](#with-docker) serves the same API on the same port, so every example below
works against it unchanged.

The service proves its judge before it serves. It builds the judge from the variables and sends
it one small call with the configured model and temperature, which costs at most 16 reply
tokens. A missing, empty or unknown variable, a refused key, an unknown model and an endpoint
that does not answer within `RUBRIC_JUDGE_MAX_ATTEMPTS` each stop uvicorn with exit code 3, and
the traceback it prints ends with the cause. Here only the model is set.

```bash
RUBRIC_JUDGE_MODEL=gpt-5.4-mini .venv/bin/uvicorn rubric_judge.api:app --port 8001
```

```
    raise RuntimeError(f"Unusable environment variables: {', '.join(unusable)}")
RuntimeError: Unusable environment variables: RUBRIC_JUDGE_ENDPOINT is missing, RUBRIC_JUDGE_API_KEY is missing

ERROR:    Application startup failed. Exiting.
```

With no `RUBRIC_JUDGE_*` variable at all, the service starts in compare-only mode instead.
`POST /compare` works, and the two endpoints that need a judge answer `503` naming the
variables to set. A single variable, like the model above, already means a judge was intended,
which is why that start was refused rather than put into this mode.

```bash
.venv/bin/uvicorn rubric_judge.api:app --port 8001
```

```
INFO:     Started server process [80494]
INFO:     Waiting for application startup.
No RUBRIC_JUDGE_* variable is set, so this service starts without a judge. POST /compare works, POST /evaluate and POST /evaluate/run answer 503.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
```

```bash
curl -sS -X POST http://localhost:8001/evaluate \
  -H 'Content-Type: application/json' \
  --data '{"id": 1, "answer": "x", "criteria": [{"id": 1, "content": "y", "weight": 1}]}'
```

```json
{"detail":"This service was started without a judge. Set RUBRIC_JUDGE_ENDPOINT, RUBRIC_JUDGE_API_KEY and RUBRIC_JUDGE_MODEL and restart it to evaluate."}
```

The service is open to anyone who can reach it. Set `RUBRIC_JUDGE_ACCESS_TOKEN` and every
endpoint except `/health` answers `401` unless the request carries that token as
`Authorization: Bearer <token>`, the scheme vLLM and the OpenAI API use too. `/health` stays
open so that an orchestrator can probe the service without knowing the token, and `/docs`
stays open so that you can read it. Pick a long random value, for example with
`openssl rand -hex 32`, and serve the port over HTTPS, since the token travels in the clear
otherwise.

```bash
RUBRIC_JUDGE_ACCESS_TOKEN=s3cret .venv/bin/uvicorn rubric_judge.api:app --env-file .env --port 8001
```

A request without the token, or with a different one, is refused before its body is
validated, so a stranger learns nothing about the service. Only a body that is not JSON at all
is answered `422` first, because the framework parses it before anything else runs.

```bash
curl -sS -i -X POST http://localhost:8001/compare -H 'Content-Type: application/json' --data '{}'
```

```
HTTP/1.1 401 Unauthorized
date: Wed, 23 Sep 2026 10:11:43 GMT
server: uvicorn
www-authenticate: Bearer
content-length: 70
content-type: application/json

{"detail":"Send the access token as 'Authorization: Bearer <token>'."}
```

With the header, the same request goes through. Every example below works against a locked
service once you add `-H 'Authorization: Bearer <token>'` to it. In `/docs` the Authorize
button sets the header for you.

```bash
jq -n \
  --slurpfile baseline examples/run_result_baseline.json \
  --slurpfile candidate examples/run_result_candidate.json \
  '{baseline: $baseline[0], candidate: $candidate[0]}' \
| curl -sS -X POST http://localhost:8001/compare \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer s3cret' --data @- \
| jq '.metrics_delta.average_score_delta'
```

```
0.16666666666666663
```

The token works in compare-only mode as well. Set to nothing but whitespace, it stops the
service rather than leaving it open, because an empty line is a lock you meant to set.

```bash
RUBRIC_JUDGE_ACCESS_TOKEN= .venv/bin/uvicorn rubric_judge.api:app --port 8001
```

```
RuntimeError: Unusable environment variables: RUBRIC_JUDGE_ACCESS_TOKEN is empty

ERROR:    Application startup failed. Exiting.
```

### Health

```bash
curl -sS http://localhost:8001/health
```

```json
{"status":"ok","judge":"ok"}
```

`status` is `ok` exactly when the answer is a `200`, and `judge` says why.

| Status | Body | Meaning |
|---|---|---|
| `200` | `{"status":"ok","judge":"ok"}` | The judge answered its last call |
| `503` | `{"status":"unhealthy","judge":"failing"}` | The endpoint refused the key, the model or the URL, a periodic check found the model no longer listed or the endpoint unreachable, or the check crashed. The cause is in the server log and never in the body. A crashed check stays unhealthy until a restart |
| `200` | `{"status":"ok","judge":"none"}` | Compare-only mode, started without any `RUBRIC_JUDGE_*` variable |
| `200` | `{"status":"ok","judge":"custom"}` | A judge installed in code by overriding `get_judge`, see [the reference](docs/REFERENCE.md#functions). Its health is yours to watch |

`/health` never calls the judge itself, so it answers at once and never times out a probe. How
the judge stays proven after startup is in [A judge that stops working](#a-judge-that-stops-working).

### 1. One answer

```bash
curl -sS -X POST http://localhost:8001/evaluate \
  -H 'Content-Type: application/json' \
  --data @- <<'JSON'
{
  "id": 1,
  "context": "The question asked was: How do I report sick leave?",
  "answer": "Call your line manager as early as you can on the first day you are ill, at the latest before 9:00. If you cannot reach them, leave a voicemail and send a short message as well.",
  "criteria": [
    {"id": 11, "content": "Tells the employee to inform their line manager.", "weight": 3},
    {"id": 12, "content": "Names the deadline: before 9:00 on the first day of absence.", "weight": 2},
    {"id": 13, "content": "Says a doctor's note is needed from the fourth day of absence.", "weight": 1}
  ],
  "labels": ["procedure"]
}
JSON
```

The response carries the judge's reasoning for every criterion, so it is long. Piping it
through `jq` gives the part you came for.

```bash
curl … | jq '{score, grades: [.criterion_results[] | {criterion_id, score, is_present}]}'
```

```json
{
  "score": 0.8333333333333333,
  "grades": [
    { "criterion_id": 11, "score": 2.0, "is_present": true },
    { "criterion_id": 12, "score": 2.0, "is_present": true },
    { "criterion_id": 13, "score": 0.0, "is_present": false }
  ]
}
```

Same `0.8333` the library gave for this case.

`context` is optional here exactly as it is in Python. Leave the field out and the answer is
judged against its criteria alone.

```bash
curl -sS -X POST http://localhost:8001/evaluate \
  -H 'Content-Type: application/json' \
  --data @- <<'JSON'
{
  "id": 2,
  "answer": "The Cologne office has an underground garage with 40 spots. Employees reserve a spot through the facility portal, at the latest on the day before. Visitors are registered at reception.",
  "criteria": [
    {"id": 21, "content": "Says what a parking spot costs per month.", "weight": 3},
    {"id": 22, "content": "Says how an employee reserves a spot.", "weight": 2},
    {"id": 23, "content": "Names the deadline for a reservation.", "weight": 1}
  ]
}
JSON
```

```json
{
  "score": 0.5,
  "grades": [
    { "criterion_id": 21, "score": 0.0, "is_present": false },
    { "criterion_id": 22, "score": 2.0, "is_present": true },
    { "criterion_id": 23, "score": 2.0, "is_present": true }
  ]
}
```

<details>
<summary>The full response body</summary>

```json
{"case_id":1,"score":0.8333333333333333,"scale":{"maximum":2,"presence_threshold":0.5,"level_descriptions":{"2":"Fully covered. Every essential part of the criterion is clearly recognizable in the answer, even if the wording, terminology or structure differs.","1":"Partially covered. Some essential information is missing, but the basic idea is still derivable from the answer.","0":"Not covered. The criterion is absent, or the answer has no recognizable connection to it."}},"criterion_results":[{"criterion_id":11,"weight":3.0,"score":2.0,"is_present":true,"reasoning":"The answer explicitly instructs the employee to call their line manager, which directly matches the criterion. The timing and fallback instructions add detail, but the core requirement to inform the line manager is clearly present."},{"criterion_id":12,"weight":2.0,"score":2.0,"is_present":true,"reasoning":"The answer explicitly states the deadline “at the latest before 9:00” and also anchors it to “on the first day you are ill,” which matches the criterion closely. The required deadline is clearly present, with no essential part missing."},{"criterion_id":13,"weight":1.0,"score":0.0,"is_present":false,"reasoning":"The answer explains how to notify the line manager about sick leave and gives a deadline, but it does not mention any doctor's note requirement. The specific condition that a doctor's note is needed from the fourth day of absence is absent, so the criterion is not covered."}],"labels":["procedure"]}
```

The `scale` travels with the result, which is what lets a stored document still say what its
2 out of 2 was worth.

</details>

### 2. A whole run

`POST /evaluate/run` takes the catalog as one document. The body is the four cases from the
library section, so it is long, and the whole run is one HTTP request with one timeout.

<details>
<summary>The request body, complete and pasteable</summary>

```bash
curl -sS -X POST http://localhost:8001/evaluate/run \
  -H 'Content-Type: application/json' \
  --data @- <<'JSON'
{
  "cases": [
    {
      "id": 1,
      "context": "The question asked was: How do I report sick leave?",
      "answer": "Call your line manager as early as you can on the first day you are ill, at the latest before 9:00. If you cannot reach them, leave a voicemail and send a short message as well.",
      "criteria": [
        {"id": 11, "content": "Tells the employee to inform their line manager.", "weight": 3},
        {"id": 12, "content": "Names the deadline: before 9:00 on the first day of absence.", "weight": 2},
        {"id": 13, "content": "Says a doctor's note is needed from the fourth day of absence.", "weight": 1}
      ],
      "labels": ["procedure"]
    },
    {
      "id": 2,
      "context": "The question asked was: How do I get the expenses for a business trip reimbursed?",
      "answer": "File the claim in the expense tool with your receipts attached. It has to be in within 30 days of the day the trip ended.",
      "criteria": [
        {"id": 21, "content": "Says the claim is filed in the expense tool.", "weight": 3},
        {"id": 22, "content": "Names the deadline: within 30 days of the end of the trip.", "weight": 1}
      ],
      "labels": ["procedure"]
    },
    {
      "id": 3,
      "context": "The question asked was: How many vacation days do I get per year?",
      "answer": "You get a generous amount of paid vacation, and most colleagues take theirs in summer. Your remaining balance is shown in the HR tool.",
      "criteria": [
        {"id": 31, "content": "Gives the number of vacation days per year.", "weight": 3},
        {"id": 32, "content": "Says up to 5 unused days carry over into the next year.", "weight": 2},
        {"id": 33, "content": "Says carried-over days expire at the end of March.", "weight": 1}
      ],
      "labels": ["policy"]
    },
    {
      "id": 4,
      "context": "The question asked was: Can I work from home?",
      "answer": "Yes, working from home is possible. Agree the days with your line manager beforehand.",
      "criteria": [
        {"id": 41, "content": "Gives the maximum number of home office days per week.", "weight": 2},
        {"id": 42, "content": "Says the days have to be agreed with the line manager.", "weight": 1}
      ],
      "labels": ["policy"]
    }
  ]
}
JSON
```

</details>

```bash
curl … | jq '{cases: [.case_results[] | {case_id, labels, score}], average: .metrics.average_score, fulfillment: .metrics.criteria_fulfillment_rate, zero: .metrics.cases_with_score_zero, per_label: [.label_metrics[] | {label, average: .metrics.average_score}]}'
```

```json
{
  "cases": [
    { "case_id": 1, "labels": ["procedure"], "score": 0.8333333333333333 },
    { "case_id": 2, "labels": ["procedure"], "score": 1.0 },
    { "case_id": 3, "labels": ["policy"], "score": 0.0 },
    { "case_id": 4, "labels": ["policy"], "score": 0.3333333333333333 }
  ],
  "average": 0.5416666666666666,
  "fulfillment": 0.5416666666666666,
  "zero": [3],
  "per_label": [
    { "label": "policy", "average": 0.16666666666666666 },
    { "label": "procedure", "average": 0.9166666666666666 }
  ]
}
```

Every one of those numbers equals the library run. The whole response document is committed
as [examples/run_result_baseline.json](examples/run_result_baseline.json).

Note that the per-label buckets come back on the result without being asked for, which is
the HTTP equivalent of calling `label_metrics` yourself.

### 3. Slicing a run by label

Add a `label_filter` to the same body and only the cases it selects are judged, which is
also all you pay for.

```json
  "label_filter": [["policy"]]
```

```bash
curl … | jq '{applied_label_filter, cases: [.case_results[] | {case_id, score}], average: .metrics.average_score}'
```

```json
{
  "applied_label_filter": [["policy"]],
  "cases": [
    { "case_id": 3, "score": 0.0 },
    { "case_id": 4, "score": 0.3333333333333333 }
  ],
  "average": 0.16666666666666666
}
```

The filter is recorded on the result as `applied_label_filter`, so a stored document says
which slice of the catalog it describes rather than pretending to be the whole run.

A filter that matches nothing is refused rather than returning an empty run, and the refusal
names the labels the catalog does carry.

```json
{"detail":[{"loc":["body"],"msg":"Value error, label_filter [['polizy']] matches no case; labels present in this run: policy (2), procedure (2)","type":"value_error"}]}
```

### 4. Comparing two runs

`POST /compare` needs no judge, so it costs nothing and is instant. Both committed example
documents can be posted straight from disk.

```bash
jq -n \
  --slurpfile baseline examples/run_result_baseline.json \
  --slurpfile candidate examples/run_result_candidate.json \
  '{baseline: $baseline[0], candidate: $candidate[0]}' \
| curl -sS -X POST http://localhost:8001/compare \
    -H 'Content-Type: application/json' --data @- \
| jq '{metrics_delta, improved: .summary.improved_case_ids, stable: .summary.stable_case_ids, worsened: .summary.worsened_case_ids, cases: [.case_comparison_results[] | {case_id, baseline_score, candidate_score, status}]}'
```

```json
{
  "metrics_delta": {
    "average_score_delta": 0.16666666666666663,
    "median_score_delta": 0.33333333333333337,
    "variance_delta": 0.018518518518518517,
    "standard_deviation_delta": 0.01974934160432723,
    "average_criterion_score_delta": 0.3999999999999999,
    "criteria_fulfillment_rate_delta": 0.125,
    "cases_with_score_zero_count_delta": 0
  },
  "improved": [3, 1],
  "stable": [2],
  "worsened": [4],
  "cases": [
    { "case_id": 1, "baseline_score": 0.8333333333333333, "candidate_score": 1.0, "status": "improved" },
    { "case_id": 2, "baseline_score": 1.0, "candidate_score": 1.0, "status": "stable" },
    { "case_id": 3, "baseline_score": 0.0, "candidate_score": 0.8333333333333333, "status": "improved" },
    { "case_id": 4, "baseline_score": 0.3333333333333333, "candidate_score": 0.0, "status": "worsened" }
  ]
}
```

That is the whole point of a run result being a plain document. A run stored months ago
compares against one produced a second ago, with no server that has to remember either.
The full response is committed as
[examples/run_comparison_result.json](examples/run_comparison_result.json).

### When a request fails

Every failure names its cause in the body. The full table of situations, exceptions and
statuses is in [the reference](docs/REFERENCE.md).

| Status | Situation | Body |
|---|---|---|
| `401` | The service was started with `RUBRIC_JUDGE_ACCESS_TOKEN` and the request did not carry it | `{"detail":"Send the access token as 'Authorization: Bearer <token>'."}` |
| `422` | The document is invalid, here a weight of 0 | `{"detail":[{"loc":["body","criteria",1,"weight"],"msg":"Input should be greater than 0","type":"greater_than"}]}` |
| `422` | A label filter matching no case | `{"detail":[{"loc":["body"],"msg":"Value error, label_filter [['polizy']] matches no case; labels present in this run: policy (2), procedure (2)","type":"value_error"}]}` |
| `422` | Two runs that do not describe the same catalog | `{"detail":"the runs are not comparable: case 4, criterion 42: weight 1.0 vs 2.0"}` |
| `503` | The judge could not answer, so the whole run is dropped | `{"detail":"Judge gave no usable answer in 3 attempts: Connection error."}` |
| `503` | `/evaluate` or `/evaluate/run` in compare-only mode | `{"detail":"This service was started without a judge. Set RUBRIC_JUDGE_ENDPOINT, RUBRIC_JUDGE_API_KEY and RUBRIC_JUDGE_MODEL and restart it to evaluate."}` |
| `500` | The judge endpoint rejects the key or the model, or the program has a bug | `Internal Server Error`, with the cause in the server log. A rejected key or model also turns `/health` unhealthy |

A healthy service does not hand you a `503` on request. The one above was produced by
starting a second server pointed at a dead endpoint, which is worth knowing if you intend to
test your own handling of it.

---

## Going further

### Your own prompt

`judge_prompt` writes the system prompt from a `Scale`, and `OpenAIJudge` sends it. You can
read exactly what your judge is being told.

```python
from rubric_judge import DEFAULT_SCALE, judge_prompt

print(judge_prompt(DEFAULT_SCALE))
```

```
You are a careful examiner.

You will be given an answer produced by some system and a single criterion, and
sometimes the context the answer was produced in. Your task is to decide to what
degree the criterion is covered by the answer. When there is no context, judge the
answer on its own terms rather than assuming something was left out.

Use this 0-2 scale:

2 = Fully covered. Every essential part of the criterion is clearly recognizable
    in the answer, even if the wording, terminology or structure differs.
1 = Partially covered. Some essential information is missing, but the basic idea
    is still derivable from the answer.
0 = Not covered. The criterion is absent, or the answer has no recognizable
    connection to it.

Judge only the criterion you are given. Do not reward correct information that
belongs to a different criterion, and do not punish it either.

First write a short, neutral argument for which score fits the scale above. Only
then decide, and output the decision as a JSON object on its own line. Never use
markdown, never use code fences.

Your reply must always look like this:

[Two or three sentences arguing which score the scale calls for.]
{"score": 0, 1 or 2}
```

The three level lines are not written into that text by hand. They are
`DEFAULT_SCALE.level_descriptions`, rendered in, which is why a scale and the prompt
explaining it cannot drift apart.

`OpenAIJudge(config, system_prompt=...)` replaces it. Your text has to keep two promises or
every reply fails to parse. The model argues first and closes with a single
`{"score": <grade>}` object on its own line, and the scale it describes is the scale the
judge was built with.

A custom prompt is for encoding a grading policy that differs from the bundled one, or for
domain vocabulary a general examiner does not have. It is not a lever for rescuing a vague
criterion. Sharpen the criterion for that, which is what the two rewritten criteria in this
manual's catalog are.

### Another scale

A scale is a maximum, a presence threshold and one sentence per grade. Describe every grade
and the prompt writes itself.

```python
from rubric_judge import Scale

FIVE_LEVELS = Scale(
    maximum=4,
    presence_threshold=2,
    level_descriptions={
        4: "Stated in full, including every number, deadline and named party the criterion has.",
        3: "Stated, but one detail of it is vague or one named party is missing.",
        2: "The requirement is recognizable, and the specifics are missing.",
        1: "Only the topic is touched; the requirement itself is not stated.",
        0: "Not covered. The criterion is absent from the answer.",
    },
)

judge = OpenAIJudge(JudgeConfig.from_env(), scale=FIVE_LEVELS)
```

Raw grades then come back on your scale while the case score stays in the range 0 to 1,
because every grade is divided by `scale.maximum` before the weights fold it together. The
scale rides along on the result, so a run read back next year still says what its 2 of 4
meant.

More levels buy the judge room to separate "stated but vague" from "stated in full" on
answers that do state something. They do not rescue an answer that states nothing. Case 4
scores `0.3333` on both the bundled scale and the five-level one, because the answer gives
no figure and level 1, "only the topic is touched", was on offer and not taken.

Two scales are refused, both before any request is sent, so neither costs a judge call.

Level descriptions that do not cover every grade, because a prompt explaining four of ten
levels is worse than one explaining none.

```python
Scale(maximum=2, presence_threshold=1, level_descriptions={2: "Covered.", 0: "Not covered."})
```

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for Scale
  Value error, level descriptions must describe every grade of the scale 0..2 and no other, got [0, 2]
```

And a scale with no level descriptions used without a prompt of your own, because an
undescribed scale is arithmetic that no model has been told about.

```python
OpenAIJudge(JudgeConfig.from_env(), scale=Scale(maximum=10, presence_threshold=5))
```

```
ValueError: the scale 0..10 (covered from 5.0) describes no levels, so no prompt can be
written from it: give it a level_descriptions entry per grade, or pass a prompt of your own
```

Refused at construction rather than at the first reply. A model told 0 to 2 while its
answers are checked against 0 to 10 would fail every criterion of every case, one paid call
at a time, and the run would still come back looking like a bad system.

### Your own judge

The `Judge` protocol is one attribute and one method. No base class, no registration.

```python
class MyJudge:
    scale: Scale

    async def score(
        self, answer: str, criterion: Criterion, context: str | None = None
    ) -> JudgeReply:
        ...
```

`context` is `None` whenever the case carries none, which is every answer that stands on its
own. Read it as background and never score it, and judge the answer on its own terms when it
is absent.

Return a `JudgeReply` or raise. `JudgeUnavailableError` means the endpoint could not answer
and invalidates the run. Anything else is read as a bug in the program.

The one most people want first replays recorded grades. Everything above a grade is
arithmetic, so pinning the grades turns the scoring, the weighting, the metrics and the
comparison into a deterministic test that still runs the real `evaluate_run`, with no key,
no network and no cost.

```python
class RecordedJudge:
    """Replays grades recorded from a real run, so a test can exercise the whole pipeline.

    Scoring, weighting, the metrics and the comparison are arithmetic over grades. Pinning
    the grades makes every number below them a deterministic test with no key, no network
    and no cost, while still going through the same `evaluate_run` production uses.

    Example:
        judge = RecordedJudge({11: 2, 12: 2, 13: 0})
        case_result = await evaluate_case(judge, sick_leave_case)
        case_result.score   # 0.8333333333333333
    """

    scale = DEFAULT_SCALE

    def __init__(self, grade_per_criterion_id: dict[int, int]):
        """Record one grade per criterion id, on `DEFAULT_SCALE`."""
        self.grade_per_criterion_id = grade_per_criterion_id

    async def score(
        self, answer: str, criterion: Criterion, context: str | None = None
    ) -> JudgeReply:
        """Hand back the recorded grade, or fail loudly when the rubric outgrew the table."""
        if criterion.id not in self.grade_per_criterion_id:
            raise LookupError(f"no grade recorded for criterion {criterion.id}")
        return JudgeReply(
            score=self.grade_per_criterion_id[criterion.id],
            reasoning="recorded grade, no model was asked",
        )


BASELINE_GRADES = {11: 2, 12: 2, 13: 0, 21: 2, 22: 2, 31: 0, 32: 0, 33: 0, 41: 0, 42: 2}

run_result = asyncio.run(evaluate_run(RecordedJudge(BASELINE_GRADES), baseline_run))
for case_result in run_result.case_results:
    print(f"case {case_result.case_id} {case_result.labels}: {case_result.score:.4f}")
print(f"average_score  {run_result.metrics.average_score:.4f}")
```

```
case 1 ['procedure']: 0.8333
case 2 ['procedure']: 1.0000
case 3 ['policy']: 0.0000
case 4 ['policy']: 0.3333
average_score  0.5417
```

Those are the committed baseline numbers, reproduced exactly with zero judge calls, because
`BASELINE_GRADES` holds the grades `examples/run_result_baseline.json` records.

A judge that grades above the scale it declares is caught rather than clamped, because a
clamped grade is a wrong number reported as a right one.

```
ValueError: criteria [11, 12, 13] scored above the scale 0..2 (covered from 0.5) they were
judged on
```

Every offending criterion is named at once, with the scale it was held against. That is a
`ValueError` and not a `JudgeUnavailableError`, and over HTTP a `500`, because a judge
disagreeing with its own declared scale is a bug in that judge rather than an outage at its
endpoint.

---

## How scoring works

### The scale

The judge sees the answer, **one** criterion and the context if the case has one, and
returns one integer.
Out of the box that is `DEFAULT_SCALE`, which runs from 0 to 2.

| Grade | Meaning |
|---|---|
| `2` | Fully covered. Every essential part is recognizable, even if worded differently |
| `1` | Partially covered. Essential information is missing, but the idea is derivable |
| `0` | Not covered. Absent, or with no recognizable connection to the criterion |

That table is not built into the prompt by hand. It **is**
`DEFAULT_SCALE.level_descriptions`, and the judge's system prompt is generated from it.
Anything at or above `presence_threshold`, `0.5` on this scale, sets `is_present`.

The scale belongs to the judge rather than to the package, so a judge of your own can grade
from 0 to 5 or from 0 to 10 with meanings of its own, which
[Another scale](#another-scale) shows. Every `CaseResult` stores the scale it was judged
on, so a stored result stays readable years later.

Raw criterion grades are therefore always reported in the judge's own units, while
`CaseResult.score` and `RunMetrics.average_score` are normalized to the range 0 to 1 and
stay comparable across scales.

### One case

Each criterion grade is turned into a fraction of what it could have reached, weighted, and
normalized over the sum of the weights.

```
score = Σᵢ ( wᵢ · sᵢ / max ) / Σᵢ wᵢ         ∈ [0, 1]
```

`max` is `CaseResult.scale.maximum`, and dividing by it is what makes the case score
scale-free. Half marks everywhere is `0.5` whether the judge counted in halves or in
fifths.

Case 1 from the catalog above, worked through. Three criteria with weights 3, 2 and 1,
graded 2, 2 and 0.

| Criterion | Weight `w` | Grade `s` | `w · s / 2` |
|---|---|---|---|
| 11 | 3 | 2 | 3.0 |
| 12 | 2 | 2 | 2.0 |
| 13 | 1 | 0 | 0.0 |
| | **6** | | **5.0** |

Five points out of six reachable, so `case_score()` returns `0.8333333333333333`, which is
the `0.83` the manual prints for case 1. Only the ratios matter, so weights of 30, 20 and
10 give the same number.

Every `s` in that sum is a grade the judge really gave. A criterion it could not answer for
has no grade and is given none. The run is dropped instead, which
[A dead judge invalidates the run](#a-dead-judge-invalidates-the-run) explains.

### One run

`RunMetrics` aggregates the case scores, and every field of it is defined in
[the reference](docs/REFERENCE.md). Two of them are easy to misread.

`average_criterion_score` is not `average_score` on a different scale. It ignores weights
and case boundaries entirely. It answers how well the judge rates an average statement,
not how good the average answer is.

`criteria_fulfillment_rate` is averaged per case first and then over the cases. Flattening
all criteria into one pool would let a single case with a twenty-criterion rubric dominate
nineteen short ones.

---

## Failure and load

### Repeatability

`temperature = 0.0` makes a judge as repeatable as its endpoint is willing to be, and
nothing here can make that a guarantee. A hosted endpoint batches your call with other
people's, floating point addition is not associative on a GPU, and the model behind a name
gets replaced without asking you.

In practice the wobble is narrower than "the numbers move". The baseline run of this
manual was judged three times over, with a fresh judge each time, and all ten criteria came
back with the same grade in all three. The reasoning prose was worded differently between
them. Temperature 0 pins the decision, not the text.

It took work to be able to write that sentence, and the work is the lesson. Two criteria of
this catalog originally read *"States the annual entitlement: 30 vacation days"* and
*"States the limit: up to 2 days of home office per week"*, against answers that circle the
topic without naming the figure. Across nine samples of one of them the judge answered
"not covered" twice and "partially covered, the idea is derivable" seven times. Both
readings are defensible, which is precisely why the judge could not hold still. They now
read *"Gives the number of vacation days per year"* and *"Gives the maximum number of home
office days per week"*. An answer containing no figure cannot partially contain one, and the
wobble is gone.

A longer measurement, taken on an **earlier three-case catalog** that is not the one in
this manual, shows the same thing from the other side. Four consecutive runs against a
hosted endpoint at temperature `0.0`, 18 criterion judgements per run. Every clear-cut
criterion returned the same grade every time, and exactly one wavered, the one whose grade
is genuinely arguable. It read *"State the expected last day of absence"*, against an answer
reading *"Send an email to hr@example.com before 10:00 on your first day of absence."* The
answer names a day, but not that day, and a human reviewer would hesitate too.

| Scale | Grades over the four runs | Case score |
|---|---|---|
| 0 to 2 | `0`, `0`, `0`, `1` | `0.750` three times, `0.875` once |
| 0 to 3 | `0`, `1`, `1`, `0` | `0.750` twice, `0.833` twice |

Note which explanation that rules out. It is not a matter of scale granularity, because the
same criterion wavered on both scales while the sharp criteria stayed put on both.

It does not waver on demand either. Four further runs of the identical catalog three days
later, same model, same temperature, 18 judgements each, returned all 72 grades identical,
the arguable criterion included. That is the honest shape of it. A run is usually
repeatable, one arguable criterion can move, and a single run cannot tell you which of the
two you are holding.

What it costs a comparison is the point. That one grade decided whether the run's headline
read `average_score_delta = +0.083` or `+0.042`, a factor of two on the number the whole
run gets reported by, with nothing about the system under test changed. So:

- A borderline criterion is a measuring instrument with a loose needle. Phrasing each
  criterion as one checkable fact is not only about the judge picking a compromise grade,
  it is what makes the grade repeatable at all.
- Read a one-grade criterion move against its weight before calling it a regression.
- Re-run the baseline before you believe a small delta. Two runs of an unchanged system
  measure your noise floor, and that is the cheapest way to learn which deltas mean
  anything.
- Nothing in a result measures this for you. Each criterion is judged exactly once, and no
  field reports how far that grade would move on a re-run, so a run does not tell you how
  steady its own numbers are.

### Unparseable replies heal themselves

If the judge replies with no JSON, with malformed JSON, or with a grade off the integral
scale such as `3` or `1.5`, the concrete cause is fed back to it together with its own
broken reply, and the call is repeated up to `RUBRIC_JUDGE_MAX_ATTEMPTS` times. It is not a
blind retry. The model is told what was wrong with what it wrote.

A refused connection, a timeout, a `429` or a `5xx` costs an attempt from the same budget
and is waited out instead, `0.5s` after the first failure and doubling after each further
one. There is nothing to correct in the conversation, so the criterion is simply asked
again.

One thing is never retried. A reply with no content at all, which truncation, a content
filter or a tool-call path can produce, fails the run immediately and names the endpoint's
own `finish_reason`. No rewording fixes any of those.

### A dead judge invalidates the run

A criterion the judge could not answer for gets no result, and neither does anything around
it. `POST /evaluate` answers `503`, `POST /evaluate/run` drops the whole run including the
cases that were already judged, and the library functions raise `JudgeUnavailableError`.

That is a deliberate reversal of the obvious alternative, which would be to score the
criterion `0` and carry on. A `0` nobody judged is indistinguishable from an answer that
really missed the criterion, so the run would come back looking like a finished measurement
and reading like a bad system. The metrics average the cases against each other, so a
single invented `0` moves every figure in the document. Nothing is stored server side, so a
retry costs only judge calls.

A bug in the program ends the run too, but as itself, with a `500`. "Your endpoint is down,
try again" and "this program is broken" are different messages to get.

### A judge that stops working

A service proves its judge once before it serves, and every judge call that gets an answer
after that is proof again. A call the endpoint refuses with `401`, `403` or `404` turns
`/health` into a `503` at once, because a refused key, a missing permission or an unknown model
condemns every later call too. The request itself is answered with a `500`. The next judge
call that gets an answer makes the service healthy again.

An outage does not flip it. Timeouts, rate limits and `5xx` answers are retried, then answered
with a `503` per request, and they heal by themselves. Marking every replica unhealthy for a
provider's blip would take the whole service down for it. A `400` does not flip it either,
because one oversized case can earn it while the next case goes through.

A quiet service gets no proof from traffic. Set `RUBRIC_JUDGE_HEALTH_INTERVAL` and it lists the
endpoint's models whenever its judge has gone that many seconds without an answered call,
`86400` for once a day. The list costs no tokens, and it proves the key still works and that
the configured model is still on it. Each such check waits at most 60 seconds for its answer. A
model that fell off the list is final at once, like a rejection. An outage is retried instead,
after 5, 10 and 20 minutes by default (`RUBRIC_JUDGE_HEALTH_RETRIES`,
`RUBRIC_JUDGE_HEALTH_FIRST_PAUSE`), and the service stays healthy while it waits. A judge call
answered in the meantime ends the wait. Only when every attempt failed does the service report
unhealthy, and the next check runs one interval later rather than in a loop. Left unset, the
service relies on its startup check and its traffic alone, and a key revoked while nobody calls
goes unnoticed until the next request.

The startup check sends one real completion instead, with the judge's model and temperature,
because only a completion proves that the endpoint accepts a request shaped like a judge call.
A model on the list does not prove that, and nothing about the request changes while the
service runs. On Azure OpenAI the model list names base models rather than deployment names, so
there the periodic check reports a working deployment as unhealthy. Leave
`RUBRIC_JUDGE_HEALTH_INTERVAL` at `0` on Azure.

The longest an outage can take to show is therefore the interval, plus four attempts of 60
seconds and 35 minutes of pauses, plus the 90 seconds Docker takes to see three failed probes
in a row. The startup check does not wait like this. It makes `RUBRIC_JUDGE_MAX_ATTEMPTS`
attempts, pausing 0.5 and then 1 second between the default 3, and then stops the service, so a
container started during an outage exits and your restart policy tries again. Each attempt waits
at most 60 seconds, so an endpoint that accepts the connection and never answers holds the start
for about three minutes with the default of 3 attempts.

### Concurrency

The criteria of a case are judged concurrently, but at most
`RUBRIC_JUDGE_MAX_CONCURRENT` calls are ever in flight. A rubric with 200 criteria costs 8
open connections, not 200.

The limit sits on the judge rather than on the fan-out. One judge instance serves the whole
process, so the budget is shared by everything using it.

| Situation | Coroutines | Connections in flight |
|---|---|---|
| one case, 200 criteria | 200 | 8 |
| 50 cases of 10 criteria in one run | 500 | 8 |
| five concurrent runs | thousands | 8 |
| ten parallel `POST /evaluate` | any | 8 in total, not 8 each |

A slot is held for one HTTP call only, so a criterion waiting to be retried does not occupy
one, and a run is still judged with its cases overlapping rather than one after another.

A run deliberately gets no second throttle. Only the judge implementation knows what its
backend tolerates, so that is the single place the limit lives. Raise
`RUBRIC_JUDGE_MAX_CONCURRENT`, not the run size.

---

## Where to go next

[docs/REFERENCE.md](docs/REFERENCE.md) has every input and output type field by field,
every function signature, every endpoint and the full error table.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the vocabulary, the reasoning behind the
shape of the code, where a new rule belongs and what the test suite covers.

## Scope

This package judges answers and nothing else. It does not generate answers, retrieve
documents, store runs, schedule anything or serve a user interface. A `RunResult` is a
plain pydantic model, so `model_dump_json()` and `RunResult.model_validate_json()` are the
whole persistence story, and where those bytes live is your decision.

## License

MIT. See [LICENSE](LICENSE).
