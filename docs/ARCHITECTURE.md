# Architecture

Why the code is shaped the way it is. Contributor material, so read
[../README.md](../README.md) first if you only want to use the package, and
[REFERENCE.md](REFERENCE.md) if you want to look a type or an endpoint up.

## The modules

Seven modules, each with one job. A request walks straight down through the first four:

| Step | File | Responsibility |
|---|---|---|
| 1 | [api.py](../src/rubric_judge/api.py) | FastAPI endpoints. Validate the body, inject the judge, hand back JSON. No domain logic |
| 2 | [evaluation.py](../src/rubric_judge/evaluation.py) | `evaluate_case()` and `evaluate_run()`, which fan out over the rubric and fold the results |
| 3 | [judge.py](../src/rubric_judge/judge.py) | `Judge` protocol, OpenAI-compatible client, reply parsing, retries, throttle, environment config |
| 4 | [prompt.py](../src/rubric_judge/prompt.py) | Every word the judge is told, written from the scale |

The other three are read from everywhere and depend on no step:

| File | Responsibility |
|---|---|
| [models.py](../src/rubric_judge/models.py) | Every public type, `Scale` and `DEFAULT_SCALE`, `CriterionResult.judged()` |
| [metrics.py](../src/rubric_judge/metrics.py) | `case_score()`, `run_metrics()` and `label_metrics()`, the formulas and nothing else |
| [comparison.py](../src/rubric_judge/comparison.py) | `compare_runs()`, two finished runs into their differences. Reads no judge and no config |

## The vocabulary

Four grains, each a pair of what goes in and what comes back:

| Grain | In | Out |
|---|---|---|
| one requirement | `Criterion` | `CriterionResult` |
| one answer | `Case` | `CaseResult` |
| a whole catalog | `Run` | `RunResult` |
| two whole runs | `RunComparison` | `RunComparisonResult` |

Then there is `RunMetrics`, which is `RunResult.metrics` and nothing else, `Scale`, which
every case result carries, and `JudgeReply`, which only a judge produces.

`Case` carries `context`, not `question`, and it is optional. A criterion is often checked
against a text that was never a reply to anything at all, a summary, a drafted email, a
report, and requiring a question there forced a caller to either invent one or pass a blank
that meant nothing. `context` is one free-form string that names itself in the caller's own
words, `"The question asked was: ..."`, `"The source policy says: ..."`, rather than a
structured type with a `kind` field and a `value` field. That costs the caller one clause to
say what the text is, and it is what lets a source document or a task instruction reuse the
same field later without a redesign of `Case` or of the judge's prompt, which renders exactly
one label, `Context:`, whatever the caller put after it.

Labels add no grain, they cut across one. `LabelMetrics` and `LabelMetricsDelta` are each a
label plus the aggregate of the grain above, composed rather than copied, so the pair
`RunMetrics` and `RunMetricsDelta` stays the single definition of what a run's numbers are. A
twin that redeclared every field would have to be edited in step with it forever, and the half
that got forgotten would go on serializing a stale number under a familiar name.

The word is "grain" and not "scale", because `Scale` is the grading scale and one word must not
mean two things.

Comparing repeats the same grains one level up, and the names say so. A
`CriterionComparisonResult` sits inside a `CaseComparisonResult` exactly as a `CriterionResult`
sits inside a `CaseResult`. Only a produced type carries `Result`. `RunComparison` is what you
hand in, and everything ending in `Result` is what comes back. That naming is also why
`compare_runs()` takes one named pair rather than two arguments. Both sides have the same type,
so a swap would be undetectable and would invert every sign.

Two rules hold the naming together, and they are worth knowing before adding a field.

A result never reuses the name of its input. `CaseResult.criterion_results` holds results, so
it is not called `criteria`. `Case.criteria` is the rubric, and one name must not mean two
things.

There is exactly one type per grain. A case evaluated alone and a case inside a run are the
same `Case`, and both come back as the same `CaseResult`. That is why `Case.id` is mandatory
rather than optional. An id that is sometimes there would have meant two near-identical types,
two result shapes, and a `null` to check for.

## Where does a new rule go?

The result types draw the layer boundary and answer the question by themselves:

```python
JudgeReply:       score, reasoning                            # what the model replied
CriterionResult:  criterion_id, weight, score, is_present,     # what the system concluded
                  reasoning
CaseResult:       case_id, score, scale, criterion_results,    # one whole case
                  labels
RunMetrics:       the aggregate over many case results
LabelMetrics:     that aggregate again, per label
```

`judge.py` speaks `JudgeReply`. Prompts, parsing and retries are its business, and another
implementation may do all three differently. Everything that has to hold no matter who judges
lives outside it, or two judges would produce incomparable scores. That covers how a grade
becomes a share of the reachable points, the weighting, and the rule that a dead judge
invalidates the run.

The grading scale is the one setting that belongs to the judge and has to be known outside it.
A judge owns it and every `CaseResult` carries a copy, so `metrics.py` and `comparison.py`
never have to ask which judge produced a run they are reading off disk. It is also the reason
`prompt.py` can generate instead of store, because the judge's words and the reader's units
come from the same object.

So nothing above `judge.py` knows what a chat completion is, nothing inside it knows what a
weight is, and `evaluation.py` is the only module that speaks both languages.

## How the code documents itself

Every public function states its inputs, its return value and what it raises, in Google style.
`help(evaluate_run)` in a REPL is the same reference as [REFERENCE.md](REFERENCE.md). Sections
always appear in this order, with the example last:

```
Args:
    judge: As for `evaluate_case`. The same instance serves every case of the run …
Returns:
    A `RunResult`: one `case_results` entry per selected case in request order …
Raises:
    As `evaluate_case`. A programming error in any single case aborts the run …
Example:
    run_result = await evaluate_run(judge, run)
```

Private helpers deliberately do not get that treatment. They are two or three lines with a name
that says what they do, and an `Args:` block there would be noise around the one sentence that
actually matters, which is why the helper exists. A helper that stops being obvious gets the
full block like a public one.

FastAPI endpoint docstrings are user-facing too. They become the descriptions in the Swagger UI
at `/docs`, so they are written for whoever is calling the API and not for whoever is
maintaining it.

Model fields are documented with attribute docstrings rather than comments:

```python
class Criterion(DocumentedModel):
    weight: float = Field(gt=0, allow_inf_nan=False)
    """How much this criterion counts next to the others. Only the ratios matter:
    weights 3 and 1 score exactly like 30 and 10."""
```

Pydantic's `use_attribute_docstrings`, set once on `DocumentedModel`, copies them into the
OpenAPI schema. The same sentence therefore serves IDE hover, the Swagger UI at `/docs` and the
field tables in [REFERENCE.md](REFERENCE.md). One text, never three, so they cannot drift apart.

## Tests

```bash
.venv/bin/python -m pytest
```

346 tests, and no real LLM is ever called. The mocking happens at two levels, with a third
helper for the comparison tests.

`FakeJudge` in [../tests/conftest.py](../tests/conftest.py) replaces the `Judge` protocol and
scores from a lookup table. `{1: 2, 2: JudgeUnavailableError("down")}` scores criterion 1 with
a `2` and lets the judge fail on criterion 2. `CASE` and `RUN` live in the same file, so the
domain tests and the HTTP tests describe literally the same input. `RUN` is deliberately uneven,
scoring `0.75`, `0.5` and `0.0` under `RUN_SCORES`, because uniform scores make most run metrics
indistinguishable.

`StubJudgeEndpoint` in [../tests/test_api.py](../tests/test_api.py) answers the OpenAI
chat-completions format from a script, keyed by criterion text rather than a single queue,
because the criteria are judged concurrently and one queue would hand out replies in a racy
order. The end-to-end tests hand a real `OpenAIJudge` an `AsyncOpenAI` client whose transport is
that stub and drive the whole chain: the HTTP request, the real `openai` SDK writing the call
and reading the reply, the parsing, and the weighted fold. That is what proves the wire format
and the self-healing retry, without a socket or an environment variable.

`run_of` in [../tests/conftest.py](../tests/conftest.py) builds a finished `RunResult` straight
from judge scores. `run_of({1: 2, 2: 0}, {21: 1})` is a two-case run, and
`labels_by_case_id={1: ["table"]}` tags one. Comparison tests are about the difference between
two runs, so going through a judge and an event loop would only stand between the test and the
numbers it asserts on.

| File | Covers |
|---|---|
| [test_metrics.py](../tests/test_metrics.py) | the scoring formula, float extremes, every run metric |
| [test_evaluation.py](../tests/test_evaluation.py) | fan-out, ordering, failure policy, run aggregation |
| [test_judge.py](../tests/test_judge.py) | the parser reply by reply, both retry loops, what is not retried, the concurrency limit |
| [test_comparison.py](../tests/test_comparison.py) | deltas and their direction, the three statuses, ordering, every refusal, and the stored documents in [../examples](../examples) against what the library recomputes from them |
| [test_prompt.py](../tests/test_prompt.py) | the prompt a scale generates, held against the hand-written original in [../tests/judge_prompt_en.txt](../tests/judge_prompt_en.txt) |
| [test_scale.py](../tests/test_scale.py) | what a valid scale is, and what it does to a grade, a run and a comparison |
| [test_labels.py](../tests/test_labels.py) | what a valid label is, which cases a filter and a bucket select, and what a relabelled case does to a comparison |
| [test_api.py](../tests/test_api.py) | validation, wiring, serialization, and the end-to-end chain |

The documents in [../examples](../examples) are real responses of this app driven by a judge
that grades from a table. They are the only place a reader can check the library against
something they did not compute themselves, and a test recomputes the stored comparison from the
two stored runs so that stays true.

## Scope

Implemented: evaluating one case or a whole catalog with run metrics, slicing a run by label,
and comparing two finished runs against each other, as a library or over HTTP.

Not implemented: rubric catalog files, a CLI, negation in a `label_filter` (so "table but not
images" cannot be written), streaming progress for long runs, and self-consistency, meaning
judging each criterion several times and reporting how far the grades spread.
