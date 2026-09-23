# Reference

Every public type, function, endpoint and error of `rubric-judge`, in one place to look
things up in. The manual with the install steps, the quickstarts and the scoring formula is
[../README.md](../README.md).

Everything here is importable from the package root:

```python
from rubric_judge import Case, Criterion, evaluate_case
```

## Inputs

All inputs are Pydantic models. Construct them in Python, or post the same shape as JSON.

### `Criterion`

One requirement a good answer has to satisfy, the atom of a rubric.

| Field | Type | Required | Rules |
|---|---|---|---|
| `id` | `int` | yes | Yours. Echoed back as `CriterionResult.criterion_id`. Must be unique within its case |
| `content` | `str` | yes | The requirement in plain language. Non-empty after stripping whitespace |
| `weight` | `float` | yes | `> 0`, finite. Only the *ratios* matter, so `3` and `1` score exactly like `30` and `10` |

Phrase `content` as one checkable fact. Two facts in one criterion cannot be scored apart,
so the judge has to pick a compromise score:

```python
Criterion(id=1, content="Send an email to hr@example.com", weight=3)             # good
Criterion(id=2, content="Email HR before 10:00 and inform your team", weight=3)  # two facts
```

### `Case`

One answer plus the rubric to hold it against.

| Field | Type | Required | Rules |
|---|---|---|---|
| `id` | `int` | yes | Yours. Echoed back as `CaseResult.case_id`. Unique within its run |
| `context` | `str \| None` | no, default `None` | What the answer was produced in response to, in your own words, for example `"The question asked was: How do I report sick leave?"`. Stripped of surrounding whitespace, and blank after stripping is rejected rather than stored. Never scored. Omit it, or pass `None`, for an answer that stands on its own |
| `answer` | `str` | yes | The answer under test, judged exactly as it comes in. May be empty, because a system that returned nothing is a valid case that scores `0` |
| `criteria` | `list[Criterion]` | yes | At least one. Criterion ids must be unique |
| `labels` | `Labels` | no, default `[]` | What kind of case this is, for example `["table", "images"]`. Your own vocabulary, never checked against a list. Rules under [the label types](#label-labels-and-labelfilter). Never shown to the judge |

`id` is mandatory even for a single evaluation. That is what makes one result type serve both
the single-case and the run path.

### `Run`

Many cases evaluated in one go.

| Field | Type | Required | Rules |
|---|---|---|---|
| `cases` | `list[Case]` | yes | At least one. Case ids must be unique |
| `label_filter` | `LabelFilter` | no, default `[]` | Which of those cases to run, as an OR of ANDs. A case runs when it carries every label of at least one group. `[]` runs all of them. A label filter matching no case is rejected, and the message names the labels the run does carry, with counts |

`cases` is what you have, `label_filter` decides what runs. Post a whole catalog with
`[["table", "split_infos"], ["agentic"]]` and the run covers the cases carrying both `table`
and `split_infos`, plus the cases carrying `agentic`. Negation cannot be written.

On a `Run` this field is an instruction. The `RunResult` carries the same groups back as
`applied_label_filter`, a record of what ran, and there it is validated.

### `Label`, `Labels` and `LabelFilter`

Constrained aliases rather than models, so a label obeys one rule wherever it appears, on a
`Case`, on a `CaseResult`, in a `Run.label_filter` or in a call to `filter_cases_by_labels`.
Only `LabelFilter` is importable from `rubric_judge`.

| Type | Shape | Rules |
|---|---|---|
| `Label` | `str` | Surrounding whitespace stripped, then at least 1 character. A blank tag is rejected, never silently dropped |
| `Labels` | `list[Label]` | Every entry a `Label`. The same label twice in one list is rejected. Order is irrelevant everywhere, since labels are read as a set |
| `LabelFilter` | `list[Labels]` | Every group a `Labels`. The same group twice is rejected in any order, so `[["a","b"], ["b","a"]]` is refused. `[]` selects everything |

```python
Case(id=1, answer="a", labels=["table", "table"],
     criteria=[Criterion(id=1, content="c", weight=1)])
# ValidationError: labels must be unique, repeated: ['table']

Run(cases=catalog, label_filter=[["table"], ["table"]])
# ValidationError: label groups must be unique, repeated: [['table']]
```

### `RunComparison`

The two finished runs to hold against each other.

| Field | Type | Required | Rules |
|---|---|---|---|
| `baseline` | `RunResult` | yes | The run compared against, the state of things before your change |
| `candidate` | `RunResult` | yes | The run under test. Must have been judged on the same `scale` and must cover the same case ids, the same criterion ids per case, the same weights and the same labels per case as `baseline` |

### `JudgeConfig`

How to reach the judge model and how hard to try. Built from the environment by
`JudgeConfig.from_env()` in production, by hand in tests or when your settings come from
somewhere else. Each field reads the environment variable named in the last column, prefixed
`RUBRIC_JUDGE_`.

| Field | Type | Required | Rules |
|---|---|---|---|
| `model` | `str` | yes | Model name as the endpoint knows it, `"gpt-4o-mini"` or `"qwen3:8b"`. `RUBRIC_JUDGE_MODEL` |
| `endpoint` | `str` | yes | OpenAI-compatible base URL including the version path. `RUBRIC_JUDGE_ENDPOINT` |
| `api_key` | `str` | yes | Key for that endpoint. `RUBRIC_JUDGE_API_KEY` |
| `temperature` | `float` | no, default `0.0` | `>= 0`. `RUBRIC_JUDGE_TEMPERATURE` |
| `max_tokens` | `int` | no, default `768` | `> 0`. Must fit the reasoning and the closing JSON object. `RUBRIC_JUDGE_MAX_TOKENS` |
| `max_attempts` | `int` | no, default `3` | `>= 1`, the first try included. The only retry budget there is, since the SDK's own is switched off. `RUBRIC_JUDGE_MAX_ATTEMPTS` |
| `max_concurrent` | `int` | no, default `8` | `>= 1`. Judge calls in flight at once, per judge *instance*. `RUBRIC_JUDGE_MAX_CONCURRENT` |
| `health_interval_seconds` | `int` | no, default `0` | `0`, or `>= 60` (`MINIMUM_HEALTH_INTERVAL_SECONDS`). How long a running service lets its judge go without an answered call before it checks it. `0` never checks. `RUBRIC_JUDGE_HEALTH_INTERVAL` |
| `health_check_retries` | `int` | no, default `3` | `>= 0`. Retries of a periodic check after an outage, before the judge counts as unhealthy. A rejection is never retried. `RUBRIC_JUDGE_HEALTH_RETRIES` |
| `health_check_first_pause_seconds` | `int` | no, default `300` | `>= 1`. The pause before the first retry, doubled before each further one. `RUBRIC_JUDGE_HEALTH_FIRST_PAUSE` |

```python
JudgeConfig(model="qwen3:8b", endpoint="http://localhost:11434/v1", api_key="ollama")
```

## Outputs

### `Scale`

The grading scale a case was judged on. Carried by every `CaseResult`, so a stored run can be
read and re-scored without the judge that produced it and still says what its grades meant.
Compared by value, descriptions included, so two runs are comparable only if their scales are
equal.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `maximum` | `int` | `> 0` | Best score one criterion can reach. The scale is integral, so `maximum = 2` offers exactly the grades `0`, `1`, `2` |
| `presence_threshold` | `float` | `> 0`, `<= maximum` | From which score `is_present` counts the criterion as covered. Above `0`, because a `0` is by definition not covered |
| `level_descriptions` | `dict[int, str]` | empty, or one entry per grade | What each grade means, keyed by the grade. Either complete or absent. A described scale writes its own judge prompt, an undescribed one is arithmetic only |

`DEFAULT_SCALE` is the `0..2` scale the bundled prompt describes, descriptions included, and
the one `OpenAIJudge` grades on unless you give it another. It is never a stand-in for a scale
a stored result failed to name.

### `JudgeReply`

What the model said, before anything is attached to it. This is what a `Judge` returns and the
only type a custom judge has to produce. It is not part of a `CaseResult`, because
`evaluate_case` turns each one into a `CriterionResult` on the judge's scale.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `score` | `int` | `0 … scale.maximum` | The grade the model named. A whole number, since the scale is integral. Only the `int` type is checked on the model itself. The bound is checked by `parse_judge_reply` against the judge's scale, and again when the `CriterionResult` is built |
| `reasoning` | `str` | any | The judge's argument, everything it wrote before the closing JSON object. Never `None` |

### `CriterionResult`

What one criterion was given.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `criterion_id` | `int` | any | The `Criterion.id` this result belongs to |
| `weight` | `float` | `> 0`, finite | Copy of `Criterion.weight`, so a result can be re-scored without the rubric at hand |
| `score` | `float` | `0.0 … scale.maximum` | The judge's raw grade, not normalized. On the default scale `2` is fully covered, `1` partially, `0` not covered. A float, so averaging repeated runs cannot change the type. Bounded by `CaseResult.scale`, which is the object that knows it |
| `is_present` | `bool` | any | `score >= scale.presence_threshold`, using the scale of the `CaseResult` above. Never asked of the judge, and a `CaseResult` refuses a result whose value here contradicts its own score |
| `reasoning` | `str \| None` | any | The judge's own argument for the score |

There is exactly one constructor, `CriterionResult.judged()`. There is none for a criterion
the judge never answered for, because such a run is invalidated instead of being completed
around the gap.

### `CaseResult`

One whole answer, and the same document whether the case was evaluated alone or inside a run.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `case_id` | `int` | any | The `Case.id` this result belongs to |
| `score` | `float` | `0.0 … 1.0` | The weighted case score. `1.0` means every criterion fully covered |
| `scale` | `Scale` | any | What the grades below mean. Stored once per case rather than per criterion result, because one case is judged by one judge on one scale and `POST /evaluate` returns this document on its own. Required, so posting a result without it is a `422` and never a silent `0..2` |
| `criterion_results` | `list[CriterionResult]` | at least 1 entry | One result per criterion, in rubric order, so it can be zipped with `Case.criteria`. Never empty, since a rubric has at least one criterion. Criterion ids must be unique, because they are what pairs the two sides when two runs are compared |
| `labels` | `Labels` | any | The `Case.labels` this result came from, copied over so a stored run can still be sliced without the catalog at hand. `[]` for an untagged case, and for a result written before labels existed |

### `RunMetrics`

The aggregate over a run. Every case counts once, whatever the size of its rubric, because
otherwise one case with twenty criteria would outvote nineteen cases with one.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `total_cases` | `int` | `>= 1` | How many cases went into these numbers |
| `average_score` | `float` | `0 … 1` | Mean of the case scores, the single number a run is usually reported by |
| `median_score` | `float` | `0 … 1` | Middle case score. Far above the mean means a few catastrophic cases drag an otherwise solid run down |
| `variance` | `float` | `0 … 0.5` | Sample variance of the case scores. `0.0` for a single case, which has no spread |
| `standard_deviation` | `float` | `0 … √0.5`, about `0.71` | Square root of it, in score units. Small means uniformly good or bad, large means it depends heavily on the question |
| `average_criterion_score` | `float` | `0 … scale.maximum` | How the judge rates an average *statement*, on the run's raw scale (`1.4` of `2`, not `0.7`), ignoring weights and case boundaries. A different question from `average_score` |
| `criteria_fulfillment_rate` | `float` | `0 … 1` | Mean share of criteria counting as `is_present`, averaged per case first so a long rubric cannot dominate |
| `cases_with_score_zero` | `list[int]` | at most `total_cases` entries | Ids of answers that missed their rubric completely. Read these first |
| `cases_with_score_zero_count` | `int` | `0 … total_cases` | Length of that list. Derived, so the two can never disagree |
| `weakest_cases_above_zero` | `list[int]` | at most 5 entries | The weakest cases that still scored something, weakest first. Kept apart from the zeros, because a total miss and a partial answer usually have different causes |

### `LabelMetrics`

One label's slice of a run.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `label` | `str` | any | The `Case.labels` entry this slice is about |
| `metrics` | `RunMetrics` | any | The table above, computed over only the cases carrying `label`. Every field means exactly what it means for the whole run, `total_cases` included, where it is how many cases carry this label |

A case counts in every bucket it carries a label for, so the buckets overlap and their case
counts add up to more than the run. That is the question a bucket answers, "how do cases
involving tables do" and not "how do cases that are only tables do". Cases with no labels are
in the run-wide `metrics` and in no bucket at all.

### `RunResult`

One whole run.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `metrics` | `RunMetrics` | any | The aggregate over every case of the run |
| `label_metrics` | `list[LabelMetrics]` | 0 up to one per label carried | The same aggregate once per label, in alphabetical label order. One entry per label occurring anywhere in `case_results` and no others, so `[]` for a run of untagged cases |
| `applied_label_filter` | `LabelFilter` | `[]` up to one group per selection | The label filter that picked these cases, copied from `Run.label_filter` so a stored run still says which subset it is. `[]` for an unfiltered run. Named apart from the input field because it is a record, so it is checked. Every case result must match it |
| `case_results` | `list[CaseResult]` | at least 1 entry | One per selected case, in request order, so fewer than `Run.cases` when a `label_filter` narrowed the run. Never empty, because `run_metrics` refuses a run of no cases and such a run was therefore never producible. Case ids must be unique, since cases are paired by id when two runs are compared |

`RunResult.scale` reads the one scale its cases were judged on. It is a Python accessor and
not a JSON field, since the cases already carry it. A run whose cases name more than one scale
is rejected, and so is a comparison of two runs whose scales differ.

### `ChangeStatus`

Which way a score moved. A string enum with three values, used at both the case and the
criterion grain. It serializes as the plain string, so JSON readers never see an object.

| Value | Meaning |
|---|---|
| `"improved"` | The candidate scored higher, by more than `SCORE_EQUALITY_TOLERANCE` |
| `"stable"` | The two scores are equal within `SCORE_EQUALITY_TOLERANCE` |
| `"worsened"` | The candidate scored lower, by more than `SCORE_EQUALITY_TOLERANCE` |

The status is derived from the raw score and not from `is_present`. A criterion that went from
`1` to `2` never crosses the presence threshold yet visibly moved the case score, so it is
reported as improved.

### `CriterionComparisonResult`

One criterion across two runs.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `criterion_id` | `int` | any | The criterion both runs judged. Identical in both by construction |
| `weight` | `float` | `> 0` | Its weight, identical in both and compared exactly, since weights are copied from the rubric and never computed. Read a movement against it. A `2.0` swing on weight `1` beside nine criteria of weight `3` barely moves the case |
| `baseline_score` | `float` | `0.0 … scale.maximum` | What the baseline run's judge gave it, on the runs' shared raw scale |
| `candidate_score` | `float` | `0.0 … scale.maximum` | What the candidate run's judge gave it, same scale |
| `score_delta` | `float` | `-maximum … maximum` | `candidate_score - baseline_score`. Both runs were judged on one scale, because a comparison of two is refused |
| `status` | `ChangeStatus` | one of three | That delta as a status |

### `CaseComparisonResult`

One case across two runs.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `case_id` | `int` | any | The case both runs evaluated |
| `baseline_score` | `float` | `0.0 … 1.0` | Its weighted score in the baseline run |
| `candidate_score` | `float` | `0.0 … 1.0` | Its weighted score in the candidate run |
| `score_delta` | `float` | `-1.0 … 1.0` | `candidate_score - baseline_score` |
| `status` | `ChangeStatus` | one of three | That delta as a status |
| `criterion_comparison_results` | `list[CriterionComparisonResult]` | at least 1 entry | One per criterion, ordered by `criterion_id`, so the list reads the same whichever order either run happened to be stored in |

A case can be `"stable"` while its criteria moved hard in opposite directions. That is exactly
why the criterion grain exists.

### `RunMetricsDelta`

The run-level difference, candidate minus baseline, one field per `RunMetrics` field that can
meaningfully be subtracted. `total_cases` has none, because a comparison of different case
sets is refused, and the two id lists have none, because a set of ids does not subtract.
`ChangeSummary` reports the movement of cases instead.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `average_score_delta` | `float` | `-1 … 1` | Change in the mean case score, the headline number |
| `median_score_delta` | `float` | `-1 … 1` | Change in the median. Read it next to the mean, since a mean that rose while the median fell means a few cases carried the win |
| `variance_delta` | `float` | `-0.5 … 0.5` | Change in the sample variance of the case scores |
| `standard_deviation_delta` | `float` | `-√0.5 … √0.5` | Change in their spread. Negative means more uniform, which is an improvement or a regression depending on which way the mean went |
| `average_criterion_score_delta` | `float` | `-maximum … maximum` | Change on the runs' shared raw scale, ignoring weights and case boundaries. Moves independently of `average_score_delta` |
| `criteria_fulfillment_rate_delta` | `float` | `-1 … 1` | Change in the mean share of criteria counting as covered |
| `cases_with_score_zero_count_delta` | `int` | `-total_cases … total_cases` | Change in how many answers missed completely. Negative is the good direction here |

### `LabelMetricsDelta`

One label's slice of a comparison. This is what answers the question "my average went up, but
did I fix `agentic_search` or break `table`".

| Field | Type | Range | Meaning |
|---|---|---|---|
| `label` | `str` | any | The label this slice is about. Always present on both runs, because a case whose labels changed between them makes the comparison incomparable |
| `metrics_delta` | `RunMetricsDelta` | any | The table above, over only the cases carrying `label`. Signs point the same way they do for the whole run |

### `ChangeMagnitude`

How large the moves on one side of a comparison were.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `largest` | `float \| null` | `-1.0 … 1.0`, or `null` | The single biggest move on this side |
| `mean` | `float \| null` | `-1.0 … 1.0`, or `null` | Mean of the moves, what a typical one was worth |
| `median` | `float \| null` | `-1.0 … 1.0`, or `null` | Median of them. Far below the mean on the improvement side means one case carries the win |

All three carry the sign of their side, so a worsening's `largest` is the most negative delta
and not its absolute value. All three are `null` when nothing moved that way. A run where
nothing got worse has no worsening to report, and a `0.0` there would read as a regression of
exactly zero.

### `ChangeSummary`

Where the run moved, case by case. The counterpart to `RunMetricsDelta`. That one says the
average rose by `0.125`, this one says whether every case rose a little or one rose a lot
while another collapsed.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `improved_case_ids` | `list[int]` | at most `total_cases` entries | Cases the candidate scored higher on, biggest improvement first. Complete rather than capped, so the top three are its first three |
| `stable_case_ids` | `list[int]` | at most `total_cases` entries | Cases whose score did not move beyond the tolerance, in id order |
| `worsened_case_ids` | `list[int]` | at most `total_cases` entries | Cases the candidate scored lower on, biggest regression first. Read these when an average went up and you want to know what it cost |
| `improvement` | `ChangeMagnitude` | any | Size of the moves behind `improved_case_ids`, all positive. All three fields `null` when nothing improved |
| `worsening` | `ChangeMagnitude` | any | Size of the moves behind `worsened_case_ids`, all negative. All three fields `null` when nothing got worse |
| `improved_case_count` | `int` | `0 … total_cases` | Length of `improved_case_ids`. Derived, so list and count cannot disagree |
| `stable_case_count` | `int` | `0 … total_cases` | Length of `stable_case_ids` |
| `worsened_case_count` | `int` | `0 … total_cases` | Length of `worsened_case_ids` |
| `improvement_rate` | `float` | `0 … 1` | Share of cases that improved |
| `stability_rate` | `float` | `0 … 1` | Share that did not move |
| `worsening_rate` | `float` | `0 … 1` | Share that got worse. Every case lands in exactly one list, so the three counts always add up to the run. The three rates are three separate divisions and sum to `1.0` only to within float rounding |

### `RunComparisonResult`

One whole comparison.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `metrics_delta` | `RunMetricsDelta` | any | Whether the run got better |
| `summary` | `ChangeSummary` | any | How that is distributed over the cases |
| `label_metrics_deltas` | `list[LabelMetricsDelta]` | 0 up to one per label carried | Which kind of case moved, in alphabetical label order. `[]` when neither run carries labels |
| `case_comparison_results` | `list[CaseComparisonResult]` | at least 1 entry | Which criterion is responsible. Ordered by `case_id`, since both runs cover the same cases and neither one's storage order is canonical |

## Constants

| Constant | Value | Meaning |
|---|---|---|
| `DEFAULT_SCALE` | `Scale(maximum=2, presence_threshold=0.5, level_descriptions={2: …, 1: …, 0: …})` | The scale the bundled prompt is written from and the one `OpenAIJudge` grades on unless you give it another. The three level descriptions are part of the scale's identity, so `Scale(maximum=2, presence_threshold=0.5)` is a different, undescribed scale and is not equal to it |
| `WEAKEST_CASES_REPORTED` | `5` | How many cases `RunMetrics.weakest_cases_above_zero` names. A shortlist to look at next, not a complete ranking |
| `MINIMUM_HEALTH_INTERVAL_SECONDS` | `60` | The shortest `health_interval_seconds` apart from `0`. A periodic check costs no tokens, but every replica sends one, and less than a minute would load the endpoint for no information a minute does not already give. From `rubric_judge.judge` |
| `COMPARE_ONLY_REFUSAL` | the `503` detail | What `POST /evaluate` and `POST /evaluate/run` answer in compare-only mode. From `rubric_judge.api` |
| `ACCESS_TOKEN_REFUSAL` | the `401` detail | What every endpoint except `GET /health` answers to a request without the access token. From `rubric_judge.api` |
| `ACCESS_TOKEN_VARIABLE` | `"RUBRIC_JUDGE_ACCESS_TOKEN"` | The one `RUBRIC_JUDGE_*` variable that configures the HTTP service instead of the judge. From `rubric_judge.judge` |
| `SCORE_EQUALITY_TOLERANCE` | `1e-9` | How close two scores must be to count as unchanged in a comparison. Far above the float noise two runs accumulate summing the same weights in a different order, and far below the smallest difference a rubric can actually produce |

## Functions

```python
async def evaluate_case(judge: Judge, case: Case) -> CaseResult
```
Judges one case. Fans out over the criteria concurrently and folds the results into one score,
or raises `JudgeUnavailableError` and returns nothing at all.

```python
async def evaluate_run(judge: Judge, run: Run) -> RunResult
```
Judges every case concurrently and adds `RunMetrics`, plus one `LabelMetrics` per label the
cases carry. A fan-out over `evaluate_case` and nothing else, so the two paths cannot drift
apart. Which cases run is the `Run`'s own business. It judges `run.selected_cases`, so a whole
catalog plus a `label_filter` runs the subset and records what picked it.

```python
def filter_cases_by_labels(cases: list[Case], label_filter: list[list[str]]) -> list[Case]
```
The cases a label filter covers, any group and every label of it. The same rule
`Run.selected_cases` runs by and the buckets bucket by, so "the run selected by `table`" and
"the `table` bucket of the full run" are the same cases. You rarely need it to *run* a subset.
Reach for it to see what a label filter would pick first. No match is an empty list and never
an exception, since `Run` is what refuses to run one. The label filter itself is held to the
same rules as `Run.label_filter`. Raises `TypeError` for a flat `["table"]`, which would
otherwise compare characters and quietly return the wrong cases.

```python
def run_metrics(case_results: list[CaseResult]) -> RunMetrics
```
The aggregate on its own. Use it when you already have case results, loaded from disk for
instance, and only want the numbers. Raises `ValueError` on an empty list, or on cases judged
on different scales, because `average_criterion_score` averages raw grades and grades in two
units do not average.

```python
def label_metrics(case_results: list[CaseResult]) -> list[LabelMetrics]
```
The per-label breakdown on its own, alphabetical. Public for the same reason `run_metrics` is,
so that a run read back from disk months later can still be sliced. `[]` when no case carries
a label. Raises `ValueError` on cases judged on different scales.

```python
def case_score(criterion_results: list[CriterionResult], scale: Scale) -> float
```
The weighted formula alone, returning a float in `[0, 1]`. Dividing by `scale.maximum` is what
makes the result scale-free. Raises `ValueError` on an empty list, or when a result is graded
above the scale.

```python
def judge_prompt(scale: Scale, examples: str = "") -> str
```
Writes a judge's system prompt from a scale that describes its levels, covering the header,
one line per level, and the reply format down to the grades the model may answer with.
`examples` is appended verbatim. Raises `ValueError` for a scale with no `level_descriptions`.

```python
def parse_judge_reply(reply: str, scale: Scale) -> JudgeReply
```
Pulls the score and the argument out of one raw judge reply. The last `{"score": …}` object in
it wins, and everything before it is the reasoning. `OpenAIJudge` calls it for you. Reach for
it to check what your own judge's endpoint replies, or to reuse the parsing in a judge of your
own. Raises `UnusableReplyError`, a `ValueError`, for a reply with no score object,
unparseable JSON, or a grade off `scale`. That message is the corrective prompt the retry loop
sends back to the model and not a report for a human.

```python
def compare_runs(run_comparison: RunComparison) -> RunComparisonResult
```
Holds two finished runs against each other at three grains, run, case and criterion, plus one
delta per label. Pure computation with no judge, no network and no cost. Raises
`RunsNotComparableError`, a `ValueError`, naming every difference at once when the runs do not
describe the same catalog.

```python
@classmethod
def CriterionResult.judged(criterion: Criterion, score: float,
                           reasoning: str | None, scale: Scale) -> CriterionResult
```
The only constructor for a criterion result. It copies `id` and `weight` off the criterion and
derives `is_present` from `scale`, so the two can never contradict the grade. `scale` is the
judge's own and never a guess. Raises `pydantic.ValidationError` for a negative or non-finite
`score`. Whether the score fits the scale is checked by the `CaseResult` it goes into, which is
the object that knows.

```python
@property
def Run.selected_cases -> list[Case]
```
The cases this run actually judges, the entries of `cases` matching `label_filter`, in their
original order, all of them when it is empty. Never empty, because a label filter matching
nothing is refused when the `Run` is built.

```python
@property
def RunResult.scale -> Scale
```
The one `Scale` every case of the run was judged on. Raises `ValueError` if the cases name more
than one, which `RunResult` already refuses at validation, so a run you are holding always
answers.

```python
@property
def Scale.grades -> range
```
Every grade the scale offers, lowest first, `range(0, maximum + 1)`. `list(DEFAULT_SCALE.grades)`
is `[0, 1, 2]`. `str(scale)` names the bounds and the threshold, `"0..2 (covered from 0.5)"`,
which is how refusals quote a scale.

```python
class Judge(Protocol):
    scale: Scale
    async def score(self, answer: str, criterion: Criterion,
                    context: str | None = None) -> JudgeReply
```
The extension point. Implement those two members and `evaluate_case` and `evaluate_run` take
your object unchanged, with no base class and no registration. `score` judges exactly one
criterion and must either return a valid `JudgeReply` or raise. Raise `JudgeUnavailableError`
for anything the endpoint did and anything else for a bug. Throttling belongs here too, because
only the implementation knows what its backend tolerates.

```python
class OpenAIJudge:
    def __init__(self, config: JudgeConfig, system_prompt: str | None = None,
                 scale: Scale = DEFAULT_SCALE, client: AsyncOpenAI | None = None)
    async def score(self, answer: str, criterion: Criterion, context: str | None = None) -> JudgeReply
    async def check(self) -> None
    async def check_once(self) -> None

    scale: Scale            # what it grades on
    health: JudgeHealth     # whether the endpoint recently answered
    system_prompt: str      # the system message it sends, generated unless you passed one
    config: JudgeConfig
    client: AsyncOpenAI
    free_call_slots: asyncio.Semaphore   # the throttle of the running event loop
```
The bundled judge, for any OpenAI-compatible endpoint. `system_prompt` replaces the system
prompt and `scale` is what it grades on. A described scale writes its own prompt, so passing
both is only for your own instructions on your own scale. `client` is an already-built
`openai.AsyncOpenAI` to talk through, a shared connection pool, an Azure client or a test
double. Left out, one is built from `config` with `max_retries=0` so `max_attempts` stays the
only retry budget. A client you pass keeps its own `max_retries` and the two budgets multiply,
so build yours with `max_retries=0` unless you mean that. `score` judges a single criterion
with no fan-out and no failure handling above its own retries, which is handy for a quick
experiment. Raises `ValueError` for a `scale` with no `level_descriptions` and no
`system_prompt` to go with it.

Build one judge and share it. `max_concurrent` is a budget of the instance, so a judge per
request would hand each request its own full set of slots. `free_call_slots` returns the
semaphore of the event loop the call runs in, one per loop, and raises `RuntimeError` outside a
running loop.

`check_once` is the periodic check. It lists the endpoint's models with `GET /v1/models`, which
costs no tokens, and waits at most 60 seconds for its answer. A response naming the configured
model marks `health` healthy. An outage, a timeout included, raises `JudgeUnavailableError`,
records nothing, and leaves the caller's own schedule to decide when the judge is unhealthy. A
response that no longer lists the model marks `health` unhealthy and raises
`ModelNotListedError`, final at once rather than retried. A `401`, `403` or `404` also marks
`health` unhealthy and is raised as the SDK's own exception. `check` is what the service's
startup uses. It sends one real completion of at most 16 reply tokens with the judge's model
and temperature, because only a completion proves the request shape a judge call uses. It waits
at most 60 seconds per attempt, repeats through outages with the quick backoff of `score`, and
marks `health` unhealthy on any failure. During `score`, only a
`401`, `403` or `404` marks it unhealthy, and any answer marks it healthy again. Judge calls
keep the SDK's own timeout of 600 seconds.

```python
class JudgeHealth:
    def __init__(self, clock: Callable[[], float] = time.monotonic)
    def record_proof(self) -> None
    def record_failure(self) -> None
    def seconds_until_check_due(self, interval_seconds: float) -> float

    clock: Callable[[], float]      # seconds, never going backwards
    is_healthy: bool                # False until the first answer
    last_evidence_at: float | None  # last answer or failure, on clock
```
The bookkeeping behind `OpenAIJudge.health`. `seconds_until_check_due` is `0.0` before any
evidence and once the last evidence is older than the interval. A failure restarts the wait
like an answer does, so a failed check is repeated one interval later. Pass your own `clock` to
let a day pass in a test.

```python
@classmethod
def JudgeConfig.from_env() -> JudgeConfig
@classmethod
def JudgeConfig.from_mapping(environment: Mapping[str, str]) -> JudgeConfig
```
`from_env()` reads the `RUBRIC_JUDGE_*` variables out of `os.environ`. `from_mapping()` applies
the same rules to any mapping you hand it, settings loaded from a file or a plain dict in a
test. Both raise `RuntimeError` naming every missing, empty or unknown variable at once, and
`pydantic.ValidationError` when a numeric variable does not parse or is out of range. An
optional variable that is absent falls back to the field default. An optional variable set to
the empty string is refused like a missing required one. Whitespace around a value is dropped
before any of that is decided, because `docker run --env-file` passes it through where `uvicorn
--env-file` strips it, so a value of nothing but spaces counts as empty. Names without the
`RUBRIC_JUDGE_` prefix are ignored, so the whole process environment can be handed in. A name
with it that no field reads, such as `RUBRIC_JUDGE_HEALTH_INTERVALL`, is refused as unknown,
because a typo would otherwise fall back to the default unnoticed. `RUBRIC_JUDGE_ACCESS_TOKEN`
is known and refused when empty, but no field reads it, because it belongs to the HTTP service.

The functions below come from `rubric_judge.api` and decide how the HTTP service starts.

```python
def judge_source(environment: Mapping[str, str], override_installed: bool) -> JudgeSource
```
Where a starting service's judge comes from. `CUSTOM` when `get_judge` is overridden, else
`ENVIRONMENT` when any `RUBRIC_JUDGE_*` variable is present, an optional one included, else
`NONE`, which is compare-only mode. `RUBRIC_JUDGE_ACCESS_TOKEN` does not count, since it locks
the service and says nothing about the judge.

| Source | At startup | `GET /health` |
|---|---|---|
| `ENVIRONMENT` | `JudgeConfig.from_env()`, then one `check()`. Any failure stops the service, uvicorn exits with code 3 | `ok` while `judge.health` is healthy, `503` `failing` otherwise |
| `CUSTOM` | nothing is built or checked | `ok` `custom` |
| `NONE` | a warning is logged | `ok` `none`. `POST /evaluate` and `POST /evaluate/run` answer `503` with `COMPARE_ONLY_REFUSAL` |

```python
def get_judge(request: Request) -> Judge
```
The FastAPI dependency every endpoint takes its judge from: the one `OpenAIJudge` the service
built and proved at startup, so one `max_concurrent` budget is shared by every request. Raises
`HTTPException` `503` in compare-only mode. A program that embeds the app brings its own judge
by overriding it **before** the app starts, which is what makes the source `CUSTOM`.

```python
app.dependency_overrides[get_judge] = lambda: my_own_judge   # before startup
```

```python
def access_token(environment: Mapping[str, str]) -> str | None
def is_authorized(expected_token: str | None, presented_token: str | None) -> bool
async def require_access_token(request: Request,
                               credentials: HTTPAuthorizationCredentials | None) -> None
```
`access_token` reads `RUBRIC_JUDGE_ACCESS_TOKEN` at startup, without surrounding whitespace.
`None` means the variable is absent and the service is open. A value of nothing but whitespace
raises `RuntimeError`. `is_authorized` is `True` for an open service and otherwise exactly when
the presented token equals the expected one, compared in constant time. `require_access_token`
is the FastAPI dependency on `POST /evaluate`, `POST /evaluate/run` and `POST /compare` that
answers `401` with `WWW-Authenticate: Bearer` when `is_authorized` says no. It runs before the
body is validated and before `get_judge`, so a caller without the token learns nothing else.
Only a body that is not JSON at all is answered `422` first, because FastAPI parses the body
before any dependency runs.

```python
is_authorized("s3cret", "s3cret")   # True
is_authorized("s3cret", None)       # False
```

```python
def health_report(source: JudgeSource, judge_health: JudgeHealth | None,
                  monitor_alive: bool) -> HealthReport
def retry_pauses_seconds(first_pause_seconds: float, retries: int) -> list[float]
async def prove_the_judge_periodically(judge: OpenAIJudge,
                                       sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None
```
`health_report` decides the body of `GET /health`, and a periodic check that crashed counts as
unhealthy. `retry_pauses_seconds(300, 3)` is `[300, 600, 1200]`, the waits before each retry of
a periodic check, and `[]` for no retries. `prove_the_judge_periodically` is the background task
the service runs when `health_interval_seconds` is positive. It calls `check_once` whenever the
evidence in `judge.health` is older than the interval, retries an outage after each of those
pauses while the judge stays as it was, and records a failure only once every attempt failed. A
rejection, `ModelNotListedError` included, is recorded at once with no retry, and traffic
answered or refused during a pause ends the schedule. It logs every failed attempt instead of
raising it, and only a bug ends it.

## Exceptions

| Exception | Base | Raised when |
|---|---|---|
| `JudgeUnavailableError` | `Exception` | The judge produced no usable reply within `max_attempts`, or its endpoint answered with no choice or no content. The case and the run it belongs to are invalid, and `503` is what the HTTP layer answers. Raise it from a custom judge for anything its endpoint does |
| `ModelNotListedError` | `Exception` | The periodic check (`OpenAIJudge.check_once`) reached the endpoint but the configured model is no longer among the models it lists. Treated like a rejection, final at once with no retry, and `GET /health` turns `503` |
| `UnusableReplyError` | `ValueError` | One reply the parser refuses, with no score object, unparseable JSON, or a grade off the scale. Its message is the correction the retry loop sends back to the model, so it is worded for the judge and not for a human. Its own type, so the retry loop repeats an attempt for this exception and for nothing else |
| `RunsNotComparableError` | `ValueError` | The two runs do not describe the same catalog or were not judged on the same scale. The message names every difference at once. A named type, so the HTTP layer can tell a refused comparison apart from a `ValidationError` or a `StatisticsError`, which are `ValueError` subclasses too |

## HTTP API

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/evaluate` | a `Case` | a `CaseResult` |
| `POST` | `/evaluate/run` | a `Run` | a `RunResult` |
| `POST` | `/compare` | a `RunComparison` | a `RunComparisonResult` |
| `GET` | `/health` | none | a `HealthReport`, `200` or `503` |

The JSON shapes are exactly the models above. The service is stateless, with no catalog, no run
ids and no persistence.

Started with `RUBRIC_JUDGE_ACCESS_TOKEN`, the three `POST` endpoints need
`Authorization: Bearer <token>`. `GET /health`, `/docs` and `/openapi.json` never do.

### `POST /evaluate`

The request is one `Case`. This is a complete response for a two-criterion rubric of weights
`3` and `1`, from a judge grading them `2` and `0`:

```json
{
  "case_id": 1,
  "score": 0.75,
  "scale": {
    "maximum": 2,
    "presence_threshold": 0.5,
    "level_descriptions": {
      "2": "Fully covered. Every essential part of the criterion is clearly recognizable in the answer, even if the wording, terminology or structure differs.",
      "1": "Partially covered. Some essential information is missing, but the basic idea is still derivable from the answer.",
      "0": "Not covered. The criterion is absent, or the answer has no recognizable connection to it."
    }
  },
  "criterion_results": [
    {
      "criterion_id": 1,
      "weight": 3.0,
      "score": 2.0,
      "is_present": true,
      "reasoning": "The answer instructs the reader to email hr@example.com before 10:00 on the first day, which is exactly what the criterion asks for."
    },
    {
      "criterion_id": 2,
      "weight": 1.0,
      "score": 0.0,
      "is_present": false,
      "reasoning": "Neither the expected last day nor any duration is mentioned."
    }
  ],
  "labels": []
}
```

Note `"level_descriptions"` keyed by `"2"`, `"1"`, `"0"` as strings. JSON has no integer keys,
and Pydantic reads them back as integers.

### `POST /evaluate/run`

The request is a list of exactly those case bodies, plus optional `labels` and an optional
`label_filter`:

```json
{
  "cases": [
    { "id": 1, "context": "The question asked was: How do I report sick leave?",
      "answer": "Send an email to hr@example.com before 10:00 on your first day.",
      "criteria": [
        { "id": 1, "content": "Report by email before 10:00 on the first day", "weight": 3 },
        { "id": 2, "content": "State the expected last day of absence", "weight": 1 }
      ],
      "labels": ["policy"] },
    { "id": 2, "context": "The question asked was: How do I request vacation?",
      "answer": "Submit the request in the HR tool.",
      "criteria": [
        { "id": 21, "content": "Submit the request in the HR tool", "weight": 1 }
      ],
      "labels": ["policy", "tool"] },
    { "id": 3, "context": "The question asked was: Who approves overtime?",
      "answer": "Your line manager approves it.",
      "criteria": [
        { "id": 31, "content": "The line manager approves it", "weight": 1 }
      ],
      "labels": ["tool"] }
  ]
}
```

Judged by a judge grading those four criteria `0`, `0`, `2` and `2`, the response's `metrics`
object is this. The numbers here come from running that small catalog, so they are checkable
line by line. The repository's own sample run, a larger catalog judged by a real model, is in
[../examples](../examples).

```json
{
  "total_cases": 3,
  "average_score": 0.6666666666666666,
  "median_score": 1.0,
  "variance": 0.3333333333333333,
  "standard_deviation": 0.5773502691896257,
  "average_criterion_score": 1.0,
  "criteria_fulfillment_rate": 0.6666666666666666,
  "cases_with_score_zero": [1],
  "weakest_cases_above_zero": [2, 3],
  "cases_with_score_zero_count": 1
}
```

`label_metrics` repeats that whole object once per label, alphabetically, over only the cases
carrying it. Here `policy` reports two cases and `tool` reports two, over a run of three,
because case 2 carries both. `case_results[i]` is the same document `POST /evaluate` returns
for that case, the same type and not a similar one.

`scale` is required on every `case_results` entry, here and when posting a stored run back to
`/compare`. Were it read as `DEFAULT_SCALE` instead, a `0..10` run that lost the field would
match a genuine `0..2` run's scale exactly and come back as deltas in a unit neither run was
judged in, with a `200`.

To run only part of the catalog, add a `label_filter` to the body. There is no query parameter,
so the library and the API take it in exactly one place and one form:

```json
{ "label_filter": [["policy", "tool"], ["agentic"]] }
```

Against the catalog above that runs case 2, which carries both `policy` and `tool`, plus any
case carrying `agentic`, of which there are none. It comes back as `applied_label_filter`.

The request is synchronous. The response arrives when the last selected case is done, so the
whole catalog is one HTTP timeout.

### `POST /compare`

The body is two `RunResult` documents back verbatim, with no reshaping:

```json
{ "baseline": { "metrics": {}, "case_results": [] },
  "candidate": { "metrics": {}, "case_results": [] } }
```

Take the catalog above judged twice. In the baseline case 1 scores `0.0`, case 2 `1.0` and
case 3 `1.0`. In the candidate case 1 rises to `0.8750000000000001`, case 2 holds at `1.0` and
case 3 falls to `0.5`. That pair produces this `metrics_delta` and `summary`:

```json
{
  "metrics_delta": {
    "average_score_delta": 0.1250000000000001,
    "median_score_delta": -0.12499999999999989,
    "variance_delta": -0.265625,
    "standard_deviation_delta": -0.3171420192563591,
    "average_criterion_score_delta": 0.5,
    "criteria_fulfillment_rate_delta": 0.33333333333333337,
    "cases_with_score_zero_count_delta": -1
  },
  "summary": {
    "improved_case_ids": [1],
    "stable_case_ids": [2],
    "worsened_case_ids": [3],
    "improvement": { "largest": 0.8750000000000001,
                     "mean": 0.8750000000000001,
                     "median": 0.8750000000000001 },
    "worsening": { "largest": -0.5, "mean": -0.5, "median": -0.5 },
    "improved_case_count": 1,
    "stable_case_count": 1,
    "worsened_case_count": 1,
    "improvement_rate": 0.3333333333333333,
    "stability_rate": 0.3333333333333333,
    "worsening_rate": 0.3333333333333333
  }
}
```

The mean rose while the median fell. One case carried the whole win, and `worsened_case_ids`
says which one paid for it. That pair of numbers is the reason both are reported. Those
trailing digits are real. `0.8750000000000001` is what summing weights `3` and `1` in that
order produces, which is also why "stable" is a tolerance and not an `==`.

The response adds `label_metrics_deltas`, the same `metrics_delta` once per label, whenever the
runs carry labels. For the pair above that is `policy` at `0.4375` and `tool` at `-0.25`, which
is how a run-wide gain of `0.125` turns out to be one kind of case improving while another got
worse. `applied_label_filter` is not compared, because two runs covering the same case ids are
comparable however each was selected.

Complete stored request and response documents of this kind live in
[../examples](../examples), and a test recomputes the stored comparison from the two stored
runs, so they stay real output.

### `GET /health`

Answers a `HealthReport`, and never calls the judge itself. The Docker image's `HEALTHCHECK`
asks this endpoint.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `status` | `str` | `ok`, `unhealthy` | `ok` with a `200`, `unhealthy` with a `503` |
| `judge` | `str` | `ok`, `failing`, `custom`, `none` | `ok` or `failing` for a judge from the environment, `custom` for one installed in code, `none` in compare-only mode |

The judge turns `failing` when the endpoint refuses a call with `401`, `403` or `404`, when a
periodic check fails on every attempt of its retry schedule, or when a periodic check finds the
model no longer listed (`ModelNotListedError`, final at once, no retry). It turns `ok` again
with the next answered call. A periodic check that crashed is a bug, is logged at once and keeps
the judge `failing` until the service restarts. An outage answered with `503` per request does
not change it. The cause of a `failing` judge is in the server log and never in the body. `POST
/compare` needs no judge at all and answers correctly in every one of these states.

## Errors

Request bodies are validated before the first LLM call, so a malformed request costs nothing.
The `422` body carries only `loc`, `msg` and `type`. The rejected value is never echoed back:

```json
{ "detail": [ { "loc": ["body", "criteria", 0, "weight"],
                "msg": "Input should be greater than 0", "type": "greater_than" } ] }
```

| Situation | Library | HTTP |
|---|---|---|
| `criteria`, `cases`, `criterion_results` or `case_results` empty; duplicate ids in any of them; `content` blank; `weight` `0`, negative, `Infinity` or `NaN`; a missing field | `pydantic.ValidationError` | `422` |
| A `Case.context` that is blank or whitespace-only. Omitting it, or passing `None`, is the one valid way to say a case has no context | `pydantic.ValidationError` | `422` |
| A run posted to `/compare` whose numbers leave the ranges the [output tables](#outputs) give: a score above its own `scale.maximum` or off `0 … 1`, a non-positive weight, any `Infinity` or `NaN`, a `variance` or `standard_deviation` no distribution of case scores could produce, or `RunMetrics` id lists naming more cases than `total_cases` | `pydantic.ValidationError` | `422` |
| A `CaseResult` posted without its `scale`. No default stands in, because one would decide the unit of every score under it | `pydantic.ValidationError` | `422` |
| A run whose cases name more than one `scale`; a result whose `is_present` contradicts its own score | `pydantic.ValidationError` | `422` |
| A `Scale` whose `presence_threshold` is above its `maximum`, so no criterion could ever count as covered | `pydantic.ValidationError` | `422` |
| A `Scale` describing only some of its grades, or a grade it does not have | `pydantic.ValidationError` | `422` |
| A blank label, or the same label twice, on one case or in one group of a label filter | `pydantic.ValidationError` | `422` |
| The same group twice in a label filter, in any order, in a `label_filter` or in `filter_cases_by_labels()` | `pydantic.ValidationError` | `422` |
| A `Run` whose `label_filter` matches no case at all | `pydantic.ValidationError` naming the labels the run does carry, with counts | `422` |
| A stored `RunResult` whose `applied_label_filter` does not describe the cases it holds | `pydantic.ValidationError` naming the offending case ids | `422` |
| A stored run whose `label_metrics` does not describe exactly the labels its cases carry | `pydantic.ValidationError` | `422` |
| A flat `["table"]` where a label filter is expected, passed to `filter_cases_by_labels()` or sent in a request body | `TypeError` showing the label filter to write instead | `422` (schema) |
| Two runs not comparable: a different grading scale, different case ids, different criteria within a case, different weights, or a case whose labels changed between the runs | `RunsNotComparableError`, a `ValueError`, naming every difference at once | `422` |
| `run_metrics([])`, or `case_score([], scale)`. An empty list has no distribution and no score, and `0.0` would be indistinguishable from a completely missed answer | `ValueError` | none, unreachable over HTTP: the models refuse an empty run or rubric first |
| `run_metrics()` or `label_metrics()` over cases judged on different scales, since raw grades in two units do not average | `ValueError` naming every scale found | none, unreachable over HTTP: `RunResult` refuses such a run first |
| `judge_prompt(scale)` for a scale with no `level_descriptions`, or `OpenAIJudge(config, scale=…)` with such a scale and no `system_prompt` | `ValueError` naming the scale | none, raised at construction time and not per request |
| A judge returning a score above the `scale` it declares | `ValueError` out of `evaluate_case()` naming the criterion, a bug in the judge and not an outage | `500` |
| Judge not configured: a required `RUBRIC_JUDGE_*` variable missing while another one is set, any of them, required or optional, exported empty or as nothing but whitespace, or a `RUBRIC_JUDGE_*` name no setting reads | `RuntimeError` naming every offending variable and which of the three it is | none, the service does not start: uvicorn exits with code 3 |
| A numeric `RUBRIC_JUDGE_*` variable that does not parse or is out of range, including a `RUBRIC_JUDGE_HEALTH_INTERVAL` from 1 to 59 | `pydantic.ValidationError` naming the setting | none, the service does not start: uvicorn exits with code 3 |
| The startup `check()` refused by the endpoint, for a wrong key, model, URL or parameter | the `openai` SDK's own exception, unretried | none, the service does not start: uvicorn exits with code 3 |
| The startup `check()` unanswered within `RUBRIC_JUDGE_MAX_ATTEMPTS` | `JudgeUnavailableError` | none, the service does not start: uvicorn exits with code 3 |
| `RUBRIC_JUDGE_ACCESS_TOKEN` exported empty or as nothing but whitespace | `RuntimeError` naming it | none, the service does not start: uvicorn exits with code 3 |
| The service started with `RUBRIC_JUDGE_ACCESS_TOKEN`, and a request to any `POST` endpoint without `Authorization: Bearer` and that token | none, the library needs no service | `401` with `ACCESS_TOKEN_REFUSAL` and `WWW-Authenticate: Bearer`, before the body is validated and before compare-only mode answers |
| No `RUBRIC_JUDGE_*` variable at all, and a request to `POST /evaluate` or `POST /evaluate/run` | none, the library needs no service | `503` with `COMPARE_ONLY_REFUSAL`, before the body is validated |
| Judge endpoint unreachable, timed out, rate limited or answering `5xx` | retried up to `RUBRIC_JUDGE_MAX_ATTEMPTS` times with a doubling wait, then `JudgeUnavailableError`. The whole run is dropped | `503` |
| Judge reply unparseable after `RUBRIC_JUDGE_MAX_ATTEMPTS` tries, with no JSON, broken JSON, or a grade off the scale | `UnusableReplyError`, a `ValueError`, per attempt, then `JudgeUnavailableError` naming the last complaint | `503` |
| Judge reply carrying no content at all, or a response with no choice, from a truncation, a content filter or a tool-call path | `JudgeUnavailableError` naming the endpoint's `finish_reason`, not retried | `503` |
| Judge endpoint rejecting the key, the model or the request | the `openai` SDK's own exception, unretried, because repeating it would not help. A `401`, `403` or `404` also marks `judge.health` unhealthy | `500`, and `GET /health` answers `503` after a `401`, `403` or `404` |
| Periodic check finds the model no longer listed | `ModelNotListedError`, final at once, no retry | none, and `GET /health` answers `503` |
| A bug in the program | propagates as itself | `500` |
| Request cancelled by a client disconnect or a shutdown | `asyncio.CancelledError` propagates | none |
