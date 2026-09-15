"""Pydantic models. The result shape is a published interface: grow it additively.

Public fields and constants are documented with docstrings rather than comments: the
IDE shows them on hover, and `use_attribute_docstrings` copies them into the OpenAPI
schema, so the editor and `/docs` can never drift apart.
"""

import math
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    field_validator,
)

SCALE_MAX = 2
"""Best score one criterion can reach. The scale is integral: 0, 1 or 2."""

PRESENCE_THRESHOLD = 0.5
"""Half a scale point. With a single judge run this is simply "not a plain 0"; with
several runs it is the majority rule applied to the averaged score."""

WEAKEST_CASES_REPORTED = 5
"""How many of the weakest non-zero cases `RunMetrics` names by id — the shortlist to look
at next, not a complete ranking."""

SCORE_EQUALITY_TOLERANCE = 1e-9
"""How close two scores have to be to count as unchanged in a comparison. Far above the
float noise two runs accumulate summing the same weights in a different order, and far below
the smallest score difference a rubric can actually produce."""


def _duplicate_ids(ids: Iterable[int]) -> list[int]:
    """One check for the rubric and the batch: both match results back to their input by id,
    so a repeat breaks both the same way. Empty when every id is unique, which is what lets
    it read as a validator condition.
    """
    counted = Counter(ids)
    return sorted(id_ for id_, count in counted.items() if count > 1)


class DocumentedModel(BaseModel):
    """Base for every model in this package: makes attribute docstrings the field
    descriptions, so one docstring feeds both IDE hover and the generated OpenAPI schema."""

    model_config = ConfigDict(use_attribute_docstrings=True)


class Criterion(DocumentedModel):
    """One statement a good answer has to contain — the atom of a rubric.

    A rubric is just a list of these, and each one is judged on its own, one LLM call per
    criterion. Phrase `content` as a single checkable fact: two facts in one criterion
    cannot be scored apart, so the judge has to pick a compromise between them.

    Example:
        Criterion(id=1, content="Report by email before 10:00", weight=3)
    """

    id: int
    """Caller-owned identifier, echoed back as `CriterionResult.criterion_id` so results
    can be matched to the rubric without relying on list order."""

    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    """The requirement in plain language, e.g. "Send an email to hr@example.com".
    Phrase it as one checkable fact — two facts in one criterion cannot be scored apart.
    Surrounding whitespace is stripped, so a blank criterion is rejected rather than judged."""

    weight: float = Field(gt=0, allow_inf_nan=False)
    """How much this criterion counts next to the others. Only the ratios matter:
    weights 3 and 1 score exactly like 30 and 10."""


class CriterionResult(DocumentedModel):
    """The verdict for a single criterion: what the judge gave, and why.

    Not built by hand — use `judged()` for an answered criterion and `unjudged()` for one
    the judge never delivered, so the derived fields stay consistent everywhere.

    Every documented range is enforced, not merely described: a stored result is postable to
    `/compare`, so this model is an *input* type there and the numbers below are arithmetic
    a comparison depends on. `NaN` in particular would survive every computation, serialize
    as JSON `null` where a float is promised, and classify as a regression.
    """

    criterion_id: int
    """The `Criterion.id` this verdict belongs to."""

    weight: float = Field(gt=0, allow_inf_nan=False)
    """Copy of `Criterion.weight`, so a result can be scored without the rubric at hand.
    Positive and finite, exactly as the rubric it was copied from."""

    score: float = Field(ge=0, le=SCALE_MAX, allow_inf_nan=False)
    """Judge score in [0, SCALE_MAX]: 2 fully covered, 1 partially, 0 not covered.
    A float rather than an int, so averaging several runs of the same criterion cannot
    change the type. Off the scale it is refused rather than folded into a case score
    above 1.0 that nothing downstream could recognize as wrong."""

    is_present: bool
    """Whether the criterion counts as covered at all: `score >= PRESENCE_THRESHOLD`.
    Derived here and never asked of the judge — one question less for it to get wrong."""

    spread: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    """Standard deviation of `score` across repeated judge runs. Stays 0.0 while every
    criterion is judged exactly once, which is the only mode implemented so far."""

    failed: bool = False
    """True when the judge produced no usable verdict even after all retries. The criterion
    still counts with score 0 and keeps its weight, so an outage lowers the score visibly."""

    reasoning: str | None = None
    """The judge's own argument for the score — or the error cause when `failed` is true."""

    @classmethod
    def judged(
        cls, criterion: Criterion, score: float, reasoning: str | None
    ) -> "CriterionResult":
        """Build the result of a criterion the judge answered for.

        Args:
            criterion: The criterion that was judged; its `id` and `weight` are copied over
                so the result can be re-scored without the rubric at hand.
            score: The judge's score, expected on the 0..SCALE_MAX scale. Not clamped here —
                `parse_verdict` is what enforces the range.
            reasoning: The judge's argument, or None when the caller does not keep it.

        Returns:
            A `CriterionResult` with `failed=False` and `is_present` derived as
            `score >= PRESENCE_THRESHOLD` — the judge is never asked for that separately,
            which is one question less for it to get wrong.
        """
        return cls(
            criterion_id=criterion.id,
            weight=criterion.weight,
            score=score,
            is_present=score >= PRESENCE_THRESHOLD,
            reasoning=reasoning,
        )

    @classmethod
    def unjudged(cls, criterion: Criterion, cause: str) -> "CriterionResult":
        """Build the result of a criterion the judge never delivered a verdict for.

        Args:
            criterion: The criterion that could not be judged.
            cause: Why — reported verbatim as the result's `reasoning`.

        Returns:
            A `CriterionResult` with `failed=True` and `score=0.0` that **keeps its weight**
            and therefore stays in the denominator of `case_score`. An outage has to lower
            the score visibly rather than silently shrink the rubric.
        """
        return cls(
            criterion_id=criterion.id,
            weight=criterion.weight,
            score=0.0,
            is_present=False,
            failed=True,
            reasoning=cause,
        )


class Case(DocumentedModel):
    """One thing to evaluate: a question, the answer some system gave, and the rubric to
    hold it against. The unit of work everywhere — `POST /evaluate` takes one, a batch takes
    a list of them.

    Stateless: the caller owns questions, answers and rubric; nothing here is stored.

    Example:
        Case(id=1, question="How do I report sick leave?", answer="Email hr@...",
             criteria=[Criterion(id=1, content="Report by email before 10:00", weight=3)])
    """

    id: int
    """Caller-owned identifier, echoed back as `CaseResult.case_id` so results can be matched
    to the cases they came from without relying on list order. Required even for a single
    evaluation, so one result shape serves both paths."""

    question: str
    """The question that was asked. Passed to the judge as context only; it is never scored."""

    answer: str
    """The answer under test, produced by some other system. Judged exactly as it comes in."""

    criteria: list[Criterion] = Field(min_length=1)
    """The rubric: at least one criterion, because an empty rubric has no meaningful score.
    Ids have to be unique — they are what results are matched by."""

    @field_validator("criteria")
    @classmethod
    def _reject_duplicate_criterion_ids(cls, criteria: list[Criterion]) -> list[Criterion]:
        """Every verdict is labelled with its `Criterion.id`, so a repeated id makes results
        ambiguous: a caller keying by id would drop one verdict or count another twice."""
        if repeated := _duplicate_ids(criterion.id for criterion in criteria):
            raise ValueError(f"criterion ids must be unique, repeated: {repeated}")
        return criteria


class CaseResult(DocumentedModel):
    """What `evaluate_case` returns: the case score plus the verdicts it was computed from.

    The same document whether the case was evaluated alone or inside a batch — which is why
    `Case.id` is mandatory: one result type, no nullable id, nothing to reconcile.

    Always complete: a criterion the judge could not answer for is present with
    `failed=True`, never missing.

    Example:
        result.score                             # 0.75  — weighted, in [0, 1]
        result.criterion_results[0].score        # 2.0   — the raw judge score, 0 / 1 / 2
        result.criterion_results[0].reasoning    # why the judge gave it
    """

    case_id: int
    """The `Case.id` this result belongs to."""

    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    """Weighted case score in [0, 1], see `metrics.case_score`. 1.0 means every criterion
    was fully covered. Bounded for the same reason the verdict scores under it are: this
    model is what `/compare` takes in, and every delta of a comparison subtracts it."""

    criterion_results: list[CriterionResult] = Field(min_length=1)
    """One verdict per criterion of the case, in rubric order. At least one, because
    `Case.criteria` rejects an empty rubric and the metrics divide by this count. Ids have to
    be unique, exactly as in the rubric this came from. Named for what it holds: `criteria`
    would promise `Criterion` objects and deliver verdicts."""

    @field_validator("criterion_results")
    @classmethod
    def _reject_duplicate_criterion_ids(
        cls, criterion_results: list[CriterionResult]
    ) -> list[CriterionResult]:
        """`Case.criteria` already rejects repeated ids, so `evaluate_case` can never produce
        them — but a stored result is postable to `/compare`, which keys verdicts by id to
        pair the two runs up and would silently drop one of a repeated pair."""
        if repeated := _duplicate_ids(verdict.criterion_id for verdict in criterion_results):
            raise ValueError(f"criterion ids must be unique, repeated: {repeated}")
        return criterion_results


class Batch(DocumentedModel):
    """The input to `evaluate_batch`: several cases evaluated in one go, so a whole test
    catalog produces one set of run metrics instead of many isolated scores.

    Example:
        Batch(cases=[
            Case(id=1, question="How do I report sick leave?", answer="...", criteria=[...]),
            Case(id=2, question="How do I request vacation?", answer="...", criteria=[...]),
        ])
    """

    cases: list[Case] = Field(min_length=1)
    """The catalog: at least one case, because an empty run has no meaningful metrics.
    Ids have to be unique — they are what results are matched by."""

    @field_validator("cases")
    @classmethod
    def _reject_duplicate_case_ids(cls, cases: list[Case]) -> list[Case]:
        """Same reason as for criterion ids: a repeated id makes the run metrics ambiguous,
        because `cases_with_score_zero` and the weakest-case shortlist name cases by id."""
        if repeated := _duplicate_ids(case.id for case in cases):
            raise ValueError(f"case ids must be unique, repeated: {repeated}")
        return cases


class RunMetrics(DocumentedModel):
    """What a whole batch is judged by: the distribution of the case scores plus the few
    numbers that say where to look when it is bad.

    Every case counts once, whatever the size of its rubric — a case with 20 criteria must
    not outweigh nineteen cases with one.

    The documented ranges are enforced rather than described, for the same reason they are on
    the result models above: a stored run is postable to `/compare`, where every one of these
    numbers is subtracted from its counterpart.
    """

    total_cases: int = Field(ge=1)
    """How many cases the metrics were computed from. At least one: a run of no cases has no
    distribution to describe, which is why `run_metrics` refuses one."""

    average_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    """Arithmetic mean of the case scores — the single number a run is usually reported by."""

    median_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    """Middle case score. Next to the mean it shows skew: far above it means a few
    catastrophic cases drag an otherwise solid run down."""

    variance: float = Field(ge=0, allow_inf_nan=False)
    """Sample variance of the case scores, 0.0 for a single case (which has no spread)."""

    standard_deviation: float = Field(ge=0, allow_inf_nan=False)
    """Square root of `variance`, in the same unit as the scores. Small means the system is
    uniformly good or bad; large means it depends heavily on the question."""

    average_criterion_score: float = Field(ge=0, le=SCALE_MAX, allow_inf_nan=False)
    """Mean judge score over *all* criteria of all cases, on the raw 0..SCALE_MAX scale and
    unweighted. Unlike `average_score` it ignores both weights and case boundaries, so it
    answers "how well does the judge rate an average statement" rather than "how good is the
    average answer"."""

    criteria_fulfillment_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    """Mean share of criteria counting as covered (`is_present`) per case, in [0, 1].
    Averaged per case first, so a long rubric does not dominate the rate."""

    cases_with_score_zero: list[int]
    """Ids of the cases that scored exactly 0 — the answers that missed the rubric
    completely. These are the ones to read first."""

    weakest_cases_above_zero: list[int]
    """Ids of the up to `WEAKEST_CASES_REPORTED` lowest-scoring cases that scored above 0,
    weakest first. Listed apart from `cases_with_score_zero` because a total miss and a
    partial answer usually have different causes."""

    failed_criteria_count: int = Field(ge=0)
    """How many criteria across the whole run got no usable verdict and were counted as 0.
    Anything above 0 means the run is depressed by judge outages, not only by the answers —
    read it before the average."""

    @computed_field
    @property
    def cases_with_score_zero_count(self) -> int:
        """Length of `cases_with_score_zero`. Derived rather than stored, so the count and
        the list can never contradict each other."""
        return len(self.cases_with_score_zero)


class BatchResult(DocumentedModel):
    """What `evaluate_batch` returns: the aggregate plus every single case result it was
    computed from, so a suspicious number can always be traced back to its cases.

    Example:
        run.metrics.average_score           # 0.5
        run.metrics.cases_with_score_zero   # [2]  — the answers to read first
        run.case_results[0]                 # the CaseResult for case 1, in full
    """

    metrics: RunMetrics
    """The aggregate over all cases of this run."""

    case_results: list[CaseResult] = Field(min_length=1)
    """One result per case of the batch, in request order. Each one is exactly what
    `POST /evaluate` returns for that case. At least one, for the same reason `Batch.cases`
    needs one: `run_metrics` refuses to describe a distribution over no cases, so a run with
    an empty list could never have been produced legitimately — and anything computed from
    it later, a comparison above all, would be dividing by zero. Case ids have to be unique,
    for the same reason they do in `Batch.cases`."""

    @field_validator("case_results")
    @classmethod
    def _reject_duplicate_case_ids(cls, case_results: list[CaseResult]) -> list[CaseResult]:
        """`Batch.cases` already rejects repeated ids, so `evaluate_batch` can never produce
        them — but a stored run is postable to `/compare`, which keys cases by id to pair the
        two runs up. A repeated id would silently drop a case there and report deltas over
        fewer cases than `metrics` describes, with a 200."""
        if repeated := _duplicate_ids(result.case_id for result in case_results):
            raise ValueError(f"case ids must be unique, repeated: {repeated}")
        return case_results


class ChangeStatus(StrEnum):
    """Which way one score moved between two runs — the vocabulary of every comparison."""

    IMPROVED = "improved"
    """The candidate scored higher than the baseline, by more than `SCORE_EQUALITY_TOLERANCE`."""

    STABLE = "stable"
    """The two scores are equal within `SCORE_EQUALITY_TOLERANCE`."""

    WORSENED = "worsened"
    """The candidate scored lower than the baseline, by more than `SCORE_EQUALITY_TOLERANCE`."""


def _change_status(score_delta: float) -> ChangeStatus:
    """The one place a score delta becomes a verdict, so a criterion and a case can never
    classify the same movement differently.

    The tolerance is what keeps float noise out of the report: two runs that summed the same
    weights in a different order can land on 0.7500000000000001 and 0.75, and exact equality
    would file that as a regression.
    """
    if math.isclose(score_delta, 0.0, abs_tol=SCORE_EQUALITY_TOLERANCE):
        return ChangeStatus.STABLE
    return ChangeStatus.IMPROVED if score_delta > 0 else ChangeStatus.WORSENED


class CriterionComparisonResult(DocumentedModel):
    """How one criterion of one case fared between two runs — the finest grain of a comparison.

    Built by `between()`, never by hand, so `score_delta` and `status` cannot disagree.
    """

    criterion_id: int
    """The `Criterion.id` both runs judged. Identical in both by construction: a comparison
    of runs with different criterion ids is refused before any of this is computed."""

    weight: float
    """The criterion's weight, identical in both runs. Carried so a movement can be read
    against how much it counted — a 2.0 swing on weight 1 next to nine criteria of weight 3
    barely moves the case score."""

    baseline_score: float
    """What the baseline run's judge gave this criterion, on the raw 0..SCALE_MAX scale."""

    candidate_score: float
    """What the candidate run's judge gave it, same scale."""

    score_delta: float
    """`candidate_score - baseline_score`, so a positive number always means the candidate
    did better. On the raw 0..SCALE_MAX scale, hence in [-SCALE_MAX, SCALE_MAX]."""

    status: ChangeStatus
    """`score_delta` as a verdict. Derived from the raw score, not from `is_present`: a
    criterion that went from 1.0 to 2.0 moved the case score and is reported as improved."""

    @classmethod
    def between(
        cls, baseline: CriterionResult, candidate: CriterionResult
    ) -> "CriterionComparisonResult":
        """Compare the two verdicts one criterion got in two runs.

        Args:
            baseline: The verdict from the run being compared *against*.
            candidate: The verdict from the run *under test*, for the same criterion id and
                weight — `comparison.compare_runs` has already refused the runs otherwise.

        Returns:
            A `CriterionComparisonResult` whose `score_delta` points from baseline to candidate and
            whose `status` is derived from exactly that delta.

        Example:
            CriterionComparisonResult.between(before, after).status   # ChangeStatus.IMPROVED
        """
        score_delta = candidate.score - baseline.score
        return cls(
            criterion_id=candidate.criterion_id,
            weight=candidate.weight,
            baseline_score=baseline.score,
            candidate_score=candidate.score,
            score_delta=score_delta,
            status=_change_status(score_delta),
        )


class CaseComparisonResult(DocumentedModel):
    """How one case fared between two runs, and which of its criteria are responsible.

    Built by `between()`, never by hand.

    Example:
        case_comparison.score_delta               # +0.25 — the candidate answered better
        case_comparison.criterion_comparison_results[0]  # which criterion moved, and by how much
    """

    case_id: int
    """The `Case.id` both runs evaluated."""

    baseline_score: float
    """The baseline run's weighted case score, in [0, 1]."""

    candidate_score: float
    """The candidate run's weighted case score, in [0, 1]."""

    score_delta: float
    """`candidate_score - baseline_score`, in [-1, 1]. Positive means the candidate is better,
    which is the sign convention of every delta in a comparison."""

    status: ChangeStatus
    """`score_delta` as a verdict, with the same tolerance a criterion gets."""

    criterion_comparison_results: list[CriterionComparisonResult] = Field(min_length=1)
    """One entry per criterion of the case, ordered by `criterion_id`. At least one, because
    a rubric cannot be empty. Named for what it holds, not for what went in."""

    @classmethod
    def between(cls, baseline: CaseResult, candidate: CaseResult) -> "CaseComparisonResult":
        """Compare the two results one case got in two runs.

        Args:
            baseline: The case result from the run being compared against.
            candidate: The case result from the run under test, for the same case id and the
                same rubric — `comparison.compare_runs` has already refused the runs otherwise.

        Returns:
            A `CaseComparisonResult` carrying both scores, their delta and one
            `CriterionComparisonResult` per criterion, ordered by `criterion_id` so the list reads
            the same whichever order the two runs happened to be stored in.

        Example:
            CaseComparisonResult.between(weak, strong).status   # ChangeStatus.IMPROVED
        """
        score_delta = candidate.score - baseline.score
        return cls(
            case_id=candidate.case_id,
            baseline_score=baseline.score,
            candidate_score=candidate.score,
            score_delta=score_delta,
            status=_change_status(score_delta),
            criterion_comparison_results=[
                CriterionComparisonResult.between(baseline_criterion, candidate_criterion)
                for baseline_criterion, candidate_criterion in zip(
                    _criterion_results_in_id_order(baseline),
                    _criterion_results_in_id_order(candidate),
                    strict=True,
                )
            ],
        )


def _criterion_results_in_id_order(result: CaseResult) -> list[CriterionResult]:
    """Both runs' verdicts brought into one order, so zipping them pairs the same criterion.
    Rubric order is not enough: two runs may have been stored with their criteria in
    different orders, and zipping those would compare unrelated verdicts. Zipped `strict`,
    so a caller reaching `CaseComparisonResult.between` past `compare_runs` gets a crash rather
    than a silently truncated comparison."""
    return sorted(result.criterion_results, key=lambda verdict: verdict.criterion_id)


class RunMetricsDelta(DocumentedModel):
    """`RunMetrics` of the candidate minus those of the baseline — one field per metric that
    can meaningfully be subtracted.

    Every delta points the same way: **positive means the candidate scored higher**. For the
    two counting fields that reads backwards on purpose — a positive
    `cases_with_score_zero_count_delta` means the candidate produced *more* total misses.

    `total_cases` has no delta because a comparison of runs with different case sets is
    refused, and the two id lists of `RunMetrics` have none because a set of ids does not
    subtract — `ChangeSummary` reports the movement of cases instead.
    """

    average_score_delta: float
    """Change in the mean case score, in [-1, 1]. The headline number of a comparison."""

    median_score_delta: float
    """Change in the median case score. Read next to the mean: a mean that rose while the
    median stood still means a few cases improved, not the run as a whole."""

    variance_delta: float
    """Change in the sample variance of the case scores."""

    standard_deviation_delta: float
    """Change in the spread of the case scores. Negative means the candidate is more uniform
    — which is an improvement or a regression depending on which way the mean went."""

    average_criterion_score_delta: float
    """Change in the unweighted mean judge score over all criteria, on the 0..SCALE_MAX
    scale. Moves independently of `average_score_delta`, because it ignores both weights and
    case boundaries."""

    criteria_fulfillment_rate_delta: float
    """Change in the mean share of criteria counting as covered, in [-1, 1]."""

    cases_with_score_zero_count_delta: int
    """Change in how many cases missed their rubric completely. **Negative is the good
    direction here** — the candidate left fewer answers at zero."""

    failed_criteria_count_delta: int
    """Change in how many criteria got no usable verdict. Read this before any other delta:
    anything but 0 means the two runs suffered different amounts of judge outage, and every
    number above is then partly an artefact of that rather than of the answers."""


class ChangeMagnitude(DocumentedModel):
    """How large the moves on one side of a comparison were — improvements or regressions.

    All three numbers carry the sign of their side, so a worsening's `largest` is the most
    negative delta, not its absolute value. Every field is 0.0 when nothing moved that way,
    which is the honest reading: a run where nothing got worse has no worsening to report.
    """

    largest: float
    """The single biggest move on this side, or 0.0 if the side is empty."""

    mean: float
    """Arithmetic mean of the moves on this side — how much a typical one was worth."""

    median: float
    """Median of the moves. Far below the mean on the improvement side means one case
    carries the win."""


class ChangeSummary(DocumentedModel):
    """Where a run moved, case by case: which cases went which way, and by how much.

    The counterpart to `RunMetricsDelta`. That one says the average rose by 0.08; this one
    says whether every case rose a little or three rose a lot while one collapsed.

    Example:
        summary.improved_case_ids[:3]   # [7, 3, 2] — the three biggest wins, in order
        summary.worsening.largest       # -0.31     — the regression to read first
    """

    improved_case_ids: list[int]
    """Ids of the cases the candidate scored higher on, **biggest improvement first**. The
    list is complete rather than capped, so the top three are simply its first three."""

    stable_case_ids: list[int]
    """Ids of the cases whose score did not move beyond `SCORE_EQUALITY_TOLERANCE`, in id
    order. A stable case score can still hide criteria that moved in opposite directions."""

    worsened_case_ids: list[int]
    """Ids of the cases the candidate scored lower on, **biggest regression first**. These
    are the ones to read when an average went up and you want to know what it cost."""

    improvement: ChangeMagnitude
    """Size of the moves behind `improved_case_ids`, all positive."""

    worsening: ChangeMagnitude
    """Size of the moves behind `worsened_case_ids`, all negative."""

    @computed_field
    @property
    def improved_case_count(self) -> int:
        """Length of `improved_case_ids`. Derived, so count and list cannot contradict."""
        return len(self.improved_case_ids)

    @computed_field
    @property
    def stable_case_count(self) -> int:
        """Length of `stable_case_ids`."""
        return len(self.stable_case_ids)

    @computed_field
    @property
    def worsened_case_count(self) -> int:
        """Length of `worsened_case_ids`."""
        return len(self.worsened_case_ids)

    @computed_field
    @property
    def improvement_rate(self) -> float:
        """Share of the run's cases that improved, in [0, 1].

        Every case lands in exactly one of the three lists, so the three *counts* always add
        up to the run exactly. The three rates are three separate divisions, so they add up
        to 1.0 only to within float rounding — six cases split 1 / 4 / 1 sum to
        0.9999999999999999. Compare the counts when an exact total is what you need.
        """
        return self.improved_case_count / self._total_cases

    @computed_field
    @property
    def stability_rate(self) -> float:
        """Share of the run's cases that did not move, in [0, 1]."""
        return self.stable_case_count / self._total_cases

    @computed_field
    @property
    def worsening_rate(self) -> float:
        """Share of the run's cases that got worse, in [0, 1]."""
        return self.worsened_case_count / self._total_cases

    @property
    def _total_cases(self) -> int:
        """Every case is in exactly one of the three lists, so they add up to the run — no
        separate total to store and keep in sync. Never 0: `BatchResult` rejects a run with
        no cases, so the three rates above can always be divided out."""
        return self.improved_case_count + self.stable_case_count + self.worsened_case_count


class RunPair(DocumentedModel):
    """The input to `compare_runs`: two finished runs to hold against each other.

    A named pair rather than two arguments on purpose. Both sides have the exact same type,
    so a swapped pair of positional arguments would be impossible to detect and would invert
    the sign of every number in the result.

    Example:
        RunPair(baseline=last_weeks_run, candidate=todays_run)
    """

    baseline: BatchResult
    """The run being compared *against* — the state of things before your change."""

    candidate: BatchResult
    """The run *under test*. Every delta in the result is `candidate - baseline`, so a
    positive number always means this one did better."""


class ComparisonResult(DocumentedModel):
    """What `compare_runs` returns: the same comparison at three grains.

    `metrics_delta` says whether the run got better, `summary` says how that is distributed
    over the cases, and `case_comparison_results` says which criterion is responsible. A
    number at any grain can always be traced down to the one below it.

    Example:
        result.metrics_delta.average_score_delta   # +0.084
        result.summary.worsened_case_ids           # [5] — what the win cost
        result.case_comparison_results[0].criterion_comparison_results[1].score_delta  # -2.0
    """

    metrics_delta: RunMetricsDelta
    """Candidate minus baseline for every `RunMetrics` field that subtracts."""

    summary: ChangeSummary
    """How the movement is distributed over the cases: who won, who lost, by how much."""

    case_comparison_results: list[CaseComparisonResult] = Field(min_length=1)
    """One entry per case, ordered by `case_id`. At least one, because both compared runs
    have at least one case. Both runs cover exactly the same cases — that is checked before
    anything here is computed — so neither run's storage order is the canonical one, and
    sorting by id gives a document that does not depend on either."""
