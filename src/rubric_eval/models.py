"""Pydantic models. The result shape is a published interface: grow it additively.

Public fields and constants are documented with docstrings rather than comments: the
IDE shows them on hover, and `use_attribute_docstrings` copies them into the OpenAPI
schema, so the editor and `/docs` can never drift apart.
"""

from collections import Counter
from collections.abc import Iterable
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


def duplicate_ids(ids: Iterable[int]) -> list[int]:
    """Find the ids that occur more than once.

    Shared by the rubric and the batch, because both match results back to their input by
    id and a repeat breaks both in exactly the same way.

    Args:
        ids: The ids to check, in any order. Consumed once, so a generator is fine.

    Returns:
        The repeated ids, sorted and each listed once. Empty when every id is unique —
        which makes it directly usable as a validator condition.

    Example:
        duplicate_ids([3, 1, 3, 1, 2])   # [1, 3]
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
    """

    criterion_id: int
    """The `Criterion.id` this verdict belongs to."""

    weight: float
    """Copy of `Criterion.weight`, so a result can be scored without the rubric at hand."""

    score: float
    """Judge score in [0, SCALE_MAX]: 2 fully covered, 1 partially, 0 not covered.
    A float rather than an int, so averaging several runs of the same criterion cannot
    change the type."""

    is_present: bool
    """Whether the criterion counts as covered at all: `score >= PRESENCE_THRESHOLD`.
    Derived here and never asked of the judge — one question less for it to get wrong."""

    spread: float = 0.0
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
        if repeated := duplicate_ids(criterion.id for criterion in criteria):
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

    score: float
    """Weighted case score in [0, 1], see `metrics.case_score`. 1.0 means every criterion
    was fully covered."""

    criterion_results: list[CriterionResult]
    """One verdict per criterion of the case, in rubric order. Named for what it holds:
    `criteria` would promise `Criterion` objects and deliver verdicts."""


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
        if repeated := duplicate_ids(case.id for case in cases):
            raise ValueError(f"case ids must be unique, repeated: {repeated}")
        return cases


class RunMetrics(DocumentedModel):
    """What a whole batch is judged by: the distribution of the case scores plus the few
    numbers that say where to look when it is bad.

    Every case counts once, whatever the size of its rubric — a case with 20 criteria must
    not outweigh nineteen cases with one.
    """

    total_cases: int
    """How many cases the metrics were computed from."""

    average_score: float
    """Arithmetic mean of the case scores — the single number a run is usually reported by."""

    median_score: float
    """Middle case score. Next to the mean it shows skew: far above it means a few
    catastrophic cases drag an otherwise solid run down."""

    variance: float
    """Sample variance of the case scores, 0.0 for a single case (which has no spread)."""

    standard_deviation: float
    """Square root of `variance`, in the same unit as the scores. Small means the system is
    uniformly good or bad; large means it depends heavily on the question."""

    average_criterion_score: float
    """Mean judge score over *all* criteria of all cases, on the raw 0..SCALE_MAX scale and
    unweighted. Unlike `average_score` it ignores both weights and case boundaries, so it
    answers "how well does the judge rate an average statement" rather than "how good is the
    average answer"."""

    criteria_fulfillment_rate: float
    """Mean share of criteria counting as covered (`is_present`) per case, in [0, 1].
    Averaged per case first, so a long rubric does not dominate the rate."""

    cases_with_score_zero: list[int]
    """Ids of the cases that scored exactly 0 — the answers that missed the rubric
    completely. These are the ones to read first."""

    weakest_cases_above_zero: list[int]
    """Ids of the up to `WEAKEST_CASES_REPORTED` lowest-scoring cases that scored above 0,
    weakest first. Listed apart from `cases_with_score_zero` because a total miss and a
    partial answer usually have different causes."""

    failed_criteria_count: int
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

    case_results: list[CaseResult]
    """One result per case of the batch, in request order. Each one is exactly what
    `POST /evaluate` returns for that case."""
