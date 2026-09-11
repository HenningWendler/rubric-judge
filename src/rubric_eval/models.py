"""Pydantic models. The result shape is a published interface: grow it additively.

Public fields and constants are documented with docstrings rather than comments: the
IDE shows them on hover, and `use_attribute_docstrings` copies them into the OpenAPI
schema, so the editor and `/docs` can never drift apart.
"""

from collections import Counter
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

SCALE_MAX = 2
"""Best score one criterion can reach. The scale is integral: 0, 1 or 2."""

PRESENCE_THRESHOLD = 0.5
"""Half a scale point. With a single judge run this is simply "not a plain 0"; with
several runs it is the majority rule applied to the averaged score."""


class DocumentedModel(BaseModel):
    """Base for every model in this package: makes attribute docstrings the field
    descriptions, so one docstring feeds both IDE hover and the generated OpenAPI schema."""

    model_config = ConfigDict(use_attribute_docstrings=True)


class Criterion(DocumentedModel):
    """One statement a good answer has to contain — the atom of a rubric.

    A rubric is just a list of these, and each one is judged on its own, one LLM call
    per criterion:

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
        """The judge answered. `is_present` is derived, never asked for separately."""
        return cls(
            criterion_id=criterion.id,
            weight=criterion.weight,
            score=score,
            is_present=score >= PRESENCE_THRESHOLD,
            reasoning=reasoning,
        )

    @classmethod
    def unjudged(cls, criterion: Criterion, cause: str) -> "CriterionResult":
        """The judge never delivered: counts as 0 and stays in the denominator (see metrics)."""
        return cls(
            criterion_id=criterion.id,
            weight=criterion.weight,
            score=0.0,
            is_present=False,
            failed=True,
            reasoning=cause,
        )


class EvaluationResult(DocumentedModel):
    """Everything the evaluator knows about one (question, answer) pair: the case score
    plus the per-criterion verdicts it was computed from."""

    score: float
    """Weighted case score in [0, 1], see `metrics.case_score`. 1.0 means every criterion
    was fully covered."""

    criteria: list[CriterionResult]
    """One verdict per criterion of the request, in request order."""


class EvaluateRequest(DocumentedModel):
    """Request body of `POST /evaluate`: one answer plus the rubric to hold it against.

    Stateless — the caller owns questions, answers and rubric; nothing here is stored.
    """

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
        counted = Counter(criterion.id for criterion in criteria)
        if repeated := sorted(id_ for id_, count in counted.items() if count > 1):
            raise ValueError(f"criterion ids must be unique, repeated: {repeated}")
        return criteria
