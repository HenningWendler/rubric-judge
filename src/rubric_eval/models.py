"""Pydantic models. The result shape is a published interface: grow it additively.

Public fields and constants are documented with docstrings rather than comments: the
IDE shows them on hover, and `use_attribute_docstrings` copies them into the OpenAPI
schema, so the editor and `/docs` can never drift apart.
"""

import math
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum
from operator import attrgetter
from typing import Annotated, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    computed_field,
    field_validator,
    model_validator,
)

WEAKEST_CASES_REPORTED = 5
"""How many of the weakest non-zero cases `RunMetrics` names by id — the shortlist to look
at next, not a complete ranking."""

SCORE_EQUALITY_TOLERANCE = 1e-9
"""How close two scores have to be to count as unchanged in a comparison. Far above the
float noise two runs accumulate summing the same weights in a different order, and far below
the smallest score difference a rubric can actually produce."""

_WIDEST_SCORE_VARIANCE = 0.5
"""The largest sample variance a run of case scores can reach, since every case score lies in
[0, 1]: the widest run there is puts half its cases at 0 and half at 1, and two such cases
land on exactly this."""

_WIDEST_SCORE_STANDARD_DEVIATION = math.sqrt(_WIDEST_SCORE_VARIANCE)
"""Its square root, and therefore the same bound expressed in score units."""


Identifier = TypeVar("Identifier", int, str)
"""The two kinds of identifier this package matches things back by: the integer ids of cases
and criteria, and the string labels of a case."""


def _reject_duplicates(subject: str, values: Iterable[Identifier]) -> None:
    """One check and one wording for every repeated identifier in this package.

    The criterion ids of a rubric, the case ids of a run, the same two again on the results
    of either, and the labels of a case: each of them matches something back by that
    identifier, so a repeat breaks all five the same way. Written out per model, the five
    refusals would eventually word one rule five ways.

    Args:
        subject: What is repeated, as the message says it — "criterion ids", "case ids",
            "labels". Plural, because the sentence reads "<subject> must be unique".
        values: The identifiers to check, all of one type. Order is irrelevant.

    Raises:
        ValueError: At least one value occurs twice. The message names every repeat at once,
            sorted, so one fix can address the whole collision.
    """
    counted = Counter(values)
    if repeated := sorted(value for value, count in counted.items() if count > 1):
        raise ValueError(f"{subject} must be unique, repeated: {repeated}")


LevelDescription = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""What one grade of a `Scale` means, in plain language. Stripped and never blank: it is
rendered into the judge's prompt, where a bare number with nothing after it would read as an
instruction to guess."""

Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""One tag on a case — `"table"`, `"multi_page_expected"`. The vocabulary is yours and is
never checked against a list: a library cannot know your catalog's taxonomy. Stripped and
never blank, because a blank tag would open a bucket in the metrics that names nothing.

Labels never reach the judge. They slice a run; they do not grade an answer. A property that
should change the grade belongs in a `Criterion`, where it is checkable and weighted."""


def _reject_duplicate_labels(labels: list[Label]) -> list[Label]:
    """Exists because a repeated label is a caller mistake no code below could act on.

    Labels are read as a set everywhere: bucketing, filtering, comparing two runs.
    """
    _reject_duplicates("labels", labels)
    return labels


Labels = Annotated[list[Label], AfterValidator(_reject_duplicate_labels)]
"""A case's tags: stripped, never blank, no repeats. One type for the input and the echo, so
`Case.labels` and `CaseResult.labels` cannot drift into two different rules."""


def _reject_duplicate_groups(label_filter: list[list[str]]) -> list[list[str]]:
    """Exists because the same group twice selects nothing extra.

    A group is read as a set of required labels, so `["a", "b"]` and `["b", "a"]` ask one
    and the same question.
    """
    counted = Counter(frozenset(group) for group in label_filter)
    if repeated := sorted(sorted(group) for group, count in counted.items() if count > 1):
        raise ValueError(f"label groups must be unique, repeated: {repeated}")
    return label_filter


LabelFilter = Annotated[list[Labels], AfterValidator(_reject_duplicate_groups)]
"""Which cases a run covers, as an **OR of ANDs**: a case is selected when it carries every
label of at least one group.

Every boolean combination of labels can be written this way, which is why no expression
grammar is needed — one more level of list is the whole feature:

    [["table", "split_infos"], ["agentic"]]   # (table AND split_infos) OR agentic
    [["table", "images"]]                     # table AND images
    [["table"], ["images"]]                   # table OR images
    []                                        # everything — what "no filter" means

Negation is deliberately absent: "table but not images" cannot be written, and adding it
would mean either a second field or a sigil inside a label, neither of which has been asked
for yet."""


read_label_filter = TypeAdapter(LabelFilter).validate_python
"""The same rule, applied to a label filter pydantic has not been through — the one
`filter_cases_by_labels` takes straight from a caller rather than off a `Run` field. Without
it a label with a stray space would match nothing where the identical label filter on a `Run`
matches two cases, and the preview would contradict the run it previews."""


class DocumentedModel(BaseModel):
    """Base for every model in this package.

    It turns attribute docstrings into field descriptions, so one docstring feeds both the
    IDE hover and the generated OpenAPI schema and the two cannot drift apart.

    Example:
        Criterion.model_json_schema()["properties"]["id"]["description"]
        # "Caller-owned identifier, echoed back as `CriterionResult.criterion_id` so results
        #  can be matched to the rubric without relying on list order."
    """

    model_config = ConfigDict(use_attribute_docstrings=True)


class Scale(DocumentedModel):
    """The grading scale one judge works on.

    How far a criterion can be covered, and from where on it counts as covered at all.

    A judge owns its scale and the `CaseResult` it produces carries it, so nothing downstream
    has to assume the bundled 0..2. Compared by value — including the level descriptions, so
    rewording what a grade means makes it a different scale and two runs judged under the two
    wordings are refused rather than subtracted.

    Describing the levels is what lets `prompt.judge_prompt` write the judge's instructions
    from the scale alone; a scale that describes none is arithmetic only and needs a prompt
    handed to the judge.

    Frozen, and revalidated whenever it is put into a result: `frozen` stops the fields being
    reassigned but not `level_descriptions` being written into, and `DEFAULT_SCALE` is one
    module-level object every judge that takes the default shares. Revalidating copies it into
    each `CaseResult`, so a finished run keeps saying what its grades meant even if someone
    reaches into that constant afterwards — which is the whole reason a result carries a scale.

    Example:
        pass_fail = Scale(
            maximum=1,
            presence_threshold=1,
            level_descriptions={1: "Covered.", 0: "Not covered."},
        )
        list(pass_fail.grades)                   # [0, 1]
        Scale(maximum=10, presence_threshold=5)  # arithmetic only, bring your own prompt
    """

    model_config = ConfigDict(frozen=True, revalidate_instances="always")

    maximum: int = Field(gt=0)
    """Best score one criterion can reach; the scale runs 0..maximum and is integral, so a
    maximum of 2 offers exactly the grades 0, 1 and 2. Keep it small — a judge asked to pick
    one of a hundred levels is guessing, not being precise, and every level has to be spelled
    out in the prompt."""

    presence_threshold: float = Field(gt=0, allow_inf_nan=False)
    """From which score `CriterionResult.is_present` counts the criterion as covered. Above 0
    because a criterion the judge scored 0 is by definition not covered, and at most
    `maximum` because a threshold beyond the scale would make every criterion absent. On the
    bundled scale it is 0.5 — "not a plain 0" for a single run, the majority rule once
    repeated runs average into fractional scores."""

    level_descriptions: dict[int, LevelDescription] = Field(default_factory=dict)
    """What each grade means, keyed by the grade: `{2: "Fully covered. …", 1: "…", 0: "…"}`.
    Either empty or complete — one entry for every grade from 0 to `maximum`, and none for a
    grade off it — because a prompt that explains four of ten levels is worse than one that
    explains none. This is judge-facing prose *and* data a result carries: it is what
    `prompt.judge_prompt` renders, and what tells a reader of a stored run six months later
    what its 7 out of 10 was supposed to mean."""

    @model_validator(mode="after")
    def _reject_descriptions_that_do_not_match_the_grades(self) -> "Scale":
        """Exists because half a description block is worse than none at all.

        A prompt that lists some levels and leaves the judge to invent the rest reads as a
        deliberate omission, where the zero-shot prompt of an undescribed scale does not.
        """
        if self.level_descriptions and set(self.level_descriptions) != set(self.grades):
            raise ValueError(
                f"level descriptions must describe every grade of the scale 0..{self.maximum} "
                f"and no other, got {sorted(self.level_descriptions)}"
            )
        return self

    @model_validator(mode="after")
    def _reject_a_threshold_off_the_scale(self) -> "Scale":
        """Exists because an unreachable threshold would make every criterion absent.

        No grade could reach it, and nothing would raise to explain the run of nothing but
        uncovered criteria that follows.
        """
        if self.presence_threshold > self.maximum:
            raise ValueError(
                f"presence threshold {self.presence_threshold} is above the highest reachable "
                f"score {self.maximum}, so no criterion could ever count as covered"
            )
        return self

    @property
    def grades(self) -> range:
        """Every grade this scale offers, lowest first: 0, 1, 2 for the bundled one.

        Returns:
            A `range` from 0 through `maximum` inclusive — the one place that "inclusive" is
            spelled out, so the validator, the prompt and the retry hints cannot disagree
            about whether the top grade is on the scale.

        Example:
            list(DEFAULT_SCALE.grades)   # [0, 1, 2]
        """
        return range(self.maximum + 1)

    def __str__(self) -> str:
        """Name this scale the way a refusal has to name it.

        Returns:
            The bounds and the presence threshold, never the level descriptions — a message
            holding two scales against each other would otherwise run to a paragraph per
            scale. `reworded_grades_clause` adds the wording back, and only when it is what
            differs.

        Example:
            str(DEFAULT_SCALE)   # "0..2 (covered from 0.5)"
        """
        return f"0..{self.maximum} (covered from {self.presence_threshold})"


DEFAULT_SCALE = Scale(
    maximum=2,
    presence_threshold=0.5,
    level_descriptions={
        2: (
            "Fully covered. Every essential part of the criterion is clearly recognizable in "
            "the answer, even if the wording, terminology or structure differs."
        ),
        1: (
            "Partially covered. Some essential information is missing, but the basic idea is "
            "still derivable from the answer."
        ),
        0: (
            "Not covered. The criterion is absent, or the answer has no recognizable "
            "connection to it."
        ),
    },
)
"""The scale `prompt.JUDGE_EN` is generated from, and the one `OpenAIJudge` grades on unless
it is given another. Never a stand-in for a scale a document failed to name: a result carries
the scale it was judged on or is refused, because guessing the unit of a set of scores is how
a ten-point run ends up compared against a three-point one.

The three sentences live here rather than in `prompt.py` because they are not only prompt
text: a `CaseResult` carries them, so a run read back months later still says what its grades
were supposed to mean. `prompt.py` renders them and owns every other word the judge sees."""


def one_scale_of(scales: Iterable[Scale]) -> Scale:
    """The single scale a set of criterion results was given on, or a refusal naming the mix.

    The one place the "a run is judged on exactly one scale" rule lives, so the models, the
    metrics and the comparison cannot come to different conclusions about the same run. Mixed
    scales are not a cosmetic problem: `RunMetrics.average_criterion_score` averages raw judge
    scores across every criterion of a run, and a 2 meaning "fully covered" averaged with a 2
    meaning "barely" produces a number with no unit.

    Compared pairwise rather than through a `set`: a `Scale` carries its level descriptions in
    a dict and is therefore not hashable. The lists are one entry per case, so the cost is
    nothing and the alternative would be a second representation to keep in sync.

    Args:
        scales: The scales to reduce — one per case of a run. Must not be empty; every caller
            reaches it past a `min_length=1` field.

    Returns:
        The `Scale` they all agree on.

    Raises:
        ValueError: They do not agree. The message names every distinct scale found.

    Example:
        one_scale_of(case_result.scale for case_result in run_result.case_results)
    """
    distinct: list[Scale] = []
    for scale in scales:
        if scale not in distinct:
            distinct.append(scale)
    if len(distinct) > 1:
        raise ValueError(
            "all cases of a run must be judged on one scale, got: "
            + ", ".join(sorted(str(scale) for scale in distinct))
            + reworded_grades_clause(distinct)
        )
    return distinct[0]


def reworded_grades_clause(scales: list[Scale]) -> str:
    """The clause that explains scales which print alike but are not the same scale.

    `Scale.__str__` names the maximum and the threshold, so scales differing only in what they
    told the judge a grade *means* all print identically — and a refusal reading "got: 0..2
    (covered from 0.5), 0..2 (covered from 0.5)" names a contradiction instead of a cause. Both
    refusals that hold scales against each other end with this, so a rewording is reported the
    same way whether it shows up between the cases of one run or between two runs.

    Args:
        scales: The scales that were found to differ. Two or more; fewer cannot disagree.

    Returns:
        A clause to append to a message that has already named them, or "" when they print
        differently and the message therefore already says what differs.

    Example:
        reworded_grades_clause([DEFAULT_SCALE, default_with_a_reworded_two])
        # " — same grades, but the wording of [2] differs"
    """
    if len({str(scale) for scale in scales}) > 1:
        return ""
    described_grades = {grade for scale in scales for grade in scale.level_descriptions}
    reworded_grades = sorted(
        grade
        for grade in described_grades
        if len({scale.level_descriptions.get(grade) for scale in scales}) > 1
    )
    if not reworded_grades:
        return ""
    return f" — same grades, but the wording of {reworded_grades} differs"


def scores_must_fit(criterion_results: list["CriterionResult"], scale: Scale) -> None:
    """Refuse criterion results graded above what the scale can carry.

    The one place that rule lives, because two paths need it and must not word it differently:
    `metrics.case_score` divides by `scale.maximum` while a case is being built, and
    `CaseResult` re-checks the same thing for a finished run read back from JSON. Caught early
    on either path, an out-of-scale grade would otherwise surface as a case score above 1.0 —
    a number that names neither the criterion nor the judge that produced it.

    Args:
        criterion_results: The results of one case. Only `score` and `criterion_id` are read.
        scale: The scale they were given on.

    Raises:
        ValueError: At least one criterion result is above `scale.maximum`. The message
            names every offending criterion at once and the scale they were held against.

    Example:
        scores_must_fit(case_result.criterion_results, case_result.scale)
    """
    if off_scale := [
        criterion_result.criterion_id
        for criterion_result in criterion_results
        if criterion_result.score > scale.maximum
    ]:
        raise ValueError(
            f"criteria {off_scale} scored above the scale {scale} they were judged on"
        )


def carries_every_label(case_labels: list[str], required_labels: list[str]) -> bool:
    """Whether one case carries **all** of the required labels — one group of a label filter.

    The one place the AND rule lives, because two callers depend on it meaning the same
    thing: `metrics.label_metrics` buckets with it and `matches_label_filter` selects with
    it. That is what makes "the run selected by `table`" and "the `table` bucket of the full
    run" the same cases — written out twice, one of them would eventually drift to "any".

    Args:
        case_labels: The labels the case carries. Order and repeats are irrelevant.
        required_labels: The labels being asked for — all of them, not any. Empty asks
            nothing, so every case matches.

    Returns:
        True when `case_labels` covers `required_labels`.

    Example:
        carries_every_label(["table", "images"], ["table"])   # True — subset, not equality
        carries_every_label(["table"], ["table", "images"])   # False
    """
    return set(required_labels) <= set(case_labels)


def matches_label_filter(case_labels: list[str], label_filter: list[list[str]]) -> bool:
    """Whether one case is covered by a `LabelFilter` — any group, every label of it.

    The OR half of the rule, on top of the AND half above. One place, for the same reason:
    `Run` selects the cases to run with it, `RunResult` validates what ran with it, and
    `filter_cases_by_labels` lets you ask which cases it would pick without running them.

    Args:
        case_labels: The labels the case carries.
        label_filter: Groups of required labels. Empty selects every case, which is what
            "no filter" means — and what keeps an unfiltered run a run.

    Returns:
        True when the case carries every label of at least one group.

    Example:
        matches_label_filter(["table", "split_infos"],
                             [["table", "split_infos"], ["agentic"]])      # True
        matches_label_filter(["agentic"], [["table", "split_infos"], ["agentic"]])  # True
        matches_label_filter(["table"], [["table", "split_infos"], ["agentic"]])    # False
    """
    return not label_filter or any(
        carries_every_label(case_labels, group) for group in label_filter
    )


def every_case_must_match(
    applied_label_filter: list[list[str]], labels_by_case_id: dict[int, list[str]]
) -> None:
    """Refuse a finished run whose `applied_label_filter` does not describe the cases it holds.

    `RunResult.applied_label_filter` claims "these are the cases that filter picked". The
    claim is checkable in one line, so it is checked rather than believed — the same stance
    `CaseResult` takes on a criterion result whose `is_present` contradicts its own score.
    Left unchecked, a stored run could call itself the `table` subset while holding the whole
    catalog, and every number read off it later would answer a different question than its
    name promises.

    Only results are held to this. On a `Run` the same groups are an *instruction* under the
    name `label_filter`, and an instruction cannot lie: the run is expected to carry cases the
    filter excludes, which is the entire point of handing it a catalog and a filter.

    Args:
        applied_label_filter: The groups the cases were selected by. Empty claims nothing and
            is always accepted — it is what an unfiltered run carries.
        labels_by_case_id: The labels of every case, keyed by case id. Keyed rather than
            listed so the message can name the offenders.

    Raises:
        ValueError: At least one case matches no group of the label filter. The message names
            all of them at once, so one fix can address the whole mismatch.

    Example:
        every_case_must_match([["table"]], {r.case_id: r.labels for r in case_results})
    """
    if missing := sorted(
        case_id
        for case_id, case_labels in labels_by_case_id.items()
        if not matches_label_filter(case_labels, applied_label_filter)
    ):
        raise ValueError(
            f"applied_label_filter {applied_label_filter} does not describe this run: "
            f"cases {missing} match none of its groups"
        )


def case_count_per_label(labels_per_case: Iterable[list[str]]) -> Counter[str]:
    """How many of these cases carry each label — one reading of "which labels are in play".

    The one place that question is answered, because two callers ask it for different
    reasons: `metrics.label_metrics` reads the labels off it, to know which buckets exist,
    and `labels_with_case_counts` reads the counts too. Answered twice, the two would
    eventually stop agreeing on what is in play — the bucket breakdown of a run and the
    refusal naming its labels have to be built from the same reading.

    Args:
        labels_per_case: The labels of each case, one list per case. A case carrying no label
            contributes nothing; an empty iterable is fine and counts nothing.

    Returns:
        How many cases carry each label. Empty when no case carries one — which is what an
        untagged catalog honestly looks like, not a missing answer.

    Example:
        case_count_per_label(case.labels for case in run.cases)   # {"table": 2, "images": 1}
    """
    return Counter(label for case_labels in labels_per_case for label in case_labels)


def labels_with_case_counts(labels_per_case: Iterable[list[str]]) -> str:
    """The labels a catalog actually carries, counted, as one line for a refusal to quote.

    What a label filter matching nothing is missing: that is a typo far more often than a
    genuinely empty subset, and the right spelling is unguessable from "nothing matched"
    alone.

    Args:
        labels_per_case: The labels of each case of the catalog, one list per case.

    Returns:
        The labels in alphabetical order with the number of cases carrying each — "images
        (1), table (2)" — or the word "none" for a catalog carrying no label at all, so the
        sentence quoting it never trails off into nothing.

    Example:
        labels_with_case_counts(case.labels for case in run.cases)   # "images (1), table (2)"
    """
    counted = case_count_per_label(labels_per_case)
    carried = ", ".join(f"{label} ({count})" for label, count in sorted(counted.items()))
    return carried or "none"


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
    """What one criterion was given: the judge's grade for it, and why.

    Not built by hand — use `judged()`, so the derived fields stay consistent everywhere.
    There is no constructor for a criterion the judge never answered for: a result nobody
    gave is not a result with a 0 in it, and the run it belongs to is invalidated instead.

    Every documented range is enforced, not merely described: a stored result is postable to
    `/compare`, so this model is an *input* type there and the numbers below are arithmetic
    a comparison depends on. `NaN` in particular would survive every computation, serialize
    as JSON `null` where a float is promised, and classify as a regression.

    Example:
        criterion_result = CriterionResult.judged(
            Criterion(id=1, content="Report by email before 10:00", weight=3),
            2.0,
            "The answer instructs the reader to email hr@example.com.",
            DEFAULT_SCALE,
        )
        criterion_result.is_present   # True
        criterion_result.spread       # 0.0 — one judge run, so no spread to report
    """

    criterion_id: int
    """The `Criterion.id` this result belongs to."""

    weight: float = Field(gt=0, allow_inf_nan=False)
    """Copy of `Criterion.weight`, so a result can be scored without the rubric at hand.
    Positive and finite, exactly as the rubric it was copied from."""

    score: float = Field(ge=0, allow_inf_nan=False)
    """The judge's **raw** grade, not normalized: on the bundled scale 2 is fully covered, 1
    partially, 0 not covered. A float rather than an int, so averaging several runs of the
    same criterion cannot change the type. Its upper bound is `CaseResult.scale.maximum` and
    is checked there — a result on its own does not know which scale it was given on."""

    is_present: bool
    """Whether the criterion counts as covered at all: `score >= scale.presence_threshold`,
    with the scale of the `CaseResult` this result belongs to. Never asked of the judge —
    one question less for it to get wrong — and `CaseResult` refuses a result whose value
    here contradicts its own score, so it cannot drift away from the number it describes."""

    spread: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    """Standard deviation of `score` across repeated judge runs. Stays 0.0 while every
    criterion is judged exactly once, which is the only mode implemented so far."""

    reasoning: str | None = None
    """The judge's own argument for the score."""

    @classmethod
    def judged(
        cls, criterion: Criterion, score: float, reasoning: str | None, scale: Scale
    ) -> "CriterionResult":
        """Build the result of a criterion the judge answered for.

        Args:
            criterion: The criterion that was judged; its `id` and `weight` are copied over
                so the result can be re-scored without the rubric at hand.
            score: The judge's score, expected on `scale`. A score above its maximum is
                refused rather than clamped.
            reasoning: The judge's argument, or None when the caller does not keep it.
            scale: The scale the judge works on — `judge.scale`, never a guess. Not stored
                on the result (the `CaseResult` above it carries it once for the whole
                case); it is what `is_present` is cut at here.

        Returns:
            A `CriterionResult` carrying the judge's raw score, with `is_present` set from
            `scale` — `False` for a 0 on every scale, since a presence threshold is always
            above it.

        Raises:
            ValidationError: `score` is negative or not finite. Whether it fits the scale is
                checked by the `CaseResult` it goes into, which is the object that knows.

        Example:
            CriterionResult.judged(
                Criterion(id=1, content="Report by email before 10:00", weight=3),
                0.0,
                "The answer never mentions email.",
                DEFAULT_SCALE,
            ).is_present   # False
        """
        return cls(
            criterion_id=criterion.id,
            weight=criterion.weight,
            score=score,
            is_present=score >= scale.presence_threshold,
            reasoning=reasoning,
        )


class Case(DocumentedModel):
    """One thing to evaluate: a question, an answer, and the rubric to hold it against.

    The unit of work everywhere — `POST /evaluate` takes one, a run takes a list of them.

    Stateless: the caller owns questions, answers and rubric; nothing here is stored.

    Example:
        Case(
            id=1,
            question="How do I report sick leave?",
            answer="Email hr@example.com before 10:00.",
            criteria=[Criterion(id=1, content="Report by email before 10:00", weight=3)],
            labels=["one_page_expected"],
        )
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

    labels: Labels = Field(default_factory=list)
    """What kind of case this is — `["table", "multi_page_expected"]`. Free-form tags in your
    own vocabulary, used to slice a run: `RunResult.label_metrics` reports a full set of
    numbers per label, and `filter_cases_by_labels` selects by them. Optional, because an
    untagged catalog is still a catalog. Never shown to the judge, so adding a label cannot
    move a single score — a requirement the answer has to meet belongs in `criteria`."""

    @field_validator("criteria")
    @classmethod
    def _reject_duplicate_criterion_ids(cls, criteria: list[Criterion]) -> list[Criterion]:
        """Exists because a repeated id makes the results of a case ambiguous.

        Every result is labelled with its `Criterion.id`, so a caller keying by id would
        drop one result or count another twice.
        """
        _reject_duplicates("criterion ids", (criterion.id for criterion in criteria))
        return criteria


class CaseResult(DocumentedModel):
    """What `evaluate_case` returns: the case score plus the criterion results behind it.

    The same document whether the case was evaluated alone or inside a run — which is why
    `Case.id` is mandatory: one result type, no nullable id, nothing to reconcile.

    Always complete: one result per criterion of the rubric, never a hole. A case the judge
    could not answer every criterion of produces no result at all.

    Example:
        case_result = CaseResult(
            case_id=1,
            score=1.0,
            scale=DEFAULT_SCALE,
            criterion_results=[
                CriterionResult.judged(
                    Criterion(id=1, content="Report by email before 10:00", weight=3),
                    2.0,
                    "The answer instructs the reader to email hr@example.com.",
                    DEFAULT_SCALE,
                )
            ],
        )
        case_result.score                          # 1.0 — weighted, in [0, 1]
        case_result.criterion_results[0].score     # 2.0 — the raw judge grade
        case_result.criterion_results[0].reasoning # why the judge gave it
    """

    case_id: int
    """The `Case.id` this result belongs to."""

    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    """Weighted case score in [0, 1], see `metrics.case_score`. 1.0 means every criterion
    was fully covered. Bounded for the same reason the criterion scores under it are: this
    model is what `/compare` takes in, and every delta of a comparison subtracts it."""

    scale: Scale
    """The grading scale every result below was given on, copied off the judge once for the
    whole case. Here rather than on each result because a case is judged by one judge on one
    scale, and `evaluate_case` returns this document on its own — so this is the lowest level
    that always exists. Required and never defaulted: a default would decide the unit of every
    score under it, so a stored 0..10 result that lost this field would validate as a 0..2 one
    and `/compare` would subtract it from a genuine 0..2 run and answer 200."""

    criterion_results: list[CriterionResult] = Field(min_length=1)
    """One result per criterion of the case, in rubric order. At least one, because
    `Case.criteria` rejects an empty rubric and the metrics divide by this count. Ids have to
    be unique, exactly as in the rubric this came from. Named for what it holds: `criteria`
    would promise `Criterion` objects and deliver results."""

    labels: Labels = Field(default_factory=list)
    """The `Case.labels` this result came from, copied over so a stored run can still be sliced
    by label months later without the catalog at hand. Defaulted, because a result that names
    none was either written before labels existed or came from an untagged case — both simply
    belong to no bucket."""

    @field_validator("criterion_results")
    @classmethod
    def _reject_duplicate_criterion_ids(
        cls, criterion_results: list[CriterionResult]
    ) -> list[CriterionResult]:
        """Exists for the stored results `evaluate_case` never produced.

        `Case.criteria` already rejects repeated ids, but a stored result is postable to
        `/compare`, which keys results by id to pair the two runs up and would silently drop
        one of a repeated pair.
        """
        _reject_duplicates(
            "criterion ids",
            (criterion_result.criterion_id for criterion_result in criterion_results),
        )
        return criterion_results

    @model_validator(mode="after")
    def _reject_scores_off_the_scale(self) -> "CaseResult":
        """Exists because a criterion result has no upper bound of its own.

        This is the object that knows the scale. A fresh case is guarded by `case_score`;
        this guards the other path, a finished run read back from JSON and posted to
        `/compare`.
        """
        scores_must_fit(self.criterion_results, self.scale)
        return self

    @model_validator(mode="after")
    def _reject_presence_that_contradicts_the_score(self) -> "CaseResult":
        """Exists because a documented invariant that is only described can be lied to.

        `is_present` is documented as `score >= scale.presence_threshold`, and a stored run
        claiming a covered zero would otherwise raise `criteria_fulfillment_rate` at
        `/compare` with nothing to catch it. Refused rather than recomputed, because silently
        rewriting a caller's number would hide whichever of the two is actually wrong.
        """
        for criterion_result in self.criterion_results:
            covered = criterion_result.score >= self.scale.presence_threshold
            if criterion_result.is_present != covered:
                raise ValueError(
                    f"criterion {criterion_result.criterion_id} scored "
                    f"{criterion_result.score} and claims "
                    f"is_present={criterion_result.is_present}, "
                    f"which the scale {self.scale} does not"
                )
        return self


class Run(DocumentedModel):
    """The input to `evaluate_run`: several cases evaluated in one go.

    A whole test catalog produces one set of run metrics instead of many isolated scores.

    Example:
        run = Run(
            cases=[
                Case(
                    id=1,
                    question="How do I report sick leave?",
                    answer="Email hr@example.com before 10:00.",
                    criteria=[Criterion(id=1, content="Report by email", weight=3)],
                    labels=["table"],
                ),
                Case(
                    id=2,
                    question="How do I request vacation?",
                    answer="Ask your team lead.",
                    criteria=[Criterion(id=21, content="Use the HR tool", weight=1)],
                ),
            ],
            label_filter=[["table"]],
        )
        [case.id for case in run.selected_cases]   # [1]
    """

    cases: list[Case] = Field(min_length=1)
    """The catalog: at least one case, because an empty run has no meaningful metrics.
    Ids have to be unique — they are what results are matched by."""

    label_filter: LabelFilter = Field(default_factory=list)
    """Which of those cases to actually run, as an OR of ANDs:
    `[["table", "split_infos"], ["agentic"]]` runs the cases carrying both `table` and
    `split_infos`, plus the cases carrying `agentic`. Empty runs all of them, which is the
    normal case.

    Hand this a whole catalog and a label filter rather than pre-filtering: `cases` is what you
    have, `selected_cases` is what runs, and the finished run records the label filter so it
    still says which subset it is months later. A label filter matching no case is refused
    here — before the first judge call, not after a catalog of them."""

    @field_validator("cases")
    @classmethod
    def _reject_duplicate_case_ids(cls, cases: list[Case]) -> list[Case]:
        """Exists for the same reason criterion ids may not repeat.

        `cases_with_score_zero` and the weakest-case shortlist name cases by id, so a
        repeated id makes the run metrics ambiguous.
        """
        _reject_duplicates("case ids", (case.id for case in cases))
        return cases

    @property
    def selected_cases(self) -> list[Case]:
        """The cases this run actually judges: those matching `label_filter`.

        Returns:
            The entries of `cases` covered by the label filter, in their original order —
            all of them when it is empty. Never empty: a label filter matching nothing is
            refused when the run is built.

        Example:
            Run(cases=catalog, label_filter=[["table"]]).selected_cases
        """
        return [
            case
            for case in self.cases
            if matches_label_filter(case.labels, self.label_filter)
        ]

    @model_validator(mode="after")
    def _reject_a_filter_that_matches_no_case(self) -> "Run":
        """Exists so a misspelled label costs nothing to find out about.

        Caught while the run is built, because the alternative is being told only after
        paying for a catalog of judge calls — and because a run of no cases has no metrics
        to report, so there would be nothing to hand back either.
        """
        if not self.selected_cases:
            raise ValueError(
                f"label_filter {self.label_filter} matches no case; labels present in this "
                f"run: {labels_with_case_counts(case.labels for case in self.cases)}"
            )
        return self


class RunMetrics(DocumentedModel):
    """What a whole run is judged by.

    The distribution of the case scores, plus the few numbers that say where to look when a
    run is bad.

    Every case counts once, whatever the size of its rubric — a case with 20 criteria must
    not outweigh nineteen cases with one.

    The documented ranges are enforced rather than described, for the same reason they are on
    the result models above: a stored run is postable to `/compare`, where every one of these
    numbers is subtracted from its counterpart.

    Example:
        metrics = run_metrics([case_result])   # the `case_result` of `CaseResult`
        metrics.total_cases              # 1
        metrics.average_score            # 1.0
        metrics.variance                 # 0.0 — a single case has no spread
        metrics.cases_with_score_zero    # []  — nothing missed its rubric completely
    """

    total_cases: int = Field(ge=1)
    """How many cases the metrics were computed from. At least one: a run of no cases has no
    distribution to describe, which is why `run_metrics` refuses one."""

    average_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    """Arithmetic mean of the case scores — the single number a run is usually reported by."""

    median_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    """Middle case score. Next to the mean it shows skew: far above it means a few
    catastrophic cases drag an otherwise solid run down."""

    variance: float = Field(ge=0, le=_WIDEST_SCORE_VARIANCE, allow_inf_nan=False)
    """Sample variance of the case scores, 0.0 for a single case (which has no spread). Never
    above 0.5, because the case scores it describes are themselves bounded 0..1 — a stored run
    claiming more is describing no distribution of case scores at all."""

    standard_deviation: float = Field(
        ge=0, le=_WIDEST_SCORE_STANDARD_DEVIATION, allow_inf_nan=False
    )
    """Square root of `variance`, in the same unit as the scores. Small means the system is
    uniformly good or bad; large means it depends heavily on the question. Bounded by the
    square root of the same 0.5, for the same reason."""

    average_criterion_score: float = Field(ge=0, allow_inf_nan=False)
    """Mean judge score over *all* criteria of all cases, on the **raw** scale of the run and
    unweighted — 1.4 of 2, not 0.7. Reported raw because the raw grades are what it is
    diagnosing; read it next to the run's `scale`. Unlike `average_score` it ignores both
    weights and case boundaries, so it answers "how well does the judge rate an average
    statement" rather than "how good is the average answer". Its upper bound is the run's
    `scale.maximum` and is enforced by `RunResult`, which is the first place that knows
    the scale."""

    criteria_fulfillment_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    """Mean share of criteria counting as covered (`is_present`) per case, in [0, 1].
    Averaged per case first, so a long rubric does not dominate the rate."""

    cases_with_score_zero: list[int]
    """Ids of the cases that scored exactly 0 — the answers that missed the rubric
    completely. These are the ones to read first."""

    weakest_cases_above_zero: list[int] = Field(max_length=WEAKEST_CASES_REPORTED)
    """Ids of the up to `WEAKEST_CASES_REPORTED` lowest-scoring cases that scored above 0,
    weakest first. Listed apart from `cases_with_score_zero` because a total miss and a
    partial answer usually have different causes. A shortlist and not a ranking, so its length
    is a promise: a longer one read back from JSON is refused rather than truncated."""

    @model_validator(mode="after")
    def _reject_naming_more_cases_than_the_run_holds(self) -> "RunMetrics":
        """Exists because the two id lists are claims, and a stored run can be edited.

        They name disjoint sets of cases — a case scored 0 or it did not — so together they
        can never name more cases than the run has. Unchecked, a stored run could report six
        total misses out of three cases, and `/compare` would subtract that count.
        """
        named = len(self.cases_with_score_zero) + len(self.weakest_cases_above_zero)
        if named > self.total_cases:
            raise ValueError(
                f"metrics over {self.total_cases} cases name {named} of them: "
                f"cases_with_score_zero {self.cases_with_score_zero} and "
                f"weakest_cases_above_zero {self.weakest_cases_above_zero}"
            )
        return self

    @computed_field
    @property
    def cases_with_score_zero_count(self) -> int:
        """How many cases missed their rubric completely.

        Derived rather than stored, so the count and the list it counts can never contradict
        each other — and serialized all the same, so a reader of the JSON gets both.

        Returns:
            The length of `cases_with_score_zero`, 0 when every case scored something.

        Example:
            run_metrics([case_result]).cases_with_score_zero_count   # 0
        """
        return len(self.cases_with_score_zero)


class LabelMetrics(DocumentedModel):
    """One label's slice of a run.

    The same `RunMetrics`, computed over only the cases carrying that label.

    Composed rather than flattened on purpose. A twin that redeclared every `RunMetrics` field
    would have to be edited in step with it forever, and the half that got forgotten would go
    on serializing a stale number under a familiar name.

    A case counts in every bucket it carries a label for, so the buckets overlap and their
    case counts add up to more than the run. That is the question a bucket answers: "how do
    cases involving tables do", not "how do cases that are *only* tables do".

    Example:
        bucket = run_result.label_metrics[0]
        bucket.label                   # "table"
        bucket.metrics.average_score   # 0.0 — next to a run average of 0.5
        bucket.metrics.total_cases     # 1
    """

    label: str
    """The `Case.labels` entry this slice is about."""

    metrics: RunMetrics
    """The run metrics over the cases carrying `label`, and nothing else. Every field means
    exactly what it means on the run as a whole — including `total_cases`, which is how many
    cases carry this label."""


class RunResult(DocumentedModel):
    """What `evaluate_run` returns: the aggregate plus every case result behind it.

    A suspicious number can always be traced back to the cases it was computed from.

    Example:
        run_result = await evaluate_run(judge, run)
        run_result.metrics.average_score                   # 0.5
        run_result.metrics.cases_with_score_zero           # [1] — the answer to read first
        run_result.label_metrics[0].label                  # "table"
        run_result.label_metrics[0].metrics.average_score  # 0.0 — where it really hurts
        run_result.case_results[0].score                   # 0.0
    """

    metrics: RunMetrics
    """The aggregate over all cases of this run."""

    label_metrics: list[LabelMetrics] = Field(default_factory=list)
    """The same aggregate once per label, in alphabetical label order — the breakdown that
    says *which kind* of case a bad average is made of. One entry per label occurring anywhere
    in `case_results` and no others, so a run of untagged cases carries none. See
    `metrics.label_metrics`, which computes this and can be called on stored results too."""

    applied_label_filter: LabelFilter = Field(default_factory=list)
    """The label filter that picked this run's cases, copied from `Run.label_filter` so a
    stored run still says which subset it is. Empty for an unfiltered run. Named apart from
    the input field because it is a *record* and not an instruction: every case result must
    match it, so a run cannot call itself the `table` subset while holding the whole
    catalog."""

    case_results: list[CaseResult] = Field(min_length=1)
    """One result per case of the run, in request order. Each one is exactly what
    `POST /evaluate` returns for that case. At least one, for the same reason `Run.cases`
    needs one: `run_metrics` refuses to describe a distribution over no cases, so a run with
    an empty list could never have been produced legitimately — and anything computed from
    it later, a comparison above all, would be dividing by zero. Case ids have to be unique,
    for the same reason they do in `Run.cases`."""

    @field_validator("case_results")
    @classmethod
    def _reject_duplicate_case_ids(cls, case_results: list[CaseResult]) -> list[CaseResult]:
        """Exists for the stored runs `evaluate_run` never produced.

        `Run.cases` already rejects repeated ids, but a stored run is postable to `/compare`,
        which keys cases by id to pair the two runs up. A repeated id would silently drop a
        case there and report deltas over fewer cases than `metrics` describes, with a 200.
        """
        _reject_duplicates(
            "case ids", (case_result.case_id for case_result in case_results)
        )
        return case_results

    @property
    def scale(self) -> Scale:
        """The scale this whole run was judged on.

        Returns:
            The one `Scale` shared by every case of the run — what `/compare` holds two runs
            against before subtracting anything.

        Raises:
            ValueError: The cases name more than one scale.

        Example:
            str(run_result.scale)   # "0..2 (covered from 0.5)"
        """
        return one_scale_of(case_result.scale for case_result in self.case_results)

    @model_validator(mode="after")
    def _reject_a_mix_of_scales(self) -> "RunResult":
        """Exists because a run graded in two units has no `average_criterion_score`.

        Reading `scale` is the whole check, exactly as in `CaseResult`.
        """
        _ = self.scale
        return self

    @model_validator(mode="after")
    def _reject_metrics_off_the_runs_scale(self) -> "RunResult":
        """Exists because `RunMetrics` cannot check this bound itself.

        It holds numbers, not results, and only here is the scale they were computed on in
        reach. Left unchecked, a stored run could claim an average of 4 on a 0..2 scale and
        every delta computed from it at `/compare` would inherit the lie.
        """
        if self.metrics.average_criterion_score > self.scale.maximum:
            raise ValueError(
                f"average criterion score {self.metrics.average_criterion_score} is off the "
                f"scale {self.scale} this run was judged on"
            )
        return self

    @model_validator(mode="after")
    def _reject_a_filter_that_does_not_describe_the_run(self) -> "RunResult":
        """Exists because `Run` cannot make this check.

        There the field selects, so the run is expected to hold cases it excludes. Here it
        describes what actually ran, which is a claim — and a stored run is the path where a
        claim can have been edited since.
        """
        every_case_must_match(
            self.applied_label_filter,
            {case_result.case_id: case_result.labels for case_result in self.case_results},
        )
        return self

    @model_validator(mode="after")
    def _reject_label_metrics_that_do_not_match_the_cases(self) -> "RunResult":
        """Exists so `/compare` can pair two runs' buckets by label.

        Which labels have a bucket is checkable in a set comparison; whether each bucket's
        numbers are right is not, short of recomputing the whole run. So the structural lie is
        refused — a bucket for a label no case carries, or a labelled run with no breakdown at
        all — and the breakdown covers exactly the labels that are there.
        """
        described = {bucket.label for bucket in self.label_metrics}
        present = {
            label for case_result in self.case_results for label in case_result.labels
        }
        if described != present:
            raise ValueError(
                f"label_metrics describes {sorted(described)} but the cases carry "
                f"{sorted(present)}"
            )
        return self


class ChangeStatus(StrEnum):
    """Which way one score moved between two runs — the vocabulary of every comparison.

    A `StrEnum`, so a stored comparison carries the plain word and a reader of the JSON
    needs no table to decode it.

    Example:
        ChangeStatus.IMPROVED == "improved"   # True
    """

    IMPROVED = "improved"
    """The candidate scored higher than the baseline, by more than `SCORE_EQUALITY_TOLERANCE`."""

    STABLE = "stable"
    """The two scores are equal within `SCORE_EQUALITY_TOLERANCE`."""

    WORSENED = "worsened"
    """The candidate scored lower than the baseline, by more than `SCORE_EQUALITY_TOLERANCE`."""


def _change_status(score_delta: float) -> ChangeStatus:
    """The one place a score delta becomes a status.

    A criterion and a case can never classify the same movement differently this way.

    The tolerance is what keeps float noise out of the report: two runs that summed the same
    weights in a different order can land on 0.7500000000000001 and 0.75, and exact equality
    would file that as a regression.
    """
    if math.isclose(score_delta, 0.0, abs_tol=SCORE_EQUALITY_TOLERANCE):
        return ChangeStatus.STABLE
    return ChangeStatus.IMPROVED if score_delta > 0 else ChangeStatus.WORSENED


class CriterionComparisonResult(DocumentedModel):
    """How one criterion of one case fared between two runs — the finest grain there is.

    Built by `between()`, never by hand, so `score_delta` and `status` cannot disagree.

    Example:
        criterion = Criterion(id=1, content="Report by email before 10:00", weight=3)
        before = CriterionResult.judged(criterion, 0.0, None, DEFAULT_SCALE)
        after = CriterionResult.judged(criterion, 2.0, None, DEFAULT_SCALE)
        CriterionComparisonResult.between(before, after).score_delta   # 2.0
    """

    criterion_id: int
    """The `Criterion.id` both runs judged. Identical in both by construction: a comparison
    of runs with different criterion ids is refused before any of this is computed."""

    weight: float
    """The criterion's weight, identical in both runs. Carried so a movement can be read
    against how much it counted — a 2.0 swing on weight 1 next to nine criteria of weight 3
    barely moves the case score."""

    baseline_score: float
    """What the baseline run's judge gave this criterion, on the run's raw scale."""

    candidate_score: float
    """What the candidate run's judge gave it, same scale."""

    score_delta: float
    """`candidate_score - baseline_score`, so a positive number always means the candidate
    did better. On the run's raw scale, hence in [-maximum, maximum] — both runs were judged
    on the same one, because a comparison of two scales is refused."""

    status: ChangeStatus
    """`score_delta` as a status. Derived from the raw score, not from `is_present`: a
    criterion that went from 1.0 to 2.0 moved the case score and is reported as improved."""

    @classmethod
    def between(
        cls, baseline: CriterionResult, candidate: CriterionResult
    ) -> "CriterionComparisonResult":
        """Compare the two results one criterion got in two runs.

        Args:
            baseline: The result from the run being compared *against*.
            candidate: The result from the run *under test*, for the same criterion id and
                weight — `comparison.compare_runs` has already refused the runs otherwise.

        Returns:
            A `CriterionComparisonResult` whose `score_delta` points from baseline to candidate and
            whose `status` is derived from exactly that delta.

        Example:
            criterion = Criterion(id=1, content="Report by email before 10:00", weight=3)
            before = CriterionResult.judged(criterion, 0.0, None, DEFAULT_SCALE)
            after = CriterionResult.judged(criterion, 2.0, None, DEFAULT_SCALE)
            CriterionComparisonResult.between(before, after).status  # ChangeStatus.IMPROVED
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
        case_comparison = compare_runs(
            RunComparison(baseline=baseline_run, candidate=candidate_run)
        ).case_comparison_results[0]
        case_comparison.score_delta   # 1.0 — the candidate answered better
        case_comparison.criterion_comparison_results[0].score_delta   # 2.0 — which one moved
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
    """`score_delta` as a status, with the same tolerance a criterion gets."""

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
            CaseComparisonResult.between(
                baseline_run.case_results[0], candidate_run.case_results[0]
            ).status   # ChangeStatus.IMPROVED
        """
        score_delta = candidate.score - baseline.score
        return cls(
            case_id=candidate.case_id,
            baseline_score=baseline.score,
            candidate_score=candidate.score,
            score_delta=score_delta,
            status=_change_status(score_delta),
            criterion_comparison_results=[
                CriterionComparisonResult.between(
                    baseline_criterion_result, candidate_criterion_result
                )
                for baseline_criterion_result, candidate_criterion_result in zip(
                    _criterion_results_in_id_order(baseline),
                    _criterion_results_in_id_order(candidate),
                    strict=True,
                )
            ],
        )


def _criterion_results_in_id_order(case_result: CaseResult) -> list[CriterionResult]:
    """Both runs' criterion results in one order, so zipping them pairs the same criterion.

    Rubric order is not enough: two runs may have been stored with their criteria in
    different orders, and zipping those would compare unrelated results. Zipped `strict`, so
    a caller reaching `CaseComparisonResult.between` past `compare_runs` gets a crash rather
    than a silently truncated comparison.
    """
    return sorted(case_result.criterion_results, key=attrgetter("criterion_id"))


class RunMetricsDelta(DocumentedModel):
    """`RunMetrics` of the candidate minus those of the baseline.

    One field per metric that can meaningfully be subtracted.

    Every delta points the same way: **positive means the candidate scored higher**. For
    `cases_with_score_zero_count_delta` that reads backwards on purpose — a positive value
    means the candidate produced *more* total misses.

    `total_cases` has no delta because a comparison of runs with different case sets is
    refused, and the two id lists of `RunMetrics` have none because a set of ids does not
    subtract — `ChangeSummary` reports the movement of cases instead.

    Example:
        metrics_delta = compare_runs(
            RunComparison(baseline=baseline_run, candidate=candidate_run)
        ).metrics_delta
        metrics_delta.average_score_delta                 # 0.5
        metrics_delta.cases_with_score_zero_count_delta   # -1 — one total miss fewer
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
    """Change in the unweighted mean judge score over all criteria, on the runs' shared raw
    scale. Moves independently of `average_score_delta`, because it ignores both weights and
    case boundaries."""

    criteria_fulfillment_rate_delta: float
    """Change in the mean share of criteria counting as covered, in [-1, 1]."""

    cases_with_score_zero_count_delta: int
    """Change in how many cases missed their rubric completely. **Negative is the good
    direction here** — the candidate left fewer answers at zero."""


class LabelMetricsDelta(DocumentedModel):
    """One label's slice of a comparison: `RunMetricsDelta` over only the cases carrying it.

    The mirror of `LabelMetrics`, and composed for the same reason. This is what answers "my
    average went up — but did I fix `agentic_search` or break `table`?"

    Example:
        bucket = compare_runs(
            RunComparison(baseline=baseline_run, candidate=candidate_run)
        ).label_metrics_deltas[0]
        bucket.label                              # "table"
        bucket.metrics_delta.average_score_delta  # 1.0
    """

    label: str
    """The label this slice is about. Present on both runs — a case whose labels changed
    between them makes the comparison incomparable and is refused before this is computed."""

    metrics_delta: RunMetricsDelta
    """Candidate minus baseline over the cases carrying `label`. Every field points the same
    way it does on the run as a whole: positive means the candidate scored higher, except for
    `cases_with_score_zero_count_delta`."""


class ChangeMagnitude(DocumentedModel):
    """How large the moves on one side of a comparison were — improvements or regressions.

    All three numbers carry the sign of their side, so a worsening's `largest` is the most
    negative delta, not its absolute value. Every field is `null` when nothing moved that way:
    a run where nothing got worse has no worsening to report, and a 0.0 there would read as a
    regression of exactly zero to anyone holding the number rather than the id list.

    Example:
        summary = compare_runs(
            RunComparison(baseline=baseline_run, candidate=candidate_run)
        ).summary
        summary.improvement.largest   # 1.0
        summary.worsening.largest     # None — nothing got worse
    """

    largest: float | None
    """The single biggest move on this side, `None` when no case moved this way."""

    mean: float | None
    """Arithmetic mean of the moves on this side — how much a typical one was worth. `None`
    when no case moved this way; there is no move to average."""

    median: float | None
    """Median of the moves, `None` when no case moved this way. Far below the mean on the
    improvement side means one case carries the win."""


class ChangeSummary(DocumentedModel):
    """Where a run moved, case by case: which cases went which way, and by how much.

    The counterpart to `RunMetricsDelta`. That one says the average rose by 0.08; this one
    says whether every case rose a little or three rose a lot while one collapsed.

    Example:
        summary = compare_runs(
            RunComparison(baseline=baseline_run, candidate=candidate_run)
        ).summary
        summary.improved_case_ids   # [1]  — the biggest win first
        summary.stable_case_ids     # [2]
        summary.worsening.largest   # None — nothing got worse
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
    """Size of the moves behind `improved_case_ids`, all positive — all three fields `null`
    when nothing improved."""

    worsening: ChangeMagnitude
    """Size of the moves behind `worsened_case_ids`, all negative — all three fields `null`
    when nothing got worse."""

    @computed_field
    @property
    def improved_case_count(self) -> int:
        """How many cases the candidate scored higher on.

        Derived rather than stored, so the count and the list it counts cannot contradict.

        Returns:
            The length of `improved_case_ids`, 0 when nothing improved.

        Example:
            summary.improved_case_count   # 1
        """
        return len(self.improved_case_ids)

    @computed_field
    @property
    def stable_case_count(self) -> int:
        """How many cases did not move beyond `SCORE_EQUALITY_TOLERANCE`.

        Returns:
            The length of `stable_case_ids`, 0 when every case moved.

        Example:
            summary.stable_case_count   # 1
        """
        return len(self.stable_case_ids)

    @computed_field
    @property
    def worsened_case_count(self) -> int:
        """How many cases the candidate scored lower on.

        Returns:
            The length of `worsened_case_ids`, 0 when nothing got worse — which is a real
            answer and not a missing one.

        Example:
            summary.worsened_case_count   # 0
        """
        return len(self.worsened_case_ids)

    @computed_field
    @property
    def improvement_rate(self) -> float:
        """Share of the run's cases that improved, in [0, 1].

        Every case lands in exactly one of the three lists, so the three *counts* always add
        up to the run exactly. The three rates are three separate divisions, so they add up
        to 1.0 only to within float rounding — six cases split 1 / 4 / 1 sum to
        0.9999999999999999. Compare the counts when an exact total is what you need.

        Returns:
            `improved_case_count` over the run's cases; 0.0 when nothing improved, 1.0 when
            every case did.

        Example:
            summary.improvement_rate   # 0.5
        """
        return self.improved_case_count / self._total_cases

    @computed_field
    @property
    def stability_rate(self) -> float:
        """Share of the run's cases that did not move, in [0, 1].

        Returns:
            `stable_case_count` over the run's cases; 1.0 for a run in which nothing moved
            at all.

        Example:
            summary.stability_rate   # 0.5
        """
        return self.stable_case_count / self._total_cases

    @computed_field
    @property
    def worsening_rate(self) -> float:
        """Share of the run's cases that got worse, in [0, 1].

        Returns:
            `worsened_case_count` over the run's cases; 0.0 when nothing got worse.

        Example:
            summary.worsening_rate   # 0.0
        """
        return self.worsened_case_count / self._total_cases

    @property
    def _total_cases(self) -> int:
        """Every case is in exactly one of the three lists, so they add up to the run.

        No separate total to store and keep in sync, then. Never 0: `RunResult` rejects a run
        with no cases, so the three rates above can always be divided out.
        """
        return self.improved_case_count + self.stable_case_count + self.worsened_case_count


class RunComparison(DocumentedModel):
    """The comparison to make: the two finished runs to hold against each other.

    One named argument rather than two positional ones on purpose. Both sides have the exact
    same type, so a swapped pair would be impossible to detect and would invert the sign of
    every number in the result. Named like every other input here — `Run` produces a
    `RunResult`, a `RunComparison` produces a `RunComparisonResult`.

    Example:
        comparison = RunComparison(baseline=baseline_run, candidate=candidate_run)
        comparison.baseline.metrics.average_score    # 0.5
        comparison.candidate.metrics.average_score   # 1.0
    """

    baseline: RunResult
    """The run being compared *against* — the state of things before your change."""

    candidate: RunResult
    """The run *under test*. Every delta in the result is `candidate - baseline`, so a
    positive number always means this one did better."""


class RunComparisonResult(DocumentedModel):
    """What `compare_runs` returns: the same comparison at three grains.

    `metrics_delta` says whether the run got better, `summary` says how that is distributed
    over the cases, `label_metrics_deltas` says which *kind* of case moved, and
    `case_comparison_results` says which criterion is responsible. A number at any grain can
    always be traced down to the one below it.

    Example:
        result = compare_runs(RunComparison(baseline=baseline_run, candidate=candidate_run))
        result.metrics_delta.average_score_delta   # 0.5
        result.label_metrics_deltas[0].label       # "table"
        result.summary.worsened_case_ids           # []  — the win cost nothing
        result.case_comparison_results[0].criterion_comparison_results[0].score_delta  # 2.0
    """

    metrics_delta: RunMetricsDelta
    """Candidate minus baseline for every `RunMetrics` field that subtracts."""

    summary: ChangeSummary
    """How the movement is distributed over the cases: who won, who lost, by how much."""

    label_metrics_deltas: list[LabelMetricsDelta] = Field(default_factory=list)
    """`metrics_delta` once per label, in alphabetical label order — which *kind* of case
    moved. Empty when neither run carries labels. Both runs always cover the same labels here:
    a case whose labels differ between the two makes them incomparable."""

    case_comparison_results: list[CaseComparisonResult] = Field(min_length=1)
    """One entry per case, ordered by `case_id`. At least one, because both compared runs
    have at least one case. Both runs cover exactly the same cases — that is checked before
    anything here is computed — so neither run's storage order is the canonical one, and
    sorting by id gives a document that does not depend on either."""
