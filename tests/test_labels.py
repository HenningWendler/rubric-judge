"""Labels as the axis a run is sliced by: what a valid label is, which cases a filter and a
bucket select, what the breakdown says, and what a relabelled case does to a comparison.

The labels story cuts across every module — model, evaluation, metrics, comparison — so it is
told in one file rather than in four fragments nobody reads together, exactly as the scale
story is in `test_scale.py`. The HTTP end of it lives in `test_api.py`.
"""

import pytest
from pydantic import ValidationError

from tests.conftest import CASE, FakeJudge, run_of
from rubric_eval import (
    Case,
    Criterion,
    LabelMetrics,
    Run,
    RunPair,
    RunResult,
    RunsNotComparableError,
    compare_runs,
    evaluate_run,
    filter_cases_by_labels,
    label_metrics,
)


def case_with(case_id: int, labels: list[str]) -> Case:
    """A valid one-criterion case that exists only to carry labels — every labels test needs
    one and none of them cares about the rubric."""
    return Case(
        id=case_id,
        question="q",
        answer="a",
        criteria=[Criterion(id=case_id, content="x", weight=1)],
        labels=labels,
    )


# --- what a label is ------------------------------------------------------------------------


def test_a_case_needs_no_labels():
    """An untagged catalog is still a catalog: labels are an axis you may add, not a tax."""
    assert Case(**CASE).labels == []


def test_a_case_carries_several_labels():
    assert case_with(1, ["table", "images"]).labels == ["table", "images"]


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_label_is_rejected(blank):
    """A blank tag would open a bucket in the breakdown that names nothing."""
    with pytest.raises(ValidationError):
        case_with(1, [blank])


def test_a_label_is_stripped():
    assert case_with(1, ["  table  "]).labels == ["table"]


def test_duplicate_labels_are_rejected():
    """Labels are read as a set everywhere, so a repeat is a mistake nothing downstream could
    act on — refused rather than silently deduplicated."""
    with pytest.raises(ValidationError, match="labels must be unique"):
        case_with(1, ["table", "table"])


def test_labels_that_differ_only_in_case_are_two_labels():
    """No normalization: silently lowercasing a caller's data would hide which of the two
    spellings is the wrong one, exactly as `is_present` is refused rather than recomputed."""
    assert case_with(1, ["Table", "table"]).labels == ["Table", "table"]


# --- filtering ------------------------------------------------------------------------------


CATALOG = [
    case_with(1, ["table", "split_infos"]),
    case_with(2, ["table"]),
    case_with(3, ["agentic"]),
    case_with(4, ["images"]),
]
"""Four cases covering every shape a selection can ask for: one carrying both labels of a
two-label group, one carrying only part of it, one matching a second group, and one matching
nothing — so a test can tell AND, OR and their combination apart."""


def test_one_group_requires_every_label_in_it():
    assert [case.id for case in filter_cases_by_labels(CATALOG, [["table"]])] == [1, 2]
    assert [
        case.id for case in filter_cases_by_labels(CATALOG, [["table", "split_infos"]])
    ] == [1]


def test_several_groups_are_an_or():
    selected = filter_cases_by_labels(CATALOG, [["table"], ["agentic"]])

    assert [case.id for case in selected] == [1, 2, 3]


def test_an_or_of_ands_is_the_whole_point():
    """`(table AND split_infos) OR agentic` — the combination no flat list could express."""
    selected = filter_cases_by_labels(CATALOG, [["table", "split_infos"], ["agentic"]])

    assert [case.id for case in selected] == [1, 3]


def test_a_case_matching_two_groups_is_selected_once():
    """The groups overlap freely; the result is a selection of cases, not of matches."""
    selected = filter_cases_by_labels(CATALOG, [["table"], ["split_infos"]])

    assert [case.id for case in selected] == [1, 2]


def test_a_case_carrying_more_than_a_group_still_matches():
    """Subset, not equality: the `table` group is every case *involving* a table, which is
    the question the bucket of the same name answers too."""
    assert filter_cases_by_labels([case_with(1, ["table", "images"])], [["table"]]) != []


def test_an_empty_selection_selects_everything():
    """What "no selection" means — the same thing an absent `label_filter` means on a run."""
    assert filter_cases_by_labels(CATALOG, []) == CATALOG


def test_no_match_is_an_empty_list_and_not_an_exception():
    """A filter behaves like a filter, so it composes. What an empty selection *means* is the
    caller's to decide — `Run` is what refuses to run one."""
    assert filter_cases_by_labels(CATALOG, [["tabel"]]) == []


def test_a_flat_list_of_labels_is_refused_rather_than_read_as_characters():
    """`["table"]` is a plausible mistake and `set("table") <= set(labels)` would quietly
    compare *characters*, returning a wrong answer with no error. The message names both
    things the caller might have meant."""
    with pytest.raises(TypeError, match=r"list of label \*groups\*"):
        filter_cases_by_labels(CATALOG, ["table"])


def test_a_selection_is_read_the_same_way_however_its_labels_are_spaced():
    """`filter_cases_by_labels` is documented as the *preview* of what a `Run` would run, so
    the two have to agree on what a label is. A case's labels are stripped on the way in, so a
    selection's have to be too — otherwise a label copied out of a spreadsheet with a trailing
    space previews as "nothing matches" and then runs two cases."""
    padded = [["  table  "]]

    assert [case.id for case in filter_cases_by_labels(CATALOG, padded)] == [
        case.id for case in Run(cases=CATALOG, label_filter=padded).selected_cases
    ] == [1, 2]


def test_a_preview_refuses_the_selections_a_run_refuses():
    """The other half of reading it the same way: a selection a `Run` would never accept has
    no cases it "would pick", so answering one would be inventing a result."""
    with pytest.raises(ValidationError, match="label groups must be unique"):
        filter_cases_by_labels(CATALOG, [["table"], ["table"]])
    with pytest.raises(ValidationError):
        filter_cases_by_labels(CATALOG, [[""]])


def test_an_empty_group_asks_for_nothing_and_so_selects_everything():
    """A group is every label of it, and a group of no labels asks nothing — so every case
    carries all of it. The same reading `[]` gets, one level down."""
    assert filter_cases_by_labels(CATALOG, [[]]) == CATALOG
    assert Run(cases=CATALOG, label_filter=[[]]).selected_cases == CATALOG


def test_an_empty_group_beside_a_real_one_widens_the_selection_to_everything():
    """The OR is what makes it widen rather than narrow: `table OR nothing-required` is every
    case, not the table cases. Worth pinning, because an empty group is what an accidentally
    empty list of labels looks like, and it selects the whole catalog in silence."""
    assert Run(cases=CATALOG, label_filter=[["table"], []]).selected_cases == CATALOG


def test_the_filter_keeps_the_catalog_order():
    shuffled = [case_with(3, ["table"]), case_with(1, ["table"]), case_with(2, ["table"])]

    assert [case.id for case in filter_cases_by_labels(shuffled, [["table"]])] == [3, 1, 2]


def test_duplicate_groups_are_rejected():
    """The same group twice, in any order, asks one question twice and selects nothing extra."""
    with pytest.raises(ValidationError, match="label groups must be unique"):
        Run(cases=CATALOG, label_filter=[["table", "split_infos"], ["split_infos", "table"]])


# --- the breakdown --------------------------------------------------------------------------


def test_every_label_gets_a_bucket_in_alphabetical_order():
    """Sorted so the document does not depend on the order the run happened to be stored in."""
    run = run_of({1: 2}, {2: 2}, labels_by_case_id={1: ["table"], 2: ["agentic_search"]})

    assert [bucket.label for bucket in run.label_metrics] == ["agentic_search", "table"]


def test_a_case_counts_in_every_bucket_it_carries_a_label_for():
    """The buckets overlap on purpose: their case counts add up to more than the run."""
    run = run_of({1: 2}, labels_by_case_id={1: ["table", "images"]})

    assert [bucket.metrics.total_cases for bucket in run.label_metrics] == [1, 1]


def test_a_bucket_reports_the_metrics_of_its_own_cases_only():
    """The point of the whole feature: a run that looks mediocre overall can be fine except
    on one kind of case, and only the breakdown shows it."""
    run = run_of(
        {1: 2}, {2: 2}, {3: 0}, labels_by_case_id={1: ["table"], 2: ["table"], 3: ["images"]}
    )
    buckets = {bucket.label: bucket.metrics for bucket in run.label_metrics}

    assert run.metrics.average_score == pytest.approx(2 / 3)
    assert buckets["table"].average_score == 1.0
    assert buckets["images"].average_score == 0.0


def test_an_unlabelled_case_lands_in_no_bucket():
    """It is counted in the run-wide metrics and nowhere else — no reserved "unlabelled"
    string that could collide with a real label."""
    run = run_of({1: 2}, {2: 0}, labels_by_case_id={1: ["table"]})

    assert [bucket.label for bucket in run.label_metrics] == ["table"]
    assert run.label_metrics[0].metrics.total_cases == 1
    assert run.metrics.total_cases == 2


def test_a_run_of_untagged_cases_has_no_breakdown():
    assert run_of({1: 2}, {2: 0}).label_metrics == []


def test_the_breakdown_can_be_computed_on_stored_results():
    """Public for the same reason `run_metrics` is: a run read back from JSON months later
    can still be sliced."""
    stored = run_of({1: 2}, labels_by_case_id={1: ["table"]}).model_dump()

    buckets = label_metrics(RunResult(**stored).case_results)

    assert [bucket.label for bucket in buckets] == ["table"]


# --- the selection a run runs -------------------------------------------------------------


def test_a_run_runs_only_the_cases_its_selection_covers():
    """Hand it the catalog and the selection: `cases` is what you have, `selected_cases` is
    what runs."""
    run = Run(cases=CATALOG, label_filter=[["table", "split_infos"], ["agentic"]])

    assert [case.id for case in run.cases] == [1, 2, 3, 4]
    assert [case.id for case in run.selected_cases] == [1, 3]


def test_a_run_without_a_selection_runs_everything():
    assert Run(cases=CATALOG).selected_cases == CATALOG


def test_a_run_whose_selection_matches_nothing_is_refused_before_any_judge_call():
    """A run of no cases has no metrics to report, and being told the label was a typo after
    paying for a catalog of judge calls is the outcome this exists to prevent."""
    with pytest.raises(ValidationError, match="matches no case"):
        Run(cases=CATALOG, label_filter=[["tabel"]])


def test_a_selection_that_matches_nothing_names_the_labels_that_do_exist():
    """That is nearly always a typo, and the right spelling is unguessable from "nothing
    matched" alone."""
    carried = r"agentic \(1\), images \(1\), split_infos \(1\), table \(2\)"
    with pytest.raises(ValidationError, match=carried):
        Run(cases=CATALOG, label_filter=[["tabel"]])


def test_a_stored_run_refuses_a_selection_its_results_do_not_match():
    """On a *result* the field is a claim, not an instruction, so here it can lie — and a
    stored run is the path where it can have been edited since. A run cannot call itself the
    `table` subset while holding the whole catalog."""
    stored = run_of({1: 2}, {2: 0}, labels_by_case_id={1: ["table"]}).model_dump()

    with pytest.raises(ValidationError, match=r"cases \[2\] match none of its groups"):
        RunResult(**{**stored, "label_filter": [["table"]]})


def test_a_run_may_hold_cases_its_selection_excludes():
    """The asymmetry with the result above, stated outright: an instruction cannot lie, and
    passing a catalog plus a selection is the entire point."""
    assert Run(cases=CATALOG, label_filter=[["agentic"]]).cases == CATALOG


# --- the breakdown cannot lie ---------------------------------------------------------------


def test_a_run_refuses_a_bucket_for_a_label_no_case_carries():
    """`/compare` pairs the two runs' buckets by label and trusts the breakdown to cover
    exactly the labels that are there."""
    stored = run_of({1: 2}, labels_by_case_id={1: ["table"]})
    invented = LabelMetrics(label="images", metrics=stored.metrics)

    with pytest.raises(ValidationError, match="label_metrics describes"):
        RunResult(**{**stored.model_dump(), "label_metrics": [invented.model_dump()]})


def test_a_labelled_run_refuses_an_empty_breakdown():
    """An empty list on a labelled run would otherwise be ambiguous between "not computed"
    and "wrong"."""
    stored = run_of({1: 2}, labels_by_case_id={1: ["table"]}).model_dump()

    with pytest.raises(ValidationError, match="label_metrics describes"):
        RunResult(**{**stored, "label_metrics": []})


# --- evaluating a labelled run ------------------------------------------------------------


async def test_a_run_covers_the_selected_cases_and_records_what_picked_them():
    run = Run(
        cases=[case_with(1, ["table"]), case_with(2, ["images"])], label_filter=[["table"]]
    )

    run_result = await evaluate_run(FakeJudge({1: 2, 2: 2}), run)

    assert [result.case_id for result in run_result.case_results] == [1]
    assert run_result.case_results[0].labels == ["table"]
    assert run_result.label_filter == [["table"]]
    assert [bucket.label for bucket in run_result.label_metrics] == ["table"]


async def test_an_or_of_ands_narrows_a_whole_catalog_in_one_call():
    run = Run(cases=CATALOG, label_filter=[["table", "split_infos"], ["agentic"]])

    run_result = await evaluate_run(FakeJudge({1: 2, 2: 2, 3: 0, 4: 2}), run)

    assert [result.case_id for result in run_result.case_results] == [1, 3]
    assert run_result.metrics.total_cases == 2


async def test_a_label_never_reaches_the_judge():
    """Labels slice a run; they do not grade an answer. A property that should change the
    grade belongs in a criterion, where it is checkable and weighted."""
    seen = []

    class RecordingJudge(FakeJudge):
        async def score(self, question, answer, criterion):
            seen.append((question, answer, criterion.content))
            return await super().score(question, answer, criterion)

    await evaluate_run(
        RecordingJudge({1: 2}), Run(cases=[case_with(1, ["secret_label"])])
    )

    assert seen == [("q", "a", "x")]


async def test_labelling_a_case_cannot_move_its_score():
    unlabelled = await evaluate_run(FakeJudge({1: 2}), Run(cases=[case_with(1, [])]))
    labelled = await evaluate_run(FakeJudge({1: 2}), Run(cases=[case_with(1, ["table"])]))

    assert unlabelled.case_results[0].score == labelled.case_results[0].score


# --- comparing two labelled runs ------------------------------------------------------------


def test_a_comparison_reports_a_delta_per_label():
    """"My average went up — but did I fix `agentic_search` or break `table`?\""""
    labels = {1: ["table"], 2: ["agentic_search"]}
    baseline = run_of({1: 2}, {2: 0}, labels_by_case_id=labels)
    candidate = run_of({1: 0}, {2: 2}, labels_by_case_id=labels)

    result = compare_runs(RunPair(baseline=baseline, candidate=candidate))

    deltas = {bucket.label: bucket.metrics_delta for bucket in result.label_metrics_deltas}
    assert result.metrics_delta.average_score_delta == pytest.approx(0.0)
    assert deltas["table"].average_score_delta == pytest.approx(-1.0)
    assert deltas["agentic_search"].average_score_delta == pytest.approx(1.0)


def test_the_label_deltas_are_alphabetical():
    labels = {1: ["table"], 2: ["agentic_search"]}
    runs = RunPair(
        baseline=run_of({1: 2}, {2: 0}, labels_by_case_id=labels),
        candidate=run_of({1: 2}, {2: 0}, labels_by_case_id=labels),
    )

    result = compare_runs(runs)

    assert [bucket.label for bucket in result.label_metrics_deltas] == [
        "agentic_search",
        "table",
    ]


def test_two_unlabelled_runs_compare_with_no_label_deltas():
    runs = RunPair(baseline=run_of({1: 2}), candidate=run_of({1: 0}))

    assert compare_runs(runs).label_metrics_deltas == []


def test_a_case_relabelled_between_the_runs_is_refused():
    """The two buckets of the same name would hold different cases, so every delta under them
    would silently compare two different populations."""
    runs = RunPair(
        baseline=run_of({1: 2}, labels_by_case_id={1: ["table"]}),
        candidate=run_of({1: 2}, labels_by_case_id={1: ["images"]}),
    )

    with pytest.raises(RunsNotComparableError, match=r"case 1: labels \['table'\] vs \['images'\]"):
        compare_runs(runs)


def test_reordering_a_case_s_labels_is_not_a_change():
    """Labels are read as a set everywhere, so the same labelling written in another order
    must not refuse a comparison that is perfectly sound."""
    runs = RunPair(
        baseline=run_of({1: 2}, labels_by_case_id={1: ["table", "images"]}),
        candidate=run_of({1: 0}, labels_by_case_id={1: ["images", "table"]}),
    )

    assert compare_runs(runs).metrics_delta.average_score_delta == pytest.approx(-1.0)


def test_runs_recorded_under_different_filters_still_compare():
    """`label_filter` is provenance, not a comparability rule: the case ids are the real check
    and two different filters can legitimately arrive at the same cases."""
    baseline = run_of({1: 2}, labels_by_case_id={1: ["table", "images"]})
    candidate = run_of({1: 0}, labels_by_case_id={1: ["table", "images"]})
    runs = RunPair(
        baseline=RunResult(**{**baseline.model_dump(), "label_filter": [["table"]]}),
        candidate=RunResult(**{**candidate.model_dump(), "label_filter": [["images"]]}),
    )

    assert compare_runs(runs).metrics_delta.average_score_delta == pytest.approx(-1.0)


def test_a_result_written_before_labels_existed_still_loads():
    """`CaseResult.labels` defaults for the same reason `scale` does: a stored run that names
    none simply belongs to no bucket."""
    stored = run_of({1: 2}).model_dump()
    for result in stored["case_results"]:
        del result["labels"]

    assert RunResult(**stored).label_metrics == []
