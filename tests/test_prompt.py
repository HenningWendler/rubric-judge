"""The judge prompt as something a scale builds, not a constant somebody keeps in sync.

`JUDGE_EN` is the generated prompt of `DEFAULT_SCALE`, so the first test here is the one that
matters: it must still come out byte-for-byte as the hand-written text it replaced, which is
kept alongside as `judge_prompt_en.txt` for exactly that purpose.
"""

from pathlib import Path

import pytest

from rubric_eval import DEFAULT_SCALE, Scale
from rubric_eval.prompt import JUDGE_EN, WORKED_EXAMPLES_EN, judge_prompt

GOLDEN = Path(__file__).with_name("judge_prompt_en.txt")
"""The prompt as it was written by hand, before the scale generated it. A plain text file so
a reviewer who does not read Python can still read every word the judge is told — and so any
change to the wording shows up as a diff instead of as a passing test."""

TEN_POINT = Scale(
    maximum=10,
    presence_threshold=5,
    level_descriptions={grade: f"Level {grade} of ten." for grade in range(11)},
)


def test_the_bundled_prompt_is_still_exactly_the_hand_written_one():
    assert JUDGE_EN == GOLDEN.read_text()


def test_the_bundled_prompt_is_what_the_default_scale_generates():
    assert judge_prompt(DEFAULT_SCALE, WORKED_EXAMPLES_EN) == JUDGE_EN


def test_the_scale_block_is_written_from_the_descriptions_highest_first():
    prompt = judge_prompt(DEFAULT_SCALE)
    scale_block = prompt.split("Use this 0-2 scale:\n\n", 1)[1].split("\n\nJudge only", 1)[0]
    assert scale_block.startswith("2 = Fully covered.")
    assert "\n1 = Partially covered." in scale_block
    assert "\n0 = Not covered." in scale_block


def test_a_custom_scale_generates_its_own_header_levels_and_reply_format():
    prompt = judge_prompt(TEN_POINT)
    assert "Use this 0-10 scale:" in prompt
    assert "10 = Level 10 of ten." in prompt
    assert "0 = Level 0 of ten." in prompt
    assert '{"score": 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 or 10}' in prompt


def test_a_long_description_is_wrapped_and_hanging_indented_like_the_bundled_ones():
    """The generated block has to read like the hand-written one, or a reviewer diffing the
    two would drown in reflowed whitespace."""
    wordy = Scale(
        maximum=1,
        presence_threshold=1,
        level_descriptions={1: "Covered. " + "word " * 30, 0: "Not covered."},
    )
    lines = judge_prompt(wordy).split("Use this 0-1 scale:\n\n", 1)[1].splitlines()
    assert lines[0].startswith("1 = Covered.")
    assert lines[1].startswith("    word")
    assert max(len(line) for line in lines[:4]) <= 80


def test_worked_examples_are_left_out_unless_they_are_handed_in():
    """The bundled ones grade 0, 1 and 2, so they belong to the bundled scale and to no
    other — a ten-point judge shown them would be shown three wrong answers."""
    assert "Example:" not in judge_prompt(TEN_POINT)
    assert "Example:" in judge_prompt(TEN_POINT, WORKED_EXAMPLES_EN)


def test_a_scale_that_describes_no_levels_cannot_generate_a_prompt():
    """There would be nothing to put under "Use this 0-2 scale:" — a judge told to pick a
    grade with no meanings attached is guessing."""
    with pytest.raises(ValueError, match="describes no levels"):
        judge_prompt(Scale(maximum=2, presence_threshold=0.5, level_descriptions={}))
