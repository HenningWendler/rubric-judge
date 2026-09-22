"""Every word the judge is told, as plain strings.

The system prompt, the per-criterion user prompt, and the complaints fed back to the judge
when a reply cannot be parsed. Wrapping them into chat messages is `judge.py`'s business —
this module knows nothing about roles, message dicts or the OpenAI SDK, so the complete
wording of an evaluation stays reviewable in one file, by someone who does not read Python.

The one exception is what the grades *mean*: those sentences live on the `Scale`, because a
`CaseResult` carries them and a stored run has to keep saying what its 7 out of 10 was worth.
`judge_prompt` renders them into the instructions, so a judge on any described scale gets a
prompt that agrees with the parser instead of one somebody had to keep in sync by hand.

A Python module rather than a `.txt`, so the prompt cannot go missing from a wheel or a
container image. Callers who want their own wording pass it to
`OpenAIJudge(config, system_prompt=...)`; reading that from a file is their job, not the
judge's.
"""

import textwrap

from rubric_judge.models import DEFAULT_SCALE, Scale

_WRAP_WIDTH = 80
"""Where a level description is broken across lines. The width the bundled prompt was written
at — a generated block has to read like the hand-written one it replaced, or a reviewer
diffing the two drowns in reflowed whitespace."""


def _allowed_scores(scale: Scale) -> str:
    """Exists so no complaint can offer the judge a grade its own parser would then reject."""
    return ", ".join(str(grade) for grade in scale.grades[:-1]) + f" or {scale.maximum}"


WORKED_EXAMPLES_EN = """\
---------------------------
Example:

Context:
The question asked was: How do I request vacation?

Answer:
Send an email to hr@example.com and state the reason for your absence.
Afterwards inform your project team and the project lead about the duration.

Criterion:
- Send an email to hr@example.com

The answer explicitly instructs the reader to send an email to hr@example.com,
which is exactly what the criterion asks for. It is covered literally, with no
deviation or omission.
{"score": 2}

---------------------------
Example:

Context:
The question asked was: How do I request vacation?

Answer:
Send an email to hr@example.com and state the reason for your absence.
Afterwards inform your project team and the project lead about the duration.

Criterion:
- The executive board has to be informed

The answer names the project team and the project lead as the parties to inform,
but never mentions the executive board, neither directly nor by implication.
Nothing in the answer lets the reader derive that requirement.
{"score": 0}

---------------------------
Example:

Context:
The question asked was: How do I report sick leave?

Answer:
If you are ill, inform your line manager without delay about your incapacity to
work and the expected duration.

Criterion:
- Submit the sick note to HR before 10:00

The answer does cover the act of reporting the absence, which is part of the
criterion. However, both the deadline (10:00) and the recipient (HR) are absent,
so the essential specifics are missing while the basic idea remains derivable.
{"score": 1}

---------------------------
Example:

Answer:
The Cologne office has an underground garage. Spots are reserved through the
facility portal, at the latest on the day before.

Criterion:
- Names how a parking spot is reserved

The answer names the facility portal as the place a spot is reserved through,
which is exactly what the criterion asks for. That there is no context here
changes nothing: the requirement is met by the answer alone.
{"score": 2}
"""
"""Four fully worked judgements, appended to the generated instructions by `JUDGE_EN`.

Separate from the generator because they cannot come from a scale: each one is a real answer
and criterion with a real argument, and each closes on a grade of 0, 1 or 2. They therefore
belong to `DEFAULT_SCALE` and to no other — a judge on a ten-point scale shown these would be
shown four wrong answers, which is why `judge_prompt` leaves them out unless they are handed
in.

The fourth carries no context, so a judge meets that shape here rather than for the first
time in production. The first three already cover every grade, 2, 0 and 1, so the fourth
reuses a clear-cut 2 instead of arguing a borderline case: what it has to teach is the
missing `Context:` block, and a debatable grade beside it would only blur that."""


def judge_prompt(scale: Scale, examples: str = "") -> str:
    """Write the judge's system prompt for one scale.

    Everything the judge is told about *grading* comes from `scale`: the header, one line per
    level in its own words, and the reply format down to the list of grades it may answer
    with. Nothing in the result can therefore contradict what `judge.parse_judge_reply`
    accepts.

    Args:
        scale: The scale to instruct the judge on. It has to describe its levels — see Raises.
        examples: Worked judgements appended verbatim after the instructions, or "" for a
            zero-shot prompt. Pass `WORKED_EXAMPLES_EN` only for `DEFAULT_SCALE`: the bundled
            examples close on grades of 0, 1 and 2.

    Returns:
        The complete system prompt, never empty. For `DEFAULT_SCALE` with
        `WORKED_EXAMPLES_EN` that is `JUDGE_EN`, byte for byte.

    Raises:
        ValueError: `scale` describes no levels, so there is nothing to put under the header.
            An undescribed scale is arithmetic only and needs a prompt written by hand.

    Example:
        pass_fail = Scale(
            maximum=1,
            presence_threshold=1,
            level_descriptions={1: "Covered.", 0: "Not covered."},
        )
        "Use this 0-1 scale:" in judge_prompt(pass_fail)   # True
        "1 = Covered." in judge_prompt(pass_fail)          # True
    """
    if not scale.level_descriptions:
        raise ValueError(
            f"the scale {scale} describes no levels, so no prompt can be written from it: "
            "give it a level_descriptions entry per grade, or pass a prompt of your own"
        )
    return _INSTRUCTIONS.format(
        maximum=scale.maximum,
        levels=_level_lines(scale),
        allowed_scores=_allowed_scores(scale),
    ) + examples


def _level_lines(scale: Scale) -> str:
    """Exists so the judge reads the scale top down, best grade first, as a marker would."""
    return "\n".join(
        textwrap.fill(
            f"{grade} = {scale.level_descriptions[grade]}",
            width=_WRAP_WIDTH,
            subsequent_indent="    ",
        )
        for grade in reversed(scale.grades)
    )


_INSTRUCTIONS = """\
You are a careful examiner.

You will be given an answer produced by some system and a single criterion, and
sometimes the context the answer was produced in. Your task is to decide to what
degree the criterion is covered by the answer. When there is no context, judge the
answer on its own terms rather than assuming something was left out.

Use this 0-{maximum} scale:

{levels}

Judge only the criterion you are given. Do not reward correct information that
belongs to a different criterion, and do not punish it either.

First write a short, neutral argument for which score fits the scale above. Only
then decide, and output the decision as a JSON object on its own line. Never use
markdown, never use code fences.

Your reply must always look like this:

[Two or three sentences arguing which score the scale calls for.]
{{"score": {allowed_scores}}}

"""
"""Everything the judge is told that is not a level description and not an example.

The scale header, the level block and the reply format are filled in from the `Scale`, so the
prompt cannot end up describing a scale the parser does not enforce. Kept as one readable
block rather than assembled from fragments: this is the text a reviewer reads."""


JUDGE_EN = judge_prompt(DEFAULT_SCALE, WORKED_EXAMPLES_EN)
"""English judge prompt, used by `OpenAIJudge` for `DEFAULT_SCALE` unless a prompt is passed.

Generated rather than written out, but byte for byte the text it replaced — `tests/
judge_prompt_en.txt` keeps the hand-written original and a test holds this against it, so a
change to the generator or to a level description shows up as a diff rather than as a passing
test.

One thing still has to stay in sync by hand: the closing-JSON instruction matches what
`judge.py` parses, which takes the *last* `{"score": ...}` object in a reply."""


def no_json_hint(scale: Scale) -> str:
    """The complaint for a reply in which nothing looks like a score object.

    Args:
        scale: The scale the judge is working on. Its grades are listed out, so a judge on a
            ten-point scale is never told to answer with 0, 1 or 2.

    Returns:
        The message, phrased as an instruction: the parser raises it as a `ValueError` and
        the retry loop sends that text to the model unchanged. Never empty — an empty
        correction would spend one of the judge's attempts telling it nothing.

    Example:
        no_json_hint(DEFAULT_SCALE)
        # 'Your reply contained no JSON object with a "score" field. End your reply with
        #  exactly one line of the form {"score": 0}, using 0, 1 or 2.'
    """
    return (
        'Your reply contained no JSON object with a "score" field. '
        'End your reply with exactly one line of the form {"score": 0}, '
        f"using {_allowed_scores(scale)}."
    )


def out_of_range_hint(score: float, scale: Scale) -> str:
    """The complaint for a score that is not on the scale — `3`, `-1`, `1.5`.

    Args:
        score: What the judge actually returned. Quoted back to it, because a model
            corrects a number it can see far more reliably than an abstract rule.
        scale: The scale it was supposed to answer on — see `no_json_hint`.

    Returns:
        The message, phrased as an instruction and never empty — see `no_json_hint`.

    Example:
        out_of_range_hint(3.0, DEFAULT_SCALE)
        # 'You returned "score": 3.0, which is not on the scale. Reconsider and answer
        #  with 0, 1 or 2.'
    """
    return (
        f'You returned "score": {score}, which is not on the scale. '
        f"Reconsider and answer with {_allowed_scores(scale)}."
    )


def malformed_json_hint(error: Exception, scale: Scale) -> str:
    """The complaint for a score object that was found but does not parse as valid JSON.

    Args:
        error: The parse failure, included verbatim so the model is told *what* is broken
            (a trailing comma, a single quote) instead of merely that something is.
        scale: The scale it was supposed to answer on — see `no_json_hint`.

    Returns:
        The message, phrased as an instruction and never empty — see `no_json_hint`. It
        stays readable for an `error` whose own text is empty, because the sentence around
        it carries the instruction.

    Example:
        import json

        try:
            json.loads('{"score": 2,}')
        except ValueError as broken:
            malformed_json_hint(broken, DEFAULT_SCALE)
        # 'Your score object could not be parsed (Expecting property name enclosed in
        #  double quotes: line 1 column 13 (char 12)). End your reply with exactly one
        #  line of valid JSON like {"score": 0}, using 0, 1 or 2.'
    """
    return (
        f'Your score object could not be parsed ({error}). '
        f'End your reply with exactly one line of valid JSON like {{"score": 0}}, '
        f"using {_allowed_scores(scale)}."
    )


def criterion_prompt(answer: str, criterion_content: str, context: str | None = None) -> str:
    """Build the user message: what came back, one criterion to judge, and what framed it.

    Args:
        answer: The answer under test, inserted unmodified.
        criterion_content: The text of a single criterion, without the `Criterion` around it.
            Its weight is deliberately *not* passed: a judge that knew how much a criterion
            counts could let that leak into the score.
        context: What the answer was produced in response to, in the caller's own words, or
            `None` for an answer that stands on its own. Background only — the system prompt
            tells the model not to score it.

    Returns:
        The complete user message, never empty. The `Answer:` and `Criterion:` labels are
        always there, even for an empty `answer`; the `Context:` block is absent entirely
        when there is no context, rather than present and blank. One criterion per call is
        the whole design; a model asked about five at once trades attention between them.

    Example:
        print(criterion_prompt("Email hr@example.com.", "Report by email"))
        # Answer:
        # Email hr@example.com.
        #
        # Criterion:
        # - Report by email

        print(criterion_prompt("Email hr@example.com.", "Report by email",
                               "The question asked was: How do I report sick leave?"))
        # Context:
        # The question asked was: How do I report sick leave?
        #
        # Answer:
        # Email hr@example.com.
        #
        # Criterion:
        # - Report by email
    """
    answer_and_criterion = f"Answer:\n{answer}\n\nCriterion:\n- {criterion_content}\n"
    if context is None:
        return answer_and_criterion
    return f"Context:\n{context}\n\n{answer_and_criterion}"
