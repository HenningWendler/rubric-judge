"""Every word the judge is told: the system prompt, the per-criterion user prompt, and the
complaints fed back to it when a reply cannot be parsed.

Strings only. Wrapping them into chat messages is `judge.py`'s business — this module knows
nothing about roles, message dicts or the OpenAI SDK, so the complete wording of an
evaluation stays reviewable in one file, by someone who does not read Python.

A Python module rather than a `.txt`, so the prompt cannot go missing from a wheel or a
container image. Callers who want their own wording pass it to
`OpenAIJudge(config, prompt=...)`; reading that from a file is their job, not the judge's.
"""

from rubric_eval.models import SCALE_MAX

#: "0, 1 or 2", derived from the scale so no complaint can ever contradict SCALE_MAX.
_ALLOWED_SCORES = ", ".join(str(score) for score in range(SCALE_MAX)) + f" or {SCALE_MAX}"

JUDGE_EN = """\
You are a careful examiner.

You will be given a question, an answer produced by some system, and a single
criterion. Your task is to decide to what degree the criterion is covered by the
answer.

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

---------------------------
Example:

Question:
How do I request vacation?

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

Question:
How do I request vacation?

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

Question:
How do I report sick leave?

Answer:
If you are ill, inform your line manager without delay about your incapacity to
work and the expected duration.

Criterion:
- Submit the sick note to HR before 10:00

The answer does cover the act of reporting the absence, which is part of the
criterion. However, both the deadline (10:00) and the recipient (HR) are absent,
so the essential specifics are missing while the basic idea remains derivable.
{"score": 1}
"""
"""English judge prompt, used by `OpenAIJudge` unless a prompt is passed explicitly.

Two things have to stay in sync with the rest of the package: the closing-JSON instruction
matches what `judge.py` parses (it takes the *last* `{"score": ...}` object in the reply),
and the prose `0-2` scale is the one place spelling the scale out by hand — every other
mention is derived from `SCALE_MAX`, so raising the scale means editing this text too."""

NO_JSON_HINT = (
    'Your reply contained no JSON object with a "score" field. '
    'End your reply with exactly one line of the form {"score": 0}, '
    f"using {_ALLOWED_SCORES}."
)
"""Fed back when nothing in the reply looks like a score object."""


def out_of_range_hint(score: float) -> str:
    """The complaint for a score that is not on the scale — `3`, `-1`, `1.5`.

    Args:
        score: What the judge actually returned. Quoted back to it, because a model
            corrects a number it can see far more reliably than an abstract rule.

    Returns:
        The message, phrased as an instruction: the parser raises it as a `ValueError` and
        the retry loop sends that text to the model unchanged.
    """
    return (
        f'You returned "score": {score}, which is not on the scale. '
        f"Reconsider and answer with {_ALLOWED_SCORES}."
    )


def malformed_json_hint(error: Exception) -> str:
    """The complaint for a score object that was found but does not parse as valid JSON.

    Args:
        error: The parse failure, included verbatim so the model is told *what* is broken
            (a trailing comma, a single quote) instead of merely that something is.

    Returns:
        The message, phrased as an instruction — see `out_of_range_hint`.
    """
    return (
        f'Your score object could not be parsed ({error}). '
        f'End your reply with exactly one line of valid JSON like {{"score": 0}}, '
        f"using {_ALLOWED_SCORES}."
    )


def criterion_prompt(question: str, answer: str, criterion: str) -> str:
    """Build the user message: what was asked, what came back, one criterion to judge.

    Args:
        question: Context only — the system prompt tells the model not to score it.
        answer: The answer under test, inserted unmodified.
        criterion: The text of a single criterion. Its weight is deliberately *not* passed:
            a judge that knew how much a criterion counts could let that leak into the score.

    Returns:
        The complete user message. One criterion per call is the whole design — a model
        asked about five at once trades attention between them.
    """
    return f"Question:\n{question}\n\nAnswer:\n{answer}\n\nCriterion:\n- {criterion}\n"
