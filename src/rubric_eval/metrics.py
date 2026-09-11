"""Scoring. Failed criteria count as 0 and stay in the denominator — a judge outage
must lower the score visibly, not silently shrink the rubric."""

from rubric_eval.models import SCALE_MAX, CriterionResult


def case_score(results: list[CriterionResult]) -> float:
    """Weighted share of the reachable points, normalized to [0, 1]."""
    if not results:
        raise ValueError("a case score needs at least one criterion")
    weights = _weights_scaled_to_at_most_one(results)
    reached_points = sum(
        weight * result.score / SCALE_MAX for weight, result in zip(weights, results)
    )
    return reached_points / sum(weights)


def _weights_scaled_to_at_most_one(results: list[CriterionResult]) -> list[float]:
    """Only weight *ratios* matter, so dividing every weight by the largest one leaves every
    score untouched — but it keeps both sums inside the float range. Summed raw, two weights
    of 1e308 overflow to `inf`, and `inf / inf` would make the whole case score `nan`.
    """
    largest_weight = max(result.weight for result in results)
    return [result.weight / largest_weight for result in results]
