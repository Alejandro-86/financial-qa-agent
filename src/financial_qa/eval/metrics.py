"""Evaluation metrics for the financial QA pipeline.

Execution accuracy with justified numerical tolerance is the headline metric.
MAPE is a diagnostic for magnitude of error on correct-format predictions.
"""

from dataclasses import dataclass, field


def is_correct(
    predicted: float,
    gold: float,
    tolerance: float = 1e-4,
    check_percentage_scale: bool = True,
) -> bool:
    """Determine whether a predicted value matches the gold answer.

    Applies a small absolute tolerance to handle floating-point rounding.
    Optionally checks whether the prediction is off by a factor of 100
    (percentage stored as decimal vs. whole number).

    Args:
        predicted: The value produced by the pipeline.
        gold: The ground-truth answer.
        tolerance: Absolute tolerance for equality.
        check_percentage_scale: If True, also accept predictions that differ
            from gold by exactly a factor of 100 (e.g. 0.20 vs 20.0).

    Returns:
        True if the prediction is within tolerance of the gold answer.
    """
    if abs(predicted - gold) <= tolerance:
        return True
    if check_percentage_scale and gold != 0:
        if abs(predicted * 100 - gold) <= tolerance:
            return True
        if abs(predicted / 100 - gold) <= tolerance:
            return True
    return False


@dataclass
class AccuracyResult:
    """Result of an execution accuracy computation.

    Attributes:
        correct: Number of predictions within tolerance.
        total: Total predictions evaluated.
        accuracy: Fraction correct (0.0–1.0).
    """

    correct: int
    total: int
    accuracy: float


class ExecutionAccuracy:
    """Computes execution accuracy over a set of (predicted, gold) pairs."""

    @staticmethod
    def compute(
        pairs: list[tuple[float, float]],
        tolerance: float = 1e-4,
    ) -> AccuracyResult:
        """Compute execution accuracy over predicted/gold pairs.

        Args:
            pairs: List of (predicted, gold) float tuples.
            tolerance: Passed through to ``is_correct``.

        Returns:
            AccuracyResult with correct count and accuracy fraction.

        Raises:
            ValueError: If pairs is empty.
        """
        if not pairs:
            raise ValueError("cannot compute accuracy over empty pairs")

        correct = sum(
            1 for pred, gold in pairs if is_correct(pred, gold, tolerance)
        )
        return AccuracyResult(correct=correct, total=len(pairs), accuracy=correct / len(pairs))

    @staticmethod
    def by_turn(
        triples: list[tuple[float, float, int]],
        tolerance: float = 1e-4,
    ) -> dict[int, AccuracyResult]:
        """Compute execution accuracy segmented by turn depth.

        Args:
            triples: List of (predicted, gold, turn) tuples.
            tolerance: Passed through to ``is_correct``.

        Returns:
            Dict mapping turn index to AccuracyResult.
        """
        by_depth: dict[int, list[tuple[float, float]]] = {}
        for pred, gold, turn in triples:
            by_depth.setdefault(turn, []).append((pred, gold))

        return {
            turn: ExecutionAccuracy.compute(pairs, tolerance)
            for turn, pairs in by_depth.items()
        }


class MeanAbsolutePercentageError:
    """MAPE — diagnostic metric for magnitude of error."""

    @staticmethod
    def compute(pairs: list[tuple[float, float]]) -> float:
        """Compute MAPE over predicted/gold pairs, skipping zero-gold entries.

        Args:
            pairs: List of (predicted, gold) float tuples.

        Returns:
            MAPE as a percentage (0–100+).  Returns 0.0 for empty input.
        """
        errors = [
            abs(pred - gold) / abs(gold) * 100
            for pred, gold in pairs
            if gold != 0.0
        ]
        if not errors:
            return 0.0
        return sum(errors) / len(errors)
