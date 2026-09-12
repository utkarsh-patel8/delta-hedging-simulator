import numpy as np


def value_at_risk(
    pnl: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Return VaR as a positive loss amount.
    """

    pnl = np.asarray(pnl, dtype=float)

    if pnl.size == 0:
        raise ValueError("pnl cannot be empty")

    if not 0 < confidence < 1:
        raise ValueError(
            "confidence must be between zero and one"
        )

    losses = -pnl

    return float(np.quantile(losses, confidence))


def conditional_value_at_risk(
    pnl: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Return the average loss in the worst tail.
    """

    pnl = np.asarray(pnl, dtype=float)
    losses = -pnl

    var = value_at_risk(
        pnl=pnl,
        confidence=confidence,
    )

    tail_losses = losses[losses >= var]

    return float(tail_losses.mean())


def root_mean_squared_error(
    pnl: np.ndarray,
) -> float:
    pnl = np.asarray(pnl, dtype=float)

    if pnl.size == 0:
        raise ValueError("pnl cannot be empty")

    return float(np.sqrt(np.mean(pnl**2)))