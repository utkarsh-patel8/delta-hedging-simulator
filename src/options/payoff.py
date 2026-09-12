import numpy as np


def european_call_payoff(
    final_prices: np.ndarray,
    strike: float,
) -> np.ndarray:
    """
    Calculate the payoff of a European call option.

    payoff = max(final stock price - strike, 0)
    """

    if strike <= 0:
        raise ValueError("strike must be positive")

    final_prices = np.asarray(final_prices, dtype=float)

    if np.any(final_prices < 0):
        raise ValueError("stock prices cannot be negative")

    return np.maximum(final_prices - strike, 0.0)