from dataclasses import dataclass

import numpy as np

from src.pricing.black76 import black76_call_delta


@dataclass
class FuturesHedgeResult:
    pnl: np.ndarray
    payoff: np.ndarray
    transaction_costs: np.ndarray
    turnover: np.ndarray


def simulate_futures_delta_hedge(
    paths: np.ndarray,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    option_premium: float,
    transaction_cost_rate: float = 0.0,
) -> FuturesHedgeResult:
    """
    Hedge a short call using futures.

    Futures require no initial purchase payment. Their price changes
    directly produce gains or losses in the cash account.
    """

    paths = np.asarray(paths, dtype=float)

    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("paths must be a two-dimensional array")

    if np.any(paths <= 0):
        raise ValueError("all forward prices must be positive")

    if strike <= 0 or maturity <= 0 or volatility <= 0:
        raise ValueError(
            "strike, maturity and volatility must be positive"
        )

    if option_premium < 0:
        raise ValueError("option_premium cannot be negative")

    if transaction_cost_rate < 0:
        raise ValueError(
            "transaction_cost_rate cannot be negative"
        )

    num_paths, num_columns = paths.shape
    num_steps = num_columns - 1
    time_step = maturity / num_steps

    cash = np.full(num_paths, option_premium, dtype=float)

    hedge = black76_call_delta(
        forward_price=paths[:, 0],
        strike=strike,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_maturity=maturity,
    )

    initial_turnover = np.abs(hedge) * paths[:, 0]
    initial_cost = transaction_cost_rate * initial_turnover

    cash -= initial_cost

    total_turnover = initial_turnover.copy()
    total_cost = initial_cost.copy()

    for step in range(1, num_steps + 1):
        cash *= np.exp(risk_free_rate * time_step)

        futures_gain = hedge * (
            paths[:, step] - paths[:, step - 1]
        )

        cash += futures_gain

        if step < num_steps:
            time_remaining = maturity - step * time_step

            new_hedge = black76_call_delta(
                forward_price=paths[:, step],
                strike=strike,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                time_to_maturity=time_remaining,
            )

            trade_turnover = (
                np.abs(new_hedge - hedge)
                * paths[:, step]
            )

            trade_cost = (
                transaction_cost_rate * trade_turnover
            )

            cash -= trade_cost
            total_turnover += trade_turnover
            total_cost += trade_cost

            hedge = new_hedge

    # Close the remaining futures position at maturity.
    closing_turnover = np.abs(hedge) * paths[:, -1]
    closing_cost = transaction_cost_rate * closing_turnover

    cash -= closing_cost
    total_turnover += closing_turnover
    total_cost += closing_cost

    payoff = np.maximum(paths[:, -1] - strike, 0.0)

    # We sold the call, so its payoff is a liability.
    pnl = cash - payoff

    return FuturesHedgeResult(
        pnl=pnl,
        payoff=payoff,
        transaction_costs=total_cost,
        turnover=total_turnover,
    )