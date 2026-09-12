from dataclasses import dataclass

import numpy as np

from src.options.payoff import european_call_payoff
from src.pricing.black_scholes import (
    black_scholes_call_delta,
    black_scholes_call_price,
)


@dataclass(frozen=True)
class HedgingResult:
    initial_option_price: float
    initial_delta: float
    terminal_portfolio: np.ndarray
    option_payoff: np.ndarray
    hedging_error: np.ndarray
    transaction_costs: np.ndarray
    turnover: np.ndarray


def simulate_delta_hedge(
    paths: np.ndarray,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    transaction_cost_rate: float = 0.0,
    rebalance_every: int = 1,
    no_trade_band: float = 0.0,
) -> HedgingResult:


    if transaction_cost_rate < 0:
        raise ValueError(
            "transaction_cost_rate cannot be negative"
        )

    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive")

    if no_trade_band < 0:
        raise ValueError("no_trade_band cannot be negative")
    
    """
    Simulate daily Black-Scholes delta hedging.

    Transaction costs are not included in this version.
    """

    paths = np.asarray(paths, dtype=float)

    if paths.ndim != 2:
        raise ValueError("paths must be a two-dimensional array")

    if np.any(paths <= 0):
        raise ValueError("all stock prices must be positive")

    num_paths, num_columns = paths.shape
    num_steps = num_columns - 1

    if num_paths == 0 or num_steps <= 0:
        raise ValueError("paths cannot be empty")

    initial_price = float(paths[0, 0])

    if not np.allclose(paths[:, 0], initial_price):
        raise ValueError(
            "all paths must have the same initial price"
        )

    dt = maturity / num_steps
    cash_growth = np.exp(risk_free_rate * dt)

    initial_option_price = black_scholes_call_price(
        stock_price=initial_price,
        strike=strike,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_maturity=maturity,
    )

    initial_delta = black_scholes_call_delta(
        stock_price=initial_price,
        strike=strike,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_maturity=maturity,
    )

    shares = np.full(num_paths, initial_delta)

    initial_turnover = (
        abs(initial_delta) * initial_price
    )

    initial_transaction_cost = (
        transaction_cost_rate * initial_turnover
    )

    cash = np.full(
        num_paths,
        initial_option_price
        - initial_delta * initial_price
        - initial_transaction_cost,
    )

    turnover = np.full(
        num_paths,
        initial_turnover,
    )

    transaction_costs = np.full(
        num_paths,
        initial_transaction_cost,
    )

    # Rebalance at days 1 through num_steps - 1.
    # No rebalance is needed at maturity.
    for step in range(1, num_steps):
        cash *= cash_growth

        if step % rebalance_every != 0:
            continue

        current_prices = paths[:, step]
        time_remaining = maturity - step * dt

        target_delta = black_scholes_call_delta(
            stock_price=current_prices,
            strike=strike,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_maturity=time_remaining,
        )

        desired_trade = target_delta - shares

        shares_bought = np.where(
            np.abs(desired_trade) >= no_trade_band,
            desired_trade,
            0.0
        )

        trade_value = (
            np.abs(shares_bought) * current_prices
        )

        trade_cost = (
            transaction_cost_rate * trade_value
        )

        cash -= (
            shares_bought * current_prices
            + trade_cost
        )

        turnover += trade_value
        transaction_costs += trade_cost

        shares += shares_bought

    # Interest accrues over the final interval.
    cash *= cash_growth

    final_prices = paths[:, -1]

    liquidation_value = (
        np.abs(shares) * final_prices
    )

    liquidation_cost = (
        transaction_cost_rate * liquidation_value
    )

    cash += (
        shares * final_prices
        - liquidation_cost
    )

    turnover += liquidation_value
    transaction_costs += liquidation_cost
    terminal_portfolio = cash

    option_payoff = european_call_payoff(
        final_prices=final_prices,
        strike=strike,
    )

    hedging_error = terminal_portfolio - option_payoff

    return HedgingResult(
        initial_option_price=initial_option_price,
        initial_delta=initial_delta,
        terminal_portfolio=terminal_portfolio,
        option_payoff=option_payoff,
        hedging_error=hedging_error,
        transaction_costs=transaction_costs,
        turnover = turnover,
    )