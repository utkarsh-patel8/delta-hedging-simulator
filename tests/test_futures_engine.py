import numpy as np

from src.hedging.futures_engine import (
    simulate_futures_delta_hedge,
)
from src.pricing.black76 import black76_call_price


def create_paths() -> np.ndarray:
    return np.array(
        [
            [100.0, 102.0, 105.0],
            [100.0, 99.0, 95.0],
            [100.0, 101.0, 100.0],
        ]
    )


def test_futures_engine_output_shapes() -> None:
    paths = create_paths()

    result = simulate_futures_delta_hedge(
        paths=paths,
        strike=100.0,
        risk_free_rate=0.05,
        volatility=0.20,
        maturity=0.25,
        option_premium=5.0,
    )

    assert result.pnl.shape == (3,)
    assert result.payoff.shape == (3,)
    assert result.transaction_costs.shape == (3,)
    assert result.turnover.shape == (3,)


def test_zero_cost_produces_zero_transaction_costs() -> None:
    result = simulate_futures_delta_hedge(
        paths=create_paths(),
        strike=100.0,
        risk_free_rate=0.05,
        volatility=0.20,
        maturity=0.25,
        option_premium=5.0,
        transaction_cost_rate=0.0,
    )

    assert np.allclose(result.transaction_costs, 0.0)


def test_transaction_costs_reduce_pnl() -> None:
    paths = create_paths()

    premium = black76_call_price(
        forward_price=100.0,
        strike=100.0,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=0.25,
    )

    without_costs = simulate_futures_delta_hedge(
        paths=paths,
        strike=100.0,
        risk_free_rate=0.05,
        volatility=0.20,
        maturity=0.25,
        option_premium=premium,
        transaction_cost_rate=0.0,
    )

    with_costs = simulate_futures_delta_hedge(
        paths=paths,
        strike=100.0,
        risk_free_rate=0.05,
        volatility=0.20,
        maturity=0.25,
        option_premium=premium,
        transaction_cost_rate=0.0001,
    )

    assert np.all(with_costs.transaction_costs > 0)
    assert np.all(with_costs.pnl < without_costs.pnl)