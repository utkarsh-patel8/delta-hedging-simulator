import numpy as np

from src.hedging.engine import simulate_delta_hedge
from src.simulation.gbm import generate_gbm_paths


def create_sample_paths() -> np.ndarray:
    return generate_gbm_paths(
        initial_price=100,
        expected_return=0.08,
        volatility=0.20,
        maturity=1,
        num_steps=50,
        num_paths=1000,
        seed=42,
    )


def test_hedging_result_shapes() -> None:
    paths = create_sample_paths()

    result = simulate_delta_hedge(
        paths=paths,
        strike=100,
        risk_free_rate=0.05,
        volatility=0.20,
        maturity=1,
    )

    assert result.terminal_portfolio.shape == (1000,)
    assert result.option_payoff.shape == (1000,)
    assert result.hedging_error.shape == (1000,)

    assert np.allclose(
        result.hedging_error,
        result.terminal_portfolio
        - result.option_payoff,
    )


def test_transaction_costs_reduce_portfolio() -> None:
    paths = create_sample_paths()

    no_cost = simulate_delta_hedge(
        paths=paths,
        strike=100,
        risk_free_rate=0.05,
        volatility=0.20,
        maturity=1,
        transaction_cost_rate=0,
    )

    with_cost = simulate_delta_hedge(
        paths=paths,
        strike=100,
        risk_free_rate=0.05,
        volatility=0.20,
        maturity=1,
        transaction_cost_rate=0.001,
    )

    assert np.all(
        with_cost.terminal_portfolio
        <= no_cost.terminal_portfolio
    )

    assert np.all(with_cost.transaction_costs >= 0)

    assert np.allclose(
        with_cost.transaction_costs,
        0.001 * with_cost.turnover,
    )


def test_no_trade_band_reduces_turnover() -> None:
    paths = create_sample_paths()

    always_trade = simulate_delta_hedge(
        paths=paths,
        strike=100,
        risk_free_rate=0.05,
        volatility=0.20,
        maturity=1,
        transaction_cost_rate=0.001,
        no_trade_band=0,
    )

    band_strategy = simulate_delta_hedge(
        paths=paths,
        strike=100,
        risk_free_rate=0.05,
        volatility=0.20,
        maturity=1,
        transaction_cost_rate=0.001,
        no_trade_band=0.05,
    )

    assert (
        band_strategy.turnover.mean()
        < always_trade.turnover.mean()
    )