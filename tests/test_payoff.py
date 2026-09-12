import numpy as np
import pytest

from src.options.payoff import european_call_payoff


def test_call_payoff_below_strike() -> None:
    payoff = european_call_payoff(
        final_prices=np.array([80.0]),
        strike=100.0,
    )

    assert payoff[0] == 0.0


def test_call_payoff_at_strike() -> None:
    payoff = european_call_payoff(
        final_prices=np.array([100.0]),
        strike=100.0,
    )

    assert payoff[0] == 0.0


def test_call_payoff_above_strike() -> None:
    payoff = european_call_payoff(
        final_prices=np.array([120.0]),
        strike=100.0,
    )

    assert payoff[0] == 20.0


def test_vectorized_call_payoff() -> None:
    payoff = european_call_payoff(
        final_prices=np.array([80, 100, 120, 150]),
        strike=100,
    )

    expected = np.array([0, 0, 20, 50])

    assert np.array_equal(payoff, expected)


def test_invalid_strike() -> None:
    with pytest.raises(ValueError):
        european_call_payoff(
            final_prices=np.array([100]),
            strike=0,
        )