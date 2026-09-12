import numpy as np

from src.pricing.black76 import (
    black76_call_delta,
    black76_call_price,
    black76_implied_volatility,
    implied_forward_from_parity,
)


def test_black76_known_call_price() -> None:
    price = black76_call_price(
        forward_price=100.0,
        strike=100.0,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1.0,
    )

    assert np.isclose(price, 7.5771, atol=1e-4)


def test_implied_volatility_recovery() -> None:
    market_price = black76_call_price(
        forward_price=100.0,
        strike=100.0,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1.0,
    )

    recovered_volatility = black76_implied_volatility(
        market_price=market_price,
        forward_price=100.0,
        strike=100.0,
        risk_free_rate=0.05,
        time_to_maturity=1.0,
    )

    assert np.isclose(
        recovered_volatility,
        0.20,
        atol=1e-6,
    )


def test_put_call_parity_forward() -> None:
    forward = implied_forward_from_parity(
        call_price=170.45,
        put_price=162.15,
        strike=25750.0,
        risk_free_rate=0.05,
        time_to_maturity=6.1875 / 365,
    )

    assert np.isclose(
        forward,
        25758.3070,
        atol=1e-3,
    )


def test_call_delta_is_between_zero_and_one() -> None:
    delta = black76_call_delta(
        forward_price=25758.3070,
        strike=25750.0,
        risk_free_rate=0.05,
        volatility=0.1244,
        time_to_maturity=6.1875 / 365,
    )

    assert 0.0 <= delta <= 1.0


def test_black76_supports_arrays() -> None:
    deltas = black76_call_delta(
        forward_price=np.array([95.0, 100.0, 105.0]),
        strike=100.0,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1.0,
    )

    assert deltas.shape == (3,)
    assert np.all(np.diff(deltas) > 0)