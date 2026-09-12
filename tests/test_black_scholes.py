import pytest
import numpy as np

from src.pricing.black_scholes import (
    black_scholes_call_delta,
    black_scholes_call_price,
)


def test_known_call_price() -> None:
    price = black_scholes_call_price(
        stock_price=100,
        strike=100,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1,
    )

    assert price == pytest.approx(10.450584, rel=1e-6)




def test_known_call_delta() -> None:
    delta = black_scholes_call_delta(
        stock_price=100,
        strike=100,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1,
    )

    assert delta == pytest.approx(0.636831, rel=1e-6)


def test_call_price_increases_with_stock_price() -> None:
    low = black_scholes_call_price(
        90, 100, 0.05, 0.20, 1
    )
    high = black_scholes_call_price(
        110, 100, 0.05, 0.20, 1
    )

    assert high > low


def test_call_delta_is_between_zero_and_one() -> None:
    delta = black_scholes_call_delta(
        100, 100, 0.05, 0.20, 1
    )

    assert 0 < delta < 1


def test_delta_accepts_array() -> None:
    stock_prices = np.array([80.0, 100.0, 120.0])

    deltas = black_scholes_call_delta(
        stock_price=stock_prices,
        strike=100,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1,
    )

    assert deltas.shape == (3,)
    assert np.all(np.diff(deltas) > 0)