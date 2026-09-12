import numpy as np
from scipy.stats import norm


def calculate_d1(
    stock_price: float | np.ndarray,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    time_to_maturity: float,
) -> float | np.ndarray:
    stock_price = np.asarray(stock_price, dtype=float)

    if np.any(stock_price <= 0):
        raise ValueError("stock_price must be positive")

    if strike <= 0:
        raise ValueError("strike must be positive")

    if volatility <= 0:
        raise ValueError("volatility must be positive")

    if time_to_maturity <= 0:
        raise ValueError("time_to_maturity must be positive")

    numerator = (
        np.log(stock_price / strike)
        + (
            risk_free_rate
            + 0.5 * volatility**2
        ) * time_to_maturity
    )

    denominator = (
        volatility * np.sqrt(time_to_maturity)
    )

    result = numerator / denominator

    if result.ndim == 0:
        return float(result)

    return result


def black_scholes_call_price(
    stock_price: float | np.ndarray,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    time_to_maturity: float,
) -> float | np.ndarray:
    d1 = calculate_d1(
        stock_price,
        strike,
        risk_free_rate,
        volatility,
        time_to_maturity,
    )

    d2 = (
        d1
        - volatility * np.sqrt(time_to_maturity)
    )

    discounted_strike = (
        strike
        * np.exp(-risk_free_rate * time_to_maturity)
    )

    result = (
        np.asarray(stock_price) * norm.cdf(d1)
        - discounted_strike * norm.cdf(d2)
    )

    if result.ndim == 0:
        return float(result)

    return result


def black_scholes_call_delta(
    stock_price: float | np.ndarray,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    time_to_maturity: float,
) -> float | np.ndarray:
    d1 = calculate_d1(
        stock_price,
        strike,
        risk_free_rate,
        volatility,
        time_to_maturity,
    )

    result = np.asarray(norm.cdf(d1))

    if result.ndim == 0:
        return float(result)

    return result