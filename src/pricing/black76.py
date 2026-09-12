import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _return_correct_type(
    result: np.ndarray,
    original_input: float | np.ndarray,
) -> float | np.ndarray:
    if np.asarray(original_input).ndim == 0:
        return float(result)

    return result


def black76_call_price(
    forward_price: float | np.ndarray,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    time_to_maturity: float,
) -> float | np.ndarray:
    """
    Calculate the Black-76 price of a European call option.
    """

    forward = np.asarray(forward_price, dtype=float)

    if np.any(forward <= 0):
        raise ValueError("forward_price must be positive")

    if strike <= 0:
        raise ValueError("strike must be positive")

    if volatility <= 0:
        raise ValueError("volatility must be positive")

    if time_to_maturity < 0:
        raise ValueError("time_to_maturity cannot be negative")

    if time_to_maturity == 0:
        result = np.maximum(forward - strike, 0.0)
        return _return_correct_type(result, forward_price)

    square_root_time = np.sqrt(time_to_maturity)

    d1 = (
        np.log(forward / strike)
        + 0.5 * volatility**2 * time_to_maturity
    ) / (volatility * square_root_time)

    d2 = d1 - volatility * square_root_time

    discount_factor = np.exp(
        -risk_free_rate * time_to_maturity
    )

    result = discount_factor * (
        forward * norm.cdf(d1)
        - strike * norm.cdf(d2)
    )

    return _return_correct_type(result, forward_price)


def black76_call_delta(
    forward_price: float | np.ndarray,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    time_to_maturity: float,
) -> float | np.ndarray:
    """
    Calculate call-option sensitivity to the forward price.
    """

    forward = np.asarray(forward_price, dtype=float)

    if np.any(forward <= 0):
        raise ValueError("forward_price must be positive")

    if strike <= 0:
        raise ValueError("strike must be positive")

    if volatility <= 0:
        raise ValueError("volatility must be positive")

    if time_to_maturity < 0:
        raise ValueError("time_to_maturity cannot be negative")

    if time_to_maturity == 0:
        result = np.where(
            forward > strike,
            1.0,
            np.where(forward < strike, 0.0, 0.5),
        )

        return _return_correct_type(result, forward_price)

    d1 = (
        np.log(forward / strike)
        + 0.5 * volatility**2 * time_to_maturity
    ) / (
        volatility * np.sqrt(time_to_maturity)
    )

    discount_factor = np.exp(
        -risk_free_rate * time_to_maturity
    )

    result = discount_factor * norm.cdf(d1)

    return _return_correct_type(result, forward_price)


def implied_forward_from_parity(
    call_price: float,
    put_price: float,
    strike: float,
    risk_free_rate: float,
    time_to_maturity: float,
) -> float:
    """
    Infer the option-expiry forward using put-call parity.

    C - P = exp(-rT) * (F - K)
    """

    if call_price < 0 or put_price < 0:
        raise ValueError("option prices cannot be negative")

    if strike <= 0:
        raise ValueError("strike must be positive")

    if time_to_maturity <= 0:
        raise ValueError("time_to_maturity must be positive")

    return strike + np.exp(
        risk_free_rate * time_to_maturity
    ) * (call_price - put_price)


def black76_implied_volatility(
    market_price: float,
    forward_price: float,
    strike: float,
    risk_free_rate: float,
    time_to_maturity: float,
) -> float:
    """
    Find the volatility that reproduces the observed call price.
    """

    if market_price <= 0:
        raise ValueError("market_price must be positive")

    if forward_price <= 0 or strike <= 0:
        raise ValueError(
            "forward_price and strike must be positive"
        )

    if time_to_maturity <= 0:
        raise ValueError("time_to_maturity must be positive")

    discount_factor = np.exp(
        -risk_free_rate * time_to_maturity
    )

    minimum_price = discount_factor * max(
        forward_price - strike,
        0.0,
    )

    maximum_price = discount_factor * forward_price

    if not minimum_price <= market_price < maximum_price:
        raise ValueError(
            "Market price violates Black-76 price bounds"
        )

    def pricing_error(volatility: float) -> float:
        model_price = black76_call_price(
            forward_price=forward_price,
            strike=strike,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_maturity=time_to_maturity,
        )

        return model_price - market_price

    return brentq(
        pricing_error,
        1e-6,
        5.0,
    )