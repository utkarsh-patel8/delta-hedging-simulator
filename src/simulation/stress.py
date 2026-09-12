import numpy as np


def generate_regime_switching_paths(
    initial_price: float,
    expected_return: float,
    initial_volatility: float,
    stressed_volatility: float,
    switch_step: int,
    maturity: float,
    num_steps: int,
    num_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate paths whose volatility changes at switch_step.
    """

    if not 0 < switch_step < num_steps:
        raise ValueError(
            "switch_step must lie inside the simulation"
        )

    if initial_volatility < 0:
        raise ValueError(
            "initial_volatility cannot be negative"
        )

    if stressed_volatility < 0:
        raise ValueError(
            "stressed_volatility cannot be negative"
        )

    rng = np.random.default_rng(seed)
    dt = maturity / num_steps

    volatility_schedule = np.full(
        num_steps,
        initial_volatility,
    )

    volatility_schedule[switch_step:] = (
        stressed_volatility
    )

    random_shocks = rng.standard_normal(
        size=(num_paths, num_steps)
    )

    log_returns = (
        (
            expected_return
            - 0.5 * volatility_schedule**2
        ) * dt
        + volatility_schedule
        * np.sqrt(dt)
        * random_shocks
    )

    cumulative_returns = np.cumsum(
        log_returns,
        axis=1,
    )

    paths = np.empty((num_paths, num_steps + 1))
    paths[:, 0] = initial_price

    paths[:, 1:] = (
        initial_price * np.exp(cumulative_returns)
    )

    return paths


def apply_price_shock(
    paths: np.ndarray,
    shock_step: int,
    shock_size: float,
) -> np.ndarray:
    """
    Apply a one-time percentage jump to every path.

    For example, shock_size=-0.20 means a 20% crash.
    """

    paths = np.asarray(paths, dtype=float)

    if paths.ndim != 2:
        raise ValueError("paths must be two-dimensional")

    if not 0 < shock_step < paths.shape[1]:
        raise ValueError(
            "shock_step must lie inside the path"
        )

    if shock_size <= -1:
        raise ValueError(
            "shock_size must be greater than -1"
        )

    stressed_paths = paths.copy()

    stressed_paths[:, shock_step:] *= (
        1 + shock_size
    )

    return stressed_paths