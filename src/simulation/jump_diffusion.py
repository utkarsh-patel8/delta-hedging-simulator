import numpy as np


def generate_jump_diffusion_paths(
    initial_price: float,
    expected_return: float,
    volatility: float,
    jump_intensity: float,
    jump_mean: float,
    jump_volatility: float,
    maturity: float,
    num_steps: int,
    num_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate stock paths using Merton's jump-diffusion model.

    jump_intensity:
        Expected number of jumps per year.

    jump_mean:
        Mean jump size in log-return terms.

    jump_volatility:
        Uncertainty in jump magnitude.
    """

    if initial_price <= 0:
        raise ValueError("initial_price must be positive")

    if volatility < 0:
        raise ValueError("volatility cannot be negative")

    if jump_intensity < 0:
        raise ValueError(
            "jump_intensity cannot be negative"
        )

    if jump_volatility < 0:
        raise ValueError(
            "jump_volatility cannot be negative"
        )

    if maturity <= 0:
        raise ValueError("maturity must be positive")

    if num_steps <= 0 or num_paths <= 0:
        raise ValueError(
            "num_steps and num_paths must be positive"
        )

    rng = np.random.default_rng(seed)
    dt = maturity / num_steps

    diffusion_shocks = rng.standard_normal(
        size=(num_paths, num_steps)
    )

    # Number of jumps during every interval.
    jump_counts = rng.poisson(
        jump_intensity * dt,
        size=(num_paths, num_steps),
    )

    jump_shocks = rng.standard_normal(
        size=(num_paths, num_steps)
    )

    # Conditional on n jumps, their sum has mean n*jump_mean
    # and standard deviation sqrt(n)*jump_volatility.
    jump_log_returns = (
        jump_counts * jump_mean
        + np.sqrt(jump_counts)
        * jump_volatility
        * jump_shocks
    )

    expected_relative_jump = (
        np.exp(
            jump_mean
            + 0.5 * jump_volatility**2
        )
        - 1
    )

    drift = (
        expected_return
        - 0.5 * volatility**2
        - jump_intensity * expected_relative_jump
    ) * dt

    log_returns = (
        drift
        + volatility
        * np.sqrt(dt)
        * diffusion_shocks
        + jump_log_returns
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