import numpy as np

def generate_gbm_paths(
        initial_price: float,
        expected_return: float,
        volatility: float,
        maturity: float,
        num_steps: int,
        num_paths: int,
        seed: int | None = None,
) -> np.ndarray:

    # Generate stock price paths using Geometric Brownian Motion
    # Returns an array with shape (num_paths, num_steps+1)
    # Every row is an stock path
    # Every column is one point in time

    if initial_price <= 0:
        raise ValueError("initial_price must be positive")

    if volatility < 0:
        raise ValueError("volatility cannot be negative")

    if maturity <= 0:
        raise ValueError("maturity must be positive")

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    if num_paths <= 0:
        raise ValueError("num_paths must be positive")


    rng = np.random.default_rng(seed)

    # Length of one simulation step, in years
    dt = maturity / num_steps;

    # One independent random shock for every path and time step
    random_shocks = rng.standard_normal(size = (num_paths, num_steps))

    # Continuously compound return during every time step
    log_returns = (
        (expected_return - 0.5*volatility**2) * dt +
        volatility * np.sqrt(dt) * random_shocks
    )

    # Accumulate the stepwise returns through time

    cumulative_logs_return = np.cumsum(log_returns, axis = 1)

    paths = np.empty((num_paths, num_steps+1))

    # Every path begins at the same initial price
    paths[:,0] = initial_price

    paths[:, 1:] = (initial_price * np.exp(cumulative_logs_return))

    return paths