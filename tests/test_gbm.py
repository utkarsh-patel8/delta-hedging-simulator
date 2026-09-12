import numpy as np

from src.simulation.gbm import generate_gbm_paths


def test_output_shape() -> None:
    paths = generate_gbm_paths(
        initial_price=100,
        expected_return=0.08,
        volatility=0.20,
        maturity=1,
        num_steps=252,
        num_paths=1000,
        seed=42,
    )

    assert paths.shape == (1000, 253)


def test_all_paths_start_at_initial_price() -> None:
    paths = generate_gbm_paths(
        initial_price=100,
        expected_return=0.08,
        volatility=0.20,
        maturity=1,
        num_steps=252,
        num_paths=1000,
        seed=42,
    )

    assert np.all(paths[:, 0] == 100)


def test_prices_remain_positive() -> None:
    paths = generate_gbm_paths(
        initial_price=100,
        expected_return=0.08,
        volatility=0.20,
        maturity=1,
        num_steps=252,
        num_paths=1000,
        seed=42,
    )

    assert np.all(paths > 0)


def test_same_seed_produces_same_paths() -> None:
    arguments = {
        "initial_price": 100,
        "expected_return": 0.08,
        "volatility": 0.20,
        "maturity": 1,
        "num_steps": 252,
        "num_paths": 100,
        "seed": 42,
    }

    first_paths = generate_gbm_paths(**arguments)
    second_paths = generate_gbm_paths(**arguments)

    assert np.array_equal(first_paths, second_paths)


def test_zero_volatility_is_deterministic() -> None:
    paths = generate_gbm_paths(
        initial_price=100,
        expected_return=0.08,
        volatility=0,
        maturity=1,
        num_steps=252,
        num_paths=10,
        seed=42,
    )

    expected_final_price = 100 * np.exp(0.08)

    assert np.allclose(
        paths[:, -1],
        expected_final_price,
    )