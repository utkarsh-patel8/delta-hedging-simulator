import numpy as np

from src.simulation.gbm import generate_gbm_paths
from src.simulation.jump_diffusion import (
    generate_jump_diffusion_paths,
)
from src.simulation.stress import (
    apply_price_shock,
    generate_regime_switching_paths,
)


def test_regime_paths_are_valid() -> None:
    paths = generate_regime_switching_paths(
        initial_price=100,
        expected_return=0.08,
        initial_volatility=0.20,
        stressed_volatility=0.40,
        switch_step=25,
        maturity=1,
        num_steps=50,
        num_paths=100,
        seed=42,
    )

    assert paths.shape == (100, 51)
    assert np.all(paths > 0)
    assert np.all(paths[:, 0] == 100)


def test_price_shock_is_applied_correctly() -> None:
    original = np.array(
        [[100.0, 110.0, 120.0, 130.0]]
    )

    stressed = apply_price_shock(
        paths=original,
        shock_step=2,
        shock_size=-0.25,
    )

    expected = np.array(
        [[100.0, 110.0, 90.0, 97.5]]
    )

    assert np.allclose(stressed, expected)


def test_price_shock_does_not_modify_original() -> None:
    original = np.array(
        [[100.0, 110.0, 120.0]]
    )

    original_copy = original.copy()

    apply_price_shock(
        paths=original,
        shock_step=1,
        shock_size=-0.20,
    )

    assert np.array_equal(original, original_copy)


def test_jump_paths_are_valid_and_reproducible() -> None:
    arguments = {
        "initial_price": 100,
        "expected_return": 0.08,
        "volatility": 0.20,
        "jump_intensity": 1.5,
        "jump_mean": -0.10,
        "jump_volatility": 0.15,
        "maturity": 1,
        "num_steps": 50,
        "num_paths": 100,
        "seed": 42,
    }

    first = generate_jump_diffusion_paths(**arguments)
    second = generate_jump_diffusion_paths(**arguments)

    assert first.shape == (100, 51)
    assert np.all(first > 0)
    assert np.array_equal(first, second)


def test_zero_jump_intensity_matches_gbm() -> None:
    gbm_paths = generate_gbm_paths(
        initial_price=100,
        expected_return=0.08,
        volatility=0.20,
        maturity=1,
        num_steps=50,
        num_paths=100,
        seed=42,
    )

    jump_paths = generate_jump_diffusion_paths(
        initial_price=100,
        expected_return=0.08,
        volatility=0.20,
        jump_intensity=0,
        jump_mean=-0.10,
        jump_volatility=0.15,
        maturity=1,
        num_steps=50,
        num_paths=100,
        seed=42,
    )

    assert np.allclose(jump_paths, gbm_paths)