import numpy as np
import pytest

from src.evaluation.metrics import (
    conditional_value_at_risk,
    root_mean_squared_error,
    value_at_risk,
)


def test_value_at_risk() -> None:
    pnl = np.array([-10, -5, 0, 5, 10])

    result = value_at_risk(
        pnl,
        confidence=0.80,
    )

    assert result == pytest.approx(6.0)


def test_conditional_value_at_risk() -> None:
    pnl = np.array([-10, -5, 0, 5, 10])

    result = conditional_value_at_risk(
        pnl,
        confidence=0.80,
    )

    assert result == pytest.approx(10.0)


def test_rmse() -> None:
    pnl = np.array([-10, -5, 0, 5, 10])

    result = root_mean_squared_error(pnl)

    assert result == pytest.approx(np.sqrt(50))


def test_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        value_at_risk(
            np.array([1, 2, 3]),
            confidence=1.0,
        )