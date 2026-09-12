import torch

from src.hedging.neural_hedger import (
    NeuralHedger,
    black_scholes_delta_torch,
)


def create_inputs():
    stock_prices = torch.tensor(
        [80.0, 100.0, 120.0]
    )

    time_remaining = torch.tensor(
        [1.0, 1.0, 1.0]
    )

    previous_hedge = torch.zeros(3)

    bs_delta = black_scholes_delta_torch(
        stock_price=stock_prices,
        strike=100,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=time_remaining,
    )

    return (
        stock_prices,
        time_remaining,
        previous_hedge,
        bs_delta,
    )


def test_neural_hedger_output_shape() -> None:
    model = NeuralHedger()

    (
        stock_prices,
        time_remaining,
        previous_hedge,
        bs_delta,
    ) = create_inputs()

    output = model(
        stock_price=stock_prices,
        strike=100,
        time_to_maturity=time_remaining,
        total_maturity=1,
        previous_hedge=previous_hedge,
        bs_delta=bs_delta,
    )

    assert output.shape == (3,)


def test_output_remains_valid_delta() -> None:
    model = NeuralHedger()

    (
        stock_prices,
        time_remaining,
        previous_hedge,
        bs_delta,
    ) = create_inputs()

    output = model(
        stock_price=stock_prices,
        strike=100,
        time_to_maturity=time_remaining,
        total_maturity=1,
        previous_hedge=previous_hedge,
        bs_delta=bs_delta,
    )

    assert torch.all(output >= 0)
    assert torch.all(output <= 1)


def test_untrained_model_matches_bs_delta() -> None:
    model = NeuralHedger()

    (
        stock_prices,
        time_remaining,
        previous_hedge,
        bs_delta,
    ) = create_inputs()

    output = model(
        stock_price=stock_prices,
        strike=100,
        time_to_maturity=time_remaining,
        total_maturity=1,
        previous_hedge=previous_hedge,
        bs_delta=bs_delta,
    )

    assert torch.allclose(output, bs_delta)