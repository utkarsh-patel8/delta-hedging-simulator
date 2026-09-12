import torch

from src.hedging.neural_engine import simulate_neural_hedge
from src.hedging.neural_hedger import NeuralHedger
from src.simulation.gbm import generate_gbm_paths


def make_paths() -> torch.Tensor:
    paths = generate_gbm_paths(
        initial_price=100.0,
        expected_return=0.08,
        volatility=0.20,
        maturity=1.0,
        num_steps=12,
        num_paths=64,
        seed=42,
    )

    return torch.tensor(paths, dtype=torch.float32)


def test_neural_engine_output_shapes() -> None:
    model = NeuralHedger()
    paths = make_paths()

    result = simulate_neural_hedge(
        model=model,
        paths=paths,
        strike=100.0,
        risk_free_rate=0.05,
        volatility=0.20,
        maturity=1.0,
        transaction_cost_rate=0.001,
    )

    assert result.pnl.shape == (64,)
    assert result.payoff.shape == (64,)
    assert result.transaction_costs.shape == (64,)
    assert torch.all(torch.isfinite(result.pnl))


def test_gradients_flow_through_simulation() -> None:
    model = NeuralHedger()
    paths = make_paths()

    result = simulate_neural_hedge(
        model=model,
        paths=paths,
        strike=100.0,
        risk_free_rate=0.05,
        volatility=0.20,
        maturity=1.0,
        transaction_cost_rate=0.001,
    )

    loss = torch.mean(result.pnl**2)
    loss.backward()

    has_nonzero_gradient = any(
        parameter.grad is not None
        and torch.any(parameter.grad != 0)
        for parameter in model.parameters()
    )

    assert has_nonzero_gradient


def test_transaction_costs_reduce_pnl() -> None:
    model = NeuralHedger()
    paths = make_paths()

    with torch.no_grad():
        without_costs = simulate_neural_hedge(
            model=model,
            paths=paths,
            strike=100.0,
            risk_free_rate=0.05,
            volatility=0.20,
            maturity=1.0,
            transaction_cost_rate=0.0,
        )

        with_costs = simulate_neural_hedge(
            model=model,
            paths=paths,
            strike=100.0,
            risk_free_rate=0.05,
            volatility=0.20,
            maturity=1.0,
            transaction_cost_rate=0.001,
        )

    assert torch.all(with_costs.pnl <= without_costs.pnl)