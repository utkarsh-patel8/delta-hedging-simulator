from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.hedging.neural_engine import simulate_neural_hedge
from src.hedging.neural_hedger import NeuralHedger
from src.simulation.gbm import generate_gbm_paths


def calculate_metrics(
    pnl: np.ndarray,
    transaction_costs: np.ndarray,
    turnover: np.ndarray,
) -> dict[str, float]:
    losses = -pnl

    var_95 = np.quantile(losses, 0.95)
    tail_losses = losses[losses >= var_95]
    cvar_95 = tail_losses.mean()

    return {
        "Mean P&L": pnl.mean(),
        "P&L Std": pnl.std(),
        "MAE": np.mean(np.abs(pnl)),
        "RMSE": np.sqrt(np.mean(pnl**2)),
        "VaR 95%": var_95,
        "CVaR 95%": cvar_95,
        "Average Cost": transaction_costs.mean(),
        "Average Turnover": turnover.mean(),
        "Worst P&L": pnl.min(),
    }


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Evaluation device: {device}")

    initial_price = 100.0
    expected_return = 0.08
    volatility = 0.20
    risk_free_rate = 0.05
    strike = 100.0
    maturity = 1.0
    transaction_cost_rate = 0.001

    num_steps = 52
    num_test_paths = 20_000

    # Different seed from training, so these are unseen paths.
    test_paths_numpy = generate_gbm_paths(
        initial_price=initial_price,
        expected_return=expected_return,
        volatility=volatility,
        maturity=maturity,
        num_steps=num_steps,
        num_paths=num_test_paths,
        seed=2026,
    )

    test_paths = torch.tensor(
        test_paths_numpy,
        dtype=torch.float32,
        device=device,
    )

    checkpoint = torch.load(
        "results/neural_hedger.pt",
        map_location=device,
        weights_only=True,
    )

    trained_model = NeuralHedger(
        hidden_size=checkpoint["hidden_size"],
        max_adjustment=checkpoint["max_adjustment"],
    ).to(device)

    trained_model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    trained_model.eval()

    # A new untrained model outputs exactly the Black-Scholes delta,
    # because its neural adjustment initially equals zero.
    black_scholes_model = NeuralHedger(
        hidden_size=checkpoint["hidden_size"],
        max_adjustment=checkpoint["max_adjustment"],
    ).to(device)

    black_scholes_model.eval()

    with torch.no_grad():
        black_scholes_result = simulate_neural_hedge(
            model=black_scholes_model,
            paths=test_paths,
            strike=strike,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            maturity=maturity,
            transaction_cost_rate=transaction_cost_rate,
        )

        neural_result = simulate_neural_hedge(
            model=trained_model,
            paths=test_paths,
            strike=strike,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            maturity=maturity,
            transaction_cost_rate=transaction_cost_rate,
        )

    black_scholes_pnl = (
        black_scholes_result.pnl.cpu().numpy()
    )
    neural_pnl = neural_result.pnl.cpu().numpy()

    black_scholes_metrics = calculate_metrics(
        pnl=black_scholes_pnl,
        transaction_costs=(
            black_scholes_result.transaction_costs.cpu().numpy()
        ),
        turnover=black_scholes_result.turnover.cpu().numpy(),
    )

    neural_metrics = calculate_metrics(
        pnl=neural_pnl,
        transaction_costs=(
            neural_result.transaction_costs.cpu().numpy()
        ),
        turnover=neural_result.turnover.cpu().numpy(),
    )

    comparison = pd.DataFrame(
        [
            {
                "Strategy": "Black-Scholes delta",
                **black_scholes_metrics,
            },
            {
                "Strategy": "Neural hedger",
                **neural_metrics,
            },
        ]
    )

    print()
    print("========== UNSEEN-PATH COMPARISON ==========")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    bs_rmse = black_scholes_metrics["RMSE"]
    neural_rmse = neural_metrics["RMSE"]

    rmse_improvement = (
        100.0 * (bs_rmse - neural_rmse) / bs_rmse
    )

    bs_cvar = black_scholes_metrics["CVaR 95%"]
    neural_cvar = neural_metrics["CVaR 95%"]

    cvar_improvement = (
        100.0 * (bs_cvar - neural_cvar) / bs_cvar
    )

    print()
    print(f"RMSE improvement:         {rmse_improvement:.2f}%")
    print(f"CVaR improvement:         {cvar_improvement:.2f}%")

    results_directory = Path("results")
    results_directory.mkdir(exist_ok=True)

    comparison.to_csv(
        results_directory / "neural_comparison.csv",
        index=False,
    )

    plt.figure(figsize=(9, 5))

    plt.hist(
        black_scholes_pnl,
        bins=80,
        alpha=0.55,
        density=True,
        label="Black-Scholes delta",
    )

    plt.hist(
        neural_pnl,
        bins=80,
        alpha=0.55,
        density=True,
        label="Neural hedger",
    )

    plt.axvline(0.0, color="black", linestyle="--")
    plt.xlabel("Final hedging P&L")
    plt.ylabel("Density")
    plt.title("Out-of-Sample Hedging P&L Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_directory / "neural_pnl_distribution.png",
        dpi=200,
    )
    plt.close()

    risk_comparison = comparison.set_index("Strategy")[
        ["RMSE", "CVaR 95%", "Average Cost"]
    ]

    risk_comparison.plot(
        kind="bar",
        figsize=(9, 5),
        rot=0,
    )

    plt.ylabel("Metric value")
    plt.title("Black-Scholes vs Neural Hedging")
    plt.tight_layout()

    plt.savefig(
        results_directory / "neural_risk_comparison.png",
        dpi=200,
    )
    plt.close()

    print("Comparison results saved in the results folder.")


if __name__ == "__main__":
    main()