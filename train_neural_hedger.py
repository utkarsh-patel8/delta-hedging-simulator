from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.hedging.neural_hedger import NeuralHedger
from src.hedging.training import train_neural_hedger
from src.simulation.gbm import generate_gbm_paths


def main() -> None:
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")

    initial_price = 100.0
    expected_return = 0.08
    volatility = 0.20
    risk_free_rate = 0.05
    strike = 100.0
    maturity = 1.0
    transaction_cost_rate = 0.001

    # We use 52 weekly rebalancing intervals during neural training.
    # This keeps sequential training fast while remaining realistic.
    num_steps = 52
    num_train_paths = 20_000

    print("Generating training paths...")

    training_paths_numpy = generate_gbm_paths(
        initial_price=initial_price,
        expected_return=expected_return,
        volatility=volatility,
        maturity=maturity,
        num_steps=num_steps,
        num_paths=num_train_paths,
        seed=42,
    )

    training_paths = torch.tensor(
        training_paths_numpy,
        dtype=torch.float32,
    )

    model = NeuralHedger(
        hidden_size=32,
        max_adjustment=0.25,
    )

    training_rmse = train_neural_hedger(
        model=model,
        paths=training_paths,
        strike=strike,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        maturity=maturity,
        transaction_cost_rate=transaction_cost_rate,
        epochs=25,
        batch_size=2048,
        learning_rate=0.001,
        device=device,
    )

    results_directory = Path("results")
    results_directory.mkdir(exist_ok=True)

    model_path = results_directory / "neural_hedger.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_size": 32,
            "max_adjustment": 0.25,
            "num_steps": num_steps,
            "strike": strike,
            "risk_free_rate": risk_free_rate,
            "volatility": volatility,
            "maturity": maturity,
            "transaction_cost_rate": transaction_cost_rate,
        },
        model_path,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(training_rmse) + 1),
        training_rmse,
        marker="o",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training P&L RMSE")
    plt.title("Neural Hedger Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        results_directory / "neural_training_loss.png",
        dpi=200,
    )
    plt.close()

    print()
    print(f"Model saved to: {model_path}")
    print("Training-loss plot saved in the results folder.")


if __name__ == "__main__":
    main()