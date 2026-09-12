import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.hedging.neural_engine import simulate_neural_hedge
from src.hedging.neural_hedger import NeuralHedger


def train_neural_hedger(
    model: NeuralHedger,
    paths: torch.Tensor,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    transaction_cost_rate: float,
    epochs: int = 25,
    batch_size: int = 2048,
    learning_rate: float = 0.001,
    device: torch.device | str | None = None,
) -> list[float]:
    """
    Train the neural hedger by minimizing mean squared terminal P&L.

    Returns the training RMSE after every epoch.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    model = model.to(device)

    # Keep the full dataset in CPU memory and move only one batch
    # at a time onto the GPU.
    dataset = TensorDataset(
        paths.detach().to(dtype=torch.float32, device="cpu")
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    training_rmse: list[float] = []

    model.train()

    for epoch in range(epochs):
        total_squared_error = 0.0
        total_paths = 0

        for (path_batch,) in loader:
            path_batch = path_batch.to(
                device,
                non_blocking=device.type == "cuda",
            )

            optimizer.zero_grad(set_to_none=True)

            result = simulate_neural_hedge(
                model=model,
                paths=path_batch,
                strike=strike,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                maturity=maturity,
                transaction_cost_rate=transaction_cost_rate,
            )

            loss = torch.mean(result.pnl**2)

            loss.backward()

            # Prevent unusually large gradients from destabilizing training.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            batch_size_actual = path_batch.shape[0]

            total_squared_error += (
                loss.detach().item() * batch_size_actual
            )
            total_paths += batch_size_actual

        epoch_mse = total_squared_error / total_paths
        epoch_rmse = math.sqrt(epoch_mse)

        training_rmse.append(epoch_rmse)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"- Training RMSE: {epoch_rmse:.4f}"
        )

    return training_rmse