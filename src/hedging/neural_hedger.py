import math

import torch
from torch import nn


def black_scholes_delta_torch(
    stock_price: torch.Tensor,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    time_to_maturity: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable Black-Scholes call delta.
    """

    d1 = (
        torch.log(stock_price / strike)
        + (
            risk_free_rate
            + 0.5 * volatility**2
        ) * time_to_maturity
    ) / (
        volatility * torch.sqrt(time_to_maturity)
    )

    return 0.5 * (
        1.0 + torch.erf(d1 / math.sqrt(2.0))
    )


class NeuralHedger(nn.Module):
    """
    Predict an adjustment to Black-Scholes delta.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        max_adjustment: float = 0.25,
    ) -> None:
        super().__init__()

        self.max_adjustment = max_adjustment

        self.network = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # The initial output adjustment is zero.
        # Therefore, the untrained model starts at BS delta.
        final_layer = self.network[-1]

        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        stock_price: torch.Tensor,
        strike: float,
        time_to_maturity: torch.Tensor,
        total_maturity: float,
        previous_hedge: torch.Tensor,
        bs_delta: torch.Tensor,
    ) -> torch.Tensor:
        log_moneyness = torch.log(
            stock_price / strike
        )

        normalized_time = (
            time_to_maturity / total_maturity
        )

        features = torch.stack(
            [
                log_moneyness,
                normalized_time,
                previous_hedge,
                bs_delta,
            ],
            dim=-1,
        )

        raw_adjustment = (
            self.network(features).squeeze(-1)
        )

        adjustment = (
            self.max_adjustment
            * torch.tanh(raw_adjustment)
        )

        neural_delta = bs_delta + adjustment

        return torch.clamp(
            neural_delta,
            min=0.0,
            max=1.0,
        )