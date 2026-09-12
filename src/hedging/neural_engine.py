from dataclasses import dataclass

import torch

from src.hedging.neural_hedger import (
    NeuralHedger,
    black_scholes_delta_torch,
)
from src.pricing.black_scholes import black_scholes_call_price


@dataclass
class NeuralHedgingResult:
    pnl: torch.Tensor
    payoff: torch.Tensor
    terminal_wealth: torch.Tensor
    transaction_costs: torch.Tensor
    turnover: torch.Tensor


def simulate_neural_hedge(
    model: NeuralHedger,
    paths: torch.Tensor,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    transaction_cost_rate: float = 0.0,
) -> NeuralHedgingResult:
    """
    Simulate a self-financing neural-network hedging strategy.

    The option seller:
    1. Receives the option premium.
    2. Uses the neural network to choose a stock position.
    3. Rebalances that position through time.
    4. Pays transaction costs for every trade.
    5. Liquidates the stock and pays the option payoff at maturity.
    """

    if paths.ndim != 2:
        raise ValueError("paths must have shape (num_paths, num_steps + 1)")

    if paths.shape[1] < 2:
        raise ValueError("paths must contain at least two time points")

    if strike <= 0:
        raise ValueError("strike must be positive")

    if volatility <= 0:
        raise ValueError("volatility must be positive")

    if maturity <= 0:
        raise ValueError("maturity must be positive")

    if transaction_cost_rate < 0:
        raise ValueError("transaction_cost_rate cannot be negative")

    if torch.any(paths <= 0):
        raise ValueError("all stock prices must be positive")

    # Ensure paths and neural-network parameters use the same device and dtype.
    model_parameter = next(model.parameters())

    paths = paths.to(
        device=model_parameter.device,
        dtype=model_parameter.dtype,
    )

    num_paths, num_columns = paths.shape
    num_steps = num_columns - 1

    dt = maturity / num_steps
    growth_factor = torch.exp(
        torch.tensor(
            risk_free_rate * dt,
            device=paths.device,
            dtype=paths.dtype,
        )
    )

    initial_price = float(paths[0, 0].detach().cpu())

    option_premium = black_scholes_call_price(
        stock_price=initial_price,
        strike=strike,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_maturity=maturity,
    )

    cash = torch.full(
        (num_paths,),
        float(option_premium),
        device=paths.device,
        dtype=paths.dtype,
    )

    shares = torch.zeros_like(cash)
    transaction_costs = torch.zeros_like(cash)
    turnover = torch.zeros_like(cash)

    for step in range(num_steps):
        # Cash earns interest since the previous time step.
        if step > 0:
            cash = cash * growth_factor

        current_prices = paths[:, step]
        time_remaining_value = maturity - step * dt

        time_remaining = torch.full_like(
            current_prices,
            time_remaining_value,
        )

        bs_delta = black_scholes_delta_torch(
            stock_price=current_prices,
            strike=strike,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_maturity=time_remaining,
        )

        target_shares = model(
            stock_price=current_prices,
            strike=strike,
            time_to_maturity=time_remaining,
            total_maturity=maturity,
            previous_hedge=shares,
            bs_delta=bs_delta,
        )

        shares_traded = target_shares - shares
        traded_value = torch.abs(shares_traded) * current_prices
        trading_cost = transaction_cost_rate * traded_value

        # Buying shares reduces cash; selling shares increases cash.
        cash = cash - shares_traded * current_prices - trading_cost

        turnover = turnover + traded_value
        transaction_costs = transaction_costs + trading_cost
        shares = target_shares

    # Cash earns interest over the final interval.
    cash = cash * growth_factor

    final_prices = paths[:, -1]

    # Sell all remaining shares at maturity.
    liquidation_value = shares * final_prices
    liquidation_turnover = torch.abs(shares) * final_prices
    liquidation_cost = transaction_cost_rate * liquidation_turnover

    cash = cash + liquidation_value - liquidation_cost

    turnover = turnover + liquidation_turnover
    transaction_costs = transaction_costs + liquidation_cost

    payoff = torch.relu(final_prices - strike)

    terminal_wealth = cash
    pnl = terminal_wealth - payoff

    return NeuralHedgingResult(
        pnl=pnl,
        payoff=payoff,
        terminal_wealth=terminal_wealth,
        transaction_costs=transaction_costs,
        turnover=turnover,
    )