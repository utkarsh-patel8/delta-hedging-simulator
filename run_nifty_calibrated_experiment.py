from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.market_data import load_nifty_minute_data
from src.hedging.futures_engine import (
    simulate_futures_delta_hedge,
)
from src.pricing.black76 import (
    black76_call_delta,
    black76_call_price,
    black76_implied_volatility,
    implied_forward_from_parity,
)
from src.simulation.gbm import generate_gbm_paths


def calculate_metrics(pnl: np.ndarray) -> dict[str, float]:
    losses = -pnl

    var_95 = np.quantile(losses, 0.95)
    cvar_95 = losses[losses >= var_95].mean()

    return {
        "Mean P&L": pnl.mean(),
        "P&L Std": pnl.std(),
        "MAE": np.abs(pnl).mean(),
        "RMSE": np.sqrt(np.mean(pnl**2)),
        "VaR 95%": var_95,
        "CVaR 95%": cvar_95,
        "Worst P&L": pnl.min(),
    }


def main() -> None:
    data_path = Path(
        "data/raw/"
        "20260204_option_minute_prices_non_expiry.csv"
    )

    results_folder = Path("results")
    results_folder.mkdir(exist_ok=True)

    market_data = load_nifty_minute_data(data_path)

    selected_time = pd.Timestamp("2026-02-04 11:00:00")
    selected_strike = 25750.0
    risk_free_rate = 0.05

    snapshot = market_data[
        market_data["timestamp"] == selected_time
    ]

    options = snapshot[
        snapshot["instrument_type"] == "option"
    ]

    # Match every call with the put having the same strike.
    option_prices = options.pivot(
        index="strike",
        columns="option_type",
        values="price",
    ).dropna()

    expiry = options["expiry"].iloc[0]

    maturity = (
        expiry - selected_time
    ).total_seconds() / (365 * 24 * 60 * 60)

    # Calculate one parity-implied forward for every strike.
    parity_forwards = [
        implied_forward_from_parity(
            call_price=row["call"],
            put_price=row["put"],
            strike=strike,
            risk_free_rate=risk_free_rate,
            time_to_maturity=maturity,
        )
        for strike, row in option_prices.iterrows()
    ]

    # Median is less sensitive to one stale option price.
    initial_forward = float(np.median(parity_forwards))

    selected_call = options[
        (options["strike"] == selected_strike)
        & (options["option_type"] == "call")
    ]

    if selected_call.empty:
        raise ValueError("Selected call was not found")

    market_premium = float(
        selected_call["price"].iloc[0]
    )

    implied_volatility = black76_implied_volatility(
        market_price=market_premium,
        forward_price=initial_forward,
        strike=selected_strike,
        risk_free_rate=risk_free_rate,
        time_to_maturity=maturity,
    )

    model_premium = black76_call_price(
        forward_price=initial_forward,
        strike=selected_strike,
        risk_free_rate=risk_free_rate,
        volatility=implied_volatility,
        time_to_maturity=maturity,
    )

    initial_delta = black76_call_delta(
        forward_price=initial_forward,
        strike=selected_strike,
        risk_free_rate=risk_free_rate,
        volatility=implied_volatility,
        time_to_maturity=maturity,
    )

    num_paths = 100_000
    num_steps = 60
    transaction_cost_rate = 0.0001

    # A forward price is a martingale under risk-neutral pricing,
    # so its expected return is set to zero.
    paths = generate_gbm_paths(
        initial_price=initial_forward,
        expected_return=0.0,
        volatility=implied_volatility,
        maturity=maturity,
        num_steps=num_steps,
        num_paths=num_paths,
        seed=2026,
    )

    no_cost_result = simulate_futures_delta_hedge(
        paths=paths,
        strike=selected_strike,
        risk_free_rate=risk_free_rate,
        volatility=implied_volatility,
        maturity=maturity,
        option_premium=market_premium,
        transaction_cost_rate=0.0,
    )

    cost_result = simulate_futures_delta_hedge(
        paths=paths,
        strike=selected_strike,
        risk_free_rate=risk_free_rate,
        volatility=implied_volatility,
        maturity=maturity,
        option_premium=market_premium,
        transaction_cost_rate=transaction_cost_rate,
    )

    unhedged_pnl = (
        market_premium * np.exp(risk_free_rate * maturity)
        - np.maximum(paths[:, -1] - selected_strike, 0.0)
    )

    summary = pd.DataFrame(
        [
            {
                "Strategy": "Unhedged short call",
                **calculate_metrics(unhedged_pnl),
                "Average Cost": 0.0,
            },
            {
                "Strategy": "Black-76 delta",
                **calculate_metrics(no_cost_result.pnl),
                "Average Cost": 0.0,
            },
            {
                "Strategy": "Black-76 delta with costs",
                **calculate_metrics(cost_result.pnl),
                "Average Cost": (
                    cost_result.transaction_costs.mean()
                ),
            },
        ]
    )

    print()
    print("========== NIFTY MARKET CALIBRATION ==========")
    print(f"Snapshot time:           {selected_time}")
    print(f"Option expiry:           {expiry}")
    print(f"Strike:                  {selected_strike:.2f}")
    print(f"Market call premium:     {market_premium:.2f}")
    print(f"Parity forward:          {initial_forward:.2f}")
    print(f"Remaining maturity:      {maturity * 365:.4f} days")
    print(f"Implied volatility:      {implied_volatility:.2%}")
    print(f"Black-76 model premium:  {model_premium:.2f}")
    print(f"Initial call delta:      {initial_delta:.4f}")

    print()
    print("========== NIFTY-CALIBRATED SIMULATION ==========")
    print(summary.round(4).to_string(index=False))

    summary.to_csv(
        results_folder / "nifty_calibrated_summary.csv",
        index=False,
    )

    plt.figure(figsize=(10, 6))

    for path in paths[:30]:
        plt.plot(path, alpha=0.6)

    plt.axhline(
        selected_strike,
        color="black",
        linestyle="--",
        label="Strike",
    )

    plt.title("Market-Calibrated NIFTY Forward Paths")
    plt.xlabel("Simulation step")
    plt.ylabel("NIFTY forward level")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_folder / "nifty_calibrated_paths.png",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(10, 6))

    plt.hist(
        unhedged_pnl,
        bins=100,
        alpha=0.55,
        label="Unhedged",
        density=True,
    )

    plt.hist(
        cost_result.pnl,
        bins=100,
        alpha=0.65,
        label="Delta hedged with costs",
        density=True,
    )

    plt.title("NIFTY Call: Hedged vs Unhedged P&L")
    plt.xlabel("Final P&L")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_folder / "nifty_calibrated_pnl.png",
        dpi=200,
    )
    plt.close()

    print()
    print("Results saved in the results folder.")


if __name__ == "__main__":
    main()