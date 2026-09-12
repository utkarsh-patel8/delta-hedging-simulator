import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulation.gbm import generate_gbm_paths
from src.options.payoff import european_call_payoff
from src.pricing.black_scholes import (
    black_scholes_call_delta,
    black_scholes_call_price,
)
from src.simulation.jump_diffusion import (
    generate_jump_diffusion_paths,
)

from src.evaluation.metrics import (
    conditional_value_at_risk,
    root_mean_squared_error,
    value_at_risk,
)

from src.simulation.stress import (
    apply_price_shock,
    generate_regime_switching_paths,
)

from src.hedging.engine import simulate_delta_hedge

from pathlib import Path

from src.visualization.plots import (
    plot_band_comparison,
    plot_frequency_comparison,
    plot_stress_comparison,
)

def main() -> None:
    initial_price = 100.0
    expected_return =  0.08
    volatility = 0.20
    maturity = 1.0
    num_steps = 252
    num_paths = 100000
    risk_free_rate = 0.05

    paths = generate_gbm_paths(
        initial_price = initial_price, expected_return = expected_return, 
        volatility = volatility, maturity = maturity, num_steps = num_steps,
        num_paths = num_paths, seed = 42
    )

    theoretical_mean = (initial_price * np.exp(expected_return*maturity))

    simulated_mean = paths[:, -1].mean()

    strike = 100.0

    final_prices = paths[:, -1]

    call_payoffs = european_call_payoff(
        final_prices=final_prices,
        strike = strike
    )


    positive_payoff_percentage = np.mean(call_payoffs>0) * 100

    call_price = black_scholes_call_price(
        stock_price=initial_price,
        strike=strike,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_maturity=maturity,
    )

    call_delta = black_scholes_call_delta(
        stock_price=initial_price,
        strike=strike,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_maturity=maturity,
    )

    hedging_result = simulate_delta_hedge(
        paths=paths,
        strike=strike,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        maturity=maturity,
    )

    errors = hedging_result.hedging_error

    unhedged_errors = (
        call_price * np.exp(risk_free_rate * maturity)
        - call_payoffs
    )

    risk_reduction = (
        1
        - errors.std() / unhedged_errors.std()
    ) * 100

    print("\n========== STOCK SIMULATION ==========")
    print(f"Paths array shape:       {paths.shape}")
    print(f"Minimum generated price: {paths.min():.2f}")
    print(f"Theoretical final mean:  {theoretical_mean:.2f}")
    print(f"Simulated final mean:    {simulated_mean:.2f}")

    print("\n========== OPTION PAYOFF ==========")
    print(f"Strike price:            {strike:.2f}")
    print(f"Average payoff:          {call_payoffs.mean():.2f}")
    print(f"Maximum payoff:          {call_payoffs.max():.2f}")
    print(f"Positive-payoff paths:   {positive_payoff_percentage:.2f}%")

    print("\n========== BLACK-SCHOLES ==========")
    print(f"Fair option premium:     {call_price:.4f}")
    print(f"Initial call delta:      {call_delta:.4f}")

    print("\n========== DELTA HEDGING ==========")
    print(f"Mean hedging error:       {errors.mean():.4f}")
    print(f"Error standard deviation:{errors.std():9.4f}")
    print(f"Mean absolute error:      {np.abs(errors).mean():.4f}")
    print(f"Worst hedging shortfall:  {errors.min():.4f}")
    print(f"Largest hedging surplus: {errors.max():9.4f}")

    print("\n========== UNHEDGED COMPARISON ==========")
    print(
        f"Unhedged error std:       "
        f"{unhedged_errors.std():.4f}"
    )
    print(
        f"Unhedged worst shortfall: "
        f"{unhedged_errors.min():.4f}"
    )
    print(
        f"Hedged error std:         "
        f"{errors.std():.4f}"
    )
    print(
        f"Risk reduction:           "
        f"{risk_reduction:.2f}%"
    )


    cost_rate = 0.001

    cost_result = simulate_delta_hedge(
        paths=paths,
        strike=strike,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        maturity=maturity,
        transaction_cost_rate=cost_rate,
    )

    
    cost_errors = cost_result.hedging_error

    print("\n========== HEDGING WITH COSTS ==========")
    print(f"Transaction-cost rate:    {cost_rate:.4%}")
    print(
        f"Average turnover:         "
        f"{cost_result.turnover.mean():.4f}"
    )
    print(
        f"Average transaction cost: "
        f"{cost_result.transaction_costs.mean():.4f}"
    )
    print(
        f"Mean hedging P&L:         "
        f"{cost_errors.mean():.4f}"
    )
    print(
        f"P&L standard deviation:  "
        f"{cost_errors.std():.4f}"
    )
    print(
        f"Worst shortfall:          "
        f"{cost_errors.min():.4f}"
    )

    frequencies = {
        "Daily": 1,
        "Weekly": 5,
        "Monthly": 21,
        "Quarterly": 63,
    }

    frequency_results = []

    for name, interval in frequencies.items():
        result = simulate_delta_hedge(
            paths=paths,
            strike=strike,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            maturity=maturity,
            transaction_cost_rate=cost_rate,
            rebalance_every=interval,
        )

        pnl = result.hedging_error

        frequency_results.append(
        {
            "Strategy": name,
            "Mean P&L": pnl.mean(),
            "P&L Std": pnl.std(),
            "MAE": np.abs(pnl).mean(),
            "RMSE": root_mean_squared_error(pnl),
            "VaR 95%": value_at_risk(
                pnl,
                confidence=0.95,
            ),
            "CVaR 95%": conditional_value_at_risk(
                pnl,
                confidence=0.95,
            ),
            "Average Cost": (
                result.transaction_costs.mean()
            ),
            "Average Turnover": result.turnover.mean(),
            "Worst P&L": pnl.min(),
        }
    )

    comparison = pd.DataFrame(frequency_results)

    print("\n========== FREQUENCY COMPARISON ==========")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    bands = [0.0, 0.01, 0.02, 0.05, 0.10]

    band_results = []

    for band in bands:
        result = simulate_delta_hedge(
            paths=paths,
            strike=strike,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            maturity=maturity,
            transaction_cost_rate=cost_rate,
            rebalance_every=1,
            no_trade_band=band,
        )

        pnl = result.hedging_error

        band_results.append(
            {
                "Band": band,
                "Mean P&L": pnl.mean(),
                "P&L Std": pnl.std(),
                "MAE": np.abs(pnl).mean(),
                "RMSE": root_mean_squared_error(pnl),
                "VaR 95%": value_at_risk(pnl, 0.95),
                "CVaR 95%": conditional_value_at_risk(
                    pnl,
                    0.95,
                ),
                "Average Cost": (
                    result.transaction_costs.mean()
                ),
                "Average Turnover": result.turnover.mean(),
            }
        )

    band_comparison = pd.DataFrame(band_results)

    print("\n========== NO-TRADE-BAND COMPARISON ==========")
    print(
        band_comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    regime_paths = generate_regime_switching_paths(
        initial_price=initial_price,
        expected_return=expected_return,
        initial_volatility=0.20,
        stressed_volatility=0.40,
        switch_step=126,
        maturity=maturity,
        num_steps=num_steps,
        num_paths=num_paths,
        seed=42,
    )

    crash_paths = apply_price_shock(
        paths=paths,
        shock_step=126,
        shock_size=-0.20,
    )

    jump_paths = generate_jump_diffusion_paths(
        initial_price=initial_price,
        expected_return=expected_return,
        volatility=0.20,
        jump_intensity=1.5,
        jump_mean=-0.10,
        jump_volatility=0.15,
        maturity=maturity,
        num_steps=num_steps,
        num_paths=num_paths,
        seed=42,
    )

    scenarios = {
        "Baseline": paths,
        "Volatility 20% -> 40%": regime_paths,
        "20% mid-year crash": crash_paths,
        "Jump diffusion": jump_paths,
    }

    stress_results = []

    strategies = {
        "Daily delta": 0.0,
        "No-trade band 0.02": 0.02,
    }

    for scenario_name, scenario_paths in scenarios.items():
        for strategy_name, band in strategies.items():
            result = simulate_delta_hedge(
                paths=scenario_paths,
                strike=strike,
                risk_free_rate=risk_free_rate,

                # The hedger continues assuming 20% volatility.
                volatility=0.20,

                maturity=maturity,
                transaction_cost_rate=cost_rate,
                rebalance_every=1,
                no_trade_band=band,
            )

            pnl = result.hedging_error

            stress_results.append(
                {
                    "Scenario": scenario_name,
                    "Strategy": strategy_name,
                    "Mean P&L": pnl.mean(),
                    "P&L Std": pnl.std(),
                    "RMSE": root_mean_squared_error(pnl),
                    "VaR 95%": value_at_risk(pnl, 0.95),
                    "CVaR 95%": conditional_value_at_risk(
                        pnl,
                        0.95,
                    ),
                    "Average Cost": (
                        result.transaction_costs.mean()
                    ),
                    "Worst P&L": pnl.min(),
                }
            )

    stress_comparison = pd.DataFrame(stress_results)

    print("\n========== STRESS COMPARISON ==========")
    print(
        stress_comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


    results_directory = Path("results")
    results_directory.mkdir(exist_ok=True)

    comparison.to_csv(
        results_directory / "frequency_comparison.csv",
        index=False,
    )

    band_comparison.to_csv(
        results_directory / "band_comparison.csv",
        index=False,
    )

    stress_comparison.to_csv(
        results_directory / "stress_comparison.csv",
        index=False,
    )

    plot_frequency_comparison(
        comparison,
        "results/frequency_comparison.png",
    )

    plot_band_comparison(
        band_comparison,
        "results/band_comparison.png",
    )

    plot_stress_comparison(
        stress_comparison,
        "results/stress_comparison.png",
    )

    print("\nResults and plots saved in the results folder.")

if __name__ == "__main__":
    main()