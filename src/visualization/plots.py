from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def plot_frequency_comparison(
    results: pd.DataFrame,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    sns.barplot(
        data=results,
        x="Strategy",
        y="P&L Std",
        ax=axes[0],
        color="steelblue",
    )

    axes[0].set_title("Hedging Risk by Frequency")
    axes[0].set_ylabel("P&L Standard Deviation")

    sns.barplot(
        data=results,
        x="Strategy",
        y="Average Cost",
        ax=axes[1],
        color="darkorange",
    )

    axes[1].set_title("Transaction Cost by Frequency")
    axes[1].set_ylabel("Average Transaction Cost")

    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_band_comparison(
    results: pd.DataFrame,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    sns.lineplot(
        data=results,
        x="Band",
        y="RMSE",
        marker="o",
        label="RMSE",
        ax=axes[0],
    )

    sns.lineplot(
        data=results,
        x="Band",
        y="CVaR 95%",
        marker="o",
        label="CVaR 95%",
        ax=axes[0],
    )

    axes[0].set_title("Risk Versus No-Trade Band")
    axes[0].set_ylabel("Risk")

    sns.lineplot(
        data=results,
        x="Band",
        y="Average Cost",
        marker="o",
        color="darkorange",
        ax=axes[1],
    )

    axes[1].set_title("Cost Versus No-Trade Band")
    axes[1].set_ylabel("Average Transaction Cost")

    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_stress_comparison(
    results: pd.DataFrame,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=results,
        x="Scenario",
        y="CVaR 95%",
        hue="Strategy",
        ax=ax,
    )

    ax.set_title("Tail Risk Under Market Stress")
    ax.set_ylabel("95% CVaR Loss")
    ax.tick_params(
        axis="x",
        rotation=15,
    )

    fig.tight_layout()
    _save_figure(fig, output_path)


def _save_figure(
    figure: plt.Figure,
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)