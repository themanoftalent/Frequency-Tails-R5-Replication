"""Plotting functions for empirical results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def set_style():
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "figure.dpi": 150, "savefig.dpi": 250})


def plot_vix_stress(data: pd.DataFrame, path: Path, threshold=30):
    set_style()
    series = data["vix_level"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(series.index, series, lw=.9, label="VIX level")
    ax.axhline(threshold, ls=":", lw=1, label=f"Threshold {threshold}")
    ax.fill_between(series.index, 0, series, where=series > threshold, alpha=.18)
    ax.set(ylabel="VIX level", xlabel="Date", title="VIX levels and stress episodes")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_dcc(paths: pd.DataFrame, path: Path):
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for column in paths:
        ax.plot(paths.index, paths[column], label=column.replace("__", " vs "), lw=.8)
    ax.axhline(0, color="black", lw=.6)
    ax.set(ylabel="Conditional correlation", xlabel="Date",
           title="Dynamic conditional correlations (DCC-GARCH)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_connectedness(matrix: pd.DataFrame, path: Path):
    set_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix.values, aspect="equal", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=55, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, f"{matrix.iloc[i,j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if matrix.iloc[i,j] > 60 else "black")
    ax.set_title("Forecast error variance decomposition (%)")
    fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_quantile_coefficients(table: pd.DataFrame, path: Path, term="vix"):
    set_style()
    data = table.loc[table.term == term].sort_values("tau")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(data.tau, data.coefficient, marker="o", lw=1)
    ax.fill_between(data.tau.to_numpy(dtype=float), data.ci_low.to_numpy(dtype=float),
                    data.ci_high.to_numpy(dtype=float), alpha=.18)
    ax.axhline(0, color="black", lw=.7)
    ax.set(xlabel="Quantile (tau)", ylabel=f"{term} coefficient",
           title=f"Quantile regression estimates for {term}")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
