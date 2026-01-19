"""Small plotting utilities for exploratory data analysis (EDA).

Provides a correlation heatmap and a time-series close-price plot. Each
function saves the generated figure to disk and also calls ``plt.show()``.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Constants import ColNames


def plot_correlation_matrix(df, title="Feature Correlation Matrix"):
    """Plot a correlation heatmap and save to `correlation_matrix.png`.

    Parameters:
        df (pandas.DataFrame): DataFrame of features to correlate.
    """
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=.5,
        cbar_kws={"shrink": .8},
    )
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.show()


def plot_close_price(df, title="Close Price Over Time"):
    """Plot close price over time and save to `close_price_over_time.png`.

    Parameters:
        df (pandas.DataFrame): DataFrame containing date and close price columns.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df[ColNames.DATE], df[ColNames.CLOSE], label="Close Price", color="blue")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig("close_price_over_time.png")
    plt.show()