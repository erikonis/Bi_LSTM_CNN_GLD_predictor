import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Constants import ColNames
import dataset_former

def plot_correlation_matrix(df, title="Feature Correlation Matrix"):
    """
    Calculates and plots a heatmap of the correlation matrix for a dataframe.
    """
    # 1. Calculate the correlation matrix (using Pearson by default)
    corr = df.corr()

    # 2. Create a mask to hide the upper triangle (optional but cleaner)
    # Since the matrix is symmetrical, we only need to see one half.
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 3. Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # 4. Draw the heatmap
    # cmap: 'coolwarm' is great for seeing pos/neg correlations (Red = Pos, Blue = Neg)
    sns.heatmap(corr, 
                mask=mask, 
                annot=True,          # Show the correlation numbers
                fmt=".2f",           # Two decimal places
                cmap='coolwarm', 
                center=0,            # 0 is the neutral midpoint
                linewidths=.5, 
                cbar_kws={"shrink": .8})

    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.show()


if __name__ == "__main__":
    dataset = dataset_former.MarketDataset2.load()

    a, b, c = dataset.get_active_cols()
    final = a + b
    #noo = [ColNames.MACD_SIG_NORM, ColNames.RSI_NORM, ColNames.SMA_20_NORM, ColNames.SMA_50_NORM]
    #final = [f for f in final if f not in noo]

    plot_correlation_matrix(dataset.dataframe[final])