import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F


def plot_close_pred_vs_actual(
    preds, actuals, filename: str, last_close=None, denorm_fn=None, c_id=-1
):
    """
    Plot predicted vs actual close price time-series.

    Args:
        preds (np.ndarray|torch.Tensor): model outputs (N, *) or (N,) — if multicolumn, c_id selects close.
        actuals (np.ndarray|torch.Tensor): ground-truth values (same shape as preds).
        last_close (np.ndarray): last-known close prices (USD) used for denorm (length N). Required if denorm_fn provided.
        denorm_fn (callable): function(pred_column, last_close) -> USD prices. If None, preds/actuals assumed USD already.
        c_id (int): column index for close in preds/actuals (default -1).
        filename (str): path to save PNG.

    Returns:
        dict: { 'rmse', 'mae', 'corr' } computed on USD prices used in the plot.
    """

    # convert to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(actuals, torch.Tensor):
        actuals = actuals.detach().cpu().numpy()

    # select close column if multi-dim
    if preds.ndim > 1:
        pred_col = preds[:, c_id]
    else:
        pred_col = preds.ravel()
    if actuals.ndim > 1:
        actual_col = actuals[:, c_id]
    else:
        actual_col = actuals.ravel()

    # denormalize if requested
    if denorm_fn is not None:
        if last_close is None:
            raise ValueError("last_close is required when denorm_fn is provided")
        pred_usd = np.asarray(denorm_fn(pred_col, np.asarray(last_close)))
        actual_usd = np.asarray(denorm_fn(actual_col, np.asarray(last_close)))
    else:
        pred_usd = np.asarray(pred_col)
        actual_usd = np.asarray(actual_col)
    # metrics
    rmse = float(
        np.sqrt(np.mean((np.nan_to_num(pred_usd) - np.nan_to_num(actual_usd)) ** 2))
    )
    mae = float(np.mean(np.abs(pred_usd - actual_usd)))
    corr = (
        float(np.corrcoef(pred_usd, actual_usd)[0, 1])
        if pred_usd.size > 1
        else float("nan")
    )
    # plot
    x = np.arange(len(pred_usd))
    plt.figure(figsize=(12, 5))
    plt.plot(x, actual_usd, label="Actual Close", color="black", linewidth=1.5)
    plt.plot(
        x, pred_usd, label="Predicted Close", color="tab:blue", linewidth=1.2, alpha=0.9
    )
    plt.fill_between(
        x,
        actual_usd,
        pred_usd,
        where=(pred_usd > actual_usd),
        color="green",
        alpha=0.18,
        interpolate=True,
        label="Overprediction",
    )
    plt.fill_between(
        x,
        actual_usd,
        pred_usd,
        where=(pred_usd < actual_usd),
        color="red",
        alpha=0.18,
        interpolate=True,
        label="Underprediction",
    )
    plt.title(
        f"Predicted vs Actual Close — RMSE={rmse:.2f}, MAE={mae:.2f}, Corr={corr:.3f}"
    )
    plt.xlabel("Sample Index")
    plt.ylabel("Close Price (USD)")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(filename, dpi=150)
    plt.close()

    return {"rmse": rmse, "mae": mae, "corr": corr}


def plot_training_history(history: dict, filename: str, metrics: dict = None):
    """
    Plot train/val loss and optional metric series (e.g. directional_accuracy) on a secondary axis.
    - history: {'train_loss': [...], 'val_loss': [...], ...}
    - metrics: optional dict of name->list (same length as losses)
    """
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs, history.get("train_loss", []), label="Train Loss", color="tab:blue"
    )
    plt.plot(epochs, history.get("val_loss", []), label="Val Loss", color="tab:orange")
    plt.xlabel("Epoch")
    plt.grid(alpha=0.3)
    ax = plt.gca()

    if metrics:
        ax2 = ax.twinx()
        colors = ["tab:green", "tab:red", "tab:purple", "tab:brown"]
        for i, (name, series) in enumerate(metrics.items()):
            ax2.plot(
                epochs,
                series,
                label=name,
                color=colors[i % len(colors)],
                linestyle="--",
            )
        ax2.set_ylabel("Metric")
        # combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")
    else:
        ax.legend(loc="upper left")

    plt.title("Training History")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_interpretability_report(
    model, val_loader, mkt_cols, sent_cols, fold, filename
):
    """Generate an interpretability report with attention and permutation plots.

    Args:
        model: Trained torch model used for inference.
        val_loader: Validation DataLoader to sample data from.
        mkt_cols: List of market feature names.
        sent_cols: List of sentiment feature names.
        fold: Fold identifier (int or str) used in titles/filenames.
        filename: Base filename where the report PNG will be saved.

    Returns:
        None (saves PNG file to disk).
    """
    # 1. Setup
    all_feature_names = mkt_cols + sent_cols
    fig, axs = plt.subplots(2, 1, figsize=(16, 12))
    plt.suptitle(
        f"Model Interpretation Report - Fold {fold}", fontsize=20, fontweight="bold"
    )

    model.eval()
    device = next(model.parameters()).device

    # Grab a batch for the Attention Heatmap
    m, s, t, _ = next(iter(val_loader))
    m, s = m.to(device), s.to(device)

    # --- 2. Attention Heatmap (Time Focus) ---
    ax2 = axs[0]
    with torch.no_grad():
        # REPLICATE THE FORWARD PASS LOGIC
        # 1. Combine Market and Sentiment into one sequence (Early Fusion)
        combined_seq = torch.cat((m, s), dim=2)

        # 2. Transpose for CNN: (Batch, Channels=15, Seq_Len=30)
        feat_vec = combined_seq.transpose(1, 2)

        # 3. Pass through Parallel CNNs
        # m_cnn3 = F.relu(model.cnn3(feat_vec))
        m_cnn5 = F.relu(model.cnn5(feat_vec))
        # m_cnn = torch.cat((m_cnn3, m_cnn5), dim=1) # Concatenate filters
        m_cnn = m_cnn5

        # 4. LSTM + Attention
        m_cnn = m_cnn.transpose(1, 2)
        lstm_out, _ = model.lstm(m_cnn)
        _, weights = model.attention(lstm_out)

    avg_w = weights.cpu().squeeze().mean(dim=0).numpy().reshape(1, -1)
    im = ax2.imshow(avg_w, cmap="YlGnBu", aspect="auto")
    plt.colorbar(im, ax=ax2)
    ax2.set_title("Temporal Attention: Which days matter most?")
    ax2.set_xlabel("Days in Past (0=Oldest, 29=Most Recent)")
    ax2.set_yticks([])

    # --- 3. Permutation Importance (All Features) ---
    ax3 = axs[1]
    # Pass the full list of 15 features
    importances = calculate_permutation_importance(
        model, val_loader, mkt_cols, sent_cols, fold, filename=None
    )

    imp_series = pd.Series(importances).sort_values()
    # Color code: Market = teal, Sentiment = orange
    sent_cols = all_feature_names[-2:]  # Last 2 are sentiment features
    colors = ["orange" if x in sent_cols else "teal" for x in imp_series.index]

    imp_series.plot(kind="barh", color=colors, ax=ax3)
    ax3.set_title("Feature Importance: Shuffling Impact on MAE")
    ax3.set_xlabel("Error Increase (BPS scaled x100)")
    ax3.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename.replace(".png", f"_fold{fold}.png"))
    plt.close()
    print(f"Report for Fold {fold} saved to {filename}")


def calculate_permutation_importance(
    model, val_loader, mkt_cols, sent_cols, fold, filename
):
    """Compute permutation importance per feature measured by MAE increase.

    Args:
        model: Trained torch model.
        val_loader: DataLoader yielding validation batches.
        mkt_cols: List of market feature names.
        sent_cols: List of sentiment feature names.
        fold: Fold identifier for labeling.
        filename: If provided, save a barplot of importances to this path.

    Returns:
        dict: Mapping feature_name -> relative importance (ratio increase in MAE).
    """
    device = next(model.parameters()).device
    model.eval()
    c_id = -1

    # 1. Calculate Base Score across the whole validation set
    base_mae = 0
    all_m, all_s, all_t = [], [], []
    with torch.no_grad():
        for m, s, t, _ in val_loader:
            all_m.append(m)
            all_s.append(s)
            all_t.append(t)

    m_orig = torch.cat(all_m, dim=0).to(device)
    s_orig = torch.cat(all_s, dim=0).to(device)
    t_orig = torch.cat(all_t, dim=0).to(device)

    with torch.no_grad():
        base_out = model(m_orig, s_orig)
        base_mae = torch.abs(base_out[:, c_id] - t_orig[:, c_id]).mean().item()

    importances = {}
    feature_names = mkt_cols + sent_cols
    m_count = len(mkt_cols)

    for i, name in enumerate(feature_names):
        m_perm = m_orig.clone()
        s_perm = s_orig.clone()

        # Shuffle
        if i < m_count:
            m_perm[:, :, i] = m_orig[torch.randperm(m_orig.size(0)), :, i]
        else:
            s_perm[:, :, i - m_count] = s_orig[
                torch.randperm(s_orig.size(0)), :, i - m_count
            ]

        with torch.no_grad():
            perm_out = model(m_perm, s_perm)
            perm_mae = torch.abs(perm_out[:, c_id] - t_orig[:, c_id]).mean().item()

        # Calculate as % change (Ratio is more stable)
        importances[name] = (perm_mae - base_mae) / (base_mae + 1e-9)

    if filename is not None:
        # 1. Correct way to create both figure and axes
        fig, ax = plt.subplots(figsize=(8, 8))

        # 2. Map names to values if importances is a dict
        imp_series = pd.Series(importances).sort_values()

        # 3. Fix the parameter: use ax=ax
        imp_series.plot(kind="barh", color="teal", ax=ax)

        plt.title(f"Feature Importance - Fold {fold}\n(Permutation Impact on MAE)")
        plt.xlabel("Error Increase when Shuffled")

        # 4. Optional: Add a grid for better scannability
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(filename.replace(".png", f"_fold{fold}.png"))
        plt.close()

    return importances


def plot_feature_time_heatmap(model, val_loader, feature_names, filename):
    """Plot a saliency heatmap showing feature importance over time.

    Args:
        model: Trained torch model (must support backward for gradients).
        val_loader: DataLoader to fetch an example batch.
        feature_names: List of feature names for Y-axis labels.
        filename: Path to save the generated heatmap PNG.

    Returns:
        None (saves PNG file).
    """
    model.eval()
    device = next(model.parameters()).device

    c_id = -1

    # 1. Get a batch of data
    with torch.backends.cudnn.flags(enabled=False):
        mkt_data, sent_data, _, _ = next(iter(val_loader))
        mkt_data = mkt_data.to(device).requires_grad_(True)
        sent_data = sent_data.to(device).requires_grad_(True)

        output = model(mkt_data, sent_data)
        loss = output[:, c_id].mean()
        model.zero_grad()
        loss.backward()

    # 4. Extract gradients (Saliency)
    # Shape of grad: (batch, seq_len, mkt_feat_dim)
    all_saliency = (
        torch.cat([mkt_data.grad.abs(), sent_data.grad.abs()], dim=2)
        .mean(dim=0)
        .cpu()
        .numpy()
    )

    # 5. Plotting
    plt.figure(figsize=(14, 8))
    # Transpose to get Features on Y and Time on X
    plt.imshow(all_saliency.T, cmap="hot", aspect="auto", interpolation="nearest")

    plt.colorbar(label="Feature Importance (Absolute Gradient)")
    plt.title("2D Feature-Time Importance (Saliency Map)")
    plt.ylabel("Features")
    plt.xlabel("Days in Sequence (0 = Oldest, 29 = Today)")

    # Set Y-ticks to feature names
    plt.yticks(range(len(feature_names)), feature_names)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_feature_weights(model, feature_names, filename):
    """Plot mean absolute CNN input-channel weights as a horizontal bar chart.

    Args:
        model: Trained torch model with attribute `cnn5` (Conv1d layer).
        feature_names: List of feature names corresponding to input channels.
        filename: Path to save the bar chart PNG.

    Returns:
        None (saves PNG file).
    """
    # model.cnn3 is the first layer. Shape: (out_channels, in_channels, kernel_size)

    weights = model.cnn5.weight.data.cpu().abs().mean(dim=(0, 2)).numpy()

    plt.figure(figsize=(10, 6))
    pd.Series(weights, index=feature_names).sort_values().plot(
        kind="barh", color="teal"
    )
    plt.title("Mean Absolute Weights: CNN Layer 1")
    plt.xlabel("Importance (Weight Magnitude)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_maw_progression(weight_history, fold_num, filename: str):
    """Plot progression of mean-absolute weights per feature across epochs.

    Args:
        weight_history: Dict mapping feature_name -> list of values per epoch.
        fold_num: Identifier for the fold (used in title).
        filename: Output PNG path to save the plot.

    Returns:
        None (saves PNG file).
    """
    plt.figure(figsize=(10, 6))
    for feature_name, values in weight_history.items():
        plt.plot(values, label=feature_name)

    plt.title(f"Feature Weight Progression - Fold {fold_num}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Weight")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_model_results(preds, targets, backtest_results, filename: str):
    """
    4-Panel Dashboard: Equity, Drawdown, Prediction Scatter, and Error Histogram.
    """
    # Document parameters and return value
    """Create a 4-panel dashboard of backtest and prediction diagnostics.

    Args:
        preds: numpy array of model predictions (num_samples, n_outputs).
        targets: numpy array of ground-truth targets.
        backtest_results: Dict returned by `backtest_with_costs` containing equity curves.
        filename: Path to save the dashboard PNG.

    Returns:
        None (saves PNG file).
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    c_id = -1

    # --- 1. Equity Curve ---
    ax1 = axes[0]
    ax1.plot(
        backtest_results["equity_curve"], label="Strategy", color="gold", linewidth=2
    )
    ax1.plot(
        backtest_results["buy_and_hold"],
        label="Buy & Hold",
        color="black",
        linestyle="--",
        alpha=0.6,
    )
    ax1.set_title("Strategy Growth ($1000 Start)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Account Value ($)")
    ax1.legend()
    ax1.grid(alpha=0.2)

    # --- 2. Drawdown (The "Pain" Graph) ---
    ax2 = axes[1]
    equity = backtest_results["equity_curve"]
    buy_and_hold = backtest_results["buy_and_hold"]

    # Plot strategy vs buy-and-hold
    ax2.plot(equity, label="Strategy", color="gold", linewidth=1.5)
    ax2.plot(buy_and_hold, label="Buy & Hold", color="black", linestyle="--", alpha=0.6)

    # Difference between strategy and buy-and-hold to show periods of out/under-performance
    diff = equity - buy_and_hold
    pos = np.clip(diff, a_min=0, a_max=None)
    neg = np.clip(diff, a_min=None, a_max=0)

    ax2.fill_between(
        range(len(diff)), 0, pos, color="green", alpha=0.3, label="Outperformance"
    )
    ax2.fill_between(
        range(len(diff)), 0, neg, color="red", alpha=0.3, label="Underperformance"
    )

    ax2.set_title(
        "Strategy vs Buy & Hold (Green = outperformance, Red = underperformance)",
        fontsize=12,
    )
    ax2.set_ylabel("Value ($)")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.2)

    # --- 3. Prediction vs Actual (Magnitude) ---
    ax3 = axes[2]
    ax3.scatter(
        targets[:, c_id] / 100, preds[:, c_id] / 100, alpha=0.5, color="#1f77b4", s=15
    )
    # Identity line (where predictions = reality)
    all_data = np.concatenate([targets[:, c_id] / 100, preds[:, c_id] / 100])
    low, high = all_data.min(), all_data.max()
    ax3.plot([low, high], [low, high], "r--", alpha=0.8)
    ax3.set_title("Prediction vs. Reality Magnitude", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Actual Log-Return")
    ax3.set_ylabel("Predicted Log-Return")
    ax3.grid(alpha=0.2)

    # --- 4. Residuals (Bias Check) ---
    ax4 = axes[3]
    errors = (targets[:, c_id] - preds[:, c_id]) / 100
    ax4.hist(errors, bins=60, color="seagreen", alpha=0.7, edgecolor="white")
    ax4.axvline(0, color="black", linestyle="-", linewidth=1)
    ax4.set_title(
        "Error Distribution (Zero-Bias Check)", fontsize=14, fontweight="bold"
    )
    ax4.set_xlabel("Forecast Error")
    ax4.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Dashboard saved as {filename}")


# --- Financial simulation
def calculate_trading_metrics(
    preds, targets, epsilon_pct, last_close_prices, dataset_denorm_func
):
    """Calculate trading-related metrics given predictions and targets.

    The function assumes Gold price normalization as logarithmic return with the division in the logarithmic function from the last day close price. All multiplied by 100.
    I.e. `normalized value = log(curr_price/last_close_price)*100`, where curr_price can be any of Open (O), High(H), Low (L) or Close (C).

    In order to calculate the metrics in regards of last_close_prices in USD, function must receive H, L, C changes of the price.

    The assumptions must be met:
    This function assumes to be passed predictions of [O,H,L,C]. The order must be held like this.
    Given such output of dimension 4, everything holds. Else:
        - If dimension is 3 and `last_close_prices` is given, we assume to receive [H,L,C].
        - If dimension is 3 and `last_close_prices == None`, we assume to receive [*,*,C].
        - If dimension is less than 3, the function will assume [*,C] ([C] accordingly) and will calculate as if `last_close_prices == None`

    :param preds: np.array of shape (num_samples, 4) - predicted log-returns [O,H,L,C]
    :param targets: np.array of shape (num_samples, 4) - actual log-returns [O,H,L,C]
    :param epsilon_pct: float - threshold for epsilon accuracy (e.g., 0.002 for 0.2%)
    :param last_close_prices: np.array of shape (num_samples, 4) - last known real prices [O,H,L,C] before prediction
    :param dataset_denorm_func: function - a function that takes (preds, last_close_prices) and returns denormalized prices in USD.
    :return: dict with metrics: range_coverage, epsilon_accuracy, directional_accuracy, mae, mape, max_pred_move, avg_pred_move. Note that that range coverage might return a `np.nan` if the target does not contain [H, L, C].
    """

    # Adapting epsilon to normalization
    epsilon_pct = epsilon_pct * 100

    # Checking dimensions
    num_cols = preds.shape[1] if len(preds.shape) > 1 else 1

    # Enforcing assumptions:
    if num_cols == 4:
        h_idx, l_idx, c_idx = 1, 2, 3
    elif num_cols == 3 and last_close_prices is not None:
        h_idx, l_idx, c_idx = 0, 1, 2
    else:
        # For dimensions < 3, we force last_close_prices to None logic for Range Coverage
        h_idx, l_idx, c_idx = None, None, -1
        last_close_prices = None

    # Directional Accuracy
    actual_dir = np.sign(targets[:, c_idx])
    pred_dir = np.sign(preds[:, c_idx])
    dir_acc = (actual_dir == pred_dir).astype(float).mean()

    # Log-space Volatility (Magnitude)
    # This tells us the largest move the model dared to predict
    max_pred_move = np.max(np.abs(preds[:, c_idx]))
    avg_pred_move = np.mean(np.abs(preds[:, c_idx]))

    # Epsilon Accuracy
    # With log returns, the difference |pred - target| IS essentially the % error.
    # Because ln(A) - ln(B) = ln(A/B) ≈ % change.
    log_error = np.abs(preds[:, c_idx] - targets[:, c_idx])
    epsilon_hit = log_error <= epsilon_pct
    epsilon_acc = epsilon_hit.astype(float).mean()

    # Log space
    mae = np.mean(np.abs(preds - targets))
    if num_cols > 2:
        if last_close_prices is not None and dataset_denorm_func is not None:
            last_close = last_close_prices[:, 3]

            # Convert predicted log-returns back to USD
            pred_high_usd = dataset_denorm_func(preds[:, h_idx], last_close)  # High
            pred_low_usd = dataset_denorm_func(preds[:, l_idx], last_close)  # Low
            pred_close_usd = dataset_denorm_func(preds[:, c_idx], last_close)  # Close

            # Convert target log-returns back to USD
            actual_close_usd = dataset_denorm_func(
                targets[:, c_idx], last_close
            )  # Close Target

            # MAPE (Price-based)
            mape = (
                np.mean(np.abs((actual_close_usd - pred_close_usd) / actual_close_usd))
                * 100
            )

            # Range Coverage (Price-based)
            # Is the actual close price between our predicted High/Low?
            covered = (actual_close_usd <= pred_high_usd) & (
                actual_close_usd >= pred_low_usd
            )
            range_coverage_acc = covered.astype(float).mean()

        else:
            # Fallback: Log-relative MAPE (less intuitive but stable)
            # We treat the log-error as the percentage itself
            mape = np.mean(np.abs(preds - targets)) * 100

            # Range Coverage
            # Is the actual close 'return' within the predicted High/Low 'return' boundaries?
            covered = (targets[:, c_idx] <= preds[:, h_idx]) & (
                targets[:, c_idx] >= preds[:, l_idx]
            )
            range_coverage_acc = covered.astype(float).mean()
    else:
        mape = mape = np.mean(np.abs(preds - targets)) * 100
        range_coverage_acc = np.nan

    return {
        "range_coverage": range_coverage_acc,
        "epsilon_accuracy": epsilon_acc,
        "directional_accuracy": dir_acc,
        "mae": mae,
        "mape": mape,
        "max_pred_move": max_pred_move,
        "avg_pred_move": avg_pred_move,
    }


def backtest_with_costs(
    preds, targets, initial_capital=1000.0, threshold=0.001, fee=0.0005
):
    """Simulate a simple threshold-based trading strategy and compute metrics.

    Args:
        preds: Array of predicted returns (num_samples, n_outputs) or (num_samples,).
        targets: Array of actual returns used for P&L calculation.
        initial_capital: Starting capital in USD.
        threshold: Threshold on predicted close-return to trigger long/short.
        fee: Proportional transaction fee applied when changing position.

    Returns:
        dict: Simulation outputs including `equity_curve`, `buy_and_hold`, `num_trades`, `sharpe_ratio`, `final_value`.
    """
    c_id = -1

    # 1. Generate Signals (Column -1 is 'Close')
    signals = np.zeros(len(preds))
    signals[preds[:, c_id] > threshold] = 1
    signals[preds[:, c_id] < -threshold] = -1

    equity = [initial_capital]
    current_position = 0  # 0=Cash, 1=Long, -1=Short
    num_trades = 0

    for i in range(len(signals)):
        new_signal = signals[i]
        current_equity = equity[-1]

        # 2. Apply Transaction Fee if the position changes
        if new_signal != current_position:
            current_equity *= 1 - fee
            num_trades += 1
            current_position = new_signal

        # 3. Calculate Market Movement for the day
        # Exp(actual_return) gives the price multiplier
        # If signal is 0 (Cash), multiplier is Exp(0) = 1 (No change)
        daily_multiplier = np.exp(current_position * targets[i, c_id] / 100)

        equity.append(current_equity * daily_multiplier)

    # Convert to array and remove the seed value for analysis
    equity_curve = np.array(equity[1:])
    buy_and_hold = initial_capital * np.exp(np.cumsum(targets[:, c_id]) / 100)

    # 4. Calculate Risk Metrics
    # Percentage daily returns of the strategy
    strategy_pct_returns = np.diff(equity_curve) / equity_curve[:-1]

    # Annualized Sharpe Ratio (assuming 252 trading days)
    # Higher is better; > 1.0 is considered good for a strategy
    sharpe = (
        np.sqrt(252)
        * np.mean(strategy_pct_returns)
        / (np.std(strategy_pct_returns) + 1e-9)
    )

    return {
        "equity_curve": equity_curve,
        "buy_and_hold": buy_and_hold,
        "num_trades": num_trades,
        "sharpe_ratio": sharpe,
        "final_value": equity_curve[-1],
    }


def find_best_threshold(preds, targets, logger=None):
    """Test several decision thresholds and print/signal summary statistics.

    Args:
        preds: Predicted returns array.
        targets: Actual returns array.
        logger: Optional logger to write results instead of printing.

    Returns:
        None (prints or logs a small summary table).
    """

    # Sensible thresholds in fractional decimal: 0.1% -> 0.001, 0.2% -> 0.002
    thresholds = [0.001, 0.002, 0.003, 0.005]

    if not logger:
        for t in thresholds:
            res = backtest_with_costs(preds, targets, threshold=t, fee=0.0003)
            print(
                f"{t * 100:>9.3f}% | {res['num_trades']:>8} | ${res['final_value']:>10.2f} | {res['sharpe_ratio']:>6.2f}"
            )
    else:
        for t in thresholds:
            res = backtest_with_costs(preds, targets, threshold=t, fee=0.0003)
            logger.info(
                f"{t * 100:>9.3f}% | {res['num_trades']:>8} | ${res['final_value']:>10.2f} | {res['sharpe_ratio']:>6.2f}"
            )


# ---- Feature Engineering

def identify_low_importance_features(importances, threshold=0.0001):
    """
    Returns a list of features to keep and a list of features to drop.
    """
    imp_series = pd.Series(importances)
    print(imp_series)
    to_keep = imp_series[imp_series > threshold].index.tolist()
    to_drop = imp_series[imp_series <= threshold].index.tolist()

    return to_keep, to_drop


# ----
