import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset_former

import logging

class LogOHLCLoss(nn.Module):
    def __init__(self, penalty_weight=5.0):
        super(LogOHLCLoss, self).__init__()
        self.mse = nn.HuberLoss() 
        self.penalty_weight = penalty_weight

    def forward(self, pred, target):
        # pred/target shape: (batch, 4) -> [r_open, r_high, r_low, r_close]
        
        # 1. Base Regression Loss (Standard Huber)
        base_loss = self.mse(pred, target)
        
        # 2. Log-Space Constraints
        # High must be >= Open, Low, and Close
        # If r_open - r_high > 0, it means Open is higher than High (Violation!)
        h_o_penalty = torch.mean(F.relu(pred[:, 0] - pred[:, 1]))
        h_l_penalty = torch.mean(F.relu(pred[:, 2] - pred[:, 1])) # Low > High
        h_c_penalty = torch.mean(F.relu(pred[:, 3] - pred[:, 1]))
        
        # Low must be <= Open, High, and Close
        # If r_low - r_open > 0, it means Low is higher than Open (Violation!)
        l_o_penalty = torch.mean(F.relu(pred[:, 2] - pred[:, 0]))
        l_c_penalty = torch.mean(F.relu(pred[:, 2] - pred[:, 3]))
        
        total_penalty = h_o_penalty + h_l_penalty + h_c_penalty + l_o_penalty + l_c_penalty
        
        return base_loss + (self.penalty_weight * total_penalty)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_dim)
        attn_weights = torch.tanh(self.attn(lstm_output)) # (batch, seq_len, 1)
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_output * soft_attn_weights, dim=1)
        return context, soft_attn_weights

class GoldPredictor(nn.Module):
    def __init__(self, mkt_feat_dim, sent_feat_dim, hidden_dim=64):
        super(GoldPredictor, self).__init__()
        self.cnn3 = nn.Conv1d(mkt_feat_dim, int(np.ceil(hidden_dim/2)), kernel_size=3, padding=1)
        self.cnn5 = nn.Conv1d(mkt_feat_dim, int(np.floor(hidden_dim/2)), kernel_size=5, padding=2)
        
        # --- Market Branch (CNN + Bi-LSTM) ---
        #self.cnn = nn.Conv1d(in_channels=mkt_feat_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2) # *2 for Bidirectional
        
        # --- Sentiment Branch ---
        sent_inner = 32
        self.sent_mlp = nn.Sequential(
        nn.Linear(sent_feat_dim, sent_inner),
        nn.BatchNorm1d(sent_inner), # Keeps sentiment features from being "washed out"
        nn.ReLU(),
        nn.Dropout(0.2))
        
        # --- Fusion Head ---
        # (hidden_dim * 2 from Bi-LSTM) + 16 from Sentiment
        self.fc_fusion = nn.Sequential(
            nn.Linear((hidden_dim * 2) + sent_inner, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4) # Predicts [Open, High, Low, Close]
        )

    def forward(self, mkt_seq, sent_vec):
        # mkt_seq: (batch, seq_len, mkt_feat_dim)
        # CNN expects (batch, channels, seq_len)
        mkt_seq = mkt_seq.transpose(1, 2)
        mkt_cnn3 = F.relu(self.cnn3(mkt_seq))
        mkt_cnn5 = F.relu(self.cnn5(mkt_seq))
        
        # Concatenate along the channel dimension
        mkt_cnn = torch.cat((mkt_cnn3, mkt_cnn5), dim=1)
        
        # Back to (batch, seq_len, hidden_dim) for LSTM
        mkt_cnn = mkt_cnn.transpose(1, 2)
        lstm_out, _ = self.lstm(mkt_cnn)
        
        # Attention pooling
        mkt_context, _ = self.attention(lstm_out)
        
        # Sentiment Branch
        sent_context = self.sent_mlp(sent_vec)
        
        # Late Fusion
        combined = torch.cat((mkt_context, sent_context), dim=1)
        output = self.fc_fusion(combined)
        return output
    
    def save(self, optimizer, fold, performance, path="output/gold_model_latest.pth"):
        """Saves the model state and training context."""
        state = {
            'fold': fold,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'performance' :  performance
        }
        torch.save(state, path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(model, path="output/gold_model_latest.pth", optimizer=None):
        """Loads the model for inference or further training."""
        if not os.path.exists(path):
            print("No saved model found.")
            return None
            
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from Fold {checkpoint['fold']}")
        return checkpoint['fold']

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, logger=None, debug=False):
    """
    Proposed hyperparameters in the paper 'Implementation of Long Short-Term Memory for Gold Prices Forecasting'
    are: epoch = 100, LR = 0.01 with Adam optimizer, and expanding window.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Allow for adaptive learning rate:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    criterion = LogOHLCLoss() # Robust to financial outliers
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for mkt_data, sent_data, targets, all_real_prices in train_loader:

            if debug:
                pretty_print_batch(mkt_data, sent_data, targets, samples=1, logger=logger)

            mkt_data, sent_data, targets = mkt_data.to(device), sent_data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(mkt_data, sent_data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        # Validation
        model.eval()
        val_losses = []

        all_val_preds = []
        all_val_targets = []
        all_real_prices = []

        with torch.no_grad():
            for mkt_data, sent_data, targets, real_price in val_loader:
                mkt_data, sent_data, targets = mkt_data.to(device), sent_data.to(device), targets.to(device)
                outputs = model(mkt_data, sent_data)
                v_loss = criterion(outputs, targets)
                val_losses.append(v_loss.item())

                all_val_preds.append(outputs.cpu())
                all_val_targets.append(targets.cpu())
                all_real_prices.append(real_price) 

        # Concatenate all batches
        all_val_preds = torch.cat(all_val_preds, dim=0).numpy()
        all_val_targets = torch.cat(all_val_targets, dim=0).numpy()
        all_real_prices = torch.cat(all_real_prices, dim=0).numpy()
        

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        scheduler.step(avg_val_loss)

        # Verbose logging every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            if logger:
                logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")


        # Calculate trading metrics on validation set
        epsilon_pct = 0.002
        val_metrics = calculate_trading_metrics(all_val_preds, all_val_targets, epsilon_pct=epsilon_pct, last_close_prices=all_real_prices)            
    
        if logger:
            logger.info(f"Epoch {epoch}:")
            logger.info(f"  - Range Coverage: {val_metrics['range_coverage']:.2%}")
            logger.info(f"  - Epsilon Hit ({epsilon_pct*100}%): {val_metrics['epsilon_accuracy']:.2%}")
            logger.info(f"  - Directional Acc: {val_metrics['directional_accuracy']:.2%}")
        else:
            print(f"Epoch {epoch}:")
            print(f"  - Range Coverage: {val_metrics['range_coverage']:.2%}")
            print(f"  - Epsilon Hit ({epsilon_pct*100}%): {val_metrics['epsilon_accuracy']:.2%}")
            print(f"  - Directional Acc: {val_metrics['directional_accuracy']:.2%}")
                
        
    return history, optimizer

def evaluate_test_set(model, test_loader, logger=None, scaler_target=None, debug=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_preds = []
    all_actuals = []
    all_real_prices = []

    with torch.no_grad():
        for mkt_data, sent_data, targets, real_price in test_loader:

            # Debug: Pretty print
            if debug:
                pretty_print_batch(mkt_data, sent_data, targets, samples=1)
            mkt_data, sent_data, targets = mkt_data.to(device), sent_data.to(device), targets.to(device)
            
            outputs = model(mkt_data, sent_data)
            
            # Inverse transform to get back to real dollar prices if you scaled them
            if scaler_target is not None:
                actuals = scaler_target.inverse_transform(targets.cpu().numpy())
                preds = scaler_target.inverse_transform(outputs.cpu().numpy())
            else:
                actuals = targets.cpu().numpy()
                preds = outputs.cpu().numpy()
            
            all_preds.append(torch.from_numpy(preds))
            all_actuals.append(torch.from_numpy(actuals))
            all_real_prices.append(real_price)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_actuals = torch.cat(all_actuals, dim=0).numpy()
    all_real_prices = torch.cat(all_real_prices, dim=0).numpy()

    # --- Metric 1: RMSE (Regression Error) ---
    mse = np.mean((all_preds - all_actuals)**2)
    rmse = np.sqrt(mse)

     # --- Metric 2: Trading Metrics ---
    epsilon_pct = 0.002
    val_metrics = calculate_trading_metrics(all_preds, all_actuals, epsilon_pct=epsilon_pct, last_close_prices=all_real_prices)

    # --- MEtric 3: Backtest Strategy ---
    results = backtest_with_costs(all_preds, all_actuals, threshold=0.002, fee=0.0003)

    # --- Metric 4: Threshold Sensitivity Analysis ---
    find_best_threshold(all_preds, all_actuals)

    if not logger:
        print(f"--- Final Test Results ---")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"  - Range Coverage: {val_metrics['range_coverage']:.2%}")
        print(f"  - Epsilon Hit ({epsilon_pct*100}%): {val_metrics['epsilon_accuracy']:.2%}")
        print(f"  - Directional Acc: {val_metrics['directional_accuracy']:.2%}")
        print(f"  - MAE: {val_metrics['mae']:.4f}")
        print(f"  - MAPE: {val_metrics['mape']:.2f}%")
        print(f"  - Max Predicted Move (log-return): {val_metrics['max_pred_move']:.4f}")
        print(f"  - Avg Predicted Move (log-return): {val_metrics['avg_pred_move']:.4f}")
    
        print(f"\n--- Trading Performance (with 0.03% fee and $1000 capital) ---")
        print(f"Total Trades: {results['num_trades']}")
        print(f"Strategy Final Value: ${results['equity_curve'][-1]:.2f}")
        print(f"Buy & Hold Final Value: ${results['buy_and_hold'][-1]:.2f}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
        if results['final_value'] > (results['buy_and_hold'][-1]/1000 - 1):
            print("STRATEGY OUTPERFORMED MARKET")
        else:
            print("MARKET OUTPERFORMED STRATEGY")
    else:
        logger.info(f"--- Final Test Results ---")
        logger.info(f"Test RMSE: {rmse:.4f}")
        logger.info(f"  - Range Coverage: {val_metrics['range_coverage']:.2%}")
        logger.info(f"  - Epsilon Hit ({epsilon_pct*100}%): {val_metrics['epsilon_accuracy']:.2%}")
        logger.info(f"  - Directional Acc: {val_metrics['directional_accuracy']:.2%}")
        logger.info(f"  - MAE: {val_metrics['mae']:.4f}")
        logger.info(f"  - MAPE: {val_metrics['mape']:.2f}%")
        logger.info(f"  - Max Predicted Move (log-return): {val_metrics['max_pred_move']:.4f}")
        logger.info(f"  - Avg Predicted Move (log-return): {val_metrics['avg_pred_move']:.4f}")
    
        logger.info(f"\n--- Trading Performance (with 0.03% fee and $1000 capital) ---")
        logger.info(f"Total Trades: {results['num_trades']}")
        logger.info(f"Strategy Final Value: ${results['equity_curve'][-1]:.2f}")
        logger.info(f"Buy & Hold Final Value: ${results['buy_and_hold'][-1]:.2f}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
        if results['final_value'] > (results['buy_and_hold'][-1]/1000 - 1):
            logger.info("STRATEGY OUTPERFORMED MARKET")
        else:
            logger.info("MARKET OUTPERFORMED STRATEGY")
    

    #plotting
    plot_model_results(all_preds, all_actuals, results)

    return all_preds, all_actuals, val_metrics

def pretty_print_batch(mkt, sent, targets, samples=1, logger = None):
    """CPU tensors -> print shapes and a compact table preview for `samples` examples."""
    
    b, seq_len, feat = mkt.shape
    if not logger:
        print(f"Batch shapes -> market: {mkt.shape}, sent: {sent.shape}, targets: {targets.shape}")
        
        for i in range(min(samples, b)):
            print(f"\n-- Sample {i} --")
            mdf = pd.DataFrame(mkt[i].cpu().numpy(), columns=[f"mkt_{j}" for j in range(feat)])
            print("Market seq (first rows):")
            print(mdf.head(1).to_string(index=True))
            print("\nSentiment:")
            print(pd.DataFrame(sent[i].cpu().numpy().reshape(1, -1),
                            columns=[f"sent_{j}" for j in range(sent.shape[1])]).to_string(index=False))
            print("\nTargets (O,H,L,C):")
            print(pd.DataFrame(targets[i].cpu().numpy().reshape(1, -1),
                            columns=['Open','High','Low','Close']).to_string(index=False))
        print("-" * 60)
    else:
        logger.info(f":DEBUG:")
        logger.info(f"Batch shapes -> market: {mkt.shape}, sent: {sent.shape}, targets: {targets.shape}")
        for i in range(min(samples, b)):
            logger.info(f"\n-- Sample {i} --")
            mdf = pd.DataFrame(mkt[i].cpu().numpy(), columns=[f"mkt_{j}" for j in range(feat)])
            logger.info("Market seq (first rows):")
            logger.info(mdf.head(30).to_string(index=True))
            logger.info("\nSentiment:")
            logger.info(pd.DataFrame(sent[i].cpu().numpy().reshape(1, -1),
                               columns=[f"sent_{j}" for j in range(sent.shape[1])]).to_string(index=False))
            logger.info("\nTargets (O,H,L,C):")
            logger.info(pd.DataFrame(targets[i].cpu().numpy().reshape(1, -1),
                               columns=['Open','High','Low','Close']).to_string(index=False))
        logger.info("-" * 60)

def setup_logger(filename : str = "output/model_train.log"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(filename),
            logging.StreamHandler() # Still prints to console
        ]
    )
    return logging.getLogger()

def calculate_trading_metrics(preds, targets, epsilon_pct=0.002, last_close_prices=None):
    """Calculate trading-related metrics given predictions and targets.
    
    :param preds: np.array of shape (num_samples, 4) - predicted log-returns [O,H,L,C]
    :param targets: np.array of shape (num_samples, 4) - actual log-returns [O,H,L,C]
    :param epsilon_pct: float - threshold for epsilon accuracy (e.g., 0.002 for 0.2%)
    :param last_close_prices: np.array of shape (num_samples, 4) - last known real prices [O,H,L,C] before prediction
    :return: dict with metrics: range_coverage, epsilon_accuracy, directional_accuracy, mae, mape
    """
    #Directional Accuracy 
    actual_dir = np.sign(targets[:, 3]) 
    pred_dir = np.sign(preds[:, 3])
    dir_acc = (actual_dir == pred_dir).astype(float).mean()

    # Log-space Volatility (Magnitude)
    # This tells us the largest move the model dared to predict
    max_pred_move = np.max(np.abs(preds[:, 3]))
    avg_pred_move = np.mean(np.abs(preds[:, 3]))

    # Epsilon Accuracy
    # With log returns, the difference |pred - target| IS essentially the % error.
    # Because ln(A) - ln(B) = ln(A/B) ≈ % change.
    log_error = np.abs(preds[:, 3] - targets[:, 3])
    epsilon_hit = log_error <= epsilon_pct # 0.002 here means 0.2% price difference
    epsilon_acc = epsilon_hit.astype(float).mean()

    # Log space
    mae = np.mean(np.abs(preds - targets))
    
    if last_close_prices is not None:
        last_close = last_close_prices[:, 3] 

        # Convert predicted log-returns back to USD
        pred_high_usd  = last_close * np.exp(preds[:, 1]) # High
        pred_low_usd   = last_close * np.exp(preds[:, 2]) # Low
        pred_close_usd = last_close * np.exp(preds[:, 3]) # Close

        # Convert target log-returns back to USD
        actual_close_usd = last_close * np.exp(targets[:, 3]) 

        # MAPE (Price-based)
        mape = np.mean(np.abs((actual_close_usd - pred_close_usd) / actual_close_usd)) * 100

        # Range Coverage (Price-based)
        # Is the actual close price between our predicted High/Low?
        covered = (actual_close_usd <= pred_high_usd) & (actual_close_usd >= pred_low_usd)
        range_coverage_acc = covered.astype(float).mean()
    
    else:
        # Fallback: Log-relative MAPE (less intuitive but stable)
        # We treat the log-error as the percentage itself
        mape = np.mean(np.abs(preds - targets)) * 100 

        # Range Coverage
        # Is the actual close 'return' within the predicted High/Low 'return' boundaries?
        covered = (targets[:, 3] <= preds[:, 1]) & (targets[:, 3] >= preds[:, 2])
        range_coverage_acc = covered.astype(float).mean()

    return {
        "range_coverage": range_coverage_acc,
        "epsilon_accuracy": epsilon_acc,
        "directional_accuracy": dir_acc,
        "mae": mae,
        "mape": mape,
        "max_pred_move": max_pred_move,
        "avg_pred_move": avg_pred_move
    }

def backtest_with_costs(preds, targets, initial_capital=1000.0, threshold=0.001, fee=0.0005):
    """
    Simulates trading with compounding returns and transaction fees.
    """
    # 1. Generate Signals (Column 3 is 'Close')
    signals = np.zeros(len(preds))
    signals[preds[:, 3] > threshold] = 1
    signals[preds[:, 3] < -threshold] = -1
    
    equity = [initial_capital]
    current_position = 0 # 0=Cash, 1=Long, -1=Short
    num_trades = 0
    
    for i in range(len(signals)):
        new_signal = signals[i]
        current_equity = equity[-1]
        
        # 2. Apply Transaction Fee if the position changes
        if new_signal != current_position:
            current_equity *= (1 - fee)
            num_trades += 1
            current_position = new_signal
        
        # 3. Calculate Market Movement for the day
        # Exp(actual_return) gives the price multiplier
        # If signal is 0 (Cash), multiplier is Exp(0) = 1 (No change)
        daily_multiplier = np.exp(current_position * targets[i, 3])
        
        equity.append(current_equity * daily_multiplier)
    
    # Convert to array and remove the seed value for analysis
    equity_curve = np.array(equity[1:])
    buy_and_hold = initial_capital * np.exp(np.cumsum(targets[:, 3]))
    
    # 4. Calculate Risk Metrics
    # Percentage daily returns of the strategy
    strategy_pct_returns = np.diff(equity_curve) / equity_curve[:-1]
    
    # Annualized Sharpe Ratio (assuming 252 trading days)
    # Higher is better; > 1.0 is considered good for a strategy
    sharpe = np.sqrt(252) * np.mean(strategy_pct_returns) / (np.std(strategy_pct_returns) + 1e-9)
    
    return {
        "equity_curve": equity_curve,
        "buy_and_hold": buy_and_hold,
        "num_trades": num_trades,
        "sharpe_ratio": sharpe,
        "final_value": equity_curve[-1]
    }

def find_best_threshold(preds, targets):
    print("\n--- Threshold Sensitivity Analysis ---")
    print(f"{'Threshold':<12} | {'Trades':<8} | {'Final Value':<12} | {'Sharpe'}")
    print("-" * 50)
    
    # Test thresholds from 0.05% to 0.5%
    for t in [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005]:
        res = backtest_with_costs(preds, targets, threshold=t, fee=0.0003)
        print(f"{t*100:>9.2f}% | {res['num_trades']:>8} | ${res['final_value']:>10.2f} | {res['sharpe_ratio']:>6.2f}")

def save_predictions_csv(preds, targets, filename="output/gold_predictions.csv"):
    df = pd.DataFrame({
        'Predicted_Return': preds.flatten(),
        'Actual_Return': targets.flatten()
    })
    # Add a column to see the error magnitude
    df['Error'] = df['Actual_Return'] - df['Predicted_Return']
    
    df.to_csv(filename, index=False)
    print(f"saved {len(df)} predictions to {filename}")

def plot_model_results(preds, targets, backtest_results, filename="output/performance_summary.png"):
    """
    4-Panel Dashboard: Equity, Drawdown, Prediction Scatter, and Error Histogram.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    # --- 1. Equity Curve ---
    ax1 = axes[0]
    ax1.plot(backtest_results['equity_curve'], label='Strategy', color='gold', linewidth=2)
    ax1.plot(backtest_results['buy_and_hold'], label='Buy & Hold', color='black', linestyle='--', alpha=0.6)
    ax1.set_title("Strategy Growth ($1000 Start)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Account Value ($)")
    ax1.legend()
    ax1.grid(alpha=0.2)

    # --- 2. Drawdown (The "Pain" Graph) ---
    ax2 = axes[1]
    equity = backtest_results['equity_curve']
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity)
    # Calculate percentage drop from that max
    drawdown = (equity - running_max) / running_max
    
    ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
    ax2.plot(drawdown, color='red', linewidth=1)
    ax2.set_title("Strategy Drawdown (%)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Drop from Peak")
    ax2.set_ylim([None, 0.02]) # Set upper limit to 0 to emphasize drops
    ax2.grid(alpha=0.2)

    # --- 3. Prediction vs Actual (Magnitude) ---
    ax3 = axes[2]
    ax3.scatter(targets[:, 3], preds[:, 3], alpha=0.5, color='#1f77b4', s=15)
    # Identity line (where predictions = reality)
    all_data = np.concatenate([targets[:, 3], preds[:, 3]])
    low, high = all_data.min(), all_data.max()
    ax3.plot([low, high], [low, high], 'r--', alpha=0.8)
    ax3.set_title("Prediction vs. Reality Magnitude", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Actual Log-Return")
    ax3.set_ylabel("Predicted Log-Return")
    ax3.grid(alpha=0.2)

    # --- 4. Residuals (Bias Check) ---
    ax4 = axes[3]
    errors = targets[:, 3] - preds[:, 3]
    ax4.hist(errors, bins=60, color='seagreen', alpha=0.7, edgecolor='white')
    ax4.axvline(0, color='black', linestyle='-', linewidth=1)
    ax4.set_title("Error Distribution (Zero-Bias Check)", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Forecast Error")
    ax4.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show() # If using Jupyter/Colab
    print(f"📊 Dashboard saved as {filename}")

def plot_feature_weights(model, feature_names, filename="output/feature_weights.png"):
    # model.cnn3 is the first layer. Shape: (out_channels, in_channels, kernel_size)
    weights = model.cnn3.weight.data.cpu().abs().mean(dim=(0, 2)).numpy()
    
    plt.figure(figsize=(10, 6))
    pd.Series(weights, index=feature_names).sort_values().plot(kind='barh', color='teal')
    plt.title("Mean Absolute Weights: CNN Layer 1")
    plt.xlabel("Importance (Weight Magnitude)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_attention_heatmap(model, val_loader, filename="output/attention_heatmap.png"):
    model.eval()
    device = next(model.parameters()).device
    
    # Grab a single batch
    mkt_data, sent_data, _, _ = next(iter(val_loader))
    mkt_data, sent_data = mkt_data.to(device), sent_data.to(device)
    
    with torch.no_grad():
        # Modify your model's forward or create a hook to get attention weights
        mkt_seq = mkt_data.transpose(1, 2)
        mkt_cnn = torch.cat((F.relu(model.cnn3(mkt_seq)), F.relu(model.cnn5(mkt_seq))), dim=1)
        mkt_cnn = mkt_cnn.transpose(1, 2)
        lstm_out, _ = model.lstm(mkt_cnn)
        _, weights = model.attention(lstm_out) # weights shape: (batch, seq_len, 1)

    plt.figure(figsize=(12, 4))
    avg_weights = weights.cpu().squeeze().mean(dim=0).reshape(1, -1)
    plt.imshow(avg_weights, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Score')
    plt.title("Where is the Model Looking? (Sequence Timeline)")
    plt.xlabel("Days in Past (Sequence Step)")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def calculate_permutation_importance(model, val_loader, mkt_cols, sent_cols):
    model.eval()
    device = next(model.parameters()).device
    criterion = nn.L1Loss()
    
    # 1. Base Score
    base_mae = 0
    with torch.no_grad():
        for m, s, t, _ in val_loader:
            outputs = model(m.to(device), s.to(device))
            base_mae += criterion(outputs, t.to(device)).item()
    base_mae /= len(val_loader)
    
    importances = {}

    # 2. Permute Market Features
    for i, col in enumerate(mkt_cols):
        perm_mae = 0
        with torch.no_grad():
            for m, s, t, _ in val_loader:
                m_p = m.clone()
                m_p[:, :, i] = m[torch.randperm(m.size(0)), :, i]
                perm_mae += criterion(model(m_p.to(device), s.to(device)), t.to(device)).item()
        importances[col] = (perm_mae / len(val_loader)) - base_mae

    # 3. Permute Sentiment Features
    for i, col in enumerate(sent_cols):
        perm_mae = 0
        with torch.no_grad():
            for m, s, t, _ in val_loader:
                s_p = s.clone()
                s_p[:, i] = s[torch.randperm(s.size(0)), i]
                perm_mae += criterion(model(m.to(device), s_p.to(device)), t.to(device)).item()
        importances[col] = (perm_mae / len(val_loader)) - base_mae
        
    return importances

def analyze_sentiment_impact(model, val_loader, logger=None):
    model.eval()
    device = next(model.parameters()).device
    
    # Grab a batch
    m, s, _, _ = next(iter(val_loader))
    m, s = m.to(device), s.to(device)
    
    # Baseline prediction
    with torch.no_grad():
        baseline_preds = model(m, s)[:, 3] # Close price prediction
        
        # Artificial "Bullish Sentiment" shock (+1 Standard Deviation)
        s_shocked = s.clone()
        s_shocked[:, 0] += 1.0 
        shock_preds = model(m, s_shocked)[:, 3]
        
    avg_impact = (shock_preds - baseline_preds).mean().item()
    if not logger:
        print(f"Sentiment Sensitivity: A +1SD move in Sentiment results in a {avg_impact*10000:.2f} bps move in predicted Gold price.")
    else:
        logger.info(f"Sentiment Sensitivity: A +1SD move in Sentiment results in a {avg_impact*10000:.2f} bps move in predicted Gold price.")
    return avg_impact

def plot_fusion_dominance(model, hidden_dim, sent_inner, filename="output/fusion_weights.png"):
    # Access the first layer of the fusion head
    fusion_weights = model.fc_fusion[0].weight.data.cpu().abs().mean(dim=0).numpy()
    
    # Split the weights into market-sourced and sentiment-sourced
    mkt_weight_sum = fusion_weights[:hidden_dim*2].mean()
    sent_weight_sum = fusion_weights[hidden_dim*2:].mean()
    
    plt.figure(figsize=(6, 6))
    plt.pie([mkt_weight_sum, sent_weight_sum], labels=['Market Branch', 'Sentiment Branch'], 
            autopct='%1.1f%%', colors=['#ffcc00', '#66b3ff'], startangle=140)
    plt.title("Model Reliance: Market vs. Sentiment")
    plt.savefig(filename)
    plt.close()

def plot_interpretability_report(model, val_loader, mkt_cols, sent_cols, hidden_dim, fold, filename="output/model_report.png"):
    fig = plt.subplots(2, 2, figsize=(20, 14))
    plt.suptitle(f"Model Interpretation Report - Fold {fold}", fontsize=20, fontweight='bold')
    
    # --- 1. Fusion Dominance (Pie Chart) ---
    ax1 = plt.subplot(2, 2, 1)
    fusion_weights = model.fc_fusion[0].weight.data.cpu().abs().mean(dim=0).numpy()
    mkt_influence = fusion_weights[:hidden_dim*2].mean()
    sent_influence = fusion_weights[hidden_dim*2:].mean()
    ax1.pie([mkt_influence, sent_influence], labels=['Market (LSTM)', 'Sentiment (MLP)'], 
            autopct='%1.1f%%', colors=['#FFD700', '#1E90FF'], explode=(0.05, 0), shadow=True)
    ax1.set_title("Source Reliance (Late Fusion Layer)")

    # --- 2. Attention Heatmap (Time Focus) ---
    ax2 = plt.subplot(2, 2, 2)
    model.eval()
    device = next(model.parameters()).device
    m, s, _, _ = next(iter(val_loader))
    with torch.no_grad():
        m_seq = m.to(device).transpose(1, 2)
        m_cnn = torch.cat((F.relu(model.cnn3(m_seq)), F.relu(model.cnn5(m_seq))), dim=1).transpose(1, 2)
        _, weights = model.attention(model.lstm(m_cnn)[0])
    avg_w = weights.cpu().squeeze().mean(dim=0).numpy().reshape(1, -1)
    im = ax2.imshow(avg_w, cmap='YlGnBu', aspect='auto')
    plt.colorbar(im, ax=ax2)
    ax2.set_title("Temporal Attention (Lookback Importance)")
    ax2.set_xlabel("Days in Past (0=Oldest, 29=Most Recent)")
    ax2.set_yticks([])

    # --- 3. Permutation Importance (All Features) ---
    ax3 = plt.subplot(2, 2, 3)
    importances = calculate_permutation_importance(model, val_loader, mkt_cols, sent_cols)
    imp_series = pd.Series(importances).sort_values()
    imp_series.plot(kind='barh', color='teal', ax=ax3)
    ax3.set_title("Feature Importance (Permutation Impact on MAE)")
    ax3.set_xlabel("Error Increase when Shuffled")

    # --- 4. Sentiment Sensitivity (Shock Test) ---
    ax4 = plt.subplot(2, 2, 4)
    # Check sensitivity for each sentiment dimension
    sensitivities = []
    with torch.no_grad():
        base = model(m.to(device), s.to(device))[:, 3]
        for i in range(len(sent_cols)):
            s_shock = s.clone().to(device)
            s_shock[:, i] += 1.0 # 1 Standard Deviation Shock
            shock_pred = model(m.to(device), s_shock)[:, 3]
            sensitivities.append((shock_pred - base).mean().item() * 10000) # In Basis Points
    
    ax4.bar(sent_cols, sensitivities, color=['green' if x > 0 else 'red' for x in sensitivities])
    ax4.set_title("Sentiment Sensitivity (1-SD Shock)")
    ax4.set_ylabel("Predicted Price Move (Basis Points)")
    ax4.axhline(0, color='black', linewidth=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename.replace(".png", f"_fold{fold}.png"))
    plt.close()

def main():
    os.makedirs("output", exist_ok=True)

    # Load Dataset
    dataset = dataset_former.MarketDataset.load()
    
    # Hyperparameters
    mkt_feat_dim = dataset.market_values.shape[1] # 13 features
    sent_feat_dim = dataset.sentiment_values.shape[1] # 2 features
    hidden_dim = 128
    batch_size = 64
    epochs = 100
    learning_rate = 0.001
    
    logger = setup_logger()

    # Initialize Model
    model = GoldPredictor(mkt_feat_dim, sent_feat_dim, hidden_dim)
    
    # Cross-Validation Folds
    fold_num = 0
    for train_loader, val_loader, test_loader, test_year in dataset.get_loaders(training_setting="expanding_window", batch_size=batch_size):

        fold_num += 1

        logger.info(f"Starting Fold {fold_num} | Test Year: {test_year}")
        
        # Train Model
        history, optimizer = train_model(model, train_loader, val_loader, epochs=epochs, lr=learning_rate, logger=logger, debug=False)
        
        # Evaluate on Test Set
        all_preds, all_actuals, performance = evaluate_test_set(model, test_loader,  logger=logger, debug=False)
        
        logger.info(f"Completed Fold {fold_num}\n")

        save_predictions_csv(all_preds, all_actuals)
        model.save(optimizer, fold=fold_num, performance=performance)


        # === FEATURE ANALYSIS ===
        mkt_cols = ['Log_Ret_Open', 'Log_Ret_High', 'Log_Ret_Low', 'Log_Ret_Close', 'Log_Ret_Vol', 'RSI_Z', 'MACD_Z', 'MACD_Sig_Z',
        'SMA_20_Rel', 'SMA_50_Rel', 'BB_Low_Rel', 'BB_High_Rel', 'ATR_Pct'] # Add your actual 13 column names

        sent_cols = ['Sentiment_Score', 'Sentiment_Volume'] # Add your actual sentiment column names

        # 1. Plot Weights
        plot_feature_weights(model, mkt_cols, filename=f"output/fold_{fold_num}_weights.png")

        # 2. Plot Attention (Which past days the model values most)
        plot_attention_heatmap(model, val_loader, filename=f"output/fold_{fold_num}_attention.png")

        # 3. Run Permutation Test (How much MAE drops when a feature is "broken")
        p_importance = calculate_permutation_importance(model, val_loader, mkt_cols, sent_cols)
        
        # Sort and log results
        sorted_imp = sorted(p_importance.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"\n--- Permutation Importance Fold {fold_num} ---")
        for feat, imp in sorted_imp:
            # We multiply by 1000 to make the small log-return errors easier to read
            logger.info(f"{feat:<15}: {imp*1000:.6f} (scaled x1000)")

        plot_fusion_dominance(model, hidden_dim, sent_inner=32)

        # 3. Log the sensitivity
        analyze_sentiment_impact(model, val_loader, logger=logger)

        # Total_report
        plot_interpretability_report(
            model=model, 
            val_loader=val_loader, 
            mkt_cols=mkt_cols, 
            sent_cols=sent_cols, 
            hidden_dim=hidden_dim, 
            fold=fold_num)

if __name__ == "__main__":
    main()