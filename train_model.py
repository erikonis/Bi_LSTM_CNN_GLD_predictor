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
import tensorflow as tf

from abc import ABC, abstractmethod

import logging 

overwrite = False

output_folder = "output"
if overwrite:
        os.makedirs(output_folder, exist_ok=True)
else:
    i = 1
    base_folder = output_folder
    while os.path.exists(output_folder):
        output_folder = f"{base_folder}_{i}"
        i += 1
    os.makedirs(output_folder, exist_ok=True)


class LogOHLCLoss(nn.Module):
    def __init__(self, penalty_weight=0.5):
        super(LogOHLCLoss, self).__init__()
        self.mse = nn.HuberLoss() 
        self.penalty_weight = penalty_weight

    def forward(self, pred, target):        
        # 1. Base Regression Loss (Standard Huber)
        base_loss = self.mse(pred, target)
        
        dim = pred.ndim

        if pred.ndim == 4:
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
        else:
            total_penalty = 0

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

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class PredictorSkeleton(ABC, nn.Module):
    def __init__(self):
        super(PredictorSkeleton, self).__init__()
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def save(self, optimizer, information, performance, path=f"{output_folder}/gold_model_latest.pth"):
        """Saves the model state and training context."""
        state = {
            'information': information,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'performance' :  performance
        }
        torch.save(state, path)
        print(f"Model saved to {path}")

    def load_checkpoint(self, path=None, optimizer=None):
        if path is None:
            path = f"{output_folder}/gold_model_latest.pth"
            
        if not os.path.exists(path):
            print(f"No saved model found at {path}")
            return None
            
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from Fold {checkpoint.get('fold', 'Unknown')}")
        return checkpoint.get('fold', 0)

class GoldPredictor(PredictorSkeleton):
    # Deprecicated
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
        nn.Dropout(0.1))
        
        # --- Fusion Head ---
        # (hidden_dim * 2 from Bi-LSTM) + 16 from Sentiment
        self.fc_fusion = nn.Sequential(
            nn.Linear((hidden_dim * 2) + sent_inner, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
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

class GoldPredictor2(PredictorSkeleton):
    def __init__(self, mkt_feat_dim, sent_feat_dim, hidden_dim=64, target_dim = 4):
        super(GoldPredictor2, self).__init__()

        total_feat_dim = mkt_feat_dim + sent_feat_dim

        # Update your CNNs to accept the combined dimension
        #self.cnn3 = nn.Conv1d(total_feat_dim, int(np.ceil(hidden_dim/2)), kernel_size=3, padding=1)
        self.cnn5 = nn.Conv1d(total_feat_dim, hidden_dim, kernel_size=5, padding=2)
        
        # --- Market Branch (CNN + Bi-LSTM) ---
        #self.cnn = nn.Conv1d(in_channels=mkt_feat_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2) # *2 for Bidirectional
        
        # --- Fusion Head ---
        # (hidden_dim * 2 from Bi-LSTM)
        self.fc_fusion = nn.Sequential(
            nn.Linear((hidden_dim * 2), target_dim*8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(target_dim*8, target_dim) # Predicts [Open, High, Low, Close]
        )

    def forward(self, mkt_seq, sent_vec):
        # mkt_seq: (batch, seq_len, mkt_feat_dim)
        # CNN expects (batch, channels, seq_len)
        combined = torch.cat((mkt_seq, sent_vec), dim=2)
        feat_vec = combined.transpose(1, 2)
        #mkt_cnn3 = F.relu(self.cnn3(feat_vec))
        mkt_cnn5 = F.relu(self.cnn5(feat_vec))
        
        # Concatenate along the channel dimension
        # mkt_cnn = torch.cat((mkt_cnn3, mkt_cnn5), dim=1)
        
        # Back to (batch, seq_len, hidden_dim) for LSTM
        #mkt_cnn = mkt_cnn.transpose(1, 2)
        mkt_cnn = mkt_cnn5.transpose(1, 2)
        lstm_out, _ = self.lstm(mkt_cnn)
        
        # Attention pooling
        mkt_context, _ = self.attention(lstm_out)
        
        # Early Fusion
        output = self.fc_fusion(mkt_context)
        return output

def train_model(model, device, train_loader, val_loader, market_cols, sent_cols, epochs=50, lr=0.0005, logger=None, debug=False, early_stop = False):
    """
    Proposed hyperparameters in the paper 'Implementation of Long Short-Term Memory for Gold Prices Forecasting'
    are: epoch = 100, LR = 0.01 with Adam optimizer, and expanding window.
    """
    model.to(device)

    feature_names = market_cols + sent_cols
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Allow for adaptive learning rate:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    if early_stop:
        stopper = EarlyStopping(patience=10)
    criterion = LogOHLCLoss() # Robust to financial outliers
    
    history = {'train_loss': [], 'val_loss': []}
    weight_history = {name: [] for name in feature_names}

    best_val_loss = float('inf')

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

            # Fetching model weights:

            #mkt_w3 = model.cnn3.weight.abs().mean(dim=(0, 2)) 
            mkt_w5 = model.cnn5.weight.abs().mean(dim=(0, 2))
            #mkt_weights = (mkt_w3 + mkt_w5) / 2 # Average of both CNN paths
            mkt_weights = mkt_w5
            all_current_weights = mkt_weights.cpu().numpy()

            for i, name in enumerate(feature_names):
                weight_history[name].append(all_current_weights[i])

            # Validation loop
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
        
        
        current_lr = scheduler.get_last_lr()[0]

        # Verbose logging every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            if logger:
                logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
                logger.info(f"Current LR: {current_lr:.6f}")

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

        # Early stop logic
        if early_stop:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                logger.info(f"Best model updated at Epoch {epoch}")

            stopper(avg_val_loss)
            if stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    
    if early_stop:
        model.load_state_dict(best_model_state)
        
    return history, optimizer, weight_history

def feature_engineering(train_loader, val_loader, mkt_cols, sent_cols, device, logger, hidden_dim=64, target_dim=4, threshold=0):
    # --- PHASE 1: Signal Discovery (Quick Run) ---
    logger.info("PHASE 1: Identifying feature signals...")
    
    # Initialize a temporary model to test importance
    mkt_dim = len(mkt_cols) # Initial count
    sent_dim = len(sent_cols)
    temp_model = GoldPredictor2(mkt_dim, sent_dim, hidden_dim=hidden_dim, target_dim = target_dim).to(device)

    train_model(temp_model, device, train_loader, val_loader,  mkt_cols, sent_cols, epochs=50, lr=0.001, logger=logger, early_stop=False)
    
    # Calculate initial importance using your existing function
    importances = calculate_permutation_importance(temp_model, val_loader, mkt_cols, sent_cols, fold="Discovery")
    
    # Determine which features to keep/drop
    to_keep, to_drop = identify_low_importance_features(importances, threshold=threshold)
    logger.info(f"Dropping noise features: {to_drop}")
    logger.info(f"Training with signal features: {to_keep}")

    # Update global feature list for the final model
    filtered_mkt = [f for f in to_keep if f in mkt_cols]
    filtered_sent = [f for f in to_keep if f in sent_cols]

    print(":::::::::::::::::::: Filtered features ::::::::::::::::::::")
    print(f"To keep features: {to_keep}")
    print(f"Dropping noise features: {to_drop}")
    print(f"Total market filtered features: {filtered_mkt}")
    print(f"Total sentiment filtered features: {filtered_sent}")
    
    # --- PHASE 2: Optimized Training ---
    logger.info("PHASE 2: Starting optimized training with Early Stopping...")
    final_model = GoldPredictor2(len(filtered_mkt), len(filtered_sent), hidden_dim=hidden_dim, target_dim=target_dim).to(device)
    
    return final_model, filtered_mkt, filtered_sent

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
    find_best_threshold(all_preds, all_actuals, logger)

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
    
        if results['final_value'] > (results['buy_and_hold'][-1]):
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
        logger.info(f"Strategy Final Value: ${results['final_value']:.2f}")
        logger.info(f"Buy & Hold Final Value: ${results['buy_and_hold'][-1]:.2f}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
        if results['final_value'] > results['buy_and_hold'][-1]:
            logger.info("STRATEGY OUTPERFORMED MARKET")
        else:
            logger.info("MARKET OUTPERFORMED STRATEGY")
    

    #plotting
    plot_model_results(all_preds, all_actuals, results)

    return all_preds, all_actuals, val_metrics

def identify_low_importance_features(importances, threshold=0.0001):
    """
    Returns a list of features to keep and a list of features to drop.
    """
    imp_series = pd.Series(importances)
    print(imp_series)
    to_keep = imp_series[imp_series > threshold].index.tolist()
    to_drop = imp_series[imp_series <= threshold].index.tolist()
    
    return to_keep, to_drop

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

def setup_logger(filename : str = f"{output_folder}/model_train.log"):
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
    :return: dict with metrics: range_coverage, epsilon_accuracy, directional_accuracy, mae, mape, max_pred_move, avg_pred_move. Note that that range coverage might return a `np.nan` if the target does not contain [H, L, C].
    """

    # Adapting epsilon to normalization
    epsilon_pct = epsilon_pct*100

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

    #Directional Accuracy 
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
        if last_close_prices is not None:
            last_close = last_close_prices[:, 3] 

            # Convert predicted log-returns back to USD
            pred_high_usd  = last_close * np.exp(preds[:, h_idx]/100)  # High
            pred_low_usd   = last_close * np.exp(preds[:, l_idx]/100)  # Low
            pred_close_usd = last_close * np.exp(preds[:, c_idx]/100)  # Close

            # Convert target log-returns back to USD
            actual_close_usd = last_close * np.exp(targets[:, c_idx] /100)  # Close

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
            covered = (targets[:, c_idx] <= preds[:, h_idx]) & (targets[:, c_idx] >= preds[:, l_idx])
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
        "avg_pred_move": avg_pred_move
    }

def backtest_with_costs(preds, targets, initial_capital=1000.0, threshold=0.01, fee=0.0005):
    """
    Simulates trading with compounding returns and transaction fees.
    """
    c_id = -1

    # 1. Generate Signals (Column -1 is 'Close')
    signals = np.zeros(len(preds))
    signals[preds[:, c_id] > threshold] = 1
    signals[preds[:, c_id] < -threshold] = -1
    
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
        daily_multiplier = np.exp(current_position * targets[i, c_id]/100) 
        
        equity.append(current_equity * daily_multiplier)
    
    # Convert to array and remove the seed value for analysis
    equity_curve = np.array(equity[1:])
    buy_and_hold = initial_capital * np.exp(np.cumsum(targets[:, c_id]) /100)
    
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

def find_best_threshold(preds, targets, logger=None):
    print("\n--- Threshold Sensitivity Analysis ---")
    print(f"{'Threshold':<12} | {'Trades':<8} | {'Final Value':<12} | {'Sharpe'}")
    print("-" * 50)
    
    if not logger:
        # Test thresholds from 0.05% to 0.5%
        for t in [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005]:
            res = backtest_with_costs(preds, targets, threshold=t, fee=0.0003)
            print(f"{t*100:>9.2f}% | {res['num_trades']:>8} | ${res['final_value']:>10.2f} | {res['sharpe_ratio']:>6.2f}")
    else:
        # Test thresholds from 0.05% to 0.5%
        for t in [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005]:
            res = backtest_with_costs(preds, targets, threshold=t, fee=0.0003)
            logger.info(f"{t*100:>9.2f}% | {res['num_trades']:>8} | ${res['final_value']:>10.2f} | {res['sharpe_ratio']:>6.2f}")

def save_predictions_csv(preds, targets, filename=f"{output_folder}/gold_predictions.csv"):
    df = pd.DataFrame({
        'Predicted_Return': preds.flatten()/100,
        'Actual_Return': targets.flatten()/100
    })
    # Add a column to see the error magnitude
    df['Error'] = df['Actual_Return'] - df['Predicted_Return']
    
    df.to_csv(filename, index=False)
    print(f"saved {len(df)} predictions to {filename}")

def plot_model_results(preds, targets, backtest_results, filename=f"{output_folder}/performance_summary.png", overwrite = False):
    """
    4-Panel Dashboard: Equity, Drawdown, Prediction Scatter, and Error Histogram.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    c_id = -1

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
    ax3.scatter(targets[:, c_id]/100, preds[:, c_id]/100, alpha=0.5, color='#1f77b4', s=15)
    # Identity line (where predictions = reality)
    all_data = np.concatenate([targets[:, c_id]/100, preds[:, c_id]/100])
    low, high = all_data.min(), all_data.max()
    ax3.plot([low, high], [low, high], 'r--', alpha=0.8)
    ax3.set_title("Prediction vs. Reality Magnitude", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Actual Log-Return")
    ax3.set_ylabel("Predicted Log-Return")
    ax3.grid(alpha=0.2)

    # --- 4. Residuals (Bias Check) ---
    ax4 = axes[3]
    errors = (targets[:, c_id] - preds[:, c_id]) / 100
    ax4.hist(errors, bins=60, color='seagreen', alpha=0.7, edgecolor='white')
    ax4.axvline(0, color='black', linestyle='-', linewidth=1)
    ax4.set_title("Error Distribution (Zero-Bias Check)", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Forecast Error")
    ax4.grid(alpha=0.2)
    
    plt.tight_layout()

    i = 1
    while os.path.isfile(filename) and not overwrite:
        parts = filename.split(".")
        filename = parts[0] + f"_{i}" + f".{parts[1]}"
        i += 1
    
    
    plt.savefig(filename, dpi=150)
    plt.show() # If using Jupyter/Colab
    print(f"Dashboard saved as {filename}")

def plot_maw_progression(weight_history, fold_num, filename=f"{output_folder}/maw_progression.png"):
    plt.figure(figsize=(10, 6))
    for feature_name, values in weight_history.items():
        plt.plot(values, label=feature_name)
    
    plt.title(f"Feature Weight Progression - Fold {fold_num}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Weight")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_feature_weights(model, feature_names, filename=f"{output_folder}/feature_weights.png"):
    # model.cnn3 is the first layer. Shape: (out_channels, in_channels, kernel_size)
    
    weights = model.cnn5.weight.data.cpu().abs().mean(dim=(0, 2)).numpy()
    
    plt.figure(figsize=(10, 6))
    pd.Series(weights, index=feature_names).sort_values().plot(kind='barh', color='teal')
    plt.title("Mean Absolute Weights: CNN Layer 1")
    plt.xlabel("Importance (Weight Magnitude)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_attention_heatmap(model, val_loader, filename=f"{output_folder}/attention_heatmap.png"):
    model.eval()
    device = next(model.parameters()).device
    
    # Grab a single batch
    mkt_data, sent_data, _, _ = next(iter(val_loader))
    mkt_data, sent_data = mkt_data.to(device), sent_data.to(device)
    
    with torch.no_grad():
        # Modify your model's forward or create a hook to get attention weights
        combined_seq = torch.cat((mkt_data, sent_data), dim=2)

        combined_seq = combined_seq.transpose(1, 2)
        #mkt_cnn = torch.cat((F.relu(model.cnn3(combined_seq)), F.relu(model.cnn5(combined_seq))), dim=1)
        mkt_cnn = F.relu(model.cnn5(combined_seq))
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

def plot_feature_time_heatmap(model, val_loader, feature_names, filename=f"{output_folder}/saliency_heatmap.png"):
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
    all_saliency = torch.cat([mkt_data.grad.abs(), sent_data.grad.abs()], dim=2).mean(dim=0).cpu().numpy()

    # 5. Plotting
    plt.figure(figsize=(14, 8))
    # Transpose to get Features on Y and Time on X
    plt.imshow(all_saliency.T, cmap='hot', aspect='auto', interpolation='nearest')
    
    plt.colorbar(label='Feature Importance (Absolute Gradient)')
    plt.title("2D Feature-Time Importance (Saliency Map)")
    plt.ylabel("Features")
    plt.xlabel("Days in Sequence (0 = Oldest, 29 = Today)")
    
    # Set Y-ticks to feature names
    plt.yticks(range(len(feature_names)), feature_names)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def calculate_permutation_importance(model, val_loader, mkt_cols, sent_cols, fold, filename=f"{output_folder}/permutation_importance.png"):
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
            s_perm[:, :, i - m_count] = s_orig[torch.randperm(s_orig.size(0)), :, i - m_count]
        
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
        imp_series.plot(kind='barh', color='teal', ax=ax)
        
        plt.title(f"Feature Importance - Fold {fold}\n(Permutation Impact on MAE)")
        plt.xlabel("Error Increase when Shuffled")
        
        # 4. Optional: Add a grid for better scannability
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(filename.replace(".png", f"_fold{fold}.png"))
        plt.close()

    return importances

def plot_interpretability_report(model, val_loader, mkt_cols, sent_cols, fold, filename=f"{output_folder}/model_report.png"):
    # 1. Setup
    all_feature_names = mkt_cols + sent_cols
    fig, axs = plt.subplots(2, 1, figsize=(16, 12))
    plt.suptitle(f"Model Interpretation Report - Fold {fold}", fontsize=20, fontweight='bold')

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
        #m_cnn3 = F.relu(model.cnn3(feat_vec))
        m_cnn5 = F.relu(model.cnn5(feat_vec))
        #m_cnn = torch.cat((m_cnn3, m_cnn5), dim=1) # Concatenate filters
        m_cnn = m_cnn5

        # 4. LSTM + Attention
        m_cnn = m_cnn.transpose(1, 2)
        lstm_out, _ = model.lstm(m_cnn)
        _, weights = model.attention(lstm_out)
        
    avg_w = weights.cpu().squeeze().mean(dim=0).numpy().reshape(1, -1)
    im = ax2.imshow(avg_w, cmap='YlGnBu', aspect='auto')
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
    colors = ['orange' if x in sent_cols else 'teal' for x in imp_series.index]
    
    imp_series.plot(kind='barh', color=colors, ax=ax3)
    ax3.set_title("Feature Importance: Shuffling Impact on MAE")
    ax3.set_xlabel("Error Increase (BPS scaled x100)")
    ax3.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename.replace(".png", f"_fold{fold}.png"))
    plt.close()
    print(f"Report for Fold {fold} saved to {filename}")

def main():
    global output_folder

    # Load Dataset
    dataset = dataset_former.MarketDataset2.load()

    # ========= HYPER PARAMS ===========
    automatic_feature_engineering = False
    early_stop = True

    # ['Target_Open', 'Target_High', 'Target_Low', 'Target_Close']
    targets = ['Target_Close']
    not_considered_feat = [] #['SMA_20_Rel', 'ATR_Pct', 'MACD_Sig_Z', 'SMA_50_Rel']
    mkt_cols = [f for f in dataset._market_cols if f not in not_considered_feat]
    sent_cols = [f for f in dataset._sent_cols if f not in not_considered_feat]

    dataset.set_active_cols(new_market_cols=mkt_cols, new_sent_cols=sent_cols, new_target_cols=targets)

    mkt_feat_dim = len(mkt_cols) # 13 features
    sent_feat_dim = len(sent_cols) # 2 features

    target_dim = len(targets)
    hidden_dim = 32
    batch_size = 64
    epochs = 100
    learning_rate = 0.0005
    feature_threshold = 0
    output_folder = "output"
    data_split = "expanding_window" 
    #data_split = "sliding_window"
    overwrite = False

    feature_names = mkt_cols + sent_cols
    #-----------------------------------------

    logger = setup_logger()

    logger.info("Hyperparameters:")
    logger.info(f"  - Automatic Feature Engineering: {automatic_feature_engineering}")
    logger.info(f"  - Early Stopping: {early_stop}")
    logger.info(f"Targets: {targets}")
    logger.info(f"  - Market Features: {mkt_cols}")
    logger.info(f"  - Sentiment Features: {sent_cols}")
    logger.info(f"  - Hidden Dim: {hidden_dim}")
    logger.info(f"  - Batch Size: {batch_size}")
    logger.info(f"  - Epochs: {epochs}")
    logger.info(f"  - Learning Rate: {learning_rate}")
    logger.info(f"  - Feature Threshold: {feature_threshold}")
    logger.info(f"  - Output Folder: {output_folder}")
    logger.info(f"  - Data Split: {data_split}")
    logger.info(f"  - Architecture 2: CNN + LSTM + Attention")
    logger.info(f"  - Overwrite: {overwrite}")
    logger.info("-" * 50)
    logger.info(f"Additional information:")
    logger.info(f"  - cnn5 only")
    
    # Initialize Device and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if automatic_feature_engineering:
        for train, val, test, test_year in dataset.get_loaders(training_setting=data_split, batch_size=batch_size):
            last_train = train
            last_val = val
        model, mkt_cols, sent_cols = feature_engineering(last_train, last_val, mkt_cols, sent_cols, device, logger, hidden_dim=hidden_dim, target_dim=target_dim, threshold=feature_threshold)
        dataset.set_active_cols(mkt_cols, sent_cols)
        feature_names = mkt_cols + sent_cols
    else:
        model = GoldPredictor2(mkt_feat_dim, sent_feat_dim, hidden_dim, target_dim)

    # Weights collection
    weights = {name: [] for name in feature_names}

    # Cross-Validation Folds
    fold_num = 0
    for train_loader, val_loader, test_loader, test_year in dataset.get_loaders(training_setting="expanding_window", batch_size=batch_size):

        fold_num += 1

        logger.info(f"\n\n ==============> Starting Fold {fold_num} | Test Year: {test_year}")
        
        # Train Model
        history, optimizer, fold_weights = train_model(model, device, train_loader, val_loader, market_cols=mkt_cols, sent_cols=sent_cols, epochs=epochs, lr=learning_rate, logger=logger, early_stop=early_stop)
        
        for name in feature_names:
            weights[name].extend(fold_weights[name])


        #                                        TODO
        # Evaluate on Test Set
        all_preds, all_actuals, performance = evaluate_test_set(model, test_loader,  logger=logger)
        
        logger.info(f"Completed Fold {fold_num}\n")

        save_predictions_csv(all_preds, all_actuals)

        # === FEATURE ANALYSIS ===

        # 3. Run Permutation Test (How much MAE drops when a feature is "broken")
        p_importance = calculate_permutation_importance(model, val_loader, mkt_cols, sent_cols, fold_num)
        
        # Sort and log results
        sorted_imp = sorted(p_importance.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"\n--- Permutation Importance Fold {fold_num} ---")
        for feat, imp in sorted_imp:
            # We multiply by 1000 to make the small log-return errors easier to read
            logger.info(f"{feat:<15}: {imp*1000:.6f} (scaled x1000)")

        # Total_report
        plot_interpretability_report(
            model=model, 
            val_loader=val_loader, 
            mkt_cols=mkt_cols,
            sent_cols=sent_cols,
            fold=fold_num)
        
        plot_feature_time_heatmap(model, val_loader, feature_names=feature_names, filename=f"{output_folder}/fold_{fold_num}_saliency_heatmap.png")

    # After folding
    plot_maw_progression(weights, fold_num="1-5")

    model.save(optimizer, information="After folds", performance=performance)

if __name__ == "__main__":
    main()