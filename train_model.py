import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

class LogOHLCLoss(nn.Module):
    def __init__(self, penalty_weight=10.0):
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
        
        # --- Market Branch (CNN + Bi-LSTM) ---
        self.cnn = nn.Conv1d(in_channels=mkt_feat_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2) # *2 for Bidirectional
        
        # --- Sentiment Branch ---
        self.sent_mlp = nn.Sequential(
            nn.Linear(sent_feat_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # --- Fusion Head ---
        # (hidden_dim * 2 from Bi-LSTM) + 16 from Sentiment
        self.fc_fusion = nn.Sequential(
            nn.Linear((hidden_dim * 2) + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4) # Predicts [Open, High, Low, Close]
        )

    def forward(self, mkt_seq, sent_vec):
        # mkt_seq: (batch, seq_len, mkt_feat_dim)
        # CNN expects (batch, channels, seq_len)
        mkt_seq = mkt_seq.transpose(1, 2)
        mkt_cnn = F.relu(self.cnn(mkt_seq))
        
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

def calculate_trading_metrics(preds, targets, epsilon_pct=0.002):
    """
    preds/targets: [Batch, 4] -> (Open, High, Low, Close)
    epsilon_pct: 0.002 means 0.2% tolerance
    """
    # 1. Range Coverage: Is Actual Close between Predicted High and Predicted Low?
    # Actual Close is index 3; Predicted High is index 1, Predicted Low is index 2
    covered = (targets[:, 3] <= preds[:, 1]) & (targets[:, 3] >= preds[:, 2])
    range_coverage_acc = covered.astype(float).mean()

    # 2. Epsilon Accuracy: Is Predicted Close within X% of Actual Close?
    close_error_pct = np.abs(preds[:, 3] - targets[:, 3]) / targets[:, 3]
    epsilon_hit = close_error_pct <= epsilon_pct
    epsilon_acc = epsilon_hit.astype(float).mean()

    # 3. Directional Accuracy: Did we predict the candle color (Red/Green) correctly?
    actual_dir = np.sign(targets[:, 3] - targets[:, 0]) # Close - Open
    pred_dir = np.sign(preds[:, 3] - preds[:, 0])
    dir_acc = (actual_dir == pred_dir).astype(float).mean()

    # 4. MAE and MAPE metrics
    mae = np.mean(np.abs(preds - targets))
    # small epsilon to avoid division by zero
    mape = np.mean(np.abs((targets - preds) / (targets + 1e-10))) * 100

    return {
        "range_coverage": range_coverage_acc,
        "epsilon_accuracy": epsilon_acc,
        "directional_accuracy": dir_acc,
        "mae": mae,
        "mape": mape
    }


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """
    Proposed hyperparameters in the paper 'Implementation of Long Short-Term Memory for Gold Prices Forecasting'
    are: epoch = 100, LR = 0.01 with Adam optimizer, and expanding window.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = LogOHLCLoss(penalty_weight=10.0) # Robust to financial outliers
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for mkt_data, sent_data, targets in train_loader:
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

        with torch.no_grad():
            for mkt_data, sent_data, targets in val_loader:
                mkt_data, sent_data, targets = mkt_data.to(device), sent_data.to(device), targets.to(device)
                outputs = model(mkt_data, sent_data)
                v_loss = criterion(outputs, targets)
                val_losses.append(v_loss.item())

                all_val_preds.append(outputs.cpu())
                all_val_targets.append(targets.cpu())
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)

        all_val_preds = torch.cat(all_val_preds, dim=0).numpy()
        all_val_targets = torch.cat(all_val_targets, dim=0).numpy()
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        

        # Verbose logging every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Calculate trading metrics on validation set
        epsilon_pct = 0.002
        val_metrics = calculate_trading_metrics(all_val_preds, all_val_targets, epsilon_pct=epsilon_pct)

        print(f"Epoch {epoch}:")
        print(f"  - Range Coverage: {val_metrics['range_coverage']:.2%}")
        print(f"  - Epsilon Hit ({epsilon_pct*100}%): {val_metrics['epsilon_accuracy']:.2%}")
        print(f"  - Directional Acc: {val_metrics['directional_accuracy']:.2%}")
            
    return history

def evaluate_test_set(model, test_loader, scaler_target):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for mkt_data, sent_data, targets in test_loader:
            mkt_data, sent_data, targets = mkt_data.to(device), sent_data.to(device), targets.to(device)
            
            outputs = model(mkt_data, sent_data)
            
            # Inverse transform to get back to real dollar prices if you scaled them
            # actuals = scaler_target.inverse_transform(targets.cpu().numpy())
            # preds = scaler_target.inverse_transform(outputs.cpu().numpy())
            
            all_preds.append(outputs.cpu())
            all_actuals.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_actuals = torch.cat(all_actuals, dim=0).numpy()
     
    # --- Metric 1: RMSE (Regression Error) ---
    mse = np.mean((all_preds - all_actuals)**2)
    rmse = np.sqrt(mse)

    
    print(f"--- Final Test Results ---")
    print(f"Test RMSE: {rmse:.4f}")
    
    # --- Metric 2: Trading Metrics ---
    epsilon_pct = 0.002
    val_metrics = calculate_trading_metrics(all_preds, all_actuals, epsilon_pct=epsilon_pct)

    print(f"  - Range Coverage: {val_metrics['range_coverage']:.2%}")
    print(f"  - Epsilon Hit ({epsilon_pct*100}%): {val_metrics['epsilon_accuracy']:.2%}")
    print(f"  - Directional Acc: {val_metrics['directional_accuracy']:.2%}")
    
    return all_preds, all_actuals

def setup_logger(fold_num):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(f"gold_model_fold_{fold_num}.log"),
            logging.StreamHandler() # Still prints to console
        ]
    )
    return logging.getLogger()