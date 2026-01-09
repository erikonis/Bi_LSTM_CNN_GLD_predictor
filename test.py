"""
test.py
--------
Self-contained pipeline to train a CNN-LSTM-Attention model predicting next-day Close price
(and an uncertainty estimate). Uses the repository's `dataset_former.MarketDataset.load_data`
function to load OHLCV + sentiment JSON, computes simple technical indicators,
applies min-max scaling, trains on historical years, plots diagnostics, and runs
a simple backtest simulation.

Usage (from repository root):
    python test.py

Notes & limitations:
- This is an experimental script. It does NOT provide financial advice.
- Results are for research / educational purposes only.
- The script expects the files:
    hist_data/stock_data/GLD.csv and News/finBERT_scores.json
  to exist (these are the defaults used by `dataset_former.MarketDataset.load_data`).
- Only `test.py` is modified as requested.

"""

import os
import math
import json
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from Constants import ColNames
import dataset_former

# -------------------------------
# Utilities: technicals + scaling
# -------------------------------

def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few lightweight technical indicators used as features."""
    # SMA ratios
    df[ColNames.SMA_20] = df[ColNames.CLOSE] / df[ColNames.CLOSE].rolling(20).mean()
    df[ColNames.SMA_50] = df[ColNames.CLOSE] / df[ColNames.CLOSE].rolling(50).mean()
    # RSI approx (14)
    delta = df[ColNames.CLOSE].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    df[ColNames.RSI] = (up / (up + down)).fillna(0)
    # ATR approx
    tr = np.maximum(df[ColNames.HIGH] - df[ColNames.LOW],
                    np.maximum((df[ColNames.HIGH] - df[ColNames.CLOSE].shift(1)).abs(),
                               (df[ColNames.LOW] - df[ColNames.CLOSE].shift(1)).abs()))
    df[ColNames.ATR] = tr.rolling(14).mean()
    # Fillna (use recommended methods)
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df


class MinMaxScalerDict:
    """Simple per-column min-max scaler that stores params for inverse transform."""
    def __init__(self):
        self.params = {}

    def fit_transform(self, series: pd.Series, name: str) -> np.ndarray:
        arr = np.asarray(series, dtype=float)
        minv = np.nanmin(arr)
        maxv = np.nanmax(arr)
        span = maxv - minv if (maxv - minv) != 0 else 1.0
        self.params[name] = (float(minv), float(maxv))
        return ((arr - minv) / span).astype(np.float32)

    def transform(self, arr: np.ndarray, name: str) -> np.ndarray:
        minv, maxv = self.params[name]
        span = maxv - minv if (maxv - minv) != 0 else 1.0
        return ((np.asarray(arr, dtype=float) - minv) / span).astype(np.float32)

    def inverse(self, arr: np.ndarray, name: str) -> np.ndarray:
        if name not in self.params:
            raise KeyError(name)
        minv, maxv = self.params[name]
        span = maxv - minv if (maxv - minv) != 0 else 1.0
        return (np.asarray(arr, dtype=float) * span + minv)


# -------------------------------
# PyTorch dataset for sequences
# -------------------------------
class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], target_col: str = ColNames.CLOSE,
                 seq_len: int = 30, scaler: MinMaxScalerDict = None):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.scaler = scaler or MinMaxScalerDict()

        # prepare matrix
        X = df[feature_cols].astype(float)
        y = df[target_col].astype(float)

        # fit scaler for each feature if not already present
        for c in feature_cols:
            if c not in self.scaler.params:
                self.scaler.fit_transform(X[c].values, c)
        if target_col not in self.scaler.params:
            self.scaler.fit_transform(y.values, target_col)

        Xs = np.stack([self.scaler.transform(X[c].values, c) for c in feature_cols], axis=1)
        Ys = self.scaler.transform(y.values, target_col)

        # Build sequences
        self.X_seq = []
        self.Y = []
        self.last_close = []  # store last day's raw close for backtest denorm reference
        raw_close = df[ColNames.CLOSE].values.astype(float)
        for i in range(seq_len, len(df)-1):
            # use window up to day i (inclusive) to predict day i+1 close
            window = Xs[i - seq_len + 1: i + 1]
            self.X_seq.append(window)
            self.Y.append(Ys[i + 1])
            # last close available to denormalize strategy
            self.last_close.append(raw_close[i])

        self.X_seq = np.asarray(self.X_seq, dtype=np.float32)
        self.Y = np.asarray(self.Y, dtype=np.float32)
        self.last_close = np.asarray(self.last_close, dtype=np.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X_seq[idx]), torch.from_numpy(np.asarray(self.last_close[idx], dtype=np.float32)), torch.from_numpy(np.asarray(self.Y[idx], dtype=np.float32))


# -------------------------------
# Model: CNN -> BiLSTM -> Attention -> FC
# Outputs: mean and log-variance for heteroscedastic loss
# -------------------------------
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        w = torch.tanh(self.attn(x))  # (b, seq, 1)
        w = torch.softmax(w, dim=1)
        out = torch.sum(x * w, dim=1)
        return out


class CNNLSTMAttention(nn.Module):
    def __init__(self, n_features, seq_len, hidden_dim=128):
        super().__init__()
        # conv over time; expect input shape (batch, seq_len, n_features)
        self.conv = nn.Conv1d(n_features, hidden_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = AttentionBlock(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)  # predict mean only (stable)
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        b, s, f = x.shape
        # conv expects (batch, channels, seq)
        x_c = x.transpose(1, 2)  # (b, f, s)
        c = self.conv(x_c)       # (b, hidden, s)
        c = F.relu(self.bn(c))
        c = self.conv2(c)
        c = F.relu(self.bn2(c))
        c = c.transpose(1, 2)    # (b, s, hidden)
        lstm_out, _ = self.lstm(c)
        context = self.attn(lstm_out)
        out = self.head(context)
        mu = out.squeeze(1)
        return mu


# -------------------------------
# Loss: heteroscedastic (predicted variance)
# -------------------------------
class HuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss()

    def forward(self, mu, _, target):
        # keep signature similar: accept (mu, unused, target)
        return self.huber(mu, target)


# -------------------------------
# Training / evaluation / plotting
# -------------------------------

def train(model, train_loader, val_loader, epochs=30, lr=1e-3, device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = HuberLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    history = {'train': [], 'val': []}
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X, last_close, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            opt.zero_grad()
            mu = model(X)
            loss = loss_fn(mu, None, y)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, last_close, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                mu = model(X)
                loss = loss_fn(mu, None, y)
                val_losses.append(loss.item())

        avg_tr = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses)) if val_losses else float('nan')
        history['train'].append(avg_tr)
        history['val'].append(avg_val)
        print(f"Epoch {epoch}/{epochs}  train={avg_tr:.6f}  val={avg_val:.6f}")
        scheduler.step(avg_val)
    return history


def evaluate_and_plot(model, loader, scaler: MinMaxScalerDict, target_col: str = ColNames.CLOSE):
    model.eval()
    device = next(model.parameters()).device
    preds = []
    actuals = []
    last_closes = []

    with torch.no_grad():
        for X, last_close, y in loader:
            X = X.to(device)
            mu = model(X)
            mu = mu.cpu().numpy()
            y = y.numpy()
            preds.append(mu)
            actuals.append(y)
            last_closes.append(last_close.numpy())

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    last_closes = np.concatenate(last_closes)

    # Inverse transform to USD
    pred_usd = scaler.inverse(preds, target_col)
    actual_usd = scaler.inverse(actuals, target_col)

    # Compute simple metrics
    rmse = np.sqrt(np.mean((pred_usd - actual_usd) ** 2))
    mae = np.mean(np.abs(pred_usd - actual_usd))
    mape = np.mean(np.abs((actual_usd - pred_usd) / (actual_usd + 1e-9))) * 100
    corr = np.corrcoef(pred_usd.flatten(), actual_usd.flatten())[0, 1]
    print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, Corr: {corr:.4f}")

    # Plot predictions vs actual
    plt.figure(figsize=(10,5))
    plt.plot(actual_usd, label='Actual Close')
    plt.plot(pred_usd, label='Predicted Close')
    plt.legend()
    plt.title('Predicted vs Actual Close (USD)')
    plt.tight_layout()
    plt.savefig('pred_vs_actual.png')
    print('Saved pred_vs_actual.png')

    # Return arrays for backtest
    return pred_usd, actual_usd, last_closes


def simple_backtest(preds_usd, actuals_usd, last_close, threshold=None, fee=0.0005, adapt_factor=0.5):
    """Very simple backtest: take position if predicted return > threshold.
    If `threshold` is None, derive an adaptive threshold from predicted return volatility.
    Assumes arrays are aligned and represent consecutive trading days.
    """
    pred_return = (preds_usd - last_close) / (last_close + 1e-9)
    if threshold is None:
        # adaptive threshold: fraction of std of predicted returns
        thr = max(1e-4, adapt_factor * np.nanstd(pred_return))
    else:
        thr = threshold

    capital = 1000.0
    equity = [capital]
    pos = 0
    trades = 0
    for i in range(len(preds_usd)):
        pred = preds_usd[i]
        last = last_close[i]
        actual = actuals_usd[i]
        ret = pred_return[i]
        signal = 1 if ret > thr else (-1 if ret < -thr else 0)
        if signal != pos:
            capital *= (1 - fee)
            trades += 1
            pos = signal
        if pos == 0:
            multiplier = 1.0
        elif pos == 1:
            multiplier = actual / (last + 1e-9)
        else:
            multiplier = (last + 1e-9) / (actual + 1e-9)
        capital = capital * multiplier
        equity.append(capital)
    return np.array(equity[1:]), trades, thr


# -------------------------------
# Main runner
# -------------------------------

def main():
    # 1. Load merged data using provided helper
    csv_path = 'hist_data/stock_data/GLD.csv'
    json_path = 'News/finBERT_scores.json'
    print('Loading data...')
    df = dataset_former.MarketDataset.load_data(csv_path, json_path)
    print('Loaded rows:', len(df))

    # 2. Feature engineering
    df = add_technicals(df)

    # 3. Choose features
    features = [ColNames.OPEN, ColNames.HIGH, ColNames.LOW, ColNames.CLOSE, ColNames.VOLUME,
                ColNames.RSI, ColNames.SMA_20, ColNames.SMA_50, ColNames.ATR, ColNames.SENTIMENT, ColNames.SENTIMENT_VOL]

    # Trim to rows with enough history
    seq_len = 30
    if len(df) < seq_len + 10:
        raise RuntimeError('Not enough data to build sequences')

    # 4. Train/val/test split by year (simple)
    years = sorted(df[ColNames.YEAR].unique())
    if len(years) < 3:
        # fallback: simple 70/15/15 split by rows
        n = len(df)
        train_df = df.iloc[:int(n*0.7)].copy()
        val_df = df.iloc[int(n*0.7):int(n*0.85)].copy()
        test_df = df.iloc[int(n*0.85):].copy()
    else:
        train_years = years[:-2]
        val_year = years[-2]
        test_year = years[-1]
        train_df = df[df[ColNames.YEAR].isin(train_years)].copy()
        val_df = df[df[ColNames.YEAR] == val_year].copy()
        test_df = df[df[ColNames.YEAR] == test_year].copy()

    print('Train/Val/Test sizes:', len(train_df), len(val_df), len(test_df))

    scaler = MinMaxScalerDict()
    train_ds = SeqDataset(train_df, features, seq_len=seq_len, scaler=scaler)
    val_ds = SeqDataset(val_df, features, seq_len=seq_len, scaler=scaler)
    test_ds = SeqDataset(test_df, features, seq_len=seq_len, scaler=scaler)

    batch = 64
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False)

    # 5. Model
    model = CNNLSTMAttention(n_features=len(features), seq_len=seq_len, hidden_dim=128)

    # 6. Train
    history = train(model, train_loader, val_loader, epochs=50, lr=1e-3)

    # Plot losses
    plt.figure(); plt.plot(history['train'], label='train'); plt.plot(history['val'], label='val'); plt.legend(); plt.savefig('loss_curve.png')
    print('Saved loss_curve.png')

    # 7. Evaluate on test
    preds_usd, actual_usd, last_closes = evaluate_and_plot(model, test_loader, scaler)

    # 8. Backtest
    equity, trades, thr = simple_backtest(preds_usd, actual_usd, last_closes, threshold=0.002, fee=0.0005)
    print('Backtest trades:', trades, 'Final equity:', equity[-1], 'Threshold used:', thr)
    plt.figure(); plt.plot(equity, label='Strategy'); plt.title('Equity Curve'); plt.savefig('equity.png')
    print('Saved equity.png')


if __name__ == '__main__':
    main()
