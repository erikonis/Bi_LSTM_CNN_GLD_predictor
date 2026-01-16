"""
dataset_former.py
-----------------
Utilities to create a MarketDataset for model training from raw OHLCV CSV
and sentiment JSON files. Provides dataset construction, technical indicator
calculations, normalization helpers, and robust save/load for dataset
artifacts (tensors + parquet DataFrames).

The file exposes `MarketDataset` (primary) and `MarketDataset2` (subclass)
which implement PyTorch `Dataset` interface and utilities used by training
scripts. Save/load keys are centralized as module constants to avoid
accidental mismatches when persisting and restoring objects.
"""

import math
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import pandas_ta as ta  # For technical indicators
import json, os
import dtale
from src.python_engine.training.Constants import ColNames
from src.utils.paths import get_dataset_dir

# Payload keys used for saving/loading the dataset payload. Centralized
# to avoid hard-to-find string mismatches when editing save/load.
PAY_TARGET_DATA = "target_data"
PAY_MARKET_DATA = "market_data"
PAY_SENTIMENT_DATA = "sentiment_data"
PAY_RAW_DATA = "raw_data"
PAY_INDICES = "indices"
PAY_YEARS = "years"
PAY_MARKET_COLS = "market_cols"
PAY_SENT_COLS = "sent_cols"
PAY_TARGET_COLS = "target_cols"
CLOSE_COL = "close_cols"
RAW_COLS = "raw_cols"
PAY_TARGET_NEW_NAMES = "target_new_names"
PAY_ORIGINAL_COLS_M = "original_cols_m"
PAY_ORIGINAL_COLS_S = "original_cols_s"
PAY_ORIGINAL_COLS_T = "original_cols_t"
PAY_TECHNICAL_COLS = "technical_cols"
PAY_NORMALIZED_COLS = "normalized_cols"
PAY_OHLC_MULTIPLIER = "ohlc_multiplier"
PAY_SEQ_LEN = "seq_len"
PAY_EARLIEST_DATE = "earliest_date"
PAY_LATEST_DATE = "latest_date"


class MarketDataset(Dataset):
    def __init__(
        self,
        ticker: str,
        data_df: pd.DataFrame,
        market_cols=ColNames.MARKET_COLS,
        sent_cols=ColNames.SENTIMENT_COLS,
        target_cols: list = [
            ColNames.OPEN,
            ColNames.HIGH,
            ColNames.LOW,
            ColNames.CLOSE,
        ],
        new_target_cols_names: list = ColNames.TARGET_COLS,
        lookback_window: int = 30,
        null: bool = False,
    ) -> None:
        """
        Market Dataset class for 1st CNN-LSTM architecture version, where

        :param data_df: DataFrame containing the merged and processed data
        :param seq_len: Length of the historical window for market data
        """

        if not null:
            self.dataframe = data_df.sort_index()
            self.ticker = ticker.upper()

            # Original price
            if ColNames.VOLUME in market_cols and ColNames.VOLUME not in target_cols:
                price_cols = target_cols + [ColNames.VOLUME]
            else:
                price_cols = target_cols
            self.raw_OHLCV = self.dataframe[price_cols]

            # Dataset-related constants
            self.seq_len = lookback_window
            self.valid_indices = np.arange(self.seq_len, len(self.dataframe))
            self.ohlc_multiplier = 100  # Upscales the normalized value
            self._original_cols_m = market_cols
            self._original_cols_s = sent_cols
            self._original_cols_t = target_cols

            # Columns
            self._price_cols = price_cols
            self._technical_cols = []
            self._normalized_cols = []
            self._market_cols = market_cols
            self._sent_cols = sent_cols
            self._close_col = target_cols[-1] if target_cols else None
            self._target_cols = self.derive_targets(target_cols, new_target_cols_names)

            # Data and Arrays
            self.years = self.dataframe[ColNames.YEAR].values.astype(np.int32)
            self._market_data = self.dataframe[self._market_cols].values.astype(
                np.float32
            )
            self._sentiment_data = self.dataframe[self._sent_cols].values.astype(
                np.float32
            )
            self._target_data = self.dataframe[self._target_cols].values.astype(
                np.float32
            )
            self._raw_prices = self.raw_OHLCV.values.astype(np.float32)

            # Dates:
            self.start_date = pd.to_datetime(
                self.dataframe[ColNames.DATE].iloc[0]
            ).date()
            self.end_date = pd.to_datetime(
                self.dataframe[ColNames.DATE].iloc[-1]
            ).date()

        else:
            self.ticker = None
            self.dataframe = None
            self.raw_OHLCV = None
            self.seq_len = None
            self.valid_indices = None
            self.ohlc_multiplier = None  # Upscales the normalized value
            self._original_cols_m = None
            self._original_cols_s = None
            self._original_cols_t = None
            # Columns
            self._price_cols = None
            self._technical_cols = None
            self._normalized_cols = None
            self._market_cols = None
            self._sent_cols = None
            self._close_col = None
            self._target_cols = None

            # Data and Arrays
            self.years = None
            self._market_data = None
            self._sentiment_data = None
            self._target_data = None
            self._raw_prices = None

            # Dates:
            self.start_date = None
            self.end_date = None

    def __len__(self) -> int:
        """Return number of valid samples in the dataset.

        Returns:
            int: Number of examples (length of `self.valid_indices`).
        """
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        """Build one training example for the `MarketDataset2` variant.

        Parameters:
            idx (int): Index in the dataset (0..len(self)-1). Mapped into
                `self.valid_indices` to compute the actual DataFrame row.

        Returns:
            tuple: (market_tensor, sentiment_tensor, target_tensor, real_price)
                - market_tensor: torch.Tensor, shape `(seq_len, n_market_feats)`
                - sentiment_tensor: torch.Tensor, shape `(seq_len, n_sent_feats)` (windowed)
                - target_tensor: torch.Tensor, shape `(n_target_feats,)`
                - real_price: np.ndarray containing raw OHLCV values for the row
        """

        curr_idx = self.valid_indices[idx]

        # 1. Market Data: seq-length window ending at curr_idx
        mkt_data = self._market_data[curr_idx - self.seq_len + 1 : curr_idx + 1]

        # 2. Sentiment Data: same-length window for MarketDataset2
        sent_data = self._sentiment_data[curr_idx - self.seq_len + 1 : curr_idx + 1]

        target = self._target_data[curr_idx]
        real_price = self._raw_prices[curr_idx]

        return (
            torch.tensor(mkt_data),
            torch.tensor(sent_data),
            torch.tensor(target),
            real_price,
        )

    def get_last_days(self, days:int):
        """Get last `days` days of the whole dataframe."""
        if days <= 0:
            raise ValueError("Days must be a positive integer.")
        return self.dataframe.iloc[-days:]

    def set_new_state(
        self,
        raw_ohlcv_cols: list,
        close_col,
        market_cols: list,
        sent_cols: list,
        target_cols: list,
        tech_cols: list,
        norm_cols: list,
        original_cols_m,
        original_cols_s,
        original_cols_t,
        dataframe,
        raw_ohlcv,
        years,
        market_data,
        sent_data,
        target_data,
        raw_prices,
        early_date,
        late_date,
        lookback_window: int,
        valid_indices: int,
        ohlc_multiplier: int,
        ticker: str,
    ):
        # Original price
        self.ticker = ticker.upper()
        self.dataframe = dataframe
        self.raw_OHLCV = raw_ohlcv

        # Dataset-related constants
        self.seq_len = lookback_window
        self.valid_indices = np.arange(self.seq_len, len(dataframe))
        self.ohlc_multiplier = 100  # Upscales the normalized value
        self._original_cols_m = original_cols_m
        self._original_cols_s = original_cols_s
        self._original_cols_t = original_cols_t

        # Columns
        self._price_cols = raw_ohlcv_cols
        self._technical_cols = tech_cols
        self._normalized_cols = norm_cols
        self._market_cols = market_cols
        self._sent_cols = sent_cols
        self._close_col = close_col
        self._target_cols = target_cols

        # Data and Arrays
        self.years = years
        self._market_data = market_data
        self._sentiment_data = sent_data
        self._target_data = target_data
        self._raw_prices = raw_prices

        # Dates:
        self.start_date = early_date
        self.end_date = late_date

        # Constants:
        self.valid_indices = valid_indices
        self.ohlc_multiplier = ohlc_multiplier

    def derive_targets(
        self, targeted_cols: list, target_cols: list = ColNames.TARGET_COLS
    ):
        """Create shifted target columns from provided source columns.

        Parameters:
            targeted_cols (list[str]): List of source column names.
            target_cols (list[str]): Destination column names to create.

        Returns:
            list[str]: Names of created target columns.
        """

        if not target_cols:
            return []

        j = len(target_cols)
        while len(targeted_cols) > len(target_cols):
            target_cols.append(f"Target_{targeted_cols[j]}")
            j += 1

        self.cols_exists(targeted_cols, strict=True)

        col_names = []
        for i, col in enumerate(targeted_cols):
            # Shift by -1 so that today's row contains tomorrow's target value
            self.dataframe[target_cols[i]] = self.dataframe[col].shift(-1)
            col_names.append(target_cols[i])
        return col_names

    def sync_dates(self):
        self.start_date = pd.to_datetime(self.dataframe[ColNames.DATE].iloc[0]).date()
        self.end_date = pd.to_datetime(self.dataframe[ColNames.DATE].iloc[-1]).date()

    def preprocess(self):
        self.apply_technicals()
        self.normalize()
        self.dropna()

    def apply_technicals(
        self,
        technicals: list = [
            ColNames.RSI,
            ColNames.SMA_20,
            ColNames.SMA_50,
            ColNames.MACD,
            ColNames.MACD_SIG,
            ColNames.ATR,
            ColNames.BB_LOWER,
            ColNames.BB_UPPER,
        ],
    ):
        """Compute and append technical indicator columns to the DataFrame.

        Parameters:
            technicals (list[str]): List of indicator keys to compute (see `ColNames`).

        Returns:
            None: Modifies `self.dataframe` and updates `self._technical_cols`.
        """
        df = self.dataframe
        performed = set()
        new = []
        for tec in technicals:
            if tec in performed:
                pass
            try:
                match tec:
                    case ColNames.RSI:
                        df[ColNames.RSI] = ta.rsi(df[ColNames.CLOSE], length=14)
                        new.append(ColNames.RSI)
                        performed.add(ColNames.RSI)
                    case ColNames.SMA_20:
                        df[ColNames.SMA_20] = df[ColNames.CLOSE] / ta.sma(
                            df[ColNames.CLOSE], length=20
                        )
                        new.append(ColNames.SMA_20)
                        performed.add(ColNames.SMA_20)
                    case ColNames.SMA_50:
                        df[ColNames.SMA_50] = df[ColNames.CLOSE] / ta.sma(
                            df[ColNames.CLOSE], length=50
                        )
                        new.append(ColNames.SMA_50)
                        performed.add(ColNames.SMA_50)
                    case ColNames.MACD:
                        macd = ta.macd(df[ColNames.CLOSE])
                        df[ColNames.MACD] = macd["MACD_12_26_9"]
                        new.append(ColNames.MACD)
                        performed.add(ColNames.MACD)
                    case ColNames.MACD_SIG:
                        macd = ta.macd(df[ColNames.CLOSE])
                        df[ColNames.MACD_SIG] = macd["MACDs_12_26_9"]
                        new.append(ColNames.MACD_SIG)
                        performed.add(ColNames.MACD_SIG)
                    case ColNames.ATR:
                        df[ColNames.ATR] = ta.atr(
                            df[ColNames.HIGH],
                            df[ColNames.LOW],
                            df[ColNames.CLOSE],
                            length=14,
                        )
                        new.append(ColNames.ATR)
                        performed.add(ColNames.ATR)
                    case ColNames.BB_LOWER:
                        bbands = ta.bbands(df[ColNames.CLOSE], length=20, std=2)
                        df[ColNames.BB_LOWER] = bbands.iloc[:, 0] / df[ColNames.CLOSE]
                        new.append(ColNames.BB_LOWER)
                        performed.add(ColNames.BB_LOWER)
                    case ColNames.BB_UPPER:
                        bbands = ta.bbands(df[ColNames.CLOSE], length=20, std=2)
                        df[ColNames.BB_UPPER] = bbands.iloc[:, 2] / df[ColNames.CLOSE]
                        new.append(ColNames.BB_UPPER)
                        performed.add(ColNames.BB_UPPER)
                    case _:
                        print(
                            f"Not implemented or Incorrect Key passed ({tec}). Skipping."
                        )
            except KeyError as e:
                raise KeyError(
                    f"The required column to calculate the technical was not found. Read below: \n{repr(e)}"
                )

        if new:
            self._technical_cols.extend(new)
            self.set_active_cols(new_market_cols=self._market_cols + new)

    def normalize(self, features: list = None, window: int = 90):
        """Normalize selected features and add normalized columns to the DataFrame.

        Parameters:
            features (list[str] | None): List of features to normalize. If None,
                normalizes all current market, sentiment and target columns.
            window (int): Rolling window size used for Z-score normalizations.

        Returns:
            None: Modifies `self.dataframe` in-place and updates the active
            column lists and `self._normalized_cols`.
        """
        if features is None:
            features = self._market_cols + self._sent_cols + self._target_cols

        df = self.dataframe

        new_market, old_market = [], []
        new_sent, old_sent = [], []
        new_target, old_target = [], []

        prev_close = df[self._close_col].shift(1)
        curr_close = df[self._close_col]

        for feat in features:
            # Checking if this column has been already normalized. Avoiding double normalization.
            if ColNames.NOT_TO_NORM_MAP.get(feat) in self._normalized_cols:
                continue

            match feat:
                case ColNames.OPEN:
                    df[ColNames.OPEN_NORM] = self.normalize_price(
                        df[ColNames.OPEN], prev_close
                    )
                    new_market.append(ColNames.OPEN_NORM)
                    old_market.append(ColNames.OPEN)
                case ColNames.HIGH:
                    df[ColNames.HIGH_NORM] = self.normalize_price(
                        df[ColNames.HIGH], prev_close
                    )
                    new_market.append(ColNames.HIGH_NORM)
                    old_market.append(ColNames.HIGH)
                case ColNames.LOW:
                    df[ColNames.LOW_NORM] = self.normalize_price(
                        df[ColNames.LOW], prev_close
                    )
                    new_market.append(ColNames.LOW_NORM)
                    old_market.append(ColNames.LOW)
                case ColNames.CLOSE:
                    df[ColNames.CLOSE_NORM] = self.normalize_price(
                        df[ColNames.CLOSE], prev_close
                    )
                    new_market.append(ColNames.CLOSE_NORM)
                    old_market.append(ColNames.CLOSE)
                case ColNames.VOLUME:
                    df[ColNames.VOLUME_NORM] = np.log(
                        df[ColNames.VOLUME] / df[ColNames.VOLUME].shift(1)
                    )
                    new_market.append(ColNames.VOLUME_NORM)
                    old_market.append(ColNames.VOLUME)
                case ColNames.RSI:
                    df[ColNames.RSI_NORM] = (
                        df[ColNames.RSI] - df[ColNames.RSI].rolling(window).mean()
                    ) / (df[ColNames.RSI].rolling(window).std() + 1e-6)
                    new_market.append(ColNames.RSI_NORM)
                    old_market.append(ColNames.RSI)
                case ColNames.MACD:
                    df[ColNames.MACD_NORM] = (
                        df[ColNames.MACD] - df[ColNames.MACD].rolling(window).mean()
                    ) / (df[ColNames.MACD].rolling(window).std() + 1e-6)
                    new_market.append(ColNames.MACD_NORM)
                    old_market.append(ColNames.MACD)
                case ColNames.MACD_SIG:
                    df[ColNames.MACD_SIG_NORM] = (
                        df[ColNames.MACD_SIG]
                        - df[ColNames.MACD_SIG].rolling(window).mean()
                    ) / (df[ColNames.MACD_SIG].rolling(window).std() + 1e-6)
                    new_market.append(ColNames.MACD_SIG_NORM)
                    old_market.append(ColNames.MACD_SIG)
                case ColNames.ATR:
                    df[ColNames.ATR_NORM] = df[ColNames.ATR]
                case ColNames.SMA_20:
                    df[ColNames.SMA_20_NORM] = (df[ColNames.SMA_20] - 1) * 10
                    new_market.append(ColNames.SMA_20_NORM)
                    old_market.append(ColNames.SMA_20)
                case ColNames.SMA_50:
                    df[ColNames.SMA_50_NORM] = (df[ColNames.SMA_50] - 1) * 10
                    new_market.append(ColNames.SMA_50_NORM)
                    old_market.append(ColNames.SMA_50)
                case ColNames.BB_LOWER:
                    df[ColNames.BB_LOWER_NORM] = (df[ColNames.BB_LOWER] - 1) * 10
                    new_market.append(ColNames.BB_LOWER_NORM)
                    old_market.append(ColNames.BB_LOWER)
                case ColNames.BB_UPPER:
                    df[ColNames.BB_UPPER_NORM] = (df[ColNames.BB_UPPER] - 1) * 10
                    new_market.append(ColNames.BB_UPPER_NORM)
                    old_market.append(ColNames.BB_UPPER)
                case ColNames.SENTIMENT:
                    pass
                case ColNames.SENTIMENT_VOL:
                    news_log = np.log1p(df[ColNames.SENTIMENT_VOL])

                    rolling_mean = news_log.rolling(window=90, min_periods=1).mean()
                    rolling_std = news_log.rolling(window=90, min_periods=1).std(ddof=0)

                    expanding_mean = news_log.expanding(min_periods=2).mean()
                    expanding_std = news_log.expanding(min_periods=2).std(ddof=0)

                    # Fill the "Cold Start" gaps with the growing history
                    final_mean = rolling_mean.fillna(expanding_mean)
                    final_std = rolling_std.fillna(expanding_std)

                    df[ColNames.SENTIMENT_VOL_NORM] = (news_log - final_mean) / (
                        final_std + 1e-6
                    )
                    new_sent.append(ColNames.SENTIMENT_VOL_NORM)
                    old_sent.append(ColNames.SENTIMENT_VOL)
                case ColNames.TARGET_O:
                    # for targets current close serves as their previous.
                    df[ColNames.TARGET_O_NORM] = self.normalize_price(
                        df[ColNames.TARGET_O], curr_close
                    )
                    new_target.append(ColNames.TARGET_O_NORM)
                    old_target.append(ColNames.TARGET_O)
                case ColNames.TARGET_H:
                    df[ColNames.TARGET_H_NORM] = self.normalize_price(
                        df[ColNames.TARGET_H], curr_close
                    )
                    new_target.append(ColNames.TARGET_H_NORM)
                    old_target.append(ColNames.TARGET_H)
                case ColNames.TARGET_L:
                    df[ColNames.TARGET_L_NORM] = self.normalize_price(
                        df[ColNames.TARGET_L], curr_close
                    )
                    new_target.append(ColNames.TARGET_L_NORM)
                    old_target.append(ColNames.TARGET_L)
                case ColNames.TARGET_C:
                    df[ColNames.TARGET_C_NORM] = self.normalize_price(
                        df[ColNames.TARGET_C], curr_close
                    )
                    new_target.append(ColNames.TARGET_C_NORM)
                    old_target.append(ColNames.TARGET_C)
                case _:
                    raise NotImplementedError(
                        f"Something Unexpected happened. There is no implementation to normalize such feature ({feat})."
                    )

        market = [f for f in self._market_cols if f not in old_market] + new_market
        sent = [f for f in self._sent_cols if f not in old_sent] + new_sent
        target = [f for f in self._target_cols if f not in old_target] + new_target
        self.set_active_cols(market, sent, target)

        self._normalized_cols.extend(new_market + new_sent + new_target)

    def dropna(self):
        print(self.dataframe.head())
        self.dataframe = self.dataframe.dropna()
        print(self.dataframe.head())
        self.raw_OHLCV = self.dataframe[self._price_cols]
        self.years = self.dataframe[ColNames.YEAR]
        self._market_data = self.dataframe[self._market_cols].values.astype(np.float32)
        self._sentiment_data = self.dataframe[self._sent_cols].values.astype(np.float32)
        self._target_data = self.dataframe[self._target_cols].values.astype(np.float32)
        self._raw_prices = self.raw_OHLCV.values.astype(np.float32)
        self.valid_indices = np.arange(self.seq_len, len(self.dataframe))
        self.sync_dates()

    def normalize_price(self, price, last_close):
        """Vectorized log-return normalization for price columns.

        Parameters:
            price (pd.Series | np.ndarray): Prices to normalize.
            last_close (pd.Series | np.ndarray): Reference previous-close prices.

        Returns:
            pd.Series | np.ndarray: Normalized values (log returns scaled).
        """
        return normalize_price(price, last_close, self.ohlc_multiplier)

    def unnormalize_price(self, value, last_close):
        """Invert `normalize_price` and return original-scale prices.

        Parameters:
            value (pd.Series | np.ndarray): Normalized value(s) to invert.
            last_close (pd.Series | np.ndarray): Reference price(s) used when
                the value was originally normalized.

        Returns:
            pd.Series | np.ndarray: Reconstructed price values.
        """
        return np.exp(value / self.ohlc_multiplier) * last_close

    def delete_features(self, features: list = None):
        """Remove specified features from the dataset.

        Parameters:
            features (list[str] | None): Column names to drop. If None, clears
                the entire DataFrame and all tracked feature lists.

        Returns:
            None: Mutates `self.dataframe` and internal column-tracking lists.
        """
        if features is None:
            self.dataframe = self.dataframe[[]]
            self._market_cols = self._sent_cols = self._target_cols = (
                self._technical_cols
            ) = self._normalized_cols = []
            self.set_active_cols([], [], [])

        else:
            self.dataframe = self.dataframe.drop(columns=features, axis=1)
            mkt = [f for f in self._market_cols if f not in features]
            sent = [f for f in self._sent_cols if f not in features]
            tgt = [f for f in self._target_cols if f not in features]
            self._technical_cols = [
                f for f in self._technical_cols if f not in features
            ]
            self._normalized_cols = [
                f for f in self._normalized_cols if f not in features
            ]
            self.set_active_cols(mkt, sent, tgt)

    def reset(self, setting="all"):
        """Reset dataset features according to `setting`.

        Parameters:
            setting (str): One of 'all', 'tech', 'norm'.
                - 'all': remove technical & normalized features and restore
                    original active columns.
                - 'tech': remove only technical indicators.
                - 'norm': remove only normalized columns.

        Returns:
            None
        """
        tech, norm = self.get_tech_norm_cols()
        match setting:
            case "all":
                self.delete_features(tech + norm)
                self.set_active_cols(
                    self._original_cols_m, self._original_cols_s, self._original_cols_t
                )
            case "tech":
                self.delete_features(tech)
            case "norm":
                self.delete_features(norm)

    def set_active_cols(
        self,
        new_market_cols: list = None,
        new_sent_cols: list = None,
        new_target_cols: list = None,
    ):
        """Update active column lists and refresh internal numpy arrays.

        Parameters:
            new_market_cols (list[str] | None): New market feature column names.
            new_sent_cols (list[str] | None): New sentiment feature column names.
            new_target_cols (list[str] | None): New target column names.

        Returns:
            None
        """

        if new_market_cols is not None:
            if self.cols_exists(new_market_cols, strict=True):
                self._market_cols = new_market_cols
                self._market_data = self.dataframe[self._market_cols].values.astype(
                    np.float32
                )

        if new_sent_cols is not None:
            if self.cols_exists(new_sent_cols, strict=True):
                self._sent_cols = new_sent_cols
                self._sentiment_data = self.dataframe[self._sent_cols].values.astype(
                    np.float32
                )

        if new_target_cols is not None:
            if self.cols_exists(new_target_cols, strict=True):
                self._target_cols = new_target_cols
                self._target_data = self.dataframe[self._target_cols].values.astype(
                    np.float32
                )

    def set_original_cols(self, new_origin_m, new_origin_s, new_origin_t):
        """Set a new tuple of original column lists used by `reset`.

        Parameters:
            new_origin_m (list[str]): Market original columns.
            new_origin_s (list[str]): Sentiment original columns.
            new_origin_t (list[str]): Target original columns.

        Raises:
            KeyError: If any of the provided feature names are not present in
                the underlying DataFrame.
        """
        all_cols = self.get_all_cols()
        bad = []
        for feat in new_origin_m + new_origin_s + new_origin_t:
            if feat not in all_cols:
                bad.append(feat)
        if bad:
            raise KeyError(
                f"Cannot set new origins. Features: {bad} are not in the dataframe. The selected feature name is wrong!"
            )

        self._original_cols_m = new_origin_m
        self._original_cols_s = new_origin_s
        self._original_cols_t = new_origin_t

    def cols_exists(self, cols: list, strict: bool = False):
        """Takes a list of columns and checks if they exist within the dataset dataframe. Returns False if at least one does not exist.
        Else returns True.
        """
        all_cols = self.get_all_cols()
        bad = []
        for col in cols:
            if col not in all_cols:
                bad.append(col)
        if bad:
            if strict:
                raise KeyError(
                    f"The following columns do not exist in the dataframe: {bad}"
                )
            else:
                return False
        return True

    def get_tech_norm_cols(self):
        """Return the lists of technical and normalized column names.

        Returns:
            tuple: (technical_cols, normalized_cols) both lists of str.
        """
        return self._technical_cols, self._normalized_cols

    def get_active_cols(self):
        """Return currently active market, sentiment and target columns.

        Returns:
            tuple: (market_cols, sent_cols, target_cols) each a list of str.
        """
        return self._market_cols, self._sent_cols, self._target_cols

    def get_original_cols(self):
        """Return the original column lists captured at initialization.

        Returns:
            tuple: (original_market, original_sentiment, original_target).
        """
        return self._original_cols_m, self._original_cols_s, self._original_cols_t

    def get_all_cols(self):
        """Return all column names currently present in the DataFrame.

        Returns:
            numpy.ndarray: Array of column name strings.
        """
        return self.dataframe.columns.values

    def get_indices_by_year(self, years_list: list):
        """Return dataset indices (relative to `__len__`) for given years.

        Parameters:
            years_list (list[int]): Years to include (e.g. [2018,2019]).

        Returns:
            list[int]: Indices into the dataset (0..len(self)-1) whose
                corresponding DataFrame rows have a year in `years_list`.
        """
        lst = []
        for idx in range(len(self.valid_indices)):
            if self.years[self.valid_indices[idx]] in years_list:
                lst.append(idx)
        return lst

    def get_loaders(
        self, batch_size: int = 64, training_setting: str = "expanding_window"
    ):
        """
        A generator that outputs a tuple of loaders `(train_set, val_set, test_set, real_price_set)` for each training cycle.
        For instance, in case of `training_setting == "expanding_window"`, each `next()` call yields new train, val, test split
        with train dataset incremented by 1 year: \n
            1. train 2015-2020, val 2021, test 2022
            2. train 2015-2021, val 2022, test 2023
            and so on...

        Default option is expanding_window, which from the start takes `math.floor(total_years*0.6)` years for the train set.

        :param batch_size: the size of each batch for the training. Applies for train and validation sets.
        :type: int
        :param training_setting: the splitting option. Default is "expanding_window". Putting wrong values raises ValueError. No other is implemented for now.
        :type: str
        :returns: (train_set, validation_set, test_set) upon next() call
        """
        start_year = self.years[0]
        end_year = self.years[-1]
        num_years = len(set(self.years))

        match training_setting:
            case "expanding_window":
                test_years = range(
                    start_year + math.floor(num_years * 0.6), end_year + 1
                )

                for test_year in test_years:
                    print(f"\n--- Starting Fold: Test Year {test_year} ---")

                    # Validation is usually the year before the test year, or a subset of training
                    val_year = test_year - 1

                    # Training is everything from start_year up to (but not including) val_year
                    train_years = list(range(start_year, val_year))

                    # 1. Get Indices
                    train_idx = self.get_indices_by_year(train_years)
                    val_idx = self.get_indices_by_year([val_year])
                    test_idx = self.get_indices_by_year([test_year])

                    # 2. Create Subsets and Loaders, make a generator that yields next loaders.
                    yield (
                        DataLoader(
                            Subset(self, train_idx),
                            batch_size=batch_size,
                            shuffle=True,
                        ),  # train
                        DataLoader(
                            Subset(self, val_idx), batch_size=batch_size, shuffle=False
                        ),  # validate
                        DataLoader(
                            Subset(self, test_idx), batch_size=1, shuffle=False
                        ),  # test
                        test_year,  # metadata - which year we are in
                    )

            # Placeholder for other datasplit options if needed to implement (rolling window, fixed, etc.)
            case "sliding_window":
                ref_start_year = start_year
                test_years = range(
                    start_year + math.floor(num_years * 0.6), end_year + 1
                )

                for test_year in test_years:
                    print(f"\n--- Starting Fold: Test Year {test_year} ---")

                    # Validation is usually the year before the test year, or a subset of training
                    val_year = test_year - 1

                    # Training is everything from start_year up to (but not including) val_year
                    train_years = list(range(ref_start_year, val_year))
                    ref_start_year += 1

                    # 1. Get Indices
                    train_idx = self.get_indices_by_year(train_years)
                    val_idx = self.get_indices_by_year([val_year])  # REVERT TO val_year
                    test_idx = self.get_indices_by_year([test_year])

                    # 2. Create Subsets and Loaders, make a generator that yields next loaders.
                    yield (
                        DataLoader(
                            Subset(self, train_idx),
                            batch_size=batch_size,
                            shuffle=True,
                        ),  # train
                        DataLoader(
                            Subset(self, val_idx), batch_size=batch_size, shuffle=False
                        ),  # validate
                        DataLoader(
                            Subset(self, test_idx), batch_size=1, shuffle=False
                        ),  # test
                        test_year,  # metadata - which year we are in
                    )
            case _:
                raise ValueError(
                    f"Incorrect argument passed. There is no option '{training_setting}'!"
                )

    def get_dataframe(self) -> pd.DataFrame:
        """Return the underlying pandas DataFrame.

        Returns:
            pandas.DataFrame: The processed DataFrame used by the dataset.
        """
        return self.dataframe

    def show_dataframe_interactive(
        self,
        subprocess: bool = False,
        which: int = 0,
        custom: bool = False,
        options: list = ["active"],
    ) -> None:
        """Open an interactive D-Tale session to explore the dataset.

        Parameters:
            subprocess (bool): If True, open D-Tale in a background subprocess
                (the Python process can continue). If False, the function will
                block until the user closes the session or presses Enter.
            which (int): Which DataFrame to show: 0 => both processed dataframe
                and raw OHLCV, 1 => processed dataframe only, 2 => raw OHLCV only.
            custom (bool): If set to true, in the dataframe will be displayed only the columns that one specifies in option.
                Else prefixed options are applied as displayed below.
            options ([str]): What to display. All options display at least Year and Date. Possible options are:
                - "norm" - displays all the normalized columns
                - "tech" - displays all the technical indicators
                - "ohlcv" - displays non-normalized OHLCV values
                - "ohlcvN" - displays normalized OHLCV values
                - "ohlc" - displays non-normalized OHLC values
                - "ohlcN" - displays non-normalized OHLC values
                - "active" - all currently active columns (default)
                - [] - displays everything
                Multiple options are possible.
        Returns:
            dtale.DeltaDisplay: The D-Tale display object (or None if subprocess
            mode hides it). Note: the function also prints and opens the URL
            in the browser for convenience.
        """
        time = [ColNames.YEAR, ColNames.DATE]

        match which:
            case 0:
                display = dtale.show(self.raw_OHLCV, subprocess=True)
                display.open_browser()
                if custom:
                    display = dtale.show(
                        self.dataframe[time + options], subprocess=True
                    )
                else:
                    if not options:
                        display = dtale.show(self.dataframe, subprocess=True)
                    else:
                        cols = self._eval_option(options)
                        display = dtale.show(
                            self.dataframe[time + cols], subprocess=True
                        )
                display.open_browser()
            case 1:
                if custom:
                    display = dtale.show(
                        self.dataframe[time + options], subprocess=True
                    )
                else:
                    if not options:
                        display = dtale.show(self.dataframe, subprocess=True)
                    else:
                        cols = self._eval_option(options)
                        display = dtale.show(
                            self.dataframe[time + cols], subprocess=True
                        )
                display.open_browser()
            case 2:
                display = dtale.show(self.raw_OHLCV, subprocess=True)
                display.open_browser()
            case _:
                raise ValueError(
                    f"Unexpected argument '{which}' passed. Use 0, 1 or 2 only."
                )

        if not subprocess:
            try:
                input("D-Tale sessions running — press Enter to exit...\n")
            except KeyboardInterrupt:
                pass

        return display

    def _eval_option(self, option: list):
        cols = []
        for opt in option:
            match opt.lower():
                case "all":
                    cols.extend(
                        [
                            f
                            for f in self.get_all_cols()
                            if f not in [ColNames.DATE, ColNames.YEAR]
                        ]
                    )
                case "norm":
                    cols.extend(self._normalized_cols)
                case "tech":
                    cols.extend(self._technical_cols)
                case "ohlcv":
                    cols.extend(self._original_cols_m)
                case "ohlcvN":
                    temp = []
                    for f in self._original_cols_m:
                        if ColNames.NOT_TO_NORM_MAP(f) in self._normalized_cols:
                            temp.append(f)
                    cols.extend(temp)
                case "ohlc":
                    cols.extend(
                        [f for f in self._original_cols_m if f is not ColNames.VOLUME]
                    )
                case "ohlcN":
                    temp = []
                    for f in self._original_cols_m:
                        if (
                            ColNames.NOT_TO_NORM_MAP(f) in self._normalized_cols
                            and ColNames.NOT_TO_NORM_MAP(f) is not ColNames.VOLUME_NORM
                        ):
                            temp.append(f)
                    cols.extend(temp)
                case "active":
                    mkt, snt, tgt = self.get_active_cols()
                    cols.extend(mkt + snt + tgt)
                case _:
                    print("Unknown option")
        return cols

    def save(self, ticker=None, folder_path=None) -> None:
        """Serialize/save the MarketDataset object to disk.

        Parameters:
            folder_path (str): Directory to write artifacts into. Created if
                it does not exist (default: 'saved_dataset').

        Persisted artifacts:
            - `dataframe.parquet` : the full processed DataFrame (human-readable)
            - `raw_OHLCV.parquet`  : the original raw OHLCV subset
            - `tensors.pt`         : binary payload with numpy arrays and metadata

        The binary payload keys are centralized as module-level constants so
        callers and future edits don't accidentally mismatch string keys.

        Returns:
            None
        """

        if ticker is None:
            ticker = self.ticker.upper()

        if folder_path is None:
            folder_path = get_dataset_dir(ticker)

        # Save Numerical Data as binary
        payload = {
            "ticker": self.ticker,
            PAY_TARGET_DATA: torch.from_numpy(self._target_data),
            PAY_MARKET_DATA: torch.from_numpy(self._market_data),
            PAY_SENTIMENT_DATA: torch.from_numpy(self._sentiment_data),
            PAY_RAW_DATA: torch.from_numpy(self._raw_prices),
            PAY_INDICES: torch.from_numpy(self.valid_indices),
            PAY_YEARS: torch.from_numpy(np.asarray(self.years).astype(np.int32)),
            PAY_MARKET_COLS: self._market_cols,
            PAY_SENT_COLS: self._sent_cols,
            PAY_TARGET_COLS: self._target_cols,
            CLOSE_COL: self._close_col,
            RAW_COLS: self._price_cols,
            PAY_TARGET_NEW_NAMES: self._target_cols,
            PAY_ORIGINAL_COLS_M: self._original_cols_m,
            PAY_ORIGINAL_COLS_S: self._original_cols_s,
            PAY_ORIGINAL_COLS_T: self._original_cols_t,
            PAY_TECHNICAL_COLS: self._technical_cols,
            PAY_NORMALIZED_COLS: self._normalized_cols,
            PAY_OHLC_MULTIPLIER: self.ohlc_multiplier,
            PAY_EARLIEST_DATE: self.start_date.isoformat(),
            PAY_LATEST_DATE: self.end_date.isoformat(),
            PAY_SEQ_LEN: self.seq_len,
        }
        torch.save(payload, os.path.join(folder_path, "tensors.pt"))

        # Save DataFrame as Parquet for easy human inspection / reuse
        self.dataframe.to_parquet(os.path.join(folder_path, "dataframe.parquet"))
        print(f"Dataset successfully saved to {folder_path}")

        # Raw OHLCV (subset) saved separately to preserve original prices
        self.raw_OHLCV.to_parquet(os.path.join(folder_path, "raw_OHLCV.parquet"))
        print(f"Raw OHLCV dataset successfully saved to {folder_path}")

    @classmethod
    def load(cls, ticker, folder_path=None):
        """Deserialize/load: Reconstruct the object from saved artifacts.

        Parameters:
            folder_path (str): Directory where `save()` wrote dataset artifacts.

        Returns:
            MarketDataset: A new instance with internal arrays and metadata
            restored to match the object that was saved.

        Raises:
            FileNotFoundError: If expected files are missing from `folder_path`.
        """

        ticker = ticker.upper()

        if folder_path is None:
            folder_path = get_dataset_dir(ticker)

        payload = torch.load(
            os.path.join(folder_path, "tensors.pt"), weights_only=False
        )

        instance = cls(
            ticker,
            None,
            market_cols=[],
            sent_cols=[],
            target_cols=[],
            new_target_cols_names=[],
            null=True,
        )

        instance.set_new_state(
            ticker=ticker,
            raw_ohlcv_cols=payload.get(RAW_COLS),
            close_col=payload[CLOSE_COL],
            market_cols=payload.get(PAY_MARKET_COLS),
            sent_cols=payload.get(PAY_SENT_COLS),
            target_cols=payload.get(PAY_TARGET_NEW_NAMES, instance._target_cols),
            tech_cols=payload.get(PAY_TECHNICAL_COLS, instance._technical_cols),
            norm_cols=payload.get(PAY_NORMALIZED_COLS, instance._normalized_cols),
            original_cols_m=payload.get(PAY_ORIGINAL_COLS_M, instance._original_cols_m),
            original_cols_s=payload.get(PAY_ORIGINAL_COLS_S, instance._original_cols_s),
            original_cols_t=payload.get(PAY_ORIGINAL_COLS_T, instance._original_cols_t),
            dataframe=pd.read_parquet(os.path.join(folder_path, "dataframe.parquet")),
            raw_ohlcv=pd.read_parquet(os.path.join(folder_path, "raw_OHLCV.parquet")),
            years=payload[PAY_YEARS].numpy(),
            market_data=payload[PAY_MARKET_DATA].numpy(),
            sent_data=payload[PAY_SENTIMENT_DATA].numpy(),
            target_data=payload[PAY_TARGET_DATA].numpy(),
            raw_prices=payload[PAY_RAW_DATA].numpy(),
            early_date=pd.to_datetime(payload.get(PAY_EARLIEST_DATE)).date(),
            late_date=pd.to_datetime(payload.get(PAY_LATEST_DATE)).date(),
            ohlc_multiplier=payload.get(PAY_OHLC_MULTIPLIER, instance.ohlc_multiplier),
            lookback_window=payload[PAY_SEQ_LEN],
            valid_indices=payload[PAY_INDICES].numpy(),
        )

        return instance

    @classmethod
    def form_dataset(
        cls,
        csv_path: str,
        json_path: str,
        ticker: str,
        market_cols=ColNames.MARKET_COLS,
        sent_cols=ColNames.SENTIMENT_COLS,
        tgt_cols=[ColNames.OPEN, ColNames.HIGH, ColNames.LOW, ColNames.CLOSE],
        seq_len: int = 30,
    ):
        """
        Loads, processes, and forms a MarketDataset ready for training.

        Parameters:
            csv_path (str): Path to the market CSV file.
            json_path (str): Path to the sentiment JSON file (JSON dict date -> [score,count]).
            ticker (str): Ticker symbol for the asset.
            market_cols (list[str]): Names of CSV columns to map to standard OHLCV.
            sent_cols (list[str]): Names of sentiment columns expected.
            tgt_cols (list[str]): Source columns to derive targets from.
            seq_len (int): Length of historical window for market data.

        Returns:
            MarketDataset: Constructed dataset instance ready for training.
        """
        # Load and merge data
        df = MarketDataset.load_data(csv_path, json_path)

        # Create Dataset
        dataset = cls(
            df,
            ticker = ticker, 
            market_cols=market_cols,
            sent_cols=sent_cols,
            target_cols=tgt_cols,
            lookback_window=seq_len,
        )

        return dataset

    @classmethod
    def load_data(
        cls,
        csv_path: str,
        json_path: str,
        market_cols=["Open", "High", "Low", "Close", "Volume"],
        market_new_cols=ColNames.MARKET_COLS,
    ) -> pd.DataFrame:
        """
        Load and merge market and sentiment data.
        Market data is in CSV, sentiment data is in JSON.
        Market CSV has columns: 'Date', '24h Open (USD)', '24h High (USD)',
        '24h Low (USD)', 'Closing Price (USD)', 'Trading Volume'.
        Sentiment JSON is a dict with {date :   [sentiment score, article count]}.

        Parameters:
            csv_path (str): Path to the market CSV file. Expected to contain
                columns matching `market_cols` (default names listed in signature).
            json_path (str): Path to a JSON file containing a dict mapping date
                strings to [sentiment_score, article_count].
            market_cols (list[str]): Original CSV column names to map from.
            market_new_cols (list[str]): Target column names to rename to (e.g. 'Open','High','Low','Close','Volume').

        Returns:
            pandas.DataFrame: Merged DataFrame containing Date, OHLCV, Sentiment and Year.
        """

        # 1. Load Market CSV
        # We map your specific column names to standard OHLCV
        mkt_df = pd.read_csv(csv_path)
        mkt_df[ColNames.DATE] = pd.to_datetime(mkt_df[ColNames.DATE])
        mkt_df = mkt_df.rename(
            columns={
                market_cols[0]: market_new_cols[0],
                market_cols[1]: market_new_cols[1],
                market_cols[2]: market_new_cols[2],
                market_cols[3]: market_new_cols[3],
                market_cols[4]: market_new_cols[4],
            }
        )

        # 2. Load Sentiment JSON
        with open(json_path, "r") as f:
            sent_dict = json.load(f)

        # Convert dict to DataFrame
        rows = []
        for date, lst in sent_dict.items():
            rows.append([date, lst[0], lst[1]])  # single score, count
        sent_df = pd.DataFrame(
            rows, columns=[ColNames.DATE, ColNames.SENTIMENT, ColNames.SENTIMENT_VOL]
        )
        sent_df[ColNames.DATE] = pd.to_datetime(sent_df[ColNames.DATE])

        # 3. Merge on Date
        # 'inner' ensures we only keep days where we have BOTH price and sentiment
        df = pd.merge(mkt_df, sent_df, on=ColNames.DATE, how="left").sort_values(
            ColNames.DATE
        )

        # Add Year column for Walk-Forward splits
        df[ColNames.YEAR] = df[ColNames.DATE].dt.year

        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index(ColNames.YEAR)))
        df = df[cols]
        return df

    def add_data(self, other_df: pd.DataFrame):
        """Append historical data from another DataFrame to the dataset.

        Parameters:
            other_df (pandas.DataFrame): DataFrame with same structure as
                `self.dataframe` to append before existing data.
        Returns:
            None: Mutates `self.dataframe` and internal arrays.
        """

        # 1. Combine dataframes
        # We put other_df first so that keep='first' picks the new data
        combined = pd.concat([other_df, self.dataframe], ignore_index=False)

        # 2. Handle duplicates: Keep the one from other_df (the first occurrences)
        # This removes the old version from self.dataframe if it existed in other_df
        combined = combined[~combined.index.duplicated(keep="first")]

        # 3. Sort by Date Index to ensure the timeline is chronological
        # Critical for sliding window functions!
        combined = combined.sort_index()

        self.set_new_state(
            raw_ohlcv_cols=self._price_cols,
            close_col=self._close_col,
            market_cols=self._market_cols,
            sent_cols=self._sent_cols,
            target_cols=self._target_cols,
            tech_cols=self._technical_cols,
            norm_cols=self._normalized_cols,
            original_cols_m=self._original_cols_m,
            original_cols_s=self._original_cols_s,
            original_cols_t=self._original_cols_t,
            dataframe=combined,
            raw_ohlcv=combined[self._price_cols],
            years=combined[ColNames.YEAR],
            market_data=combined[self._market_cols].values.astype(np.float32),
            sent_data=combined[self._sent_cols].values.astype(np.float32),
            target_data=combined[self._target_cols].values.astype(np.float32),
            raw_prices=combined[self._price_cols].values.astype(np.float32),
            early_date=None,
            late_date=None,
            lookback_window=self.seq_len,
            valid_indices=np.arange(self.seq_len, len(combined)),
            ohlc_multiplier=self.ohlc_multiplier,
        )
        self.sync_dates()

    def add_hist_data(
        self,
        csv_path: str,
        json_path: str,
        market_cols=["Open", "High", "Low", "Close", "Volume"],
        market_new_cols=ColNames.MARKET_COLS,
    ):
        """
        Add historical data from a csv and json file. Method expects to find market data in csv file.
        Code expects:
            - In csv file to find a column 'Date' (YYYY-MM-DD) and columns specified in `market_cols`. These columns
                will be renamed positionally to column names specified in `market_new_cols`. New names must match column names
                of original dataframe.
            - In json file to find dates as keys (YYYY-MM-DD) and list as values: `[sentiment_value_for_day, article_count_per_day]`
        While loading, data will be joined on the market data (i.e. we can expect to have days with NaN values, but can never expect
        to have days with such values for market data).

        Parameters:
            csv_path (str): path to the market data csv file. E.g. `hist/GLD.csv`
            json_path (str): path to the sentiment data json file. E.g. `hist/sent_scores.json`
            market_cols (list(str)): list of columns names to be searched aside of `Data` within market csv file.
            market_new_cols (list(str)): list of column names to which the `market_cols` will be renamed to match the original dataframe.
        Returns:
            None: Mutates `self.dataframe` and internal arrays.

        """
        df = MarketDataset.load_data(csv_path, json_path, market_cols, market_new_cols)
        self.add_data(df)

    def form_infere_sample(self, open, close, high, low, volume, sentiment, sentiment_vol, mkt_cols, sent_cols, window=30, lookback=134) -> pd.DataFrame:
        """
        Forms a single inference sample as a pandas DataFrame row.

        Parameters:
            open (float): Opening price.
            close (float): Closing price.
            high (float): Highest price.
            low (float): Lowest price.
            volume (float): Trading volume.
            sentiment (float): Sentiment score.
            sentiment_vol (int): Number of sentiment articles.
        Returns:
            pandas.DataFrame: A single-row DataFrame with the provided data.
        """
        data = {
            ColNames.OPEN: [open],
            ColNames.CLOSE: [close],
            ColNames.HIGH: [high],
            ColNames.LOW: [low],
            ColNames.VOLUME: [volume],
            ColNames.SENTIMENT: [sentiment],
            ColNames.SENTIMENT_VOL: [sentiment_vol],
        }
        df = pd.DataFrame(data)

        context_df = self.dataframe[self._original_cols_m + self._original_cols_s].tail(lookback)
        df = pd.concat([context_df, df], ignore_index=True)
        df = apply_technicals(df)
        df = normalize(df)
        sample = df.tail(1)
        window_context = self.dataframe.tail(29)
        final = pd.concat([window_context, sample])
        mkt = final[mkt_cols]
        snt = final[sent_cols]
        mkt_data = mkt.values.astype(np.float32)
        snt_data = snt.values.astype(np.float32)

        return mkt_data, snt_data
    
def apply_technicals(
        df,
        technicals: list = [
            ColNames.RSI,
            ColNames.SMA_20,
            ColNames.SMA_50,
            ColNames.MACD,
            ColNames.MACD_SIG,
            ColNames.ATR,
            ColNames.BB_LOWER,
            ColNames.BB_UPPER,
        ],
    ):
        """Compute and append technical indicator columns to the DataFrame.

        Parameters:
            technicals (list[str]): List of indicator keys to compute (see `ColNames`).

        Returns:
            None: Modifies `df`.
        """
        performed = set()
        for tec in technicals:
            if tec in performed:
                pass
            try:
                match tec:
                    case ColNames.RSI:
                        df[ColNames.RSI] = ta.rsi(df[ColNames.CLOSE], length=14)
                    case ColNames.SMA_20:
                        df[ColNames.SMA_20] = df[ColNames.CLOSE] / ta.sma(
                            df[ColNames.CLOSE], length=20
                        )
                    case ColNames.SMA_50:
                        df[ColNames.SMA_50] = df[ColNames.CLOSE] / ta.sma(
                            df[ColNames.CLOSE], length=50
                        )
                    case ColNames.MACD:
                        macd = ta.macd(df[ColNames.CLOSE])
                        df[ColNames.MACD] = macd["MACD_12_26_9"]
                    case ColNames.MACD_SIG:
                        macd = ta.macd(df[ColNames.CLOSE])
                        df[ColNames.MACD_SIG] = macd["MACDs_12_26_9"]
                    case ColNames.ATR:
                        df[ColNames.ATR] = ta.atr(
                            df[ColNames.HIGH],
                            df[ColNames.LOW],
                            df[ColNames.CLOSE],
                            length=14,
                        )
                    case ColNames.BB_LOWER:
                        bbands = ta.bbands(df[ColNames.CLOSE], length=20, std=2)
                        df[ColNames.BB_LOWER] = bbands.iloc[:, 0] / df[ColNames.CLOSE]
                    case ColNames.BB_UPPER:
                        bbands = ta.bbands(df[ColNames.CLOSE], length=20, std=2)
                        df[ColNames.BB_UPPER] = bbands.iloc[:, 2] / df[ColNames.CLOSE]
                    case _:
                        print(
                            f"Not implemented or Incorrect Key passed ({tec}). Skipping."
                        )
            except KeyError as e:
                raise KeyError(
                    f"The required column to calculate the technical was not found. Read below: \n{repr(e)}"
                )
            performed.add(tec)
        return df
            
def normalize(df, features: list = None, window: int = 90):
        """Normalize selected features and add normalized columns to the DataFrame.

        Parameters:
            features (list[str] | None): List of features to normalize. If None,
                normalizes all current market, sentiment and target columns.
            window (int): Rolling window size used for Z-score normalizations.

        Returns:
            None: Modifies `self.dataframe` in-place and updates the active
            column lists and `self._normalized_cols`.
        """
        if features is None:
            features = df.columns.tolist()


        prev_close = df[ColNames.CLOSE].shift(1)
        curr_close = df[ColNames.CLOSE]

        for feat in features:
            match feat:
                case ColNames.OPEN:
                    df[ColNames.OPEN_NORM] = normalize_price(
                        df[ColNames.OPEN], prev_close
                    )
                case ColNames.HIGH:
                    df[ColNames.HIGH_NORM] = normalize_price(
                        df[ColNames.HIGH], prev_close
                    )
                case ColNames.LOW:
                    df[ColNames.LOW_NORM] = normalize_price(
                        df[ColNames.LOW], prev_close
                    )
                case ColNames.CLOSE:
                    df[ColNames.CLOSE_NORM] = normalize_price(
                        df[ColNames.CLOSE], prev_close
                    )
                case ColNames.VOLUME:
                    df[ColNames.VOLUME_NORM] = np.log(
                        df[ColNames.VOLUME] / df[ColNames.VOLUME].shift(1)
                    )
                case ColNames.RSI:
                    df[ColNames.RSI_NORM] = (
                        df[ColNames.RSI] - df[ColNames.RSI].rolling(window).mean()
                    ) / (df[ColNames.RSI].rolling(window).std() + 1e-6)
                case ColNames.MACD:
                    df[ColNames.MACD_NORM] = (
                        df[ColNames.MACD] - df[ColNames.MACD].rolling(window).mean()
                    ) / (df[ColNames.MACD].rolling(window).std() + 1e-6)
                case ColNames.MACD_SIG:
                    df[ColNames.MACD_SIG_NORM] = (
                        df[ColNames.MACD_SIG]
                        - df[ColNames.MACD_SIG].rolling(window).mean()
                    ) / (df[ColNames.MACD_SIG].rolling(window).std() + 1e-6)
                case ColNames.ATR:
                    df[ColNames.ATR_NORM] = df[ColNames.ATR]
                case ColNames.SMA_20:
                    df[ColNames.SMA_20_NORM] = (df[ColNames.SMA_20] - 1) * 10
                case ColNames.SMA_50:
                    df[ColNames.SMA_50_NORM] = (df[ColNames.SMA_50] - 1) * 10
                case ColNames.BB_LOWER:
                    df[ColNames.BB_LOWER_NORM] = (df[ColNames.BB_LOWER] - 1) * 10
                case ColNames.BB_UPPER:
                    df[ColNames.BB_UPPER_NORM] = (df[ColNames.BB_UPPER] - 1) * 10
                case ColNames.SENTIMENT:
                    pass
                case ColNames.SENTIMENT_VOL:
                    news_log = np.log1p(df[ColNames.SENTIMENT_VOL])

                    rolling_mean = news_log.rolling(window=90, min_periods=1).mean()
                    rolling_std = news_log.rolling(window=90, min_periods=1).std(ddof=0)

                    expanding_mean = news_log.expanding(min_periods=2).mean()
                    expanding_std = news_log.expanding(min_periods=2).std(ddof=0)

                    # Fill the "Cold Start" gaps with the growing history
                    final_mean = rolling_mean.fillna(expanding_mean)
                    final_std = rolling_std.fillna(expanding_std)

                    df[ColNames.SENTIMENT_VOL_NORM] = (news_log - final_mean) / (
                        final_std + 1e-6
                    )
                case ColNames.TARGET_O:
                    # for targets current close serves as their previous.
                    df[ColNames.TARGET_O_NORM] = normalize_price(
                        df[ColNames.TARGET_O], curr_close
                    )
                case ColNames.TARGET_H:
                    df[ColNames.TARGET_H_NORM] = normalize_price(
                        df[ColNames.TARGET_H], curr_close
                    )
                case ColNames.TARGET_L:
                    df[ColNames.TARGET_L_NORM] = normalize_price(
                        df[ColNames.TARGET_L], curr_close
                    )
                case ColNames.TARGET_C:
                    df[ColNames.TARGET_C_NORM] = normalize_price(
                        df[ColNames.TARGET_C], curr_close
                    )
                case _:
                    raise NotImplementedError(
                        f"Something Unexpected happened. There is no implementation to normalize such feature ({feat})."
                    )
                
        return df

def normalize_price(price, last_close, multiplier=100):
        """Vectorized log-return normalization for price columns.

        Parameters:
            price (pd.Series | np.ndarray): Prices to normalize.
            last_close (pd.Series | np.ndarray): Reference previous-close prices.

        Returns:
            pd.Series | np.ndarray: Normalized values (log returns scaled).
        """
        return np.log(price / last_close) * multiplier

def unnormalize_price(value, last_close, multiplier=100):
    """Invert `normalize_price` and return original-scale prices.

    Parameters:
        value (pd.Series | np.ndarray): Normalized value(s) to invert.
        last_close (pd.Series | np.ndarray): Reference price(s) used when
            the value was originally normalized.

    Returns:
        pd.Series | np.ndarray: Reconstructed price values.
    """
    return np.exp(value / multiplier) * last_close

if __name__ == "__main__":
    dataset = MarketDataset.load("GLD")
    # dataset.show_dataframe_interactive(options=["all"])
    print(dataset.form_infere_sample(180.0, 185.0, 190.0, 175.0, 1500000.0, 0.5, 10,
                               dataset._market_cols, dataset._sent_cols))
