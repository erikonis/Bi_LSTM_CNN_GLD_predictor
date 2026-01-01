import math
from re import sub
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import pandas_ta as ta # For technical indicators
import json, os
import dtale

class MarketDataset(Dataset):
    def __init__(self, data_df:pd.DataFrame, seq_len:int=30) -> None:
        """
        :param data_df: DataFrame containing the merged and processed data
        :param seq_len: Length of the historical window for market data
        """
        self.dataframe = data_df
        
        self.seq_len = seq_len

        self.years = data_df['Year'].values.astype(np.int32)

        # Convert dataframes to numpy for speed
        self.market_values = data_df[[
        'Log_Ret_Open', 'Log_Ret_High', 'Log_Ret_Low', 'Log_Ret_Close', 'Log_Ret_Vol', 'RSI_Z', 'MACD_Z', 'MACD_Sig_Z',
        'SMA_20_Rel', 'SMA_50_Rel', 'BB_Low_Rel', 'BB_High_Rel', 'ATR_Pct']].values.astype(np.float32)
        
        self.sentiment_values = data_df[['Sentiment', 'News_Vol_Z']].values.astype(np.float32)
        
        # Targets are the NEXT day's OHLC (Open, High, Low, Close)
        self.targets = data_df[['Target_Open', 'Target_High', 'Target_Low', 'Target_Close']].values.astype(np.float32)
        
        # Valid indices start from seq_len to ensure we have enough history
        self.valid_indices = np.arange(seq_len, len(data_df))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx:int):
        # Actual index in the dataframe
        curr_idx = self.valid_indices[idx]
        
        # 1. Market Data: The 30-day "Window" (Shape: 30, feature_dim)
        mkt_data = self.market_values[curr_idx - self.seq_len : curr_idx]
        
        # 2. Sentiment Data: Just today's mood (Shape: 4)
        # Note: We use the sentiment from the yesterday to predict Today.
        sent_data = self.sentiment_values[curr_idx-1]
        
        # 3. Target: today's OHLC (Shape: 4)
        target = self.targets[curr_idx-1]
        
        return torch.tensor(mkt_data), torch.tensor(sent_data), torch.tensor(target)

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the whole dataframe."""
        return self.dataframe

    def show_dataframe_interactive(self, subprocess:bool=False) -> None:
        """
        A method useful to conveniently explore the dataset contained within. Usable for debugging, performing analysis and finding outliers.
        It creates a webpage on the local host that can be clicked and be opened in the browser.
        
        :param subprocess: if set to True, the tab is shown asynchronically. Beware that if the code terminates, the window immediately closes too. False prevents that by stopping the control flow.
        :type subprocess: bool
        """
        display = dtale.show(self.dataframe, subprocess=subprocess)
        display.open_browser()

    def save(self, folder_path="saved_dataset") -> None:
        """Serialize/save the MarketDataset object in the specified folder. It is created if it does not exist."""
        
        os.makedirs(folder_path, exist_ok=True)
            
        # Save Numerical Data as binary
        payload = {
            'mkt': torch.from_numpy(self.market_values),
            'sent': torch.from_numpy(self.sentiment_values),
            'targets': torch.from_numpy(self.targets),
            'indices': torch.from_numpy(self.valid_indices),
            'years': torch.from_numpy(self.years),
            'seq_len': self.seq_len
        }
        torch.save(payload, os.path.join(folder_path, "tensors.pt"))
        
        # Save DataFrame as Parquet
        self.dataframe.to_parquet(os.path.join(folder_path, "metadata.parquet"))
        print(f"Dataset successfully saved to {folder_path}")

    @classmethod
    def load(cls, folder_path="saved_dataset"):
        """Deserialize/load: Reconstructs the object from parquet and pickle files found in the specified folder."""
        # 1. Load the DataFrame
        df = pd.read_parquet(os.path.join(folder_path, "metadata.parquet"))
        
        # 2. Load the Tensors
        payload = torch.load(os.path.join(folder_path, "tensors.pt"), weights_only=True)
        
        # Create instance
        instance = cls(df, seq_len=payload['seq_len'])
        
        # Override the values with the loaded tensors to ensure exact precision
        instance.market_values = payload['mkt'].numpy()
        instance.sentiment_values = payload['sent'].numpy()
        instance.targets = payload['targets'].numpy()
        instance.valid_indices = payload['indices'].numpy()
        instance.years = payload['years'].numpy()
        
        return instance

    def get_indices_by_year(self, years_list:list) :
        """
        Returns the indices of the dataset that belong to the specified years.
        
        :param years_list: list of years to filter by
        :type years_list: list
        :return: list of indices corresponding to the specified years
        """
        lst = []
        for idx in self.valid_indices:
            if self.years[idx] in years_list:
                lst.append(idx)
        return lst

    def get_loaders(self, batch_size:int=64, training_setting : str = "expanding_window"):
        """
        A generator that outputs a tuple of loaders `(train_set, val_set, test_set)` for each training cycle.
        For instance, in case of `training_setting == "expanding_window"`, each `next()` call yields new train, val, test split
        with train dataset incremented by 1 year: \n
            1. train 2015-2020, val 2021, test 2022
            2. train 2015-2021, val 2022, test 2023
            and so on...

        Default option is expanding_window, which from the start takes `math.floor(total_years*0.6)` years for the train set.

        :param batch_size: the size of each batch for the training. Applies for train and validation sets.
        :type: int
        :param training_setting: the splitting option. Default is "expanding_window". Putting wrong values raises KeyError. No other is implemented for now.
        :type: str
        :returns: (train_set, validation_set, test_set) upon next() call
        """
        start_year = self.years[0]
        end_year = self.years[-1]
        num_years = len(set(self.years))

        match training_setting:
            case "expanding_window":

                test_years = range(start_year + math.floor(num_years*0.6), end_year + 1)
                
                for test_year in test_years:
                    print(f"\n--- Starting Fold: Test Year {test_year} ---")
                    
                    # Training is everything from start_year up to (but not including) test_year
                    train_years = list(range(start_year, test_year))
                    
                    # Validation is usually the year before the test year, or a subset of training
                    val_year = [test_year - 1] 
                    
                    # 1. Get Indices
                    train_idx = self.get_indices_by_year(train_years)
                    val_idx = self.get_indices_by_year(val_year)
                    test_idx = self.get_indices_by_year([test_year])
                    
                    # 2. Create Subsets and Loaders, make a generator that yields next loaders.
                    yield (
                        DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True), #train
                        DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False),  #validate
                        DataLoader(Subset(dataset, test_idx), batch_size=1, shuffle=False),          #test
                        test_year #metadata - which year we are in
                        )
                    
            # Placeholder for other datasplit options if needed to implement (rolling window, fixed, etc.)
            
            case _ :
                raise KeyError(f"Incorrect argument passed. There is no option '{training_setting}'!")
        
def load_data(csv_path:str, json_path:str) -> pd.DataFrame:
    """
    Load and merge market and sentiment data.
    Market data is in CSV, sentiment data is in JSON.
    Market CSV has columns: 'Date', '24h Open (USD)', '24h High (USD)', 
    '24h Low (USD)', 'Closing Price (USD)', 'Trading Volume'.
    Sentiment JSON is a dict with {date :   [sentiment score, article count]}.
    
    :param csv_path: path to the market CSV file
    :param json_path: path to the sentiment JSON file
    :return: Merged DataFrame with Date, OHLCV, Sentiment, Year. Any missing days are skipped.
    """
    
    # 1. Load Market CSV
    # We map your specific column names to standard OHLCV
    mkt_df = pd.read_csv(csv_path)
    mkt_df['Date'] = pd.to_datetime(mkt_df['Date'])
    mkt_df = mkt_df.rename(columns={
        '24h Open (USD)': 'Open',
        '24h High (USD)': 'High',
        '24h Low (USD)': 'Low',
        'Closing Price (USD)': 'Close',
        'Trading Volume': 'Volume'
    })

    # 2. Load Sentiment JSON
    with open(json_path, 'r') as f:
        sent_dict = json.load(f)
    
    # Convert dict to DataFrame
    rows = []
    for date, lst in sent_dict.items():
        rows.append([date, lst[0], lst[1]])  # single score, count
    sent_df = pd.DataFrame(rows, columns=['Date', 'Sentiment', 'Count'])
    sent_df['Date'] = pd.to_datetime(sent_df['Date'])

    # 3. Merge on Date
    # 'inner' ensures we only keep days where we have BOTH price and sentiment
    df = pd.merge(mkt_df, sent_df, on='Date', how='left').sort_values('Date')
    
    # Add Year column for Walk-Forward splits
    df['Year'] = df['Date'].dt.year

    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Year')))
    df = df[cols]
    return df

def apply_technicals(df:pd.DataFrame) -> pd.DataFrame:
    """
    Takes a pandas DataFrame with one day per row parameters and derives technical indicators for the dataset.
    If there aren't sufficient amount of days, NaN is put.

    Expected column names (at least):\n
        1. 'Close'
        2. 'Low'
        3. 'High'
    
    Applied indicators and added columns: \n
        1. 'RSI' length 14
        2. 'SMA_20_Ratio'
        3. 'SMA_50_Ratio'
        4. 'MACD'
        5. 'MACD_Sig'
        6. 'ATR'
        7. 'BB_Lower_Ratio'
        8. 'BB_Upper_Ratio

    :Returns: the same DataFrame with added technical indicators.
    """

    # Momentum
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # Trend (using ratios to make them scale-invariant)
    df['SMA_20_Ratio'] = df['Close'] / ta.sma(df['Close'], length=20)
    df['SMA_50_Ratio'] = df['Close'] / ta.sma(df['Close'], length=50)
    
    # MACD
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Sig'] = macd['MACDs_12_26_9']
    
    # Volatility (Essential for your High/Low range prediction)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Bollinger Bands
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df['BB_Lower_Ratio'] = bbands.iloc[:, 0] / df['Close']
    df['BB_Upper_Ratio'] = bbands.iloc[:, 2] / df['Close']
    
    return df

def normalize_dataframe(df:pd.DataFrame, window:int=90) -> pd.DataFrame:
    """
    Normalizes a Gold Market dataframe for Neural Network training.
    Assumes dataframe contains all values to which the normalization is applied:\n
        1. 'Open', 'High', 'Low', 'Close' - log returns w.r.t. previous day 'Close'
        2. 'Volume' - log returns w.r.t. previous day 'Volume'
        3. 'Sentiment' - kept as is (naturally bounded -1 to 1)
        4. 'Article_Count' - log(1+x) + Rolling Z-Score
        5. 'RSI', 'MACD', 'MACD_Sig' - Rolling Z-Score
        6. 'ATR' - as % of 'Close' price
        7. 'SMA_20_Ratio', 'SMA_50_Ratio', 'BB_Lower_Ratio', 'BB_Upper_Ratio' - Indicator-1

    Returned columns:\n
        'Date', 'Year', 'Log_Ret_Open', 'Log_Ret_High', 'Log_Ret_Low', 'Log_Ret_Close', 
        'Log_Ret_Vol','Sentiment', 'News_Vol_Z', 'RSI_Z', 'MACD_Z', 'MACD_Sig_Z',
        'SMA_20_Rel', 'SMA_50_Rel', 'BB_Low_Rel', 'BB_High_Rel', 'ATR_Pct',
        'Target_Open', 'Target_High', 'Target_Low', 'Target_Close'
    
    :param df: DataFrame with raw OHLCV, Sentiment, Technicals
    :param window: rolling window size for Z-Score calculations
    :return: Normalized DataFrame ready for training
    """

    # Create a copy to avoid modifying the original dataframe
    pdf = df.copy().sort_index()

    # 1. PRICE & VOLUME: Close-Anchored Log Returns
    # We use shift(1) to anchor today's OHLC to yesterday's Close
    prev_close = pdf['Close'].shift(1)
    pdf['Log_Ret_Open']  = np.log(pdf['Open'] / prev_close)
    pdf['Log_Ret_High']  = np.log(pdf['High'] / prev_close)
    pdf['Log_Ret_Low']   = np.log(pdf['Low'] / prev_close)
    pdf['Log_Ret_Close'] = np.log(pdf['Close'] / prev_close)
    pdf['Log_Ret_Vol']   = np.log(pdf['Volume'] / pdf['Volume'].shift(1))

    # Setting the targets:
    pdf['Target_Open']  = pdf['Log_Ret_Open'].shift(-1)
    pdf['Target_High']  = pdf['Log_Ret_High'].shift(-1)
    pdf['Target_Low']   = pdf['Log_Ret_Low'].shift(-1)
    pdf['Target_Close'] = pdf['Log_Ret_Close'].shift(-1)

    # 2. SENTIMENT: Naturally bounded (-1 to 1), no change needed
    # (Ensure column name matches your dataset)
    pdf['Sentiment'] = pdf['Sentiment'] 

    # 3. NEWS VOLUME: Log(1+x) + Rolling Z-Score
    # Squashes spikes and then centers around 0
    news_log = np.log1p(pdf['Count'])

    # Expanding window for the beginning, rolling for the rest
    rolling_mean = news_log.rolling(window=90, min_periods=1).mean()
    rolling_std  = news_log.rolling(window=90, min_periods=1).std(ddof=0)
    
    expanding_mean = news_log.expanding(min_periods=2).mean()
    expanding_std  = news_log.expanding(min_periods=2).std(ddof=0)

    # Fill the "Cold Start" gaps with the growing history
    final_mean = rolling_mean.fillna(expanding_mean)
    final_std  = rolling_std.fillna(expanding_std)

    pdf['News_Vol_Z'] = (news_log - final_mean) / (final_std + 1e-6)
    
    # 4. OSCILLATORS: Rolling Z-Score
    # Standardizes RSI and MACD to show "extremes" relative to the last 3 months
    for col in ['RSI', 'MACD', 'MACD_Sig']:
        pdf[f'{col}_Z'] = (pdf[col] - pdf[col].rolling(window).mean()) / (pdf[col].rolling(window).std() + 1e-6)

    # 5. TREND RATIOS: (Price / Indicator) - 1
    # Centers the relationship at 0 (0 = Price is exactly on the SMA)
    pdf['SMA_20_Rel'] = pdf['SMA_20_Ratio'] - 1
    pdf['SMA_50_Rel'] = pdf['SMA_50_Ratio'] - 1
    pdf['BB_Low_Rel'] = pdf['BB_Lower_Ratio'] - 1
    pdf['BB_High_Rel'] = pdf['BB_Upper_Ratio'] - 1

    # 6. VOLATILITY: ATR as % of Price
    pdf['ATR_Pct'] = pdf['ATR'] / pdf['Close']

    # --- CLEANUP ---
    # Drop the original unscaled columns so we only keep the processed ones
    cols_to_keep = ['Date', 'Year',
        'Log_Ret_Open', 'Log_Ret_High', 'Log_Ret_Low', 'Log_Ret_Close', 'Log_Ret_Vol',
        'Sentiment', 'News_Vol_Z', 'RSI_Z', 'MACD_Z', 'MACD_Sig_Z',
        'SMA_20_Rel', 'SMA_50_Rel', 'BB_Low_Rel', 'BB_High_Rel', 'ATR_Pct',
        'Target_Open', 'Target_High', 'Target_Low', 'Target_Close'
    ]
    
    final_df = pdf[cols_to_keep]

    # Drop the first 'window' rows (the burn-in period) to remove NaNs
    final_df = final_df.dropna()

    return final_df

def form_dataset(csv_path:str, json_path:str, seq_len:int=30):
    """
    Loads, processes, and forms a MarketDataset ready for training.
    
    :param csv_path: path to the market CSV file
    :param json_path: path to the sentiment JSON file
    :param seq_len: length of historical window for market data
    :return: MarketDataset instance
    """
    # Load and merge data
    df = load_data(csv_path, json_path)
    
    # Apply technical indicators
    df = apply_technicals(df)
    
    # Normalize and prepare features
    df = normalize_dataframe(df)
    
    # Create Dataset
    dataset = MarketDataset(df, seq_len=seq_len)
    
    return dataset

if __name__ == "__main__":

    dataset = MarketDataset.load()

    for i, k, l, c in dataset.get_loaders():
        print("--- New Fold ---")
        print(f"{i}")
        print(f"{k}")
        print(f"{l}")
        print(f"{c}")