import logging
import yfinance as yf
import os
import pandas as pd

logger = logging.getLogger("BRAIN")


def fetchStockOHLCV(stock: str,
                    start_date: str,
                    end_date: str,
                    output_dir: str):
    """Download OHLCV for `stock` and save CSV to `output_dir`.

    Args:
        stock: Ticker symbol (e.g., 'GLD').
        start_date: ISO date string for start (YYYY-MM-DD).
        end_date: ISO date string for end (YYYY-MM-DD).
        output_dir: Directory to write CSV into (created if missing).

    Returns:
        None (writes CSV file to disk).
    """
    df = fetchStockOHLCVdataframe(stock, start_date, end_date)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{stock}.csv")

    df.to_csv(output_path, sep=",", index=False)

    print(f"Saved to {output_path}")
    print(df.head())


def fetchStockOHLCVdataframe(stock: str, start_date: str, end_date: str):
    """Return a DataFrame of OHLCV data fetched via `yfinance`.

    Ensures Date column is parsed as datetime and set as the index.
    """
    # Expand doc: arguments and return
    """Args:
        stock: Ticker symbol to fetch.
        start_date: ISO start date string.
        end_date: ISO end date string.

    Returns:
        pandas.DataFrame: OHLCV DataFrame indexed by Date. Empty DataFrame if fetch failed.
    """
    logger.info(f"Fetching OHLCV data for {stock} from {start_date} to {end_date}.")

    # Use multi_level_index=False to ensure clean column names
    df = yf.download(stock, start=start_date, end=end_date, auto_adjust=True, multi_level_index=False)

    if df.empty:
        logger.warning(f"No data fetched for {stock}.")
        return pd.DataFrame()

    # Standardize and ensure Date is datetime index
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    logger.info(f"Fetched {len(df)} rows for {stock}. Index range: {df.index.min()} to {df.index.max()}")

    return df