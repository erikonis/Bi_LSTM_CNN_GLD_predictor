import yfinance as yf
import os

# function to download the historical data of the crypto and store it as a CSV
def fetchStockOHLCV(stock: str,
                    start_date: str,
                    end_date: str,
                    output_dir: str):

    # Fetch OHLCV
    df = yf.download(stock, start=start_date, end=end_date, auto_adjust=True)

    if df.empty:
        print("No data fetched.")
        return

    # Keep only OHLCV
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Rename columns
    df.columns = [
        "24h Open (USD)",
        "24h High (USD)",
        "24h Low (USD)",
        "Closing Price (USD)",
        "Trading Volume"
    ]

    # ove index to column "Date"
    df = df.reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Reorder columns
    df = df[[
        "Date",
        "24h Open (USD)",
        "24h High (USD)",
        "24h Low (USD)",
        "Closing Price (USD)",
        "Trading Volume"
    ]]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{stock}.csv")

    # Save as CSV
    df.to_csv(output_path, sep=",", index=False)

    print(f"Saved to {output_path}")
    print(df.head())

if __name__ == "__main__":
    stock = "GLD"
    fetchStockOHLCV(stock, "2014-11-30", "2025-11-30", "stock_data")