import pandas as pd

def main():
    stock_csv = "stock_data/META.csv"
    df = pd.read_csv(stock_csv, index_col=0)
    print(df.head())
    print(df.describe())

if __name__ == "__main__":
    main()
    