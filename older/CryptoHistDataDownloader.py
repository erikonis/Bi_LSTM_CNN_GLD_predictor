# Kevin Hang
# BSP05 - CryptoHistDataDownloader.py
import yfinance as yf

# Please use tickers that work on Yahoo Finance
cryptos = [
    "BTC-USD",
    "ADA-USD",
    "LTC-USD",
    "LINK-USD",
    "BNB-USD",
    "VET-USD",
    "ETH-USD",
    "CRO-USD",
    "DOGE-USD",
    "USDT-USD",
    "XRP-USD",
    "AVAX-USD",
]

print(
    "Available cryptocurrencies: BTC, ADA, LTC, LINK, BNB, VET, ETH, CRO, DOGE, USDT, XRP, AVAX"
)
print(
    "Historical data will be downloaded as a CSV file in the same directory as this file."
)
userInput = input(
    "Please enter '1' to be able to select a cryptocurrency; '2' to download all.\n> "
)


# function to download the historical data of the crypto and store it as a CSV
def dwToCSV(crypto):
    crypto1 = yf.download(crypto, period="max")
    crypto1 = crypto1[["Close", "Open", "High", "Low"]]
    crypto1.index = crypto1.index.strftime("%d/%m/%Y")
    crypto1.index.names = ["Timestamp"]
    crypto1 = crypto1.rename(
        columns={
            "Close": "Closing Price (USD)",
            "Open": "24h Open (USD)",
            "High": "24h High (USD)",
            "Low": "24h Low (USD)",
        }
    )
    crypto1.to_csv(crypto + ".csv", sep="\t")


# download historical data for all the cryptos in cryptos array
if userInput == "2":
    for crypto in cryptos:
        dwToCSV(crypto)

if userInput == "1":
    while True:
        i = input(
            "Please select the cryptocurrency: (enter 'q' to quit) or ('2' to download all)\n> "
        )
        # checks what the input is and downloads the correct crypto
        if i.upper() == "BTC":
            dwToCSV("BTC-USD")
        elif i.upper() == "ADA":
            dwToCSV("ADA-USD")
        elif i.upper() == "LTC":
            dwToCSV("LTC-USD")
        elif i.upper() == "LINK":
            dwToCSV("LINK-USD")
        elif i.upper() == "BNB":
            dwToCSV("BNB-USD")
        elif i.upper() == "VET":
            dwToCSV("VET-USD")
        elif i.upper() == "ETH":
            dwToCSV("ETH-USD")
        elif i.upper() == "CRO":
            dwToCSV("CRO-USD")
        elif i.upper() == "DOGE":
            dwToCSV("DOGE-USD")
        elif i.upper() == "USDT":
            dwToCSV("USDT-USD")
        elif i.upper() == "XRP":
            dwToCSV("XRP-USD")
        elif i.upper() == "AVAX":
            dwToCSV("AVAX-USD")
        elif i.upper() == "2":
            for crypto in cryptos:
                dwToCSV(crypto)
        elif i == "q":
            break
        else:
            print(
                "Cryptocurrency not found. Please make sure that you select a cryptocurrency from the list."
            )
