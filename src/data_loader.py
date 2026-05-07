import yfinance as yf
import pandas as pd


def load_stock_data(ticker="AAPL", start="2020-01-01", end="2025-01-01"):

    # Download stock data
    df = yf.download(
        ticker,
        start=start,
        end=end
    )

    # Reset index
    df.reset_index(inplace=True)

    # =====================================
    # FLATTEN MULTIINDEX COLUMNS
    # =====================================

    new_columns = []

    for col in df.columns:

        # If tuple/multiindex
        if isinstance(col, tuple):
            new_columns.append(col[0])

        else:
            new_columns.append(col)

    df.columns = new_columns

    return df