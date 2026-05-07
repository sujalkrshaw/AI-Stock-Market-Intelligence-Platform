import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator

from ta.trend import MACD

from ta.volatility import BollingerBands

# ==========================================
# PREPROCESS DATA
# ==========================================

def preprocess_data(df):

    # ======================================
    # CLEAN COLUMN NAMES
    # ======================================

    clean_columns = []

    for col in df.columns:

        if isinstance(col, tuple):

            clean_columns.append(col[0])

        else:

            clean_columns.append(str(col))

    df.columns = clean_columns

    # ======================================
    # REMOVE NULLS
    # ======================================

    df.dropna(inplace=True)

    # ======================================
    # DAILY RETURN
    # ======================================

    df["Daily_Return"] = (

        df["Close"].pct_change()
    )

    # ======================================
    # VOLATILITY
    # ======================================

    df["Volatility"] = (

        df["Daily_Return"]
        .rolling(20)
        .std()
    )

    # ======================================
    # SMA
    # ======================================

    df["MA20"] = (

        df["Close"]
        .rolling(20)
        .mean()
    )

    df["MA50"] = (

        df["Close"]
        .rolling(50)
        .mean()
    )

    # ======================================
    # EMA
    # ======================================

    df["EMA20"] = (

        df["Close"]
        .ewm(span=20)
        .mean()
    )

    # ======================================
    # PRICE FEATURES
    # ======================================

    df["Price_Change"] = (

        df["Close"] - df["Open"]
    )

    df["HL_Spread"] = (

        df["High"] - df["Low"]
    )

    # ======================================
    # RSI
    # ======================================

    rsi_indicator = RSIIndicator(

        close=df["Close"],

        window=14
    )

    df["RSI"] = rsi_indicator.rsi()

    # ======================================
    # MACD
    # ======================================

    macd_indicator = MACD(

        close=df["Close"]
    )

    df["MACD"] = (

        macd_indicator.macd()
    )

    df["MACD_Signal"] = (

        macd_indicator.macd_signal()
    )

    # ======================================
    # BOLLINGER BANDS
    # ======================================

    bb_indicator = BollingerBands(

        close=df["Close"],

        window=20
    )

    df["BB_High"] = (

        bb_indicator.bollinger_hband()
    )

    df["BB_Low"] = (

        bb_indicator.bollinger_lband()
    )

    # ======================================
    # MOMENTUM SCORE
    # ======================================

    df["Momentum"] = (

        df["Close"] -
        df["Close"].shift(10)
    )

    # ======================================
    # TREND STRENGTH
    # ======================================

    df["Trend_Strength"] = (

        df["MA20"] -
        df["MA50"]
    )

    # ======================================
    # BUY / SELL SIGNALS
    # ======================================

    df["Signal"] = "HOLD"

    df.loc[
        df["MA20"] > df["MA50"],
        "Signal"
    ] = "BUY"

    df.loc[
        df["MA20"] < df["MA50"],
        "Signal"
    ] = "SELL"

    # ======================================
    # RISK LEVEL
    # ======================================

    df["Risk_Level"] = "LOW"

    df.loc[
        df["Volatility"] > 0.03,
        "Risk_Level"
    ] = "HIGH"

    df.loc[
        (
            df["Volatility"] > 0.015
        ) &
        (
            df["Volatility"] <= 0.03
        ),
        "Risk_Level"
    ] = "MEDIUM"

    # ======================================
    # REMOVE NULLS
    # ======================================

    df.dropna(inplace=True)

    # ======================================
    # RESET INDEX
    # ======================================

    df.reset_index(
        drop=True,
        inplace=True
    )

    # ======================================
    # FINAL DEBUG
    # ======================================

    print("\nFINAL FEATURES:\n")

    print(df.columns)

    return df