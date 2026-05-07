from fastapi import FastAPI

from src.data_loader import load_stock_data

from src.preprocessing import preprocess_data

from src.prediction import (

    load_model,

    predict_latest
)

from src.sentiment import (

    analyze_sentiment,

    get_sample_news
)

# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(

    title="AI Stock Market Intelligence API",

    description="Professional Financial AI Backend",

    version="2.0"
)

# ==========================================
# LOAD MODEL
# ==========================================

try:

    model = load_model()

except Exception as e:

    print("MODEL LOAD ERROR:")

    print(e)

    model = None

# ==========================================
# HOME ROUTE
# ==========================================

@app.get("/")

def home():

    return {

        "message":
        "AI Financial Intelligence API Running Successfully"
    }

# ==========================================
# ANALYZE STOCK
# ==========================================

@app.get("/analyze/{ticker}")

def analyze_stock(ticker: str):

    try:

        # ==================================
        # LOAD DATA
        # ==================================

        df = load_stock_data(ticker)

        df = preprocess_data(df)

        # ==================================
        # EMPTY CHECK
        # ==================================

        if df.empty:

            return {

                "error":
                "No stock data found"
            }

        # ==================================
        # LATEST ROW
        # ==================================

        latest_row = df.iloc[-1]

        # ==================================
        # AI PREDICTION
        # ==================================

        if model is not None:

            prediction, probability, signal = predict_latest(

                model,

                latest_row
            )

            prediction = int(prediction)

            probability = float(probability)

            signal = str(signal)

            prediction_label = (

                "UPTREND"
                if prediction == 1
                else "DOWNTREND"
            )

        else:

            prediction_label = "MODEL ERROR"

            probability = 0.0

            signal = "HOLD"

        # ==================================
        # NEWS SENTIMENT
        # ==================================

        news_data = get_sample_news(
            ticker
        )

        sentiments = []

        for news in news_data:

            sentiment = analyze_sentiment(
                news
            )

            sentiments.append({

                "headline": str(news),

                "sentiment": str(sentiment)
            })

        # ==================================
        # SAFE FLOAT CONVERSION
        # ==================================

        latest_price = float(

            round(
                float(latest_row["Close"]),
                2
            )
        )

        volatility = float(

            round(
                float(latest_row["Volatility"]) * 100,
                2
            )
        )

        rsi = float(

            round(
                float(latest_row["RSI"]),
                2
            )
        )

        # ==================================
        # FINAL RESPONSE
        # ==================================

        return {

            "ticker": str(ticker),

            "latest_price": latest_price,

            "prediction": str(prediction_label),

            "confidence": probability,

            "signal": signal,

            "volatility": volatility,

            "rsi": rsi,

            "market_sentiment": sentiments
        }

    except Exception as e:

        return {

            "error": str(e)
        }