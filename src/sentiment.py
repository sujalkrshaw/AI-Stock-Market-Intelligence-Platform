from textblob import TextBlob

# ==========================================
# SENTIMENT ANALYSIS
# ==========================================

def analyze_sentiment(text):

    analysis = TextBlob(text)

    polarity = analysis.sentiment.polarity

    # ======================================
    # SENTIMENT LABEL
    # ======================================

    if polarity > 0:

        return "Positive"

    elif polarity < 0:

        return "Negative"

    else:

        return "Neutral"

# ==========================================
# SAMPLE NEWS DATA
# ==========================================

def get_sample_news(stock):

    sample_news = [

        f"{stock} stock surges after strong quarterly earnings.",

        f"Analysts remain optimistic about {stock} growth potential.",

        f"{stock} faces temporary market volatility amid global uncertainty.",

        f"Investors continue monitoring {stock} market performance.",

        f"{stock} announces strategic expansion plans."
    ]

    return sample_news