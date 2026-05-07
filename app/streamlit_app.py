import sys
import os

# ==========================================
# PYTHON PATH FIX
# ==========================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..'
        )
    )
)

# ==========================================
# IMPORTS
# ==========================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from streamlit_autorefresh import st_autorefresh

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

from src.portfolio import (
    add_stock,
    get_portfolio,
    portfolio_summary,
    delete_stock,
    clear_portfolio
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Stock Market Intelligence Platform",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# AUTO REFRESH
# ==========================================

st_autorefresh(
    interval=60000,
    key="market_refresh"
)

# ==========================================
# PREMIUM CSS
# ==========================================
# Premium 3D Dashboard Upgrade for `app/streamlit_app.py`


st.markdown("""
<style>

/* =========================================
GLOBAL APP
========================================= */

.stApp {

    background:
        radial-gradient(circle at top left, #0f172a 0%, #020617 45%, #000000 100%);

    color: white;

    font-family: 'Inter', sans-serif;
}

/* =========================================
ANIMATED GLOW ORBS
========================================= */

.stApp::before {

    content: "";

    position: fixed;

    width: 600px;

    height: 600px;

    background: radial-gradient(circle, rgba(0,255,255,0.18), transparent 70%);

    top: -250px;

    left: -200px;

    filter: blur(40px);

    z-index: -1;

    animation: glowMove 8s ease-in-out infinite;
}

.stApp::after {

    content: "";

    position: fixed;

    width: 500px;

    height: 500px;

    background: radial-gradient(circle, rgba(255,0,255,0.16), transparent 70%);

    bottom: -180px;

    right: -150px;

    filter: blur(40px);

    z-index: -1;

    animation: glowMove2 10s ease-in-out infinite;
}

@keyframes glowMove {

    0% {transform: translateY(0px) scale(1);}

    50% {transform: translateY(30px) scale(1.1);}

    100% {transform: translateY(0px) scale(1);}
}

@keyframes glowMove2 {

    0% {transform: translateX(0px) scale(1);}

    50% {transform: translateX(-30px) scale(1.08);}

    100% {transform: translateX(0px) scale(1);}
}

/* =========================================
SIDEBAR
========================================= */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(180deg, rgba(15,23,42,0.96), rgba(2,6,23,0.98));

    border-right: 1px solid rgba(0,255,255,0.18);

    backdrop-filter: blur(22px);

    box-shadow: 0 0 30px rgba(0,255,255,0.08);
}

/* =========================================
TITLE
========================================= */

h1 {

    color: #ffffff;

    font-size: 3.3rem;

    font-weight: 900;

    text-shadow:
        0 0 12px rgba(0,255,255,0.8),
        0 0 28px rgba(0,255,255,0.4);
}

h2, h3 {

    color: #f8fafc;
}

/* =========================================
3D KPI CARDS
========================================= */

[data-testid="stMetric"] {

    background: linear-gradient(
        145deg,
        rgba(17,25,40,0.88),
        rgba(15,23,42,0.92)
    );

    border: 1px solid rgba(0,255,255,0.18);

    border-radius: 28px;

    padding: 22px;

    backdrop-filter: blur(24px);

    box-shadow:
        0 12px 30px rgba(0,0,0,0.45),
        inset 0 1px 0 rgba(255,255,255,0.05),
        0 0 25px rgba(0,255,255,0.08);

    transition: all 0.35s ease;

    transform-style: preserve-3d;
}

[data-testid="stMetric"]:hover {

    transform:
        perspective(1200px)
        rotateX(4deg)
        rotateY(-4deg)
        translateY(-8px)
        scale(1.02);

    box-shadow:
        0 18px 45px rgba(0,255,255,0.22),
        0 0 40px rgba(0,255,255,0.18);
}

/* =========================================
BUTTONS
========================================= */

.stButton button {

    background:
        linear-gradient(135deg, #00c6ff, #0072ff);

    border: none;

    border-radius: 16px;

    color: white;

    font-weight: 700;

    padding: 0.7rem 1.3rem;

    box-shadow:
        0 10px 20px rgba(0,114,255,0.35),
        0 0 18px rgba(0,198,255,0.3);

    transition: all 0.3s ease;
}

.stButton button:hover {

    transform:
        translateY(-4px)
        scale(1.03);

    box-shadow:
        0 16px 30px rgba(0,198,255,0.45),
        0 0 28px rgba(0,198,255,0.5);
}

/* =========================================
TABS
========================================= */

button[data-baseweb="tab"] {

    background: rgba(255,255,255,0.05);

    border-radius: 14px;

    margin-right: 10px;

    color: white;

    border: 1px solid rgba(255,255,255,0.06);

    transition: all 0.25s ease;
}

button[data-baseweb="tab"]:hover {

    background: rgba(0,255,255,0.15);

    transform: translateY(-2px);
}

/* =========================================
PLOTLY CHART CONTAINER
========================================= */

.js-plotly-plot {

    border-radius: 26px;

    overflow: hidden;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
        0 14px 40px rgba(0,0,0,0.35),
        0 0 25px rgba(0,255,255,0.08);

    background: rgba(15,23,42,0.55);

    backdrop-filter: blur(18px);
}

/* =========================================
DATAFRAMES
========================================= */

[data-testid="stDataFrame"] {

    border-radius: 22px;

    overflow: hidden;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
        0 10px 25px rgba(0,0,0,0.25);
}

/* =========================================
SCROLLBAR
========================================= */

::-webkit-scrollbar {

    width: 10px;
}

::-webkit-scrollbar-thumb {

    background:
        linear-gradient(180deg, #00ffff, #0072ff);

    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)



# ==========================================
# TITLE
# ==========================================

st.title("🚀 AI Stock Market Intelligence Platform")

st.markdown("""
### Real-Time AI Financial Intelligence Dashboard
""")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("📊 Dashboard Controls")

watchlist = st.sidebar.multiselect(

    "Select Stocks",

    [
        "TCS.NS",
        "INFY.NS",
        "RELIANCE.NS",
        "HDFCBANK.NS",
        "AAPL",
        "TSLA",
        "NVDA",
        "MSFT"
    ],

    default=["TCS.NS"]
)

ticker = watchlist[0]

# ==========================================
# LOAD DATA
# ==========================================

df = load_stock_data(ticker)

df = preprocess_data(df)

# ==========================================
# LOAD MODEL
# ==========================================

try:

    model = load_model()

except:

    model = None

# ==========================================
# KPI CARDS
# ==========================================

latest_price = round(
    float(df["Close"].iloc[-1]),
    2
)

avg_return = round(
    float(df["Daily_Return"].mean() * 100),
    2
)

volatility = round(
    float(df["Volatility"].mean() * 100),
    2
)

high_price = round(
    float(df["High"].max()),
    2
)

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "💲 Latest Price",
    f"${latest_price}"
)

k2.metric(
    "📈 Avg Return",
    f"{avg_return}%"
)

k3.metric(
    "⚠ Volatility",
    f"{volatility}%"
)

k4.metric(
    "🚀 Highest Price",
    f"${high_price}"
)

# ==========================================
# TABS
# ==========================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([

    "📈 Live Market",

    "📊 Technical Analysis",

    "🤖 AI Prediction",

    "📋 Watchlist",

    "💼 Portfolio",

    "📊 Analytics",

    "🗂 Dataset",

    "📰 Sentiment"
])

# ==========================================
# TAB 1 — LIVE MARKET
# ==========================================

with tab1:

    st.subheader("📈 Live Market Dashboard")

    live_fig = go.Figure()

    for stock in watchlist:

        temp_df = load_stock_data(stock)

        temp_df = preprocess_data(temp_df)

        latest = temp_df["Close"].iloc[-1]

        previous = temp_df["Close"].iloc[-2]

        color = '#00ff99'

        if latest < previous:

            color = '#ff3366'

        live_fig.add_trace(

            go.Scatter(

                x=temp_df["Date"],

                y=temp_df["Close"],

                mode='lines',

                name=stock,

                line=dict(
                    color=color,
                    width=3
                )
            )
        )

    live_fig.update_layout(

        template="plotly_dark",

         height=650,

         title="⚡ Live Market Dashboard",

         paper_bgcolor='rgba(0,0,0,0)',

         plot_bgcolor='rgba(0,0,0,0)',

         font=dict(color='white'),

         hovermode='x unified'
    )

    st.plotly_chart(
        live_fig,
        use_container_width=True
    )

# ==========================================
# TAB 2 — TECHNICAL ANALYSIS
# ==========================================

with tab2:

    st.subheader("📊 Technical Analysis")

    tech_fig = go.Figure()

    tech_fig.add_trace(

        go.Scatter(

            x=df["Date"],

            y=df["Close"],

            name="Close Price"
        )
    )

    tech_fig.add_trace(

        go.Scatter(

            x=df["Date"],

            y=df["MA20"],

            name="MA20"
        )
    )

    tech_fig.add_trace(

        go.Scatter(

            x=df["Date"],

            y=df["MA50"],

            name="MA50"
        )
    )

    tech_fig.add_trace(

        go.Scatter(

            x=df["Date"],

            y=df["EMA20"],

            name="EMA20"
        )
    )

    tech_fig.update_layout(

         template="plotly_dark",

         height=650,

         paper_bgcolor='rgba(0,0,0,0)',

         plot_bgcolor='rgba(0,0,0,0)',

         font=dict(color='white'),

         hovermode='x unified'
    )



    st.plotly_chart(
        tech_fig,
        use_container_width=True
    )

    rsi_fig = px.line(
        df,
        x="Date",
        y="RSI",
        title="RSI Indicator"
    )

    rsi_fig.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        rsi_fig,
        use_container_width=True
    )

# ==========================================
# TAB 3 — AI PREDICTION
# ==========================================

with tab3:

    st.subheader("🤖 AI Prediction Engine")

    if model is not None:

        latest_row = df.iloc[-1]

        prediction, probability, signal = predict_latest(
            model,
            latest_row
        )

        label = "UPTREND"

        if prediction == 0:

            label = "DOWNTREND"

        a1, a2, a3 = st.columns(3)

        a1.metric(
            "Prediction",
            label
        )

        a2.metric(
            "Confidence",
            f"{probability}%"
        )

        a3.metric(
            "Signal",
            signal
        )

# ==========================================
# TAB 4 — WATCHLIST
# ==========================================

with tab4:

    st.subheader("📋 Watchlist")

    watchlist_data = []

    for stock in watchlist:

        temp_df = load_stock_data(stock)

        temp_df = preprocess_data(temp_df)

        watchlist_data.append({

            "Stock": stock,

            "Latest Price": round(
                float(temp_df["Close"].iloc[-1]),
                2
            ),

            "Volatility": round(
                float(
                    temp_df["Volatility"].mean()
                ) * 100,
                2
            )
        })

    watchlist_df = pd.DataFrame(
        watchlist_data
    )

    st.dataframe(
        watchlist_df,
        use_container_width=True
    )

# ==========================================
# TAB 5 — PORTFOLIO
# ==========================================

with tab5:

    st.subheader("💼 Portfolio Tracker")

    portfolio_stock = st.selectbox(
        "Select Stock",
        watchlist
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=10
    )

    buy_price = st.number_input(
        "Buy Price",
        min_value=1.0,
        value=100.0
    )

    current_price = float(
        df["Close"].iloc[-1]
    )

    if st.button("➕ Add Stock"):

        add_stock(

            portfolio_stock,

            quantity,

            buy_price,

            current_price
        )

        st.success("Stock added!")

    portfolio_df = get_portfolio()

    if not portfolio_df.empty:

        st.dataframe(
            portfolio_df,
            use_container_width=True
        )

        summary = portfolio_summary(
            portfolio_df
        )

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "Investment",
            f"${summary['Investment']}"
        )

        p2.metric(
            "Current Value",
            f"${summary['Current Value']}"
        )

        p3.metric(
            "Profit",
            f"${summary['Profit']}"
        )

        p4.metric(
            "Return %",
            f"{summary['Return %']}%"
        )


# ==========================================
# TAB 6 — ANALYTICS
# ==========================================

with tab6:

    st.subheader("📊 Portfolio Analytics")

    portfolio_df = get_portfolio()

    if not portfolio_df.empty:

        c1, c2 = st.columns(2)

        # ==================================
        # PIE CHART
        # ==================================

        with c1:

            pie_fig = px.pie(

                portfolio_df,

                names="Stock",

                values="Investment",

                hole=0.45,

                color_discrete_sequence=[

                    '#00FFFF',
                    '#00FF99',
                    '#FFD700',
                    '#FF00FF',
                    '#FF4500',
                    '#7B68EE',
                    '#1E90FF'
                ]
            )

            pie_fig.update_traces(

                textposition='inside',

                textinfo='percent+label'
            )

            pie_fig.update_layout(

                template="plotly_dark",

                height=500,

                paper_bgcolor='rgba(0,0,0,0)',

                plot_bgcolor='rgba(0,0,0,0)',

                font=dict(
                    color='white',
                    size=14
                ),

                hovermode='x unified'
            )

            st.plotly_chart(

                pie_fig,

                use_container_width=True
            )

        # ==================================
        # BAR CHART
        # ==================================

        with c2:

            bar_fig = px.bar(

                portfolio_df,

                x="Stock",

                y="Profit/Loss",

                color="Profit/Loss",

                text_auto=True,

                color_continuous_scale=[

                    "#ff0000",

                    "#ffff00",

                    "#00ff99"
                ]
            )

            bar_fig.update_layout(

                template="plotly_dark",

                height=500,

                paper_bgcolor='rgba(0,0,0,0)',

                plot_bgcolor='rgba(0,0,0,0)',

                font=dict(
                    color='white',
                    size=14
                ),

                hovermode='x unified'
            )

            st.plotly_chart(

                bar_fig,

                use_container_width=True
            )

        # ==================================
        # RETURN DISTRIBUTION
        # ==================================

        st.markdown("---")

        hist_fig = px.histogram(

            portfolio_df,

            x="Return %",

            nbins=20,

            title="📈 Return Distribution",

            color_discrete_sequence=['#00FFFF']
        )

        hist_fig.update_layout(

            template="plotly_dark",

            height=500,

            paper_bgcolor='rgba(0,0,0,0)',

            plot_bgcolor='rgba(0,0,0,0)',

            font=dict(color='white')
        )

        st.plotly_chart(

            hist_fig,

            use_container_width=True
        )

    else:

        st.warning(

            "⚠ Add stocks in Portfolio tab to view analytics."
        )
# ==========================================
# TAB 7 — DATASET
# ==========================================

with tab7:

    st.subheader("🗂 Dataset")

    st.dataframe(
        df.tail(50),
        use_container_width=True
    )

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="📥 Download Dataset",

        data=csv,

        file_name=f"{ticker}_dataset.csv",

        mime="text/csv"
    )

# ==========================================
# TAB 8 — SENTIMENT
# ==========================================

with tab8:

    st.subheader("📰 AI Sentiment Analysis")

    news = get_sample_news(ticker)

    for item in news:

        sentiment = analyze_sentiment(item)

        if sentiment == "Positive":

            st.success(item)

        elif sentiment == "Negative":

            st.error(item)

        else:

            st.warning(item)

        st.markdown(
            f"### Sentiment: {sentiment}"
        )

        st.markdown("---")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""

<div style='text-align:center;'>

<h3>
🚀 AI Stock Market Intelligence Platform
</h3>

<p>
Built with Streamlit • XGBoost • NLP • Financial AI
</p>

</div>

""", unsafe_allow_html=True)