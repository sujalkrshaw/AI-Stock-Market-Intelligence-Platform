import pandas as pd

# ==========================================
# TEMP STORAGE
# ==========================================

portfolio_data = []

# ==========================================
# ADD STOCK
# ==========================================

def add_stock(

    stock,
    quantity,
    buy_price,
    current_price
):

    investment = quantity * buy_price

    current_value = quantity * current_price

    profit_loss = (

        current_value - investment
    )

    return_percentage = (

        (profit_loss / investment) * 100
    )

    portfolio_data.append({

        "Stock": stock,

        "Quantity": quantity,

        "Buy Price": round(
            buy_price,
            2
        ),

        "Current Price": round(
            current_price,
            2
        ),

        "Investment": round(
            investment,
            2
        ),

        "Current Value": round(
            current_value,
            2
        ),

        "Profit/Loss": round(
            profit_loss,
            2
        ),

        "Return %": round(
            return_percentage,
            2
        )
    })

# ==========================================
# GET PORTFOLIO
# ==========================================

def get_portfolio():

    return pd.DataFrame(
        portfolio_data
    )

# ==========================================
# SUMMARY
# ==========================================

def portfolio_summary(df):

    if df.empty:

        return {

            "Investment": 0,

            "Current Value": 0,

            "Profit": 0,

            "Return %": 0
        }

    total_investment = round(

        df["Investment"].sum(),

        2
    )

    current_value = round(

        df["Current Value"].sum(),

        2
    )

    total_profit = round(

        df["Profit/Loss"].sum(),

        2
    )

    total_return = round(

        (
            total_profit /
            total_investment
        ) * 100,

        2
    )

    return {

        "Investment": total_investment,

        "Current Value": current_value,

        "Profit": total_profit,

        "Return %": total_return
    }

# ==========================================
# DELETE STOCK
# ==========================================

def delete_stock(stock_id):

    global portfolio_data

    if (

        stock_id > 0 and
        stock_id <= len(portfolio_data)

    ):

        portfolio_data.pop(
            stock_id - 1
        )

# ==========================================
# CLEAR
# ==========================================

def clear_portfolio():

    global portfolio_data

    portfolio_data = []