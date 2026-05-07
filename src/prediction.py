import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# ==========================================
# MODEL COMPARISON
# ==========================================

def compare_models(

    X_train,
    X_test,
    y_train,
    y_test
):

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "Random Forest": RandomForestClassifier(

            n_estimators=250,

            random_state=42
        ),

        "XGBoost": XGBClassifier(

            n_estimators=400,

            learning_rate=0.03,

            max_depth=6,

            subsample=0.85,

            colsample_bytree=0.85,

            random_state=42,

            eval_metric="logloss"
        )
    }

    results = []

    print("\n==============================")

    print("MODEL COMPARISON")

    print("==============================")

    for name, model in models.items():

        # ==================================
        # TRAIN
        # ==================================

        model.fit(
            X_train,
            y_train
        )

        # ==================================
        # PREDICT
        # ==================================

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        # ==================================
        # METRICS
        # ==================================

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions
        )

        recall = recall_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        results.append({

            "Model": name,

            "Accuracy": round(
                accuracy,
                4
            ),

            "Precision": round(
                precision,
                4
            ),

            "Recall": round(
                recall,
                4
            ),

            "F1 Score": round(
                f1,
                4
            ),

            "ROC-AUC": round(
                roc_auc,
                4
            )
        })

    results_df = pd.DataFrame(
        results
    )

    print(results_df)

    return results_df

# ==========================================
# TRAIN MODEL
# ==========================================

def train_model(df):

    # ======================================
    # TARGET
    # ======================================

    df["Target"] = (

        df["Close"].shift(-1) >
        df["Close"]

    ).astype(int)

    # ======================================
    # DROP NULLS
    # ======================================

    df.dropna(inplace=True)

    # ======================================
    # FEATURES
    # ======================================

    features = [

        "Daily_Return",

        "Volatility",

        "MA20",

        "MA50",

        "EMA20",

        "Price_Change",

        "HL_Spread",

        "RSI",

        "MACD",

        "MACD_Signal",

        "BB_High",

        "BB_Low",

        "Momentum",

        "Trend_Strength"
    ]

    # ======================================
    # SAFE FEATURES
    # ======================================

    available_features = []

    for feature in features:

        if feature in df.columns:

            available_features.append(feature)

    print("\nUSING FEATURES:\n")

    print(available_features)

    # ======================================
    # INPUTS
    # ======================================

    X = df[available_features]

    y = df["Target"]

    # ======================================
    # TIME SERIES SPLIT
    # ======================================

    split_index = int(
        len(df) * 0.8
    )

    X_train = X[:split_index]

    X_test = X[split_index:]

    y_train = y[:split_index]

    y_test = y[split_index:]

    # ======================================
    # MODEL COMPARISON
    # ======================================

    comparison_results = compare_models(

        X_train,
        X_test,
        y_train,
        y_test
    )

    # ======================================
    # FINAL MODEL
    # ======================================

    model = XGBClassifier(

        n_estimators=400,

        learning_rate=0.03,

        max_depth=6,

        subsample=0.85,

        colsample_bytree=0.85,

        random_state=42,

        eval_metric="logloss"
    )

    # ======================================
    # TRAIN
    # ======================================

    model.fit(
        X_train,
        y_train
    )

    # ======================================
    # PREDICT
    # ======================================

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # ======================================
    # METRICS
    # ======================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    # ======================================
    # RESULTS
    # ======================================

    print("\n==============================")

    print("FINAL MODEL PERFORMANCE")

    print("==============================")

    print(f"Accuracy  : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:\n")

    print(

        classification_report(
            y_test,
            predictions
        )
    )

    # ======================================
    # SAVE MODEL
    # ======================================

    joblib.dump(

        model,

        "models/stock_prediction_model.pkl"
    )

    print("\nModel saved successfully!")

    return (

        model,

        X_test,

        y_test,

        predictions,

        probabilities,

        available_features,

        comparison_results
    )

# ==========================================
# LOAD MODEL
# ==========================================

def load_model():

    model = joblib.load(

        "models/stock_prediction_model.pkl"
    )

    return model

# ==========================================
# SMART AI PREDICTION
# ==========================================

def predict_latest(

    model,
    latest_row
):

    features = [

        "Daily_Return",

        "Volatility",

        "MA20",

        "MA50",

        "EMA20",

        "Price_Change",

        "HL_Spread",

        "RSI",

        "MACD",

        "MACD_Signal",

        "BB_High",

        "BB_Low",

        "Momentum",

        "Trend_Strength"
    ]

    # ======================================
    # SAFE FEATURE COLLECTION
    # ======================================

    row_data = {}

    for feature in features:

        if feature in latest_row.index:

            row_data[feature] = latest_row[feature]

        else:

            row_data[feature] = 0

    # ======================================
    # CREATE DATAFRAME
    # ======================================

    latest_data = pd.DataFrame(
        [row_data]
    )

    # ======================================
    # MODEL PREDICTION
    # ======================================

    prediction = model.predict(
        latest_data
    )[0]

    probabilities = model.predict_proba(
        latest_data
    )[0]

    # ======================================
    # CONFIDENCE
    # ======================================

    probability = round(

        max(probabilities) * 100,

        2
    )

    # ======================================
    # SMART SIGNAL
    # ======================================

    signal = "HOLD"

    rsi = row_data.get("RSI", 50)

    trend_strength = row_data.get(
        "Trend_Strength",
        0
    )

    if (

        trend_strength > 0 and
        rsi < 70

    ):

        signal = "BUY"

    elif (

        trend_strength < 0 and
        rsi > 30

    ):

        signal = "SELL"

    return (

        prediction,

        probability,

        signal
    )