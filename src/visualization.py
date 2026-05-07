import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)

# ==========================================
# CREATE CHART FOLDER
# ==========================================

os.makedirs(
    "charts",
    exist_ok=True
)

# ==========================================
# STOCK PRICE CHART
# ==========================================

def plot_stock_chart(df, ticker):

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["Date"],
        df["Close"]
    )

    plt.title(
        f"{ticker} Stock Price"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Closing Price"
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        f"charts/{ticker}_stock_chart.png"
    )

    plt.close()

    print(
        "Stock chart saved!"
    )

# ==========================================
# CONFUSION MATRIX
# ==========================================

def plot_confusion_matrix(
    y_test,
    predictions
):

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.savefig(
        "charts/confusion_matrix.png"
    )

    plt.close()

    print(
        "Confusion matrix saved!"
    )

# ==========================================
# ROC CURVE
# ==========================================

def plot_roc_curve(
    y_test,
    probabilities
):

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.2f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle='--'
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()

    plt.savefig(
        "charts/roc_curve.png"
    )

    plt.close()

    print(
        "ROC curve saved!"
    )

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

def plot_feature_importance(
    model,
    features
):

    importance = model.feature_importances_

    feature_df = pd.DataFrame({

        "Feature": features,

        "Importance": importance

    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=feature_df,
        x="Importance",
        y="Feature"
    )

    plt.title(
        "Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        "charts/feature_importance.png"
    )

    plt.close()

    print(
        "Feature importance chart saved!"
    )

# ==========================================
# MODEL COMPARISON CHART
# ==========================================

def plot_model_comparison(
    results_df
):

    plt.figure(figsize=(10, 6))

    comparison_df = results_df.melt(
        id_vars="Model",
        value_vars=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        var_name="Metric",
        value_name="Score"
    )

    sns.barplot(
        data=comparison_df,
        x="Metric",
        y="Score",
        hue="Model"
    )

    plt.title(
        "Model Performance Comparison"
    )

    plt.ylim(0, 1)

    plt.tight_layout()

    plt.savefig(
        "charts/model_comparison.png"
    )

    plt.close()

    print(
        "Model comparison chart saved!"
    )    