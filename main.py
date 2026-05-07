from src.data_loader import load_stock_data

from src.preprocessing import preprocess_data

from src.prediction import train_model

from src.visualization import (

    plot_stock_chart,

    plot_confusion_matrix,

    plot_roc_curve,

    plot_feature_importance,

    plot_model_comparison
)

# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    # ======================================
    # STOCK INPUT
    # ======================================

    ticker = input(
        "Enter Stock Symbol: "
    ).upper()

    # ======================================
    # LOAD DATA
    # ======================================

    print(f"\nFetching stock data for {ticker}...\n")

    df = load_stock_data(ticker)

    print("Raw Data:")

    print(df.head())

    # ======================================
    # PREPROCESS
    # ======================================

    df = preprocess_data(df)

    print("\nProcessed Data:")

    print(df.head())

    print("\nDataset Shape:")

    print(df.shape)

    print("\nColumns:")

    print(df.columns)

    # ======================================
    # SAVE DATASET
    # ======================================

    save_path = f"data/{ticker}_processed.csv"

    df.to_csv(
        save_path,
        index=False
    )

    print(f"\nProcessed data saved to: {save_path}")

    # ======================================
    # TRAIN MODEL
    # ======================================

    print("\nTraining AI model...\n")

    (
        model,
        X_test,
        y_test,
        predictions,
        probabilities,
        features,
        comparison_results

    ) = train_model(df)

    # ======================================
    # VISUALIZATIONS
    # ======================================

    print("\nGenerating charts...\n")

    plot_stock_chart(
        df,
        ticker
    )

    plot_confusion_matrix(
        y_test,
        predictions
    )

    plot_roc_curve(
        y_test,
        probabilities
    )

    plot_feature_importance(
        model,
        features
    )

    plot_model_comparison(
        comparison_results
    )

    print("\nCharts saved in charts/ folder!")

    print("\nProject execution completed successfully!")