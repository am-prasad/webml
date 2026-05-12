import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)


def generate_synthetic_data(num_samples=3000):

    np.random.seed(42)

    plant_types = np.random.choice(
        [0, 1, 2, 3, 4],
        num_samples
    )

    fuel_flow = np.random.uniform(
        0,
        200,
        num_samples
    )

    boiler_load = np.random.uniform(
        0,
        500,
        num_samples
    )

    ambient_temp = np.random.uniform(
        -20,
        80,
        num_samples
    )

    carbon_capture = np.random.choice(
        [0, 1],
        num_samples
    )

    plant_factor = []

    for plant in plant_types:

        if plant == 0:
            plant_factor.append(35)

        elif plant == 1:
            plant_factor.append(22)

        elif plant == 4:
            plant_factor.append(15)

        elif plant == 2:
            plant_factor.append(5)

        else:
            plant_factor.append(2)

    plant_factor = np.array(
        plant_factor
    )

    base_co2 = (
        (fuel_flow * 0.25)
        + (boiler_load * 0.12)
        - (ambient_temp * 0.02)
        + plant_factor
    )

    capture_reduction = (
        carbon_capture * 20
    )

    co2_emission = np.clip(
        base_co2
        - capture_reduction
        + np.random.normal(
            0,
            3,
            num_samples
        ),
        0,
        150
    )

    df = pd.DataFrame({
        "plant_type": plant_types,
        "fuel_flow": fuel_flow,
        "boiler_load": boiler_load,
        "ambient_temp": ambient_temp,
        "carbon_capture": carbon_capture,
        "co2_emission": co2_emission
    })

    return df


def train_model(csv_path=None):

    if csv_path and os.path.exists(csv_path):

        print(
            f"Loading data from {csv_path}"
        )

        df = pd.read_csv(csv_path)

    else:

        print(
            "Generating synthetic dataset..."
        )

        df = generate_synthetic_data()

    # ------------------------------------------
    # CLEAN DATA
    # ------------------------------------------

    df["plant_type"] = (
        pd.to_numeric(
            df["plant_type"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    df = df.fillna(0)

    # ------------------------------------------
    # FEATURES
    # ------------------------------------------

    feature_columns = [
        "plant_type",
        "fuel_flow",
        "boiler_load",
        "ambient_temp",
        "carbon_capture"
    ]

    X = df[feature_columns]

    y = df["co2_emission"]

    # ------------------------------------------
    # TRAIN TEST SPLIT
    # ------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    # ------------------------------------------
    # FEATURE SCALING
    # ------------------------------------------

    scaler = StandardScaler()

    numeric_cols = [
        "plant_type",
        "fuel_flow",
        "boiler_load",
        "ambient_temp"
    ]

    X_train_scaled = X_train.copy()

    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = (
        scaler.fit_transform(
            X_train[numeric_cols]
        )
    )

    X_test_scaled[numeric_cols] = (
        scaler.transform(
            X_test[numeric_cols]
        )
    )

    # ------------------------------------------
    # TRAIN MODEL
    # ------------------------------------------

    print(
        "Training Multiple Linear Regression Model..."
    )

    model = LinearRegression()

    model.fit(
        X_train_scaled,
        y_train
    )

    # ------------------------------------------
    # PREDICTIONS
    # ------------------------------------------

    train_preds = model.predict(
        X_train_scaled
    )

    test_preds = model.predict(
        X_test_scaled
    )

    # ------------------------------------------
    # METRICS
    # ------------------------------------------

    metrics = {

        "algorithm":
            "Multiple Linear Regression",

        "train_r2_score":
            round(
                float(
                    r2_score(
                        y_train,
                        train_preds
                    )
                ),
                4
            ),

        "test_r2_score":
            round(
                float(
                    r2_score(
                        y_test,
                        test_preds
                    )
                ),
                4
            ),

        "train_mae":
            round(
                float(
                    mean_absolute_error(
                        y_train,
                        train_preds
                    )
                ),
                4
            ),

        "test_mae":
            round(
                float(
                    mean_absolute_error(
                        y_test,
                        test_preds
                    )
                ),
                4
            ),

        "train_rmse":
            round(
                float(
                    np.sqrt(
                        mean_squared_error(
                            y_train,
                            train_preds
                        )
                    )
                ),
                4
            ),

        "test_rmse":
            round(
                float(
                    np.sqrt(
                        mean_squared_error(
                            y_test,
                            test_preds
                        )
                    )
                ),
                4
            ),

        "test_mse":
            round(
                float(
                    mean_squared_error(
                        y_test,
                        test_preds
                    )
                ),
                4
            ),

        "test_mape":
            round(
                float(
                    mean_absolute_percentage_error(
                        y_test,
                        test_preds
                    )
                ),
                4
            )
    }

    # ------------------------------------------
    # PRINT METRICS
    # ------------------------------------------

    print(
        "\n===== REGRESSION METRICS ====="
    )

    for k, v in metrics.items():

        print(f"{k}: {v}")

    # ------------------------------------------
    # REGRESSION PLOTS
    # ------------------------------------------

    print(
        "\nGenerating regression plots..."
    )

    # ------------------------------------------
    # 1. ACTUAL VS PREDICTED
    # ------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        test_preds,
        alpha=0.6
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linewidth=2
    )

    plt.xlabel(
        "Actual CO₂ Emission"
    )

    plt.ylabel(
        "Predicted CO₂ Emission"
    )

    plt.title(
        "Actual vs Predicted CO₂"
    )

    plt.grid(True)

    plt.savefig(
        "actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ------------------------------------------
    # 2. RESIDUAL PLOT
    # ------------------------------------------

    residuals = (
        y_test - test_preds
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        test_preds,
        residuals,
        alpha=0.6
    )

    plt.axhline(
        y=0,
        linestyle='--'
    )

    plt.xlabel(
        "Predicted CO₂"
    )

    plt.ylabel(
        "Residual Error"
    )

    plt.title(
        "Residual Error Plot"
    )

    plt.grid(True)

    plt.savefig(
        "residual_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ------------------------------------------
    # 3. TRAIN VS TEST ERROR
    # ------------------------------------------

    metrics_names = [
        "MAE",
        "RMSE"
    ]

    train_values = [
        metrics["train_mae"],
        metrics["train_rmse"]
    ]

    test_values = [
        metrics["test_mae"],
        metrics["test_rmse"]
    ]

    x = np.arange(
        len(metrics_names)
    )

    width = 0.35

    plt.figure(figsize=(8, 6))

    plt.bar(
        x - width/2,
        train_values,
        width,
        label="Train"
    )

    plt.bar(
        x + width/2,
        test_values,
        width,
        label="Test"
    )

    plt.xticks(
        x,
        metrics_names
    )

    plt.ylabel("Error")

    plt.title(
        "Train vs Test Error"
    )

    plt.legend()

    plt.grid(True)

    plt.savefig(
        "train_vs_test_error.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ------------------------------------------
    # 4. CO₂ DISTRIBUTION
    # ------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.hist(
        y,
        bins=30
    )

    plt.xlabel(
        "CO₂ Emission"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "CO₂ Distribution"
    )

    plt.grid(True)

    plt.savefig(
        "co2_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ------------------------------------------
    # 5. CORRELATION HEATMAP
    # ------------------------------------------

    correlation_matrix = df[
        feature_columns
        + ["co2_emission"]
    ].corr()

    plt.figure(figsize=(8, 6))

    plt.imshow(
        correlation_matrix,
        aspect='auto'
    )

    plt.colorbar()

    plt.xticks(
        range(
            len(
                correlation_matrix.columns
            )
        ),
        correlation_matrix.columns,
        rotation=45
    )

    plt.yticks(
        range(
            len(
                correlation_matrix.columns
            )
        ),
        correlation_matrix.columns
    )

    plt.title(
        "Feature Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        "correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\n Regression plots generated successfully"
    )

    # ------------------------------------------
    # SAVE MODEL
    # ------------------------------------------

    joblib.dump(
        model,
        "linear_model.pkl"
    )

    preprocessor = {
        "scaler": scaler,
        "feature_columns": feature_columns
    }

    joblib.dump(
        preprocessor,
        "linear_preprocessor.joblib"
    )

    with open(
        "linear_metadata.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        "\nMultiple Linear Regression Training Completed"
    )


if __name__ == "__main__":

    csv_file_path = (
        r"data/processed_synthetic_data_30days.csv"
    )

    train_model(
        csv_path=csv_file_path
    )