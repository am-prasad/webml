import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

import lightgbm as lgb


def generate_synthetic_data(num_samples=2000):
    """Generate synthetic thermal plant data."""

    np.random.seed(42)

    fuel_flow = np.random.uniform(0, 200, num_samples)
    boiler_load = np.random.uniform(0, 500, num_samples)
    ambient_temp = np.random.uniform(-20, 80, num_samples)
    carbon_capture = np.random.choice([0, 1], num_samples)

    base_co2 = (
        (fuel_flow * 0.15)
        + (boiler_load * 0.05)
        - (ambient_temp * 0.01)
    )

    capture_reduction = carbon_capture * 15.0

    co2_emission = np.clip(
        base_co2 - capture_reduction + np.random.normal(0, 2, num_samples),
        0,
        50
    )

    df = pd.DataFrame({
        'fuel_flow': fuel_flow,
        'boiler_load': boiler_load,
        'ambient_temp': ambient_temp,
        'carbon_capture': carbon_capture,
        'co2_emission': co2_emission
    })

    return df


def train_model(csv_path=None):

    # Load data
    if csv_path and os.path.exists(csv_path):

        print(f"Loading data from {csv_path}")

        df = pd.read_csv(csv_path)

        if 'carbon_capture' in df.columns:
            df['carbon_capture'] = df['carbon_capture'].astype(int)

    else:
        print("CSV not found. Using synthetic data.")
        df = generate_synthetic_data(3000)

    # Required columns
    required_cols = [
        'fuel_flow',
        'boiler_load',
        'ambient_temp',
        'carbon_capture',
        'co2_emission'
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Features and target
    feature_columns = [
        'fuel_flow',
        'boiler_load',
        'ambient_temp',
        'carbon_capture'
    ]

    X = df[feature_columns]
    y = df['co2_emission']

    # Train-test split
    print("Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Scaling
    print("Scaling features...")

    scaler = StandardScaler()

    numeric_cols = [
        'fuel_flow',
        'boiler_load',
        'ambient_temp'
    ]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(
        X_train[numeric_cols]
    )

    X_test_scaled[numeric_cols] = scaler.transform(
        X_test[numeric_cols]
    )

    # Model
    print("Training LightGBM model...")

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # Predictions
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)

    # Metrics
    metrics = {
        "algorithm": "LightGBM Regressor",

        "train_r2_score":
            round(float(r2_score(y_train, train_preds)), 4),

        "test_r2_score":
            round(float(r2_score(y_test, test_preds)), 4),

        "train_mae":
            round(float(mean_absolute_error(y_train, train_preds)), 4),

        "test_mae":
            round(float(mean_absolute_error(y_test, test_preds)), 4),

        "train_rmse":
            round(float(np.sqrt(mean_squared_error(y_train, train_preds))), 4),

        "test_rmse":
            round(float(np.sqrt(mean_squared_error(y_test, test_preds))), 4),

        "test_mse":
            round(float(mean_squared_error(y_test, test_preds)), 4),

        "test_mape":
            round(float(mean_absolute_percentage_error(y_test, test_preds)), 4)
    }

    print("\n===== MODEL METRICS =====")

    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Save model
    print("\nSaving artifacts...")

    joblib.dump(model, "model.pkl")

    preprocessor = {
        "scaler": scaler,
        "feature_columns": feature_columns
    }

    joblib.dump(preprocessor, "preprocessor.joblib")

    with open("metadata.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n✅ Training completed successfully!")


if __name__ == "__main__":

    csv_file_path = r"data\processed_synthetic_data_30days.csv"

    train_model(csv_path=csv_file_path)