import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import lightgbm as lgb

def generate_synthetic_data(num_samples=2000):
    """Generate realistic synthetic thermal plant data for training."""
    print("Generating synthetic data...")
    np.random.seed(42)
    
    # Generate random features within realistic ranges
    fuel_flow = np.random.uniform(0, 200, num_samples)
    boiler_load = np.random.uniform(0, 500, num_samples)
    ambient_temp = np.random.uniform(-20, 80, num_samples)
    capture_on = np.random.choice([0, 1], num_samples)
    
    # Calculate CO2 emission (Target variable) 
    base_co2 = (fuel_flow * 0.15) + (boiler_load * 0.05) - (ambient_temp * 0.01)
    capture_reduction = capture_on * 15.0  # Capture reduces CO2 by roughly 15%
    
    co2_emission = np.clip(base_co2 - capture_reduction + np.random.normal(0, 2, num_samples), 0, 50)
    
    df = pd.DataFrame({
        'fuelflow': fuel_flow,
        'boilerload': boiler_load,
        'ambient_temp': ambient_temp,
        'capture_on': capture_on,
        'co2_emission': co2_emission
    })
    return df

def train_model(csv_path=None):
    # 1. Get Data
    if csv_path and os.path.exists(csv_path):
        print(f"Loading actual data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Ensure 'capture_on' is numeric (0 or 1)
        if 'capture_on' in df.columns:
            df['capture_on'] = df['capture_on'].astype(int)
    else:
        print("CSV not found or not provided. Falling back to synthetic data generation...")
        df = generate_synthetic_data(3000)
    
    # Ensure all required columns exist in the loaded data
    required_cols = ['fuel_flow', 'boiler_load', 'ambient_temp', 'carbon_capture', 'co2_emission']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")

    feature_columns = ['fuel_flow', 'boiler_load', 'ambient_temp', 'carbon_capture']
    X = df[feature_columns]
    y = df['co2_emission']

    # 2. Train/Test Split (80/20)
    print("Splitting data 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Normalize/Scale the numeric features
    print("Scaling features...")
    scaler = StandardScaler()
    numeric_cols = ['fuel_flow', 'boiler_load', 'ambient_temp']
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # 4. Train the Model using LightGBM (Best suited for this data)
    print("Training LightGBM Regressor...")
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate the Model
    print("Evaluating model performance...")
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)

    # Compile ALL metrics into a dictionary for the JSON file
    metrics = {
        "algorithm": "LightGBM Regressor",
        "train_r2_score": round(float(r2_score(y_train, train_preds)), 4),
        "test_r2_score": round(float(r2_score(y_test, test_preds)), 4),
        "train_mae": round(float(mean_absolute_error(y_train, train_preds)), 4),
        "test_mae": round(float(mean_absolute_error(y_test, test_preds)), 4),
        "train_rmse": round(float(np.sqrt(mean_squared_error(y_train, train_preds))), 4),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, test_preds))), 4),
        "test_mse": round(float(mean_squared_error(y_test, test_preds)), 4),
        "test_mape": round(float(mean_absolute_percentage_error(y_test, test_preds)), 4)
    }

    print("\n--- Training Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 6. Save Artifacts
    print("\nSaving artifacts to backend folder...")
    
    # Save Model
    joblib.dump(model, "model.pkl")
    
    # Save Preprocessor (Scaler + Feature column list)
    preprocessor = {
        "scaler": scaler,
        "feature_columns": feature_columns
    }
    joblib.dump(preprocessor, "preprocessor.joblib")
    
    # Save Comprehensive Metrics to JSON
    with open("metadata.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("✅ Training complete! Artifacts saved: model.pkl, preprocessor.joblib, metadata.json")

if __name__ == "__main__":
    # Point this to your actual CSV file
    csv_file_path = "data\processed_synthetic_data_30days.csv"
    train_model(csv_path=csv_file_path)