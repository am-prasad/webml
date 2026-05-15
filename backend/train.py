import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import IsolationForest
import lightgbm as lgb

def generate_synthetic_data(num_samples=2000):
    """Generate synthetic thermal plant data as fallback."""
    np.random.seed(42)
    fuel_flow = np.random.uniform(0, 200, num_samples)
    boiler_load = np.random.uniform(0, 500, num_samples)
    ambient_temp = np.random.uniform(-20, 80, num_samples)
    carbon_capture = np.random.choice([0, 1], num_samples)

    base_co2 = (fuel_flow * 0.15) + (boiler_load * 0.05) - (ambient_temp * 0.01)
    capture_reduction = carbon_capture * 15.0
    co2_emission = np.clip(base_co2 - capture_reduction + np.random.normal(0, 2, num_samples), 0, 50)

    return pd.DataFrame({
        'fuel_flow': fuel_flow, 'boiler_load': boiler_load,
        'ambient_temp': ambient_temp, 'carbon_capture': carbon_capture,
        'co2_emission': co2_emission
    })

def train_model(csv_path=None):
    # 1. Load Data
    if csv_path and os.path.exists(csv_path):
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        if 'carbon_capture' in df.columns:
            df['carbon_capture'] = df['carbon_capture'].astype(int)
    else:
        print("CSV not found. Using synthetic data.")
        df = generate_synthetic_data(3000)

    feature_columns = ['fuel_flow', 'boiler_load', 'ambient_temp', 'carbon_capture']
    X = df[feature_columns]
    
    # ---------------------------------------------------------
    # THE FIX: DYNAMIC THRESHOLDING
    # ---------------------------------------------------------
    # Find the exact middle value of your specific CO2 data
    EMISSION_THRESHOLD = df['co2_emission'].median()
    print(f"\nCalculated Dynamic Threshold: {EMISSION_THRESHOLD:.4f}")
    
    # Create classes: 1 if above median, 0 if below
    y = (df['co2_emission'] > EMISSION_THRESHOLD).astype(int)
    
    print("\nClass Distribution (Should be balanced now):")
    print(y.value_counts())
    print("-" * 50)

    # 2. Split Data
    print("Splitting dataset 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Scale Features
    print("Scaling features...")
    scaler = StandardScaler()
    numeric_cols = ['fuel_flow', 'boiler_load', 'ambient_temp']
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # 4. ENGINE 1: LightGBM Classifier
    print("Training LightGBM Classifier...")
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=42)

    # 5-Fold Stratified Cross-Validation
    print("Performing 5-Fold Stratified Cross-Validation...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='accuracy')
    mean_cv_accuracy = cv_scores.mean()
    std_cv_accuracy = cv_scores.std()

    # Fit final model
    model.fit(X_train_scaled, y_train)
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)

    # 5. ENGINE 2: Isolation Forest (Anomaly Detection)
    print("Training Isolation Forest Anomaly Detector...")
    anomaly_detector = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    anomaly_detector.fit(X_train_scaled)

    # 6. Compile Metrics
    metrics = {
        "algorithm": "LightGBM Classifier + Isolation Forest",
        "train_accuracy": round(float(accuracy_score(y_train, train_preds)), 4),
        "test_accuracy": round(float(accuracy_score(y_test, test_preds)), 4),
        "cv_mean_accuracy": round(float(mean_cv_accuracy), 4),
        "cv_std_deviation": round(float(std_cv_accuracy), 4),
        "train_f1_score": round(float(f1_score(y_train, train_preds)), 4),
        "test_f1_score": round(float(f1_score(y_test, test_preds)), 4)
    }

    print("\n===== FINAL METRICS =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 7. Save Artifacts
    print("\nSaving artifacts...")
    joblib.dump(model, "model.pkl")
    joblib.dump(anomaly_detector, "anomaly_detector.pkl")
    
    # Save the dynamic threshold in the preprocessor so the backend knows what it is!
    joblib.dump({
        "scaler": scaler, 
        "feature_columns": feature_columns, 
        "threshold": float(EMISSION_THRESHOLD)
    }, "preprocessor.joblib")
    
    with open("metadata.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Training completed successfully! The 'Only One Class' error is resolved.")

if __name__ == "__main__":
    # Ensure this path points to your actual CSV file
    train_model(csv_path=r"data\processed_synthetic_data_30days.csv")