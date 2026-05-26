import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, r2_score, mean_absolute_error, mean_squared_error
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

def generate_and_save_plots(model, anomaly_detector, X_test_scaled, y_test_binary, test_preds_binary, feature_columns, numeric_cols, df_original):
    """Generates and saves all essential machine learning visualizations."""
    print("\nGenerating evaluation plots...")
    os.makedirs("plots", exist_ok=True)
    sns.set_theme(style="whitegrid")

    # --- 1. Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test_binary, test_preds_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'High'], yticklabels=['Normal', 'High'])
    plt.title('Confusion Matrix - Hybrid Threshold Evaluation', fontsize=14)
    plt.xlabel('Predicted State')
    plt.ylabel('Actual State')
    plt.savefig('plots/1_confusion_matrix.png', bbox_inches='tight')
    plt.close()

    # --- 2. Feature Importance ---
    plt.figure(figsize=(10, 6))
    feature_imp = pd.DataFrame({'Importance': model.feature_importances_, 'Feature': feature_columns})
    feature_imp = feature_imp.sort_values(by="Importance", ascending=False)
    sns.barplot(x="Importance", y="Feature", data=feature_imp, palette="viridis")
    plt.title('Feature Importance', fontsize=14)
    plt.savefig('plots/2_feature_importance.png', bbox_inches='tight')
    plt.close()

    # --- 3. ROC Curve ---
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test_binary, test_preds_binary)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14)
    plt.legend(loc="lower right")
    plt.savefig('plots/3_roc_curve.png', bbox_inches='tight')
    plt.close()

    # --- 4. Anomaly Scatter Plot ---
    plt.figure(figsize=(10, 6))
    # Data is already normalized [0,1] — use feature columns directly
    X_all = df_original[feature_columns]

    df_plot = df_original.copy()
    df_plot['Anomaly_Status'] = anomaly_detector.predict(X_all)
    df_plot['Anomaly_Status'] = df_plot['Anomaly_Status'].map({1: 'Normal', -1: 'Anomaly'})
    
    sns.scatterplot(data=df_plot, x='fuel_flow', y='boiler_load', 
                    hue='Anomaly_Status', palette={'Normal': '#2ecc71', 'Anomaly': '#e74c3c'}, alpha=0.6, s=50)
    plt.title('Anomaly Detection Watchdog', fontsize=14)
    plt.xlabel('Fuel Flow (tons/hr)')
    plt.ylabel('Boiler Load (MW)')
    plt.savefig('plots/4_anomaly_scatter.png', bbox_inches='tight')
    plt.close()
    
    print("✅ All plots saved successfully in the 'plots/' directory!")

def train_model(csv_path=None):
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
    y = df['co2_emission'] # REGRESSION: Predicting exact number

    # Threshold for Hybrid Evaluation
    EMISSION_THRESHOLD = y.median()

    print("Splitting dataset 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Features are already Min-Max normalized [0, 1] in the CSV — no scaling needed.
    print("Skipping StandardScaler — features are already normalized [0, 1].")
    numeric_cols = ['fuel_flow', 'boiler_load', 'ambient_temp']
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # ENGINE 1: LightGBM REGRESSOR
    print("Training LightGBM Regressor...")
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)

    # Convert Regressor outputs to Classification for Evaluation plots
    y_test_binary = (y_test > EMISSION_THRESHOLD).astype(int)
    test_preds_binary = (test_preds > EMISSION_THRESHOLD).astype(int)

    # ENGINE 2: Isolation Forest Anomaly Detector
    print("Training Isolation Forest Anomaly Detector...")
    anomaly_detector = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    anomaly_detector.fit(X_train_scaled)

    # Generate Plots
    generate_and_save_plots(model, anomaly_detector, X_test_scaled, y_test_binary, test_preds_binary, feature_columns, numeric_cols, df)

    # Compile Hybrid Metrics
    metrics = {
        "algorithm": "Hybrid LGBM Regressor + Isolation Forest",
        "train_r2": round(float(r2_score(y_train, train_preds)), 4),
        "test_r2": round(float(r2_score(y_test, test_preds)), 4),
        "train_mae": round(float(mean_absolute_error(y_train, train_preds)), 4),
        "test_mae": round(float(mean_absolute_error(y_test, test_preds)), 4),
        "train_rmse": round(float(np.sqrt(mean_squared_error(y_train, train_preds))), 4),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, test_preds))), 4),
        "test_accuracy": round(float(accuracy_score(y_test_binary, test_preds_binary)), 4),
        "test_f1_score": round(float(f1_score(y_test_binary, test_preds_binary)), 4)
    }

    print("\n===== FINAL METRICS =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nSaving artifacts...")
    joblib.dump(model, "model.pkl")
    joblib.dump(anomaly_detector, "anomaly_detector.pkl")
    
    joblib.dump({
        "scaler": None,  # No scaling needed — CSV features are already normalized [0, 1]
        "feature_columns": feature_columns,
        "threshold": float(EMISSION_THRESHOLD)
    }, "preprocessor.joblib")
    
    with open("metadata.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    train_model(csv_path=r"data\processed_synthetic_data_30days.csv")