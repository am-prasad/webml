from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="EcoGridAI", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class PredictionInput(BaseModel):
    fuel_flow: float
    boiler_load: float
    ambient_temp: float
    carbon_capture: int

# Normalization ranges (from original project data generation)
NORMALIZATION_RANGES = {
    "fuel_flow": (0, 200),
    "boiler_load": (0, 500),
    "ambient_temp": (-20, 80)
}

def normalize_input(value: float, feature: str) -> float:
    """Normalize input based on training ranges to a 0-1 scale"""
    min_val, max_val = NORMALIZATION_RANGES[feature]
    return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))

# Global variables for ML Artifacts
MODEL_PATH = "model.pkl"
METADATA_PATH = "metadata.json"
PREPROCESSOR_PATH = "preprocessor.joblib"

model = None
preprocessor = None

@app.on_event("startup")
def load_artifacts():
    """Load ML artifacts into memory when the server starts"""
    global model, preprocessor
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("✅ Model loaded successfully.")
        if os.path.exists(PREPROCESSOR_PATH):
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print("✅ Preprocessor loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load ML artifacts: {e}")

@app.post("/predict")
async def predict(input_data: PredictionInput):
    """Predict CO2 emissions based on normalized inputs"""
    
    # Fallback to simulation if files haven't been moved yet
    if model is None or preprocessor is None:
        print("Using simulated prediction (Artifacts missing)")
        fuel_flow_norm = normalize_input(input_data.fuel_flow, "fuel_flow")
        boiler_load_norm = normalize_input(input_data.boiler_load, "boiler_load")
        prediction = (
         fuel_flow_norm * 150
        + boiler_load_norm * 200
        + ambient_temp_norm * 50
        + input_data.carbon_capture * 30
            )

    try:
        # 1. Normalize Inputs
        raw_data = {
            "fuel_flow": normalize_input(input_data.fuel_flow, "fuel_flow"),
            "boiler_load": normalize_input(input_data.boiler_load, "boiler_load"),
            "ambient_temp": normalize_input(input_data.ambient_temp, "ambient_temp"),
            "carbon_capture": input_data.carbon_capture
        }
        
        feature_columns = preprocessor.get("feature_columns",["fuel_flow", "boiler_load", "ambient_temp", "carbon_capture"]
)
        scaler = preprocessor.get("scaler", None)

        # 2. Build DataFrame in the exact order the model expects
        df = pd.DataFrame(0.0, index=[0], columns=feature_columns)
        for col in ["fuel_flow", "boiler_load", "ambient_temp", "carbon_capture"]:
            if col in df.columns:
                df.at[0, col] = raw_data[col]

        # 3. Apply the training scaler to the numeric columns
        if scaler:
            numeric_cols = [c for c in ["fuel_flow", "boiler_load", "ambient_temp"] if c in df.columns]
            if numeric_cols:
                df[numeric_cols] = scaler.transform(df[numeric_cols])

        # 4. Predict
        prediction = model.predict(df)[0]
        
        # Clip to realistic CO2 bounds (0 to 50%)
        final_pred = float(np.clip(prediction, 0.0, 50.0))
        
        return {
            "prediction": round(final_pred, 4),
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.get("/metrics")
async def metrics():
    """Return actual training and testing metrics from metadata.json"""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            data = json.load(f)
            return {
    "r2_score": data.get("test_r2_score", 0.0),
    "mae": data.get("test_mae", 0.0),
    "rmse": data.get("test_rmse", 0.0),
    "algorithm": data.get("algorithm", "LightGBM Regressor"),

    "train_mae": data.get("train_mae", 0.0),
    "train_rmse": data.get("train_rmse", 0.0),

    "test_mae": data.get("test_mae", 0.0),
    "test_rmse": data.get("test_rmse", 0.0)
}
    
    # Fallback response if metadata.json is missing
    return {
        "r2_score": 0.00, "mae": 0.00, "rmse": 0.00,
        "algorithm": "Artifacts Missing (Simulated)",
        "train_mae": 0.00, "train_rmse": 0.00,
        "test_mae": 0.00, "test_rmse": 0.00
    }

@app.get("/")
async def root():
    return {"message": "EcoGridAI API - Real ML Predictor is Running"}