from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json, os, joblib
import pandas as pd
import numpy as np

app = FastAPI(title="EcoGridAI", version="2.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class PredictionInput(BaseModel):
    fuelflow: float
    boilerload: float
    ambient_temp: float
    capture_on: int

MODEL_PATH = "model.pkl"
ANOMALY_PATH = "anomaly_detector.pkl"
PREPROCESSOR_PATH = "preprocessor.joblib"
METADATA_PATH = "metadata.json"

model = None
anomaly_detector = None
preprocessor = None

@app.on_event("startup")
def load_artifacts():
    global model, anomaly_detector, preprocessor
    try:
        if os.path.exists(MODEL_PATH): model = joblib.load(MODEL_PATH)
        if os.path.exists(ANOMALY_PATH): anomaly_detector = joblib.load(ANOMALY_PATH)
        if os.path.exists(PREPROCESSOR_PATH): preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("✅ Dual-Engine Models loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error loading ML artifacts: {e}")

@app.post("/predict")
async def predict(input_data: PredictionInput):
    if model is None or anomaly_detector is None or preprocessor is None:
        return {"error": "Models not loaded", "status": "failed"}

    try:
        # ⚠️ FIX: Removed the normalize_input function here. 
        # Pass the raw values directly to the dataframe!
        raw_data = {
            "fuel_flow": input_data.fuelflow,
            "boiler_load": input_data.boilerload,
            "ambient_temp": input_data.ambient_temp,
            "carbon_capture": input_data.capture_on
        }
        
        feature_columns = preprocessor.get("feature_columns")
        scaler = preprocessor.get("scaler")

        df = pd.DataFrame(0.0, index=[0], columns=feature_columns)
        for col in feature_columns:
            df.at[0, col] = raw_data[col]

        
        if scaler:
            numeric_cols = [c for c in ['fuel_flow', 'boiler_load', 'ambient_temp'] if c in df.columns]
            df[numeric_cols] = scaler.transform(df[numeric_cols])

        # Engine 1: Predict exact CO2 Regressive Number
        prediction = float(np.clip(model.predict(df)[0], 0.0, 50.0))
        
        # Engine 2: Predict Anomaly Watchdog (-1 means Anomaly, 1 means Normal)
        is_anomaly = anomaly_detector.predict(df)[0] == -1
        
        # Threshold Check for UI coloring
        threshold = preprocessor.get("threshold", 30.0)
        is_high_emission = prediction > threshold

        return {
            "prediction": round(prediction, 4),
            "is_high_emission": bool(is_high_emission),
            "is_anomaly": bool(is_anomaly), 
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.get("/metrics")
async def metrics():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)
    return {"error": "Metrics missing"}