from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import json
import os
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="EcoGridAI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionInput(BaseModel):

    plant_type: str
    fuel_flow: float
    boiler_load: float
    ambient_temp: float
    carbon_capture: int


NORMALIZATION_RANGES = {
    "fuel_flow": (0, 200),
    "boiler_load": (0, 500),
    "ambient_temp": (-20, 80)
}

PLANT_MAPPING = {
    "Coal": 0,
    "Gas": 1,
    "Nuclear": 2,
    "Solar": 3,
    "Biomass": 4
}

MODEL_PATH = "model.pkl"
PREPROCESSOR_PATH = "preprocessor.joblib"
METADATA_PATH = "metadata.json"

model = None
preprocessor = None


def normalize_input(value, feature):

    min_val, max_val = NORMALIZATION_RANGES[feature]

    return float(
        np.clip(
            (value - min_val) / (max_val - min_val),
            0.0,
            1.0
        )
    )


@app.on_event("startup")
def load_artifacts():

    global model, preprocessor

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)

    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)

    print("✅ Artifacts Loaded")


@app.post("/predict")
async def predict(input_data: PredictionInput):

    try:

        scaler = preprocessor["scaler"]

        raw_data = {
            "plant_type":
                PLANT_MAPPING[input_data.plant_type],

            "fuel_flow":
                normalize_input(
                    input_data.fuel_flow,
                    "fuel_flow"
                ),

            "boiler_load":
                normalize_input(
                    input_data.boiler_load,
                    "boiler_load"
                ),

            "ambient_temp":
                normalize_input(
                    input_data.ambient_temp,
                    "ambient_temp"
                ),

            "carbon_capture":
                input_data.carbon_capture
        }

        feature_columns = preprocessor["feature_columns"]

        df = pd.DataFrame(
            [raw_data],
            columns=feature_columns
        )

        numeric_cols = [
            "plant_type",
            "fuel_flow",
            "boiler_load",
            "ambient_temp"
        ]

        df[numeric_cols] = scaler.transform(
            df[numeric_cols]
        )

        prediction = model.predict(df)[0]

        final_prediction = float(
            np.clip(prediction, 0.0, 150.0)
        )

        return {
            "prediction": round(final_prediction, 4),
            "status": "success"
        }

    except Exception as e:

        return {
            "error": str(e),
            "status": "failed"
        }


@app.get("/metrics")
async def metrics():

    if os.path.exists(METADATA_PATH):

        with open(METADATA_PATH, "r") as f:

            data = json.load(f)

            return {
                "r2_score":
                    data.get("test_r2_score", 0.0),

                "mae":
                    data.get("test_mae", 0.0),

                "rmse":
                    data.get("test_rmse", 0.0),

                "algorithm":
                    data.get("algorithm",
                             "LightGBM Regressor"),

                "train_mae":
                    data.get("train_mae", 0.0),

                "train_rmse":
                    data.get("train_rmse", 0.0),

                "test_mae":
                    data.get("test_mae", 0.0),

                "test_rmse":
                    data.get("test_rmse", 0.0)
            }

    return {
        "r2_score": 0.0,
        "mae": 0.0,
        "rmse": 0.0,
        "algorithm": "Unavailable"
    }


@app.get("/")
async def root():

    return {
        "message":
            "EcoGridAI API Running Successfully 🚀"
    }