# EcoGrid — Anomaly Detection Workflow

## Overview

**EcoGrid** uses a **Dual-Engine ML Architecture** to monitor CO₂ emissions from thermal power plants. Engine 1 predicts *how much* CO₂ is emitted; Engine 2 acts as an unsupervised **anomaly watchdog** that flags physically impossible or dangerous sensor states — independent of the CO₂ value.

---

## 1.  The Dataset

**File:** `backend/data/processed_synthetic_data_30days.csv`

| Property | Value |
|---|---|
| Total rows | **86,400** (1 reading/minute × 60 mins × 24 hrs × 30 days) |
| Total columns | 7 |
| Time range | 2025-10-11 → 30 days of simulation |

### Columns

| Column | Type | Range | Description |
|---|---|---|---|
| `timestamp` | string | — | Per-minute timestamp |
| `plant_type` | string | Coal | Plant category |
| `fuel_flow` | float | 0.0 – 1.0 | Normalized fuel consumption rate (tons/hr) |
| `boiler_load` | float | 0.0 – 1.0 | Normalized boiler power output (MW) |
| `ambient_temp` | float | 0.0 – 1.0 | Normalized external temperature (°C) |
| `carbon_capture` | int | 0 or 1 | Carbon capture system ON/OFF |
| `co2_emission` | float | 4.34 – 15.0 | Target: CO₂ emission in g/kWh |

>

### CO₂ Emission Statistics

| Statistic | Value |
|---|---|
| Mean | 8.83 g/kWh |
| Median (dynamic threshold) | **~8.14 g/kWh** |
| Std Dev | 2.68 |
| Min / Max | 4.34 / 15.0 |

---

## 2.  Training Pipeline (`train.py`)

```mermaid
flowchart TD
    A[" Load CSV\n86,400 rows × 7 cols"] --> B[" Feature Selection\nfuel_flow, boiler_load,\nambient_temp, carbon_capture"]
    B --> C[" Dynamic Threshold\nEMISSION_THRESHOLD = median(co2_emission)\n≈ 8.14 g/kWh"]
    C --> D[" Train / Test Split\n80% Train · 20% Test\nrandom_state=42"]
    D --> E[" StandardScaler\nFit on numeric cols of train set\nTransform both splits"]

    E --> F[" Engine 1\nLightGBM Regressor\nn_estimators=200\nlearning_rate=0.05\nmax_depth=8"]
    E --> G[" Engine 2\nIsolation Forest\nn_estimators=100\ncontamination=0.05\nFit on X_train_scaled"]

    F --> H[" Regression Output\nPredicts exact CO₂ value"]
    H --> I[" Binary Classification\nPred > threshold → High\nPred ≤ threshold → Normal"]

    G --> J[" Anomaly Labels\n+1 = Normal\n-1 = Anomaly"]

    I --> K[" Evaluation & Plots\nConfusion Matrix, ROC, Feature Importance,\nAnomaly Scatter"]
    J --> K

    K --> L[" Save Artifacts\nmodel.pkl · anomaly_detector.pkl\npreprocessor.joblib · metadata.json"]
```

---

## 3.  How Isolation Forest Detects Anomalies

**Isolation Forest** is an *unsupervised* algorithm — it **does not need CO₂ labels** to detect anomalies. It learns the *shape of normal operational data* and flags outliers.

### Core Mechanism

```mermaid
flowchart LR
    A["Training Data\nX_train_scaled\n~69,120 rows"] --> B["Build 100 Isolation Trees\nRandom feature splits\nRandom threshold cuts"]
    B --> C["Compute Anomaly Score\nShort path = Easy to isolate = Anomaly\nLong path = Hard to isolate = Normal"]
    C --> D{"Score < threshold?\ncontamination=0.05\n(top 5% most isolated)"}
    D -- Yes --> E[" Label: -1 (Anomaly)"]
    D -- No --> F[" Label: +1 (Normal)"]
```

### Key Parameters

| Parameter | Value | Effect |
|---|---|---|
| `n_estimators` | 100 | 100 isolation trees built; more = more stable scores |
| `contamination` | 0.05 | Expects **5%** of data to be anomalous; sets the decision threshold |
| `random_state` | 42 | Reproducibility |

### What Gets Flagged as Anomalous?

The model learns the **joint distribution** of all 4 features together. It will flag combinations that are statistically rare in the training set, such as:

-  **Max fuel flow + near-zero boiler load** → physically impossible, likely sensor failure
-  **Extreme ambient temperature** with contradictory operational readings
-  **Sensor drift** — values slowly creeping outside learned normal bounds
-  Any feature vector that lies far from the dense training cluster

---

## 4.  Inference Pipeline (`main.py`)

Every time the user submits plant readings on the frontend, both engines run in **parallel on the same preprocessed input**:

```mermaid
sequenceDiagram
    actor User as  Operator (Browser)
    participant API as  FastAPI /predict
    participant PP as  Preprocessor
    participant E1 as  LightGBM (Engine 1)
    participant E2 as  Isolation Forest (Engine 2)

    User->>API: POST /predict {fuelflow, boilerload, ambient_temp, capture_on}
    API->>PP: Build DataFrame from raw input
    PP->>PP: StandardScaler.transform(numeric_cols)
    PP-->>API: df_scaled (1 row × 4 features)

    par Engine 1
        API->>E1: model.predict(df_scaled)
        E1-->>API: co2_value (float, clipped 0–50)
    and Engine 2
        API->>E2: anomaly_detector.predict(df_scaled)
        E2-->>API: +1 (Normal) or -1 (Anomaly)
    end

    API->>API: is_high_emission = co2_value > threshold (~8.14)
    API->>API: is_anomaly = (E2 result == -1)

    API-->>User: { prediction, is_high_emission, is_anomaly, status }
```

### API Response Fields

| Field | Type | Meaning |
|---|---|---|
| `prediction` | float | Predicted CO₂ emission in g/kWh (0–50) |
| `is_high_emission` | bool | `true` if prediction > median threshold |
| `is_anomaly` | bool | `true` if Isolation Forest returns -1 |
| `status` | string | `"success"` or `"failed"` |

---

## 5.  Frontend Response (`script.js`)

The frontend reacts to all three signals independently:

```mermaid
flowchart TD
    R["API Response"] --> A["prediction value"]
    R --> B["is_high_emission"]
    R --> C["is_anomaly"]

    A --> D["Gauge Chart\nShows % of 50 g/kWh max"]
    B -- true --> E[" Gauge turns RED\nborderColor = #ff3333"]
    B -- false --> F[" Gauge stays GREEN\nborderColor = #00ff88"]
    C -- true --> G[" anomalyAlert div\nremoves 'hidden' class\nadds 'visible' class"]
    C -- false --> H[" Alert stays hidden"]
```

> [!IMPORTANT]
> **`is_anomaly` and `is_high_emission` are independent signals.** A reading can be:
> -  Normal emission AND Normal (no alert)
> -  High emission AND Normal (gauge turns red, no watchdog alert)
> -  Normal emission AND Anomaly (watchdog fires even though CO₂ looks fine — sensor fault!)
> -  High emission AND Anomaly (both alerts fire simultaneously)

---

## 6.  Model Performance

From `metadata.json`:

| Metric | Value |
|---|---|
| Algorithm | Hybrid LGBM Regressor + Isolation Forest |
| Train R² | 0.6822 |
| Test R² | 0.6746 |
| Train MAE | 1.2663 g/kWh |
| Test MAE | **1.2852 g/kWh** |
| Train RMSE | 1.5089 |
| Test RMSE | **1.5313** |
| Test Accuracy (binary) | **76.03%** |
| Test F1 Score | **78.35%** |

> [!NOTE]
> Train ≈ Test metrics show **no overfitting** — the model generalises well to unseen data.

---

## 7.  Complete End-to-End Flow

```mermaid
flowchart TD
    DS[" Raw Dataset\n86,400 rows · 30 days\n1 reading/minute"] --> FE["Feature Engineering\nSelect 4 features\nApply StandardScaler"]
    FE --> TR1["Train LightGBM Regressor\nPredicts CO₂ continuously"]
    FE --> TR2["Train Isolation Forest\nLearns normal operating envelope\ncontamination = 5%"]
    TR1 --> PKL1["model.pkl"]
    TR2 --> PKL2["anomaly_detector.pkl"]
    FE --> PKL3["preprocessor.joblib\n(scaler + feature_cols + threshold)"]

    PKL1 --> API
    PKL2 --> API
    PKL3 --> API

    subgraph API[" FastAPI Server (main.py)"]
        direction LR
        IN["POST /predict"] --> SC["Scale input"] --> E1["LightGBM → CO₂ value"]
        SC --> E2["Isolation Forest → Normal / Anomaly"]
        E1 --> OUT["JSON Response"]
        E2 --> OUT
    end

    API --> UI[" Frontend Dashboard"]
    UI --> G["Gauge Chart\n(CO₂ level)"]
    UI --> AL[" Anomaly Alert Banner\n(watchdog fires)"]
    UI --> BC["Bar Chart\n(MAE / RMSE metrics)"]
```

---

