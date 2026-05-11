#  EcoGridAI: Real-Time CO₂ Emission Monitor

EcoGridAI is a full-stack, machine-learning-powered web application designed to predict and monitor **CO₂ emissions from thermal power plants in real time**.

By analyzing industrial operational parameters such as plant type, fuel flow, boiler load, ambient temperature, and carbon capture status, the system provides actionable environmental insights for sustainability monitoring and compliance analysis.

---

#  Features

* Real-time CO₂ emission prediction
* Machine Learning powered by Linear Regression
* Interactive frontend dashboard using Chart.js
* FastAPI REST API backend
* Performance metrics visualization
* Carbon capture impact analysis
* Multi-plant industrial simulation
* Lightweight frontend with zero framework dependencies

---

#  Tech Stack

## Backend

* Python
* FastAPI
* Uvicorn

## Machine Learning

* Scikit-Learn
* Linear Regression
* Pandas
* NumPy
* Joblib

## Frontend

* HTML5
* CSS3
* JavaScript
* Chart.js

---

#  Machine Learning Architecture

The predictive engine uses a **Linear Regression model**
to estimate CO₂ emissions from industrial operational parameters.

Linear Regression was selected in Phase 1 because of its:

* Simplicity and interpretability
* Fast training performance
* Low computational overhead
* Strong baseline regression capability
* Easy explainability for industrial analytics

The model predicts continuous CO₂ emission values using supervised learning regression techniques.

---

#  Input Features

The model accepts the following operational parameters:

| Feature | Description | Unit |
|---|---|---|
| `plant_type` | Type of thermal plant | Encoded Integer |
| `fuel_flow` | Fuel consumption rate | tons/hr |
| `boiler_load` | Power plant operational load | MW |
| `ambient_temp` | External environmental temperature | °C |
| `carbon_capture` | Carbon capture enabled | Boolean |

---





# Data Pipeline

## 1. Feature Ingestion

Operational industrial sensor parameters are collected from the plant environment.

---

## 2. Data Preprocessing

Continuous numeric features are standardized using Scikit-Learn's `StandardScaler`.

This improves regression stability and ensures balanced feature contribution during model optimization.

---

## 3. Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

This enables proper evaluation on unseen industrial operating conditions.

---
# Model Evaluation Metrics

To evaluate regression performance, multiple statistical metrics are computed.

---

## 1. Mean Absolute Error (MAE)

Measures average prediction error magnitude.

$$
\text{MAE} =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i - \hat{y}_i|
$$

---

## 2. Root Mean Squared Error (RMSE)

Penalizes larger prediction errors more heavily.

$$
\text{RMSE} =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i - \hat{y}_i)^2
}
$$

---

## 3. R-Squared Score ($R^2$)

Represents how much variance in emissions is explained by the model.

$$
R^2 =
1 -
\frac{
\sum_{i=1}^{n}
(y_i - \hat{y}_i)^2
}{
\sum_{i=1}^{n}
(y_i - \bar{y})^2
}
$$

---

## 4. Mean Absolute Percentage Error (MAPE)

Measures prediction error percentage.

$$
\text{MAPE} =
\frac{100}{n}
\sum_{i=1}^{n}
\left|
\frac{
y_i - \hat{y}_i
}{
y_i
}
\right|
$$

---

Where:

* $y_i$ = Actual value
* $\hat{y}_i$ = Predicted value
* $\bar{y}$ = Mean actual value
* $n$ = Number of samples

---

#  Model Optimization

## Overfitting (High Variance)

### Problem

The regression model performs well on training data but poorly on unseen industrial conditions.

### Mitigations

* Proper train-test split
* Feature scaling
* Balanced synthetic dataset generation
* Missing value handling
* Noise reduction during preprocessing

---

## Underfitting (High Bias)

### Problem

Linear Regression may not fully capture highly nonlinear industrial emission behavior.

### Mitigations

* Feature engineering
* Plant-type operational modeling
* Proper feature scaling
* Future migration to advanced boosting regression models

---

# Project Evolution

## Phase 1

Baseline implementation using **Linear Regression**
for industrial CO₂ emission prediction.

### Goals

* Establish predictive pipeline
* Validate industrial feature relationships
* Build real-time dashboard
* Create regression baseline metrics

---

## Phase 2

Planned migration to advanced regression systems such as:

* LightGBM Regressor
* XGBoost Regressor

### Expected Improvements

* Better nonlinear learning
* Higher prediction accuracy
* Improved scalability
* Enhanced industrial deployment capability

---

#  Project Structure

```bash
EcoGridAI/
│
├── backend/
│   ├── main.py
│   ├── train_linear.py
│   ├── requirements.txt
│   ├── linear_model.pkl
│   ├── linear_preprocessor.joblib
│   └── linear_metadata.json
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
└── README.md
```

---
# Installation & Setup

## 1. Clone Repository

```bash
git clone https://github.com/am-prasad/webml.git

cd webml
```

---

## 2. Backend Setup

Navigate to backend directory:

```bash
cd backend
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

#  Train the Model

Generate trained regression artifacts:

```bash
python train_linear.py
```

Generated files:

* `linear_model.pkl`
* `linear_preprocessor.joblib`
* `linear_metadata.json`

---

#  Run API Server

Start FastAPI backend:

```bash
uvicorn main:app --reload
```

Server runs at:

```bash
http://127.0.0.1:8000
```

---

#  Launch Frontend

Open:

```bash
frontend/index.html
```

in any modern browser.

No frontend build tools or npm setup required.

---

#  API Endpoint

## Predict CO₂ Emissions

### POST `/predict`

---

## Request Example

```json
{
  "plant_type": 0,
  "fuel_flow": 180,
  "boiler_load": 420,
  "ambient_temp": 34,
  "carbon_capture": 1
}
```

---

## Response Example

```json
{
  "prediction": 92.4418,
  "status": "success"
}
```

---

#  Future Enhancements

* Real industrial IoT integration
* Cloud deployment
* Historical emissions analytics
* Advanced industrial dashboards
* Live environmental compliance tracking
* Deep learning experimentation
* Smart industrial alert systems

---

#  Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a pull request

---

