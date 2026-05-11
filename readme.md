#  EcoGridAI: Real-Time CO₂ Emission Monitor

EcoGridAI is a full-stack, machine-learning-powered web application designed to predict and monitor **CO₂ emissions from thermal power plants in real time**.

By analyzing operational parameters such as fuel flow, boiler load, ambient temperature, and carbon capture status, the system provides actionable environmental insights for sustainability monitoring and compliance analysis.

---

#  Features

*  Real-time CO₂ emission prediction
*  Machine Learning powered by LightGBM
*  Interactive frontend dashboard using Chart.js
*  FastAPI REST API backend
*  Performance metrics visualization
*  Carbon capture impact analysis
*  Industrial operational parameter simulation
*  Lightweight frontend with zero framework dependencies

---

#  Tech Stack

## Backend

* Python
* FastAPI
* Uvicorn

## Machine Learning

* LightGBM
* Scikit-Learn
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

The predictive engine uses a **Light Gradient Boosting Machine (LightGBM) Regressor**.

LightGBM is selected because of its:

* Fast training performance
* High efficiency on large datasets
* Strong handling of non-linear industrial relationships
* Better optimization compared to traditional ensemble methods like Random Forests

---

#  Input Features

The model accepts the following operational parameters:

| Feature          | Description            | Unit    |
| ---------------- | ---------------------- | ------- |
| `fuel_flow`      | Fuel consumption rate  | tons/hr |
| `boiler_load`    | Power plant load       | MW      |
| `ambient_temp`   | External temperature   | °C      |
| `carbon_capture` | Carbon capture enabled | Boolean |

---

#  Data Pipeline

## 1. Feature Ingestion

Operational sensor data is collected from the thermal plant environment.

## 2. Data Preprocessing

Continuous numeric features are normalized using Scikit-Learn's `StandardScaler`.

This ensures that features with larger magnitudes do not dominate the optimization process.

## 3. Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

This helps evaluate model generalization on unseen operational conditions.

---

#  Model Evaluation Metrics

To evaluate prediction quality, the backend computes multiple regression metrics.

---

## 1. Mean Absolute Error (MAE)

Measures the average magnitude of prediction errors.

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
$$


---

## 2. Root Mean Squared Error (RMSE)

Penalizes larger prediction errors more heavily.

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$


---

## 3. R-Squared Score ($R^2$)

Represents how much variance in emissions is explained by the model.

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
$$


---

## 4. Mean Absolute Percentage Error (MAPE)

Measures prediction error in percentage form.

$$
\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$


---

Where:

* (y_i) = Actual value
* (\hat{y}_i) = Predicted value
* (\bar{y}) = Mean actual value
* (n) = Number of samples

---

# Model Optimization

## Overfitting (High Variance)

### Problem

The model memorizes training data but performs poorly on unseen data.

### Mitigations

* `max_depth=8`
* Strict train-test separation
* Early stopping (optional)
* Regularized tree growth

---

## Underfitting (High Bias)

### Problem

The model fails to capture the underlying relationship between operational parameters and emissions.

### Mitigations

* Using LightGBM instead of linear regression
* `learning_rate=0.05`
* `n_estimators=200`
* Feature scaling and tuning

---

# 📂 Project Structure

```bash
EcoGridAI/
│
├── backend/
│   ├── main.py
│   ├── train.py
│   ├── requirements.txt
│   ├── model.pkl
│   ├── preprocessor.joblib
│   └── metadata.json
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
└── README.md
```

---

#  Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/am-prasad/webml.git
cd webml
```

---

## 2. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

#  Train the Model

Generate the trained model and preprocessing artifacts:

```bash
python train.py
```

Generated files:

* `model.pkl`
* `preprocessor.joblib`
* `metadata.json`

---

# ▶️ Run the API Server

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Server runs at:

```bash
http://127.0.0.1:8000
```

---

# 🌐 Launch the Frontend

Open:

```bash
frontend/index.html
```

in any modern browser.

No npm installation or frontend build tools are required.

---

# 📡 API Endpoint

## Predict CO₂ Emissions

### POST `/predict`

### Request Example

```json
{
  "fuel_flow": 120,
  "boiler_load": 450,
  "ambient_temp": 32,
  "carbon_capture": 1
}
```

### Response Example

```json
{
  "predicted_co2": 18.74
}
```

---

# 📈 Future Enhancements

*  Real industrial IoT integration
*  Cloud deployment
* Advanced analytics dashboard
*  Historical emissions reporting
* Deep learning experimentation
*  Live environmental compliance monitoring

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a pull request






