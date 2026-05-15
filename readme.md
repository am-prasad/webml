# EcoGridAI: Dual-Engine CO₂ Emission & Anomaly Monitor

EcoGridAI is a full-stack, machine-learning-powered web application designed to monitor CO₂ emissions and detect equipment anomalies in thermal power plants in real time.

Moving beyond simple regression, this system utilizes a **Dual-Engine AI Architecture** to predict environmental compliance states (**Normal vs. High Emissions**) while simultaneously acting as an automated watchdog for equipment failure and sensor drift.

---

#  Features

- **Dual-Engine ML:** Simultaneous state classification and anomaly detection.
- **State Prediction:** Machine Learning powered by LightGBM Classifier.
- **Anomaly Detection:** Unsupervised learning via Isolation Forest.
- **Interactive Frontend Dashboard:** Built with Chart.js, featuring real-time anomaly alerts.
- **FastAPI REST API Backend:** High-performance, asynchronous data processing.
- **Automated Visualizations:** Auto-generates Confusion Matrices, ROC Curves, and Feature Importance plots.
- **Lightweight Frontend:** Zero framework dependencies (No npm required).

---

#  Tech Stack

## Backend
- Python
- FastAPI
- Uvicorn

## Machine Learning
- LightGBM
- Scikit-Learn
- Pandas & NumPy
- Joblib
- Matplotlib & Seaborn (for automated plotting)

## Frontend
- HTML5
- CSS3
- JavaScript
- Chart.js

---

# Machine Learning Architecture

The predictive system relies on a **Dual-Engine Architecture** running in parallel:

## Engine 1: LightGBM Classifier (State Prediction)

Instead of predicting raw CO₂ numbers, it classifies the plant's state into:
- Normal Emission (0)
- High Emission (1)

### Advantages
- Fast training performance via leaf-wise tree growth.
- Strong handling of non-linear thermodynamic relationships.

---

## Engine 2: Isolation Forest (Anomaly Detection)

An unsupervised learning model that acts as an equipment watchdog.

### Capabilities
- Isolates observations to detect physically impossible states  
  (e.g., max fuel flow but zero boiler load).
- Alerts operators to potential sensor drift or mechanical failures.

---

#  Input Features

The models accept the following operational parameters:

| Feature | Description | Unit |
|---|---|---|
| `fuel_flow` | Fuel consumption rate | tons/hr |
| `boiler_load` | Power plant load | MW |
| `ambient_temp` | External temperature | °C |
| `carbon_capture` | Carbon capture enabled | Boolean |

---

#  Data Pipeline

## 1. Feature Ingestion
Operational sensor data is collected from the thermal plant environment.

---

## 2. Dynamic Thresholding (Binarization)

Continuous CO₂ data is converted into binary classes using the dataset's median.

This ensures a perfectly balanced 50/50 class distribution, preventing the **"Accuracy Paradox"** associated with imbalanced data.

---

## 3. Data Preprocessing

Continuous numeric features are normalized using Scikit-Learn's `StandardScaler`.

This ensures that features with larger magnitudes do not dominate the optimization process.

---

## 4. Stratified Train-Test Split

The dataset is divided into:
- 80% Training Data
- 20% Testing Data

Stratification ensures the exact ratio of High/Low emissions is maintained in both sets.

---

#  Model Evaluation Metrics

Because the primary engine is a **Classifier**, we evaluate prediction quality using **Confusion Matrix metrics** rather than regression errors.

---

## 1. Accuracy

The overall proportion of correct predictions across both High and Normal emission states.

```math
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
```

---

## 2. Precision

When the model triggers a **"High Emission"** alert, how often is it actually correct?  
(Measures False Alarm rate).

```math
\text{Precision} = \frac{TP}{TP + FP}
```

---

## 3. Recall (Sensitivity)

Out of all the actual **"High Emission"** events, how many did the model successfully catch?  
(Measures Missed Alarm rate).

```math
\text{Recall} = \frac{TP}{TP + FN}
```

---

## 4. F1 Score

The harmonic mean of Precision and Recall.

This is the ultimate metric for classification, penalizing the model for false alarms and missed alarms alike.

```math
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
```

Where:
- TP = True Positives
- TN = True Negatives
- FP = False Positives
- FN = False Negatives

---

# Model Optimization

## Overfitting (High Variance)

### Problem
The model memorizes training data but performs poorly on unseen data.

### Mitigations
- 5-Fold Stratified Cross-Validation to prove consistency across multiple data slices.
- `max_depth=8` restriction.
- Strict train-test separation.

---

## Underfitting (High Bias)

### Problem
The model fails to capture the underlying relationship between operational parameters and emissions.

### Mitigations
- Using Gradient Boosting (LightGBM) instead of simple linear models.
- `learning_rate=0.05` to ensure methodical pattern learning.
- `n_estimators=200` to build sufficient tree depth.

---

# Project Structure

```bash
EcoGrid/
│
├── backend/
│   ├── main.py
│   ├── train.py
│   ├── requirements.txt
│   ├── model.pkl                 # LightGBM Classifier
│   ├── anomaly_detector.pkl      # Isolation Forest Model
│   ├── preprocessor.joblib       
│   ├── metadata.json
│   └── plots/                   
│       ├── 1_confusion_matrix.png
│       ├── 2_feature_importance.png
│       ├── 3_roc_curve.png
│       └── 4_anomaly_scatter.png
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── frontend.js
│
└── README.md
```

---

# ⚙️ Installation & Setup

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

# Train the Models

Generate the trained models, preprocessing artifacts, and visualization plots:

```bash
python train.py
```

## Generated Files
- `model.pkl`
- `anomaly_detector.pkl`
- `preprocessor.joblib`
- `metadata.json`
- `/plots` directory images

---

#  Run the API Server

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Server runs at:

```bash
http://127.0.0.1:8000
```

---

#  Launch the Frontend

Open:

```bash
frontend/index.html
```

in any modern browser.

No npm installation or frontend build tools are required.

---

#  API Endpoint

## Predict State & Anomalies

### POST `/predict`

### Request Example

```json
{
  "fuelflow": 120.5,
  "boilerload": 450.0,
  "ambient_temp": 32.0,
  "capture_on": 1
}
```

### Response Example

```json
{
  "prediction": 1,
  "is_anomaly": false,
  "status": "success"
}
```

> Note: `is_anomaly: true` will trigger the frontend UI alert banner.

---

# Future Enhancements

- Real industrial IoT (SCADA) integration
- Cloud deployment (AWS/Render)
- Advanced analytics dashboard with historical plots
- Historical emissions reporting
- Deep learning experimentation (Autoencoders for Anomaly Detection)
- Live environmental compliance monitoring

---

# Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a pull request