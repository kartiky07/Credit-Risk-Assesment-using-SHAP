# Credit Risk Assessment & SHAP Explainability — MLOps Guide

This repository contains an end-to-end production **MLOps Pipeline** for Credit Risk Assessment and Explainability, migrating from experimental Jupyter notebooks into an automated, observable, and containerized machine learning architecture.

---

## Architecture Overview

```
                                  +-----------------------+
                                  | credit_risk_dataset   |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |  src/data_loader.py   |  (Validation & Cleaning)
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |  src/preprocessing.py |  (ColumnTransformer Pipeline)
                                  +-----------+-----------+
                                              |
                                              v
+-----------------------+         +-----------------------+
|    MLflow Tracking    |<------->|     src/train.py      |  (XGBoost + Platt Scaling + Threshold Optimization)
| (Params/Metrics/Plots)|         +-----------+-----------+
+-----------------------+                     |
                                              +---------------------------------------+
                                              |                                       |
                                              v                                       v
                                  +-----------------------+               +-----------------------+
                                  | credit_risk_model.pkl |               |  shap_explainer.pkl   |
                                  |   best_threshold.pkl  |               +-----------+-----------+
                                  +-----------+-----------+                           |
                                              |                                       |
                                              +-------------------+-------------------+
                                                                  |
                                                                  v
                                                      +-----------------------+
                                                      |   FastAPI Service     |
                                                      |  /predict  &  /explain|
                                                      +-----------+-----------+
                                                                  |
                                                                  v
                                                      +-----------------------+
                                                      | Modern Web Interface  |
                                                      +-----------------------+
```

---

## 1. Quickstart

### Environment Setup
Activate the virtual environment:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Run Serving Application
```bash
python main.py
# or
uvicorn main:app --reload --port 8000
```
- **Web Interface:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 2. Running the MLOps Pipeline

### End-to-End Retraining & Tracking
To re-run data validation, model training, probability calibration, threshold optimization, and SHAP explainer generation with MLflow tracking:
```bash
python src/train.py
```

### Launch MLflow UI
To view experiment runs, metrics (ROC-AUC, Brier score, F1), calibration reliability curves, and ROC curves:
```bash
mlflow ui --port 5000
```
Then navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## 3. Testing Suite

Run the automated test suite covering data validation, preprocessing integrity, threshold optimization, and API contracts:
```bash
pytest tests/ -v
```

---

## 4. API Endpoints

### 1. `/predict` (POST)
Assesses credit default risk for an applicant.
- **Request:**
```json
{
  "person_age": 25,
  "person_income": 50000.0,
  "person_home_ownership": "RENT",
  "person_emp_length": 3.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 10000.0,
  "loan_int_rate": 10.5,
  "loan_percent_income": 0.2,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 3
}
```
- **Response:**
```json
{
  "default_probability": 0.0189,
  "default_prediction": 0,
  "threshold": 0.6952,
  "Result": "Low Risk"
}
```

### 2. `/explain` (POST)
Computes real-time SHAP feature importance for the individual applicant (vital for regulatory adverse action notices).
- **Response:**
```json
{
  "status": "success",
  "top_risk_drivers": [
    {"feature": "loan_percent_income", "impact": 0.42, "direction": "Increases Risk"},
    {"feature": "loan_int_rate", "impact": 0.21, "direction": "Increases Risk"}
  ],
  "top_mitigating_factors": [
    {"feature": "person_income", "impact": -0.38, "direction": "Decreases Risk"},
    {"feature": "cb_person_default_on_file_N", "impact": -0.15, "direction": "Decreases Risk"}
  ]
}
```

---

## 5. Docker Deployment

### Run with Docker Compose
To run both the production API server and the MLflow tracking server together:
```bash
docker compose up --build
```
- API: [http://localhost:8000](http://localhost:8000)
- MLflow: [http://localhost:5000](http://localhost:5000)

---

## 6. Continuous Integration (CI/CD)

The GitHub Actions workflow at `.github/workflows/ci.yml` automatically triggers on every pull request to:
- Verify data loader & preprocessing transformations.
- Run `pytest` unit & integration test suites.
- Ensure zero breaking changes reach the `main` branch.
