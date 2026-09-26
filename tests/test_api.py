"""
API Contract and Integration Tests for FastAPI Serving.
"""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_openapi(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "paths" in response.json()


def test_predict_endpoint_valid(client):
    payload = {
        "person_age": 28,
        "person_income": 65000.0,
        "person_home_ownership": "RENT",
        "person_emp_length": 4.0,
        "loan_intent": "EDUCATION",
        "loan_grade": "A",
        "loan_amnt": 8000.0,
        "loan_int_rate": 7.9,
        "loan_percent_income": 0.12,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "default_probability" in data
    assert "default_prediction" in data
    assert "threshold" in data
    assert "Result" in data
    assert 0.0 <= data["default_probability"] <= 1.0
    assert data["default_prediction"] in [0, 1]


def test_predict_endpoint_missing_fields(client):
    invalid_payload = {
        "person_age": 28,
        "person_home_ownership": "RENT",
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_explain_endpoint(client):
    payload = {
        "person_age": 28,
        "person_income": 65000.0,
        "person_home_ownership": "RENT",
        "person_emp_length": 4.0,
        "loan_intent": "EDUCATION",
        "loan_grade": "A",
        "loan_amnt": 8000.0,
        "loan_int_rate": 7.9,
        "loan_percent_income": 0.12,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4,
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_monitoring_status_endpoint(client):
    response = client.get("/monitoring/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data or "dataset_drift_detected" in data
