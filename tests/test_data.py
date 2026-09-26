"""
Unit Tests for Data Loader and Cleaning Logic.
"""

import pandas as pd
import pytest
from src.data_loader import validate_and_clean_data, split_data


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        "person_age": [25, 120, 16, 30, 25],
        "person_income": [50000, 60000, 30000, 70000, 50000],
        "person_home_ownership": ["RENT", "OWN", "RENT", "MORTGAGE", "RENT"],
        "person_emp_length": [3.0, 5.0, 1.0, 35.0, 3.0], # 35 > 30 age! (invalid)
        "loan_intent": ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "PERSONAL"],
        "loan_grade": ["A", "B", "C", "D", "A"],
        "loan_amnt": [5000.0, 10000.0, 0.0, 15000.0, 5000.0], # 0 is invalid
        "loan_int_rate": [7.5, 11.0, 14.0, 16.0, 7.5],
        "loan_status": [0, 1, 1, 0, 0],
        "loan_percent_income": [0.1, 0.17, 0.0, 0.21, 0.1],
        "cb_person_default_on_file": ["N", "N", "Y", "N", "N"],
        "cb_person_cred_hist_length": [3, 4, 2, 8, 3],
    })


def test_validate_and_clean_data(sample_raw_data):
    val_config = {
        "min_age": 18,
        "max_age": 100,
        "max_emp_length": 60,
        "min_loan_amnt": 0,
    }
    cleaned = validate_and_clean_data(sample_raw_data, val_config)

    # Row 4 is duplicate of Row 0 -> dropped
    # Row 1 has age 120 > 100 -> dropped
    # Row 2 has age 16 < 18 and loan_amnt 0 -> dropped
    # Row 3 has emp_length 35 > age 30 -> dropped
    # Only Row 0 should survive!
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["person_age"] == 25
    assert cleaned.iloc[0]["loan_amnt"] == 5000.0


def test_split_data():
    df = pd.DataFrame({
        "feature1": list(range(100)),
        "loan_status": [0] * 80 + [1] * 20,
    })
    X_train, X_test, y_train, y_test = split_data(df, target_col="loan_status", test_size=0.2, random_state=42)

    assert len(X_train) == 80
    assert len(X_test) == 20
    # Stratification check: 20% of class 1 in test is 4
    assert sum(y_test == 1) == 4
