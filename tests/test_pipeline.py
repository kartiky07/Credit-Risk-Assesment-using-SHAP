"""
Unit Tests for Preprocessor, Thresholding, and Evaluation Metrics.
"""

import numpy as np
import pandas as pd
import pytest
from src.preprocessing import build_preprocessor
from src.threshold import find_optimal_threshold
from src.evaluate import compute_metrics


def test_preprocessor():
    num_cols = ["age", "income"]
    cat_cols = ["home", "grade"]

    preprocessor = build_preprocessor(num_cols, cat_cols)

    train_df = pd.DataFrame({
        "age": [25, np.nan, 35],
        "income": [50000.0, 70000.0, np.nan],
        "home": ["RENT", "OWN", None],
        "grade": ["A", "B", "A"],
    })

    preprocessor.fit(train_df)
    transformed = preprocessor.transform(train_df)

    # Output should be 2D array without NaNs
    assert transformed.shape[0] == 3
    assert not np.isnan(transformed).any()

    # Unseen category during inference should not raise error
    test_df = pd.DataFrame({
        "age": [40],
        "income": [60000.0],
        "home": ["UNKNOWN_CATEGORY"],
        "grade": ["C"],
    })
    test_transformed = preprocessor.transform(test_df)
    assert test_transformed.shape[0] == 1


def test_threshold_optimizer():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.35, 0.6, 0.75, 0.8, 0.4, 0.9])

    best_thresh, details = find_optimal_threshold(y_true, y_prob)

    assert 0.0 < best_thresh < 1.0
    assert details["max_f1"] > 0.5


def test_compute_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
