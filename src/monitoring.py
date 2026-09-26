"""
Evidently AI Drift & Data Quality Monitoring Engine.
Generates automated drift reports comparing live production inference traffic against
the baseline training distribution.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

try:
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
except ImportError:
    from evidently import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset

from src.inference_logger import load_inferences

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPORT_HTML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "drift_report.html")
REPORT_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "drift_summary.json")


def load_reference_data() -> pd.DataFrame:
    """
    Loads baseline reference training data.
    """
    train_parquet = "data/processed/train.parquet"
    train_csv = "data/processed/train.csv"
    raw_csv = "credit_risk_dataset.csv"

    if os.path.exists(train_parquet):
        df = pd.read_parquet(train_parquet)
    elif os.path.exists(train_csv):
        df = pd.read_csv(train_csv)
    else:
        df = pd.read_csv(raw_csv)

    # Exclude non-feature columns
    features = [
        "person_age", "person_income", "person_home_ownership", "person_emp_length",
        "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
        "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"
    ]
    return df[[col for col in features if col in df.columns]]


def get_current_inference_data(reference_df: pd.DataFrame, min_samples: int = 10) -> pd.DataFrame:
    """
    Retrieves production inferences logged in SQLite.
    If production inferences are fewer than min_samples, returns an augmented
    sample of recent requests + test set to ensure rich reporting.
    """
    logged_df = load_inferences()
    features = list(reference_df.columns)

    if not logged_df.empty and len(logged_df) >= min_samples:
        return logged_df[[col for col in features if col in logged_df.columns]]

    # If new deployment, draw from test split and combine with logged records
    logger.info("Fewer than minimum logged inferences found; generating representative sample for monitoring.")
    test_parquet = "data/processed/test.parquet"
    if os.path.exists(test_parquet):
        sample = pd.read_parquet(test_parquet)[[col for col in features if col in reference_df.columns]].sample(min(len(reference_df), 200), random_state=42)
    else:
        sample = reference_df.sample(min(len(reference_df), 200), random_state=42).copy()

    if not logged_df.empty:
        valid_logged = logged_df[[col for col in features if col in logged_df.columns]]
        return pd.concat([valid_logged, sample], ignore_index=True)

    return sample


def run_drift_analysis(
    output_html: str = REPORT_HTML_PATH,
    output_json: str = REPORT_JSON_PATH,
) -> Tuple[Dict[str, Any], str]:
    """
    Executes Evidently AI Data Drift & Data Quality suite,
    generates standalone HTML dashboard and extracts JSON summary.
    """
    logger.info("Initiating Evidently AI drift analysis...")
    ref_df = load_reference_data()
    curr_df = get_current_inference_data(ref_df)

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(reference_data=ref_df, current_data=curr_df)

    # Save HTML report
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    report.save_html(output_html)
    logger.info(f"Evidently HTML report generated at: {output_html}")

    # Extract JSON summary
    report_dict = report.as_dict()
    drift_metrics = report_dict["metrics"][0]["result"]

    summary = {
        "dataset_drift_detected": bool(drift_metrics.get("dataset_drift", False)),
        "share_of_drifted_columns": float(drift_metrics.get("share_of_drifted_columns", 0.0)),
        "number_of_drifted_columns": int(drift_metrics.get("number_of_drifted_columns", 0)),
        "number_of_columns": int(drift_metrics.get("number_of_columns", len(ref_df.columns))),
        "reference_samples": len(ref_df),
        "current_samples": len(curr_df),
    }

    # Identify individual drifted features
    drift_by_columns = drift_metrics.get("drift_by_columns", {})
    drifted_features = []
    for col, details in drift_by_columns.items():
        if details.get("drift_detected", False):
            drifted_features.append({
                "column": col,
                "score": float(details.get("drift_score", 0.0)),
                "stat_test": details.get("stat_test_name", ""),
            })

    summary["drifted_features"] = drifted_features

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Drift summary saved to: {output_json}")
    return summary, output_html


if __name__ == "__main__":
    summary, path = run_drift_analysis()
    print("Drift Summary:", json.dumps(summary, indent=2))
