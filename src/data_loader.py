"""
Data Ingestion, Validation, and Train/Test Splitting Module.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import logging
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_and_clean_data(df: pd.DataFrame, val_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Cleans raw data according to business rules:
    - Removes duplicates
    - Removes unrealistic ages
    - Removes employment lengths exceeding age or realistic working span
    - Filters out non-positive loan amounts
    """
    initial_rows = len(df)
    df = df.copy()

    # Deduplication
    df.drop_duplicates(inplace=True)
    dedup_rows = len(df)
    logger.info(f"Deduplication: removed {initial_rows - dedup_rows} duplicate rows.")

    min_age = val_config.get("min_age", 18)
    max_age = val_config.get("max_age", 100)
    max_emp = val_config.get("max_emp_length", 60)
    min_loan = val_config.get("min_loan_amnt", 0)

    # Age bounds
    df = df[(df["person_age"] >= min_age) & (df["person_age"] <= max_age)]

    # Employment length checks
    df = df[df["person_emp_length"] <= df["person_age"]]
    df = df[df["person_emp_length"] <= max_emp]

    # Positive loan amounts
    df = df[df["loan_amnt"] > min_loan]

    logger.info(f"Data cleaning complete: {initial_rows} -> {len(df)} rows retained.")
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "loan_status",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train/test split.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Split sizes: Train={len(X_train)} samples, Test={len(X_test)} samples.")
    return X_train, X_test, y_train, y_test


def run_data_pipeline(config_path: str = "config/config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    config = load_config(config_path)
    raw_path = config["data"]["raw_path"]
    logger.info(f"Loading raw data from: {raw_path}")
    raw_df = pd.read_csv(raw_path)

    cleaned_df = validate_and_clean_data(raw_df, config.get("validation", {}))

    X_train, X_test, y_train, y_test = split_data(
        cleaned_df,
        target_col=config["data"]["target_col"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # Optionally persist processed data
    processed_dir = config["data"].get("processed_dir", "data/processed")
    os.makedirs(processed_dir, exist_ok=True)
    train_full = pd.concat([X_train, y_train], axis=1)
    test_full = pd.concat([X_test, y_test], axis=1)

    train_path = config["data"].get("train_path", os.path.join(processed_dir, "train.parquet"))
    test_path = config["data"].get("test_path", os.path.join(processed_dir, "test.parquet"))

    try:
        train_full.to_parquet(train_path, index=False)
        test_full.to_parquet(test_path, index=False)
        logger.info(f"Saved processed train split to {train_path} and test split to {test_path}")
    except (ImportError, Exception) as e:
        csv_train = train_path.replace(".parquet", ".csv")
        csv_test = test_path.replace(".parquet", ".csv")
        train_full.to_csv(csv_train, index=False)
        test_full.to_csv(csv_test, index=False)
        logger.info(f"Saved processed train split to {csv_train} and test split to {csv_test} (CSV fallback: {e})")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_data_pipeline()
