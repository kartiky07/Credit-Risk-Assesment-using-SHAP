"""
Inference Logger Module.
Asynchronously logs incoming applicant feature vectors and predictions into an SQLite database
for continuous drift and data quality monitoring with Evidently AI.
"""

import os
import sqlite3
import datetime
import pandas as pd
from typing import Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "inferences.db")


def init_db(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                person_age INTEGER,
                person_income REAL,
                person_home_ownership TEXT,
                person_emp_length REAL,
                loan_intent TEXT,
                loan_grade TEXT,
                loan_amnt REAL,
                loan_int_rate REAL,
                loan_percent_income REAL,
                cb_person_default_on_file TEXT,
                cb_person_cred_hist_length INTEGER,
                default_probability REAL,
                default_prediction INTEGER
            )
        """)
        conn.commit()


def log_inference(input_data: Dict[str, Any], probability: float, prediction: int, db_path: str = DB_PATH):
    """
    Appends an individual prediction event to the database.
    """
    try:
        init_db(db_path)
        timestamp = datetime.datetime.utcnow().isoformat()
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO inferences (
                    timestamp,
                    person_age,
                    person_income,
                    person_home_ownership,
                    person_emp_length,
                    loan_intent,
                    loan_grade,
                    loan_amnt,
                    loan_int_rate,
                    loan_percent_income,
                    cb_person_default_on_file,
                    cb_person_cred_hist_length,
                    default_probability,
                    default_prediction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                input_data.get("person_age"),
                input_data.get("person_income"),
                input_data.get("person_home_ownership"),
                input_data.get("person_emp_length"),
                input_data.get("loan_intent"),
                input_data.get("loan_grade"),
                input_data.get("loan_amnt"),
                input_data.get("loan_int_rate"),
                input_data.get("loan_percent_income"),
                input_data.get("cb_person_default_on_file"),
                input_data.get("cb_person_cred_hist_length"),
                probability,
                prediction
            ))
            conn.commit()
    except Exception as e:
        print(f"Warning: Failed to log inference: {e}")


def load_inferences(limit: Optional[int] = 1000, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Loads historical inference records into a pandas DataFrame.
    """
    if not os.path.exists(db_path):
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        query = "SELECT * FROM inferences ORDER BY id DESC"
        if limit:
            query += f" LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
    return df
