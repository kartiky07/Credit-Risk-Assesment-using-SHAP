"""
Explainability & Interpretability Module using SHAP.
Builds TreeExplainer and generates local instance explanations for credit decisions.
"""

import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, List


def get_feature_names(preprocessor, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
    """
    Extracts transformed feature names from ColumnTransformer.
    """
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_cols))
    return numeric_cols + cat_feature_names


def build_and_save_explainer(
    model,
    preprocessor,
    numeric_cols: List[str],
    categorical_cols: List[str],
    save_path: str = "models/shap_explainer.pkl",
):
    """
    Extracts underlying tree model from pipeline or CalibratedClassifierCV,
    fits a TreeExplainer, and saves it.
    """
    # If wrapped in CalibratedClassifierCV
    if hasattr(model, "calibrated_classifiers_"):
        base_pipeline = model.calibrated_classifiers_[0].estimator
    elif hasattr(model, "named_steps"):
        base_pipeline = model
    else:
        base_pipeline = model

    if hasattr(base_pipeline, "named_steps") and "classifier" in base_pipeline.named_steps:
        xgb_clf = base_pipeline.named_steps["classifier"]
    else:
        xgb_clf = base_pipeline

    explainer = shap.TreeExplainer(xgb_clf)
    feature_names = get_feature_names(preprocessor, numeric_cols, categorical_cols)

    payload = {
        "explainer": explainer,
        "feature_names": feature_names,
    }

    joblib.dump(payload, save_path)
    return payload


def explain_instance(
    explainer_payload: Dict[str, Any],
    preprocessor,
    instance_df: pd.DataFrame,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Calculates local SHAP waterfall / force values for a single applicant.
    Returns the top factors increasing risk and factors decreasing risk.
    """
    explainer = explainer_payload["explainer"]
    feature_names = explainer_payload["feature_names"]

    # Transform input
    transformed = preprocessor.transform(instance_df)
    shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        # Class 1 (Default) SHAP values
        vals = shap_values[1][0]
    elif len(shap_values.shape) == 2:
        vals = shap_values[0]
    else:
        vals = shap_values

    contributions = []
    for name, val in zip(feature_names, vals):
        contributions.append({
            "feature": name,
            "impact": float(val),
            "direction": "Increases Risk" if val > 0 else "Decreases Risk",
        })

    # Sort by absolute impact
    sorted_contributions = sorted(contributions, key=lambda x: abs(x["impact"]), reverse=True)

    risk_increasing = [c for c in sorted_contributions if c["impact"] > 0][:top_k]
    risk_decreasing = [c for c in sorted_contributions if c["impact"] < 0][:top_k]

    return {
        "top_risk_drivers": risk_increasing,
        "top_mitigating_factors": risk_decreasing,
        "all_sorted_factors": sorted_contributions[:top_k * 2],
    }
