"""
End-to-End Training and MLOps Pipeline Orchestrator with MLflow.
Trains XGBoost, calibrates probabilities, optimizes threshold, builds SHAP explainer,
and logs all parameters, metrics, plots, and models to MLflow.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import yaml
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from src.data_loader import load_config, run_data_pipeline
from src.preprocessing import build_preprocessor
from src.threshold import find_optimal_threshold
from src.evaluate import compute_metrics, generate_calibration_plot, generate_roc_plot
from src.explain import build_and_save_explainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_pipeline(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    config = load_config(config_path)

    # 1. Load and prepare data
    logger.info("Step 1: Running data pipeline...")
    X_train, X_test, y_train, y_test = run_data_pipeline(config_path)

    # Calculate class imbalance weighting
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = float(neg_count / max(pos_count, 1))
    logger.info(f"Class counts: 0={neg_count}, 1={pos_count}. scale_pos_weight={scale_pos_weight:.2f}")

    # 2. Preprocessor
    numeric_cols = config["features"]["numeric_cols"]
    categorical_cols = config["features"]["categorical_cols"]
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # 3. XGBoost Model
    xgb_params = {
        "n_estimators": config["model"].get("n_estimators", 300),
        "max_depth": config["model"].get("max_depth", 5),
        "learning_rate": config["model"].get("learning_rate", 0.1),
        "scale_pos_weight": scale_pos_weight,
        "random_state": config["model"].get("random_state", 42),
        "eval_metric": "logloss",
    }

    xgb_clf = XGBClassifier(**xgb_params)
    base_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb_clf),
    ])

    # 4. Probability Calibration
    calib_method = config["model"].get("calibration_method", "sigmoid")
    calib_cv = config["model"].get("calibration_cv", 5)
    logger.info(f"Step 2: Training CalibratedClassifierCV (method={calib_method}, cv={calib_cv})...")

    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method=calib_method,
        cv=calib_cv,
    )
    calibrated_model.fit(X_train, y_train)
    logger.info("Model fitting complete.")

    # 5. Predictions & Threshold Optimization
    logger.info("Step 3: Evaluating probabilities and optimizing threshold...")
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    best_threshold, threshold_details = find_optimal_threshold(y_test.values, y_prob)
    logger.info(f"Optimal F1 Threshold: {best_threshold:.4f} (Max F1: {threshold_details['max_f1']:.4f})")

    # 6. Metrics & Plots
    metrics = compute_metrics(y_test.values, y_prob, threshold=best_threshold)
    metrics["scale_pos_weight"] = scale_pos_weight
    logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f} | Brier Score: {metrics['brier_score']:.4f} | F1: {metrics['f1']:.4f}")

    calib_fig = generate_calibration_plot(y_test.values, y_prob, model_name="Calibrated XGBoost")
    roc_fig = generate_roc_plot(y_test.values, y_prob, roc_auc=metrics["roc_auc"])

    # 7. Persist Local Artifacts
    models_dir = config["artifacts"].get("models_dir", "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = config["artifacts"].get("model_path", "credit_risk_model.pkl")
    threshold_path = config["artifacts"].get("threshold_path", "best_threshold.pkl")
    explainer_path = config["artifacts"].get("explainer_path", "models/shap_explainer.pkl")
    metrics_path = config["artifacts"].get("metrics_path", "models/metrics.json")

    joblib.dump(calibrated_model, model_path)
    joblib.dump(best_threshold, threshold_path)

    # Save metrics JSON
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # 8. SHAP Explainer
    logger.info("Step 4: Building and saving SHAP Explainer...")
    # Extract fitted preprocessor
    fitted_preprocessor = calibrated_model.calibrated_classifiers_[0].estimator.named_steps["preprocessor"]
    build_and_save_explainer(calibrated_model, fitted_preprocessor, numeric_cols, categorical_cols, explainer_path)

    # 9. MLflow Tracking
    try:
        import mlflow
        experiment_name = config["mlflow"].get("experiment_name", "credit-risk-assessment")
        tracking_uri = config["mlflow"].get("tracking_uri", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="calibrated_xgboost_run"):
            # Log params
            mlflow.log_params(xgb_params)
            mlflow.log_param("calibration_method", calib_method)
            mlflow.log_param("calibration_cv", calib_cv)

            # Log metrics
            mlflow.log_metrics({
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "f1_score": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "accuracy": metrics["accuracy"],
                "best_threshold": best_threshold,
            })

            # Log plots
            calib_plot_path = os.path.join(models_dir, "calibration_curve.png")
            roc_plot_path = os.path.join(models_dir, "roc_curve.png")
            calib_fig.savefig(calib_plot_path)
            roc_fig.savefig(roc_plot_path)

            mlflow.log_artifact(calib_plot_path, artifact_path="plots")
            mlflow.log_artifact(roc_plot_path, artifact_path="plots")
            mlflow.log_artifact(metrics_path, artifact_path="metrics")
            mlflow.log_artifact(model_path, artifact_path="model")
            mlflow.log_artifact(threshold_path, artifact_path="threshold")
            mlflow.log_artifact(explainer_path, artifact_path="explainer")

            logger.info("Successfully logged parameters, metrics, plots, and artifacts to MLflow.")
    except Exception as e:
        logger.warning(f"MLflow logging skipped or encountered an error: {e}")

    logger.info("Training pipeline completed successfully.")
    return {
        "metrics": metrics,
        "model_path": model_path,
        "threshold_path": threshold_path,
        "explainer_path": explainer_path,
    }


if __name__ == "__main__":
    train_pipeline()
