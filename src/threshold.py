"""
Decision Threshold Optimization Module.
Calculates optimal classification threshold based on F1-score maximization
or cost-sensitive financial risk matrices.
"""

import numpy as np
from typing import Tuple, Dict, Any
from sklearn.metrics import precision_recall_curve, f1_score


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Finds the optimal threshold on Precision-Recall curve.
    Evaluates both max F1 threshold and minimum cost threshold.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Avoid zero division in F1 calculation
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_f1_idx = int(np.argmax(f1_scores))
    best_f1_threshold = float(thresholds[best_f1_idx])
    max_f1 = float(f1_scores[best_f1_idx])

    # Cost-sensitive evaluation
    # FN = Default approved (costly loss of capital)
    # FP = Non-default rejected (loss of interest)
    costs = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        fn = np.sum((y_true == 1) & (preds == 0))
        fp = np.sum((y_true == 0) & (preds == 1))
        cost = fn * cost_fn + fp * cost_fp
        costs.append(cost)

    min_cost_idx = int(np.argmin(costs))
    best_cost_threshold = float(thresholds[min_cost_idx])

    details = {
        "best_f1_threshold": best_f1_threshold,
        "max_f1": max_f1,
        "best_cost_threshold": best_cost_threshold,
        "min_cost": float(costs[min_cost_idx]),
    }

    return best_f1_threshold, details
