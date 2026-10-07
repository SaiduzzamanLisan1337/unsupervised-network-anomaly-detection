from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def quantile_threshold(normal_scores: np.ndarray, quantile: float) -> float:
    if len(normal_scores) == 0:
        raise ValueError("No normal validation scores supplied.")
    return float(np.quantile(normal_scores, quantile))


def supervised_f1_threshold(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Diagnostic threshold optimization using labels; not part of the unsupervised default."""
    unique = np.unique(scores)
    if len(unique) > 512:
        candidates = np.quantile(scores, np.linspace(0.001, 0.999, 512))
    else:
        candidates = unique
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        value = f1_score(y_true, pred, zero_division=0)
        if value > best_f1:
            best_f1 = value
            best_threshold = float(threshold)
    return best_threshold
