"""Metric helpers to compute MAE, RMSE, and R2 for single- or multi-target predictions."""
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def compute_metrics(y_true, y_pred):
    """
    Compute MAE, RMSE, R2 for predictions.
    Returns a dict with metrics for each target (if multi-target).
    """
    results = {}
    n_targets = y_true.shape[1] if len(y_true.shape) > 1 else 1
    for t in range(n_targets):
        yt = y_true[:, t] if n_targets > 1 else y_true
        yp = y_pred[:, t] if n_targets > 1 else y_pred
        mae = mean_absolute_error(yt, yp)
        # rmse = root_mean_squared_error(yt, yp, squared=False)
        rmse = root_mean_squared_error(yt, yp)
        r2 = r2_score(yt, yp)
        results[f'target_{t}'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    return results

def metrics_explanation():
    return (
        "MAE (Mean Absolute Error): Lower is better, shows average error in units of target.\n"
        "RMSE (Root Mean Squared Error): Like MAE but penalizes larger errors more.\n"
        "R2 (Coefficient of Determination): 1.0 is perfect, 0.0 means model predicts no better than mean."
    )