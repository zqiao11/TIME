"""
Evaluation module for per-window time series forecasting evaluation.

This module provides functionality to:
- Compute per-window metrics for forecasting models
- Save predictions, ground truth, and metrics in structured format
- Support probabilistic forecasts with quantile and CRPS metrics

Based on GIFT-Eval framework.
"""

from timebench.evaluation.metrics import compute_per_window_metrics
from timebench.evaluation.saver import save_window_predictions

__all__ = [
    "compute_per_window_metrics",
    "save_window_predictions",
]



