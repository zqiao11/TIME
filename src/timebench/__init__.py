"""
TIME Benchmark: A comprehensive Time Series Benchmarking framework.

Modules:
    - feature: Feature extraction and computation
    - preprocessing: Data preprocessing pipeline
    - evaluation: Model evaluation with per-window metrics
"""

__version__ = "0.1.0"

from timebench.evaluation import compute_per_window_metrics_from_quantiles, save_window_predictions

__all__ = [
    "__version__",
    "compute_per_window_metrics_from_quantiles",
    "save_window_predictions",
]



