"""
Utility functions for evaluation module.
"""

import json
import os
from pathlib import Path

import numpy as np


def load_predictions(result_dir: str) -> dict:
    """
    Load predictions from a result directory.

    Args:
        result_dir: Path to directory containing predictions.npz

    Returns:
        Dictionary with keys:
            - predictions_mean: (num_series, num_windows, num_variates, pred_len)
            - predictions_samples: (num_series, num_windows, num_samples, num_variates, pred_len)
            - ground_truth: (num_series, num_windows, num_variates, pred_len)
            - context: (num_series, num_windows, num_variates, max_ctx_len)
    """
    npz_path = os.path.join(result_dir, "predictions.npz")
    data = np.load(npz_path)
    return {
        "predictions_mean": data["predictions_mean"],
        "predictions_samples": data["predictions_samples"],
        "ground_truth": data["ground_truth"],
        "context": data["context"],
    }


def load_metrics(result_dir: str) -> dict:
    """
    Load metrics from a result directory.

    Args:
        result_dir: Path to directory containing metrics.npz

    Returns:
        Dictionary of metric arrays, each with shape (num_series, num_windows, num_variates)
    """
    metrics_path = os.path.join(result_dir, "metrics.npz")
    data = np.load(metrics_path)
    return {key: data[key] for key in data.files}


def load_metadata(result_dir: str) -> dict:
    """
    Load metadata from a result directory.

    Args:
        result_dir: Path to directory containing metadata.json

    Returns:
        Metadata dictionary
    """
    metadata_path = os.path.join(result_dir, "metadata.json")
    with open(metadata_path) as f:
        return json.load(f)


def aggregate_metrics(
    metrics: dict,
    aggregation: str = "mean",
    axis: tuple | None = None,
) -> dict:
    """
    Aggregate metrics across specified dimensions.

    Args:
        metrics: Dictionary of metric arrays with shape (num_series, num_windows, num_variates)
        aggregation: Aggregation method ("mean", "median", "std", "min", "max")
        axis: Axes to aggregate over. None for all axes.
            - (0,): aggregate over series
            - (1,): aggregate over windows
            - (2,): aggregate over variates
            - (0, 1): aggregate over series and windows
            - None: aggregate over all dimensions (returns scalar)

    Returns:
        Dictionary of aggregated metric values
    """
    agg_funcs = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "std": np.nanstd,
        "min": np.nanmin,
        "max": np.nanmax,
    }

    if aggregation not in agg_funcs:
        raise ValueError(f"Unknown aggregation: {aggregation}. Must be one of {list(agg_funcs.keys())}")

    agg_func = agg_funcs[aggregation]

    return {name: agg_func(values, axis=axis) for name, values in metrics.items()}


def find_result_dirs(base_dir: str, pattern: str = "*") -> list[str]:
    """
    Find all result directories matching a pattern.

    Args:
        base_dir: Base directory to search
        pattern: Glob pattern for directory names

    Returns:
        List of result directory paths
    """
    base_path = Path(base_dir)
    result_dirs = []

    for path in base_path.glob(pattern):
        if path.is_dir() and (path / "metadata.json").exists():
            result_dirs.append(str(path))

    return sorted(result_dirs)


def compare_models(
    model_results: dict[str, str],
    metric_name: str = "CRPS",
    aggregation: str = "mean",
) -> dict[str, float]:
    """
    Compare multiple models on a specific metric.

    Args:
        model_results: Dictionary mapping model names to result directories
        metric_name: Name of metric to compare
        aggregation: Aggregation method

    Returns:
        Dictionary mapping model names to aggregated metric values
    """
    comparison = {}

    for model_name, result_dir in model_results.items():
        metrics = load_metrics(result_dir)
        if metric_name in metrics:
            agg_metrics = aggregate_metrics({metric_name: metrics[metric_name]}, aggregation)
            comparison[model_name] = agg_metrics[metric_name]
        else:
            comparison[model_name] = np.nan

    return comparison


def get_available_terms(dataset_name: str, config: dict) -> list[str]:
    """Get the terms that are actually defined in the config for a dataset."""
    datasets_config = config.get("datasets", {})
    if dataset_name not in datasets_config:
        return []
    dataset_config = datasets_config[dataset_name]
    available_terms = []
    for term in ["short", "medium", "long"]:
        if term in dataset_config and dataset_config[term].get("prediction_length") is not None:
            available_terms.append(term)
    return available_terms
