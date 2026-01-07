"""
Prediction saving utilities for per-window evaluation.

Output structure:
    output_dir/
        {dataset_config}/
            predictions.npz  # Contains predictions and ground truth
            metrics.npz      # Contains per-window metrics
            config.json    # Contains dataset config
"""

import json
import os

import numpy as np

from timebench.evaluation.metrics import compute_per_window_metrics


def save_window_predictions(
    dataset,
    predictor,
    ds_config: str,
    output_base_dir: str,
    seasonality: int = 1,
    model_hyperparams: dict = None,
) -> dict:
    """
    Save predictions and metrics for each test window.

    Args:
        dataset: Dataset object with attributes:
            - test_data: Test data iterator
            - windows: Number of test windows
            - target_dim: Number of variates
            - prediction_length: Forecast horizon
            - freq: Data frequency string
        predictor: Model predictor with predict() method
        ds_config: Dataset configuration string, e.g., "m4_weekly/W/short"
        output_base_dir: Base directory for output files
        seasonality: Seasonal period length for MASE computation
        model_hyperparams: Dictionary of model hyperparameters to save in config

    Output files:
        predictions.npz:
            - predictions_mean: (num_series, num_windows, num_variates, prediction_length)
            - predictions_samples: (num_series, num_windows, num_samples, num_variates, prediction_length)

        metrics.npz:
            - Each metric array with shape (num_series, num_windows, num_variates)
            - Keys: MSE, MAE, RMSE, MAPE, sMAPE, MASE, ND, QuantileLoss_*, CRPS

        config.json:
            - Dataset configuration and config (without shapes and metric_shape)

    Returns:
        config: Dictionary containing dataset config
    """
    # Create output directory for this dataset config
    ds_output_dir = os.path.join(output_base_dir, ds_config)
    os.makedirs(ds_output_dir, exist_ok=True)

    test_data = dataset.test_data
    num_windows = dataset.windows
    num_variates = dataset.target_dim
    prediction_length = dataset.prediction_length

    # Get all forecasts
    print("    Generating predictions...")
    forecasts = list(predictor.predict(test_data.input))

    # Count number of series (test_data contains num_series * num_windows instances)
    num_total_instances = len(forecasts)
    num_series = num_total_instances // num_windows

    print(f"    Total instances: {num_total_instances}, Series: {num_series}, Windows: {num_windows}")

    # Collect ground truth labels and contexts for metrics computation
    print("    Collecting ground truth and context...")
    ground_truths = []
    contexts = []
    for inp, label in test_data:
        ground_truths.append(label["target"])
        contexts.append(inp["target"])

    # Initialize arrays
    num_samples = 100  # Fixed to 100 for all models

    predictions_samples = np.zeros((num_series, num_windows, num_samples, num_variates, prediction_length))
    ground_truth = np.zeros((num_series, num_windows, num_variates, prediction_length))

    # Find max context length for metrics computation
    # context_length is the maximum length among all context windows.
    # Different windows may have different context lengths (due to different series lengths
    # or different window positions), so we find the maximum for reference.
    context_len = max(ctx.shape[-1] for ctx in contexts) if contexts else 0
    context_array = np.full((num_series, num_windows, num_variates, context_len), np.nan)

    print("    Organizing data into arrays...")
    for idx, (fc, gt, ctx) in enumerate(zip(forecasts, ground_truths, contexts)):
        series_idx = idx // num_windows
        window_idx = idx % num_windows

        # Get forecast mean and samples
        fc_samples = fc.samples

        # Handle univariate case (add dimension if needed)
        if fc_samples.ndim == 2:
            fc_samples = fc_samples[:, np.newaxis, :]
        elif fc_samples.shape[1] == prediction_length and fc_samples.shape[2] == num_variates:
            fc_samples = fc_samples.transpose(0, 2, 1)

        if gt.ndim == 1:
            gt = gt[np.newaxis, :]
        elif gt.shape[0] == prediction_length and gt.shape[1] == num_variates:
            gt = gt.T

        if ctx.ndim == 1:
            ctx = ctx[np.newaxis, :]
        elif ctx.shape[0] != num_variates:
            ctx = ctx.T

        predictions_samples[series_idx, window_idx] = fc_samples
        ground_truth[series_idx, window_idx] = gt

        # Store context (padded with NaN for shorter contexts) for metrics computation
        ctx_len = ctx.shape[-1]
        context_array[series_idx, window_idx, :, :ctx_len] = ctx

    # Save predictions to npz file (only predictions, no ground_truth or context)
    # Use float16 to reduce storage (sufficient for visualization purposes)
    npz_path = os.path.join(ds_output_dir, "predictions.npz")
    np.savez_compressed(
        npz_path,
        predictions_samples=predictions_samples.astype(np.float16),
    )
    print(f"    Saved predictions to {npz_path}")

    # Compute per-window metrics
    print("    Computing per-window metrics...")
    metrics = compute_per_window_metrics(
        predictions_samples=predictions_samples,
        ground_truth=ground_truth,
        context=context_array,
        seasonality=seasonality,
    )

    # Save metrics to npz file
    metrics_path = os.path.join(ds_output_dir, "metrics.npz")
    np.savez_compressed(metrics_path, **metrics)
    print(f"    Saved metrics to {metrics_path}")

    # Save config (without shapes and metric_shape)
    config = {
        "dataset_config": ds_config,
        "num_series": num_series,
        "num_windows": num_windows,
        "num_variates": num_variates,
        "prediction_length": prediction_length,
        "num_samples": num_samples,
        "freq": dataset.freq,
        "seasonality": seasonality,
        "context_length": context_len,
        "metric_names": list(metrics.keys()),
    }

    # Add model hyperparameters if provided
    if model_hyperparams:
        config.update(model_hyperparams)

    config_path = os.path.join(ds_output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"    Saved config to {config_path}")

    # Print average metrics summary
    print("    Metrics summary (averaged over all series/windows/variates):")
    for metric_name, metric_values in metrics.items():
        mean_val = np.nanmean(metric_values)
        print(f"      {metric_name}: {mean_val:.4f}")

    return config



