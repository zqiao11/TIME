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

from timebench.evaluation.metrics import (
    compute_per_window_metrics,
    compute_per_window_metrics_from_quantiles,
)

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
            - predictions_samples: (num_series, num_windows, num_samples, num_variates, prediction_length)

        metrics.npz:
            - Each metric array with shape (num_series, num_windows, num_variates)

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
    prediction_length = dataset.prediction_length
    original_num_variates = dataset.target_dim  # Original variates before to_univariate

    # Get all forecasts
    print("    Generating predictions...")
    forecasts = list(predictor.predict(test_data.input))

    # Count number of series (test_data contains num_series * num_windows instances)
    num_total_instances = len(forecasts)
    num_series_flat = num_total_instances // num_windows

    print(f"    Total instances: {num_total_instances}, Series (flat): {num_series_flat}, Windows: {num_windows}")

    # Collect ground truth labels and contexts for metrics computation
    print("    Collecting ground truth and context...")
    ground_truths = []
    contexts = []
    for inp, label in test_data:
        ground_truths.append(label["target"])
        contexts.append(inp["target"])

    # Initialize arrays
    num_samples = 100  # Fixed to 100 for all models

    # Determine if this is a to_univariate case by checking forecast shape
    # In to_univariate case: forecasts are univariate but original dataset is multivariate
    if len(forecasts) > 0:
        first_fc = forecasts[0]
        if hasattr(first_fc, 'samples'):
            fc_samples_check = first_fc.samples
        elif hasattr(first_fc, 'to_sample_forecast'):
            sample_fc = first_fc.to_sample_forecast(num_samples=num_samples)
            fc_samples_check = sample_fc.samples
        else:
            raise ValueError(f"Unknown forecast type: {type(first_fc)}")

        # Check if forecasts are univariate
        forecast_is_univariate = (fc_samples_check.ndim == 2) or \
                                  (fc_samples_check.ndim == 3 and min(fc_samples_check.shape[1:]) == 1)
    else:
        forecast_is_univariate = False

    # Detect to_univariate case: forecasts are univariate but original dataset has multiple variates
    # In this case, num_series_flat = original_num_series * original_num_variates
    is_to_univariate = forecast_is_univariate and original_num_variates > 1 and \
                       (num_series_flat % original_num_variates == 0)

    if is_to_univariate:
        # Reverse the to_univariate transformation
        # MultivariateToUnivariate expands as: series_0_dim0, series_0_dim1, ..., series_1_dim0, ...
        num_series = num_series_flat // original_num_variates
        num_variates = original_num_variates
        print(f"    Detected to_univariate case: {num_series_flat} flat series -> {num_series} series x {num_variates} variates")
    else:
        num_series = num_series_flat
        num_variates = original_num_variates if not forecast_is_univariate else 1

    predictions_samples = np.zeros((num_series, num_windows, num_samples, num_variates, prediction_length))
    ground_truth = np.zeros((num_series, num_windows, num_variates, prediction_length))

    # Find max context length for metrics computation
    context_len = max(ctx.shape[-1] for ctx in contexts) if contexts else 0
    context_array = np.full((num_series, num_windows, num_variates, context_len), np.nan)

    print("    Organizing data into arrays...")
    for idx, (fc, gt, ctx) in enumerate(zip(forecasts, ground_truths, contexts)):
        # Get forecast samples
        if hasattr(fc, 'samples'):
            fc_samples = fc.samples
        elif hasattr(fc, 'to_sample_forecast'):
            sample_fc = fc.to_sample_forecast(num_samples=num_samples)
            fc_samples = sample_fc.samples
        else:
            raise ValueError(f"Unknown forecast type: {type(fc)}. Expected SampleForecast or DistributionForecast.")

        # Normalize fc_samples to shape: (num_samples, 1, prediction_length) for univariate
        # or (num_samples, num_variates, prediction_length) for multivariate
        if fc_samples.ndim == 2:
            fc_samples = fc_samples[:, np.newaxis, :]
        elif fc_samples.ndim == 3:
            if fc_samples.shape[1] == prediction_length and fc_samples.shape[2] != prediction_length:
                fc_samples = fc_samples.transpose(0, 2, 1)

        # Normalize ground truth
        if gt.ndim == 1:
            gt = gt[np.newaxis, :]
        elif gt.ndim == 2 and gt.shape[0] == prediction_length and gt.shape[1] != prediction_length:
            gt = gt.T

        # Normalize context
        if ctx.ndim == 1:
            ctx = ctx[np.newaxis, :]

        if is_to_univariate:
            # In to_univariate case:
            # idx iterates as: series_0_dim0_win0, series_0_dim0_win1, ..., series_0_dim1_win0, ...
            # We need to map flat_series_idx to (series_idx, variate_idx)
            flat_series_idx = idx // num_windows
            window_idx = idx % num_windows
            series_idx = flat_series_idx // num_variates
            variate_idx = flat_series_idx % num_variates

            # fc_samples shape: (num_samples, 1, prediction_length)
            predictions_samples[series_idx, window_idx, :, variate_idx, :] = fc_samples[:, 0, :]
            ground_truth[series_idx, window_idx, variate_idx, :] = gt[0, :]

            # Store context
            ctx_len_actual = ctx.shape[-1]
            context_array[series_idx, window_idx, variate_idx, :ctx_len_actual] = ctx[0, :]
        else:
            # Normal multivariate case
            series_idx = idx // num_windows
            window_idx = idx % num_windows

            predictions_samples[series_idx, window_idx] = fc_samples
            ground_truth[series_idx, window_idx] = gt

            # Handle context shape mismatch
            if ctx.shape[0] != num_variates and ctx.shape[-1] == num_variates:
                ctx = ctx.T
            ctx_len_actual = ctx.shape[-1]
            context_array[series_idx, window_idx, :, :ctx_len_actual] = ctx

    # Pre-compute quantiles for visualization (saves storage and simplifies frontend)
    # Quantiles: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 (9 quantiles)
    # predictions_samples shape: (num_series, num_windows, num_samples, num_variates, prediction_length)
    # Output shape: (num_series, num_windows, 9, num_variates, prediction_length)
    quantile_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    predictions_quantiles = np.quantile(
        predictions_samples, quantile_levels, axis=2
    )  # shape: (9, num_series, num_windows, num_variates, prediction_length)
    # Transpose to (num_series, num_windows, 9, num_variates, prediction_length)
    predictions_quantiles = predictions_quantiles.transpose(1, 2, 0, 3, 4)

    # Save quantiles to npz file (instead of full samples)
    # Use float16 to reduce storage (sufficient for visualization purposes)
    npz_path = os.path.join(ds_output_dir, "predictions.npz")
    np.savez_compressed(
        npz_path,
        predictions_quantiles=predictions_quantiles.astype(np.float16),
        quantile_levels=quantile_levels.astype(np.float16),
    )
    print(f"    Saved predictions (quantiles) to {npz_path}")

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
        "num_quantiles": len(quantile_levels),
        "quantile_levels": quantile_levels.tolist(),
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






def save_window_quantile_predictions(
    dataset,
    predictor,
    ds_config: str,
    output_base_dir: str,
    seasonality: int = 1,
    model_hyperparams: dict = None,
    quantile_levels: list[float] = None,
) -> dict:
    """
    Save quantile predictions and metrics for each test window.

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
        quantile_levels: Quantile levels to extract from forecasts

    Output files:
        predictions.npz:
            - predictions_quantiles: (num_series, num_windows, num_quantiles, num_variates, prediction_length)
            - quantile_levels

        metrics.npz:
            - Each metric array with shape (num_series, num_windows, num_variates)

        config.json:
            - Dataset configuration and config (without shapes and metric_shape)

    Returns:
        config: Dictionary containing dataset config
    """
    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    quantile_levels_array = np.array(quantile_levels, dtype=float)
    quantile_levels_list = [float(q) for q in quantile_levels_array.tolist()]
    num_quantiles = len(quantile_levels_list)

    # Create output directory for this dataset config
    ds_output_dir = os.path.join(output_base_dir, ds_config)
    os.makedirs(ds_output_dir, exist_ok=True)

    test_data = dataset.test_data
    num_windows = dataset.windows
    prediction_length = dataset.prediction_length
    original_num_variates = dataset.target_dim  # Original variates before to_univariate

    # Get all forecasts
    print("    Generating predictions...")
    forecasts = list(predictor.predict(test_data.input))

    # Count number of series (test_data contains num_series * num_windows instances)
    num_total_instances = len(forecasts)
    num_series_flat = num_total_instances // num_windows

    print(
        f"    Total instances: {num_total_instances}, Series (flat): {num_series_flat}, Windows: {num_windows}"
    )

    # Collect ground truth labels and contexts for metrics computation
    print("    Collecting ground truth and context...")
    ground_truths = []
    contexts = []
    for inp, label in test_data:
        ground_truths.append(label["target"])
        contexts.append(inp["target"])

    # Determine if this is a to_univariate case by checking forecast shape
    if len(forecasts) > 0:
        first_fc = forecasts[0]
        if not hasattr(first_fc, "quantile"):
            raise ValueError(f"Unknown forecast type: {type(first_fc)}")
        q_check = np.asarray(first_fc.quantile(0.5))
        forecast_is_univariate = (q_check.ndim == 1) or (
            q_check.ndim == 2 and min(q_check.shape) == 1
        )
    else:
        forecast_is_univariate = False

    # Detect to_univariate case: forecasts are univariate but original dataset has multiple variates
    is_to_univariate = (
        forecast_is_univariate
        and original_num_variates > 1
        and (num_series_flat % original_num_variates == 0)
    )

    if is_to_univariate:
        # Reverse the to_univariate transformation
        num_series = num_series_flat // original_num_variates
        num_variates = original_num_variates
        print(
            "    Detected to_univariate case: "
            f"{num_series_flat} flat series -> {num_series} series x {num_variates} variates"
        )
    else:
        num_series = num_series_flat
        num_variates = original_num_variates if not forecast_is_univariate else 1

    predictions_quantiles = np.zeros(
        (num_series, num_windows, num_quantiles, num_variates, prediction_length)
    )
    ground_truth = np.zeros((num_series, num_windows, num_variates, prediction_length))

    # Find max context length for metrics computation
    context_len = max(ctx.shape[-1] for ctx in contexts) if contexts else 0
    context_array = np.full((num_series, num_windows, num_variates, context_len), np.nan)

    def _normalize_quantile_pred(q_pred: np.ndarray) -> np.ndarray:
        q_pred = np.asarray(q_pred)
        if q_pred.ndim == 1:
            q_pred = q_pred[np.newaxis, :]
        elif q_pred.ndim == 2:
            if q_pred.shape[0] == prediction_length and q_pred.shape[1] != prediction_length:
                q_pred = q_pred.T
        else:
            raise ValueError(f"Unsupported quantile prediction shape: {q_pred.shape}")

        if q_pred.shape[-1] != prediction_length:
            raise ValueError(
                "Quantile prediction length mismatch: "
                f"expected {prediction_length}, got {q_pred.shape[-1]}"
            )
        return q_pred

    print("    Organizing data into arrays...")
    for idx, (fc, gt, ctx) in enumerate(zip(forecasts, ground_truths, contexts)):
        if not hasattr(fc, "quantile"):
            raise ValueError(f"Unknown forecast type: {type(fc)}")

        q_preds = []
        for q in quantile_levels_list:
            q_pred = fc.quantile(q)
            q_pred = _normalize_quantile_pred(q_pred)
            q_preds.append(q_pred)
        q_preds = np.stack(q_preds, axis=0)  # (num_quantiles, num_variates, prediction_length)

        # Normalize ground truth
        if gt.ndim == 1:
            gt = gt[np.newaxis, :]
        elif gt.ndim == 2 and gt.shape[0] == prediction_length and gt.shape[1] != prediction_length:
            gt = gt.T

        # Normalize context
        if ctx.ndim == 1:
            ctx = ctx[np.newaxis, :]

        if is_to_univariate:
            # In to_univariate case:
            # idx iterates as: series_0_dim0_win0, series_0_dim0_win1, ..., series_0_dim1_win0, ...
            # We need to map flat_series_idx to (series_idx, variate_idx)
            flat_series_idx = idx // num_windows
            window_idx = idx % num_windows
            series_idx = flat_series_idx // num_variates
            variate_idx = flat_series_idx % num_variates

            # q_preds shape: (num_quantiles, 1, prediction_length)
            predictions_quantiles[series_idx, window_idx, :, variate_idx, :] = q_preds[:, 0, :]
            ground_truth[series_idx, window_idx, variate_idx, :] = gt[0, :]

            # Store context
            ctx_len_actual = ctx.shape[-1]
            context_array[series_idx, window_idx, variate_idx, :ctx_len_actual] = ctx[0, :]
        else:
            # Normal multivariate case
            series_idx = idx // num_windows
            window_idx = idx % num_windows

            predictions_quantiles[series_idx, window_idx] = q_preds
            ground_truth[series_idx, window_idx] = gt

            # Handle context shape mismatch
            if ctx.shape[0] != num_variates and ctx.shape[-1] == num_variates:
                ctx = ctx.T
            ctx_len_actual = ctx.shape[-1]
            context_array[series_idx, window_idx, :, :ctx_len_actual] = ctx

    # Save quantiles to npz file (instead of full samples)
    # Use float16 to reduce storage (sufficient for visualization purposes)
    npz_path = os.path.join(ds_output_dir, "predictions.npz")
    np.savez_compressed(
        npz_path,
        predictions_quantiles=predictions_quantiles.astype(np.float16),
        quantile_levels=quantile_levels_array.astype(np.float16),
    )
    print(f"    Saved predictions (quantiles) to {npz_path}")

    # Compute per-window metrics
    print("    Computing per-window metrics...")
    metrics = compute_per_window_metrics_from_quantiles(
        predictions_quantiles=predictions_quantiles,
        ground_truth=ground_truth,
        context=context_array,
        seasonality=seasonality,
        quantile_levels=quantile_levels_list,
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
        "num_quantiles": num_quantiles,
        "quantile_levels": quantile_levels_list,
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
