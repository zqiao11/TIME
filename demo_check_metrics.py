"""
Test if the modified metrics.py is aligned with GluonTS implementation.
"""

import numpy as np
import importlib

# Reload the metrics module to get the latest changes
import timebench.evaluation.metrics as metrics_module
importlib.reload(metrics_module)
from timebench.evaluation.metrics import compute_per_window_metrics
from gluonts.time_feature import get_seasonality

from gluonts.ev.metrics import (
    MSE,
    MAE,
    MASE,
    MAPE,
    SMAPE,
    MSIS,
    RMSE,
    NRMSE,
    ND,
    MeanWeightedSumQuantileLoss,
)

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# Create synthetic test data
# =============================================================================
pred_len = 24
num_samples = 100
seasonality = 7

# Create synthetic ground truth and context
ground_truth = np.array([10.0, 12.0, 11.0, 13.0, 14.0, 15.0, 16.0, 15.5, 14.5, 13.5,
                         12.5, 11.5, 10.5, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                         18.0, 17.0, 16.0, 15.0])

# Context (historical data)
context = np.array([8.0, 9.0, 10.0, 11.0, 10.0, 9.0, 8.0, 9.0, 10.0, 11.0,
                    12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0,
                    10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0])

# Create sample predictions (forecast samples)
predictions_samples = ground_truth[np.newaxis, :] + np.random.normal(0, 2, (num_samples, pred_len))
predictions_mean = np.mean(predictions_samples, axis=0)
predictions_median = np.median(predictions_samples, axis=0)

print("=" * 80)
print("TEST: Comparing Modified Timebench Metrics with GluonTS")
print("=" * 80)
print(f"\nTest Data:")
print(f"  Ground Truth Shape: {ground_truth.shape}")
print(f"  Predictions Samples Shape: {predictions_samples.shape}")
print(f"  Context Shape: {context.shape}")
print(f"  Seasonality: {seasonality}")

# =============================================================================
# Compute metrics using modified timebench implementation
# =============================================================================
print("\n" + "=" * 80)
print("[1] Computing metrics using MODIFIED TIMEBENCH implementation...")
print("=" * 80)

# Reshape data for timebench format: (num_series, num_windows, num_variates, pred_len)
tb_pred_mean = predictions_mean.reshape(1, 1, 1, pred_len)
tb_samples = predictions_samples.reshape(1, 1, num_samples, 1, pred_len)
tb_gt = ground_truth.reshape(1, 1, 1, pred_len)
tb_ctx = context.reshape(1, 1, 1, len(context))

quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

tb_metrics = compute_per_window_metrics(
    predictions_samples=tb_samples,
    ground_truth=tb_gt,
    context=tb_ctx,
    seasonality=seasonality,
    quantile_levels=quantile_levels,
)

print("\nTimebench Results:")
for k, v in tb_metrics.items():
    print(f"  {k}: {v[0, 0, 0]:.10f}")

# =============================================================================
# Compute metrics using GluonTS formulas (manually)
# =============================================================================
print("\n" + "=" * 80)
print("[2] Computing metrics using GLUONTS formulas (manual calculation)...")
print("=" * 80)

# Calculate seasonal error for MASE
seasonal_errors = np.abs(context[seasonality:] - context[:-seasonality])
seasonal_error = np.mean(seasonal_errors)
print(f"\nSeasonal Error (scale for MASE): {seasonal_error:.6f}")

# Compute median forecast (used for all metrics in GluonTS 0.5 mode)
median_pred = np.median(predictions_samples, axis=0)
error = ground_truth - median_pred
abs_error = np.abs(error)
abs_label_sum = np.sum(np.abs(ground_truth))

# MSE[0.5] - using median
gluonts_mse = np.mean(error ** 2)

# MAE[0.5] - using median
gluonts_mae = np.mean(abs_error)

# RMSE[0.5] - sqrt of MSE
gluonts_rmse = np.sqrt(gluonts_mse)

# MAPE[0.5] - using median, returns fraction
with np.errstate(divide='ignore', invalid='ignore'):
    mape_vals = abs_error / np.abs(ground_truth)
    mape_vals = np.where(np.isfinite(mape_vals), mape_vals, np.nan)
    gluonts_mape = np.nanmean(mape_vals)

# sMAPE[0.5] - using median, range [0, 2]
with np.errstate(divide='ignore', invalid='ignore'):
    smape_vals = 2 * abs_error / (np.abs(ground_truth) + np.abs(median_pred))
    smape_vals = np.where(np.isfinite(smape_vals), smape_vals, np.nan)
    gluonts_smape = np.nanmean(smape_vals)

# MASE[0.5] - using median
gluonts_mase = gluonts_mae / seasonal_error

# ND[0.5] - using median
gluonts_nd = np.sum(abs_error) / abs_label_sum

# CRPS (MeanWeightedSumQuantileLoss)
# GluonTS implementation:
# 1. For each quantile q: quantile_loss = 2 * |error * ((pred >= label) - q)|
# 2. weighted_sum_quantile_loss[q] = sum(quantile_loss) / sum(|label|)
# 3. CRPS = mean(weighted_sum_quantile_loss across all quantiles)
weighted_quantile_losses = []
for q in quantile_levels:
    q_pred = np.quantile(predictions_samples, q, axis=0)
    q_error = ground_truth - q_pred
    indicator = (q_pred >= ground_truth).astype(float)
    q_loss = 2 * np.abs(q_error * (indicator - q))
    weighted_ql = np.sum(q_loss) / abs_label_sum
    weighted_quantile_losses.append(weighted_ql)
gluonts_crps = np.mean(weighted_quantile_losses)

print("\nGluonTS Results (manual calculation):")
print(f"  MSE[0.5]:  {gluonts_mse:.10f}")
print(f"  MAE[0.5]:  {gluonts_mae:.10f}")
print(f"  RMSE[0.5]: {gluonts_rmse:.10f}")
print(f"  MAPE[0.5]: {gluonts_mape:.10f}")
print(f"  sMAPE[0.5]: {gluonts_smape:.10f}")
print(f"  MASE[0.5]: {gluonts_mase:.10f}")
print(f"  ND[0.5]:   {gluonts_nd:.10f}")
print(f"  CRPS:      {gluonts_crps:.10f}")

# =============================================================================
# Compare results
# =============================================================================
print("\n" + "=" * 80)
print("[3] COMPARISON: Timebench vs GluonTS")
print("=" * 80)

def check_match(name, tb_val, gl_val, rtol=1e-9):
    diff = abs(tb_val - gl_val)
    match = np.isclose(tb_val, gl_val, rtol=rtol)
    status = "✓ MATCH" if match else "✗ MISMATCH"
    print(f"\n{name}:")
    print(f"  Timebench: {tb_val:.10f}")
    print(f"  GluonTS:   {gl_val:.10f}")
    print(f"  Diff:      {diff:.2e}")
    print(f"  Status:    {status}")
    return match

results = []
results.append(check_match("MSE", tb_metrics["MSE"][0, 0, 0], gluonts_mse))
results.append(check_match("MAE", tb_metrics["MAE"][0, 0, 0], gluonts_mae))
results.append(check_match("RMSE", tb_metrics["RMSE"][0, 0, 0], gluonts_rmse))
results.append(check_match("MAPE", tb_metrics["MAPE"][0, 0, 0], gluonts_mape))
results.append(check_match("sMAPE", tb_metrics["sMAPE"][0, 0, 0], gluonts_smape))
results.append(check_match("MASE (IMPORTANT)", tb_metrics["MASE"][0, 0, 0], gluonts_mase))
results.append(check_match("ND", tb_metrics["ND"][0, 0, 0], gluonts_nd))
results.append(check_match("CRPS (IMPORTANT)", tb_metrics["CRPS"][0, 0, 0], gluonts_crps))

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

all_match = all(results)
if all_match:
    print("\n✅ ALL METRICS MATCH! Timebench is now aligned with GluonTS.")
else:
    print("\n❌ SOME METRICS DO NOT MATCH. Please check the implementation.")

print(f"\nTotal: {sum(results)}/{len(results)} metrics match")
