"""
Per-window metrics computation for time series forecasting evaluation.
Aligned with GluonTS implementation.

Supported metrics:
- MSE: Mean Squared Error (using median forecast, aligned with GluonTS MSE[0.5])
- MAE: Mean Absolute Error (using median forecast)
- RMSE: Root Mean Squared Error (using median forecast)
- MAPE: Mean Absolute Percentage Error (using median forecast, returns fraction)
- sMAPE: Symmetric Mean Absolute Percentage Error (using median forecast, range [0, 2])
- MASE: Mean Absolute Scaled Error (using median forecast)
- ND: Normalized Deviation (using median forecast)
- CRPS: Continuous Ranked Probability Score (MeanWeightedSumQuantileLoss)

================================================================================
聚合逻辑概述 (Aggregation Logic Summary):
================================================================================

本文件实现 **第一层聚合: pred_len → window 级别**

输入形状:
    predictions_samples: (num_series, num_windows, num_samples, num_variates, pred_len)
    ground_truth:        (num_series, num_windows, num_variates, pred_len)

输出形状:
    每个指标数组:        (num_series, num_windows, num_variates)

聚合方式 (对 pred_len 维度):
- 单个 timestep 级别的平均值
    - MSE:   np.mean(error ** 2)                    # 均值
    - MAE:   np.mean(abs_error)                     # 均值
    - RMSE:  sqrt(MSE)
    - MAPE:  np.nanmean(abs_error / abs_gt)         # 均值，忽略 NaN
    - sMAPE: np.nanmean(2*|e|/(|gt|+|pred|))        # 均值，忽略 NaN
    - MASE:  MAE / seasonal_error                   # MAE 除以季节性误差
- 整个窗口级别的归一化值；尺度不受windows长度影响
    - ND:    sum(abs_error) / sum(abs_gt)           # 和比值
    - CRPS:  mean(sum(q_loss) / sum(abs_gt))        # 分位点损失的加权平均

注意: 代码使用 valid_mask 过滤 ground_truth 中的 NaN 值（防御性编程）
      正常情况下 ground_truth 不应有 NaN padding

--------------------------------------------------------------------------------
第二层聚合: window 级别 → 数据集级别 (在 saver.py 中实现)
--------------------------------------------------------------------------------
    使用 np.nanmean(metric_values) 将 3D 数组平均成标量:
    所有 series × 所有 windows × 所有 variates → 简单平均，忽略 NaN
================================================================================
"""

import numpy as np


# Default quantile levels for CRPS computation (aligned with GluonTS)
DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def compute_per_window_metrics(
    predictions_samples: np.ndarray,
    ground_truth: np.ndarray,
    context: np.ndarray,
    seasonality: int = 1,
    quantile_levels: list[float] = None,
) -> dict[str, np.ndarray]:
    """
    Compute evaluation metrics for each prediction window.
    All metrics are aligned with GluonTS implementation.

    Args:
        predictions_samples: Sampled predictions with shape
            (num_series, num_windows, num_samples, num_variates, pred_len)
        ground_truth: Ground truth values with shape
            (num_series, num_windows, num_variates, pred_len)
        context: Historical context with shape
            (num_series, num_windows, num_variates, max_ctx_len)
            Note: Shorter contexts are NaN-padded
        seasonality: Seasonal period length for MASE computation
        quantile_levels: Quantile levels for CRPS computation.
            Defaults to [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    Returns:
        Dictionary of metric arrays, each with shape (num_series, num_windows, num_variates)
        Keys: MSE, MAE, RMSE, MAPE, sMAPE, MASE, ND, CRPS
    """
    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILE_LEVELS

    num_series, num_windows, num_samples, num_variates, pred_len = predictions_samples.shape

    # Initialize metric arrays: (num_series, num_windows, num_variates)
    mse = np.zeros((num_series, num_windows, num_variates))
    mae = np.zeros((num_series, num_windows, num_variates))
    rmse = np.zeros((num_series, num_windows, num_variates))
    mape = np.zeros((num_series, num_windows, num_variates))
    smape = np.zeros((num_series, num_windows, num_variates))
    mase = np.zeros((num_series, num_windows, num_variates))
    nd = np.zeros((num_series, num_windows, num_variates))

    # CRPS (Continuous Ranked Probability Score) - using weighted quantile loss
    crps = np.zeros((num_series, num_windows, num_variates))

    for s in range(num_series):
        for w in range(num_windows):
            for v in range(num_variates):
                samples = predictions_samples[s, w, :, v]  # (num_samples, pred_len)
                gt = ground_truth[s, w, v]  # (pred_len,)
                ctx = context[s, w, v]  # (max_ctx_len,)
                ctx = ctx[~np.isnan(ctx)]  # Remove NaN padding

                # Compute median (0.5 quantile) forecast - used for most metrics
                median_pred = np.median(samples, axis=0)  # (pred_len,)

                # Create valid mask to handle potential NaN padding in ground truth
                # (defensive programming - normally gt should not have NaN)
                valid_mask = ~np.isnan(gt)
                if not np.any(valid_mask):
                    # No valid timesteps - set all metrics to NaN
                    mse[s, w, v] = mae[s, w, v] = rmse[s, w, v] = np.nan
                    mape[s, w, v] = smape[s, w, v] = mase[s, w, v] = np.nan
                    nd[s, w, v] = crps[s, w, v] = np.nan
                    continue

                # Filter to valid timesteps only
                gt = gt[valid_mask]
                median_pred = median_pred[valid_mask]
                samples = samples[:, valid_mask]  # (num_samples, valid_len)

                # Compute error using median forecast
                error = gt - median_pred
                abs_error = np.abs(error)

                # MSE (using median forecast, aligned with GluonTS MSE[0.5])
                # GluonTS: squared_error = (label - forecast)^2
                mse[s, w, v] = np.mean(error ** 2)

                # MAE (using median forecast)
                # GluonTS: absolute_error = |label - forecast|
                mae[s, w, v] = np.mean(abs_error)

                # RMSE (sqrt of MSE)
                rmse[s, w, v] = np.sqrt(mse[s, w, v])

                # MAPE (using median forecast, returns fraction not percentage)
                # GluonTS: absolute_percentage_error = |error| / |label|
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape_vals = abs_error / np.abs(gt)
                    mape_vals = np.where(np.isfinite(mape_vals), mape_vals, np.nan)
                    mape[s, w, v] = np.nanmean(mape_vals)

                # sMAPE (using median forecast, range [0, 2])
                # GluonTS: 2 * |error| / (|label| + |forecast|)
                with np.errstate(divide='ignore', invalid='ignore'):
                    smape_vals = 2 * abs_error / (np.abs(gt) + np.abs(median_pred))
                    smape_vals = np.where(np.isfinite(smape_vals), smape_vals, np.nan)
                    smape[s, w, v] = np.nanmean(smape_vals)

                # MASE (Mean Absolute Scaled Error, using median forecast)
                # GluonTS: |error| / seasonal_error
                if len(ctx) > seasonality:
                    naive_errors = np.abs(ctx[seasonality:] - ctx[:-seasonality])
                    seasonal_error = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
                    if seasonal_error > 0:
                        mase[s, w, v] = mae[s, w, v] / seasonal_error
                    else:
                        mase[s, w, v] = np.nan
                else:
                    mase[s, w, v] = np.nan

                # ND (Normalized Deviation, using median forecast)
                # GluonTS: sum(|error|) / sum(|label|)
                abs_label_sum = np.sum(np.abs(gt))
                if abs_label_sum > 0:
                    nd[s, w, v] = np.sum(abs_error) / abs_label_sum
                else:
                    nd[s, w, v] = np.nan

                # CRPS (MeanWeightedSumQuantileLoss)
                # GluonTS implementation:
                # 1. For each quantile q: quantile_loss = 2 * |error * ((pred >= label) - q)|
                # 2. weighted_sum_quantile_loss[q] = sum(quantile_loss) / sum(|label|)
                # 3. CRPS = mean(weighted_sum_quantile_loss across all quantiles)
                if abs_label_sum > 0:
                    weighted_quantile_losses = []
                    for q in quantile_levels:
                        q_pred = np.quantile(samples, q, axis=0)  # (valid_len,)
                        q_error = gt - q_pred
                        # GluonTS quantile_loss formula: 2 * |error * ((pred >= label) - q)|
                        indicator = (q_pred >= gt).astype(float)
                        q_loss = 2 * np.abs(q_error * (indicator - q))
                        # Weighted by sum of absolute labels
                        weighted_ql = np.sum(q_loss) / abs_label_sum
                        weighted_quantile_losses.append(weighted_ql)
                    crps[s, w, v] = np.mean(weighted_quantile_losses)
                else:
                    crps[s, w, v] = np.nan

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "MASE": mase,
        "ND": nd,
        "CRPS": crps,
    }
