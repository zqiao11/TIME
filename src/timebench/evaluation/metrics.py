"""
Per-window metrics computation for time series forecasting evaluation.

Supported metrics:
- MSE: Mean Squared Error
- MAE: Mean Absolute Error  
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error
- sMAPE: Symmetric Mean Absolute Percentage Error
- MASE: Mean Absolute Scaled Error
- ND: Normalized Deviation
- CRPS: Continuous Ranked Probability Score
- Quantile Losses: at 0.1, 0.5, 0.9 quantiles
"""

import numpy as np


def compute_per_window_metrics(
    predictions_mean: np.ndarray,
    predictions_samples: np.ndarray,
    ground_truth: np.ndarray,
    context: np.ndarray,
    seasonality: int = 1,
) -> dict[str, np.ndarray]:
    """
    Compute evaluation metrics for each prediction window.
    
    Args:
        predictions_mean: Mean predictions with shape 
            (num_series, num_windows, num_variates, pred_len)
        predictions_samples: Sampled predictions with shape
            (num_series, num_windows, num_samples, num_variates, pred_len)
        ground_truth: Ground truth values with shape
            (num_series, num_windows, num_variates, pred_len)
        context: Historical context with shape
            (num_series, num_windows, num_variates, max_ctx_len)
            Note: Shorter contexts are NaN-padded
        seasonality: Seasonal period length for MASE computation
        
    Returns:
        Dictionary of metric arrays, each with shape (num_series, num_windows, num_variates)
        Keys: MSE, MAE, RMSE, MAPE, sMAPE, MASE, ND, CRPS, 
              QuantileLoss_0.1, QuantileLoss_0.5, QuantileLoss_0.9
    """
    num_series, num_windows, num_variates, pred_len = predictions_mean.shape
    
    # Initialize metric arrays: (num_series, num_windows, num_variates)
    mse = np.zeros((num_series, num_windows, num_variates))
    mae = np.zeros((num_series, num_windows, num_variates))
    rmse = np.zeros((num_series, num_windows, num_variates))
    mape = np.zeros((num_series, num_windows, num_variates))
    smape = np.zeros((num_series, num_windows, num_variates))
    mase = np.zeros((num_series, num_windows, num_variates))
    nd = np.zeros((num_series, num_windows, num_variates))
    
    # Quantile losses for specific quantiles
    quantiles = [0.1, 0.5, 0.9]
    quantile_losses = {q: np.zeros((num_series, num_windows, num_variates)) for q in quantiles}
    
    # CRPS (Continuous Ranked Probability Score)
    crps = np.zeros((num_series, num_windows, num_variates))
    
    for s in range(num_series):
        for w in range(num_windows):
            for v in range(num_variates):
                pred = predictions_mean[s, w, v]  # (pred_len,)
                samples = predictions_samples[s, w, :, v]  # (num_samples, pred_len)
                gt = ground_truth[s, w, v]  # (pred_len,)
                ctx = context[s, w, v]  # (max_ctx_len,)
                ctx = ctx[~np.isnan(ctx)]  # Remove NaN padding
                
                # MSE
                mse[s, w, v] = np.mean((pred - gt) ** 2)
                
                # MAE
                mae[s, w, v] = np.mean(np.abs(pred - gt))
                
                # RMSE
                rmse[s, w, v] = np.sqrt(mse[s, w, v])
                
                # MAPE (handle zeros in ground truth)
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape_vals = np.abs((pred - gt) / gt) * 100
                    mape_vals = np.where(np.isfinite(mape_vals), mape_vals, np.nan)
                    mape[s, w, v] = np.nanmean(mape_vals)
                
                # sMAPE (symmetric MAPE)
                with np.errstate(divide='ignore', invalid='ignore'):
                    smape_vals = 200 * np.abs(pred - gt) / (np.abs(pred) + np.abs(gt))
                    smape_vals = np.where(np.isfinite(smape_vals), smape_vals, np.nan)
                    smape[s, w, v] = np.nanmean(smape_vals)
                
                # MASE (Mean Absolute Scaled Error)
                if len(ctx) > seasonality:
                    naive_errors = np.abs(ctx[seasonality:] - ctx[:-seasonality])
                    scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
                    if scale > 0:
                        mase[s, w, v] = mae[s, w, v] / scale
                    else:
                        mase[s, w, v] = np.nan
                else:
                    mase[s, w, v] = np.nan
                
                # ND (Normalized Deviation)
                gt_sum = np.sum(np.abs(gt))
                if gt_sum > 0:
                    nd[s, w, v] = np.sum(np.abs(pred - gt)) / gt_sum
                else:
                    nd[s, w, v] = np.nan
                
                # Quantile losses
                for q in quantiles:
                    q_pred = np.quantile(samples, q, axis=0)  # (pred_len,)
                    errors = gt - q_pred
                    quantile_losses[q][s, w, v] = np.mean(
                        np.where(errors >= 0, q * errors, (q - 1) * errors)
                    ) * 2  # multiply by 2 for standard quantile loss definition
                
                # CRPS (approximate using samples)
                # CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
                # where X, X' are independent samples from forecast distribution
                abs_errors = np.mean(np.abs(samples - gt), axis=0)  # (pred_len,)
                sample_spread = np.mean(
                    np.abs(samples[:, np.newaxis, :] - samples[np.newaxis, :, :]), 
                    axis=(0, 1)
                )
                crps[s, w, v] = np.mean(abs_errors - 0.5 * sample_spread)
    
    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "MASE": mase,
        "ND": nd,
        "QuantileLoss_0.1": quantile_losses[0.1],
        "QuantileLoss_0.5": quantile_losses[0.5],
        "QuantileLoss_0.9": quantile_losses[0.9],
        "CRPS": crps,
    }



