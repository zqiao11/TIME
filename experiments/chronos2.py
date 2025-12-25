"""
Chronos model experiments for time series forecasting.
(Mirrors the structure of experiments/moirai.py)

Usage:
    python experiments/chronos2.py
    python experiments/chronos2.py --model-size chronos2
    python experiments/chronos2.py --dataset "TSBench_IMOS_v2/15T" --terms short medium long
    python experiments/chronos2.py --val  # Evaluate on validation data (no saving)
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality
from chronos import BaseChronosPipeline

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.metrics import compute_per_window_metrics

load_dotenv()

# --- Helper Class: Adapts Chronos-2 output for Timebench ---
class MultivariateForecast:
    def __init__(self, quantiles_tensor, mean_tensor, target_num_samples=100):
        # 1. Handle Mean
        self._mean = mean_tensor.cpu().float().numpy() if isinstance(mean_tensor, torch.Tensor) else mean_tensor

        # 2. Handle Quantiles -> Samples (via interpolation)
        q_data = quantiles_tensor.cpu().float().numpy() if isinstance(quantiles_tensor, torch.Tensor) else quantiles_tensor
        
        # q_data expected shape: (Num_Quantiles, Variates, Time)
        num_quantiles, num_vars, pred_len = q_data.shape

        if num_quantiles != target_num_samples:
            target_indices = np.linspace(0, num_quantiles - 1, target_num_samples)
            source_indices = np.arange(num_quantiles)
            
            # Initialize result array (Samples, V, T)
            resampled = np.zeros((target_num_samples, num_vars, pred_len))
            
            for v in range(num_vars):
                for t in range(pred_len):
                    # Interpolate along the quantile dimension (dim 0)
                    resampled[:, v, t] = np.interp(target_indices, source_indices, q_data[:, v, t])
            self._samples = resampled
        else:
            self._samples = q_data

    @property
    def samples(self): return self._samples
    @property
    def mean(self): return self._mean

class MockPredictor:
    """Mock predictor to bypass internal inference in save_window_predictions"""
    def __init__(self, forecasts): self.forecasts = forecasts
    def predict(self, *args, **kwargs): return self.forecasts


def run_chronos_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] = None,
    model_size: str = "chronos2",
    output_dir: str | None = None,
    batch_size: int = 32,
    num_samples: int = 100,
    context_length: int = 2048,
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
):
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device_map = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset configuration
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = ["short", "medium", "long"]

    if output_dir is None:
        output_dir = f"./output/results/chronos2_{model_size}"

    os.makedirs(output_dir, exist_ok=True)

    # Model mapping (Preserving your specific setup)
    model_map = {
        "chronos2": "amazon/chronos-2", 
    }
    hf_model_path = model_map.get(model_size, "amazon/chronos-2")

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {hf_model_path}")
    print(f"Terms: {terms}")
    print(f"Evaluation on: {'Validation data (no saving)' if use_val else 'Test data'}")
    print(f"{'='*60}")

    for term in terms:
        print(f"\n--- Term: {term} ---")

        # Get settings from config
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_split = settings.get("test_split")
        val_split = settings.get("val_split")

        print(f"  Config: prediction_length={prediction_length}, test_split={test_split}, val_split={val_split}")

        # Initialize model
        print(f"  Initializing Chronos pipeline ({hf_model_path})...")
        pipeline = BaseChronosPipeline.from_pretrained(
            hf_model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )

        # Load dataset
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=False, # Chronos 2 supports multivariate natively
            prediction_length=prediction_length,
            test_split=test_split,
            val_split=val_split,
        )

        # Calculate actual lengths
        if use_val:
            data_length = int(dataset.val_split * dataset._min_series_length)
            num_windows = dataset.val_windows
            split_name = "Val split"
            split_value = dataset.val_split
            eval_data = dataset.val_data
        else:
            data_length = int(dataset.test_split * dataset._min_series_length)
            num_windows = dataset.windows
            split_name = "Test split"
            split_value = dataset.test_split
            eval_data = dataset.test_data

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - Series length: min={dataset._min_series_length}, max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
        print(f"    - {split_name}: {split_value} ({data_length} steps)")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        season_length = get_seasonality(dataset.freq)

        # Generate predictions
        data_type = "validation" if use_val else "test"
        print(f"  Running predictions on {data_type} data (Batch size: {batch_size})...")

        # 1. Prepare Context List
        context_list = []
        for d in eval_data.input:
            target = np.asarray(d["target"])
            
            # --- [FIX START] Manually truncate context ---
            # 确保输入长度不超过 context_length，从末尾截取
            seq_len = target.shape[-1]
            if seq_len > context_length:
                target = target[..., -context_length:]
            # --- [FIX END] ---

            if target.ndim == 1: 
                target = target[np.newaxis, :]
                
            context_list.append(torch.tensor(target))

            
        # 2. Batch Inference
        forecasts = []
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        if batch_size > 0:
            total_items = len(context_list)
            for start in range(0, total_items, batch_size):
                end = min(start + batch_size, total_items)
                batch_contexts = context_list[start:end]
                
                with torch.no_grad():
                    batch_q, batch_m = pipeline.predict_quantiles(
                        inputs=batch_contexts,
                        prediction_length=prediction_length,
                        quantile_levels=quantile_levels
                    )
                
                # Handle return types (Tensor vs List)
                if isinstance(batch_q, torch.Tensor):
                    # Check shape: (Batch, Variates, Horizon, Quantiles)
                    # If last dim is quantiles, permute it
                    if batch_q.ndim == 4 and batch_q.shape[-1] == len(quantile_levels):
                        # Permute to (Batch, Quantiles, Variates, Horizon)
                        batch_q = batch_q.permute(0, 3, 1, 2)
                    
                    batch_q = batch_q.cpu()
                    batch_m = batch_m.cpu()
                    
                    for i in range(batch_q.shape[0]):
                        forecasts.append(MultivariateForecast(
                            batch_q[i], batch_m[i], target_num_samples=num_samples
                        ))
                
                elif isinstance(batch_q, list):
                    # List of Tensors
                    for q, m in zip(batch_q, batch_m):
                        if q.ndim == 3 and q.shape[-1] == len(quantile_levels):
                            # (Variates, Horizon, Quantiles) -> (Quantiles, Variates, Horizon)
                            q = q.permute(2, 0, 1)
                        
                        forecasts.append(MultivariateForecast(
                            q, m, target_num_samples=num_samples
                        ))

        # Count series
        num_total_instances = len(forecasts)
        num_series = num_total_instances // num_windows
        num_variates = dataset.target_dim

        print(f"    Total instances: {num_total_instances}, Series: {num_series}, Windows: {num_windows}")

        # --- LOGIC BRANCH: Validation (Manual) vs Test (Save) ---
        
        if use_val:
            # === Validation Mode: Compute and Print Manually (No Saving) ===
            print("    [Validation] Organizing data and computing metrics manually...")
            
            # Collect Ground Truth
            ground_truths = []
            contexts = []
            for inp, label in eval_data:
                ground_truths.append(label["target"])
                contexts.append(inp["target"])

            # Initialize Arrays
            predictions_mean = np.zeros((num_series, num_windows, num_variates, prediction_length))
            predictions_samples = np.zeros((num_series, num_windows, num_samples, num_variates, prediction_length))
            ground_truth = np.zeros((num_series, num_windows, num_variates, prediction_length))
            
            max_context_len = max(ctx.shape[-1] for ctx in contexts) if contexts else 0
            context_array = np.full((num_series, num_windows, num_variates, max_context_len), np.nan)

            # Fill Arrays
            for idx, (fc, gt, ctx) in enumerate(zip(forecasts, ground_truths, contexts)):
                series_idx = idx // num_windows
                window_idx = idx % num_windows

                fc_mean = fc.mean
                fc_samples = fc.samples

                if fc_mean.ndim == 1: fc_mean = fc_mean[np.newaxis, :]
                if fc_samples.ndim == 2: fc_samples = fc_samples[:, np.newaxis, :]
                if gt.ndim == 1: gt = gt[np.newaxis, :]
                if gt.shape[0] == prediction_length and gt.shape[1] == num_variates: gt = gt.T
                if ctx.ndim == 1: ctx = ctx[np.newaxis, :]
                elif ctx.shape[0] != num_variates: ctx = ctx.T

                predictions_mean[series_idx, window_idx] = fc_mean
                predictions_samples[series_idx, window_idx] = fc_samples
                ground_truth[series_idx, window_idx] = gt
                
                ctx_len = ctx.shape[-1]
                context_array[series_idx, window_idx, :, :ctx_len] = ctx

            # Compute and Print
            metrics = compute_per_window_metrics(
                predictions_mean=predictions_mean,
                predictions_samples=predictions_samples,
                ground_truth=ground_truth,
                context=context_array,
                seasonality=season_length,
            )
            print("    Metrics summary (averaged over all series/windows/variates):")
            for metric_name, metric_values in metrics.items():
                mean_val = np.nanmean(metric_values)
                print(f"      {metric_name}: {mean_val:.4f}")
            print("    (No results saved - validation mode)")

        else:
            # === Test Mode: Delegate to save_window_predictions ===
            # This function handles array organization, metric computation, printing, AND saving.
            # We don't need to do it manually here.
            
            ds_config = f"{dataset_name}/{term}"
            model_hyperparams = {
                "model": f"chronos-{model_size}",
                "context_length": context_length,
            }

            mock_predictor = MockPredictor(forecasts)

            metadata = save_window_predictions(
                dataset=dataset,
                predictor=mock_predictor,
                ds_config=ds_config,
                output_base_dir=output_dir,
                seasonality=season_length,
                model_hyperparams=model_hyperparams,
            )
            print(f"  Completed: {metadata['num_series']} series × {metadata['num_windows']} windows")
            print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Chronos experiments")
    parser.add_argument("--dataset", type=str, default="IMOS/15T", help="Dataset name")
    parser.add_argument("--terms", type=str, nargs="+", default=["short", "medium", "long"], choices=["short", "medium", "long"], help="Terms to evaluate")
    parser.add_argument("--model-size", type=str, default="chronos2", help="Chronos model size (use 'chronos2' for amazon/chronos-2)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for prediction")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples for probabilistic forecasting")
    parser.add_argument("--context-length", type=int, default=4000, help="Maximum context length")
    parser.add_argument("--cuda-device", type=str, default="0", help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None, help="Path to datasets.yaml config file")
    parser.add_argument("--val", action="store_true", help="Evaluate on validation data")

    args = parser.parse_args()

    run_chronos_experiment(
        dataset_name=args.dataset,
        terms=args.terms,
        model_size=args.model_size,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        context_length=args.context_length,
        cuda_device=args.cuda_device,
        config_path=Path(args.config) if args.config else None,
        use_val=args.val,
    )

if __name__ == "__main__":
    main()