"""
VisionTS model experiments for time series forecasting.

Usage:
    python experiments/visionts_model.py
    python experiments/visionts_model.py --model-size mae_base
    python experiments/visionts_model.py --dataset "TSBench_IMOS_v2/15T" --terms short medium long
    python experiments/visionts_model.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/visionts_model.py --dataset all_datasets  # Run all datasets from config
    python experiments/visionts_model.py --val  # Evaluate on validation data (no saving)
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
from dotenv import load_dotenv

# VisionTS Imports
from visionts import VisionTS, freq_to_seasonality_list

# TimeBench Imports
from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)

# Load environment variables
load_dotenv()


# --- Helper Classes for VisionTS Integration ---
class UnivariateForecast:
    """Wraps VisionTS point forecasts into samples for TimeBench compatibility."""
    def __init__(self, point_forecast, target_num_samples=100):
        # Handle NaNs safety net
        if np.isnan(point_forecast).any():
            point_forecast = np.nan_to_num(point_forecast, nan=0.0)
            
        self._mean = point_forecast
        # Broadcast mean to create samples: (Num_Samples, Prediction_Length)
        self._samples = np.repeat(point_forecast[np.newaxis, :], target_num_samples, axis=0)

    @property
    def samples(self): return self._samples
    @property
    def mean(self): return self._mean
    def cpu(self): return self

class MockPredictor:
    """Mock predictor to pass pre-computed forecasts to save_window_predictions."""
    def __init__(self, forecasts): 
        self.forecasts = forecasts
    def predict(self, dataset_input, **kwargs): 
        return self.forecasts

def visionts_inference(model, batch_input, prediction_length, periodicity):
    """Core inference wrapper for VisionTS."""
    seq_len = batch_input.shape[1]
    
    # Update config for current batch properties
    model.update_config(
        context_len=seq_len,
        pred_len=prediction_length,
        periodicity=periodicity,
        align_const=1,
        norm_const=0.4
    )
    
    output = model(batch_input, export_image=False)
    return output[0] if isinstance(output, tuple) else output


def run_visionts_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] = None,
    model_size: str = "mae_base",
    output_dir: str | None = None,
    batch_size: int = 16, # Unused in loop but kept for signature alignment
    num_samples: int = 100,
    context_length: int = 4000,
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
):
    """
    Run VisionTS model experiments on a dataset with specified terms.
    """
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset configuration
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = ["short", "medium", "long"]

    if output_dir is None:
        output_dir = f"./output/results/visionts_{model_size}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Terms: {terms}")
    print(f"Evaluation on: {'Validation data (no saving)' if use_val else 'Test data'}")
    print(f"{'='*60}")

    # --- Load Model (Once per dataset wrapper) ---
    print(f"Loading VisionTS model ({model_size}) on {device}...")
    try:
        # Assuming ckpt_dir is relative to running script or fixed
        model = VisionTS(model_size, ckpt_dir='./ckpt/').to(device)
        model.eval()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model {model_size}: {e}")
        return

    for term in terms:
        print(f"\n--- Term: {term} ---")

        # Get settings from config
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        # Load dataset with config settings
        # Note: Using to_univariate=True for VisionTS standard processing
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=True, 
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )

        # Calculate actual info for logging
        if use_val:
            data_length = val_length
            num_windows = dataset.val_windows
            split_name = "Val split"
            eval_data = dataset.val_data
        else:
            data_length = test_length
            num_windows = dataset.windows
            split_name = "Test split"
            eval_data = dataset.test_data

        periodicity = freq_to_seasonality_list(dataset.freq)[0]

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Seasonality (Periodicity): {periodicity}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        # When to_univariate=True, target_dim is effectively 1 per processed item
        print(f"    - Target dim: {dataset.target_dim} (Processing as Univariate)")
        print(f"    - Series length: min={dataset._min_series_length}, max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
        print(f"    - {split_name}: {data_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        # Generate predictions
        data_type = "validation" if use_val else "test"
        print(f"  Running predictions on {data_type} data...")
        
        flat_forecasts = []
        
        # --- Inference Loop ---
        # VisionTS processes items sequentially or can be batched manually. 
        # Here we use sequential for safety/simplicity to match the previous logic, 
        # but with aligned logging.
        total_items = len(eval_data.input)
        
        for i, d in enumerate(eval_data.input):
            target = np.asarray(d["target"])
            
            # Handle NaNs in input
            if np.isnan(target).any():
                target = np.nan_to_num(target, nan=0.0)

            # Prepare Input Tensor: (1, Time, 1)
            if target.ndim == 1:
                target = target[..., np.newaxis]
            
            # Truncate context if too long
            if context_length and target.shape[0] > context_length:
                target = target[-context_length:]

            input_tensor = torch.tensor(target).float().unsqueeze(0).to(device)

            with torch.no_grad():
                try:
                    pred = visionts_inference(model, input_tensor, prediction_length, periodicity)
                    pred_np = pred.cpu().numpy()[0, :, 0] # Extract (Pred_Len,)
                except Exception as e:
                    print(f"    Error in inference at index {i}: {e}")
                    pred_np = np.zeros(prediction_length)
            
            flat_forecasts.append(pred_np)
            
            if (i + 1) % 100 == 0 or (i + 1) == total_items:
                 print(f"    Processed {i + 1}/{total_items} series...", end='\r')
        
        print("") # Newline after progress

        # Wrap results
        forecasts = [
            UnivariateForecast(f, target_num_samples=num_samples) 
            for f in flat_forecasts
        ]

        # Calculate statistics for logging
        num_total_instances = len(forecasts)
        # Because to_univariate=True, num_series in eval structure maps directly 
        # but logically it is:
        num_series_log = num_total_instances // num_windows if num_windows > 0 else 0
        
        print(f"    Total instances: {num_total_instances}, Series: {num_series_log}, Windows: {num_windows}")

        if use_val:
            print("    (No results saved - validation data used for hyperparameter selection)")
        else:
            # Save predictions and metrics for test data
            ds_config = f"{dataset_name}/{term}"

            model_hyperparams = {
                "model": f"visionts-{model_size}",
                "context_length": context_length,
                "periodicity": periodicity,
                "align_const": 1,
                "norm_const": 0.4
            }

            # Use MockPredictor to interface with timebench
            mock_predictor = MockPredictor(forecasts)

            metadata = save_window_predictions(
                dataset=dataset,
                predictor=mock_predictor,
                ds_config=ds_config,
                output_base_dir=output_dir,
                seasonality=periodicity,
                model_hyperparams=model_hyperparams,
            )
            print(f"  Completed: {metadata.get('num_series', 'N/A')} series × {metadata.get('num_windows', 'N/A')} windows")
            print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run VisionTS experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["TSBench_IMOS_v2/15T"],
                        help="Dataset name(s). Can be a single dataset, multiple datasets, or 'all_datasets'")
    parser.add_argument("--terms", type=str, nargs="+", default=["short", "medium", "long"],
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate")
    parser.add_argument("--model-size", type=str, default="mae_base",
                        help="VisionTS model size (e.g., mae_base)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (placeholder for compatibility)")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples for probabilistic forecasting simulation")
    parser.add_argument("--context-length", type=int, default=4000,
                        help="Maximum context length")
    parser.add_argument("--cuda-device", type=str, default="0",
                        help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--val", action="store_true",
                        help="Evaluate on validation data (no saving)")

    args = parser.parse_args()

    # Handle dataset list or 'all_datasets'
    config_path = Path(args.config) if args.config else None

    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        # Load all datasets from config
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
        print(f"Running all {len(datasets)} datasets from config:")
        for ds in datasets:
            print(f"  - {ds}")
    else:
        datasets = args.dataset

    # Iterate over all datasets
    total_datasets = len(datasets)
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        try:
            run_visionts_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                model_size=args.model_size,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                context_length=args.context_length,
                cuda_device=args.cuda_device,
                config_path=config_path,
                use_val=args.val,
            )
        except Exception as e:
            print(f"ERROR: Failed to run experiment for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'#'*60}")
    print(f"# All {total_datasets} dataset(s) completed!")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
