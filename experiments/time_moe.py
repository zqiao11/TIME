"""
TimeMoE model experiments for time series forecasting.
(Aligned with TimeBench Moirai/VisionTS templates)

Usage:
    python experiments/timemoe.py
    python experiments/timemoe.py --model-size base
    python experiments/timemoe.py --dataset "TSBench_IMOS_v2/15T" --terms short medium long
    python experiments/timemoe.py --dataset all_datasets
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)

load_dotenv()

# --- Helper Class: Univariate Forecast ---
class UnivariateForecast:
    """
    Wraps TimeMoE point forecasts into sample format.
    Correctly handles shape: (Num_Samples, Prediction_Length)
    """
    def __init__(self, point_forecast, target_num_samples=100):
        # point_forecast shape: (Prediction_Length,)
        
        # 1. Convert Tensor to Numpy if needed
        if isinstance(point_forecast, torch.Tensor):
            point_forecast = point_forecast.cpu().float().numpy()

        # 2. Handle potential NaNs (Safety net)
        if np.isnan(point_forecast).any():
            point_forecast = np.nan_to_num(point_forecast, nan=0.0)

        # 3. Construct Samples
        # Shape: (Num_Samples, Prediction_Length)
        self._mean = point_forecast
        self._samples = np.repeat(point_forecast[np.newaxis, :], target_num_samples, axis=0)

    @property
    def samples(self): return self._samples
    @property
    def mean(self): return self._mean
    def cpu(self): return self

# --- Helper Class: Mock Predictor ---
class MockPredictor:
    def __init__(self, forecasts): 
        self.forecasts = forecasts
    def predict(self, dataset_input, **kwargs): 
        return self.forecasts

# --- Helper Function: TimeMoE Inference ---
def timemoe_inference(model, batch_input, prediction_length):
    """
    Executes TimeMoE Normalize -> Generate -> Denormalize flow
    batch_input: [Batch, Context_Length]
    """
    # 1. Normalize (Reversible Instance Normalization)
    mean = batch_input.mean(dim=-1, keepdim=True)
    std = batch_input.std(dim=-1, keepdim=True) + 1e-5 # Prevent division by zero
    normed_seqs = (batch_input - mean) / std

    # 2. Forecast (Autoregressive Generation)
    # TimeMoE is a causal LM, we generate `prediction_length` new tokens
    output = model.generate(
        normed_seqs, 
        max_new_tokens=prediction_length,
        pad_token_id=0,     
        eos_token_id=None,  
        do_sample=False,    # Greedy search (deterministic point forecast)
        use_cache=True      # Optimize generation speed
    ) 
    # output shape: [Batch, Context + Pred]

    # 3. Slice Output (Take only the new tokens)
    normed_predictions = output[:, -prediction_length:]

    # 4. Inverse Normalize
    predictions = normed_predictions * std + mean
    
    return predictions

def run_timemoe_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] = None,
    model_size: str = "base",
    output_dir: str | None = None,
    batch_size: int = 1, # Kept for signature compatibility, TimeMoE usually runs sequential here
    num_samples: int = 100,
    context_length: int = 4000, 
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
):
    """
    Run TimeMoE model experiments.
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
        output_dir = f"./output/results/timemoe_{model_size}"

    os.makedirs(output_dir, exist_ok=True)

    # Resolve Model Name
    hf_repo_map = {
        "base": "Maple728/TimeMoE-50M",
        "large": "Maple728/TimeMoE-200M", 
    }
    model_name = hf_repo_map.get(model_size, "Maple728/TimeMoE-50M")

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name} (Size: {model_size})")
    print(f"Terms: {terms}")
    print(f"Evaluation on: {'Validation data (no saving)' if use_val else 'Test data'}")
    print(f"{'='*60}")

    # --- Load Model ---
    print(f"Loading TimeMoE model ({model_name}) on {device}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model {model_name}: {e}")
        return

    for term in terms:
        print(f"\n--- Term: {term} ---")

        # Get settings from config
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        # Load dataset
        # Using to_univariate=True to simplify multivariate processing into flat streams
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=True, 
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )

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

        season_length = get_seasonality(dataset.freq)

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim} (Processing as Univariate)")
        print(f"    - Series length: min={dataset._min_series_length}, max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
        print(f"    - {split_name}: {data_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        # --- Inference Loop ---
        data_type = "validation" if use_val else "test"
        print(f"  Running predictions on {data_type} data...")
        
        flat_forecasts = []
        total_items = len(eval_data.input)

        for i, d in enumerate(eval_data.input):
            target = np.asarray(d["target"]) # Shape: (Time,) or (1, Time) due to to_univariate=True

            # Handle NaNs
            if np.isnan(target).any():
                target = np.nan_to_num(target, nan=0.0)
            
            # Ensure shape is (Time,) for processing
            if target.ndim > 1:
                target = target.squeeze()

            # Truncate context if too long
            if context_length and len(target) > context_length:
                target = target[-context_length:]

            # To Tensor: (1, Time) -> Batch dim required for generate
            input_tensor = torch.tensor(target).float().unsqueeze(0).to(device)

            with torch.no_grad():
                try:
                    # Run Inference
                    pred_tensor = timemoe_inference(model, input_tensor, prediction_length)
                    # Extract result: (Prediction_Length,)
                    pred_np = pred_tensor[0].cpu().numpy()
                except Exception as e:
                    print(f"    Error in inference at index {i}: {e}")
                    pred_np = np.zeros(prediction_length)

            flat_forecasts.append(pred_np)

            if (i + 1) % 100 == 0 or (i + 1) == total_items:
                 print(f"    Processed {i + 1}/{total_items} series...", end='\r')

        print("") # Newline

        # Wrap results
        forecasts = [
            UnivariateForecast(f, target_num_samples=num_samples) 
            for f in flat_forecasts
        ]

        # Statistics
        num_total_instances = len(forecasts)
        num_series_log = num_total_instances // num_windows if num_windows > 0 else 0
        print(f"    Total instances: {num_total_instances}, Series: {num_series_log}, Windows: {num_windows}")

        if use_val:
             print("    (No results saved - validation data used for hyperparameter selection)")
        else:
            # Save
            ds_config = f"{dataset_name}/{term}"
            model_hyperparams = {
                "model": f"timemoe-{model_size}",
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
            print(f"  Completed: {metadata.get('num_series', 'N/A')} series × {metadata.get('num_windows', 'N/A')} windows")
            print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run TimeMoE experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["TSBench_IMOS_v2/15T"],
                        help="Dataset name(s). Can be a single dataset, multiple datasets, or 'all_datasets'")
    parser.add_argument("--terms", type=str, nargs="+", default=["short", "medium", "long"],
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate")
    parser.add_argument("--model-size", type=str, default="base",
                        choices=["base", "large"],
                        help="TimeMoE model size: base(50M) or large(200M)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (unused in sequential mode but kept for compat)")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples (duplicated for deterministic models)")
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
            run_timemoe_experiment(
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