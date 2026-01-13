"""
Chronos model experiments for time series forecasting.
(Rewritten to match Moirai interface and logic exactly)

Usage:
    python experiments/chronos_t5.py
    python experiments/chronos_t5.py --model-size tiny
    python experiments/chronos_t5.py --dataset "SG_Weather/D" --terms short medium long
    python experiments/chronos_t5.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/chronos_t5.py --dataset all_datasets  # Run all datasets from config
    python experiments/chronos_t5.py --val  # Evaluate on validation data (no saving)
"""

import argparse
import os
import sys
from pathlib import Path
import traceback

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from dotenv import load_dotenv
from chronos import ChronosPipeline
from gluonts.time_feature import get_seasonality

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)

# Load environment variables
load_dotenv()

def _impute_nans_1d(series: np.ndarray) -> np.ndarray:
    series = series.astype(np.float32, copy=False)
    if not np.isnan(series).any():
        return series
    idx = np.arange(series.shape[0])
    mask = np.isfinite(series)
    if mask.sum() == 0:
        return np.nan_to_num(series, nan=0.0)
    series[~mask] = np.interp(idx[~mask], idx[mask], series[mask])
    return series


class MultivariateForecast:
    """
    Wraps Chronos T5 outputs (list of univariate samples) to match Timebench/GluonTS Forecast interface.
    """
    def __init__(self, samples_list):
        # samples_list elements shape: (num_samples, prediction_length)
        # Stack to: (num_samples, num_variates, prediction_length)
        
        # Convert tensor to numpy if necessary
        np_samples = [s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in samples_list]
        
        # If input is a single array (univariate case from Chronos), wrap it in list
        if isinstance(np_samples, np.ndarray):
             np_samples = [np_samples]
             
        self._samples = np.stack(np_samples, axis=1)
        # Mean shape: (num_variates, prediction_length)
        self._mean = np.mean(self._samples, axis=0)

    @property
    def samples(self):
        return self._samples

    @property
    def mean(self):
        return self._mean
    
    def cpu(self):
        return self


class MockPredictor:
    """
    Wraps pre-calculated forecasts to satisfy save_window_predictions interface.
    """
    def __init__(self, forecasts):
        self.forecasts = forecasts

    def predict(self, dataset_input, **kwargs):
        return self.forecasts


def _prepare_context(series, target_length):
    """
    Crop to at most target_length (no padding).
    """
    if isinstance(series, torch.Tensor):
        series = series.detach().cpu().numpy()
    else:
        series = np.asarray(series)

    series = series.astype(np.float32, copy=False)
    length = series.shape[0]

    if length >= target_length:
        series = series[-target_length:]

    series = _impute_nans_1d(series)
    return torch.from_numpy(series).float()


def run_chronos_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] = None,
    model_size: str = "small",
    output_dir: str | None = None,
    batch_size: int = 16, # Default batch size
    num_samples: int = 100,
    context_length: int = 4000,
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
):
    """
    Run Chronos model experiments on a dataset with specified terms.
    """
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device_map = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset configuration
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = ["short", "medium", "long"]

    if output_dir is None:
        output_dir = f"./output/results/chronos_{model_size}"

    os.makedirs(output_dir, exist_ok=True)

    # Model Mapping
    model_map = {
        "tiny": "amazon/chronos-t5-tiny",
        "mini": "amazon/chronos-t5-mini",
        "small": "amazon/chronos-t5-small",
        "base": "amazon/chronos-t5-base",
        "large": "amazon/chronos-t5-large",
    }
    hf_model_path = model_map.get(model_size, "amazon/chronos-t5-small")

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
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        # Initialize Chronos Pipeline
        # (Re-initialized per term to ensure clean state, though not strictly necessary for Chronos)
        print(f"  Initializing Chronos pipeline ({hf_model_path})...")
        pipeline = ChronosPipeline.from_pretrained(
            hf_model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )

        # Load dataset with config settings
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=True, # Force univariate processing
            prediction_length=prediction_length,
            test_length=test_length, 
            val_length=val_length,
        )

        # Calculate actual test/val length
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

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim} (Processing as Univariate)")
        print(f"    - Series length: min={dataset._min_series_length}, max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
        print(f"    - {split_name}: {data_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        season_length = get_seasonality(dataset.freq)

        # Generate predictions
        data_type = "validation" if use_val else "test"
        print(f"  Running predictions on {data_type} data...")

        # ---------------------------------------------------------
        # Chronos Inference Logic (Replacing predictor.predict loop)
        # ---------------------------------------------------------
        
        # A. Prepare Contexts (Univariate) & Prepare Batches
        # We pre-process everything to tensor of shape (context_length,) for stacking
        flat_context_tensors = []
        
        for d in eval_data.input:
            target = np.asarray(d["target"])
            if target.ndim > 1:
                target = target.squeeze()

            # (time,)
            ts = _prepare_context(target, context_length)
            flat_context_tensors.append(ts)

        # B. Batch Prediction with Tensor Input (grouped by length, no padding)
        total_items = len(flat_context_tensors)
        flat_forecasts = [None] * total_items
        if batch_size > 0:
            length_groups = {}
            for idx, tensor in enumerate(flat_context_tensors):
                length_groups.setdefault(tensor.shape[0], []).append(idx)

            processed = 0
            for length in sorted(length_groups.keys()):
                indices = length_groups[length]
                for start in range(0, len(indices), batch_size):
                    batch_indices = indices[start : start + batch_size]
                    batch_contexts = torch.stack([flat_context_tensors[i] for i in batch_indices])

                    batch_forecasts = pipeline.predict(
                        batch_contexts,
                        prediction_length=prediction_length,
                        num_samples=num_samples,
                    )
                    batch_list = list(batch_forecasts)
                    for offset, item_idx in enumerate(batch_indices):
                        flat_forecasts[item_idx] = batch_list[offset]

                    processed += len(batch_indices)
                    if processed % (batch_size * 5) == 0 or processed == total_items:
                        sys.stdout.write(f"\r    Processed {processed}/{total_items} items...")
                        sys.stdout.flush()
        else:
            for idx, context_tensor in enumerate(flat_context_tensors, 1):
                batch_forecasts = pipeline.predict(
                    context_tensor.unsqueeze(0),
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                )
                flat_forecasts[idx - 1] = batch_forecasts[0]

                if idx % 100 == 0 or idx == total_items:
                    sys.stdout.write(f"\r    Processed {idx}/{total_items} items...")
                    sys.stdout.flush()

        print(f"\r    Processed {total_items}/{total_items} items. Done.")
            
        # C. Wrap Results into Forecast Objects
        forecasts = [MultivariateForecast([fc]) for fc in flat_forecasts]

        # Count number of series
        num_total_instances = len(forecasts)
        num_series = num_total_instances // num_windows
        num_variates = forecasts[0].samples.shape[1] if forecasts else 1

        print(f"    Total instances: {num_total_instances}, Series: {num_series}, Windows: {num_windows}")

        # Collect ground truth labels and contexts
        print("    Collecting ground truth and context...")
        ground_truths = []
        contexts = []
        for inp, label in eval_data:
            ground_truths.append(label["target"])
            contexts.append(inp["target"])

        # Initialize arrays (Included to match Moirai log flow 'Organizing data...')
        predictions_mean = np.zeros((num_series, num_windows, num_variates, prediction_length))
        predictions_samples = np.zeros((num_series, num_windows, num_samples, num_variates, prediction_length))
        ground_truth = np.zeros((num_series, num_windows, num_variates, prediction_length))

        max_context_len = max(ctx.shape[-1] for ctx in contexts) if contexts else 0
        context_array = np.full((num_series, num_windows, num_variates, max_context_len), np.nan)

        print("    Organizing data into arrays...")
        for idx, (fc, gt, ctx) in enumerate(zip(forecasts, ground_truths, contexts)):
            series_idx = idx // num_windows
            window_idx = idx % num_windows

            fc_mean = fc.mean
            fc_samples = fc.samples

            if fc_mean.ndim == 1:
                fc_mean = fc_mean[np.newaxis, :]
            
            if fc_samples.ndim == 2: 
                fc_samples = fc_samples[:, np.newaxis, :]
            
            if gt.ndim == 1:
                gt = gt[np.newaxis, :]
            elif gt.shape[0] == prediction_length and gt.shape[1] == num_variates:
                gt = gt.T

            if ctx.ndim == 1:
                ctx = ctx[np.newaxis, :]
            elif ctx.shape[0] != num_variates:
                ctx = ctx.T

            predictions_mean[series_idx, window_idx] = fc_mean
            predictions_samples[series_idx, window_idx] = fc_samples 
            ground_truth[series_idx, window_idx] = gt

            ctx_len = ctx.shape[-1]
            context_array[series_idx, window_idx, :, :ctx_len] = ctx

        if use_val:
            print("    (No results saved - validation data used for hyperparameter selection)")
        else:
            # Save predictions and metrics for test data
            ds_config = f"{dataset_name}/{term}"
            
            model_hyperparams = {
                "model": f"chronos-{model_size}",
                "context_length": context_length,
                "num_samples": num_samples,
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
    parser.add_argument("--dataset", type=str, nargs="+", default=["SG_Weather/D"],
                        help="Dataset name(s). Can be a single dataset, multiple datasets, or 'all_datasets' to run all datasets from config")
    parser.add_argument("--terms", type=str, nargs="+", default=["short", "medium", "long"],
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate")
    parser.add_argument("--model-size", type=str, default="tiny",
                        choices=["tiny", "mini", "small", "base", "large"],
                        help="Chronos model size")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for prediction")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples for probabilistic forecasting")
    parser.add_argument("--context-length", type=int, default=4000,
                        help="Maximum context length")
    parser.add_argument("--cuda-device", type=str, default="0",
                        help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--val", action="store_true",
                        help="Evaluate on validation data (for hyperparameter selection, no saving)")

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
            run_chronos_experiment(
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
            traceback.print_exc()
            continue

    print(f"\n{'#'*60}")
    print(f"# All {total_datasets} dataset(s) completed!")
    print(f"{'#'*60}")

if __name__ == "__main__":
    main()
