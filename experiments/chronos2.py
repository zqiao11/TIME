"""
Chronos-2 model experiments for time series forecasting.
(Refactored to match Moirai interface, logging format, and evaluation logic)

Usage:
    python experiments/chronos2.py
    python experiments/chronos2.py --model-size chronos2
    python experiments/chronos2.py --dataset "TSBench_IMOS_v2/15T" --terms short medium long
    python experiments/chronos2.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/chronos2.py --dataset all_datasets  # Run all datasets from config
    python experiments/chronos2.py --val  # Evaluate on validation data (no saving)
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from dotenv import load_dotenv
from chronos import Chronos2Pipeline
from gluonts.time_feature import get_seasonality

from timebench.evaluation.saver import save_window_quantile_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)

# Load environment variables
load_dotenv()

logging.getLogger("chronos").setLevel(logging.ERROR)

# ------------------------------
# NaN cleaning helpers
# ------------------------------
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


def _clean_nan_target(series: np.ndarray) -> np.ndarray:
    if series.ndim == 1:
        return _impute_nans_1d(series)
    if series.ndim == 2:
        cleaned = np.empty_like(series, dtype=np.float32)
        for i in range(series.shape[0]):
            cleaned[i] = _impute_nans_1d(series[i])
        return cleaned
    return np.nan_to_num(series, nan=0.0)

# --- Helper Classes for Chronos-2 adaptation ---

class MultivariateForecast:
    """
    Adapts Chronos-2 quantile output for Timebench.
    """
    def __init__(self, quantiles_tensor, mean_tensor, quantile_levels: list[float]):
        self._mean = mean_tensor.cpu().float().numpy() if isinstance(mean_tensor, torch.Tensor) else mean_tensor
        q_data = quantiles_tensor.cpu().float().numpy() if isinstance(quantiles_tensor, torch.Tensor) else quantiles_tensor
        q_data = np.asarray(q_data, dtype=np.float32)
        if q_data.ndim == 2:
            q_data = q_data[:, np.newaxis, :]
        self._quantiles = q_data
        self._quantile_levels = [float(q) for q in quantile_levels]

    def quantile(self, q: float):
        q_levels = np.asarray(self._quantile_levels, dtype=float)
        matches = np.where(np.isclose(q_levels, q, atol=1e-6))[0]
        if matches.size == 0:
            raise ValueError(f"Quantile {q} not available. Supported: {self._quantile_levels}")
        return self._quantiles[int(matches[0])]

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


def run_chronos2_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] = None,
    model_size: str = "chronos2",
    output_dir: str | None = None,
    batch_size: int = 32,
    context_length: int = 2048,
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
    quantile_levels: list[float] | None = None,
):
    """
    Run Chronos-2 model experiments.
    """
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device_map = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset configuration
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    # Auto-detect available terms from config if not specified
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if output_dir is None:
        output_dir = f"./output/results/chronos2_{model_size}"

    os.makedirs(output_dir, exist_ok=True)

    # Model Mapping
    model_map = {
        "chronos2": "amazon/chronos-2",
        # Add other sizes if released
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
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        # Initialize Chronos Pipeline
        print(f"  Initializing Chronos pipeline ({hf_model_path})...")
        pipeline = Chronos2Pipeline.from_pretrained(
            hf_model_path,
            device_map=device_map,
        )

        # Dataset Initialization
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=False, # Chronos 2 supports multivariate natively
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )

        # Determine split
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
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - Series length: min={dataset._min_series_length}, max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
        print(f"    - {split_name}: {data_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        season_length = get_seasonality(dataset.freq)

        # ---------------------------------------------------------
        # 1. Running Inference (Chronos-2 Specific Logic)
        # ---------------------------------------------------------
        data_type = "validation" if use_val else "test"
        print(f"  Running predictions on {data_type} data (Batch size: {batch_size})...")

        # Helper function to prepare a single context
        def _prepare_context(d):
            target = np.asarray(d["target"])

            if target.ndim == 2 and target.shape[0] > target.shape[1]:
                target = target.T

            # Manually truncate context
            seq_len = target.shape[-1]
            if context_length is not None and seq_len > context_length:
                target = target[..., -context_length:]

            target = _clean_nan_target(target)

            if target.ndim == 1:
                target = target[np.newaxis, :]

            return torch.tensor(target)

        # Batch Inference with lazy loading
        forecasts = []
        eval_input_list = list(eval_data.input)  # Convert to list for indexing
        total_items = len(eval_input_list)

        if batch_size > 0:
            for start in range(0, total_items, batch_size):
                end = min(start + batch_size, total_items)
                # Load context only for current batch
                batch_contexts = [_prepare_context(eval_input_list[i]) for i in range(start, end)]

                class ContentFilterStderr:
                    def __init__(self, original_stream):
                        self.original_stream = original_stream

                    def write(self, data):
                        if "Quantiles to be predicted" in data and "Chronos-2" in data:
                            return
                        self.original_stream.write(data)

                    def flush(self):
                        self.original_stream.flush()

                original_stderr = sys.stderr
                sys.stderr = ContentFilterStderr(original_stderr)
                try:
                    with torch.no_grad():
                        batch_q, batch_m = pipeline.predict_quantiles(
                            inputs=batch_contexts,
                            prediction_length=prediction_length,
                            quantile_levels=quantile_levels,
                        )
                finally:
                    sys.stderr = original_stderr

                # Handle return types (Tensor vs List)
                if isinstance(batch_q, torch.Tensor):
                    if batch_q.ndim == 4 and batch_q.shape[-1] == len(quantile_levels):
                        batch_q = batch_q.permute(0, 3, 1, 2)

                    batch_q = batch_q.cpu()
                    batch_m = batch_m.cpu()

                    for i in range(batch_q.shape[0]):
                        forecasts.append(MultivariateForecast(
                            batch_q[i], batch_m[i], quantile_levels=quantile_levels
                        ))

                elif isinstance(batch_q, list):
                    for q, m in zip(batch_q, batch_m):
                        if q.ndim == 3 and q.shape[-1] == len(quantile_levels):
                            q = q.permute(2, 0, 1)

                        forecasts.append(MultivariateForecast(
                            q, m, quantile_levels=quantile_levels
                        ))

                # Optional progress logging to match Bolt/Moirai if needed
                if (start // batch_size + 1) % 10 == 0:
                     pass

        # ---------------------------------------------------------
        # 3. Saving Results
        # ---------------------------------------------------------
        if use_val:
            print("    (No results saved - validation data used for hyperparameter selection)")
        else:
            ds_config = f"{dataset_name}/{term}"
            model_hyperparams = {
                "model": f"chronos2-{model_size}",
                "context_length": context_length,
                "quantile_levels": quantile_levels,
            }

            mock_predictor = MockPredictor(forecasts)

            metadata = save_window_quantile_predictions(
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
                        help="Dataset name(s). Can be a single dataset, multiple datasets, or 'all_datasets'")
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--model-size", type=str, default="chronos2",
                        help="Chronos model size (use 'chronos2' for amazon/chronos-2)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for prediction")
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Quantile levels to predict",
    )
    parser.add_argument("--context-length", type=int, default=8192,
                        help="Maximum context length")
    parser.add_argument("--cuda-device", type=str, default="0",
                        help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--val", action="store_true",
                        help="Evaluate on validation data")

    args = parser.parse_args()

    # [Logic] Handle dataset list or 'all_datasets'
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

    # [Logic] Iterate over all datasets with progress logging
    total_datasets = len(datasets)
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        try:
            run_chronos2_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                model_size=args.model_size,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                context_length=args.context_length,
                cuda_device=args.cuda_device,
                config_path=config_path,
                use_val=args.val,
                quantile_levels=args.quantiles,
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
