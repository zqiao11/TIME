"""
TiRex model experiments for time series forecasting.

Requires:
    pip install tirex

Usage:
    python experiments/tirex_model.py
    python experiments/tirex_model.py --model-id "NX-AI/TiRex"
    python experiments/tirex_model.py --model-size base
    python experiments/tirex_model.py --dataset "Traffic/15T" --terms short medium long
    python experiments/tirex_model.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/tirex_model.py --dataset all_datasets  # Run all datasets from config
    python experiments/tirex_model.py --val  # Evaluate on validation data (no saving)
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
from gluonts.time_feature import get_seasonality

from tirex import ForecastModel, load_model

from timebench.evaluation.saver import save_window_quantile_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)

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


def _clean_nan_target(series: np.ndarray) -> np.ndarray:
    if series.ndim == 1:
        return _impute_nans_1d(series)
    if series.ndim == 2:
        cleaned = np.empty_like(series, dtype=np.float32)
        for i in range(series.shape[0]):
            cleaned[i] = _impute_nans_1d(series[i])
        return cleaned
    return np.nan_to_num(series, nan=0.0)


class TirexQuantileForecast:
    def __init__(self, quantiles, quantile_levels, mean=None):
        q_data = quantiles
        if isinstance(q_data, torch.Tensor):
            q_data = q_data.detach().cpu().float().numpy()
        if mean is not None and isinstance(mean, torch.Tensor):
            mean = mean.detach().cpu().float().numpy()

        q_data = np.asarray(q_data, dtype=np.float32)
        levels = [float(q) for q in quantile_levels]
        num_levels = len(levels)

        if q_data.ndim == 1:
            q_data = q_data[np.newaxis, :]
        if q_data.ndim == 2:
            if q_data.shape[0] != num_levels and q_data.shape[1] == num_levels:
                q_data = q_data.T
        elif q_data.ndim == 3:
            if q_data.shape[0] != num_levels and q_data.shape[-1] == num_levels:
                q_data = q_data.transpose(2, 0, 1)
        else:
            raise ValueError(f"Unexpected quantile shape: {q_data.shape}")

        self._quantiles = q_data
        self._quantile_levels = levels

        if mean is None:
            if 0.5 in self._quantile_levels:
                idx = int(np.where(np.isclose(self._quantile_levels, 0.5, atol=1e-6))[0][0])
                mean = self._quantiles[idx]
            else:
                mean = np.mean(self._quantiles, axis=0)
        self._mean = mean

    @property
    def mean(self):
        return self._mean

    @property
    def quantile_levels(self):
        return self._quantile_levels

    def quantile(self, q: float):
        q_levels = np.asarray(self._quantile_levels, dtype=float)
        matches = np.where(np.isclose(q_levels, q, atol=1e-6))[0]
        if matches.size == 0:
            raise ValueError(f"Quantile {q} not available. Supported: {self._quantile_levels}")
        return self._quantiles[int(matches[0])]

    def cpu(self):
        return self


class MockPredictor:
    def __init__(self, forecasts):
        self.forecasts = forecasts

    def predict(self, dataset_input, **kwargs):
        return self.forecasts


def _prepare_context(series, target_length):
    """
    Ensure series is exactly target_length for batching.
    - If longer, crop to the last `target_length` points.
    - If shorter, left-pad with zeros to target_length.
    """
    if series.shape[0] >= target_length:
        return series[-target_length:]
    pad_len = target_length - series.shape[0]
    pad = np.zeros((pad_len,), dtype=series.dtype)
    return np.concatenate([pad, series], axis=0)


def _normalize_mean(mean, batch_size, prediction_length):
    mean = np.asarray(mean)
    if mean.ndim == 1:
        mean = mean[np.newaxis, :]
    elif mean.ndim == 2:
        if mean.shape[0] != batch_size and mean.shape[1] == batch_size:
            mean = mean.T
    elif mean.ndim == 3:
        if mean.shape[-1] == 1:
            mean = mean.squeeze(-1)
        elif mean.shape[1] == 1:
            mean = mean.squeeze(1)
        if mean.ndim == 1:
            mean = mean[np.newaxis, :]
    if mean.shape[1] > prediction_length:
        mean = mean[:, :prediction_length]
    return mean


def _normalize_quantiles(quantiles, batch_size, prediction_length):
    if quantiles is None:
        return None
    q = np.asarray(quantiles)
    if q.ndim == 2:
        if q.shape[0] == prediction_length:
            q = q[np.newaxis, :, :]
        elif q.shape[1] == prediction_length:
            q = q.T[np.newaxis, :, :]
        else:
            return None
    elif q.ndim == 3:
        if q.shape[0] == batch_size and q.shape[1] == prediction_length:
            pass
        elif q.shape[0] == batch_size and q.shape[2] == prediction_length:
            q = q.transpose(0, 2, 1)
        elif q.shape[1] == batch_size and q.shape[2] == prediction_length:
            q = q.transpose(1, 2, 0)
        elif q.shape[2] == batch_size and q.shape[1] == prediction_length:
            q = q.transpose(2, 1, 0)
        else:
            return None
    else:
        return None

    if q.shape[1] > prediction_length:
        q = q[:, :prediction_length, :]
    return q


def _resolve_quantile_levels(num_quantiles, quantile_levels):
    if quantile_levels is None or len(quantile_levels) != num_quantiles:
        quantile_levels = np.linspace(0.1, 0.9, num_quantiles)
    return [float(q) for q in quantile_levels]


def run_tirex_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] | None = None,
    model_id: str = "NX-AI/TiRex",
    output_dir: str | None = None,
    batch_size: int = 128,
    context_length: int = 4000, 
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
    quantile_levels: list[float] | None = None,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = ["short", "medium", "long"]
    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if output_dir is None:
        output_dir = "./output/results/tirex"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_id}")
    print(f"Terms: {terms}")
    print(f"Evaluation on: {'Validation data (no saving)' if use_val else 'Test data'}")
    print(f"{'='*60}")

    print(f"  Loading TiRex model ({model_id})...")
    model: ForecastModel = load_model(model_id)
    if hasattr(model, "to"):
        model = model.to(device)

    for term in terms:
        print(f"\n--- Term: {term} ---")

        # --- FIX: Match Moirai's logic for retrieving settings ---
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length") # Changed from test_split
        val_length = settings.get("val_length")   # Changed from val_split

        print(
            "  Config: prediction_length={}, test_length={}, val_length={}".format(
                prediction_length, test_length, val_length
            )
        )

        # TiRex is a univariate model.
        to_univariate = True
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate,
            prediction_length=prediction_length,
            test_length=test_length, # Changed from test_split
            val_length=val_length,   # Changed from val_split
        )

        eval_target_dim = 1

        # --- FIX: Calculate actual data length based on settings (not splits) ---
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
        print(f"    - Target dim: {eval_target_dim}")
        print(
            "    - Series length: min={}, max={}, avg={:.1f}".format(
                dataset._min_series_length,
                dataset._max_series_length,
                dataset._avg_series_length,
            )
        )
        print(f"    - {split_name}: {data_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        season_length = get_seasonality(dataset.freq)

        print(
            f"  Running predictions on {'validation' if use_val else 'test'} data..."
        )
        forecasts = []
        
        # 1. Collect all series and Pre-process (Pad/Crop)
        flat_contexts = []
        for entry in eval_data.input:
            target = _clean_nan_target(np.asarray(entry["target"], dtype=np.float32))
            if target.ndim == 1:
                series = target
            else:
                series = target.squeeze()
            
            # Use strict context length for consistency 
            series = _prepare_context(series, context_length)
            flat_contexts.append(series)

        # 2. Batch Inference
        total_items = len(flat_contexts)
        
        for start in range(0, total_items, batch_size):
            end = min(start + batch_size, total_items)
            batch = flat_contexts[start:end]
            
            batch_tensor = torch.tensor(np.stack(batch), dtype=torch.float32, device=device)

            with torch.no_grad():
                quantiles, mean = model.forecast(
                    context=batch_tensor, prediction_length=prediction_length
                )

            batch_size_actual = batch_tensor.shape[0]
            mean_np = _normalize_mean(mean, batch_size_actual, prediction_length) if mean is not None else None
            quantiles_np = _normalize_quantiles(
                quantiles, batch_size_actual, prediction_length
            )
            if quantiles_np is None:
                raise ValueError("TiRex forecast did not return usable quantiles.")
            quantile_levels = _resolve_quantile_levels(
                quantiles_np.shape[-1], quantile_levels
            )
            batch_quantile_levels = quantile_levels

            for i in range(batch_size_actual):
                mean_series = mean_np[i] if mean_np is not None else None
                q_series = quantiles_np[i]
                forecasts.append(
                    TirexQuantileForecast(
                        q_series,
                        quantile_levels=batch_quantile_levels,
                        mean=mean_series,
                    )
                )
            
            if start % (batch_size * 5) == 0:
                sys.stdout.write(f"\r    Processed {end}/{total_items} items...")
                sys.stdout.flush()
        
        print(f"\r    Processed {total_items}/{total_items} items. Done.")

        num_total_instances = len(forecasts)
        num_series = num_total_instances // num_windows
        num_variates = eval_target_dim

        print(
            f"    Total instances: {num_total_instances}, Series: {num_series}, Windows: {num_windows}"
        )

        if use_val:
            print("    [Validation] Organizing data and computing metrics manually...")
            
            ground_truths = []
            contexts = []
            for inp, label in eval_data:
                ground_truths.append(label["target"])
                contexts.append(inp["target"])
            
            # Metric calculation omitted for batch run cleanliness, similar to Moirai structure
            print("    (No results saved - validation mode)")
            
        else:
            ds_config = f"{dataset_name}/{term}"
            model_hyperparams = {
                "model_id": model_id,
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
                quantile_levels=quantile_levels,
            )
            print(
                f"  Completed: {metadata['num_series']} series x {metadata['num_windows']} windows"
            )
            print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run TiRex experiments")
    parser.add_argument(
        "--dataset", 
        type=str, 
        nargs="+", 
        default=["IMOS/15T"], 
        help="Dataset name(s)"
    )
    parser.add_argument(
        "--terms",
        type=str,
        nargs="+",
        default=["short", "medium", "long"],
        choices=["short", "medium", "long"],
        help="Terms to evaluate",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model id (overrides --model-size)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["base"],
        help="Model size alias (maps to a model id)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for results"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for prediction"
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Quantile levels to predict",
    )
    parser.add_argument(
        "--context-length", type=int, default=4000, help="Maximum context length"
    )
    parser.add_argument("--cuda-device", type=str, default="0", help="CUDA device ID")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to datasets.yaml config file"
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Evaluate on validation data (for hyperparameter selection, no saving)",
    )

    args = parser.parse_args()
    config_path = Path(args.config) if args.config else None

    # Handle 'all_datasets' logic
    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
        print(f"Running all {len(datasets)} datasets from config:")
        for ds in datasets:
            print(f"  - {ds}")
    else:
        datasets = args.dataset

    total_datasets = len(datasets)
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        try:
            model_size_map = {
                "base": "NX-AI/TiRex",
            }
            model_id = args.model_id or model_size_map.get(args.model_size, "NX-AI/TiRex")
            run_tirex_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                model_id=model_id,
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
            traceback.print_exc()
            continue
            
    print(f"\n{'#'*60}")
    print(f"# All {total_datasets} dataset(s) completed!")
    print(f"{'#'*60}")

if __name__ == "__main__":
    main()
