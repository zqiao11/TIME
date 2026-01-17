"""
TimesFM model experiments for time series forecasting.

Usage:
    python experiments/timesfm2.5.py
    python experiments/timesfm2.5.py --model-size base
    python experiments/timesfm2.5.py --dataset "TSBench_IMOS_v2/15T" --terms short medium long
    python experiments/timesfm2.5.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/timesfm2.5.py --dataset "SG_Weather/D" --batch-size 32
    python experiments/timesfm2.5.py --dataset all_datasets  # Run all datasets from config
    python experiments/timesfm2.5.py --val  # Evaluate on validation data (no saving)
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import timesfm  # 引入 TimesFM
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality

from timebench.evaluation.saver import save_window_quantile_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)

# Load environment variables
load_dotenv()

DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


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


class TimesFMForecast:
    """
    Wrap TimesFM quantile outputs to match TimeBench Forecast interface.
    """
    def __init__(self, quantiles, quantile_levels: list[float]):
        q_data = np.asarray(quantiles, dtype=np.float32)
        if q_data.ndim == 2:
            if q_data.shape[0] != len(quantile_levels) and q_data.shape[1] == len(quantile_levels):
                q_data = q_data.T
            q_data = q_data[:, np.newaxis, :]
        elif q_data.ndim != 3:
            raise ValueError(f"Unexpected TimesFM quantile shape: {q_data.shape}")

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
    Wrap pre-computed forecasts to satisfy save_window_quantile_predictions interface.
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


def run_timesfm_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] = None,
    model_size: str = "base",
    output_dir: str | None = None,
    batch_size: int = 32,
    context_length: int = 1024,  # TimesFM default max context
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
    quantile_levels: list[float] | None = None,
    # TimesFM specific flags (can be exposed as args if needed)
    use_continuous_quantile_head: bool = True,
    force_flip_invariance: bool = True,
):
    """
    Run TimesFM model experiments on a dataset with specified terms.
    """
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading configuration...")
    config = load_dataset_config(config_path)

    # Auto-detect available terms from config if not specified
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILE_LEVELS

    if output_dir is None:
        output_dir = f"./output/results/timesfm_{model_size}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Terms: {terms}")
    print(f"Device: {device}")
    print(f"Evaluation on: {'Validation data (no saving)' if use_val else 'Test data'}")
    print(f"{'='*60}")

    model_map = {
        "base": "google/timesfm-2.5-200m-pytorch",
    }
    model_id = model_map.get(model_size, "google/timesfm-2.5-200m-pytorch")

    # Initialize TimesFM Model (Torch version)
    print(f"Initializing TimesFM-2.5 ({model_size})...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        model_id,
        torch_compile=True,
    )

    # Move internal model to device manually if needed,
    # though from_pretrained usually handles loading.
    # Note: TimesFM `forecast` method handles device placement logic internally based on backend.

    for term in terms:
        print(f"\n--- Term: {term} ---")

        # Get settings from config
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        # Compile TimesFM for this specific horizon and context
        # This prepares the graph/execution plan
        print(f"  Compiling TimesFM with max_horizon={prediction_length}, max_context={context_length}...")
        model.compile(
            timesfm.ForecastConfig(
                max_context=context_length,
                max_horizon=prediction_length,
                normalize_inputs=True,
                use_continuous_quantile_head=use_continuous_quantile_head,
                force_flip_invariance=force_flip_invariance,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

        # Initialize the dataset
        to_univariate = False if Dataset(name=dataset_name, term=term,to_univariate=False).target_dim == 1 else True

        # Load dataset with config settings
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate, # TimesFM is univariate
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
        print(f"    - Target dim: {dataset.target_dim} (Forced Univariate for TimesFM)")
        print(f"    - {split_name}: {data_length} steps")
        print(f"    - Windows: {num_windows}")

        season_length = get_seasonality(dataset.freq)

        # Prepare Data for Batch Prediction
        print(f"  Preparing input batches from {split_name} data...")

        # eval_data is an iterable of dictionaries. We need to batch inputs for TimesFM.
        # Each item has 'target' (history) and 'label' (ground truth).
        all_inputs = []
        all_ground_truths = []

        for inp, label in eval_data:
            # TimesFM expects 1D numpy arrays for univariate
            history = inp["target"]
            # if context_length is not None and history.shape[0] > context_length:
            #     history = history[-context_length:]
            # history = _impute_nans_1d(np.asarray(history))

            all_inputs.append(history)
            all_ground_truths.append(label["target"])

        num_total_instances = len(all_inputs)
        print(f"    Total instances to forecast: {num_total_instances}")

        # Initialize Results Lists
        fc_quantiles = []

        # Run Batch Prediction
        print(f"  Running predictions (Batch size: {batch_size})...")

        for i in range(0, num_total_instances, batch_size):
            batch_inputs = all_inputs[i : i + batch_size]

            # TimesFM forecast
            point_forecast, quantile_forecast = model.forecast(
                horizon=prediction_length,
                inputs=batch_inputs,
            )

            # quantile_forecast: (batch, horizon, 10) -> mean, 10th...90th
            fc_quantiles.append(quantile_forecast)

            if (i + batch_size) % (batch_size * 10) == 0:
                print(f"    Processed {i + batch_size}/{num_total_instances}...")

        # Concatenate results
        full_quantiles = np.concatenate(fc_quantiles, axis=0) # (Total, Horizon, 10)

        forecasts = []
        for idx in range(num_total_instances):
            q_with_mean = full_quantiles[idx]  # (Horizon, 10)
            q_only = q_with_mean[:, 1:]  # (Horizon, 9)
            forecasts.append(
                TimesFMForecast(
                    q_only.T,
                    quantile_levels=quantile_levels,
                )
            )

        if use_val:
            print("    (No results saved - validation data used for hyperparameter selection)")
        else:
            ds_config = f"{dataset_name}/{term}"

            model_hyperparams = {
                "model": "TimesFM-2.5-200M",
                "context_length": context_length,
                "force_flip_invariance": force_flip_invariance,
                "output_type": "quantiles",
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
            print(f"  Completed: {metadata['num_series']} series x {metadata['num_windows']} windows")
            print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run TimesFM experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["SG_Weather/D"],
                        help="Dataset name(s). 'all_datasets' for all.")
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--model-size", type=str, default="base",
                        choices=["base"],
                        help="TimesFM model size")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for prediction")
    parser.add_argument("--context-length", type=int, default=4096,
                        help="Maximum context length; 15360 is the max for TimesFM-2.5-200M")
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=DEFAULT_QUANTILE_LEVELS,
        help="Quantile levels to predict",
    )
    parser.add_argument("--cuda-device", type=str, default="0",
                        help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--val", action="store_true",
                        help="Evaluate on validation data")

    args = parser.parse_args()

    # Handle dataset list or 'all_datasets'
    config_path = Path(args.config) if args.config else None

    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
        print(f"Running all {len(datasets)} datasets from config:")
    else:
        datasets = args.dataset

    # Iterate over all datasets
    total_datasets = len(datasets)
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        try:
            run_timesfm_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                model_size=args.model_size,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                context_length=args.context_length,
                quantile_levels=args.quantiles,
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
