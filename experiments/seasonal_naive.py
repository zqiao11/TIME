"""
Seasonal Naive baseline experiments for time series forecasting.

The Seasonal Naive method forecasts the value from the same season
in the previous seasonal cycle. This is a simple but effective baseline
for seasonal time series.

Usage:
    python experiments/seasonal_naive.py
    python experiments/seasonal_naive.py --dataset "SG_Weather/D" --terms short medium long
    python experiments/seasonal_naive.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/seasonal_naive.py --dataset all_datasets  # Run all datasets from config
    python experiments/seasonal_naive.py --val  # Evaluate on validation data (no saving)
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.metrics import compute_per_window_metrics
from timebench.models import SeasonalNaivePredictor

# Load environment variables
load_dotenv()


# Custom seasonality mapping for frequencies where get_seasonality returns 1
# These are more meaningful seasonal periods for forecasting
CUSTOM_SEASONALITY = {
    "D": 7,      # Daily → weekly seasonality (7 days)
    "W": 52,     # Weekly → yearly seasonality (52 weeks)
    "M": 12,     # Monthly → yearly seasonality (12 months)
    "Q": 4,      # Quarterly → yearly seasonality (4 quarters)
    "A": 1,      # Annual → no meaningful sub-annual seasonality
    "Y": 1,      # Yearly → no meaningful sub-annual seasonality
}


def get_effective_seasonality(freq: str) -> int:
    """
    Get effective seasonality for a given frequency.

    Uses custom mapping for frequencies where gluonts.get_seasonality returns 1,
    otherwise falls back to get_seasonality.

    Args:
        freq: Frequency string (e.g., 'D', 'H', '15T')

    Returns:
        Seasonal period length
    """
    # Normalize frequency string (handle cases like '1D', '1H')
    freq_upper = freq.upper().lstrip('0123456789')

    # Check custom mapping first
    if freq_upper in CUSTOM_SEASONALITY:
        return CUSTOM_SEASONALITY[freq_upper]

    # Fall back to gluonts seasonality
    season_length = get_seasonality(freq)

    # # If get_seasonality returns 1, try to infer a reasonable default
    # if season_length == 1:
    #     # For sub-daily frequencies not in our mapping, use daily seasonality
    #     if 'T' in freq_upper or 'MIN' in freq_upper:
    #         # Minutes per day / minutes per period
    #         return 24 * 60 // max(1, int(''.join(filter(str.isdigit, freq)) or 1))
    #     elif 'S' in freq_upper:
    #         # Seconds: use hourly seasonality
    #         return 3600

    return season_length


def run_seasonal_naive_experiment(
    dataset_name: str = "SG_Weather/D",
    terms: list[str] = None,
    output_dir: str | None = None,
    num_samples: int = 100,
    config_path: Path | None = None,
    use_val: bool = False,
):
    """
    Run Seasonal Naive baseline experiments on a dataset with specified terms.

    Args:
        dataset_name: Dataset name (e.g., "SG_Weather/D")
        terms: List of terms to evaluate ("short", "medium", "long")
        output_dir: Output directory for results
        num_samples: Number of samples for forecast (all identical for point forecast)
        config_path: Path to datasets.yaml config file
        use_val: If True, evaluate on validation data (for hyperparameter selection, no saving)
    """
    # Load dataset configuration
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = ["short", "medium", "long"]

    if output_dir is None:
        output_dir = "./output/results/seasonal_naive"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model: Seasonal Naive")
    print(f"Dataset: {dataset_name}")
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

        # Load dataset with config settings
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=False,
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )

        # Get seasonality for the dataset frequency
        # Use custom mapping for frequencies where get_seasonality returns 1
        season_length = get_effective_seasonality(dataset.freq)
        gluonts_season = get_seasonality(dataset.freq)
        if season_length != gluonts_season:
            print(f"  Seasonality: {season_length} (custom, gluonts default was {gluonts_season}, freq={dataset.freq})")
        else:
            print(f"  Detected seasonality: {season_length} (freq={dataset.freq})")

        # Initialize Seasonal Naive predictor
        predictor = SeasonalNaivePredictor(
            prediction_length=dataset.prediction_length,
            season_length=season_length,
            freq=dataset.freq,
            num_samples=num_samples,
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
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - Series length: min={dataset._min_series_length}, max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
        print(f"    - {split_name}: {data_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")
        print(f"    - Season length: {season_length}")

        # Generate predictions
        data_type = "validation" if use_val else "test"
        print(f"  Running predictions on {data_type} data...")
        forecasts = list(predictor.predict(eval_data.input))

        # Count number of series
        num_total_instances = len(forecasts)
        num_series = num_total_instances // num_windows
        num_variates = dataset.target_dim
        pred_len = dataset.prediction_length

        print(f"    Total instances: {num_total_instances}, Series: {num_series}, Windows: {num_windows}")

        # Collect ground truth labels and contexts
        print("    Collecting ground truth and context...")
        ground_truths = []
        contexts = []
        for inp, label in eval_data:
            ground_truths.append(label["target"])
            contexts.append(inp["target"])

        # Initialize arrays
        fc_num_samples = forecasts[0].samples.shape[0] if len(forecasts) > 0 else num_samples

        predictions_samples = np.zeros((num_series, num_windows, fc_num_samples, num_variates, pred_len))
        ground_truth = np.zeros((num_series, num_windows, num_variates, pred_len))

        # Find max context length to pad contexts
        max_context_len = max(ctx.shape[-1] for ctx in contexts)
        context_array = np.full((num_series, num_windows, num_variates, max_context_len), np.nan)

        print("    Organizing data into arrays...")
        for idx, (fc, gt, ctx) in enumerate(zip(forecasts, ground_truths, contexts)):
            series_idx = idx // num_windows
            window_idx = idx % num_windows

            # Get forecast samples
            fc_samples = fc.samples

            # Handle shape: fc_samples is (num_samples, num_variates, pred_len)
            # No transpose needed as our predictor outputs in correct shape

            # Handle ground truth shape
            if gt.ndim == 1:
                gt = gt[np.newaxis, :]
            elif gt.shape[0] == pred_len and gt.shape[1] == num_variates:
                gt = gt.T

            # Handle context shape
            if ctx.ndim == 1:
                ctx = ctx[np.newaxis, :]
            elif ctx.shape[0] != num_variates:
                ctx = ctx.T

            predictions_samples[series_idx, window_idx] = fc_samples
            ground_truth[series_idx, window_idx] = gt

            # Store context (padded with NaN for shorter contexts)
            ctx_len = ctx.shape[-1]
            context_array[series_idx, window_idx, :, :ctx_len] = ctx

        # Compute metrics for validation or save for test
        if use_val:
            # For validation: just compute and print metrics
            print("    Computing metrics...")
            metrics = compute_per_window_metrics(
                predictions_samples=predictions_samples,
                ground_truth=ground_truth,
                context=context_array,
                seasonality=season_length,
            )
            print("    Metrics summary (averaged over all series/windows/variates):")
            for metric_name, metric_values in metrics.items():
                mean_val = np.nanmean(metric_values)
                print(f"      {metric_name}: {mean_val:.4f}")
            print("    (No results saved - validation data used for hyperparameter selection)")
        else:
            # For test: save predictions and metrics
            ds_config = f"{dataset_name}/{term}"

            # Prepare model hyperparameters for metadata
            model_hyperparams = {
                "season_length": season_length,
            }

            metadata = save_window_predictions(
                dataset=dataset,
                predictor=predictor,
                ds_config=ds_config,
                output_base_dir=output_dir,
                seasonality=season_length,
                model_hyperparams=model_hyperparams,
            )
            print(f"  Completed: {metadata['num_series']} series x {metadata['num_windows']} windows")
            print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Seasonal Naive baseline experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["SG_Weather/D"],
                        help="Dataset name(s). Can be a single dataset, multiple datasets, or 'all_datasets' to run all datasets from config")
    parser.add_argument("--terms", type=str, nargs="+", default=["short", "medium", "long"],
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples for probabilistic forecasting (all identical for Seasonal Naive)")
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
            run_seasonal_naive_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                output_dir=args.output_dir,
                num_samples=args.num_samples,
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

