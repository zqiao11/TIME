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
from timebench.evaluation.utils import get_available_terms
from timebench.models import SeasonalNaivePredictor

# Load environment variables
load_dotenv()



def run_seasonal_naive_experiment(
    dataset_name: str,
    terms: list[str] = None,
    output_dir: str | None = None,
    num_samples: int = 100,
    config_path: Path | None = None,
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

    # Auto-detect available terms from config if not specified
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if output_dir is None:
        output_dir = "./output/results/seasonal_naive"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model: Seasonal Naive")
    print(f"Dataset: {dataset_name}")
    print(f"Terms: {terms}")
    print(f"{'='*60}")

    for term in terms:
        print(f"\n--- Term: {term} ---")

        # Get settings from config
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        # Initialize the dataset
        to_univariate = False if Dataset(name=dataset_name, term=term,to_univariate=False).target_dim == 1 else True

        # Load dataset with config settings
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate,
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )

        season_length = get_seasonality(dataset.freq)
        # Initialize Seasonal Naive predictor
        predictor = SeasonalNaivePredictor(
            prediction_length=dataset.prediction_length,
            season_length=season_length,
            freq=dataset.freq,
            num_samples=num_samples,
        )

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
        forecasts = list(predictor.predict(eval_data.input))

        fc_samples = []
        for fc in forecasts:
            fc_samples.append(fc.samples[np.newaxis, ...])
        fc_samples = np.concatenate(fc_samples, axis=0)  # (num_total_instances, num_samples, 1, prediction_length)

        # Convert samples to quantiles
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        quantile_levels_array = np.array(quantile_levels, dtype=float)

        fc_quantiles = np.quantile(fc_samples, quantile_levels_array, axis=1)
        # np.quantile returns (num_quantiles, num_total_instances, 1, prediction_length), need to transpose
        fc_quantiles = fc_quantiles.transpose(1, 0, 2, 3)  # (num_total_instances, num_quantiles, 1, prediction_length)
        fc_quantiles = fc_quantiles.squeeze(axis=2)

        # Compute metrics
        ds_config = f"{dataset_name}/{term}"

        # Prepare model hyperparameters for metadata
        model_hyperparams = {
            "season_length": season_length,
        }

        metadata = save_window_predictions(
            dataset=dataset,
            fc_quantiles=fc_quantiles,
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
    parser.add_argument("--dataset", type=str, nargs="+", default=["Port_Activity/D"],
                        help="Dataset name(s). Can be a single dataset, multiple datasets, or 'all_datasets' to run all datasets from config")
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples for probabilistic forecasting (all identical for Seasonal Naive)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
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

