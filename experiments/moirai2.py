"""
Moirai-2.0 model experiments for time series forecasting.

Usage:
    python experiments/moriai2.py
    python experiments/moriai2.py --model-size small
    python experiments/moriai2.py --dataset "Coastal_T_S/5T" --terms short medium long
    python experiments/moriai2.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/moriai2.py --dataset all_datasets  # Run all datasets from config
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
# Strict import based on your demo
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.utils import get_available_terms
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)

load_dotenv()


def run_moirai2_experiment(
    dataset_name: str,
    terms: list[str] = None,
    model_size: str = "base",
    output_dir: str | None = None,
    batch_size: int = 16,
    context_length: int = 1680,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
):
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    # Auto-detect available terms from config if not specified
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")
    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    model_size_used = model_size
    hf_model_path = f"Salesforce/moirai-2.0-R-{model_size}"
    try:
        module = Moirai2Module.from_pretrained(hf_model_path)
    except Exception as e:
        if model_size != "small":
            model_size_used = "small"
            hf_model_path = "Salesforce/moirai-2.0-R-small"
            print(f"WARNING: Failed to load '{model_size}', falling back to '{model_size_used}'.")
            module = Moirai2Module.from_pretrained(hf_model_path)
        else:
            raise e

    # Print model parameter count
    total_params = sum(p.numel() for p in module.parameters())
    print(f"Model: {hf_model_path}, Total parameters: {total_params:,}")

    if output_dir is None:
        output_dir = "./output/results/moirai2"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Terms: {terms}")
    print(f"Model: Moirai-2.0 ({model_size_used})")
    print(f"{'='*60}")

    for term in terms:
        print(f"\n--- Term: {term} ---")

        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        # Moirai2 only supports univariate forecasting
        to_univariate = False if Dataset(name=dataset_name, term=term,to_univariate=False).target_dim == 1 else True

        # Load dataset first to get dimensions
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate,
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )

        # Initialize model STRICTLY following your demo structure
        print(f"  Initializing Moirai-2.0-{model_size_used} model...")

        model = Moirai2Forecast(
            module=module,
            prediction_length=dataset.prediction_length,
            context_length=context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        num_windows = dataset.windows

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")


        print(f"  Running predictions...")
        predictor = model.create_predictor(batch_size=batch_size)
        test_data = dataset.test_data
        forecasts = list(predictor.predict(test_data.input))
        fc_quantiles = []
        for fc in forecasts:
            fc_quantiles.append(fc.forecast_array[np.newaxis, ...])
        fc_quantiles = np.concatenate(fc_quantiles, axis=0)  # (num_total_instances, num_quantiles, prediction_length)

        season_length = get_seasonality(dataset.freq)

        ds_config = f"{dataset_name}/{term}"
        model_hyperparams = {
            "model_version": "2.0",
            "model_size": model_size_used,
            "context_length": context_length,
            "quantile_levels": quantile_levels,
        }

        metadata = save_window_predictions(
            dataset=dataset,
            fc_quantiles=fc_quantiles,
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
    parser = argparse.ArgumentParser(description="Run Moirai-2.0 experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["Global_Influenza/W"],
                        help="Dataset name(s)")
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"], help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--model-size", type=str, default="base",
                        choices=["small", "base", "large"], help="Moirai model size")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Quantile levels to predict",
    )
    parser.add_argument("--context-length", type=int, default=4000, help="Max context length")
    parser.add_argument("--config", type=str, default=None, help="Path to datasets.yaml")

    args = parser.parse_args()
    config_path = Path(args.config) if args.config else None

    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
    else:
        datasets = args.dataset

    total_datasets = len(datasets)
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")
        try:
            run_moirai2_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                model_size=args.model_size,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                context_length=args.context_length,
                config_path=config_path,
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