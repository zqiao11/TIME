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
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality

from tirex import ForecastModel, load_model

from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.utils import get_available_terms

load_dotenv()

DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def run_tirex_experiment(
    dataset_name: str,
    terms: list[str] | None = None,
    model_id: str = "NX-AI/TiRex",
    output_dir: str | None = None,
    batch_size: int = 128,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
):
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
        output_dir = "./output/results/TiRex"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_id}")
    print(f"Terms: {terms}")
    print(f"{'='*60}")

    print(f"  Loading TiRex model ({model_id})...")

    model: ForecastModel = load_model(model_id, device=device)

    for term in terms:
        print(f"\n--- Term: {term} ---")

        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(
            "  Config: prediction_length={}, test_length={}, val_length={}".format(
                prediction_length, test_length, val_length
            )
        )

        to_univariate = (
            False
            if Dataset(name=dataset_name, term=term, to_univariate=False).target_dim == 1
            else True
        )
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate,
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )

        eval_target_dim = 1

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

        fc_quantiles = []

        flat_contexts = []
        for entry in eval_data.input:
            target = np.asarray(entry["target"], dtype=np.float32)
            if target.ndim == 1:
                series = target
            else:
                series = target.squeeze()

            flat_contexts.append(series)

        total_items = len(flat_contexts)

        for start in range(0, total_items, batch_size):
            end = min(start + batch_size, total_items)
            batch = flat_contexts[start:end]

            with torch.no_grad():
                quantiles, mean = model.forecast(
                    context=batch, prediction_length=prediction_length
                )

            if isinstance(quantiles, torch.Tensor):
                quantiles_np = quantiles.detach().cpu().float().numpy()
            else:
                quantiles_np = np.asarray(quantiles)

            # (batch, prediction_length, num_quantiles) -> (batch, num_quantiles, prediction_length)
            quantiles_np = quantiles_np.transpose(0, 2, 1)

            fc_quantiles.append(quantiles_np)

            if start % (batch_size * 5) == 0:
                sys.stdout.write(f"\r    Processed {end}/{total_items} items...")
                sys.stdout.flush()

        print(f"\r    Processed {total_items}/{total_items} items. Done.")

        fc_quantiles = np.concatenate(fc_quantiles, axis=0).astype(np.float32, copy=False)

        ds_config = f"{dataset_name}/{term}"
        model_hyperparams = {
            "model_id": model_id,
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
        default=["Global_Influenza/W"],
        help="Dataset name(s)",
    )
    parser.add_argument(
        "--terms",
        type=str,
        nargs="+",
        default=None,
        choices=["short", "medium", "long"],
        help="Terms to evaluate. If not specified, auto-detect from config.",
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
        default=DEFAULT_QUANTILE_LEVELS,
        help="Quantile levels to predict",
    )

    parser.add_argument(
        "--config", type=str, default=None, help="Path to datasets.yaml config file"
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
                config_path=config_path,
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
