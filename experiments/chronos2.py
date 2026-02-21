"""
Chronos-2 model experiments for time series forecasting.
(Refactored to match Moirai interface, logging format, and evaluation logic)

Usage:
    python experiments/chronos2.py
    python experiments/chronos2.py --model-size chronos2
    python experiments/chronos2.py --dataset "TSBench_IMOS_v2/15T" --terms short medium long
    python experiments/chronos2.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/chronos2.py --dataset all_datasets  # Run all datasets from config
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

from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.utils import get_available_terms
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)

# Load environment variables
load_dotenv()

logging.getLogger("chronos").setLevel(logging.ERROR)


def run_chronos2_experiment(
    dataset_name: str,
    terms: list[str] = None,
    model_size: str = "chronos2",
    output_dir: str | None = None,
    batch_size: int = 32,
    context_length: int = 2048,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
):
    """
    Run Chronos-2 model experiments.
    """
    # Set CUDA device
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
        output_dir = "./output/results/chronos2_base"

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
        # Helper function to prepare a single context
        def _prepare_context(d):
            target = np.asarray(d["target"])

            # Manually truncate context
            seq_len = target.shape[-1]
            if seq_len > context_length:
                target = target[..., -context_length:]

            if target.ndim == 1:
                target = target[np.newaxis, :]

            return torch.tensor(target)

        # Batch Inference with lazy loading
        fc_quantiles_batches = []
        eval_input_list = list(eval_data.input)  # Convert to list for indexing
        total_items = len(eval_input_list)


        for start in range(0, total_items, batch_size):
            end = min(start + batch_size, total_items)
            # Load context only for current batch
            batch_contexts = [_prepare_context(eval_input_list[i]) for i in range(start, end)]

            # Filter out verbose warnings from Chronos-2 during prediction to keep output clean
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

            batch_quantiles_list = []
            for q in batch_q:
                if isinstance(q, torch.Tensor):
                    if q.ndim == 3 and q.shape[-1] == len(quantile_levels):
                        # Shape: (num_variates, pred_len, num_quantiles) -> (num_quantiles, num_variates, pred_len)
                        q = q.permute(2, 0, 1)
                    q = q.cpu().float().numpy()
                # q shape: (num_quantiles, num_variates, prediction_length)
                # Add batch dimension: (1, num_quantiles, num_variates, prediction_length)
                batch_quantiles_list.append(q[np.newaxis, ...])


            # Stack into batch: (batch_size, num_quantiles, num_variates, prediction_length)
            batch_q_array = np.concatenate(batch_quantiles_list, axis=0)
            fc_quantiles_batches.append(batch_q_array)

            # Optional progress logging
            if (start // batch_size + 1) % 10 == 0:
                print(f"    Processed {min(start + batch_size, total_items)}/{total_items}...")

        # Concatenate all batches into a single array
        # Shape: (num_total_instances, num_quantiles, num_variates, prediction_length)
        fc_quantiles = np.concatenate(fc_quantiles_batches, axis=0)

        # ---------------------------------------------------------
        # 3. Saving Results
        # ---------------------------------------------------------
        ds_config = f"{dataset_name}/{term}"
        model_hyperparams = {
            "model": f"chronos2-{model_size}",
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

        print(f"  Completed: {metadata['num_series']} series × {metadata['num_windows']} windows")
        print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Chronos experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["Global_Influenza/W"],
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

    # Iterate over all datasets with progress logging
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
