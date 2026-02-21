"""
Chronos-Bolt model experiments for time series forecasting.

Usage:
    python experiments/chronos_bolt.py
    python experiments/chronos_bolt.py --model-size base
    python experiments/chronos_bolt.py --dataset "TSBench_IMOS_v2/15T" --terms short medium long
    python experiments/chronos_bolt.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/chronos_bolt.py --dataset all_datasets  # Run all datasets from config
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality

# Chronos Import
from chronos import BaseChronosPipeline

from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.utils import get_available_terms
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)

# Load environment variables
load_dotenv()


def run_chronos_bolt_experiment(
    dataset_name: str,
    terms: list[str] = None,
    model_size: str = "base",
    output_dir: str | None = None,
    batch_size: int = 32,
    context_length: int = 512,
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

    if output_dir is None:
        output_dir = f"./output/results/chronos_bolt_{model_size}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: amazon/chronos-bolt-{model_size}")
    print(f"Terms: {terms}")
    print(f"{'='*60}")

    model_name = f"amazon/chronos-bolt-{model_size}"
    print(f"Loading Chronos-Bolt model: {model_name}...")

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map=device_map,
    )

    for term in terms:
        print(f"\n--- Term: {term} ---")
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        # Chronos-Bolt only supports univariate forecasting
        to_univariate = False if Dataset(name=dataset_name, term=term,to_univariate=False).target_dim == 1 else True
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate,
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
        # Running Inference
        # ---------------------------------------------------------
        print(f"  Preparing input batches from {split_name} data...")

        # eval_data is an iterable of dictionaries. Each item has 'target' (history) and 'label' (ground truth).
        # Since to_univariate=True, the dataset is already flattened to univariate instances
        all_inputs = []

        for inp, label in eval_data:
            # Chronos-Bolt expects 1D numpy arrays for univariate
            history = inp["target"]

            # Truncate context if needed
            if context_length is not None and len(history) > context_length:
                history = history[-context_length:]

            all_inputs.append(history)

        num_total_instances = len(all_inputs)
        print(f"    Total instances to forecast: {num_total_instances}")

        # Initialize Results Lists
        fc_quantiles = []

        # Run Batch Prediction
        print(f"  Running predictions (Batch size: {batch_size})...")

        for i in range(0, num_total_instances, batch_size):
            batch_inputs = all_inputs[i : i + batch_size]
            # Convert to torch tensors for Chronos-Bolt
            batch_contexts = [torch.from_numpy(inp).float() for inp in batch_inputs]

            quantiles, _ = pipeline.predict_quantiles(
                batch_contexts,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels
            )

            batch_q_list = []
            for q in quantiles:
                q_np = q.cpu().float().numpy() if isinstance(q, torch.Tensor) else q
                if (
                    q_np.ndim == 2
                    and q_np.shape[0] == prediction_length
                    and q_np.shape[1] == len(quantile_levels)
                ):
                    q_np = q_np.T
                # q_np shape: (num_quantiles, prediction_length)
                batch_q_list.append(q_np[np.newaxis, ...])
            batch_q_array = np.concatenate(batch_q_list, axis=0)
            fc_quantiles.append(batch_q_array)

            if (i + batch_size) % (batch_size * 10) == 0:
                print(f"    Processed {min(i + batch_size, num_total_instances)}/{num_total_instances}...")

        # Concatenate all batches into a single array
        # Shape: (num_total_instances, num_quantiles, prediction_length) for univariate
        # or (num_total_instances, num_quantiles, num_variates, prediction_length) for multivariate
        fc_quantiles = np.concatenate(fc_quantiles, axis=0)

        # ---------------------------------------------------------
        # Saving Results
        # ---------------------------------------------------------
        ds_config = f"{dataset_name}/{term}"
        model_hyperparams = {
            "model": f"chronos-bolt-{model_size}",
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
        print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Chronos-Bolt experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["Global_Influenza/W"],
                        help="Dataset name(s). 'all_datasets' for all.")
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--model-size", type=str, default="base",
                        choices=["tiny", "mini", "small", "base"],
                        help="Chronos-Bolt model size")
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
    parser.add_argument("--context-length", type=int, default=2048,
                        help="Maximum context length")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")

    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
    else:
        datasets = args.dataset

    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{len(datasets)}: {dataset_name}")
        print(f"{'#'*60}")

        try:
            run_chronos_bolt_experiment(
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
    print("# All datasets completed!")
    print(f"{'#'*60}")

if __name__ == "__main__":
    main()
