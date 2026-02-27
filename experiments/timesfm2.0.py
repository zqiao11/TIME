"""
TimesFM-2.0 model experiments for time series forecasting.

Usage:
    python experiments/timesfm2.0.py
    python experiments/timesfm2.0.py --model-size base
    python experiments/timesfm2.0.py --dataset "TSBench_IMOS_v2/15T" --terms short medium long
    python experiments/timesfm2.0.py --dataset "SG_Weather/D" "SG_PM25/H"
    python experiments/timesfm2.0.py --dataset all_datasets
    python experiments/timesfm2.0.py --val
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Prefer local TimesFM v1 source if available (TimesFM-2.0 API).
TIMESFM_SRC = Path(__file__).resolve().parents[1] / "timesfm" / "v1" / "src"
if TIMESFM_SRC.exists():
    sys.path.insert(0, str(TIMESFM_SRC))

import numpy as np
import torch
import timesfm
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality

from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.utils import get_available_terms
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)

load_dotenv()

DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _get_model_config(model_size: str) -> dict:
    model_map = {
        "base": {
            "repo_id": "google/timesfm-2.0-500m-pytorch",
            "num_layers": 50,
            "model_dims": 1280,
            "use_positional_embedding": False,
            "input_patch_len": 32,
            "output_patch_len": 128,
            "num_quantiles": 9,
            "default_context_len": 2048,
        },
    }
    return model_map.get(model_size, model_map["base"])


def run_timesfm_experiment(
    dataset_name: str,
    terms: list[str] = None,
    model_size: str = "base",
    output_dir: str | None = None,
    batch_size: int = 512,
    per_core_batch_size: int = 32,
    context_length: int | None = None,
    input_patch_len: int = 32,
    output_patch_len: int = 128,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
    normalize_inputs: bool = True,
):
    """
    Run TimesFM-2.0 model experiments on a dataset with specified terms.
    """
    backend = "gpu" if torch.cuda.is_available() else "cpu"

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
        output_dir = "./output/results/TimesFM-2.0"

    os.makedirs(output_dir, exist_ok=True)

    model_cfg = _get_model_config(model_size)
    if context_length is None:
        context_length = model_cfg["default_context_len"]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Terms: {terms}")
    print(f"Backend: {backend}")
    print(f"{'='*60}")

    for term in terms:
        print(f"\n--- Term: {term} ---")

        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        if len(quantile_levels) != model_cfg["num_quantiles"]:
            raise ValueError(
                "TimesFM-2.0 checkpoints expect "
                f"{model_cfg['num_quantiles']} quantiles. "
                f"Got {len(quantile_levels)}."
            )
        if input_patch_len != model_cfg["input_patch_len"]:
            print(
                f"  Overriding input_patch_len={input_patch_len} to "
                f"{model_cfg['input_patch_len']} to match TimesFM-2.0 checkpoint."
            )
        if output_patch_len != model_cfg["output_patch_len"]:
            print(
                f"  Overriding output_patch_len={output_patch_len} to "
                f"{model_cfg['output_patch_len']} to match TimesFM-2.0 checkpoint."
            )
        input_patch_len = model_cfg["input_patch_len"]
        effective_output_patch_len = model_cfg["output_patch_len"]

        print(f"  Initializing TimesFM-2.0 ({model_size})...")
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=per_core_batch_size,
                horizon_len=prediction_length,
                context_len=context_length,
                input_patch_len=input_patch_len,
                output_patch_len=effective_output_patch_len,
                num_layers=model_cfg["num_layers"],
                model_dims=model_cfg["model_dims"],
                use_positional_embedding=model_cfg["use_positional_embedding"],
                quantiles=quantile_levels,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=model_cfg["repo_id"],
            ),
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
        freq_code = timesfm.freq_map(dataset.freq)

        print(f"  Preparing input batches from {split_name} data...")

        all_inputs = []
        for inp, _label in eval_data:
            history = inp["target"]
            all_inputs.append(history)

        num_total_instances = len(all_inputs)
        print(f"    Total instances to forecast: {num_total_instances}")

        # Initialize Results Lists
        fc_quantiles = []

        print(f"  Running predictions (Batch size: {batch_size})...")

        for i in range(0, num_total_instances, batch_size):
            batch_inputs = all_inputs[i : i + batch_size]
            batch_freq = [freq_code] * len(batch_inputs)

            # TimesFM forecast
            point_forecast, quantile_forecast = tfm.forecast(
                batch_inputs,
                freq=batch_freq,
                forecast_context_len=context_length,
                normalize=normalize_inputs,
            )

            processed_quantile_forecast = quantile_forecast[:, :, 1:].transpose(0, 2, 1)  # (batch, 9, horizon)
            fc_quantiles.append(processed_quantile_forecast)

            if (i + batch_size) % (batch_size * 10) == 0:
                print(f"    Processed {i + batch_size}/{num_total_instances}...")

        fc_quantiles = np.concatenate(fc_quantiles, axis=0)  # (total_instances, 9, horizon)
        ds_config = f"{dataset_name}/{term}"

        model_hyperparams = {
            "model": "TimesFM-2.0-500M",
            "context_length": context_length,
            "horizon_len": prediction_length,
            "input_patch_len": input_patch_len,
            "output_patch_len": effective_output_patch_len,
            "num_layers": model_cfg["num_layers"],
            "model_dims": model_cfg["model_dims"],
            "use_positional_embedding": model_cfg["use_positional_embedding"],
            "backend": backend,
            "output_type": "quantiles",
            "quantile_levels": quantile_levels,
            "normalize_inputs": normalize_inputs,
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
    parser = argparse.ArgumentParser(description="Run TimesFM-2.0 experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["ECDC_COVID/W"],
                        help="Dataset name(s). 'all_datasets' for all.")
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--model-size", type=str, default="base",
                        choices=["base"],
                        help="TimesFM model size")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for prediction")
    parser.add_argument("--per-core-batch-size", type=int, default=32,
                        help="Per-core batch size for TimesFM")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Maximum context length (default per model)")
    parser.add_argument("--input-patch-len", type=int, default=32,
                        help="Input patch length for TimesFM (fixed for 2.0)")
    parser.add_argument("--output-patch-len", type=int, default=128,
                        help="Output patch length for TimesFM (fixed for 2.0)")
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=DEFAULT_QUANTILE_LEVELS,
        help="Quantile levels to predict",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")

    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None

    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
        print(f"Running all {len(datasets)} datasets from config:")
    else:
        datasets = args.dataset

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
                per_core_batch_size=args.per_core_batch_size,
                context_length=args.context_length,
                input_patch_len=args.input_patch_len,
                output_patch_len=args.output_patch_len,
                quantile_levels=args.quantiles,
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
