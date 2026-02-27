"""
Toto model experiments for time series forecasting.
Reference: https://github.com/DataDog/toto/blob/main/toto/notebooks/inference_tutorial.ipynb

Usage:
    python experiments/toto_model.py
    python experiments/toto_model.py --model-id "Datadog/Toto-Open-Base-1.0"
    python experiments/toto_model.py --model-size base
    python experiments/toto_model.py --dataset "TSBench_IMOS_v2/15T" --terms short medium long
    python experiments/toto_model.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/toto_model.py --dataset all_datasets  # Run all datasets from config
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality
from pandas.tseries.frequencies import to_offset

from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config
from timebench.evaluation.utils import get_available_terms, clean_nan_target

load_dotenv()

DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _freq_to_seconds(freq: str) -> float:
    try:
        offset = to_offset(freq)
        if getattr(offset, "delta", None) is not None:
            return float(offset.delta.total_seconds())
    except Exception:
        pass
    return 0.0


def _prepare_series(series: np.ndarray, context_length: int) -> np.ndarray:
    if series.ndim == 1:
        series = series[np.newaxis, :]
    if series.shape[-1] > context_length:
        series = series[..., -context_length:]
    return series


def _build_masked_timeseries(
    series: np.ndarray,
    device: str,
    interval_seconds: float,
) -> MaskedTimeseries:
    series_t = torch.as_tensor(series, dtype=torch.float32, device=device)
    padding_mask_t = torch.ones_like(series_t, dtype=torch.bool, device=device)
    id_mask = torch.zeros_like(series_t)
    timestamp_seconds = torch.zeros(series_t.shape, dtype=torch.float32, device=device)
    time_interval_seconds = torch.full(
        (series_t.shape[0],),
        interval_seconds,
        dtype=torch.float32,
        device=device,
    )
    return MaskedTimeseries(
        series=series_t,
        padding_mask=padding_mask_t,
        id_mask=id_mask,
        timestamp_seconds=timestamp_seconds,
        time_interval_seconds=time_interval_seconds,
    )


def run_toto_experiment(
    dataset_name: str,
    terms: list[str] | None = None,
    model_id: str = "Datadog/Toto-Open-Base-1.0",
    output_dir: str | None = None,
    num_samples: int = 100,
    samples_per_batch: int = 100,
    context_length: int = 4000,
    config_path: Path | None = None,
    compile_model: bool = True,
    quantile_levels: list[float] = DEFAULT_QUANTILE_LEVELS,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if output_dir is None:
        output_dir = "./output/results/toto"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    print(f"  Initializing Toto model ({model_id})...")
    toto = Toto.from_pretrained(model_id)
    toto.to(device)

    if compile_model:
        try:
            toto.compile()
        except Exception as exc:
            print(f"  Warning: model compile failed, continuing without compile: {exc}")

    forecaster = TotoForecaster(toto.model)

    for term in terms:
        print(f"\n--- Term: {term} ---")

        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(
            "  Config: prediction_length="
            f"{prediction_length}, test_length={test_length}, val_length={val_length}"
        )

        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=False,
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
        print(f"    - Target dim: {dataset.target_dim}")
        print(
            "    - Series length: "
            f"min={dataset._min_series_length}, "
            f"max={dataset._max_series_length}, "
            f"avg={dataset._avg_series_length:.1f}"
        )
        print(f"    - {split_name}: {data_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        freq = dataset.freq.lower()
        season_length = get_seasonality(freq)
        interval_seconds = _freq_to_seconds(freq)

        print(f"  Running predictions on {split_name.lower()} data...")
        fc_quantiles = []

        with torch.no_grad():
            for idx, item in enumerate(eval_data.input, 1):
                target = np.asarray(item["target"])
                target = _prepare_series(target, context_length)
                target = clean_nan_target(target)
                inputs = _build_masked_timeseries(
                    target, device, interval_seconds
                )

                fc = forecaster.forecast(
                    inputs,
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                    samples_per_batch=samples_per_batch,
                )

                q_preds = []
                for q in quantile_levels:
                    q_pred = fc.quantile(q).detach().cpu().numpy()[0]
                    q_preds.append(q_pred)

                fc_quantiles.append(np.stack(q_preds, axis=0))

                if idx % 100 == 0:
                    print(f"    Processed {idx} windows")

        ds_config = f"{dataset_name}/{term}"
        fc_quantiles = np.stack(fc_quantiles, axis=0).astype(np.float32, copy=False)
        model_hyperparams = {
            "model_id": model_id,
            "context_length": context_length,
            "requested_num_samples": num_samples,
            "samples_per_batch": samples_per_batch,
        }

        save_window_predictions(
            dataset=dataset,
            fc_quantiles=fc_quantiles,
            ds_config=ds_config,
            output_base_dir=output_dir,
            seasonality=season_length,
            model_hyperparams=model_hyperparams,
            quantile_levels=quantile_levels,
        )

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Toto experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["ECDC_COVID/W"],
        help="Dataset name(s). 'all_datasets' for all.",
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
        "--model-size",
        type=str,
        default=None,
        choices=["base"],
        help="Model size alias (maps to a model id)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Datadog/Toto-Open-Base-1.0",
        help="Toto model ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples for probabilistic forecasting",
    )
    parser.add_argument(
        "--samples-per-batch",
        type=int,
        default=100,
        help="Samples per batch (controls memory during inference)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=4096,
        help="Maximum context length",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to datasets.yaml config file",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch compile for Toto",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=DEFAULT_QUANTILE_LEVELS,
        help="Quantile levels to predict",
    )

    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    model_id = args.model_id
    if args.model_size:
        model_size_map = {
            "base": "Datadog/Toto-Open-Base-1.0",
        }
        model_id = model_size_map[args.model_size]

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
            run_toto_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                model_id=model_id,
                output_dir=args.output_dir,
                num_samples=args.num_samples,
                samples_per_batch=args.samples_per_batch,
                context_length=args.context_length,
                config_path=config_path,
                compile_model=not args.no_compile,
                quantile_levels=args.quantiles,
            )
        except Exception as exc:
            print(f"ERROR: Failed to run experiment for {dataset_name}: {exc}")
            traceback.print_exc()
            continue

    print(f"\n{'#'*60}")
    print(f"# All {total_datasets} dataset(s) completed!")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
