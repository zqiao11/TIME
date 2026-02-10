"""
Moirai-2.0 model experiments for time series forecasting.

Usage:
    python experiments/moriai2.py
    python experiments/moriai2.py --model-size small
    python experiments/moriai2.py --dataset "Coastal_T_S/5T" --terms short medium long
    python experiments/moriai2.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/moriai2.py --dataset all_datasets  # Run all datasets from config
    python experiments/moriai2.py --val  # Evaluate on validation data (no saving)
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


def _prepare_entry(entry: dict, context_length: int | None) -> dict:
    target = np.asarray(entry["target"], dtype=np.float32)
    if target.ndim == 2 and target.shape[0] > target.shape[1]:
        target = target.T
    if context_length is not None and target.shape[-1] > context_length:
        target = target[..., -context_length:]
    target = _clean_nan_target(target)

    cleaned = dict(entry)
    cleaned["target"] = target

    for key in ("past_feat_dynamic_real", "feat_dynamic_real"):
        if key not in cleaned:
            continue
        feat = np.asarray(cleaned[key], dtype=np.float32)
        if feat.ndim == 2 and feat.shape[0] > feat.shape[1]:
            feat = feat.T
        if context_length is not None and feat.shape[-1] > context_length:
            feat = feat[..., -context_length:]
        cleaned[key] = feat

    return cleaned


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


def run_moirai2_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] = None,
    model_size: str = "base",
    output_dir: str | None = None,
    batch_size: int = 16,
    context_length: int = 1680,
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
    quantile_levels: list[float] | None = None,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

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
        output_dir = f"./output/results/moirai2_{model_size_used}"

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

        if use_val:
            num_windows = dataset.val_windows
        else:
            num_windows = dataset.windows

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        predictor = model.create_predictor(batch_size=batch_size)
        season_length = get_seasonality(dataset.freq)

        print(f"  Running predictions...")
        # processed_inputs = [_prepare_entry(entry, context_length) for entry in dataset.test_data.input]
        # forecasts = list(predictor.predict(processed_inputs))

        if use_val:
            print("    (No results saved - validation data used for hyperparameter selection)")
        else:
            ds_config = f"{dataset_name}/{term}"
            model_hyperparams = {
                "model_version": "2.0",
                "model_size": model_size_used,
                "context_length": context_length,
                "quantile_levels": quantile_levels,
            }

            metadata = save_window_quantile_predictions(
                dataset=dataset,
                predictor=predictor,
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
    parser.add_argument("--dataset", type=str, nargs="+", default=["ECDC_COVID/D"],
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
    parser.add_argument("--cuda-device", type=str, default="0", help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None, help="Path to datasets.yaml")
    parser.add_argument("--val", action="store_true", help="Evaluate on validation data")

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
                cuda_device=args.cuda_device,
                config_path=config_path,
                use_val=args.val,
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