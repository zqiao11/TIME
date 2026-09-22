"""t0-beta experiments for the TIME benchmark.

This module evaluates The Forecasting Company's t0-beta model with the
PyTorch ``tfc-t0`` runtime.  It follows TIME's native evaluation contract:
produce quantile forecasts for each rolling test window and persist them with
``save_window_predictions``.

Examples:
    python experiments/t0_beta.py --dataset "SG_Weather/D" --terms short medium long
    python experiments/t0_beta.py --dataset "SG_Weather/D" "SG_PM25/H"
    python experiments/t0_beta.py --dataset all_datasets
"""

import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality
from t0 import T0Forecaster, TimeSeries

from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config
from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.utils import get_available_terms

load_dotenv()

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_ID = "theforecastingcompany/t0-beta"
DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _select_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _select_dtype(dtype_name: str, device: torch.device) -> torch.dtype | None:
    if dtype_name == "auto":
        return torch.bfloat16 if device.type == "cuda" else None
    if dtype_name == "float32":
        return None
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]


def _target_to_context(target: object, context_length: int | None) -> np.ndarray:
    context = np.asarray(target, dtype=np.float32)
    if context.ndim == 0:
        raise ValueError("Expected a 1-D or 2-D target array, got scalar target")
    if context.ndim > 2:
        raise ValueError(f"Expected a 1-D or 2-D target array, got shape {context.shape}")
    if context_length is not None:
        context = context[..., -context_length:]
    return np.ascontiguousarray(context)


def _context_batch_to_timeseries(contexts: list[np.ndarray]) -> TimeSeries:
    context_tensor = torch.as_tensor(np.stack(contexts), dtype=torch.float32)
    return TimeSeries.from_array(context_tensor)


def _predict_quantiles(
    model: T0Forecaster,
    eval_input: list[dict],
    *,
    prediction_length: int,
    quantile_levels: list[float],
    batch_size: int,
    context_length: int | None,
) -> np.ndarray:
    device = next(model.parameters()).device
    contexts = [_target_to_context(entry["target"], context_length) for entry in eval_input]
    indices_by_shape: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for idx, context in enumerate(contexts):
        indices_by_shape[context.shape].append(idx)

    fc_quantiles: list[np.ndarray | None] = [None] * len(contexts)
    processed = 0
    for indices in indices_by_shape.values():
        start = 0
        current_batch_size = batch_size
        while start < len(indices):
            batch_indices = indices[start : start + current_batch_size]
            try:
                # Batch only equal-shaped contexts. This avoids extra left padding
                # beyond t0's own patch alignment and matches the benchmark path
                # used for the reported numbers.
                model_input = _context_batch_to_timeseries([contexts[i] for i in batch_indices])
                with torch.inference_mode():
                    forecast = model.predict(
                        model_input,
                        horizon=prediction_length,
                        quantile_levels=quantile_levels,
                    )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                is_oom = "out of memory" in str(err).lower()
                if not is_oom or current_batch_size <= 1:
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                LOGGER.warning("Out of memory; reducing batch size to %d", current_batch_size)
                continue

            quantiles = forecast.quantiles.detach().cpu().numpy().astype(np.float32, copy=False)
            target_dim = contexts[batch_indices[0]].shape[0] if contexts[batch_indices[0]].ndim == 2 else 1
            row_start = 0
            for original_idx in batch_indices:
                row_stop = row_start + target_dim
                item_quantiles = quantiles[row_start:row_stop]
                if target_dim == 1:
                    # t0: (1, prediction_length, quantiles)
                    # TIME univariate: (quantiles, prediction_length)
                    fc_quantiles[original_idx] = np.transpose(item_quantiles[0], (1, 0))
                else:
                    # t0: (variates, prediction_length, quantiles)
                    # TIME multivariate: (quantiles, variates, prediction_length)
                    fc_quantiles[original_idx] = np.transpose(item_quantiles, (2, 0, 1))
                row_start = row_stop

            processed += len(batch_indices)
            start += len(batch_indices)
            if processed % (10 * batch_size) == 0 or processed == len(contexts):
                print(f"    Processed {processed}/{len(contexts)} instances")

    if any(quantiles is None for quantiles in fc_quantiles):
        raise RuntimeError("Internal error: not all forecasts were produced")
    return np.stack(fc_quantiles)


def run_t0_beta_experiment(
    dataset_name: str,
    terms: list[str] | None = None,
    output_dir: str | None = None,
    batch_size: int = 32,
    context_length: int | None = 8192,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "auto",
    torch_dtype: str = "auto",
) -> None:
    """Run t0-beta on one TIME dataset across one or more forecast terms."""
    config = load_dataset_config(config_path)
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    quantile_levels = quantile_levels or DEFAULT_QUANTILE_LEVELS
    output_dir = output_dir or "./output/results/t0_beta"
    os.makedirs(output_dir, exist_ok=True)

    selected_device = _select_device(device)
    selected_dtype = _select_dtype(torch_dtype, selected_device)

    print(f"\n{'=' * 60}")
    print(f"Model: {model_id}")
    print(f"Device: {selected_device}")
    print(f"Dataset: {dataset_name}")
    print(f"Terms: {terms}")
    print(f"{'=' * 60}")

    model = T0Forecaster.from_pretrained(model_id, dtype=selected_dtype).to(selected_device).eval()

    for term in terms:
        print(f"\n--- Term: {term} ---")
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(
            "  Config: "
            f"prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}"
        )

        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=False,
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(
            "    - Series length: "
            f"min={dataset._min_series_length}, max={dataset._max_series_length}, "
            f"avg={dataset._avg_series_length:.1f}"
        )
        print(f"    - Test split: {test_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {dataset.windows}")

        eval_input = list(dataset.test_data.input)
        fc_quantiles = _predict_quantiles(
            model,
            eval_input,
            prediction_length=dataset.prediction_length,
            quantile_levels=quantile_levels,
            batch_size=batch_size,
            context_length=context_length,
        )

        season_length = get_seasonality(dataset.freq)
        metadata = save_window_predictions(
            dataset=dataset,
            fc_quantiles=fc_quantiles,
            ds_config=f"{dataset_name}/{term}",
            output_base_dir=output_dir,
            seasonality=season_length,
            model_hyperparams={
                "model": "t0-beta",
                "model_id": model_id,
                "runtime": "tfc-t0==0.5.0",
                "batch_size": batch_size,
                "context_length_limit": context_length,
            },
            quantile_levels=quantile_levels,
        )
        print(f"  Completed: {metadata['num_series']} series × {metadata['num_windows']} windows")
        print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'=' * 60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run t0-beta on TIME benchmark datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["Port_Activity/D"],
        help="Dataset name(s), or 'all_datasets' to run every dataset from the config.",
    )
    parser.add_argument(
        "--terms",
        type=str,
        nargs="+",
        default=None,
        choices=["short", "medium", "long"],
        help="Terms to evaluate. If omitted, terms are read from the dataset config.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument(
        "--context-length",
        type=int,
        default=8192,
        help="Right-aligned historical context length. Use <=0 for full available history.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to datasets.yaml config file")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    parser.add_argument("--device", type=str, default="auto", help="Torch device, e.g. auto, cuda, mps, cpu")
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "bfloat16", "float16"],
        help="Autocast dtype passed to t0. 'float32' disables autocast.",
    )
    parser.add_argument(
        "--quantile-levels",
        type=float,
        nargs="+",
        default=DEFAULT_QUANTILE_LEVELS,
        help="Quantile levels to save for TIME evaluation.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    config_path = Path(args.config) if args.config else None
    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
        print(f"Running all {len(datasets)} datasets from config")
    else:
        datasets = args.dataset

    context_length = args.context_length if args.context_length and args.context_length > 0 else None
    for index, dataset_name in enumerate(datasets, start=1):
        print(f"\n{'#' * 60}")
        print(f"# Dataset {index}/{len(datasets)}: {dataset_name}")
        print(f"{'#' * 60}")
        try:
            run_t0_beta_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                context_length=context_length,
                config_path=config_path,
                quantile_levels=args.quantile_levels,
                model_id=args.model_id,
                device=args.device,
                torch_dtype=args.torch_dtype,
            )
        except Exception:
            LOGGER.exception("Failed to run t0-beta for %s", dataset_name)
            continue

    print(f"\n{'#' * 60}")
    print(f"# All {len(datasets)} dataset(s) completed")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
