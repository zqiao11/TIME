"""
Sundial model experiments for time series forecasting.

Usage:
    python experiments/sundial.py
    python experiments/sundial.py --model-size base
    python experiments/sundial.py --model-id "thuml/sundial-base-128m"
    python experiments/sundial.py --dataset "SG_Weather/D" --terms short medium long
    python experiments/sundial.py --dataset all_datasets  # Run all datasets from config
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality
from transformers import AutoModelForCausalLM

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.utils import get_available_terms, clean_nan_target

load_dotenv()


def _prepare_context(series, target_length):
    """
    Ensure series is exactly target_length for batching.
    - If longer, crop to the last `target_length` points (standard context window).
    - If shorter, left pad (required for torch.stack).
    """
    if isinstance(series, torch.Tensor):
        series_t = series.float()
    else:
        series_t = torch.tensor(series, dtype=torch.float32)

    if series_t.shape[-1] >= target_length:
        return series_t[..., -target_length:]

    pad_len = target_length - series_t.shape[-1]
    pad = torch.zeros((*series_t.shape[:-1], pad_len), dtype=series_t.dtype)
    return torch.cat([pad, series_t], dim=-1)


def _normalize_samples_array(samples: np.ndarray, prediction_length: int) -> np.ndarray:
    """
    Normalize Sundial output to shape:
        (num_instances, num_samples, prediction_length)
    """
    samples = np.asarray(samples)

    if samples.ndim == 2:
        # Single instance: (num_samples, prediction_length)
        if samples.shape[1] != prediction_length:
            raise ValueError(
                f"Expected samples shape (num_samples, prediction_length), got {samples.shape}"
            )
        return samples[np.newaxis, ...]

    if samples.ndim == 3:
        # (num_instances, num_samples, prediction_length)
        if samples.shape[2] != prediction_length:
            raise ValueError(
                f"Expected samples shape (num_instances, num_samples, prediction_length), got {samples.shape}"
            )
        return samples

    raise ValueError(
        f"Unsupported samples ndim: {samples.ndim}, shape: {samples.shape}"
    )


def run_sundial_experiment(
    dataset_name: str,
    terms: list[str] | None = None,
    model_id: str = "thuml/sundial-base-128m",
    output_dir: str | None = None,
    batch_size: int = 16,
    num_samples: int = 100,
    context_length: int = 2880,
    quantile_levels: list[float] | None = None,
    config_path: Path | None = None,
):
    """
    Run Sundial model experiments on a dataset with specified terms.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if output_dir is None:
        output_dir = f"./output/results/sundial_base"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Terms: {terms}")
    print(f"{'='*60}")

    print(f"  Initializing Sundial model ({model_id})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
    ).to(device)
    model.eval()

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

        season_length = get_seasonality(dataset.freq)

        all_inputs = []

        print("    Preparing input batches...")
        for i, item in enumerate(eval_data.input):
            target = item["target"]
            target = clean_nan_target(np.asarray(target))

            if target.ndim > 1:
                target = target.squeeze(0)
            ts = torch.tensor(target, dtype=torch.float32)
            ts = _prepare_context(ts, context_length)
            all_inputs.append(ts)

        total_items = len(all_inputs)
        raw_predictions = []

        steps = range(0, total_items, batch_size)

        with torch.no_grad():
            for idx, start in enumerate(steps):
                end = min(start + batch_size, total_items)
                batch_seqs = torch.stack(all_inputs[start:end]).to(device)

                batch_out = model.generate(
                    batch_seqs,
                    max_new_tokens=prediction_length,
                    num_samples=num_samples,
                )

                if isinstance(batch_out, tuple):
                    batch_out = batch_out[0]

                raw_predictions.append(batch_out.cpu().numpy())

                if idx % 10 == 0:
                    sys.stdout.write(f"\r    Processed {end}/{total_items} items...")
                    sys.stdout.flush()
        print(f"\r    Processed {total_items}/{total_items} items. Done.")

        flat_preds = np.concatenate(raw_predictions, axis=0)

        if quantile_levels is None:
            quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        quantile_levels_array = np.asarray(quantile_levels, dtype=float)

        samples = _normalize_samples_array(flat_preds, prediction_length)
        fc_quantiles = np.quantile(samples, quantile_levels_array, axis=1)
        fc_quantiles = np.moveaxis(fc_quantiles, 0, 1).astype(np.float32, copy=False)

        ds_config = f"{dataset_name}/{term}"

        model_hyperparams = {
            "model_id": model_id,
            "context_length": context_length,
            "num_samples": num_samples,
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
    parser = argparse.ArgumentParser(description="Run Sundial experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["SG_Weather/D"],
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
        "--model-size",
        type=str,
        default="base",
        choices=["base"],
        help="Sundial model size (maps to HF ID)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Sundial model HF ID (overrides --model-size)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for prediction",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,  # Their demo in Gift-Eval set it as 20, but we use 100 for more stable results
        help="Number of samples for probabilistic forecasting",
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=2880,
        help="Maximum context length (lookback)",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Quantile levels to compute from samples",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to datasets.yaml config file",
    )

    args = parser.parse_args()

    model_id_map = {
        "base": "thuml/sundial-base-128m",
    }
    model_id = args.model_id or model_id_map.get(args.model_size)
    if model_id is None:
        raise ValueError(f"Unsupported Sundial model size: {args.model_size}")

    config_path = Path(args.config) if args.config else None

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
            run_sundial_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                model_id=model_id,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                context_length=args.context_length,
                quantile_levels=args.quantiles,
                config_path=config_path,
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
