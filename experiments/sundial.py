"""
Sundial model experiments for time series forecasting.

Usage:
    python experiments/sundial.py
    python experiments/sundial.py --model-size base
    python experiments/sundial.py --model-id "thuml/sundial-base-128m"
    python experiments/sundial.py --dataset "SG_Weather/D" --terms short medium long
    python experiments/sundial.py --dataset all_datasets  # Run all datasets from config
    python experiments/sundial.py --val  # Evaluate on validation data (no saving)
"""

import argparse
import os
import sys
from pathlib import Path
import traceback

# Ensure timebench is importable
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

# Load environment variables
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


class SundialForecast:
    """Wrapper to make Sundial output compatible with TimeBench/GluonTS evaluation"""
    def __init__(self, samples):
        # samples shape: (num_samples, target_dim, prediction_length)
        # or (num_samples, prediction_length) if univariate
        self._samples = samples
        self._mean = np.mean(samples, axis=0)

    @property
    def samples(self):
        return self._samples

    @property
    def mean(self):
        return self._mean

    def cpu(self):
        return self


class MockPredictor:
    """Mock predictor to pass pre-computed forecasts to save_window_predictions"""
    def __init__(self, forecasts):
        self.forecasts = forecasts

    def predict(self, dataset_input, **kwargs):
        return self.forecasts


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

def run_sundial_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] | None = None,
    model_id: str = "thuml/sundial-base-128m",
    output_dir: str | None = None,
    batch_size: int = 16,
    num_samples: int = 100,  # Default aligned to 100 to prevent broadcasting errors in saver
    context_length: int = 2880,
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
):
    """
    Run Sundial model experiments on a dataset with specified terms.
    """
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset configuration
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    # Auto-detect available terms from config if not specified
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if output_dir is None:
        model_slug = model_id.split('/')[-1]
        output_dir = f"./output/results/sundial_{model_slug}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Terms: {terms}")
    print(f"Evaluation on: {'Validation data (no saving)' if use_val else 'Test data'}")
    print(f"{'='*60}")

    # Initialize Model
    print(f"  Initializing Sundial model ({model_id})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True
    ).to(device)
    model.eval()

    for term in terms:
        print(f"\n--- Term: {term} ---")

        # Get settings from config
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        # Sundial uses univariate inputs.
        to_univariate = False if Dataset(name=dataset_name, term=term,to_univariate=False).target_dim == 1 else True
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate,
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )

        # Calculate actual test/val length
        if use_val:
            data_length = val_length
            num_windows = dataset.val_windows
            split_name = "Val split"
            eval_data = dataset.val_data
        else:
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

        # Generate predictions
        data_type = "validation" if use_val else "test"
        print(f"  Running predictions on {data_type} data...")

        forecasts = []

        # Prepare input batches
        all_inputs = []

        print("    Preparing input batches...")
        for i, item in enumerate(eval_data.input):
            target = item["target"]
            target = _clean_nan_target(np.asarray(target))

            if target.ndim > 1:
                target = target.squeeze(0)
            ts = torch.tensor(target, dtype=torch.float32)
            # Prepare context (Crop/Pad to exactly context_length for batching)
            ts = _prepare_context(ts, context_length)
            all_inputs.append(ts)

        # Batched Inference
        total_items = len(all_inputs)
        raw_predictions = []

        steps = range(0, total_items, batch_size)

        with torch.no_grad():
            for idx, start in enumerate(steps):
                end = min(start + batch_size, total_items)
                batch_seqs = torch.stack(all_inputs[start:end]).to(device)

                # Model Inference
                # Pass num_samples here as required by Sundial
                batch_out = model.generate(
                    batch_seqs,
                    max_new_tokens=prediction_length,
                    num_samples=num_samples
                )

                # Output handling: Sundial usually returns tensor or tuple
                if isinstance(batch_out, tuple):
                    batch_out = batch_out[0]

                # Ensure output is on CPU numpy
                raw_predictions.append(batch_out.cpu().numpy())

                if idx % 10 == 0:
                    sys.stdout.write(f"\r    Processed {end}/{total_items} items...")
                    sys.stdout.flush()
        print(f"\r    Processed {total_items}/{total_items} items. Done.")

        # Reconstruct Forecast Objects
        # Combine all batches
        flat_preds = np.concatenate(raw_predictions, axis=0)


        for pred in flat_preds:
            forecasts.append(SundialForecast(pred))

        # if to_univariate:
        #     for pred in flat_preds:
        #         forecasts.append(SundialForecast(pred))
        # else:
        #     current_idx = 0
        #     vars_per_instance = dataset.target_dim
        #     num_instances = len(eval_data.input)

        #     for _ in range(num_instances):
        #         instance_chunk = flat_preds[current_idx : current_idx + vars_per_instance]
        #         current_idx += vars_per_instance

        #         # Format: (samples, prediction_length, variates) if multivariate
        #         if vars_per_instance == 1:
        #             # Chunk shape: (1, num_samples, pred_len) -> (num_samples, pred_len)
        #             final_pred = instance_chunk[0]
        #         else:
        #             # Chunk shape: (vars, num_samples, pred_len) -> (num_samples, pred_len, vars)
        #             final_pred = np.transpose(instance_chunk, (1, 2, 0))

        #         forecasts.append(SundialForecast(final_pred))

        num_total_instances = len(forecasts)
        num_series = num_total_instances // num_windows

        print(f"    Total instances: {num_total_instances}, Series: {num_series}, Windows: {num_windows}")

        if use_val:
            print("    (No results saved - validation data used for hyperparameter selection)")
        else:
            ds_config = f"{dataset_name}/{term}"

            model_hyperparams = {
                "model_id": model_id,
                "context_length": context_length,
                "num_samples": num_samples
            }

            predictor = MockPredictor(forecasts)

            # Correct call: Do NOT pass num_samples to save_window_predictions
            metadata = save_window_predictions(
                dataset=dataset,
                predictor=predictor,
                ds_config=ds_config,
                output_base_dir=output_dir,
                seasonality=season_length,
                model_hyperparams=model_hyperparams,
            )
            print(f"  Completed: {metadata['num_series']} series × {metadata['num_windows']} windows")
            print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Sundial experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["SG_Weather/D"],
                        help="Dataset name(s)")
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--model-size", type=str, default="base",
                        choices=["base"],
                        help="Sundial model size (maps to HF ID)")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Sundial model HF ID (overrides --model-size)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for prediction")

    # IMPORTANT: Default set to 100 to match TimeBench/Moirai defaults
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples for probabilistic forecasting")

    # UPDATED: Default set to 4000 to match Moirai
    parser.add_argument("--context-length", type=int, default=2880,
                        help="Maximum context length (lookback)")
    parser.add_argument("--cuda-device", type=str, default="0",
                        help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--val", action="store_true",
                        help="Evaluate on validation data")

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
                cuda_device=args.cuda_device,
                config_path=config_path,
                use_val=args.val,
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
