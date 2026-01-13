"""
Kairos model experiments for time series forecasting.

Usage:
    python experiments/kairos_model.py
    python experiments/kairos_model.py --model-size 50m
    python experiments/kairos_model.py --model-id "mldi-lab/Kairos_50m"
    python experiments/kairos_model.py --dataset "SG_Weather/D" --terms short medium long
    python experiments/kairos_model.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/kairos_model.py --dataset all_datasets  # Run all datasets from config
    python experiments/kairos_model.py --val  # Evaluate on validation data (no saving)
"""

import argparse
import os
import sys
from pathlib import Path
import traceback

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Ensure Kairos tsfm is importable without shadowing HF datasets
KAIROS_DIR = Path(__file__).parent / "Kairos"
if str(KAIROS_DIR) not in sys.path:
    sys.path.append(str(KAIROS_DIR))

import numpy as np
import torch
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality

from tsfm.model.kairos import AutoModel

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


def _prepare_context(series, target_length):
    """
    Ensure series is exactly target_length for batching.
    - If longer, crop to the last `target_length` points (standard context window).
    - If shorter, keep original length (no padding).
    """
    if isinstance(series, torch.Tensor):
        series_t = series.float()
    else:
        series_t = torch.tensor(series, dtype=torch.float32)

    if series_t.shape[-1] >= target_length:
        return series_t[..., -target_length:]

    return series_t


def _normalize_quantile_output(quantiles: np.ndarray, prediction_length: int) -> np.ndarray:
    q_data = quantiles
    if isinstance(q_data, torch.Tensor):
        q_data = q_data.detach().cpu().float().numpy()
    q = np.asarray(q_data)
    if q.ndim != 3:
        raise ValueError(f"Unexpected quantile output shape: {q.shape}")
    if q.shape[2] == prediction_length:
        return q
    if q.shape[1] == prediction_length:
        return q.transpose(0, 2, 1)
    raise ValueError(
        "Quantile output length mismatch: "
        f"expected {prediction_length}, got {q.shape}"
    )


def _resolve_quantile_levels(num_quantiles: int, quantile_levels: list[float] | None) -> list[float]:
    if quantile_levels is None or len(quantile_levels) != num_quantiles:
        quantile_levels = np.linspace(0.1, 0.9, num_quantiles)
    return [float(q) for q in quantile_levels]


class KairosForecast:
    def __init__(self, quantiles, quantile_levels):
        q_data = quantiles
        if isinstance(q_data, torch.Tensor):
            q_data = q_data.detach().cpu().float().numpy()
        q_data = np.asarray(q_data, dtype=np.float32)

        levels = [float(q) for q in quantile_levels]
        num_levels = len(levels)
        if q_data.ndim == 2:
            q_data = q_data[:, np.newaxis, :]
        elif q_data.ndim == 3:
            if q_data.shape[0] != num_levels and q_data.shape[1] == num_levels:
                q_data = q_data.transpose(1, 0, 2)
        else:
            raise ValueError(f"Unexpected quantile shape: {q_data.shape}")

        self._quantiles = q_data
        self._quantile_levels = levels
        if 0.5 in self._quantile_levels:
            idx = int(np.where(np.isclose(self._quantile_levels, 0.5, atol=1e-6))[0][0])
            self._mean = self._quantiles[idx]
        else:
            self._mean = np.mean(self._quantiles, axis=0)

    @property
    def mean(self):
        return self._mean

    @property
    def quantile_levels(self):
        return self._quantile_levels

    def quantile(self, q: float):
        q_levels = np.asarray(self._quantile_levels, dtype=float)
        matches = np.where(np.isclose(q_levels, q, atol=1e-6))[0]
        if matches.size == 0:
            raise ValueError(f"Quantile {q} not available. Supported: {self._quantile_levels}")
        return self._quantiles[int(matches[0])]

    def cpu(self):
        return self


class MockPredictor:
    def __init__(self, forecasts):
        self.forecasts = forecasts

    def predict(self, dataset_input, **kwargs):
        return self.forecasts


def run_kairos_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] | None = None,
    model_id: str = "mldi-lab/Kairos_50m",
    output_dir: str | None = None,
    batch_size: int = 16,
    context_length: int = 2048,
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
    quantile_levels: list[float] | None = None,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = ["short", "medium", "long"]

    if output_dir is None:
        model_slug = model_id.split("/")[-1]
        output_dir = f"./output/results/kairos_{model_slug}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_id}")
    print(f"Terms: {terms}")
    print(f"Evaluation on: {'Validation data (no saving)' if use_val else 'Test data'}")
    print(f"{'='*60}")

    print(f"  Loading Kairos model ({model_id})...")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    if quantile_levels is None:
        quantile_levels = getattr(getattr(model, "config", None), "quantiles", None)

    for term in terms:
        print(f"\n--- Term: {term} ---")
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        to_univariate = True
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate,
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )

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

        forecasts = []
        all_inputs = []
        print("  Preparing input batches...")
        for item in eval_data.input:
            target = _clean_nan_target(np.asarray(item["target"]))

            if dataset.target_dim > 1 and target.ndim == 2:
                for v in range(target.shape[0]):
                    ts = torch.tensor(target[v], dtype=torch.float32)
                    ts = _prepare_context(ts, context_length)
                    all_inputs.append(ts)
            else:
                if target.ndim == 2:
                    target = target.squeeze(0)
                ts = torch.tensor(target, dtype=torch.float32)
                ts = _prepare_context(ts, context_length)
                all_inputs.append(ts)

        total_items = len(all_inputs)
        raw_predictions = []
        print(f"  Running predictions on {split_name.lower()} data...")

        with torch.no_grad():
            for start in range(0, total_items, batch_size):
                end = min(start + batch_size, total_items)
                batch_seqs = torch.stack(all_inputs[start:end]).to(device)
                outputs = model(
                    past_target=batch_seqs,
                    prediction_length=prediction_length,
                    generation=True,
                    preserve_positivity=True,
                    average_with_flipped_input=True,
                )
                quantiles = outputs.get("prediction_outputs")
                if quantiles is None:
                    raise ValueError("Kairos output missing 'prediction_outputs'")
                quantiles = _normalize_quantile_output(quantiles, prediction_length)
                raw_predictions.append(quantiles)

                if start % (batch_size * 5) == 0:
                    sys.stdout.write(f"\r    Processed {end}/{total_items} items...")
                    sys.stdout.flush()
        print(f"\r    Processed {total_items}/{total_items} items. Done.")

        flat_preds = np.concatenate(raw_predictions, axis=0) if raw_predictions else np.empty((0, 0, 0))
        if flat_preds.size == 0:
            raise ValueError("No predictions were generated.")

        quantile_levels = _resolve_quantile_levels(flat_preds.shape[1], quantile_levels)

        num_instances = len(eval_data.input)
        if to_univariate:
            for idx in range(num_instances):
                q_stack = flat_preds[idx][:, np.newaxis, :]
                forecasts.append(KairosForecast(q_stack, quantile_levels))
        else:
            current_idx = 0
            vars_per_instance = dataset.target_dim

            for _ in range(num_instances):
                instance_chunk = flat_preds[current_idx: current_idx + vars_per_instance]
                current_idx += vars_per_instance

                if vars_per_instance == 1:
                    q_stack = instance_chunk[0][:, np.newaxis, :]
                else:
                    q_stack = np.transpose(instance_chunk, (1, 0, 2))

                forecasts.append(KairosForecast(q_stack, quantile_levels))

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
                "quantile_levels": quantile_levels,
            }

            predictor = MockPredictor(forecasts)
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
    parser = argparse.ArgumentParser(description="Run Kairos experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["SG_Weather/D"],
                        help="Dataset name(s). 'all_datasets' for all.")
    parser.add_argument("--terms", type=str, nargs="+", default=["short", "medium", "long"],
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate")
    parser.add_argument("--model-size", type=str, default="base",
                        choices=["small", "base", "large", "10m", "23m", "50m"],
                        help="Kairos model size (maps to HF ID)")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Kairos model HF ID (overrides --model-size)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for prediction")
    parser.add_argument("--context-length", type=int, default=2048,
                        help="Maximum context length")
    parser.add_argument("--cuda-device", type=str, default="0",
                        help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--val", action="store_true",
                        help="Evaluate on validation data (no saving)")
    parser.add_argument("--quantiles", type=float, nargs="+", default=None,
                        help="Override quantile levels (must match model output)")

    args = parser.parse_args()
    config_path = Path(args.config) if args.config else None

    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
        print(f"Running all {len(datasets)} datasets from config:")
        for ds in datasets:
            print(f"  - {ds}")
    else:
        datasets = args.dataset

    model_size_map = {
        "large": "mldi-lab/Kairos_50m",
        "base": "mldi-lab/Kairos_23m",
        "small": "mldi-lab/Kairos_10m",
        "50m": "mldi-lab/Kairos_50m",
        "23m": "mldi-lab/Kairos_23m",
        "10m": "mldi-lab/Kairos_10m",
    }

    model_id = args.model_id or model_size_map.get(args.model_size)
    if model_id is None:
        raise ValueError(f"Unsupported Kairos model size: {args.model_size}")

    total_datasets = len(datasets)
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        try:
            run_kairos_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                model_id=model_id,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                context_length=args.context_length,
                cuda_device=args.cuda_device,
                config_path=config_path,
                use_val=args.val,
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
