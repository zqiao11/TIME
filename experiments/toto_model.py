"""
Toto model experiments for time series forecasting.

Usage:
    python experiments/toto_model.py
    python experiments/toto_model.py --model-id "Datadog/Toto-Open-Base-1.0"
    python experiments/toto_model.py --model-size base
    python experiments/toto_model.py --dataset "TSBench_IMOS_v2/15T" --terms short medium long
    python experiments/toto_model.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/toto_model.py --dataset all_datasets  # Run all datasets from config
    python experiments/toto_model.py --val  # Evaluate on validation data (no saving)
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
from pandas.tseries.frequencies import to_offset

from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config

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


def _freq_to_seconds(freq: str) -> float:
    try:
        offset = to_offset(freq)
        if getattr(offset, "delta", None) is not None:
            return float(offset.delta.total_seconds())
    except Exception:
        pass
    return 0.0


def _prepare_series(series: np.ndarray, context_length: int | None) -> np.ndarray:
    if series.ndim == 1:
        series = series[np.newaxis, :]
    if context_length is not None and series.shape[-1] > context_length:
        series = series[..., -context_length:]
    return series


def _build_masked_timeseries(series: np.ndarray, device: str, interval_seconds: float) -> MaskedTimeseries:
    series_t = torch.as_tensor(series, dtype=torch.float32, device=device)
    padding_mask = torch.ones_like(series_t, dtype=torch.bool)
    id_mask = torch.zeros_like(series_t)
    timestamp_seconds = torch.zeros_like(series_t)
    time_interval_seconds = torch.full(
        (series_t.shape[0],),
        interval_seconds,
        dtype=torch.float32,
        device=device,
    )
    return MaskedTimeseries(
        series=series_t,
        padding_mask=padding_mask,
        id_mask=id_mask,
        timestamp_seconds=timestamp_seconds,
        time_interval_seconds=time_interval_seconds,
    )


def _normalize_samples(samples: np.ndarray, prediction_length: int, num_variates: int) -> np.ndarray:
    if isinstance(samples, (list, tuple)):
        if len(samples) == 0:
            raise ValueError("Toto returned empty samples list.")
        elems = []
        for s in samples:
            if isinstance(s, torch.Tensor):
                elems.append(s.detach().cpu().numpy())
            else:
                elems.append(np.asarray(s))
        if len(elems) == 1 and elems[0].ndim >= 2:
            samples = elems[0]
        else:
            first = elems[0]
            try:
                if first.ndim == 1:
                    samples = np.stack(elems, axis=0)
                elif first.ndim == 2 and len(elems) == num_variates and first.shape[1] == prediction_length:
                    samples = np.stack(elems, axis=1)
                elif first.ndim == 2 and len(elems) == num_variates and first.shape[0] == prediction_length:
                    samples = np.stack(elems, axis=1).transpose(2, 1, 0)
                else:
                    samples = np.stack(elems, axis=0)
            except ValueError as exc:
                raise ValueError(f"Toto samples list has inconsistent shapes: {exc}") from exc

    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    else:
        samples = np.asarray(samples)

    if samples.dtype == object:
        samples = np.stack([np.asarray(s) for s in samples], axis=0)

    if samples.ndim == 4:
        # Demo output: (batch, variates, pred_len, num_samples)
        if samples.shape[0] == 1:
            samples = samples.squeeze(0)
        else:
            raise ValueError(f"Unexpected Toto batch dimension: {samples.shape}")

    if samples.ndim == 3:
        # (variates, pred_len, num_samples) -> (num_samples, variates, pred_len)
        if samples.shape[0] == num_variates and samples.shape[1] == prediction_length:
            return samples.transpose(2, 0, 1)
        # (variates, num_samples, pred_len) -> (num_samples, variates, pred_len)
        if samples.shape[0] == num_variates and samples.shape[2] == prediction_length:
            return samples.transpose(1, 0, 2)
        # (pred_len, variates, num_samples) -> (num_samples, variates, pred_len)
        if samples.shape[0] == prediction_length and samples.shape[1] == num_variates:
            return samples.transpose(2, 1, 0)
        # (pred_len, num_samples, variates) -> (num_samples, variates, pred_len)
        if samples.shape[0] == prediction_length and samples.shape[2] == num_variates:
            return samples.transpose(1, 2, 0)
        # (num_samples, variates, pred_len)
        if samples.shape[1] == num_variates and samples.shape[2] == prediction_length:
            return samples
        # (num_samples, pred_len, variates) -> (num_samples, variates, pred_len)
        if samples.shape[1] == prediction_length and samples.shape[2] == num_variates:
            return samples.transpose(0, 2, 1)

    if samples.ndim > 3:
        squeezed = np.squeeze(samples)
        if squeezed.ndim <= 3:
            samples = squeezed
        else:
            raise ValueError(f"Unexpected Toto samples shape: {samples.shape}")

    if samples.ndim == 2:
        if samples.shape[1] != prediction_length:
            raise ValueError("Unexpected Toto samples shape for univariate output.")
        return samples[:, np.newaxis, :]

    if samples.ndim != 3:
        raise ValueError("Unexpected Toto samples shape; expected 2D or 3D tensor.")

    raise ValueError(
        f"Unexpected Toto samples shape: {samples.shape}, "
        f"prediction_length={prediction_length}, num_variates={num_variates}"
    )


def _coerce_num_samples(samples: np.ndarray, target_num_samples: int) -> np.ndarray:
    if samples.shape[0] == target_num_samples:
        return samples
    if samples.shape[0] > target_num_samples:
        return samples[:target_num_samples]
    pad_count = target_num_samples - samples.shape[0]
    pad = np.repeat(samples[-1:], pad_count, axis=0)
    return np.concatenate([samples, pad], axis=0)


class TotoForecast:
    def __init__(self, samples: np.ndarray):
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
    def __init__(self, forecasts):
        self.forecasts = forecasts

    def predict(self, dataset_input, **kwargs):
        return self.forecasts



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

def run_toto_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] | None = None,
    model_id: str = "Datadog/Toto-Open-Base-1.0",
    output_dir: str | None = None,
    num_samples: int = 100,
    samples_per_batch: int = 100,
    context_length: int = 4000,
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
    compile_model: bool = True,
):
    """
    Run Toto model experiments on a dataset with specified terms.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading configuration...")
    config = load_dataset_config(config_path)

    # Auto-detect available terms from config if not specified
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
    print(f"Evaluation on: {'Validation data (no saving)' if use_val else 'Test data'}")
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

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=False,
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
        interval_seconds = _freq_to_seconds(dataset.freq)

        print(f"  Running predictions on {split_name.lower()} data...")
        forecasts = []

        with torch.no_grad():
            for idx, item in enumerate(eval_data.input, 1):
                target = np.asarray(item["target"])
                target = _prepare_series(target, context_length)
                target = _clean_nan_target(target)

                inputs = _build_masked_timeseries(target, device, interval_seconds)

                fc = forecaster.forecast(
                    inputs,
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                    samples_per_batch=samples_per_batch,
                )

                samples = fc.samples
                if isinstance(samples, torch.Tensor):
                    samples = samples.detach().cpu().numpy()
                samples = np.asarray(samples)

                samples = _normalize_samples(samples, prediction_length, dataset.target_dim)
                samples = _coerce_num_samples(samples, 100)

                forecasts.append(TotoForecast(samples))

                # Clear GPU cache to reduce memory fragmentation
                if device == "cuda":
                    torch.cuda.empty_cache()

                if idx % 100 == 0:
                    print(f"    Processed {idx} windows")

        if use_val:
            print("    (No results saved - validation data used for hyperparameter selection)")
            continue

        ds_config = f"{dataset_name}/{term}"
        predictor = MockPredictor(forecasts)
        model_hyperparams = {
            "model_id": model_id,
            "context_length": context_length,
            "requested_num_samples": num_samples,
            "samples_per_batch": samples_per_batch,
        }

        save_window_predictions(
            dataset=dataset,
            predictor=predictor,
            ds_config=ds_config,
            output_base_dir=output_dir,
            seasonality=season_length,
            model_hyperparams=model_hyperparams,
        )

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Toto experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["SG_Weather/D"],
                        help="Dataset name(s). 'all_datasets' for all.")
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--model-size", type=str, default=None,
                        choices=["base"],
                        help="Model size alias (maps to a model id)")
    parser.add_argument("--model-id", type=str, default="Datadog/Toto-Open-Base-1.0",
                        help="Toto model ID")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples for probabilistic forecasting")
    parser.add_argument("--samples-per-batch", type=int, default=100,
                        help="Samples per batch (controls memory during inference)")
    parser.add_argument("--context-length", type=int, default=4096,
                        help="Maximum context length")
    parser.add_argument("--cuda-device", type=str, default="0",
                        help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--val", action="store_true",
                        help="Evaluate on validation data (no saving)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch compile for Toto")

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
                cuda_device=args.cuda_device,
                config_path=config_path,
                use_val=args.val,
                compile_model=not args.no_compile,
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
