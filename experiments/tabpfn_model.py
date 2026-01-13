"""
TabPFN-TS model experiments for time series forecasting.

Usage:
    python experiments/tabpfn_model.py
    python experiments/tabpfn_model.py --model-size base
    python experiments/tabpfn_model.py --dataset "TSBench_IMOS_v2/15T" --terms short medium long
    python experiments/tabpfn_model.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/tabpfn_model.py --dataset all_datasets  # Run all datasets from config
    python experiments/tabpfn_model.py --val  # Evaluate on validation data (no saving)
"""

import argparse
import os
import sys
from pathlib import Path
import traceback
import warnings

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality

from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode

from timebench.evaluation.saver import save_window_quantile_predictions
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



def _to_timestamp(start_value) -> pd.Timestamp:
    if hasattr(start_value, "item"):
        start_value = start_value.item()
    if isinstance(start_value, pd.Period):
        return start_value.to_timestamp()
    try:
        return pd.Timestamp(start_value)
    except ValueError:
        return pd.Period(start_value).to_timestamp()


def _build_context_df(series: np.ndarray, timestamps, item_id: str, covariates: dict | None) -> pd.DataFrame:
    data = {
        "item_id": item_id,
        "timestamp": timestamps,
        "target": series,
    }
    if covariates:
        data.update(covariates)
    return pd.DataFrame(data)


def _build_future_df(timestamps, item_id: str, covariates: dict | None) -> pd.DataFrame:
    data = {
        "item_id": item_id,
        "timestamp": timestamps,
    }
    if covariates:
        data.update(covariates)
    return pd.DataFrame(data)


def _iter_item_groups(pred_df: pd.DataFrame):
    if "item_id" in pred_df.columns:
        for item_id, group in pred_df.groupby("item_id"):
            yield item_id, group
        return

    if isinstance(pred_df.index, pd.MultiIndex) and "item_id" in pred_df.index.names:
        for item_id, group in pred_df.groupby(level="item_id"):
            yield item_id, group
        return

    raise ValueError("TabPFN prediction output does not include item_id information.")


def _extract_prediction(group: pd.DataFrame):
    if "timestamp" in group.columns:
        group = group.sort_values("timestamp")
    else:
        group = group.sort_index()

    columns = list(group.columns)
    mean = None
    for mean_col in ("target", "mean", "median"):
        if mean_col in columns:
            mean = group[mean_col].to_numpy()
            break

    quantile_cols = []
    for col in columns:
        if col in ("item_id", "timestamp", "target", "mean", "median"):
            continue
        try:
            quantile_cols.append((float(col), col))
        except (TypeError, ValueError):
            continue

    if not quantile_cols:
        raise ValueError("No quantile columns found in TabPFN prediction output.")

    quantile_cols.sort(key=lambda x: x[0])
    quantile_levels = np.array([q for q, _ in quantile_cols], dtype=np.float32)
    quantiles = group[[col for _, col in quantile_cols]].to_numpy()

    if mean is None:
        if 0.5 in quantile_levels:
            idx = int(np.where(quantile_levels == 0.5)[0][0])
            mean = quantiles[:, idx]
        else:
            mean = np.mean(quantiles, axis=1)

    return mean, quantiles, quantile_levels


class TabPFNForecast:
    def __init__(self, quantiles, quantile_levels, mean):
        q_data = np.asarray(quantiles)
        if q_data.ndim != 2:
            raise ValueError(f"Unexpected quantile shape: {q_data.shape}")

        quantile_levels = np.asarray(quantile_levels, dtype=np.float32)
        order = np.argsort(quantile_levels)
        quantile_levels = quantile_levels[order]
        q_data = q_data[:, order]

        q_data = q_data.T  # (num_quantiles, pred_len)
        self._quantiles = q_data
        self._quantile_levels = [float(q) for q in quantile_levels]
        self._samples = q_data

        mean_arr = np.asarray(mean)
        if mean_arr.ndim > 1:
            mean_arr = mean_arr.squeeze()
        self._mean = mean_arr

    @property
    def samples(self):
        return self._samples

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


class MultivariateForecast:
    def __init__(self, forecasts):
        samples = [fc.samples for fc in forecasts]
        self._samples = np.stack(samples, axis=1)
        self._mean = np.stack([fc.mean for fc in forecasts], axis=0)
        self._quantile_levels = forecasts[0].quantile_levels
        self._quantiles = np.stack([fc._quantiles for fc in forecasts], axis=1)

    @property
    def samples(self):
        return self._samples

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


def run_tabpfn_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] | None = None,
    model_size: str = "base",
    output_dir: str | None = None,
    batch_size: int = 64,
    context_length: int | None = 4000,
    num_samples: int = 100,
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = ["short", "medium", "long"]

    if output_dir is None:
        output_dir = "./output/results/tabpfn"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: TabPFN-TS ({model_size})")
    print(f"Terms: {terms}")
    print(f"Evaluation on: {'Validation data (no saving)' if use_val else 'Test data'}")
    print(f"{'='*60}")

    model_size_map = {
        "base": TabPFNMode.LOCAL,
    }
    tabpfn_mode = model_size_map.get(model_size, TabPFNMode.LOCAL)
    pipeline = TabPFNTSPipeline(tabpfn_mode=tabpfn_mode)

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

        eval_inputs = list(eval_data.input)
        total_entries = len(eval_inputs)
        print(f"  Running predictions on {split_name.lower()} data (instances: {total_entries})...")

        entry_variates = {}
        entry_num_vars = {}
        batch_frames = []
        batch_future_frames = []
        batch_meta = []
        forecasts = []
        output_quantile_levels = None

        def _flush_batch():
            if not batch_frames:
                return
            nonlocal output_quantile_levels

            context_df = pd.concat(batch_frames, ignore_index=True)
            future_df = pd.concat(batch_future_frames, ignore_index=True)
            context_df = context_df.sort_values(["item_id", "timestamp"])
            future_df = future_df.sort_values(["item_id", "timestamp"])
            context_df = context_df.dropna(axis=1, how="all")
            future_df = future_df.dropna(axis=1, how="all")
            covariate_cols = [
                col for col in context_df.columns
                if col not in ("item_id", "timestamp", "target")
            ]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                pred_df = pipeline.predict_df(
                    context_df=context_df,
                    future_df=future_df.drop(columns=covariate_cols, errors="ignore"),
                )

            pred_map = {}
            for item_id, group in _iter_item_groups(pred_df):
                pred_map[item_id] = _extract_prediction(group)

            for entry_idx, var_idx, item_id in batch_meta:
                if item_id not in pred_map:
                    raise ValueError(f"Missing predictions for item_id {item_id}")
                mean, quantiles, quantile_levels = pred_map[item_id]
                if output_quantile_levels is None:
                    output_quantile_levels = quantile_levels
                elif not np.allclose(output_quantile_levels, quantile_levels, atol=1e-6):
                    raise ValueError("Inconsistent quantile levels across TabPFN outputs.")
                entry_variates[entry_idx][var_idx] = TabPFNForecast(
                    quantiles=quantiles,
                    quantile_levels=quantile_levels,
                    mean=mean,
                )

            batch_frames.clear()
            batch_future_frames.clear()
            batch_meta.clear()

        for entry_idx, entry in enumerate(eval_inputs):
            target = np.asarray(entry["target"])
            if target.ndim == 2 and target.shape[0] > target.shape[1]:
                target = target.T
            if target.ndim == 1:
                target = target[np.newaxis, :]

            if context_length is not None and target.shape[1] > context_length:
                target = target[:, -context_length:]

            target = np.stack([_impute_nans_1d(target[v]) for v in range(target.shape[0])], axis=0)

            entry_num_vars[entry_idx] = target.shape[0]
            entry_variates[entry_idx] = [None] * target.shape[0]

            start_ts = _to_timestamp(entry["start"]) if "start" in entry else pd.Timestamp("1970-01-01")
            context_ts = pd.date_range(start=start_ts, periods=target.shape[1], freq=dataset.freq)
            future_ts = pd.date_range(start=context_ts[-1], periods=prediction_length + 1, freq=dataset.freq)[1:]

            for var_idx in range(target.shape[0]):
                series = target[var_idx]
                item_id = f"e{entry_idx}_v{var_idx}"

                covariates_ctx = {}
                for j in range(target.shape[0]):
                    if j == var_idx:
                        continue
                    name = f"covar_{j}"
                    covariates_ctx[name] = target[j]

                batch_frames.append(_build_context_df(series, context_ts, item_id, covariates_ctx))
                batch_future_frames.append(_build_future_df(future_ts, item_id, None))
                batch_meta.append((entry_idx, var_idx, item_id))

                if batch_size and len(batch_meta) >= batch_size:
                    _flush_batch()

        _flush_batch()

        for entry_idx in range(total_entries):
            var_forecasts = entry_variates[entry_idx]
            if any(fc is None for fc in var_forecasts):
                raise ValueError(f"Missing forecasts for entry {entry_idx}")
            if len(var_forecasts) == 1:
                forecasts.append(var_forecasts[0])
            else:
                forecasts.append(MultivariateForecast(var_forecasts))

        if use_val:
            print("    (No results saved - validation data used for hyperparameter selection)")
            continue

        ds_config = f"{dataset_name}/{term}"
        model_hyperparams = {
            "model": "TabPFN-TS",
            "context_length": context_length,
            "quantile_levels": output_quantile_levels.tolist() if output_quantile_levels is not None else None,
        }

        predictor = MockPredictor(forecasts)
        metadata = save_window_quantile_predictions(
            dataset=dataset,
            predictor=predictor,
            ds_config=ds_config,
            output_base_dir=output_dir,
            seasonality=season_length,
            model_hyperparams=model_hyperparams,
            quantile_levels=output_quantile_levels.tolist() if output_quantile_levels is not None else None,
        )
        print(f"  Completed: {metadata['num_series']} series × {metadata['num_windows']} windows")
        print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run TabPFN-TS experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["SG_Weather/D"],
                        help="Dataset name(s). Can be a single dataset, multiple datasets, or 'all_datasets'")
    parser.add_argument("--terms", type=str, nargs="+", default=["short", "medium", "long"],
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate")
    parser.add_argument("--model-size", type=str, default="base",
                        choices=["base"],
                        help="TabPFN-TS model size")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size (number of univariate series per batch)")
    parser.add_argument("--context-length", type=int, default=4000,
                        help="Maximum context length")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples for probabilistic forecasting")
    parser.add_argument("--cuda-device", type=str, default="0",
                        help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--val", action="store_true",
                        help="Evaluate on validation data (no saving)")

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

    total_datasets = len(datasets)
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        try:
            run_tabpfn_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                model_size=args.model_size,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                context_length=args.context_length,
                num_samples=args.num_samples,
                cuda_device=args.cuda_device,
                config_path=config_path,
                use_val=args.val,
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
