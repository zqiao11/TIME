"""
Chronos-Bolt model experiments for time series forecasting.

Usage:
    python experiments/chronos_bolt.py
    python experiments/chronos_bolt.py --model-size base
    python experiments/chronos_bolt.py --dataset "TSBench_IMOS_v2/15T" --terms short medium long
    python experiments/chronos_bolt.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/chronos_bolt.py --dataset all_datasets  # Run all datasets from config
    python experiments/chronos_bolt.py --val  # Evaluate on validation data (no saving)
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
from transformers import logging as hf_logging

# Chronos Import
from chronos import BaseChronosPipeline

from timebench.evaluation.saver import save_window_quantile_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)

# Load environment variables
load_dotenv()


# ==========================================

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


class MultivariateForecast:
    """
    Adapts Chronos-Bolt quantiles to Timebench forecast API.
    """
    def __init__(
        self,
        quantiles,
        quantile_levels: list[float],
    ):
        q_data = quantiles
        if isinstance(q_data, torch.Tensor):
            q_data = q_data.cpu().float().numpy()

        q_data = np.asarray(q_data, dtype=np.float32)
        if q_data.ndim == 2:
            q_data = q_data[:, np.newaxis, :]

        self._quantiles = q_data
        self._quantile_levels = [float(q) for q in quantile_levels]

        self._samples = q_data
        self._mean = np.mean(self._samples, axis=0)

    @property
    def samples(self):
        return self._samples

    @property
    def mean(self):
        return self._mean

    def quantile(self, q: float):
        q_levels = np.asarray(self._quantile_levels, dtype=float)
        matches = np.where(np.isclose(q_levels, q, atol=1e-6))[0]
        if matches.size == 0:
            raise ValueError(f"Quantile {q} not available. Supported: {self._quantile_levels}")
        return self._quantiles[int(matches[0])]

    def cpu(self):
        return self


class ChronosBoltPredictor:
    """
    Wrapper to make Chronos-Bolt behave like a GluonTS Predictor.
    """
    def __init__(
        self, 
        pipeline: BaseChronosPipeline, 
        prediction_length: int, 
        quantile_levels: list[float],
        batch_size: int = 32,
        context_length: int = 512
    ):
        self.pipeline = pipeline
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.context_length = context_length
        self.quantile_levels = quantile_levels

    def predict(self, dataset, **kwargs):
        """
        Iterates over the dataset, batches inputs, and yields forecast objects.
        """
        flat_contexts = []
        instance_dims = []

        for entry in dataset:
            target = entry["target"]

            if isinstance(target, torch.Tensor):
                target_np = target.detach().cpu().numpy()
            else:
                target_np = np.asarray(target)

            target_np = target_np.astype(np.float32, copy=False)

            if target_np.ndim == 1:
                target_np = target_np[np.newaxis, :]
            elif target_np.ndim == 2 and target_np.shape[0] > target_np.shape[1]:
                target_np = target_np.T

            if self.context_length is not None and target_np.shape[-1] > self.context_length:
                target_np = target_np[..., -self.context_length:]

            target_np = _clean_nan_target(target_np)

            num_vars = target_np.shape[0]
            for v in range(num_vars):
                flat_contexts.append(torch.from_numpy(target_np[v]).float())
            instance_dims.append(num_vars)

        flat_quantiles = []
        for start in range(0, len(flat_contexts), self.batch_size):
            end = min(start + self.batch_size, len(flat_contexts))
            flat_quantiles.extend(self._process_batch(flat_contexts[start:end]))

        forecasts = []
        cursor = 0
        for dim in instance_dims:
            component_q = flat_quantiles[cursor: cursor + dim]
            cursor += dim

            q_list = []
            for q in component_q:
                q_np = q.cpu().float().numpy() if isinstance(q, torch.Tensor) else q
                if (
                    q_np.ndim == 2
                    and q_np.shape[0] == self.prediction_length
                    and q_np.shape[1] == len(self.quantile_levels)
                ):
                    q_np = q_np.T
                q_list.append(q_np)

            q_stack = np.stack(q_list, axis=1) if q_list else np.empty((0, 0, 0))
            forecasts.append(
                MultivariateForecast(
                    q_stack,
                    quantile_levels=self.quantile_levels,
                )
            )

        for fc in forecasts:
            yield fc

    def _process_batch(self, contexts):

        # 在这个上下文里，所有的 stderr 输出都会经过 ContentFilterStderr
        quantiles, _ = self.pipeline.predict_quantiles(
            contexts,
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels
        )

        if isinstance(quantiles, torch.Tensor):
            if quantiles.ndim == 3 and quantiles.shape[-1] == len(self.quantile_levels):
                quantiles = quantiles.permute(0, 2, 1)
            return [quantiles[i] for i in range(quantiles.shape[0])]
        return quantiles


def run_chronos_bolt_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] = None,
    model_size: str = "base",
    output_dir: str | None = None,
    batch_size: int = 32,
    context_length: int = 512, 
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
    quantile_levels: list[float] | None = None,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = ["short", "medium", "long"]
    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if output_dir is None:
        output_dir = f"./output/results/chronos_bolt_{model_size}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: amazon/chronos-bolt-{model_size}")
    print(f"Terms: {terms}")
    print(f"Evaluation on: {'Validation data (no saving)' if use_val else 'Test data'}")
    print(f"{'='*60}")

    model_name = f"amazon/chronos-bolt-{model_size}"
    print(f"Loading Chronos-Bolt model: {model_name}...")
    
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )

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
            eval_data = dataset.val_data
        else:
            data_length = test_length
            num_windows = dataset.windows
            eval_data = dataset.test_data

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        predictor = ChronosBoltPredictor(
            pipeline=pipeline,
            prediction_length=dataset.prediction_length,
            quantile_levels=quantile_levels,
            batch_size=batch_size,
            context_length=context_length
        )
        
        season_length = get_seasonality(dataset.freq)

        print(f"  Running predictions...")
        forecasts = list(predictor.predict(eval_data.input))

        num_total_instances = len(forecasts)
        num_series = num_total_instances // num_windows
        num_variates = dataset.target_dim

        print(f"    Total instances: {num_total_instances}, Series: {num_series}, Windows: {num_windows}")
        print("    Collecting ground truth and context...")
        
        ground_truths = []
        contexts = []
        for inp, label in eval_data:
            ground_truths.append(label["target"])
            contexts.append(inp["target"])

        if not to_univariate:
            actual_num_samples = (
                forecasts[0].samples.shape[0]
                if len(forecasts) > 0
                else len(predictor.quantile_levels)
            )

            predictions_mean = np.zeros((num_series, num_windows, num_variates, prediction_length))
            predictions_samples = np.zeros((num_series, num_windows, actual_num_samples, num_variates, prediction_length))
            ground_truth = np.zeros((num_series, num_windows, num_variates, prediction_length))

            max_context_len = max(ctx.shape[-1] for ctx in contexts)
            max_context_len = min(max_context_len, context_length * 2)
            context_array = np.full((num_series, num_windows, num_variates, max_context_len), np.nan)

            print("    Organizing data into arrays...")
            for idx, (fc, gt, ctx) in enumerate(zip(forecasts, ground_truths, contexts)):
                series_idx = idx // num_windows
                window_idx = idx % num_windows

                fc_mean = fc.mean
                fc_samples = fc.samples

                if fc_mean.ndim == 1:
                    fc_mean = fc_mean[np.newaxis, :]

                if fc_samples.ndim == 2:
                    fc_samples = fc_samples[:, np.newaxis, :]

                if gt.ndim == 1:
                    gt = gt[np.newaxis, :]
                elif gt.shape[0] == prediction_length and gt.shape[1] == num_variates:
                    gt = gt.T

                if ctx.ndim == 1:
                    ctx = ctx[np.newaxis, :]
                elif ctx.shape[0] != num_variates:
                    ctx = ctx.T

                predictions_mean[series_idx, window_idx] = fc_mean
                predictions_samples[series_idx, window_idx] = fc_samples
                ground_truth[series_idx, window_idx] = gt

                ctx_len = min(ctx.shape[-1], max_context_len)
                context_array[series_idx, window_idx, :, :ctx_len] = ctx[:, -ctx_len:]

        if use_val:
            print("    (No results saved - validation data used for hyperparameter selection)")
        else:
            ds_config = f"{dataset_name}/{term}"
            model_hyperparams = {
                "model": f"chronos-bolt-{model_size}",
                "context_length": context_length,
                "quantile_levels": predictor.quantile_levels,
            }

            metadata = save_window_quantile_predictions(
                dataset=dataset,
                predictor=predictor,
                ds_config=ds_config,
                output_base_dir=output_dir,
                seasonality=season_length,
                model_hyperparams=model_hyperparams,
            )
            print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Chronos-Bolt experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["SG_Weather/D"],
                        help="Dataset name(s). 'all_datasets' for all.")
    parser.add_argument("--terms", type=str, nargs="+", default=["short", "medium", "long"],
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate")
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
    parser.add_argument("--context-length", type=int, default=4000,
                        help="Maximum context length")
    parser.add_argument("--cuda-device", type=str, default="0",
                        help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--val", action="store_true",
                        help="Evaluate on validation data")

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
    print(f"# All datasets completed!")
    print(f"{'#'*60}")

if __name__ == "__main__":
    main()
