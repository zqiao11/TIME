"""
FlowState model inference script for time series forecasting.

Usage:
    python experiments/flowstate.py
    python experiments/flowstate.py --model-id "ibm-granite/granite-timeseries-flowstate-r1"
    python experiments/flowstate.py --dataset "TSBench_IMOS_v2/15T" --terms short medium long
    python experiments/flowstate.py --dataset "SG_Weather/D" "SG_PM25/H"
    python experiments/flowstate.py --dataset all_datasets
    python experiments/flowstate.py --val
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from pandas.tseries.frequencies import to_offset

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality

from timebench.evaluation.saver import save_window_quantile_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)

load_dotenv()

DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BASE_SEASONALITY = 24.0


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


class FlowStateForecast:
    """
    Wrap FlowState quantile outputs to match TimeBench Forecast interface.
    """
    def __init__(self, quantiles, quantile_levels: list[float]):
        q_data = np.asarray(quantiles, dtype=np.float32)
        if q_data.ndim == 2:
            q_data = q_data[:, np.newaxis, :]
        elif q_data.ndim == 1:
            q_data = q_data[np.newaxis, np.newaxis, :]
        elif q_data.ndim != 3:
            raise ValueError(f"Unexpected FlowState quantile shape: {q_data.shape}")

        self._quantiles = q_data
        self._quantile_levels = [float(q) for q in quantile_levels]
        self._samples = q_data

    @property
    def samples(self):
        return self._samples

    @property
    def mean(self):
        if 0.5 in self._quantile_levels:
            return self.quantile(0.5)
        return np.mean(self._quantiles, axis=0)

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


def _normalize_freq(freq: str) -> str:
    try:
        return str(to_offset(freq).name).upper()
    except Exception:
        return str(freq).upper()


def _infer_scale_factor(
    freq: str,
    season_length: int,
    override: float | None,
    daily_has_weekly_cycle: bool,
    seasonality_override: float | None,
) -> tuple[float, str]:
    if override is not None:
        return float(override), "manual"
    if seasonality_override is not None and seasonality_override > 0:
        return BASE_SEASONALITY / float(seasonality_override), "seasonality_override"
    norm = _normalize_freq(freq)
    if norm in ("15T", "15MIN"):
        return 0.25, "recommended_table"
    if norm in ("30T", "30MIN"):
        return 0.5, "recommended_table"
    if norm == "H":
        return 1.0, "recommended_table"
    if norm in ("D", "B"):
        return (3.43 if daily_has_weekly_cycle else 0.0656), "recommended_table"
    if norm == "W" or norm.startswith("W-"):
        return 0.46, "recommended_table"
    if norm == "M":
        return 2.0, "recommended_table"
    if season_length and season_length > 0:
        return BASE_SEASONALITY / float(season_length), "seasonality"
    return 1.0, "fallback"


def _extract_quantiles(quantile_outputs: np.ndarray, prediction_length: int) -> np.ndarray:
    """
    Normalize FlowState quantile outputs to shape (num_quantiles, prediction_length).
    """
    outputs = np.asarray(quantile_outputs)
    if outputs.ndim == 5:
        # (num_channels, batch, quantiles, prediction_length, 1)
        outputs = outputs[0, 0]
        if outputs.shape[-1] == 1:
            outputs = outputs[..., 0]
    elif outputs.ndim == 4:
        # (batch, quantiles, prediction_length, 1) or (quantiles, prediction_length, 1)
        if outputs.shape[-1] == 1:
            outputs = outputs[..., 0]
        if outputs.shape[0] == 1 and outputs.ndim == 3:
            outputs = outputs[0]
    elif outputs.ndim == 3:
        # (quantiles, prediction_length, 1) or (batch, prediction_length, 1)
        if outputs.shape[-1] == 1:
            outputs = outputs[..., 0]
        if outputs.shape[0] == 1 and outputs.ndim == 2:
            outputs = outputs[0]
    elif outputs.ndim != 2:
        raise ValueError(f"Unsupported FlowState quantile output shape: {outputs.shape}")

    if outputs.shape[0] != prediction_length and outputs.shape[1] == prediction_length:
        return outputs
    if outputs.shape[1] != prediction_length and outputs.shape[0] == prediction_length:
        return outputs.T
    if outputs.shape[-1] != prediction_length:
        raise ValueError(
            "FlowState output length mismatch: "
            f"expected {prediction_length}, got {outputs.shape[-1]}"
        )
    return outputs


def _extract_point_outputs(prediction_outputs: np.ndarray, prediction_length: int) -> np.ndarray:
    outputs = np.asarray(prediction_outputs)
    if outputs.ndim == 4:
        # (num_channels, batch, prediction_length, 1) or (batch, prediction_length, 1)
        outputs = outputs[0]
        if outputs.shape[-1] == 1:
            outputs = outputs[..., 0]
    elif outputs.ndim == 3:
        if outputs.shape[-1] == 1:
            outputs = outputs[..., 0]
        if outputs.shape[0] == 1:
            outputs = outputs[0]
    elif outputs.ndim != 2 and outputs.ndim != 1:
        raise ValueError(f"Unsupported FlowState point output shape: {outputs.shape}")

    if outputs.ndim == 1:
        outputs = outputs[np.newaxis, :]
    if outputs.shape[-1] != prediction_length:
        raise ValueError(
            "FlowState prediction length mismatch: "
            f"expected {prediction_length}, got {outputs.shape[-1]}"
        )
    return outputs


def run_flowstate_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] = None,
    model_id: str = "ibm-granite/granite-timeseries-flowstate-r1",
    output_dir: str | None = None,
    batch_size: int = 1,
    context_length: int = 2048,
    scale_factor: float | None = None,
    daily_has_weekly_cycle: bool = True,
    seasonality_override: float | None = None,
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
    quantile_levels: list[float] | None = None,
):
    """
    Run FlowState model inference on a dataset with specified terms.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = ["short", "medium", "long"]

    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILE_LEVELS

    if output_dir is None:
        output_dir = "./output/results/flowstate"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_id}")
    print(f"Terms: {terms}")
    print(f"Device: {device}")
    print(f"Evaluation on: {'Validation data (no saving)' if use_val else 'Test data'}")
    print(f"{'='*60}")

    print(f"Loading FlowState model ({model_id}) on {device}...")
    try:
        from tsfm_public import FlowStateForPrediction
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to import FlowStateForPrediction: {e}")
        return

    try:
        predictor = FlowStateForPrediction.from_pretrained(model_id).to(device)
        predictor.eval()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model {model_id}: {e}")
        return

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
            to_univariate=True,
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

        season_length = get_seasonality(dataset.freq)
        term_scale, scale_source = _infer_scale_factor(
            dataset.freq,
            season_length,
            scale_factor,
            daily_has_weekly_cycle,
            seasonality_override,
        )
        model_quantiles = getattr(predictor.config, "quantiles", None)
        if model_quantiles and quantile_levels != model_quantiles:
            print("    Overriding quantile_levels to match model config.")
            quantile_levels = [float(q) for q in model_quantiles]

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim} (Forced Univariate for FlowState)")
        print(f"    - {split_name}: {data_length} steps")
        print(f"    - Windows: {num_windows}")
        print(f"    - Scale factor: {term_scale:.4f} ({scale_source})")

        if season_length and prediction_length > season_length * 30:
            print("    Warning: prediction_length exceeds 30 seasonal cycles.")

        print(f"  Running predictions (Batch size: {batch_size}, sequential)...")

        flat_forecasts = []
        total_items = len(eval_data.input)

        for i, d in enumerate(eval_data.input):
            target = np.asarray(d["target"])
            if np.isnan(target).any():
                target = np.nan_to_num(target, nan=0.0)
            if target.ndim > 1:
                target = target.squeeze()
            if context_length and len(target) > context_length:
                target = target[-context_length:]
            target = _impute_nans_1d(target)

            input_tensor = torch.tensor(target, device=device).view(-1, 1, 1)

            with torch.no_grad():
                try:
                    forecast = predictor(
                        input_tensor,
                        scale_factor=term_scale,
                        prediction_length=prediction_length,
                        batch_first=False,
                    )
                    if getattr(forecast, "quantile_outputs", None) is not None:
                        outputs = forecast.quantile_outputs
                        outputs = outputs.detach().cpu().float().numpy()
                        q_values = _extract_quantiles(outputs, prediction_length)
                    else:
                        outputs = forecast.prediction_outputs
                        outputs = outputs.detach().cpu().float().numpy()
                        point_values = _extract_point_outputs(outputs, prediction_length)
                        q_values = point_values
                        if quantile_levels != [0.5]:
                            print("    Quantile outputs unavailable; using 0.5 only.")
                            quantile_levels = [0.5]
                    if q_values.shape[0] != len(quantile_levels) and q_values.shape[1] == len(quantile_levels):
                        q_values = q_values.T
                    if q_values.shape[0] != len(quantile_levels):
                        raise ValueError(
                            "Quantile count mismatch: "
                            f"model={q_values.shape[0]}, expected={len(quantile_levels)}"
                        )
                except Exception as e:
                    print(f"    Error in inference at index {i}: {e}")
                    q_values = np.zeros((len(quantile_levels), prediction_length), dtype=np.float32)

            flat_forecasts.append(
                FlowStateForecast(q_values, quantile_levels=quantile_levels)
            )

            if (i + 1) % 100 == 0 or (i + 1) == total_items:
                print(f"    Processed {i + 1}/{total_items} series...", end="\r")

        print("")

        num_total_instances = len(flat_forecasts)
        num_series_log = num_total_instances // num_windows if num_windows > 0 else 0
        print(f"    Total instances: {num_total_instances}, Series: {num_series_log}, Windows: {num_windows}")

        if use_val:
            print("    (No results saved - validation data used for hyperparameter selection)")
        else:
            ds_config = f"{dataset_name}/{term}"
            model_hyperparams = {
                "model": model_id,
                "context_length": context_length,
                "prediction_length": prediction_length,
                "scale_factor": term_scale,
                "quantile_levels": quantile_levels,
            }

            mock_predictor = MockPredictor(flat_forecasts)
            metadata = save_window_quantile_predictions(
                dataset=dataset,
                predictor=mock_predictor,
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
    parser = argparse.ArgumentParser(description="Run FlowState experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["TSBench_IMOS_v2/15T"],
                        help="Dataset name(s). 'all_datasets' for all.")
    parser.add_argument("--terms", type=str, nargs="+", default=["short", "medium", "long"],
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate")
    parser.add_argument("--model-id", type=str, default="ibm-granite/granite-timeseries-flowstate-r1",
                        help="HuggingFace model id for FlowState")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (sequential mode)")
    parser.add_argument("--context-length", type=int, default=2048,
                        help="Maximum context length")
    parser.add_argument("--scale-factor", type=float, default=None,
                        help="Override scale factor (default uses base_seasonality/seasonality)")
    parser.add_argument("--daily-no-weekly-cycle", action="store_true",
                        help="For daily data, use 0.0656 instead of 3.43")
    parser.add_argument("--seasonality", type=float, default=None,
                        help="Override seasonality N and use scale_factor=24/N")
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=DEFAULT_QUANTILE_LEVELS,
        help="Quantile levels to predict",
    )
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
            run_flowstate_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                model_id=args.model_id,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                context_length=args.context_length,
                scale_factor=args.scale_factor,
                daily_has_weekly_cycle=not args.daily_no_weekly_cycle,
                seasonality_override=args.seasonality,
                quantile_levels=args.quantiles,
                cuda_device=args.cuda_device,
                config_path=config_path,
                use_val=args.val,
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
