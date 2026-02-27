"""
Kairos model experiments for time series forecasting.

Usage:
    python experiments/kairos_model.py
    python experiments/kairos_model.py --model-size 50m
    python experiments/kairos_model.py --model-id "mldi-lab/Kairos_50m"
    python experiments/kairos_model.py --dataset "SG_Weather/D" --terms short medium long
    python experiments/kairos_model.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/kairos_model.py --dataset all_datasets  # Run all datasets from config
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

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

from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.utils import get_available_terms, clean_nan_target

load_dotenv()

DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _prepare_context(series, target_length, pad_value=float('nan')):
    if isinstance(series, torch.Tensor):
        series_t = series.float()
    else:
        series_t = torch.tensor(series, dtype=torch.float32)

    current_length = series_t.shape[-1]

    if current_length < target_length:
        padding_size = target_length - current_length
        return torch.nn.functional.pad(series_t, (padding_size, 0), mode='constant', value=pad_value)
    else:
        return series_t[..., -target_length:]


def run_kairos_experiment(
    dataset_name: str,
    terms: list[str] | None = None,
    model_id: str = "mldi-lab/Kairos_50m",
    output_dir: str | None = None,
    batch_size: int = 16,
    context_length: int = 2048,
    config_path: Path | None = None,
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
        output_dir = "./output/results/kairos"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_id}")
    print(f"Terms: {terms}")
    print(f"{'='*60}")

    print(f"  Loading Kairos model ({model_id})...")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    for term in terms:
        print(f"\n--- Term: {term} ---")
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

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
        print("  Preparing input batches...")
        for item in eval_data.input:
            target = clean_nan_target(np.asarray(item["target"]))

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
                quantiles = outputs["prediction_outputs"].detach().cpu().float().numpy()
                raw_predictions.append(quantiles)

                if start % (batch_size * 5) == 0:
                    sys.stdout.write(f"\r    Processed {end}/{total_items} items...")
                    sys.stdout.flush()
        print(f"\r    Processed {total_items}/{total_items} items. Done.")

        flat_preds = np.concatenate(raw_predictions, axis=0)

        fc_quantiles = flat_preds.astype(np.float32, copy=False)
        ds_config = f"{dataset_name}/{term}"
        model_hyperparams = {
            "model_id": model_id,
            "context_length": context_length,
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
        print(f"  Completed: {metadata['num_series']} series x {metadata['num_windows']} windows")
        print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Kairos experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["Global_Influenza/W"],
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
        default="base",
        choices=["small", "base", "large", "10m", "23m", "50m"],
        help="Kairos model size (maps to HF ID)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Kairos model HF ID (overrides --model-size)",
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
        "--context-length",
        type=int,
        default=2048,
        help="Maximum context length",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to datasets.yaml config file",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=DEFAULT_QUANTILE_LEVELS,
        help="Override quantile levels (must match model output)",
    )

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
                config_path=config_path,
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
