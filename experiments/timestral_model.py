"""Timestral experiments for the TIME benchmark.

Timestral (https://github.com/mistralai/timestral) is a family of forecasting models from
Mistral AI; the weights are on the Hugging Face Hub (mistralai/Timestral-{5M,28M,123M,209M}).

Protocol:
- Multivariate targets are forecast jointly (`to_univariate=False`): one window is a
  `(1, V, T)` context.
- The full available history before each window is the context; the forecaster keeps the
  most recent 32768 steps (the trained context length).
- The forecaster defaults are used.
- One forecast per window; TIME's 9 quantile levels 0.1..0.9 are taken from the model's 21.

Examples:
    python experiments/timestral_model.py --dataset all_datasets
    python experiments/timestral_model.py --model-id mistralai/Timestral-5M --dataset "SG_PM25/H" "Crypto/D"
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality
from timestral import TimestralForecaster

from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config
from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.utils import get_available_terms

load_dotenv()

DEFAULT_MODEL_ID = "mistralai/Timestral-209M"
TIME_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _batches(lengths, num_variates, patch_size, max_patches, max_batch_size):
    """Group window indices (sorted by context length) into batches under a patch budget.

    Contexts in a batch are left-padded with NaN to the longest one, which leaves the
    forecasts unchanged; sorting by length keeps the padding small.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batch = []
    for i in order:
        n_patches = math.ceil(lengths[i] / patch_size) * num_variates
        if batch and (len(batch) + 1) * n_patches > max_patches or len(batch) >= max_batch_size:
            yield batch
            batch = []
        batch.append(i)
    if batch:
        yield batch


def _forecast_batch(forecaster, contexts, prediction_length, q_idx):
    """Forecast a list of `(V, T_i)` contexts; returns `(B, Q, V, H)` (TIME's layout)."""
    t_max = max(c.shape[-1] for c in contexts)
    batch = np.full((len(contexts), contexts[0].shape[0], t_max), np.nan, dtype=np.float32)
    for j, c in enumerate(contexts):
        batch[j, :, t_max - c.shape[-1] :] = c
    try:
        fc = forecaster.forecast(batch, prediction_length)
    except torch.OutOfMemoryError:
        if len(contexts) == 1:
            raise
        torch.cuda.empty_cache()
        half = len(contexts) // 2
        return np.concatenate(
            [
                _forecast_batch(forecaster, contexts[:half], prediction_length, q_idx),
                _forecast_batch(forecaster, contexts[half:], prediction_length, q_idx),
            ]
        )
    q = fc.quantiles[..., q_idx]  # (B, V, H, 9)
    return q.permute(0, 3, 1, 2).float().cpu().numpy()


def run_task(
    forecaster,
    dataset_name,
    term,
    config,
    output_dir,
    model_hyperparams,
    max_patches,
    max_batch_size,
):
    settings = get_dataset_settings(dataset_name, term, config)
    prediction_length = settings["prediction_length"]
    dataset = Dataset(
        name=dataset_name,
        term=term,
        to_univariate=False,  # Timestral forecasts the variates of a series jointly.
        prediction_length=prediction_length,
        test_length=settings["test_length"],
        val_length=settings["val_length"],
    )
    num_variates = dataset.target_dim
    print(
        f"  {dataset_name}/{term}: {len(dataset.hf_dataset)} series x {dataset.windows} windows, "
        f"V={num_variates}, H={prediction_length}, freq={dataset.freq}",
        flush=True,
    )

    contexts = []
    for entry in dataset.test_data.input:
        target = np.asarray(entry["target"], dtype=np.float32)
        contexts.append(target[None] if target.ndim == 1 else target)  # (V, T): full history

    levels = forecaster.quantile_levels
    q_idx = [
        min(range(len(levels)), key=lambda k: abs(levels[k] - q)) for q in TIME_QUANTILE_LEVELS
    ]
    assert all(abs(levels[k] - q) < 1e-6 for k, q in zip(q_idx, TIME_QUANTILE_LEVELS)), levels

    fc_quantiles = np.empty(
        (len(contexts), len(TIME_QUANTILE_LEVELS), num_variates, prediction_length), np.float32
    )
    lengths = [c.shape[-1] for c in contexts]
    for idx in _batches(lengths, num_variates, forecaster.patch_size, max_patches, max_batch_size):
        fc_quantiles[idx] = _forecast_batch(
            forecaster, [contexts[i] for i in idx], prediction_length, q_idx
        )

    save_window_predictions(
        dataset=dataset,
        fc_quantiles=fc_quantiles,
        ds_config=f"{dataset_name}/{term}",
        output_base_dir=str(output_dir),
        seasonality=get_seasonality(dataset.freq),
        model_hyperparams=model_hyperparams,
        quantile_levels=TIME_QUANTILE_LEVELS,
    )


def main():
    parser = argparse.ArgumentParser(description="Run Timestral on TIME benchmark datasets")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    parser.add_argument(
        "--model-name", default=None, help="Results folder name (default: the model id's name)"
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all_datasets"],
        help="'all_datasets' or names like 'Crypto/D'",
    )
    parser.add_argument("--terms", nargs="+", default=None, choices=["short", "medium", "long"])
    parser.add_argument("--output-dir", default="./output/results")
    parser.add_argument("--config", default=None, help="Path to datasets.yaml config file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max-patches",
        type=int,
        default=2**19,
        help="Batch budget in context patches (x variates)",
    )
    parser.add_argument("--max-batch-size", type=int, default=512, help="Max windows per batch")
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute tasks that already have metrics.npz"
    )
    args = parser.parse_args()

    if not os.getenv("TIME_DATASET"):
        sys.exit("TIME_DATASET is not set; add it to .env or export it")

    config = load_dataset_config(Path(args.config) if args.config else None)
    datasets = list(config["datasets"]) if args.dataset == ["all_datasets"] else args.dataset
    model_name = args.model_name or args.model_id.rstrip("/").split("/")[-1]
    output_dir = Path(args.output_dir) / model_name

    forecaster = TimestralForecaster.from_pretrained(args.model_id, device=args.device)
    model_hyperparams = {
        "model": model_name,
        "model_id": args.model_id,
        "max_context": forecaster.max_context,
        "per_shot_horizon_patches": forecaster.per_shot_horizon_patches,
        "symmetric_averaging": True,
        "clip_nonnegative": True,
        "to_univariate": False,
    }
    print(f"Model {args.model_id} on {args.device}; {len(datasets)} dataset(s)", flush=True)

    failed = []
    for d_idx, dataset_name in enumerate(datasets, 1):
        terms = args.terms or get_available_terms(dataset_name, config)
        for term in terms:
            if term not in config["datasets"][dataset_name]:
                continue
            task_dir = output_dir / dataset_name / term
            if (task_dir / "metrics.npz").exists() and not args.overwrite:
                print(
                    f"[{d_idx}/{len(datasets)}] {dataset_name}/{term}: done, skipping", flush=True
                )
                continue
            print(f"[{d_idx}/{len(datasets)}] {dataset_name}/{term}", flush=True)
            start = time.time()
            try:
                run_task(
                    forecaster,
                    dataset_name,
                    term,
                    config,
                    output_dir,
                    model_hyperparams,
                    args.max_patches,
                    args.max_batch_size,
                )
            except Exception:
                traceback.print_exc()
                failed.append(f"{dataset_name}/{term}")
                continue
            metrics = np.load(task_dir / "metrics.npz")
            summary = {k: float(np.nanmean(metrics[k])) for k in ("MASE", "CRPS")}
            print(f"  -> {json.dumps(summary)} ({time.time() - start:.0f}s)", flush=True)

    if failed:
        sys.exit(f"Failed tasks: {failed}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
