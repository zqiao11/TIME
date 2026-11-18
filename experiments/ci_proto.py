"""CI-Proto experiments for the TIME benchmark.

CI-Proto is Continuous Intelligence's time-series foundation model. It runs through the
``continuousintelligence`` package, which loads the checkpoint and its ``tta.json`` from the
Hugging Face Hub (ContinuousIntelligence/ci-proto) and applies the release's preprocessing and
calibration.

Protocol:
- Each window supplies at most its most recent 16,384 steps, keeping inference on the
  standard checkpoint. Timestamps retain the original forecast origin after truncation.
- Each window's horizon is forecast in one call. For horizons above 320 steps, the checkpoint's
  forecast window, the package widens the window to cover the horizon.
- A task with one target forecasts all of a window's series in one call, as a cohort identified
  by TIME's item ids. Shorter series are padded at the start.
- A task with several targets forecasts each series' targets jointly, one call per window.
- Each window's timestamps run from its series' stored start at the dataset's frequency, over
  the history and the horizon. With the frequency, they give the release's calendar features.
- The release's ``tta.json`` is used as published.
- TIME's 9 quantile levels 0.1..0.9 are selected from the model's quantile levels.

Examples:
    python experiments/ci_proto.py --dataset all_datasets
    python experiments/ci_proto.py --dataset "SG_PM25/H" "Crypto/D" --terms short
"""

import argparse
import json
import os
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import continuousintelligence as ci
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality

from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config
from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.utils import get_available_terms

load_dotenv()

DEFAULT_MODEL_ID = "ContinuousIntelligence/ci-proto"
DEFAULT_REVISION = "ee3031351a151e3f6febf8c62b53e6ea1008bef6"
MAX_CONTEXT = 16_384
TIME_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# The files TIME's saver writes for a task, all of which a finished task has.
RESULT_FILES = ("predictions.npz", "metrics.npz", "config.json")
# The timestamp of the padding before a shorter series' first observation.
PAD_TIME = np.iinfo(np.int64).min


def _quantile_columns(model_levels: list[float]) -> list[int]:
    """Return the positions of TIME's quantile levels among the model's levels."""
    columns = []
    for level in TIME_QUANTILE_LEVELS:
        matches = [i for i, candidate in enumerate(model_levels) if abs(candidate - level) < 1e-6]
        if not matches:
            raise ValueError(f"the model has no quantile level {level}")
        columns.append(matches[0])
    return columns


def _series_timestamps(start: object, freq: str, length: int) -> np.ndarray:
    """Return ``length`` timestamps from ``start`` at ``freq``, as epoch nanoseconds."""
    return pd.date_range(start=pd.Timestamp(start), periods=length, freq=freq).asi8


def _left_padded(rows: list[np.ndarray], fill: float | int) -> np.ndarray:
    """Stack rows of different lengths, padding each at the start with ``fill``."""
    length = max(len(row) for row in rows)
    stacked = np.full((len(rows), length), fill, dtype=rows[0].dtype)
    for i, row in enumerate(rows):
        stacked[i, length - len(row) :] = row
    return stacked


def _finished(task_dir: Path) -> bool:
    """Return whether a task's results are all written; raise if only some of them are."""
    present = [name for name in RESULT_FILES if (task_dir / name).exists()]
    if present and len(present) < len(RESULT_FILES):
        raise ValueError(
            f"{task_dir} holds only {present}, from an interrupted run; remove it to rerun the task"
        )
    return bool(present)


def run_task(
    model: ci.Forecaster,
    dataset_name: str,
    term: str,
    config: dict,
    output_dir: Path,
    model_hyperparams: dict,
) -> None:
    """Forecast every window of one task and save it with TIME's saver."""
    settings = get_dataset_settings(dataset_name, term, config)
    horizon = settings["prediction_length"]
    dataset = Dataset(
        name=dataset_name,
        term=term,
        to_univariate=False,  # CI-Proto forecasts the targets of a series jointly.
        prediction_length=horizon,
        test_length=settings["test_length"],
        val_length=settings["val_length"],
    )
    num_series, num_windows = len(dataset.hf_dataset), dataset.windows
    num_variates, freq = dataset.target_dim, dataset.freq
    print(
        f"  {dataset_name}/{term}: {num_series} series x {num_windows} windows, "
        f"V={num_variates}, H={horizon}, freq={freq}",
        flush=True,
    )

    # TIME gives the windows series by series: window w of series s is entry s * windows + w.
    entries = list(dataset.test_data.input)
    contexts = [np.atleast_2d(np.asarray(entry["target"], dtype=np.float32)) for entry in entries]
    item_ids = [str(entries[s * num_windows]["item_id"]) for s in range(num_series)]
    starts = dataset.hf_dataset.data.column("start").to_pylist()
    stamps = [
        _series_timestamps(starts[s], freq, contexts[(s + 1) * num_windows - 1].shape[-1] + horizon)
        for s in range(num_series)
    ]
    columns = _quantile_columns(model.quantiles.tolist())

    fc_quantiles = np.empty(
        (len(entries), len(TIME_QUANTILE_LEVELS), num_variates, horizon), np.float32
    )
    if num_variates == 1:
        for w in range(num_windows):
            indices = [s * num_windows + w for s in range(num_series)]
            lengths = [contexts[i].shape[-1] for i in indices]
            forecast = model.forecast(
                torch.from_numpy(
                    _left_padded([contexts[i][0, -MAX_CONTEXT:] for i in indices], np.nan)
                ),
                horizon,
                timestamps=torch.from_numpy(
                    _left_padded(
                        [
                            stamps[s][max(0, length - MAX_CONTEXT) : length + horizon]
                            for s, length in enumerate(lengths)
                        ],
                        PAD_TIME,
                    )
                ),
                freq=freq,
                series_ids=item_ids,
            )  # (S, Q, H)
            fc_quantiles[indices, :, 0] = forecast[:, columns].numpy()
    else:
        for i, context in enumerate(contexts):
            length = context.shape[-1]
            forecast = model.forecast(
                torch.from_numpy(context[:, -MAX_CONTEXT:]),
                horizon,
                timestamps=torch.from_numpy(
                    stamps[i // num_windows][max(0, length - MAX_CONTEXT) : length + horizon]
                ),
                freq=freq,
            )  # (V, Q, H)
            fc_quantiles[i] = forecast[:, columns].permute(1, 0, 2).numpy()

    save_window_predictions(
        dataset=dataset,
        fc_quantiles=fc_quantiles,
        ds_config=f"{dataset_name}/{term}",
        output_base_dir=str(output_dir),
        seasonality=get_seasonality(freq),
        model_hyperparams=model_hyperparams,
        quantile_levels=TIME_QUANTILE_LEVELS,
    )


def main() -> None:
    """Run CI-Proto on the chosen TIME tasks, skipping the tasks already saved."""
    parser = argparse.ArgumentParser(description="Run CI-Proto on TIME benchmark datasets")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION, help="Hugging Face commit of the model's files"
    )
    parser.add_argument("--model-name", default="ci-proto", help="Results folder name")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all_datasets"],
        help="'all_datasets' or names like 'Crypto/D'",
    )
    parser.add_argument("--terms", nargs="+", default=None, choices=["short", "medium", "long"])
    parser.add_argument("--output-dir", default="./output/results")
    parser.add_argument("--config", default=None, help="Path to datasets.yaml config file")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if not os.getenv("TIME_DATASET"):
        sys.exit("TIME_DATASET is not set; add it to .env or export it")

    config = load_dataset_config(Path(args.config) if args.config else None)
    datasets = list(config["datasets"]) if args.dataset == ["all_datasets"] else args.dataset
    output_dir = Path(args.output_dir) / args.model_name

    model = ci.load_model(args.model_id, device=args.device, revision=args.revision)
    model_hyperparams = {
        "model": args.model_name,
        "model_id": args.model_id,
        "revision": args.revision,
        "max_context": MAX_CONTEXT,
        "device": str(model.device),
        "backend": model.backend,
        "continuousintelligence": version("continuousintelligence"),
        "torch": torch.__version__,
        "to_univariate": False,
    }
    print(
        f"Model {args.model_id}@{args.revision} on {model.device} ({model.backend}); "
        f"{len(datasets)} dataset(s)",
        flush=True,
    )

    failed = []
    for d_idx, dataset_name in enumerate(datasets, 1):
        terms = args.terms or get_available_terms(dataset_name, config)
        for term in terms:
            if term not in config["datasets"][dataset_name]:
                continue
            task_dir = output_dir / dataset_name / term
            if _finished(task_dir):
                print(
                    f"[{d_idx}/{len(datasets)}] {dataset_name}/{term}: done, skipping", flush=True
                )
                continue
            print(f"[{d_idx}/{len(datasets)}] {dataset_name}/{term}", flush=True)
            start = time.time()
            try:
                run_task(model, dataset_name, term, config, output_dir, model_hyperparams)
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
