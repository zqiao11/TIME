"""Run t0-gamma on the TIME benchmark.

Copy it into the TIME repo as ``experiments/t0_gamma.py`` and run, with
``TIME_DATASET`` pointing at the downloaded ``Real-TSF/TIME`` snapshot::

    python experiments/t0_gamma.py --device mps

From apps/t0/, point ``--time-repo`` at a TIME clone instead::

    uv run --group gift-eval python tools/evaluate_timebench.py --time-repo /path/to/TIME

The variates of a multivariate series are forecast jointly.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_CONTEXT = 8192
# The ``predict`` settings behind the published scores, all on by default.
PREDICT_SETTINGS = (
    "trim_leading_missing",
    "symmetric_averaging",
    "clip_nonnegative",
    "snap_integers",
)


def _predict_task(
    model,
    eval_input,
    prediction_length: int,
    quantile_levels,
    context_length: int,
    batch_size: int,
    options=None,
):
    """Forecast every evaluation window, returning (N, Q, [V,] H) as TIME expects."""
    from t0 import batch_series

    entries = list(eval_input)
    device = next(model.parameters()).device
    per_entry: list[np.ndarray] = []

    for start in range(0, len(entries), batch_size):
        chunk = entries[start : start + batch_size]
        # A (V, T) target is one series of V variates; batch_series gives its rows
        # a shared group id so they are forecast jointly.
        contexts = [np.asarray(e["target"], dtype=np.float32)[..., -context_length:] for e in chunk]
        widths = [c.shape[0] if c.ndim == 2 else 1 for c in contexts]

        context, mask, group_ids = batch_series(contexts)
        with torch.inference_mode():
            out = model.predict(
                context.to(device),
                horizon=prediction_length,
                quantile_levels=list(quantile_levels),
                mask=mask.to(device),
                group_ids=group_ids.to(device),
                **(options or {}),
            )
        quantiles = out.quantiles.float().cpu().numpy()  # (rows, H, Q)

        row = 0
        for width in widths:
            block = quantiles[row : row + width]  # (V, H, Q)
            row += width
            # TIME wants quantiles on axis 1: (Q, H) univariate, (Q, V, H) multivariate.
            per_entry.append(block[0].T if width == 1 else block.transpose(2, 0, 1))

    return np.stack(per_entry, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--time-repo", type=Path, default=None, help="Clone of github.com/zqiao11/TIME"
    )
    parser.add_argument("--model-path", default="theforecastingcompany/t0-gamma")
    parser.add_argument("--model-name", default="t0-gamma")
    parser.add_argument("--output-dir", default="output/results/t0_gamma")
    parser.add_argument(
        "--dataset", nargs="*", default=None, help="dataset/freq ids; default every configured task"
    )
    parser.add_argument("--terms", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument(
        "--fresh", action="store_true", help="Recompute tasks that already have metrics"
    )
    for setting in PREDICT_SETTINGS:
        parser.add_argument(
            f"--{setting.replace('_', '-')}",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=f"T0Forecaster.predict({setting}=...)",
        )
    args = parser.parse_args()

    if args.time_repo is not None:
        sys.path.insert(0, str(args.time_repo / "src"))
    from gluonts.time_feature import get_seasonality
    from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config
    from timebench.evaluation.saver import save_window_predictions
    from timebench.evaluation.utils import get_available_terms

    from t0 import T0Forecaster

    if not os.getenv("TIME_DATASET"):
        raise SystemExit(
            "TIME_DATASET is not set — point it at the downloaded Real-TSF/TIME snapshot"
        )

    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = T0Forecaster.from_pretrained(args.model_path).to(device).eval()
    print(
        f"[setup] {args.model_path} on {device} | ctx {args.context_length} | batch {args.batch_size}"
        f" | {', '.join(s for s in PREDICT_SETTINGS if getattr(args, s))}"
    )

    config = load_dataset_config(None)
    # The config nests every task under a "datasets" key; iterating the top level
    # yields that single literal name and silently evaluates nothing.
    catalog = config.get("datasets", config)
    dataset_names = args.dataset or list(catalog)
    print(f"[setup] {len(dataset_names)} dataset(s)")
    completed = 0

    for name in dataset_names:
        for term in args.terms or get_available_terms(name, config):
            settings = get_dataset_settings(name, term, config)
            done_marker = Path(args.output_dir) / name / term / "metrics.npz"
            if done_marker.is_file() and not args.fresh:
                print(f"{name}/{term}: already done — skipping")
                continue
            dataset = Dataset(
                name=name,
                term=term,
                to_univariate=False,
                prediction_length=settings["prediction_length"],
                test_length=settings["test_length"],
                val_length=settings["val_length"],
            )
            start = time.monotonic()
            fc_quantiles = _predict_task(
                model,
                dataset.test_data.input,
                dataset.prediction_length,
                DEFAULT_QUANTILES,
                args.context_length,
                args.batch_size,
                options={setting: getattr(args, setting) for setting in PREDICT_SETTINGS},
            )
            metadata = save_window_predictions(
                dataset=dataset,
                fc_quantiles=fc_quantiles,
                ds_config=f"{name}/{term}",
                output_base_dir=args.output_dir,
                seasonality=get_seasonality(dataset.freq),
                model_hyperparams={
                    "model_name": args.model_name,
                    "context_length": args.context_length,
                },
                quantile_levels=DEFAULT_QUANTILES,
            )
            completed += 1
            print(
                f"{name}/{term}: {fc_quantiles.shape} ({time.monotonic() - start:.1f}s) -> {metadata}"
            )

    if completed == 0 and not any(
        (Path(args.output_dir) / n / t / "metrics.npz").is_file()
        for n in dataset_names
        for t in (args.terms or [])
    ):
        raise SystemExit("no tasks were evaluated — check --dataset/--terms against the config")


if __name__ == "__main__":
    main()
