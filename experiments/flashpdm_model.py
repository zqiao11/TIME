"""Evaluate Flashmind PDM on TIME through the published flashpdm client."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from flashpdm import Client
from flashpdm.batching import ForecastPool
from gluonts.time_feature import get_seasonality
from huggingface_hub import snapshot_download

from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config
from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.utils import get_available_terms


REVISIONS = {
    "pdm-femto": "46bb0bde1c836a6b1a82e6d0108f766adb47c8639cf9c695637c583a617fe06c.2026-10-27",
    "pdm-pico": "cf380714c0e6eda4a43a1e62a12cf228b3c0795fa21be7078ddaf6b866c501bd.2026-10-27",
    "pdm-nano": "48cc3fd4918228d313179822d616fca5c282c5bcc9e59d55bc33c31b2c98eb5e.2026-10-27",
}
QUANTILES = [i / 10 for i in range(1, 10)]
RESULT_FILES = {"predictions.npz", "metrics.npz", "config.json"}


def project(bands, history):
    bands = bands.astype(np.float32)
    for v, row in enumerate(history):
        observed = row[np.isfinite(row)]
        if len(observed) and (observed >= 0).all():
            bands[v] = np.maximum(bands[v], 0)
            if len(observed) >= 8 and (observed == np.rint(observed)).all():
                bands[v] = np.rint(bands[v])
    return bands


def forecast_task(pool, dataset):
    """TIME input order is series, window; output axes are instance, Q, V, H."""
    from itertools import tee

    contexts, histories = tee(
        np.atleast_2d(np.asarray(entry["target"], dtype=np.float32))
        for entry in dataset.test_data.input
    )
    forecasts = pool.forecast_quantiles(
        contexts, dataset.prediction_length, quantile_levels=QUANTILES,
    )
    return np.stack([
        project(bands, history).transpose(2, 0, 1)
        for bands, history in zip(forecasts, histories, strict=True)
    ])


def completed(path, contract):
    """A task is reusable only after all official files were committed together."""
    receipt = path / "complete.json"
    if not receipt.exists():
        return False
    saved = json.loads(receipt.read_text())
    if saved["contract"] != contract:
        raise ValueError(f"Different evaluation settings at {path}; use another output directory")
    if set(saved["sha256"]) != RESULT_FILES:
        raise ValueError(f"Incomplete result manifest: {path}")
    for name, digest in saved["sha256"].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed or incomplete result: {path / name}")
    return True


def run_task(pool, dataset_name, term, config, output, contract):
    settings = get_dataset_settings(dataset_name, term, config)
    dataset = Dataset(
        name=dataset_name, term=term, to_univariate=False,
        prediction_length=settings["prediction_length"],
        test_length=settings["test_length"], val_length=settings["val_length"],
    )
    if dataset.target_dim > pool.capabilities["max_channels_per_input"]:
        raise ValueError(f"{dataset_name} exceeds the model's joint channel limit")
    started = time.monotonic()
    forecasts = forecast_task(pool, dataset)
    task_id = f"{dataset_name}/{term}"
    # An interrupted scorer must never leave a task that looks complete.
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        save_window_predictions(
            dataset=dataset, fc_quantiles=forecasts, ds_config=task_id,
            output_base_dir=temporary, seasonality=get_seasonality(dataset.freq),
            model_hyperparams=contract, quantile_levels=QUANTILES,
        )
        source = Path(temporary) / task_id
        destination = output / task_id
        destination.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name in ("predictions.npz", "metrics.npz", "config.json"):
            hashes[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
            os.replace(source / name, destination / name)
        receipt = dict(contract=contract, sha256=hashes, elapsed_s=time.monotonic() - started)
        (source / "complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(source / "complete.json", destination / "complete.json")
    with np.load(destination / "metrics.npz") as metrics:
        scores = {k: float(np.nanmean(metrics[k])) for k in ("MASE", "CRPS")}
    print(json.dumps(dict(task=task_id, elapsed_s=receipt["elapsed_s"], **scores)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=REVISIONS, default="pdm-nano")
    parser.add_argument("--revision")
    parser.add_argument("--dataset", nargs="+", default=["all_datasets"])
    parser.add_argument("--terms", nargs="+", choices=["short", "medium", "long"])
    parser.add_argument("--output-dir", type=Path, default=Path("output/results"))
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    if not os.environ.get("FLASHMIND_API_KEY"):
        parser.error("Set FLASHMIND_API_KEY to an evaluator API key")
    if not os.environ.get("TIME_DATASET"):
        os.environ["TIME_DATASET"] = snapshot_download("Real-TSF/TIME", repo_type="dataset")
    config = load_dataset_config()
    names = list(config["datasets"]) if args.dataset == ["all_datasets"] else args.dataset
    for name in names:
        if name not in config["datasets"]:
            parser.error(f"Unknown TIME dataset: {name}")
    output = args.output_dir / f"PDM-{args.model.split('-')[1].title()}"
    output.mkdir(parents=True, exist_ok=True)
    with Client(timeout=180, max_retries=2) as client, ForecastPool(
        client, model=args.model, revision=args.revision or REVISIONS[args.model],
        concurrency=args.concurrency, cache_dir=args.output_dir / ".forecast-cache" / args.model,
        max_attempts=3,
    ) as pool:
        contract = dict(
            model=args.model, revision=pool.revision, client="flashpdm==0.1.12",
            model_url=f"https://www.flashmind.com/models/pdm#{args.model}",
            organization="Flashmind UAE", max_context=pool.capabilities["max_context"],
            to_univariate=False, flip_averaging=True, clamp_nonnegative=True,
            integer_counts=True, count_min_observed=8, rollout="auto", quantile_levels=QUANTILES,
            benchmark_config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        )
        for name in names:
            for term in args.terms or get_available_terms(name, config):
                if term not in config["datasets"][name]:
                    continue
                if completed(output / name / term, contract):
                    print(f"Already complete: {name}/{term}", flush=True)
                    continue
                run_task(pool, name, term, config, output, contract)
        print(f"Results: {output}; batches: {pool.stats}", flush=True)


if __name__ == "__main__":
    main()
