"""ChakraTS-OS-10M experiments for the TIME benchmark.

ChakraTS-OS-10M (https://huggingface.co/yhatlabs/ChakraTS-OS-10M) is an open-weight forecasting model from YHat Labs
(10.6M parameters), run with the yhatmodels library (https://github.com/yhatlabs/yhatmodels).

Protocol:
- A series' variates are forecast together (`to_univariate=False`).
- Single-variate series are read 16 at a time within a test window (`cross_series`) when all of that window's series end
  at the same timestamp; TIME cuts each series' windows from its own end, so otherwise they are forecast one by one.
- The context is the full history before each window (the model reads up to 32768 steps).
- The model's default `pro` preset; TIME's nine deciles are requested directly; one forecast per window.
- Results are written with TIME's own `save_window_predictions`.

Access: the weights are gated on Hugging Face during early access (request access on the model page, then
`hf auth login`). The first download needs a one-time license acceptance with a yhatlabs.com account: run
`yhatmodels accept-license ChakraTS-OS-10M` in a terminal, or set YHAT_MODELS_TOKEN.

Examples:
    bash scripts/run_chakrats_os.sh
    bash scripts/run_chakrats_os.sh --dataset "Crypto/D" --terms short
"""

import argparse
import os
from pathlib import Path

import numpy as np
from gluonts.time_feature import get_seasonality
from huggingface_hub import snapshot_download

from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config
from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.utils import get_available_terms

QUANTILES = [i / 10 for i in range(1, 10)]


def group_size(dataset, contexts) -> int:
    """16 for single-variate series (read together when a window's series share a cutoff), else 1: a series' variates
    are already forecast together."""
    return 16 if all(c.shape[0] == 1 for c in contexts) else 1


def forecast_task(model, dataset, preset):
    """TIME's test instances (series x window, in that order) -> (instances, Q, V, H), one window per call. A window's
    series are read together only when all of their contexts end at the same timestamp: TIME cuts each series' windows
    from its own end, so where series end on different dates the same window is a different moment for each, and reading
    them together would let one series see another's values after its own cutoff."""
    entries = list(dataset.test_data.input)
    contexts = [np.atleast_2d(np.asarray(e["target"], dtype=np.float32)) for e in entries]
    group = group_size(dataset, contexts)
    seen, by_window = {}, {}
    for i, e in enumerate(entries):  # an item's windows come oldest first
        j = seen.get(e["item_id"], 0)
        seen[e["item_id"]] = j + 1
        by_window.setdefault(j, []).append(i)
    out, used = [None] * len(entries), 1
    for idx in by_window.values():
        aligned = len({str(entries[i]["start"] + contexts[i].shape[-1]) for i in idx}) == 1  # the step after each context
        g = group if aligned else 1
        used = max(used, g)
        qs = model.forecast([contexts[i] for i in idx], dataset.prediction_length, quantiles=QUANTILES, preset=preset, cross_series=g)
        for i, q in zip(idx, qs, strict=True):
            out[i] = q  # (V, H, Q)
    return np.stack([q.transpose(2, 0, 1) for q in out]), used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ChakraTS-OS-10M")
    parser.add_argument("--preset", default="pro")
    parser.add_argument("--dataset", nargs="+", default=["all_datasets"])
    parser.add_argument("--terms", nargs="+", choices=["short", "medium", "long"])
    parser.add_argument("--output-dir", type=Path, default=Path("output/results"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    from yhatmodels import ChakraTS

    if not os.environ.get("TIME_DATASET"):
        os.environ["TIME_DATASET"] = snapshot_download("Real-TSF/TIME", repo_type="dataset")
    model = ChakraTS.from_pretrained(args.model, device=args.device)
    config = load_dataset_config()
    names = list(config["datasets"]) if args.dataset == ["all_datasets"] else args.dataset
    for name in names:
        if name not in config["datasets"]:
            parser.error(f"Unknown TIME dataset: {name} (names include the frequency, e.g. Auto_Production_SF/M)")
    output = args.output_dir / "ChakraTS-OS-10M"
    for name in names:
        for term in args.terms or get_available_terms(name, config):
            if term not in config["datasets"][name]:
                continue
            s = get_dataset_settings(name, term, config)
            dataset = Dataset(name=name, term=term, to_univariate=False, prediction_length=s["prediction_length"],
                              test_length=s["test_length"], val_length=s["val_length"])  # fmt: skip
            fc, group = forecast_task(model, dataset, args.preset)
            save_window_predictions(dataset=dataset, fc_quantiles=fc, ds_config=f"{name}/{term}", output_base_dir=str(output),
                                    seasonality=get_seasonality(dataset.freq), quantile_levels=QUANTILES,
                                    model_hyperparams={"model": "ChakraTS-OS-10M", "preset": args.preset, "cross_series": group})  # fmt: skip
            with np.load(output / name / term / "metrics.npz") as m:
                print(f"{name}/{term}: MASE {np.nanmean(m['MASE']):.4f} CRPS {np.nanmean(m['CRPS']):.4f}", flush=True)


if __name__ == "__main__":
    main()
