"""
VisionTS++ model experiments for time series forecasting.
(Supports native multivariate input and quantile outputs)
(Adaptive: Batched for Multivariate, Sequential for Univariate/Variable Lengths)

Usage:
    python experiments/visionts++.py
    python experiments/visionts++.py --model-size base
    python experiments/visionts++.py --dataset "SG_Weather/D" --terms short medium long
    python experiments/visionts++.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/visionts++.py --dataset all_datasets  # Run all datasets from config
"""

import argparse
import os
import random
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gluonts.time_feature import get_seasonality
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from visionts import VisionTSpp, freq_to_seasonality_list
import visionts.util as visionts_util

from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.utils import get_available_terms, clean_nan_target


warnings.filterwarnings("ignore")

load_dotenv()


DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _normalize_offset_name(name: str) -> str:
    base = name.split("-")[0]
    base_lower = base.lower()
    if base_lower == "min":
        return "T"
    if base_lower == "h":
        return "H"
    if base_lower == "s":
        return "S"
    if base_lower == "d":
        return "D"
    if base_lower == "w":
        return "W"
    if base_lower in ("me", "ms", "m"):
        return "M"
    if base_lower.startswith("q"):
        return "Q"
    if base_lower.startswith(("y", "a")):
        return "A"
    if base_lower == "b":
        return "B"
    return base.upper()


def freq_to_seasonality_list_compat(freq: str) -> list[int]:
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
        base = _normalize_offset_name(offset.name)
        base_seasonality_list = visionts_util.POSSIBLE_SEASONALITIES.get(base, [])
        seasonality_list = []
        for base_seasonality in base_seasonality_list:
            seasonality, remainder = divmod(base_seasonality, offset.n)
            if not remainder:
                seasonality_list.append(seasonality)
        seasonality_list.append(1)
        return seasonality_list
    except Exception:
        return freq_to_seasonality_list(freq)


def run_visiontspp_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] = None,
    model_size: str = "base",
    output_dir: str | None = None,
    batch_size: int = 32,
    context_length: int = 4000,
    config_path: Path | None = None,
    seed: int = 42,
    quantile_levels: list[float] | None = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILE_LEVELS

    if output_dir is None:
        output_dir = f"./output/results/visiontspp_{model_size}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: VisionTS++ ({model_size})")
    print(f"Terms: {terms}")
    print(f"{'='*60}")

    print(f"  Preparing VisionTS++ model...")

    local_dir = "./hf_models/VisionTSpp"
    if model_size == "base":
        arch = "mae_base"
        ckpt_filename = "visiontspp_base.ckpt"
    else:  # large
        arch = "mae_large"
        ckpt_filename = "visiontspp_large.ckpt"

    ckpt_path = os.path.join(local_dir, ckpt_filename)

    if not os.path.exists(ckpt_path):
        print(f"  Downloading checkpoint to {local_dir}...")
        try:
            snapshot_download(
                repo_id="Lefei/VisionTSpp",
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            print(f"  Error downloading model: {e}")
            return

    print(f"  Loading model from {ckpt_path}...")
    model = VisionTSpp(
        arch,
        ckpt_path=ckpt_path,
        quantile=True,
        clip_input=True,
        complete_no_clip=False,
        color=True,
    ).to(device)
    model.eval()

    for term in terms:
        print(f"\n--- Term: {term} ---")

        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(
            "  Config: prediction_length="
            f"{prediction_length}, test_length={test_length}, val_length={val_length}"
        )

        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=False,
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )

        data_length = test_length if test_length else int(
            dataset.test_split * dataset._min_series_length
        )
        num_windows = dataset.windows
        split_name = "Test split"
        eval_data = dataset.test_data

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        periodicity_list = freq_to_seasonality_list_compat(dataset.freq)
        periodicity = periodicity_list[0]
        print(f"    - Derived Periodicity: {periodicity}")

        data_type = "test"
        print(f"  Running predictions on {data_type} data...")

        context_list = []

        for d in eval_data.input:
            target = np.asarray(d["target"])

            seq_len = target.shape[-1]
            if seq_len > context_length:
                target = target[..., -context_length:]

            target = clean_nan_target(target)
            if target.ndim == 1:
                target = target[np.newaxis, :]

            target_t = target.T
            context_list.append(torch.tensor(target_t).float())

        fc_quantiles = []

        def run_inference(input_tensor):
            curr_ctx_len = input_tensor.shape[1]
            nvars_input = input_tensor.shape[2]
            max_vars_per_pass = 16

            def _run_single_pass(tensor_chunk):
                chunk_vars = tensor_chunk.shape[2]
                model.update_config(
                    context_len=curr_ctx_len,
                    pred_len=prediction_length,
                    periodicity=periodicity,
                    num_patch_input=7,
                    padding_mode="constant",
                )
                color_list = [i % 3 for i in range(chunk_vars)]
                with torch.no_grad():
                    return model(tensor_chunk, export_image=False, color_list=color_list)

            if nvars_input <= max_vars_per_pass:
                return _run_single_pass(input_tensor)

            median_chunks = []
            quantile_chunks = None
            for start in range(0, nvars_input, max_vars_per_pass):
                end = min(start + max_vars_per_pass, nvars_input)
                chunk_output = _run_single_pass(input_tensor[:, :, start:end])
                preds_data = chunk_output[0] if isinstance(chunk_output, tuple) else chunk_output
                med_chunk = preds_data[0]
                q_chunk_list = preds_data[1]

                median_chunks.append(med_chunk)
                if quantile_chunks is None:
                    quantile_chunks = [[] for _ in range(len(q_chunk_list))]
                for qi, q_tensor in enumerate(q_chunk_list):
                    quantile_chunks[qi].append(q_tensor)

            med_full = torch.cat(median_chunks, dim=2)
            q_full = [torch.cat(parts, dim=2) for parts in quantile_chunks]
            return [med_full, q_full]

        print(
            "  [Strategy] Using Multivariate Path "
            f"(dim={dataset.target_dim}). Trying Batch Inference..."
        )

        total_items = len(context_list)
        for start in range(0, total_items, batch_size):
            end = min(start + batch_size, total_items)
            batch_list = context_list[start:end]

            try:
                batch_input = torch.stack(batch_list).to(device)
                output = run_inference(batch_input)

                if isinstance(output, tuple):
                    preds_data = output[0]
                else:
                    preds_data = output

                medians = preds_data[0].cpu().numpy()
                quantiles_list = preds_data[1]
                q_np_list = [q.cpu().numpy() for q in quantiles_list]

                for i in range(medians.shape[0]):
                    qs_sample = [q[i] for q in q_np_list]
                    qs_stacked = np.stack(qs_sample, axis=0)
                    med_expanded = np.expand_dims(medians[i], axis=0)
                    full_quantiles = np.concatenate(
                        [qs_stacked[:4], med_expanded, qs_stacked[4:]], axis=0
                    )
                    full_quantiles = full_quantiles.transpose(0, 2, 1)
                    fc_quantiles.append(full_quantiles)
            except RuntimeError as e:
                if "stack" in str(e):
                    for item in batch_list:
                        single_input = item.unsqueeze(0).to(device)
                        output = run_inference(single_input)

                        if isinstance(output, tuple):
                            preds_data = output[0]
                        else:
                            preds_data = output

                        medians = preds_data[0].cpu().numpy()
                        quantiles_list = preds_data[1]
                        q_np_list = [q.cpu().numpy() for q in quantiles_list]

                        qs_sample = [q[0] for q in q_np_list]
                        qs_stacked = np.stack(qs_sample, axis=0)
                        med_expanded = np.expand_dims(medians[0], axis=0)
                        full_quantiles = np.concatenate(
                            [qs_stacked[:4], med_expanded, qs_stacked[4:]], axis=0
                        )
                        full_quantiles = full_quantiles.transpose(0, 2, 1)
                        fc_quantiles.append(full_quantiles)
                else:
                    raise

            if (start // batch_size + 1) % 10 == 0:
                print(f"    Processed {min(end, total_items)}/{total_items}", end="\r")
        print("")

        fc_quantiles = np.stack(fc_quantiles, axis=0).astype(np.float32, copy=False)

        ds_config = f"{dataset_name}/{term}"
        model_hyperparams = {
            "model": f"visiontspp-{model_size}",
            "context_length": context_length,
            "periodicity": periodicity,
            "quantile_levels": quantile_levels,
        }

        gluonts_season = get_seasonality(dataset.freq)

        metadata = save_window_predictions(
            dataset=dataset,
            fc_quantiles=fc_quantiles,
            ds_config=ds_config,
            output_base_dir=output_dir,
            seasonality=gluonts_season,
            model_hyperparams=model_hyperparams,
            quantile_levels=quantile_levels,
        )

        print(f"  Completed: {metadata['num_series']} series × {metadata['num_windows']} windows")
        print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run VisionTS++ experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["Water_Quality_Darwin/15T"],
        help="Dataset name(s)",
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
        choices=["base", "large"],
        help="Model size",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=DEFAULT_QUANTILE_LEVELS,
        help="Quantile levels to predict",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=4000,
        help="Context length",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed",
    )

    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
    else:
        datasets = args.dataset

    total_datasets = len(datasets)
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")
        try:
            run_visiontspp_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                model_size=args.model_size,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                context_length=args.context_length,
                config_path=config_path,
                seed=args.seed,
                quantile_levels=args.quantiles,
            )
        except Exception as e:
            print(f"ERROR: Failed to run experiment for {dataset_name}: {e}")
            traceback.print_exc()
            continue

    print(f"\n{'#'*60}")
    print(f"# All completed!")


if __name__ == "__main__":
    main()
