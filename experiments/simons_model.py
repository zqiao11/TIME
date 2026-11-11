"""Simons forecasts TIME tasks and saves predictions, metrics and config.

    pip install "simons @ git+https://github.com/Abel-ai-lab/Simons"
    python experiments/simons_model.py --model abel-lab/Simons-297M --dataset all_datasets

scripts/run_simons.sh sets up the environment and runs every task.

Tasks are loaded with to_univariate=False; related variates are forecast in groups of at most 32.
The last context_length steps are read, and the checkpoint's forecast rules are applied.
CUDA uses bf16 for the trunk and fp32 for embedding, norm and head."""

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
from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config
from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.utils import get_available_terms

from simons import Simons, forecast_series

load_dotenv()

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # the quantiles Simons forecasts


def run_simons_experiment(
    model: Simons,
    model_name: str,
    dataset_name: str,
    terms: list[str] | None,
    output_dir: str,
    batch_size: int,
    config_path: Path | None = None,
):
    """Forecast every test window of one dataset's terms and save TIME's predictions, metrics and
    config."""
    config = load_dataset_config(config_path)
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    for term in terms:
        settings = get_dataset_settings(dataset_name, term, config)
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=False,  # Simons supports multivariate natively
            prediction_length=settings.get("prediction_length"),
            test_length=settings.get("test_length"),
            val_length=settings.get("val_length"),
        )
        print(f"\n--- {dataset_name} / {term}: {len(dataset.hf_dataset)} series, "
              f"{dataset.target_dim} variates, {dataset.windows} windows, "
              f"horizon {dataset.prediction_length}")

        # One history per (series, window), in TIME's order: [time] for univariate,
        # [variates, time] otherwise.
        histories = [np.asarray(entry["target"], dtype=np.float32)
                     for entry in dataset.test_data.input]
        with torch.no_grad():
            outputs = forecast_series(model, histories, dataset.prediction_length,
                                      batch_size=batch_size)
        # [horizon, 9] or [variates, horizon, 9] -> (instances, 9, horizon) or
        # (instances, 9, variates, horizon)
        fc_quantiles = np.moveaxis(np.stack(outputs), -1, 1)

        metadata = save_window_predictions(
            dataset=dataset,
            fc_quantiles=fc_quantiles,
            ds_config=f"{dataset_name}/{term}",
            output_base_dir=output_dir,
            seasonality=get_seasonality(dataset.freq),
            model_hyperparams={
                "model": model_name,
                "context_length": model.config.context_length,
                "quantile_levels": QUANTILE_LEVELS,
            },
            quantile_levels=QUANTILE_LEVELS,
        )
        print(f"  Completed: {metadata['num_series']} series x {metadata['num_windows']} windows")


def main():
    parser = argparse.ArgumentParser(description="Run Simons experiments")
    parser.add_argument("--model", type=str, default="abel-lab/Simons-297M",
                        help="Hugging Face checkpoint: abel-lab/Simons-25M, -60M, -178M or -297M")
    parser.add_argument("--dataset", type=str, nargs="+", default=["Global_Influenza/W"],
                        help="Dataset name(s). Can be a single dataset, multiple datasets, "
                             "or 'all_datasets'")
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (default: output/results/<model name>)")
    parser.add_argument("--batch-size", type=int, default=256, help="Series per forward pass")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    output_dir = args.output_dir or f"./output/results/{model_name}"
    config_path = Path(args.config) if args.config else None
    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        datasets = list(load_dataset_config(config_path).get("datasets", {}).keys())
    else:
        datasets = args.dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Simons.from_pretrained(args.model, device=device)
    print(f"{model_name}: {sum(p.numel() for p in model.parameters()):,} parameters on {device}; "
          f"{len(datasets)} datasets")
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n# Dataset {idx}/{len(datasets)}: {dataset_name}")
        run_simons_experiment(model, model_name, dataset_name, args.terms, output_dir,
                              args.batch_size, config_path)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
