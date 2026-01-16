"""
Moirai model experiments for time series forecasting.

Usage:
    python experiments/moirai.py
    python experiments/moirai.py --model-size base
    python experiments/moirai.py --dataset "TSBench_IMOS_v2/15T" --terms short medium long
    python experiments/moirai.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/moirai.py --dataset all_datasets  # Run all datasets from config
    python experiments/moirai.py --val  # Evaluate on validation data (no saving)
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)

# Load environment variables
load_dotenv()

patch_size_dict = {"S": 64, "T": 32, "H": 32, "D": 16, "B": 16, "W": 16, "M": 8, "Q": 8, "Y": 8, "A": 8}

def get_available_terms(dataset_name: str, config: dict) -> list[str]:
    """Get the terms that are actually defined in the config for a dataset."""
    datasets_config = config.get("datasets", {})
    if dataset_name not in datasets_config:
        return []

    dataset_config = datasets_config[dataset_name]
    available_terms = []
    for term in ["short", "medium", "long"]:
        if term in dataset_config and dataset_config[term].get("prediction_length") is not None:
            available_terms.append(term)
    return available_terms


def run_moirai_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] = None,
    model_size: str = "small",
    output_dir: str | None = None,
    batch_size: int = 512,
    num_samples: int = 100,
    context_length: int = 4000,
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
):
    """
    Run Moirai model experiments on a dataset with specified terms.

    Args:
        dataset_name: Dataset name (e.g., "TSBench_IMOS_v2/15T")
        terms: List of terms to evaluate ("short", "medium", "long").
               If None, auto-detect from config.
        model_size: Moirai model size ("small", "base", "large")
        output_dir: Output directory for results
        batch_size: Batch size for prediction
        num_samples: Number of samples for probabilistic forecasting
        context_length: Maximum context length
        cuda_device: CUDA device ID
        config_path: Path to datasets.yaml config file
        use_val: If True, evaluate on validation data (for hyperparameter selection, no saving)
    """
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # Load dataset configuration
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    # Auto-detect available terms from config if not specified
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if output_dir is None:
        output_dir = f"./output/results/moirai_{model_size}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Terms: {terms}")
    print(f"Evaluation on: {'Validation data (no saving)' if use_val else 'Test data'}")
    print(f"{'='*60}")

    for term in terms:
        print(f"\n--- Term: {term} ---")

        # Get settings from config
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        # Moirai hyperparameter (constant across all terms)
        freq = dataset_name.split("/")[1]           # 例如 "15T", "H", "5T"
        freq_unit = freq.lstrip('0123456789')       # 去掉前面的数字: "15T" -> "T", "H" -> "H"
        patch_size = patch_size_dict[freq_unit]     # 用单位查找 patch_size

        # Try multivariate first, fallback to univariate on OOM
        for to_univariate in [False, True]:
            mode_name = "univariate" if to_univariate else "multivariate"

            # Initialize model for this term with the term's prediction_length
            print(f"  Initializing Moirai-{model_size} model ({mode_name} mode) with prediction_length={prediction_length}...")
            model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{model_size}"),
                prediction_length=prediction_length,  # From config for this term
                context_length=context_length,
                patch_size=patch_size,
                num_samples=num_samples,
                target_dim=1,  # Will be updated per dataset
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )

            # Load dataset with config settings
            dataset = Dataset(
                name=dataset_name,
                term=term,
                to_univariate=to_univariate,
                prediction_length=prediction_length,
                test_length=test_length,
                val_length=val_length,
            )

            # Calculate actual test/val length (based on min series length)
            if use_val:
                data_length = val_length
                num_windows = dataset.val_windows
                split_name = "Val split"
            else:
                data_length = test_length
                num_windows = dataset.windows
                split_name = "Test split"

            print("  Dataset info:")
            print(f"    - Mode: {mode_name}")
            print(f"    - Frequency: {dataset.freq}")
            print(f"    - Num series: {len(dataset.hf_dataset)}")
            print(f"    - Target dim: {dataset.target_dim}")
            print(f"    - Series length: min={dataset._min_series_length}, max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
            print(f"    - {split_name}: {data_length} steps)")
            print(f"    - Prediction length: {dataset.prediction_length}")
            print(f"    - Windows: {num_windows}")

            # Configure model for this dataset
            model.hparams.prediction_length = dataset.prediction_length
            model.hparams.target_dim = 1 if to_univariate else dataset.target_dim
            model.hparams.past_feat_dynamic_real_dim = dataset.past_feat_dynamic_real_dim

            predictor = model.create_predictor(batch_size=batch_size)
            season_length = get_seasonality(dataset.freq)

            # Generate predictions
            data_type = "validation" if use_val else "test"
            print(f"  Running predictions on {data_type} data...")

            try:
                if use_val:
                    print("    (No results saved - validation data used for hyperparameter selection)")
                else:
                    # Save predictions and metrics for test data
                    ds_config = f"{dataset_name}/{term}"

                    # Prepare model hyperparameters for metadata
                    # Note: num_samples is stored in predictions.npz, not in metadata.json
                    model_hyperparams = {
                        "patch_size": patch_size,
                        "context_length": context_length,
                        "mode": mode_name,
                    }

                    metadata = save_window_predictions(
                        dataset=dataset,
                        predictor=predictor,
                        ds_config=ds_config,
                        output_base_dir=output_dir,
                        seasonality=season_length,
                        model_hyperparams=model_hyperparams,
                    )
                    print(f"  Completed: {metadata['num_series']} series × {metadata['num_windows']} windows")
                    print(f"  Output: {metadata.get('output_dir', output_dir)}")

                # Success - break out of the retry loop
                break

            except torch.cuda.OutOfMemoryError:
                if to_univariate:
                    # Already in univariate mode, re-raise the error
                    raise
                else:
                    # Multivariate failed, try univariate
                    print("  ⚠️  CUDA OOM in multivariate mode, switching to univariate mode...")
                    # Clean up GPU memory
                    del model, predictor, dataset
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Moirai experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["US_Term_Structure/B"],
                        help="Dataset name(s). Can be a single dataset, multiple datasets, or 'all_datasets' to run all datasets from config")
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--model-size", type=str, default="base",
                        choices=["small", "base", "large"],
                        help="Moirai model size")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for prediction")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples for probabilistic forecasting")
    parser.add_argument("--context-length", type=int, default=4000,
                        help="Maximum context length")
    parser.add_argument("--cuda-device", type=str, default="0",
                        help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--val", action="store_true",
                        help="Evaluate on validation data (for hyperparameter selection, no saving)")

    args = parser.parse_args()

    # Handle dataset list or 'all_datasets'
    config_path = Path(args.config) if args.config else None

    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        # Load all datasets from config
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
        print(f"Running all {len(datasets)} datasets from config:")
        for ds in datasets:
            print(f"  - {ds}")
    else:
        datasets = args.dataset

    # Iterate over all datasets
    total_datasets = len(datasets)
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        try:
            run_moirai_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                model_size=args.model_size,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                context_length=args.context_length,
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
