"""
Moirai-MoE model experiments for time series forecasting.
(Aligned with TimeBench Moirai template)

Usage:
    python experiments/moirai_moe.py
    python experiments/moirai_moe.py --model-size base
    python experiments/moirai_moe.py --dataset "Traffic/15T" --terms short
    python experiments/moirai_moe.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/moirai_moe.py --dataset all_datasets  # Run all datasets from config
    python experiments/moirai_moe.py --val  # Evaluate on validation data (no saving)
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality

# Import Moirai-MoE specific modules
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)

# Load environment variables
load_dotenv()

def run_moirai_moe_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] = None,
    model_size: str = "small",  # options: small, base
    output_dir: str | None = None,
    batch_size: int = 16,
    num_samples: int = 100,
    context_length: int = 4000,
    patch_size: int = 16,
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
):
    """
    Run Moirai-MoE model experiments.
    """
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    # Load dataset configuration
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = ["short", "medium", "long"]

    if output_dir is None:
        output_dir = f"./output/results/moirai_moe_{model_size}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: Moirai-MoE ({model_size})")
    print(f"Terms: {terms}")
    print(f"Evaluation on: {'Validation data (no saving)' if use_val else 'Test data'}")
    print(f"{'='*60}")

    # --- Initialize Moirai-MoE Model (Once per dataset wrapper) ---
    print(f"Loading Moirai-MoE-{model_size} model...")
    hf_model_path = f"Salesforce/moirai-moe-1.0-R-{model_size}"
    
    try:
        module = MoiraiMoEModule.from_pretrained(hf_model_path)
        # We initialize the wrapper later per term because prediction_length changes
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model {hf_model_path}: {e}")
        return

    for term in terms:
        print(f"\n--- Term: {term} ---")

        # Get settings from config
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        # Initialize Forecast Wrapper with current prediction length
        model = MoiraiMoEForecast(
            module=module,
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=patch_size,
            num_samples=num_samples,
            target_dim=1,  # Will be updated per dataset
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        
        # Load Dataset
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=False, 
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

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - Series length: min={dataset._min_series_length}, max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
        print(f"    - {split_name}: {data_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        # Configure model for this dataset
        model.hparams.prediction_length = dataset.prediction_length
        model.hparams.target_dim = dataset.target_dim
        model.hparams.past_feat_dynamic_real_dim = dataset.past_feat_dynamic_real_dim

        predictor = model.create_predictor(batch_size=batch_size)

        # --- Inference ---
        data_type = "validation" if use_val else "test"
        print(f"  Running predictions on {data_type} data...")
        forecasts = list(predictor.predict(eval_data.input))

        # Count number of series
        num_total_instances = len(forecasts)
        num_series_log = num_total_instances // num_windows if num_windows > 0 else 0
        num_variates = dataset.target_dim
        prediction_length = dataset.prediction_length

        print(f"    Total instances: {num_total_instances}, Series: {num_series_log}, Windows: {num_windows}")

        # Collect ground truth labels and contexts
        print("    Collecting ground truth and context...")
        ground_truths = []
        contexts = []
        for inp, label in eval_data:
            ground_truths.append(label["target"])
            contexts.append(inp["target"])

        # Initialize arrays
        actual_num_samples = forecasts[0].samples.shape[0] if len(forecasts) > 0 else num_samples

        predictions_mean = np.zeros((num_series_log, num_windows, num_variates, prediction_length))
        predictions_samples = np.zeros((num_series_log, num_windows, actual_num_samples, num_variates, prediction_length))
        ground_truth = np.zeros((num_series_log, num_windows, num_variates, prediction_length))

        max_context_len = max(ctx.shape[-1] for ctx in contexts) if contexts else 0
        context_array = np.full((num_series_log, num_windows, num_variates, max_context_len), np.nan)

        print("    Organizing data into arrays...")
        for idx, (fc, gt, ctx) in enumerate(zip(forecasts, ground_truths, contexts)):
            series_idx = idx // num_windows
            window_idx = idx % num_windows

            fc_mean = fc.mean
            fc_samples = fc.samples

            if fc_mean.ndim == 1:
                fc_mean = fc_mean[np.newaxis, :]
            elif fc_mean.shape[0] == prediction_length and fc_mean.shape[1] == num_variates:
                fc_mean = fc_mean.T

            if fc_samples.ndim == 2:
                fc_samples = fc_samples[:, np.newaxis, :]
            elif fc_samples.shape[1] == prediction_length and fc_samples.shape[2] == num_variates:
                fc_samples = fc_samples.transpose(0, 2, 1)

            if gt.ndim == 1:
                gt = gt[np.newaxis, :]
            elif gt.shape[0] == prediction_length and gt.shape[1] == num_variates:
                gt = gt.T

            if ctx.ndim == 1:
                ctx = ctx[np.newaxis, :]
            elif ctx.shape[0] != num_variates:
                ctx = ctx.T

            predictions_mean[series_idx, window_idx] = fc_mean
            predictions_samples[series_idx, window_idx] = fc_samples
            ground_truth[series_idx, window_idx] = gt

            ctx_len = ctx.shape[-1]
            context_array[series_idx, window_idx, :, :ctx_len] = ctx

        if use_val:
             print("    (No results saved - validation data used for hyperparameter selection)")
        else:
            # Save Results
            ds_config = f"{dataset_name}/{term}"
            model_hyperparams = {
                "model": f"moirai-moe-{model_size}",
                "patch_size": patch_size,
                "context_length": context_length,
            }

            metadata = save_window_predictions(
                dataset=dataset,
                predictor=predictor,
                ds_config=ds_config,
                output_base_dir=output_dir,
                seasonality=season_length,
                model_hyperparams=model_hyperparams,
            )
            print(f"  Completed: {metadata.get('num_series', 'N/A')} series × {metadata.get('num_windows', 'N/A')} windows")
            print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Moirai-MoE experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["TSBench_IMOS_v2/15T"],
                        help="Dataset name(s). Can be a single dataset, multiple datasets, or 'all_datasets'")
    parser.add_argument("--terms", type=str, nargs="+", default=["short", "medium", "long"],
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate")
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["small", "base"],
                        help="Moirai-MoE model size")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples for probabilistic forecasting")
    parser.add_argument("--context-length", type=int, default=4000,
                        help="Maximum context length")
    parser.add_argument("--patch-size", type=int, default=16,
                        help="Patch size (default 16 for MoE)")
    parser.add_argument("--cuda-device", type=str, default="0",
                        help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--val", action="store_true",
                        help="Evaluate on validation data (no saving)")

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
            run_moirai_moe_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                model_size=args.model_size,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                context_length=args.context_length,
                patch_size=args.patch_size,
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
