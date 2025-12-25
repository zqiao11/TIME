"""
VisionTS model experiments for time series forecasting.
(Corrected for variable length inputs and VisionTS return values)
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
import einops

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from visionts import VisionTS, freq_to_seasonality_list
from gluonts.time_feature import get_seasonality

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.metrics import compute_per_window_metrics

# Load environment variables
load_dotenv()

# --- Helper Class 1: Wrap Point Forecasts as Samples ---
class MultivariateForecast:
    def __init__(self, point_forecast, num_samples=100):
        self._mean = point_forecast
        self._samples = np.expand_dims(point_forecast, axis=0).repeat(num_samples, axis=0)

    @property
    def samples(self): return self._samples
    @property
    def mean(self): return self._mean
    def cpu(self): return self

class MultivariateCompositeForecast:
    def __init__(self, forecast_list):
        self._samples = np.stack([f.samples for f in forecast_list], axis=1)
        self._mean = np.stack([f.mean for f in forecast_list], axis=0)
    
    @property
    def samples(self): return self._samples
    @property
    def mean(self): return self._mean

# --- Helper Class 2: Mock Predictor ---
class MockPredictor:
    def __init__(self, forecasts): self.forecasts = forecasts
    def predict(self, *args, **kwargs): return self.forecasts


def run_visionts_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] = None,
    model_size: str = "mae_base",
    output_dir: str | None = None,
    batch_size: int = 32,
    num_samples: int = 100,
    context_length: int = 4000,
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
):
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = ["short", "medium", "long"]

    if output_dir is None:
        output_dir = f"./output/results/visionts_{model_size}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: VisionTS-{model_size}")
    print(f"Terms: {terms}")
    print(f"Evaluation on: {'Validation data' if use_val else 'Test data'}")
    print(f"{'='*60}")

    # Initialize VisionTS Model
    print(f"  Initializing VisionTS-{model_size} model...")
    model = VisionTS(model_size, ckpt_dir='./ckpt/').to(device)
    model.eval()

    for term in terms:
        print(f"\n--- Term: {term} ---")

        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_split = settings.get("test_split")
        val_split = settings.get("val_split")

        print(f"  Config: prediction_length={prediction_length}")

        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=False, 
            prediction_length=prediction_length,
            test_split=test_split,
            val_split=val_split,
        )

        if use_val:
            data_length = int(dataset.val_split * dataset._min_series_length)
            num_windows = dataset.val_windows
            split_name = "Val split"
            split_value = dataset.val_split
            eval_data = dataset.val_data
        else:
            data_length = int(dataset.test_split * dataset._min_series_length)
            num_windows = dataset.windows
            split_name = "Test split"
            split_value = dataset.test_split
            eval_data = dataset.test_data

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - Series length: min={dataset._min_series_length}, max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
        print(f"    - {split_name}: {split_value} ({data_length} steps)")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        periodicity_list = freq_to_seasonality_list(dataset.freq)
        periodicity = periodicity_list[0]
        print(f"  Derived Periodicity: {periodicity}")

        # --- Manual Prediction Loop ---
        print(f"  Running predictions (manual batch processing)...")

        flat_context_tensors = []
        instance_dims = []
        
        # Flatten input
        for d in eval_data.input:
            target = np.asarray(d["target"])


            if target.ndim == 2:
                num_vars = target.shape[0]
                for v in range(num_vars):
                    # Transpose to (Time, 1) for VisionTS
                    tensor = torch.tensor(target[v]).float().unsqueeze(-1)
                    flat_context_tensors.append(tensor)
                instance_dims.append(num_vars)
            else:
                tensor = torch.tensor(target).float().unsqueeze(-1)
                flat_context_tensors.append(tensor)
                instance_dims.append(1)

        # Batch Inference
        flat_forecasts = []
        print(f"  Starting sequential inference on {len(flat_context_tensors)} samples...")
        
        # Loop one by one to handle variable lengths
        for i, input_tensor in enumerate(flat_context_tensors):
            # 1. Add Batch Dimension: [Time, 1] -> [1, Time, 1]
            single_input = input_tensor.unsqueeze(0).to(device)
            
            # 2. Get actual length of this sample
            curr_ctx_len = single_input.shape[1]
            
            # 3. Dynamic Config Update (Crucial for VisionTS to set patches correctly)
            # Note: We pass args positionally/keyword based on the previous working code
            model.update_config(
                curr_ctx_len, 
                prediction_length, 
                periodicity=periodicity,
                align_const=1, 
                norm_const=0.4
            )
            
            # 4. Inference
            with torch.no_grad():
                output = model(single_input, export_image=False)
                
                # Handle return tuple
                if isinstance(output, tuple):
                    y_pred = output[0]
                else:
                    y_pred = output
            
            # 5. Extract Result
            # y_pred shape: [1, pred_len, 1]
            y_pred_np = y_pred.cpu().numpy()
            
            # Append 1D array [pred_len]
            flat_forecasts.append(y_pred_np[0, :, 0])

            # Print progress
            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(flat_context_tensors)}", end="\r")
        
        print("\n  Inference done.")


        # 3. Re-assemble Multivariate Results
        forecasts = []
        cursor = 0
        for dim in instance_dims:
            component_forecasts_np = flat_forecasts[cursor : cursor + dim]
            cursor += dim
            uni_wrappers = [MultivariateForecast(f, num_samples=num_samples) for f in component_forecasts_np]
            forecasts.append(MultivariateCompositeForecast(uni_wrappers))
        
        print(f"  Predictions generated. Merged instances: {len(forecasts)}")

        # --- Validation (Manual Calc) vs Test (Save) ---
        if use_val:
            print("    [Validation Mode] Computing metrics manually...")
            
            ground_truths = []
            contexts = []
            for inp, label in eval_data:
                ground_truths.append(label["target"])
                contexts.append(inp["target"])

            num_total_instances = len(forecasts)
            num_series = num_total_instances // num_windows
            num_variates = dataset.target_dim

            predictions_mean = np.zeros((num_series, num_windows, num_variates, prediction_length))
            predictions_samples = np.zeros((num_series, num_windows, num_samples, num_variates, prediction_length))
            ground_truth = np.zeros((num_series, num_windows, num_variates, prediction_length))
            
            max_context_len = max(ctx.shape[-1] for ctx in contexts) if contexts else 0
            context_array = np.full((num_series, num_windows, num_variates, max_context_len), np.nan)

            for idx, (fc, gt, ctx) in enumerate(zip(forecasts, ground_truths, contexts)):
                series_idx = idx // num_windows
                window_idx = idx % num_windows

                fc_mean = fc.mean
                fc_samples = fc.samples

                if fc_mean.ndim == 1: fc_mean = fc_mean[np.newaxis, :]
                if fc_samples.ndim == 2: fc_samples = fc_samples[:, np.newaxis, :]
                if gt.ndim == 1: gt = gt[np.newaxis, :]
                if gt.shape[0] == prediction_length and gt.shape[1] == num_variates: gt = gt.T
                if ctx.ndim == 1: ctx = ctx[np.newaxis, :]
                elif ctx.shape[0] != num_variates: ctx = ctx.T

                predictions_mean[series_idx, window_idx] = fc_mean
                predictions_samples[series_idx, window_idx] = fc_samples
                ground_truth[series_idx, window_idx] = gt
                
                ctx_len = ctx.shape[-1]
                context_array[series_idx, window_idx, :, :ctx_len] = ctx

            metrics = compute_per_window_metrics(
                predictions_mean=predictions_mean,
                predictions_samples=predictions_samples,
                ground_truth=ground_truth,
                context=context_array,
                seasonality=periodicity,
            )
            print("    Metrics summary (averaged over all series/windows/variates):")
            for metric_name, metric_values in metrics.items():
                mean_val = np.nanmean(metric_values)
                print(f"      {metric_name}: {mean_val:.4f}")
            print("    (No results saved - validation mode)")

        else:
            ds_config = f"{dataset_name}/{term}"
            model_hyperparams = {
                "model": f"visionts-{model_size}",
                "context_length": context_length,
                "periodicity": periodicity
            }

            mock_predictor = MockPredictor(forecasts)

            metadata = save_window_predictions(
                dataset=dataset,
                predictor=mock_predictor,
                ds_config=ds_config,
                output_base_dir=output_dir,
                seasonality=periodicity,
                model_hyperparams=model_hyperparams,
            )
            
            print(f"  Completed: {metadata['num_series']} series × {metadata['num_windows']} windows")
            print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run VisionTS experiments")
    parser.add_argument("--dataset", type=str, default="IMOS/15T", help="Dataset name")
    parser.add_argument("--terms", type=str, nargs="+", default=["short", "medium", "long"], choices=["short", "medium", "long"], help="Terms to evaluate")
    parser.add_argument("--model-size", type=str, default="mae_base", choices=["mae_base", "mae_large", "mae_huge"], help="VisionTS model size")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for prediction")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples (broadcasted for point forecast)")
    parser.add_argument("--context-length", type=int, default=4000, help="Maximum context length")
    parser.add_argument("--cuda-device", type=str, default="0", help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None, help="Path to datasets.yaml config file")
    parser.add_argument("--val", action="store_true", help="Evaluate on validation data")

    args = parser.parse_args()

    run_visionts_experiment(
        dataset_name=args.dataset,
        terms=args.terms,
        model_size=args.model_size,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        context_length=args.context_length,
        cuda_device=args.cuda_device,
        config_path=Path(args.config) if args.config else None,
        use_val=args.val,
    )

if __name__ == "__main__":
    main()