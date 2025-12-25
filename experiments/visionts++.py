"""
VisionTS++ model experiments for time series forecasting.
(Supports native multivariate input and quantile outputs)
(Adaptive: Batched for Multivariate, Sequential for Univariate/Variable Lengths)

Usage:
    python experiments/visiontspp.py
    python experiments/visiontspp.py --model-size base
    python experiments/visiontspp.py --dataset "volicity/10T" --terms short
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
from huggingface_hub import snapshot_download
import random

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from visionts import VisionTSpp, freq_to_seasonality_list
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

# --- Helper Class: Wrap Quantiles as Samples ---
class MultivariateForecast:
    """
    Wraps VisionTS++ quantile forecasts into sample-based format via interpolation.
    """
    def __init__(self, quantiles_tensor, mean_tensor, target_num_samples=100):
        # 1. Handle Mean
        # mean_tensor shape: [Variates, Prediction_Length]
        self._mean = mean_tensor.cpu().float().numpy() if isinstance(mean_tensor, torch.Tensor) else mean_tensor

        # 2. Handle Quantiles -> Samples (via interpolation)
        # quantiles_tensor shape: [Num_Quantiles, Variates, Prediction_Length]
        q_data = quantiles_tensor.cpu().float().numpy() if isinstance(quantiles_tensor, torch.Tensor) else quantiles_tensor
        
        num_quantiles, num_vars, pred_len = q_data.shape

        if num_quantiles != target_num_samples:
            target_indices = np.linspace(0, num_quantiles - 1, target_num_samples)
            source_indices = np.arange(num_quantiles)
            
            # Initialize result array (Samples, Variates, Pred_Length)
            resampled = np.zeros((target_num_samples, num_vars, pred_len))
            
            for v in range(num_vars):
                for t in range(pred_len):
                    # Interpolate along the quantile dimension (dim 0)
                    resampled[:, v, t] = np.interp(target_indices, source_indices, q_data[:, v, t])
            
            self._samples = resampled
        else:
            self._samples = q_data

    @property
    def samples(self): return self._samples
    @property
    def mean(self): return self._mean
    def cpu(self): return self

# --- Helper Class 2: Mock Predictor ---
class MockPredictor:
    def __init__(self, forecasts): self.forecasts = forecasts
    def predict(self, *args, **kwargs): return self.forecasts


def run_visiontspp_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] = None,
    model_size: str = "base", # 'base' or 'large'
    output_dir: str | None = None,
    batch_size: int = 32,
    num_samples: int = 100,
    context_length: int = 4000,
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
    seed: int = 42,
):
    # Set Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset configuration
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = ["short", "medium", "long"]

    if output_dir is None:
        output_dir = f"./output/results/visiontspp_{model_size}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: VisionTS++ ({model_size})")
    print(f"Terms: {terms}")
    print(f"Evaluation on: {'Validation data' if use_val else 'Test data'}")
    print(f"{'='*60}")

    # --- Initialize VisionTS++ Model ---
    print(f"  Preparing VisionTS++ model...")
    
    # Define paths
    local_dir = "./hf_models/VisionTSpp"
    if model_size == "base":
        arch = 'mae_base'
        ckpt_filename = "visiontspp_base.ckpt"
    else: # large
        arch = 'mae_large'
        ckpt_filename = "visiontspp_large.ckpt"
    
    ckpt_path = os.path.join(local_dir, ckpt_filename)

    # Download if missing
    if not os.path.exists(ckpt_path):
        print(f"  Downloading checkpoint to {local_dir}...")
        try:
            snapshot_download(
                repo_id="Lefei/VisionTSpp",
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
        except Exception as e:
            print(f"  Error downloading model: {e}")
            print("  Please ensure internet access or manually place checkpoints in ./hf_models/VisionTSpp/")
            return

    print(f"  Loading model from {ckpt_path}...")
    model = VisionTSpp(
        arch,
        ckpt_path=ckpt_path,
        quantile=True,
        clip_input=True,
        complete_no_clip=False,
        color=True
    ).to(device)
    model.eval()

    for term in terms:
        print(f"\n--- Term: {term} ---")

        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_split = settings.get("test_split")
        val_split = settings.get("val_split")

        print(f"  Config: prediction_length={prediction_length}")

        # [Important] Keep to_univariate=False because VisionTS++ supports multivariate input
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
            eval_data = dataset.val_data
        else:
            data_length = int(dataset.test_split * dataset._min_series_length)
            num_windows = dataset.windows
            eval_data = dataset.test_data

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")
        print(f"    - Target Dim (Variates): {dataset.target_dim}")


        periodicity_list = freq_to_seasonality_list(dataset.freq)
        periodicity = periodicity_list[0]
        print(f"  Derived Periodicity: {periodicity}")
  

        # --- Prediction Logic ---
        print(f"  Preparing input data...")

        # 1. Prepare Data
        context_list = []
        is_multivariate_dataset = (dataset.target_dim > 1)

        for d in eval_data.input:
            target = np.asarray(d["target"]) # [V, T] or [T]
            
            # Truncate left to max context_length
            seq_len = target.shape[-1]
            if seq_len > context_length:
                target = target[..., -context_length:]

            # Normalize shape to [Time, Variates]
            if target.ndim == 1:
                target = target[np.newaxis, :] # [1, T]
            
            target_t = target.T # [T, V]
            context_list.append(torch.tensor(target_t).float())

        forecasts = []

        # ---------------------------------------------------------------------
        # [Strategy Branching]
        # ---------------------------------------------------------------------
        if is_multivariate_dataset:
            # === STRATEGY A: Native Multivariate ===
            print(f"  [Strategy] Detected Multivariate (dim={dataset.target_dim}). Trying Batch Inference...")
            
            if batch_size > 0:
                total_items = len(context_list)
                for start in range(0, total_items, batch_size):
                    end = min(start + batch_size, total_items)
                    batch_list = context_list[start:end]
                    
                    # [FIX START] Enhanced Fallback Logic for Variable Lengths
                    try:
                        # 尝试堆叠。如果该 batch 内所有样本长度一致，成功。
                        # 如果不一致（例如 volicity），会抛出 RuntimeError。
                        batch_input = torch.stack(batch_list).to(device) # [Batch, Time, Variates]
                        
                        # --- Path A1: Successful Batching ---
                        curr_ctx_len = batch_input.shape[1]
                        nvars_input = batch_input.shape[2] 
                        
                        model.update_config(
                            context_len=curr_ctx_len, 
                            pred_len=prediction_length, 
                            periodicity=periodicity,
                            num_patch_input=7, 
                            padding_mode='constant'
                        )
                        
                        color_list = [i % 3 for i in range(nvars_input)]
                        
                        with torch.no_grad():
                            output = model(batch_input, export_image=False, color_list=color_list)
                        
                        if isinstance(output, tuple): preds_data = output[0]
                        else: preds_data = output

                        medians = preds_data[0].cpu().numpy() 
                        quantiles_list = preds_data[1] 
                        q_np_list = [q.cpu().numpy() for q in quantiles_list]
                        
                        for i in range(medians.shape[0]):
                            med_sample = medians[i].T 
                            qs_sample = [q[i] for q in q_np_list]
                            qs_stacked = np.stack(qs_sample, axis=0)
                            med_expanded = np.expand_dims(medians[i], axis=0)
                            full_quantiles = np.concatenate([qs_stacked[:4], med_expanded, qs_stacked[4:]], axis=0)
                            full_quantiles = full_quantiles.transpose(0, 2, 1)
                            forecasts.append(MultivariateForecast(full_quantiles, med_sample, target_num_samples=num_samples))

                    except RuntimeError as e:
                        # --- Path A2: Fallback to Sequential due to variable lengths ---
                        if "stack" in str(e):
                            # print(f"    [Batch {start//batch_size}] Variable lengths detected. Switching to sequential processing.")
                            # 对当前 batch_list 里的数据进行逐个处理
                            for item in batch_list:
                                single_input = item.unsqueeze(0).to(device) # [1, T, V]
                                curr_ctx_len = single_input.shape[1]
                                nvars_input = single_input.shape[2]
                                
                                model.update_config(
                                    context_len=curr_ctx_len, 
                                    pred_len=prediction_length, 
                                    periodicity=periodicity,
                                    num_patch_input=7, 
                                    padding_mode='constant'
                                )
                                color_list = [i % 3 for i in range(nvars_input)]
                                
                                with torch.no_grad():
                                    output = model(single_input, export_image=False, color_list=color_list)
                                
                                if isinstance(output, tuple): preds_data = output[0]
                                else: preds_data = output

                                medians = preds_data[0].cpu().numpy() 
                                quantiles_list = preds_data[1] 
                                q_np_list = [q.cpu().numpy() for q in quantiles_list]
                                
                                # Process single item
                                med_sample = medians[0].T 
                                qs_sample = [q[0] for q in q_np_list]
                                qs_stacked = np.stack(qs_sample, axis=0)
                                med_expanded = np.expand_dims(medians[0], axis=0)
                                full_quantiles = np.concatenate([qs_stacked[:4], med_expanded, qs_stacked[4:]], axis=0)
                                full_quantiles = full_quantiles.transpose(0, 2, 1)
                                forecasts.append(MultivariateForecast(full_quantiles, med_sample, target_num_samples=num_samples))
                        else:
                            raise e 
                    # [FIX END]

        else:
            # === STRATEGY B: Univariate (Sequential) ===
            print(f"  [Strategy] Detected Univariate. Using Sequential Inference with Random Coloring.")
            
            for i, context_tensor in enumerate(context_list):
                single_input = context_tensor.unsqueeze(0).to(device)
                curr_ctx_len = single_input.shape[1]
                
                model.update_config(
                    context_len=curr_ctx_len, 
                    pred_len=prediction_length, 
                    periodicity=periodicity,
                    num_patch_input=7, 
                    padding_mode='constant'
                )
                
                rand_color = random.randint(0, 2)
                color_list = [rand_color] 
                
                with torch.no_grad():
                    output = model(single_input, export_image=False, color_list=color_list)
                
                if isinstance(output, tuple): preds_data = output[0]
                else: preds_data = output

                medians = preds_data[0].cpu().numpy() 
                quantiles_list = preds_data[1] 
                q_np_list = [q.cpu().numpy() for q in quantiles_list]
                
                med_sample = medians[0].T 
                qs_sample = [q[0] for q in q_np_list]
                qs_stacked = np.stack(qs_sample, axis=0)
                med_expanded = np.expand_dims(medians[0], axis=0)
                full_quantiles = np.concatenate([qs_stacked[:4], med_expanded, qs_stacked[4:]], axis=0)
                full_quantiles = full_quantiles.transpose(0, 2, 1)
                forecasts.append(MultivariateForecast(full_quantiles, med_sample, target_num_samples=num_samples))

                if (i + 1) % 50 == 0:
                    print(f"    Processed {i + 1}/{len(context_list)}", end="\r")
            print("")

        print(f"  Predictions generated. Instances: {len(forecasts)}")

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
                "model": f"visiontspp-{model_size}",
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
    parser = argparse.ArgumentParser(description="Run VisionTS++ experiments")
    parser.add_argument("--dataset", type=str, default="IMOS/15T", help="Dataset name")
    parser.add_argument("--terms", type=str, nargs="+", default=["short", "medium", "long"], choices=["short", "medium", "long"], help="Terms to evaluate")
    parser.add_argument("--model-size", type=str, default="base", choices=["base", "large"], help="VisionTS++ model size")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for prediction")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples (interpolated from quantiles)")
    parser.add_argument("--context-length", type=int, default=4000, help="Maximum context length")
    parser.add_argument("--cuda-device", type=str, default="0", help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None, help="Path to datasets.yaml config file")
    parser.add_argument("--val", action="store_true", help="Evaluate on validation data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    run_visiontspp_experiment(
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
        seed=args.seed
    )

if __name__ == "__main__":
    main()