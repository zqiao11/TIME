"""
TimeMoE model experiments for time series forecasting.
(Based on Maple728/TimeMoE)
(Sequential Inference Mode - No Batching)

Usage:
    python experiments/timemoe.py
    python experiments/timemoe.py --model-size base
    python experiments/timemoe.py --dataset "TSBench_IMOS_v2/15T" --terms short medium long
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.metrics import compute_per_window_metrics

load_dotenv()

# --- Helper Class 1: Wrap Multivariate Results ---
class MultivariateForecast:
    """
    Wraps TimeMoE point forecasts into sample format.
    Since TimeMoE is deterministic, results are duplicated to create samples.
    """
    def __init__(self, point_forecast, target_num_samples=100):
        # point_forecast shape: (Prediction_Length,) or (Variates, Prediction_Length)
        
        # 1. Convert to CPU Numpy
        data = point_forecast.cpu().float().numpy() if isinstance(point_forecast, torch.Tensor) else point_forecast
        
        # 2. Unified dim to (Variates, Prediction_Length)
        if data.ndim == 1:
            data = data[np.newaxis, :] # (1, T)
            
        self._mean = data
        num_vars, pred_len = data.shape

        # 3. Construct Samples: (Num_Samples, Variates, Prediction_Length)
        # Broadcast Mean value
        self._samples = np.repeat(data[np.newaxis, :, :], target_num_samples, axis=0)

    @property
    def samples(self): return self._samples
    @property
    def mean(self): return self._mean
    def cpu(self): return self

# --- Helper Class 2: Mock Predictor ---
class MockPredictor:
    def __init__(self, forecasts): self.forecasts = forecasts
    def predict(self, dataset_input, **kwargs): return self.forecasts

# --- Helper Function: TimeMoE Inference Logic ---
def timemoe_inference(model, batch_input, prediction_length):
    """
    Executes TimeMoE Normalize -> Generate -> Denormalize flow
    batch_input: [Batch, Context_Length] (Here Batch=1)
    """
    # 1. Normalize (Reversible Instance Normalization)
    
    mean = batch_input.mean(dim=-1, keepdim=True)
    std = batch_input.std(dim=-1, keepdim=True) + 1e-5 # Prevent division by zero
    normed_seqs = (batch_input - mean) / std

    # 2. Forecast (Autoregressive Generation)
    output = model.generate(
        normed_seqs, 
        max_new_tokens=prediction_length,
        pad_token_id=0,     # TimeMoE usually doesn't need pad_token, kept for interface compat
        eos_token_id=None,  # Ensure generation of specific length
        do_sample=False     # Greedy search (point forecast)
    ) 
    # output shape: [Batch, Context + Pred]

    # 3. Slice Output
    normed_predictions = output[:, -prediction_length:]

    # 4. Inverse Normalize
    predictions = normed_predictions * std + mean
    
    return predictions

def run_timemoe_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] = None,
    model_size: str = "base", # 'base' (50M) or 'large' (200M)
    output_dir: str | None = None,
    # batch_size: int = 32, # [Removed] No longer needed
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
        output_dir = f"./output/results/timemoe_{model_size}"
    os.makedirs(output_dir, exist_ok=True)

    # --- [MODIFIED] Map 'base'/'large' to HF repos ---
    hf_repo_map = {
        "base": "Maple728/TimeMoE-50M",
        "large": "Maple728/TimeMoE-200M", 
    }
    model_name = hf_repo_map.get(model_size, "Maple728/TimeMoE-50M")

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name} (Size: {model_size})")
    print(f"Terms: {terms}")
    print(f"Evaluation on: {'Validation data' if use_val else 'Test data'}")
    print(f"Mode: Sequential Inference (One-by-One)")
    print(f"{'='*60}")

    # --- Initialize Model ---
    print(f"  Loading TimeMoE model ({model_name})...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval() 
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    for term in terms:
        print(f"\n--- Term: {term} ---")
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_split = settings.get("test_split")
        val_split = settings.get("val_split")
        print(f"  Config: prediction_length={prediction_length}")

        # [Important] to_univariate=True
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=True, 
            prediction_length=prediction_length,
            test_split=test_split,
            val_split=val_split,
        )

        if use_val:
            eval_data = dataset.val_data
        else:
            eval_data = dataset.test_data

        season_length = get_seasonality(dataset.freq)

        # --- Prediction Logic ---
        print(f"  Running predictions (Sequential processing)...")

        # 1. Prepare Data & Truncate Context
        flat_context_tensors = []
        instance_dims = []

        for d in eval_data.input:
            target = np.asarray(d["target"])
            
            # Manual truncate context
            seq_len = target.shape[-1]
            if context_length is not None and seq_len > context_length:
                target = target[..., -context_length:]
            
            # Convert to Tensor
            if target.ndim == 2:
                num_vars = target.shape[0]
                for v in range(num_vars):
                    flat_context_tensors.append(torch.tensor(target[v]).float())
                instance_dims.append(num_vars)
            else:
                flat_context_tensors.append(torch.tensor(target).float())
                instance_dims.append(1)

        # 2. Sequential Inference
        flat_forecasts = [] 
        total_items = len(flat_context_tensors)
        print(f"  Total univariate series: {total_items}")
        
        for i, item in enumerate(flat_context_tensors):
            # Add Batch Dim: [Time] -> [1, Time]
            single_input = item.unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Direct inference on single sequence
                pred = timemoe_inference(model, single_input, prediction_length)
            
            # pred shape: [1, Pred_Len], take first element
            flat_forecasts.append(pred[0])

            if (i + 1) % 50 == 0:
                 print(f"    Processed {i + 1}/{total_items}", end="\r")
        print("")
        
        # 3. Assemble Multivariate Results
        forecasts = []
        cursor = 0
        for dim in instance_dims:
            # Extract all variable forecasts for current Series
            component_forecasts = flat_forecasts[cursor : cursor + dim]
            cursor += dim
            
            # Stack to (Variates, Pred_Len)
            stacked_comp = torch.stack(component_forecasts) 
            
            # Wrap
            forecasts.append(MultivariateForecast(stacked_comp, target_num_samples=num_samples))

        print(f"  Predictions generated. Merged instances: {len(forecasts)}")

        # --- Save / Validate ---
        if not use_val:
            ds_config = f"{dataset_name}/{term}"
            model_hyperparams = {
                "model": f"timemoe-{model_size}",
                "context_length": context_length,
            }

            mock_predictor = MockPredictor(forecasts)

            metadata = save_window_predictions(
                dataset=dataset,
                predictor=mock_predictor,
                ds_config=ds_config,
                output_base_dir=output_dir,
                seasonality=season_length,
                model_hyperparams=model_hyperparams,
            )
            print(f"  Completed: {metadata['num_series']} series × {metadata['num_windows']} windows")
            print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Run TimeMoE experiments")
    parser.add_argument("--dataset", type=str, default="IMOS/15T", help="Dataset name")
    parser.add_argument("--terms", type=str, nargs="+", default=["short", "medium", "long"], choices=["short", "medium", "long"], help="Terms to evaluate")
    
    # --- [MODIFIED] Changed choices to base/large ---
    parser.add_argument("--model-size", type=str, default="base", choices=["base", "large"], help="TimeMoE model size: base(50M) or large(200M)")
    
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    # parser.add_argument("--batch-size", type=int, default=32, help="Batch size for prediction") # [Removed]
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples (duplicated for deterministic models)")
    parser.add_argument("--context-length", type=int, default=4000, help="Maximum context length")
    parser.add_argument("--cuda-device", type=str, default="0", help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None, help="Path to datasets.yaml config file")
    parser.add_argument("--val", action="store_true", help="Evaluate on validation data")

    args = parser.parse_args()

    run_timemoe_experiment(
        dataset_name=args.dataset,
        terms=args.terms,
        model_size=args.model_size,
        output_dir=args.output_dir,
        # batch_size=args.batch_size, # [Removed]
        num_samples=args.num_samples,
        context_length=args.context_length,
        cuda_device=args.cuda_device,
        config_path=Path(args.config) if args.config else None,
        use_val=args.val,
    )

if __name__ == "__main__":
    main()