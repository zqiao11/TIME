##需要首先安装包：pip install chronos-forecasting


import argparse
import os
import sys
from pathlib import Path
import torch

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from dotenv import load_dotenv
from chronos import ChronosPipeline
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

# --- 辅助类 1: 包装多变量结果 ---
class MultivariateForecast:
    """
    将 Chronos 的多个单变量预测结果包装成类似 Moirai 的多变量预测对象。
    """
    def __init__(self, samples_list):
        # samples_list elements shape: (num_samples, prediction_length)
        # Stack to: (num_samples, num_variates, prediction_length)
        np_samples = [s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in samples_list]
        self._samples = np.stack(np_samples, axis=1)
        # Mean shape: (num_variates, prediction_length)
        self._mean = np.mean(self._samples, axis=0)

    @property
    def samples(self):
        return self._samples

    @property
    def mean(self):
        return self._mean
    
    def cpu(self):
        return self

# --- 辅助类 2: 欺骗 save_window_predictions 的假 Predictor ---
class MockPredictor:
    """
    这是一个假的 Predictor。
    它的作用是当 save_window_predictions 调用 .predict() 时，
    直接返回我们已经计算好并整理好的 forecasts 列表，
    从而跳过 Chronos 的真实推理，并避免类型错误。
    """
    def __init__(self, precomputed_forecasts):
        self.forecasts = precomputed_forecasts

    def predict(self, dataset_input, **kwargs):
        # 忽略传入的 dataset_input（因为格式不对且不需要再跑一遍）
        # 直接返回预先算好的结果
        return self.forecasts


def run_chronos_experiment(
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
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device_map = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset configuration
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = ["short", "medium", "long"]

    if output_dir is None:
        output_dir = f"./output/results/chronos_{model_size}"

    os.makedirs(output_dir, exist_ok=True)

    model_map = {
        "tiny": "amazon/chronos-t5-tiny",
        "mini": "amazon/chronos-t5-mini",
        "small": "amazon/chronos-t5-small",
        "base": "amazon/chronos-t5-base",
        "large": "amazon/chronos-t5-large",
    }
    hf_model_path = model_map.get(model_size, "amazon/chronos-t5-small")

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {hf_model_path}")
    print(f"Terms: {terms}")
    print(f"Evaluation on: {'Validation data (no saving)' if use_val else 'Test data'}")
    print(f"{'='*60}")

    for term in terms:
        print(f"\n--- Term: {term} ---")

        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_split = settings.get("test_split")
        val_split = settings.get("val_split")

        print(f"  Config: prediction_length={prediction_length}")

        print(f"  Initializing Chronos pipeline ({hf_model_path})...")
        real_predictor = ChronosPipeline.from_pretrained(
            hf_model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )

        # 保持 to_univariate=False，以便 metrics 正常计算
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

        season_length = get_seasonality(dataset.freq)

        # --- 手动执行预测流程 (为了处理多变量) ---
        print(f"  Running predictions (manual batch processing)...")

        # 1. 拆解多变量输入
        flat_context_tensors = []
        instance_dims = []
        
        for d in eval_data.input:
            target = np.asarray(d["target"])
            
            # --- [FIX START] Manually truncate context ---
            # 确保输入长度不超过 context_length，从末尾截取
            # target 可能是 (Time,) 或 (Variates, Time)，shape[-1] 总是时间维度
            seq_len = target.shape[-1]
            if context_length is not None and seq_len > context_length:
                target = target[..., -context_length:]
            # --- [FIX END] ---

            if target.ndim == 2:
                # (dim, time)
                num_vars = target.shape[0]
                for v in range(num_vars):
                    flat_context_tensors.append(torch.tensor(target[v]))
                instance_dims.append(num_vars)
            else:
                # (time,)
                flat_context_tensors.append(torch.tensor(target))
                instance_dims.append(1)

        # 2. 批量推理
        flat_forecasts = []
        if batch_size > 0:
            for start in range(0, len(flat_context_tensors), batch_size):
                batch_contexts = flat_context_tensors[start:start + batch_size]
                # 这里调用真正的 Chronos 进行推理
                batch_forecasts = real_predictor.predict(
                    batch_contexts,
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                )

                flat_forecasts.extend(list(batch_forecasts))

        # 3. 组装结果 (Re-assemble)
        forecasts = []
        cursor = 0
        for dim in instance_dims:
            component_forecasts = flat_forecasts[cursor : cursor + dim]
            cursor += dim
            # 包装成 Moirai 风格的对象
            forecasts.append(MultivariateForecast(component_forecasts))
        
        print(f"  Predictions generated. Merged instances: {len(forecasts)}")

        # --- 手动计算 Metrics (用于控制台输出) ---
        # ... (此处省略部分冗余的 metrics 打印代码，重点是下面的保存逻辑) ... 
        # 你可以保留你原来的 metrics 计算逻辑用于 verify

        if not use_val:
            ds_config = f"{dataset_name}/{term}"
            model_hyperparams = {
                "model": f"chronos-{model_size}",
                "context_length": context_length,
            }

            # --- [关键修改] 使用 MockPredictor ---
            # 1. 创建 Mock 对象，装入算好的 forecasts
            mock_predictor = MockPredictor(forecasts)

            # 2. 调用 save_window_predictions
            # save_window_predictions 内部会调用 mock_predictor.predict(eval_data.input)
            # mock_predictor 会忽略 eval_data.input，直接返回 forecasts
            # 从而完美避开类型错误和重复计算
            metadata = save_window_predictions(
                dataset=dataset,
                predictor=mock_predictor, # <--- 传入 Mock 对象
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
    parser = argparse.ArgumentParser(description="Run Chronos experiments")
    parser.add_argument("--dataset", type=str, default="IMOS/15T", help="Dataset name")
    parser.add_argument("--terms", type=str, nargs="+", default=["short", "medium", "long"], choices=["short", "medium", "long"], help="Terms to evaluate")
    parser.add_argument("--model-size", type=str, default="tiny", choices=["tiny", "mini", "small", "base", "large"], help="Chronos model size")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for prediction")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples for probabilistic forecasting")
    parser.add_argument("--context-length", type=int, default=4000, help="Maximum context length")
    parser.add_argument("--cuda-device", type=str, default="0", help="CUDA device ID")
    parser.add_argument("--config", type=str, default=None, help="Path to datasets.yaml config file")
    parser.add_argument("--val", action="store_true", help="Evaluate on validation data")

    args = parser.parse_args()

    run_chronos_experiment(
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