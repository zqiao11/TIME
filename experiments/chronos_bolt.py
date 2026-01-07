import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from chronos import BaseChronosPipeline
from gluonts.time_feature import get_seasonality

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)

load_dotenv()

# --- 辅助类 1: 包装多变量结果 (修正版) ---
class MultivariateForecast:
    """
    将 Chronos-Bolt 的输出包装成 Timebench 可用的对象。
    策略：计算分位数的均值作为单样本，利用广播机制适配 num_samples=100。
    """
    def __init__(self, quantile_input):
        # quantile_input 可以是:
        # 1. 单个 Tensor: (num_quantiles, prediction_length) [单变量]
        # 2. Tensor 列表: [Tensor(Q, T), Tensor(Q, T), ...] [多变量拆解后]

        # [修改 1] 处理输入类型: List[Tensor] -> Numpy Array
        if isinstance(quantile_input, list):
            # 列表中的每个元素应该是 (num_quantiles, prediction_length) 的 tensor
            # 我们需要把它们堆叠成 (num_quantiles, num_variates, prediction_length)
            # 注意：Chronos Bolt 输出是 (Q, T)

            # 先转 CPU Numpy
            np_list = [t.cpu().float().numpy() if isinstance(t, torch.Tensor) else t for t in quantile_input]

            # Stack along axis 1 (variates)
            # data shape: (num_quantiles, num_variates, prediction_length)
            data = np.stack(np_list, axis=1)

        elif isinstance(quantile_input, torch.Tensor):
            data = quantile_input.cpu().float().numpy()
            # 如果是单个 Tensor，可能是 (Q, T) 或 (Q, V, T)
            if data.ndim == 2:
                # (Q, T) -> (Q, 1, T)
                data = data[:, np.newaxis, :]
        else:
            # 假设已经是 numpy array
            data = quantile_input
            if data.ndim == 2:
                data = data[:, np.newaxis, :]

        # [修改 2] 此时 data 必然是 (Q, V, T) 格式的 Numpy Array

        # Bolt 输出的是分位数，我们取这些分位数的均值作为对分布的最佳点估计
        # Shape 变为: (num_variates, prediction_length)
        # axis=0 是 quantile 维度
        mean_val = np.mean(data, axis=0)

        # 设置 _mean
        self._mean = mean_val

        # 设置 _samples
        # 为了适配 Timebench 的 (100, V, T) 数组，我们将 sample shape 设为 (1, V, T)。
        # Numpy 在赋值时会将 (1, V, T) 自动广播(复制)填充到 (10, V, T)。
        self._samples = mean_val[np.newaxis, :, :]

    @property
    def samples(self):
        return self._samples

    @property
    def mean(self):
        return self._mean

# --- 辅助类 2: Mock Predictor (保持不变) ---
class MockPredictor:
    def __init__(self, precomputed_forecasts):
        self.forecasts = precomputed_forecasts
    def predict(self, dataset_input, **kwargs):
        return self.forecasts


def run_chronos_experiment(
    dataset_name: str = "TSBench_IMOS_v2/15T",
    terms: list[str] = None,
    model_size: str = "base",
    output_dir: str | None = None,
    batch_size: int = 512,
    num_samples: int = 100,
    context_length: int = 512,
    cuda_device: str = "0",
    config_path: Path | None = None,
    use_val: bool = False,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device_map = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = ["short", "medium", "long"]

    if output_dir is None:
        output_dir = f"./output/results/chronos_bolt_{model_size}"
    os.makedirs(output_dir, exist_ok=True)

    model_map = {
        "tiny": "amazon/chronos-bolt-tiny",
        "mini": "amazon/chronos-bolt-mini",
        "small": "amazon/chronos-bolt-small",
        "base": "amazon/chronos-bolt-base",
    }
    hf_model_path = model_map.get(model_size, "amazon/chronos-bolt-base")

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {hf_model_path}")
    print(f"{'='*60}")

    for term in terms:
        print(f"\n--- Term: {term} ---")
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_split = settings.get("test_split")
        val_split = settings.get("val_split")
        print(f"  Config: prediction_length={prediction_length}")

        print(f"  Initializing Chronos Bolt pipeline ({hf_model_path})...")
        pipeline = BaseChronosPipeline.from_pretrained(
            hf_model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )

        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=False,
            prediction_length=prediction_length,
            test_split=test_split,
            val_split=val_split,
        )

        if use_val:
            eval_data = dataset.val_data
        else:
            eval_data = dataset.test_data

        season_length = get_seasonality(dataset.freq)

        # --- 手动执行预测流程 ---
        print(f"  Running predictions (manual batch processing)...")

        # 1. 准备数据
        flat_context_tensors = []
        instance_dims = []

        for d in eval_data.input:
            target = np.asarray(d["target"])

            # --- [FIX START] Manually truncate context ---
            # 确保输入长度不超过 context_length，从末尾截取
            seq_len = target.shape[-1]
            if seq_len > context_length:
                target = target[..., -context_length:]
            # --- [FIX END] ---

            if target.ndim == 2:
                num_vars = target.shape[0]
                for v in range(num_vars):
                    flat_context_tensors.append(torch.tensor(target[v]))
                instance_dims.append(num_vars)
            else:
                flat_context_tensors.append(torch.tensor(target))
                instance_dims.append(1)

        # 2. 批量推理
        flat_forecast_tensors = []

        if batch_size > 0:
            total_items = len(flat_context_tensors)
            # Bolt 推理非常快，为了进度可见，可以加上简单的计数或者使用 tqdm (如果可用)
            for start in range(0, total_items, batch_size):
                end = min(start + batch_size, total_items)
                batch_contexts = flat_context_tensors[start:end]

                with torch.no_grad():
                    # Bolt 输出 shape: (Batch, Quantiles, Time)
                    batch_output = pipeline.predict(
                        inputs=batch_contexts,
                        prediction_length=prediction_length
                    )

                # 将 batch tensor 切片后存入列表 (转回 CPU 以节省显存)
                batch_output = batch_output.cpu()
                for i in range(batch_output.shape[0]):
                    flat_forecast_tensors.append(batch_output[i])

        # 3. 组装结果
        forecasts = []
        cursor = 0
        for dim in instance_dims:
            # 取出当前实例对应的 D 个 Tensor
            # component_tensors 是 List[Tensor]，每个 shape (Q, T)
            component_tensors = flat_forecast_tensors[cursor : cursor + dim]
            cursor += dim

            # 正确调用: 传入 Tensor 列表
            forecasts.append(MultivariateForecast(component_tensors))

        print(f"  Predictions generated. Merged instances: {len(forecasts)}")

        if not use_val:
            ds_config = f"{dataset_name}/{term}"
            model_hyperparams = {
                "model": f"chronos-bolt-{model_size}",
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
    parser = argparse.ArgumentParser(description="Run Chronos Bolt experiments")
    parser.add_argument("--dataset", type=str, default="IMOS/15T", help="Dataset name")
    parser.add_argument("--terms", type=str, nargs="+", default=["short", "medium", "long"], choices=["short", "medium", "long"], help="Terms to evaluate")
    parser.add_argument("--model-size", type=str, default="base", choices=["tiny", "mini", "small", "base"], help="Chronos model size")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples (ignored/broadcasted)")
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