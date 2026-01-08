"""
计算数据集中每个series的第k个test window的起始时间戳。

根据 Dataset 类的实现:
- test split 在 offset = -prediction_length * windows 处
- windows = ceil(test_length / prediction_length)
- 每个 window 之间的 distance = prediction_length

所以第 k 个 test window (1-indexed) 的起始位置:
  start_idx = series_length - prediction_length * windows + (k-1) * prediction_length
"""

import math
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from datasets import load_from_disk
from timebench.evaluation.data import Dataset


def load_dataset_config(config_path: Optional[Path] = None) -> dict:
    """Load dataset configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "datasets.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_test_window_start_timestamp(
    dataset_with_freq: str,
    term: str = "short",
    series_idx: int = 0,
    window_idx: int = 1,
    hf_dataset_root: Optional[Path] = None,
    config_path: Optional[Path] = None,
):
    """
    计算数据集中每个series的第k个test window的起始时间戳。

    Parameters
    ----------
    dataset_with_freq : str
        数据集名称，如 "SG_PM25/H", "Finland_Traffic/15T"
    term : str
        预测范围: "short", "medium", "long"
    k : int
        test window 索引 (1-indexed)，默认为1
    hf_dataset_root : Path, optional
        HF数据集根目录
    config_path : Path, optional
        配置文件路径

    Returns
    -------
    dict
        每个 item_id 对应的第 k 个 test window 起始时间戳
    """
    # 默认路径
    if hf_dataset_root is None:
        hf_dataset_root = Path(__file__).parent / "data" / "hf_dataset"
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "datasets.yaml"

    # 加载配置
    config = load_dataset_config(config_path)
    datasets_config = config.get("datasets", {})

    if dataset_with_freq not in datasets_config:
        raise ValueError(f"Dataset '{dataset_with_freq}' not found in config.")

    dataset_config = datasets_config[dataset_with_freq]
    test_length = dataset_config["test_length"]
    prediction_length = dataset_config[term]["prediction_length"]


    dataset_obj = Dataset(
        name=dataset_with_freq,
        term=term,
        prediction_length=prediction_length,
        test_length=test_length,
    )
    # Load full time series data
    full_series = None
    train_end_idx = None
    test_window_start_idx = None
    test_window_end_idx = None

    # Get full target time series for this series
    full_target = dataset_obj.hf_dataset[series_idx]["target"]
    start_timestamp = dataset_obj.hf_dataset[series_idx]["start"]
    freq = dataset_with_freq.split("/")[1]

    # Handle multivariate case: extract specific variate
    if full_target.ndim > 1:
        full_series = full_target[0, :]  # Shape: (series_length,)
    else:
        full_series = full_target  # Shape: (series_length,)

    # Calculate train/test split point
    # Test data starts at: series_length - prediction_length * windows
    num_windows = dataset_obj.windows
    assert window_idx <= num_windows-1, f"window_idx must be less than or equal to num_windows-1: {window_idx} <= {num_windows-1}"
    series_length = len(full_series)
    train_end_idx = series_length - prediction_length * num_windows  #TODO: 这么算train_end_idx不对，最后window可能ceil，不是完整的window

    # Calculate current test window position
    test_window_start_idx = train_end_idx + window_idx * prediction_length
    test_window_end_idx = test_window_start_idx + prediction_length

    # === 新增代码开始 ===
    # 1. 确保 start_timestamp 是 pd.Timestamp 对象 (HuggingFace datasets 有时返回 datetime 或 str)
    start_ts = pd.Timestamp(start_timestamp.item())
    # start_ts = start_ts.tz_localize('UTC')  # TODO: 这里需要根据数据集的时区进行转换
    # start_ts  = start_ts.tz_convert('Asia/Shanghai')

    # 2. 将字符串频率 (如 '15T', 'H') 转换为 pandas 的 TimeOffset 对象
    time_offset = pd.tseries.frequencies.to_offset(freq)

    # 3. 计算具体的时间戳
    # 公式：绝对起始时间 + (索引偏移量 * 单个时间步长)
    window_start_ts = start_ts + (test_window_start_idx * time_offset)
    window_end_ts = start_ts + (test_window_end_idx * time_offset)
    # === 新增代码结束 ===

    return window_start_ts, window_end_ts


# window_start_ts, window_end_ts = get_test_window_start_timestamp("SG_PM25/H", "short", 0, 0)
# window_start_ts, window_end_ts = get_test_window_start_timestamp("Finland_Traffic/15T", "medium", 0, 30)
window_start_ts, window_end_ts = get_test_window_start_timestamp("SG_Weather/D", "short", 0, 0)
print(window_start_ts, window_end_ts)
