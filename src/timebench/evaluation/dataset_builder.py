"""
Dataset builder utilities for converting time series data to Arrow format.

Reference: https://github.com/SalesforceAIResearch/uni2ts/blob/main/example/prepare_data.ipynb

Arrow Dataset Schema:
- item_id: str - 时间序列唯一标识符
- start: timestamp[s] - 起始时间戳
- freq: str - 时间频率 (如 "5T", "15T", "D")
- target: list[list[float]] - 形状 [num_dims, seq_len]
- past_feat_dynamic_real: list[list[float]] - 形状 [num_features, seq_len] (可选)
"""

from collections.abc import Generator
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value


def dataframes_to_generator(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    freq: str = "D",
    to_univariate: bool = False,
    item_prefix: str = "",
    num_dims: Optional[int] = None,
    include_past_feat: bool = False,
    num_past_feat: int = 0,
) -> tuple[Callable[[], Generator[dict[str, Any], None, None]], Features]:
    """
    Convert pandas DataFrames to a HuggingFace-compatible generator and schema.

    Parameters
    ----------
    dfs : Union[pd.DataFrame, List[pd.DataFrame]]
        One or a list of DataFrames. Each DataFrame must have a datetime index
        (or first column as timestamp) and one or more value columns.
    freq : str
        Fallback frequency if pd.infer_freq() fails.
    to_univariate : bool
        If True: each column of each DataFrame becomes a separate item (UTS mode)
        If False: each DataFrame becomes a single multivariate item (MTS mode)
    item_prefix : str
        Prefix for item_id naming.
    num_dims : int, optional
        Fixed number of dimensions for target. If None, inferred from data.
    include_past_feat : bool
        Whether to include past_feat_dynamic_real field.
    num_past_feat : int
        Number of past dynamic features (only used if include_past_feat=True).

    Returns
    -------
    tuple[Callable, Features]
        A generator function and the corresponding Features schema.
    """
    # Normalize input to list
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    # Infer num_dims from first DataFrame if not provided
    if num_dims is None and not to_univariate:
        first_df = dfs[0]
        if isinstance(first_df.index, pd.DatetimeIndex):
            num_dims = len(first_df.columns)
        else:
            num_dims = len(first_df.columns) - 1  # exclude timestamp column

    # Prepare schema
    if to_univariate:
        features = Features(dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            target=Sequence(Value("float32")),
        ))
    else:
        feature_dict = dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            target=Sequence(
                feature=Sequence(Value("float32")),
                length=num_dims
            ),
        )
        if include_past_feat and num_past_feat > 0:
            feature_dict["past_feat_dynamic_real"] = Sequence(
                feature=Sequence(Value("float32")),
                length=num_past_feat
            )
        features = Features(feature_dict)

    # Build generator
    def gen_func() -> Generator[dict[str, Any], None, None]:
        for df_idx, df in enumerate(dfs):
            # Allow timestamp either as index or first column
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
                df = df.sort_values(df.columns[0])
                df = df.set_index(df.columns[0])

            col_names = df.columns.tolist()
            infer_freq = pd.infer_freq(df.index) or freq

            if to_univariate:
                # Each variate is one item
                for i, col in enumerate(col_names):
                    item = {
                        "item_id": f"{item_prefix}_{df_idx}_{col}",
                        "start": df.index[0],
                        "freq": infer_freq,
                        "target": df[col].to_numpy(dtype=np.float32),
                    }
                    yield item
            else:
                # Each DataFrame is one multivariate item
                target = df.to_numpy(dtype=np.float32).T  # shape: (var, time)
                item = {
                    "item_id": f"{item_prefix}{df_idx}",
                    "start": df.index[0],
                    "freq": infer_freq,
                    "target": target,
                }
                if include_past_feat and num_past_feat > 0:
                    seq_len = target.shape[1]
                    item["past_feat_dynamic_real"] = [
                        [np.nan] * seq_len for _ in range(num_past_feat)
                    ]
                yield item

    return gen_func, features


def build_dataset_from_csvs(
    csv_dir: Union[str, Path],
    output_path: Union[str, Path],
    pattern: str = "*.csv",
    freq: str = "D",
    to_univariate: bool = False,
    item_prefix: str = "",
    include_past_feat: bool = False,
    num_past_feat: int = 0,
) -> Dataset:
    """
    Build Arrow dataset from a directory of CSV files.

    Parameters
    ----------
    csv_dir : str or Path
        Directory containing CSV files.
    output_path : str or Path
        Where to save the resulting dataset.
    pattern : str
        Glob pattern to match CSV files.
    freq : str
        Time frequency.
    to_univariate : bool
        Convert to univariate mode.
    item_prefix : str
        Prefix for item IDs.
    include_past_feat : bool
        Include past_feat_dynamic_real field.
    num_past_feat : int
        Number of past features.

    Returns
    -------
    Dataset
        The created HuggingFace Dataset.
    """
    csv_dir = Path(csv_dir)
    output_path = Path(output_path)

    csv_paths = sorted(csv_dir.glob(pattern))
    if not csv_paths:
        raise ValueError(f"No CSV files found matching pattern '{pattern}' in {csv_dir}")

    print(f"Found {len(csv_paths)} CSV files")

    # Load all DataFrames
    dfs = [pd.read_csv(path, parse_dates=[0]) for path in csv_paths]

    # Print statistics
    lengths = [len(df) for df in dfs]
    print(f"Time series lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.2f}")

    # Create generator and features
    gen_func, features = dataframes_to_generator(
        dfs=dfs,
        freq=freq,
        to_univariate=to_univariate,
        item_prefix=item_prefix,
        include_past_feat=include_past_feat,
        num_past_feat=num_past_feat,
    )

    # Create and save dataset
    hf_dataset = Dataset.from_generator(gen_func, features=features)
    hf_dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")
    print(f"  - Number of samples: {len(hf_dataset)}")
    print(f"  - Features: {list(hf_dataset.features.keys())}")

    return hf_dataset


# Aliases for backward compatibility
make_generator_from_df = dataframes_to_generator
convert_csvs_to_gift_eval = build_dataset_from_csvs


if __name__ == "__main__":
    # Example: Convert IMOS data
    build_dataset_from_csvs(
        csv_dir="/home/zhongzheng/TIME/data/processed_csv/IMOS/15T",
        output_path="/home/zhongzheng/TIME/data/hf_dataset/IMOS/15T",
        # pattern="item_*.csv",
        # freq="15T",
        to_univariate=False,
        item_prefix="imos_",
    )

