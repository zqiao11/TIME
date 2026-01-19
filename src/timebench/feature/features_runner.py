"""
Feature extraction runner for batch processing datasets.

Usage:
    python -m timebench.feature.features_runner --dataset IMOS/15T
    python -m timebench.feature.features_runner --all

Input format:
    Expects preprocessed CSV files from preprocess.py located at:
    ./data/processed_csv/{dataset}/{freq}/*.csv

    Each CSV file has format:
    - First column: timestamp
    - Other columns: variates (may include tags like [rw], [sp])
"""

import argparse
import glob
import os
import time
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from timebench.feature.features import (
    extended_stl_features,
    extract_meta_features_batch,
    preprocess_for_tsfeatures,
    safe_parse_datetime,
    tsfeatures_with_uid_freq_map,
)

# Default config path relative to this module
# features_runner.py is at src/timebench/feature/
# datasets.yaml is at src/timebench/config/
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"


# Define the desired output column order
FEATURE_COLUMNS_ORDER = [
    # Identifiers (for joining with results)
    "dataset_id",
    "series_name",
    "variate_name",
    "unique_id",
    # Meta features (from preprocess tags)
    "is_random_walk",
    "has_spike_presence",
    "x_entropy",  # Entropy of raw series (predictability/signal-to-noise)
    # Trend features (from STL)
    "trend_strength",
    "trend_stability",
    "trend_hurst",
    "trend_nonlinearity",
    "linearity",
    "curvature",
    # Seasonal features (from STL) - before residual features
    "seasonal_strength",
    "seasonal_corr",
    "seasonal_lumpiness",
    "seasonal_entropy",
    # Residual features (from STL)
    "e_acf1",
    "e_acf10",
    "e_entropy",
    "e_kurtosis",
    "e_shapiro_w",
    "spike",
    # Statistics (from preprocessing)
    "mean",
    "std",
    "missing_rate",
    "length",
    "period1",
    "period2",
    "period3",
    "p_strength1",
    "p_strength2",
    "p_strength3",
]


def parse_dataset_key(dataset_key: str) -> tuple[str, str]:
    """
    解析 datasets.yaml 中的 key 格式 '{dataset}/{freq}'

    Args:
        dataset_key: 如 'exchange_rate/D', 'ETTh1/H', 'bitbrains_rnd/5T'

    Returns:
        (dataset_name, freq): 如 ('exchange_rate', 'D')
    """
    parts = dataset_key.split('/')
    if len(parts) != 2:
        raise ValueError(f"Invalid dataset key format: {dataset_key}. Expected 'dataset/freq'")
    return parts[0], parts[1]


def find_dataset_config(datasets_config: dict, dataset_key: str) -> tuple[str, str, dict]:
    """
    在 datasets.yaml 中查找指定数据集的配置

    Args:
        datasets_config: datasets.yaml 中的 'datasets' 字典
        dataset_key: 数据集 key，格式为 '{dataset_name}/{freq}'（如 'IMOS/15T'）

    Returns:
        (dataset_key, freq, config): 完整的 key、频率、配置字典
    """
    if dataset_key in datasets_config:
        name, freq = parse_dataset_key(dataset_key)
        return dataset_key, freq, datasets_config[dataset_key]

    # 兼容旧的只传 dataset_name 的情况
    for key, config in datasets_config.items():
        name, freq = parse_dataset_key(key)
        if name == dataset_key:
            return key, freq, config

    raise ValueError(f"Dataset '{dataset_key}' not found in config")


def convert_multi_csv_to_panel(
    csv_dir: str,
    test_length: int | None = None,
    mode: str = "test"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert multiple CSV files from preprocess.py output to tsfeatures panel format.

    Args:
        csv_dir: Directory containing *.csv files
        test_length: Number of timesteps for test portion (required if mode="test")
        mode: Which portion to compute features on:
            - "full": Use entire series
            - "test": Use only the last `test_length` timesteps

    Returns:
        panel_df: DataFrame with columns ['unique_id', 'ds', 'y']
                  unique_id format: "{series_name}_{variate_name}"
        uid_info_df: DataFrame with columns ['unique_id', 'series_name', 'variate_name']
                     for later joining with features
    """
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

    if not csv_files:
        raise ValueError(f"No *.csv files found in {csv_dir}")

    all_records = []
    uid_info_records = []

    for csv_path in csv_files:
        series_name = os.path.splitext(os.path.basename(csv_path))[0]  # e.g., "item_0" or any filename

        df = pd.read_csv(csv_path, parse_dates=[0])
        time_col = df.columns[0]
        var_cols = df.columns[1:].tolist()

        # Ensure time is sorted
        df = df.sort_values(time_col).reset_index(drop=True)

        # Filter based on mode
        if mode == "test":
            if test_length is None:
                raise ValueError("test_length must be provided when mode='test'")
            # Keep only the last test_length rows
            df = df.iloc[-test_length:].reset_index(drop=True)

        # Convert each variate to panel format
        for var in var_cols:
            unique_id = f"{series_name}_{var}"
            temp = pd.DataFrame({
                "unique_id": unique_id,
                "ds": safe_parse_datetime(df[time_col]),
                "y": df[var],
            })
            all_records.append(temp)

            # Record series_name and variate_name for this unique_id
            uid_info_records.append({
                "unique_id": unique_id,
                "series_name": series_name,
                "variate_name": var,
            })

    panel_df = pd.concat(all_records, ignore_index=True)
    uid_info_df = pd.DataFrame(uid_info_records)
    return panel_df, uid_info_df


def compute_dataset_features(
    dataset_name: str,
    freq: str,
    csv_dir: str,
    output_dir: str = "./output",
    test_length: int | None = None,
    split_mode: str = "test",
) -> None:
    """
    Compute and save the full set of time series features for a given dataset.

    Pipeline:
        1. Load or generate the raw panel from the dataset CSV files.
        2. Filter data based on split_mode and test_length.
        3. Preprocess the time series (interpolation, scaling, frequency analysis).
        4. Compute statistical and STL-based time series features.
        5. Merge all features with dataset statistics.
        6. Save the resulting feature DataFrame into the output directory.

    Args:
        dataset_name: The name of the dataset (e.g., "IMOS", "ETTh1").
        freq: Frequency string (e.g., "H", "D", "15T").
        csv_dir: Path to directory containing processed CSV files (*.csv).
        output_dir: Base directory for output files.
        test_length: Number of timesteps for test portion (required if split_mode="test").
        split_mode: Which portion to compute features on:
            - "full": 完整序列
            - "test": 只在最后 test_length 个时间步上计算（默认）

    Returns:
        None. Saves features to {output_dir}/features/{dataset}/{freq}/{split_mode}.csv
    """
    start = time.time()

    # dataset_id for joining with results (format: "{dataset_name}/{freq}")
    dataset_id = f"{dataset_name}/{freq}"

    # Set the directories & paths (format: {dataset}/{freq}/)
    feature_dir = os.path.join(output_dir, "features", dataset_name, freq)
    os.makedirs(feature_dir, exist_ok=True)

    output_csv_path = os.path.join(feature_dir, f'{split_mode}.csv')

    # Skip if already computed
    if os.path.exists(output_csv_path):
        print(f"[Skip] Features for {dataset_name}/{freq} ({split_mode}) already exist at {output_csv_path}")
        return

    print(f"[Start] Processing {dataset_name}/{freq} ({split_mode}) from {csv_dir}")
    if split_mode == "test":
        print(f"        test_length={test_length}")

    # Generate panel from CSV directory with appropriate filtering
    print("Loading CSV files and converting to panel format...")
    panel, uid_info_df = convert_multi_csv_to_panel(csv_dir, test_length=test_length, mode=split_mode)
    print(f"Loaded panel: {len(panel)} rows, {panel['unique_id'].nunique()} unique_ids")

    # Interpolate, Scale, Freq_analysis
    print("Running preprocessing...")
    series, stats_df = preprocess_for_tsfeatures(panel, freq=freq)
    assert series['y'].isna().sum() == 0, "There are still NaNs in preprocessed series!"

    # Compute features
    uid_freq_map = stats_df.set_index('unique_id')['period1'].to_dict()
    unique_ids = stats_df['unique_id'].tolist()

    # Step 1: Extract meta features from unique_id (from preprocess tags: [rw], [sp])
    meta_start = time.perf_counter()
    meta_features_df = extract_meta_features_batch(unique_ids)
    # Drop is_stationary (redundant with is_random_walk)
    if 'is_stationary' in meta_features_df.columns:
        meta_features_df = meta_features_df.drop(columns=['is_stationary'])
    meta_time = time.perf_counter() - meta_start
    print(f"[Timing] meta_features_df took {meta_time:.4f} seconds")

    # Step 2: Compute STL-based features (trend, seasonal, residual)
    stl_start = time.perf_counter()
    stl_features_df = tsfeatures_with_uid_freq_map(
        series,
        uid_freq_map=uid_freq_map,
        features=[extended_stl_features],
        scale=False
    )
    stl_time = time.perf_counter() - stl_start
    print(f"[Timing] stl_features_df took {stl_time:.2f} seconds")

    # Merge all features
    features_df = meta_features_df.merge(stl_features_df, on='unique_id', how='left')
    features_df = features_df.merge(stats_df, on='unique_id', how='left')

    # Add identifier columns (dataset_id, series_name, variate_name)
    features_df = features_df.merge(uid_info_df, on='unique_id', how='left')
    features_df['dataset_id'] = dataset_id

    # Reorder columns according to FEATURE_COLUMNS_ORDER
    ordered_cols = [col for col in FEATURE_COLUMNS_ORDER if col in features_df.columns]
    # Add any remaining columns not in the order list
    remaining_cols = [col for col in features_df.columns if col not in ordered_cols]
    features_df = features_df[ordered_cols + remaining_cols]

    # Check for NaN values and remove rows with NaN (protection against STL decomposition failures)
    # Exclude period2/3 and p_strength2/3 which are legitimately NaN for some frequencies
    exclude_cols = ['period2', 'period3', 'p_strength2', 'p_strength3']
    check_cols = [c for c in features_df.columns if c not in exclude_cols and c != 'unique_id']

    nan_rows = features_df[check_cols].isna().any(axis=1)
    if nan_rows.sum() > 0:
        nan_uids = features_df.loc[nan_rows, 'unique_id'].tolist()
        # Find which features have NaN for each row
        nan_features = features_df.loc[nan_rows, check_cols].apply(
            lambda row: [c for c in check_cols if pd.isna(row[c])], axis=1
        )
        # Get unique NaN features across all rows
        all_nan_features = set()
        for feats in nan_features:
            all_nan_features.update(feats)
        print(f"[Warning] Removing {nan_rows.sum()} rows with NaN values: {nan_uids[:10]}{'...' if len(nan_uids) > 5 else ''}")
        print(f"          NaN features: {sorted(all_nan_features)}")
        features_df = features_df[~nan_rows]

    # Save all features
    features_df.to_csv(output_csv_path, index=False)
    print(f"[Done] {dataset_name}/{freq} ({split_mode}): Saved {len(features_df)} features to {output_csv_path} (elapsed {time.time() - start:.2f}s)")


def count_variates_in_dir(csv_dir: str) -> tuple[int, int]:
    """
    Count total variates across all CSV files in directory.

    Returns:
        (n_series, n_total_variates): Number of series and total variate count
    """
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_files:
        return 0, 0

    n_series = len(csv_files)
    total_variates = 0

    for csv_path in csv_files:
        df = pd.read_csv(csv_path, nrows=5)
        # First column is timestamp, rest are variates
        total_variates += len(df.columns) - 1

    return n_series, total_variates


def load_datasets_config(config_path: str) -> dict:
    """加载 datasets.yaml 配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_test_length(dataset_config: dict) -> int | None:
    """
    获取数据集的 test_length 值

    Args:
        dataset_config: 数据集特定配置

    Returns:
        test_length 值，如果未配置则返回 None
    """
    if dataset_config and "test_length" in dataset_config:
        return dataset_config["test_length"]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run tsfeatures extraction on preprocessed datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single dataset (expects data at ./data/processed_csv/IMOS/15T/*.csv)
    # Uses test_length from config/datasets.yaml
    python -m timebench.feature.features_runner --dataset IMOS/15T

    # Process all datasets in config
    python -m timebench.feature.features_runner --all

    # Compute features on full series instead of last test_length timesteps
    python -m timebench.feature.features_runner --dataset IMOS/15T --split full
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Oil_Price/B",
        help="Dataset key in format '{name}/{freq}' (e.g., 'IMOS/15T')"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all datasets in the config file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["full", "test"],
        help="Which portion to compute features on: 'full' (完整序列), 'test' (只在最后 test_length 个时间步上, 默认)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to datasets.yaml config file (default: src/timebench/config/datasets.yaml)"
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default='./data/processed_csv',
        help="Base directory for processed CSV files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./output',
        help="Base directory for output files"
    )

    args = parser.parse_args()

    # Load config
    config = load_datasets_config(args.config)
    datasets_config = config.get("datasets", {})

    if not datasets_config:
        raise ValueError(f"No datasets found in config file: {args.config}")

    if args.all:
        # Process all datasets in config
        summary_records = []

        for dataset_key in tqdm(datasets_config.keys(), desc="Processing datasets", unit="dataset"):
            dataset_name, freq = parse_dataset_key(dataset_key)
            # Path: ./data/processed_csv/{dataset_name}/{freq}/
            dataset_csv_dir = os.path.join(args.csv_dir, dataset_name, freq)

            if not os.path.isdir(dataset_csv_dir):
                print(f"[Warning] Directory not found: {dataset_csv_dir}, skipping {dataset_key}")
                continue

            # Get test_length from config
            dataset_cfg = datasets_config.get(dataset_key, {})
            test_length = get_test_length(dataset_cfg)

            # Validate test_length when mode is "test"
            if args.split == "test" and test_length is None:
                print(f"[Warning] test_length not found in config for {dataset_key}, skipping")
                continue

            # If test_length < 500, use full series instead
            effective_split_mode = args.split
            if args.split == "test" and test_length is not None and test_length < 500:
                print(f"[Info] test_length={test_length} < 500 for {dataset_key}, using full series instead")
                effective_split_mode = "full"

            n_series, n_variates = count_variates_in_dir(dataset_csv_dir)
            summary_records.append({
                "dataset": dataset_name,
                "freq": freq,
                "test_length": test_length,
                "n_series": n_series,
                "n_variates": n_variates
            })

            compute_dataset_features(
                dataset_name=dataset_name,
                freq=freq,
                csv_dir=dataset_csv_dir,
                output_dir=args.output_dir,
                test_length=test_length,
                split_mode=effective_split_mode,
            )

        # Save summary
        if summary_records:
            summary_df = pd.DataFrame(summary_records)
            summary_csv_path = os.path.join(args.output_dir, "features", "summary.csv")
            os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
            summary_df = pd.concat(
                [summary_df, pd.DataFrame([{
                    "dataset": "TOTAL",
                    "freq": "-",
                    "test_length": "-",
                    "n_series": summary_df["n_series"].sum(),
                    "n_variates": summary_df["n_variates"].sum()
                }])],
                ignore_index=True
            )
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"Summary saved to {summary_csv_path}")

    elif args.dataset:
        # Process single dataset
        dataset_key, freq, dataset_cfg = find_dataset_config(datasets_config, args.dataset)
        dataset_name, _ = parse_dataset_key(dataset_key)

        # Path: ./data/processed_csv/{dataset_name}/{freq}/
        dataset_csv_dir = os.path.join(args.csv_dir, dataset_name, freq)

        if not os.path.isdir(dataset_csv_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_csv_dir}\n"
                f"Expected preprocessed CSV files at: {dataset_csv_dir}/*.csv\n"
                f"Run preprocess.py first to generate the data."
            )

        # Get test_length from config
        test_length = get_test_length(dataset_cfg)

        # Validate test_length when mode is "test"
        if args.split == "test" and test_length is None:
            raise ValueError(
                f"test_length not found in config for {dataset_key}. "
                f"Please add test_length to the config or use --split full."
            )

        # If test_length < 500, use full series instead
        effective_split_mode = args.split
        if args.split == "test" and test_length is not None and test_length < 500:
            print(f"[Info] test_length={test_length} < 500, using full series instead")
            effective_split_mode = "full"

        compute_dataset_features(
            dataset_name=dataset_name,
            freq=freq,
            csv_dir=dataset_csv_dir,
            output_dir=args.output_dir,
            test_length=test_length,
            split_mode=effective_split_mode,
        )
    else:
        parser.print_help()
        raise ValueError("You must provide either --dataset or --all")


if __name__ == "__main__":
    main()
