# Time Series Features Extraction

This module extracts time series features from preprocessed CSV data for pattern-based evaluation.

## Input/Output

**Input**: `./data/processed_csv/{dataset}/{freq}/*.csv`
- First column: `timestamp` (datetime)
- Other columns: variate values

**Output**: `./output/features/{dataset}/{freq}/{split_mode}.csv`
- CSV file where each row represents one time series (one variate of one series)
- `split_mode`: `test` (test split only) or `full` (entire variate)
- All features are computed on the specified split

**Split Selection Logic**:
- By default, `split_mode="test"`
- When `split_mode="test"` and `test_length < 500`, the module automatically uses `"full"` mode instead.


## Usage
Before use, configure the new dataset in `src/timebench/config/datasets.yaml`.

```bash
# Process single dataset (default: test split)
python -m timebench.feature.features_runner --dataset Water_Quality_Darwin/15T

# Use full series
python -m timebench.feature.features_runner --dataset Water_Quality_Darwin/15T --split full

# Process all datasets in config
python -m timebench.feature.features_runner --all
```

## Feature Types

The module extracts three types of features:

1. **Meta features**: Extracted from raw series (stationarity & entropy).
2. **STL features**: Trend, seasonal, and residual features computed via STL decomposition. The implementation is adapted from [tsfeatures library](https://github.com/Nixtla/tsfeatures)
3. **Statistical features**: Basic statistics (mean, std, missing_rate, length) and frequency-domain features (periods, period strengths)

**Note**: Data is standardized and interpolated (if needed) internally during feature computation. Original CSV files are not modified.
