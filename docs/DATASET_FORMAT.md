# Dataset Format Specification

This document provides detailed specifications for time series dataset formats and explains how to convert processed CSV data into HF Dataset Arrow format using `dataset_builder.py`.



## Input CSV Directory Structure

By default, the processed CSV files follow the directory structure below. The feature extraction tool (`features_runner.py`) takes the path to `{dataset_name}/{freq}` as input:
```
data/
└── processed_csv/
    └── {dataset_name}/
        └── {freq}/
            └── {file(s)}.csv
```

The `{freq}` directory can contain a single CSV file or multiple CSV files. Each csv can represent:

* A single Univariate Time Series (UTS)

* Multiple Univariate Time Series

* A single Multivariate Time Series (MTS)



## Conversion Workflow

The `dataset_builder` follows this workflow:

```
┌──────────────────────────────────────────────────────────────────┐
│                     build_dataset_from_csvs                      │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. Scan Directory                                                │
│    csv_paths = sorted(csv_dir.glob(pattern))                     │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. Load Data                                                     │
│    dfs = [pd.read_csv(p, parse_dates=[0]) for p in csv_paths]    │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. Setup Generator (dataframes_to_generator)                     │
│    - Set first column as datetime index                          │
│    - Resolve frequency (explicit `freq` or `pd.infer_freq`)      │
└──────────────────────────────────────────────────────────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
         ┌───────────────────┐       ┌───────────────────┐
         │     MTS Mode      │       │     UTS Mode      │
         │ to_univariate=False       │ to_univariate=True│
         ├───────────────────┤       ├───────────────────┤
         │ 1 CSV = 1 Series  │       │ 1 CSV = D Series  │
         │ target: [D, T]    │       │ target: [T]       │
         └───────────────────┘       └───────────────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. Build & Save HF Dataset                                       │
│    dataset = Dataset.from_generator(gen_func, features=features) │
│    dataset.save_to_disk(output_path)                             │
└──────────────────────────────────────────────────────────────────┘
```

The resulting dataset can be loaded directly using the `Dataset` class from `timebench.evaluation.data`



## Usage Examples
 It is strongly recommended to explicitly provide the `--freq` parameter, as pandas frequency inference (`pd.infer_freq`) may be unreliable.

Additionally, ensure your `--output-path` matches the `TIME_DATASET` variable in your `.env` file

* Example 1: Multiple CSVs, Each is a UTS


```bash
python -m timebench.evaluation.dataset_builder \
  --csv-dir data/processed_csv/CPHL/30T \
  --output-path data/hf_dataset/CPHL/30T \
  --freq 30T \
  --to-univariate
```


* Example 2: Multiple CSVs, Each is a MTS

```bash
# Example: 7 series, each has 6 variates
python -m timebench.evaluation.dataset_builder \
  --csv-dir data/processed_csv/Water_Quality_Darwin/15T \
  --output-path data/hf_dataset/Water_Quality_Darwin/15T \
  --freq 15T
```

* Example 3: Single CSV, MTS

```bash
# Example: 1 series with 5 variates
python -m timebench.evaluation.dataset_builder \
  --csv-dir data/processed_csv/SG_PM25/H \
  --output-path data/hf_dataset/SG_PM25/H \
  --freq H
```
