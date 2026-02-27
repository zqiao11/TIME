# Time Series Preprocessing Pipeline

This document outlines the functionality and usage of the `timebench.preprocess` module. It is designed for quality inspection, cleaning, and statistical summarization of time series data.



## 1. Overview

The PreprocessPipeline performs the following automated checks and operations:


| Feature | Description |
|---------|-------------|
| Data Type Check | Ensures data is numeric. |
| Length Check | Validates minimum sequence length based on frequency. |
| Missing Value Check | Validates missing rates. Raw NaNs are preserved in the output. |
| Constant Detection | Detects constant or zero-variance sequences. |
| White Noise Detection | Uses the Ljung-Box test to flag white noise. |
| Outlier Cleaning | Uses a rolling IQR method. Extreme outliers are replaced with the last valid observation |
| Multivariate Correlation | Detects highly correlated redundant variables (Pearson r > 0.95). |
| Dataset Summarization | Aggregates stats across multiple series to guide manual cleanup decisions. |

All diagnostic results (e.g., failed variates & failure reasons) are saved exclusively in the corresponding JSON metadata files. Failed columns are recorded under `recommended_drop_columns`.



## 2. Data Screening & Processing

The preprocessing pipeline can handle both individual files and batches of files. It automatically runs quality checks, aligns timestamps, interpolates minor outliers, and generates detailed metadata JSONs for every processed series.

### Single CSV
To process a single file, point the `--input_path` directly to your `.csv` file.
```bash
python -m timebench.preprocess \
    --input_path docs/SG_PM25_raw.csv \
    --dataset SG_PM25 \
    --freq H
```
The pipeline will log any automatic fixes it applies (such as filling missing timestamps). If all variates pass the quality checks, it simply outputs a brief success summary and saves the metadata.

```
[Preprocess] Single-file mode: docs/SG_PM25_raw.csv
[Preprocess] Filled 1330 missing timestamps.
[Preprocess] Filled 1330 missing timestamps.
[Preprocess] Filled 1330 missing timestamps.
[Preprocess] Filled 1330 missing timestamps.
[Preprocess] Filled 1330 missing timestamps.
[Preprocess] Saved details to: ./data/processed_summary/SG_PM25/H/SG_PM25_raw.json

============================================================
[Finished] Dataset: SG_PM25 | Freq: H
Success: 1/1 files processed
Total rows: 38688 | Total cols: 5
Summary saved to: ./data/processed_summary/SG_PM25/H/_summary.json
============================================================
```


### Multiple CSV
For datasets of multiple files, point the `--input_path` to the directory containing your `.csv` files.
```bash
python -m timebench.preprocess \
    --input_path docs/Water_Quality_Darwin_raw \
    --dataset Water_Quality_Darwin \
    --freq 15T
```
If any data fails the quality checks (e.g., high missing rates, pure noise), it will trigger an Action Required prompt. This prompt breaks down the problematic variables and provides direct, copy-pasteable commands for you to execute the cleanup.
```
[Preprocess] Batch mode: 8 files found in docs/Water_Quality_Darwin_raw
[Preprocess] Saved details to: ./data/processed_summary/Water_Quality_Darwin/15T/item_0.json
[Preprocess] Saved details to: ./data/processed_summary/Water_Quality_Darwin/15T/item_1.json
[Preprocess] Saved details to: ./data/processed_summary/Water_Quality_Darwin/15T/item_2.json
[Preprocess] Saved details to: ./data/processed_summary/Water_Quality_Darwin/15T/item_3.json
[Preprocess] Saved details to: ./data/processed_summary/Water_Quality_Darwin/15T/item_4.json
[Preprocess] Saved details to: ./data/processed_summary/Water_Quality_Darwin/15T/item_5.json
[Preprocess] Saved details to: ./data/processed_summary/Water_Quality_Darwin/15T/item_6.json
[Preprocess] Saved details to: ./data/processed_summary/Water_Quality_Darwin/15T/item_7.json

============================================================
[Finished] Dataset: Water_Quality_Darwin | Freq: 15T
Success: 8/8 files processed
Total rows: 117099 | Total cols: 48
Summary saved to: ./data/processed_summary/Water_Quality_Darwin/15T/_summary.json
============================================================

⚠️  [Action Required] Problematic Data Detected
------------------------------------------------------------

📌 Variates failed in a FEW series (< 50%):
   - 'TURB': Failed in 1/8 series.
     └─ item_7.csv: ❌ Missing rate failed: 52.38%

   -> Option A (Recommended): Remove the affected series to keep the variate clean
      python -m timebench.preprocess --remove_series item_7.csv --target_dir ./data/processed_csv/Water_Quality_Darwin/15T
   -> Option B: Remove the variates entirely from all series
      python -m timebench.preprocess --remove_variate TURB --target_dir ./data/processed_csv/Water_Quality_Darwin/15T

💡 Tip: Append '--dry_run' to preview deletions without modifying files.
============================================================
```


## 4. Output Files
### Individual Series JSON (`{series_name}.json`)
Contains metadata and check results for a specific CSV file. Key fields inside `_meta`:

* `kept_columns`: List of valid variates.

* `recommended_drop_columns`: Variates that failed the quality checks.

* `num_observations`: Total count of non-NaN values.

### Dataset Summary JSON (`_summary.json`)
Generated in batch mode to summarize the entire dataset:

* `num_series`, `success_count`, `num_observations`

* `avg_series_length`, `max_series_length`, `min_series_length`

* `variates`: Aggregated stats per variate.



## 5. Actionable Decisions & Cleanup

The preprocessing pipeline acts as an automated diagnostic tool. It does not automatically delete any data during the initial run. Instead, it flags problematic variables and categorizes them in the terminal prompt and JSON metadata.

Human Review is highly recommended. Not all failed variates must be deleted. Depending on your downstream forecasting tasks and domain knowledge, you must review the "Action Required" prompt and decide whether to keep or drop the data. Crucially, any deletions in MTS datasets need to preserve the dataset's structural integrity, ensuring every series retains the exact same set of variables. Thus, you must either drop a problematic variate globally (across all series), or drop the affected series entirely.

If you decide that cleanup is necessary, you have two targeted options:

### Option 1: Remove Specific Variates (Column-level)

Best for: A variable fails the quality checks across the majority of the series. Removing it globally keeps the rest of the dataset uniform.

```
python -m timebench.preprocess --remove_variate VARIATE_NAME --target_dir ./data/processed_csv/Dataset/Freq
```

### Option 2: Remove Specific Series (File-level)
Best for: A variable fails in only a few specific series (e.g., < 50%). Instead of deleting that variable from the entire dataset, you can delete the few corrupted CSV files to preserve the variable globally.

```
python -m timebench.preprocess --remove_series SERIES_NAME.csv --target_dir ./data/processed_csv/Dataset/Freq
```
After executing a cleanup command, the pipeline automatically synchronizes your dataset:

* **CSV Files Updated**: The specified variates or series are deleted from the CSV dataset.

* **JSON Metadata Recalculated**: Both the individual `{series_name}.json` and the global `_summary.json` are automatically recomputed to reflect the new data shape, remaining variables, and updated observation counts.


## 6. Next Steps

* Proceed to convert the cleaned CSVs into Arrow formats (see [DATA_FORMAT.md](./DATA_FORMAT.md)).

* Extract time series features using the `features_runner.py` module (see [FEATURES.md](./FEATURES.md)).
