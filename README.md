# TIME: Time Series Benchmarking Framework

A comprehensive framework for time series forecasting benchmarking, providing a full workflow spanning from data collection to model evaluation.


## Installation

We recommend using Conda to manage the environment

```bash
conda create -n timebench python=3.11 -y
conda activate timebench
pip install -e .
```

Define the output path for processed HF datasets in .env. [Used as `storage_env_var` in [`Dataset`](src/timebench/evaluation/data.py#L120). This implementation follows Gift-Eval and may be subject to change.]

```bash
echo "GIFT_EVAL=PATH_TO_SAVE" >> .env
```

## Overview

TIME provides a complete workflow for time series benchmarking:

```text
Preprocessing → Set config
                    ├─→ Feature Extraction
                    └─→ hf_dataset Building → Model Evaluation
```

After preprocessing and configuration, the pipeline splits into two parallel branches: feature extraction for pattern analysis, and dataset building followed by model evaluation.

## Pipeline Architecture

### 1. Data Curation (数据收集)
Due to the varying methods required for data scraping, this stage is not included in the main framework. However, we provide some sample crawling scripts for reference.
---

### 2. Preprocessing (数据预处理)

**Purpose**: Clean and validate time series data to generate preprocessed datasets.

**Input**: Raw CSV files of a time series dataset

**Output**: Cleaned CSV files in `data/processed_csv/{dataset}/{freq}/` and summary reports in `data/processed_summary/{freq}/`

**Processing Steps**:

1. Data type and length validation
2. Missing value detection and imputation
3. Constant series and white noise detection
4. Random walk detection (marked with `[rw]`)
5. Outlier detection and cleaning (marked with `[sp]` for spike presence)
6. High correlation detection between variables


**Details**: See [docs/PREPROCESS.md](docs/PREPROCESS.md)

---

### 3. Dataset Building (数据集构建)

**Purpose**: Convert preprocessed CSV files to Arrow format (HuggingFace Datasets) for evaluation. Based on gift-eval. Supports both univariate (UTS) and multivariate (MTS) time series.

**Input**: Cleaned CSV files from preprocessing

**Output**: Arrow dataset saved to `$GIFT_EVAL/{dataset_name}/{freq}/`



**Details**: See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)

---

### 4. Feature Extraction (特征提取)

**Purpose**: Extract statistical and tsfeatures from preprocessed data for pattern-based benchmarking.

**Input**: Preprocessed CSV files from `data/processed_csv/`

**Output**: Feature CSV files in `output/features/{dataset}/{freq}/{split_mode}.csv`

**Tsfeatures for variate pattern definition**:

- Meta features: random walk flag, spike presence flag
- Trend features: strength, stability, Hurst exponent
- Seasonal features: strength, correlation, lumpiness
- Residual features: ACF, entropy

**Statistical features for archiving**:
- Statistical features: mean, std, missing rate, length
- Periodicity features: FFT-detected periods and strengths


**Details**: See [docs/FEATURES.md](docs/FEATURES.md)

---

### 5. Model Evaluation (模型评估)

**Purpose**: Evaluate forecasting models and save window-level results for future visualization.

**Input**: Arrow dataset from Dataset Building

**Output**: Window-level predictions and metrics in `output/results/{model_name}/{dataset}_{term}/`

**Evaluation Features**:

- Rolling-window evaluation
- Use multiple quantiles both point and probabilistic forecasts (same as gift-eval)
- Test/val split support. Val split is only for hyperprameter tuning, and no results are saved..


**Configuration**: See [docs/PRED_CONFIG.md](docs/PRED_CONFIG.md) for setting configs in `config/datasets.yaml`

**Details**: See [docs/EVALUATION.md](docs/EVALUATION.md)



## Quick Start

### Complete Workflow Example

1. **Preprocess data**:

   ```bash
   python -m timebench.preprocess --input_path PATH_RAW_IMOS --dataset IMOS --freq 15T
   ```

2. **Build dataset**:

   ```bash
   python -m timebench.evaluation.dataset_builder \
     --csv-dir ./data/processed_csv/IMOS/15T \
     --output-path ./datasets/GIFT-Eval/IMOS/15T \
     --freq 15T
   ```

3. **Extract features**:

   ```bash
   python -m timebench.feature.features_runner --dataset IMOS/15T
   ```

4. **Run evaluation**:

   ```bash
   python experiments/moirai.py --dataset "IMOS/15T" --terms short medium long
   ```


## Configuration

Dataset and prediction configurations are stored in `config/datasets.yaml`. Each dataset can specify:

- `test_split`: Test set ratio (shared across all terms)
- `val_split`: Validation set ratio (optional)
- `prediction_length`: Prediction length for each term (short, medium, long)

Windows are automatically calculated based on `test_split` and `prediction_length` to ensure fair comparison across different terms.

**Example**:

```yaml
datasets:
  IMOS/15T:
    test_split: 0.1
    val_split: 0.1
    short:
      prediction_length: 16     # 16 * 15min = 4 hours
    medium:
      prediction_length: 96    # 96 * 15min = 1 day
    long:
      prediction_length: 672   # 672 * 15min = 7 days
```

See [docs/PRED_CONFIG.md](docs/PRED_CONFIG.md) for detailed configuration guide.


## Documentation

- [Data Format Specification](docs/DATA_FORMAT.md): CSV and Arrow format requirements
- [Preprocessing Guide](docs/PREPROCESS.md): Data cleaning and validation pipeline
- [Feature Extraction Guide](docs/FEATURES.md): Statistical and temporal feature computation
- [Prediction Configuration](docs/PRED_CONFIG.md): Setting up prediction tasks
- [Evaluation Guide](docs/EVALUATION.md): Model evaluation workflow

