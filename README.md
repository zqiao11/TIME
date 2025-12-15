# TIME: Time Series Benchmarking Framework

A comprehensive framework for time series forecasting benchmarking, providing an end-to-end pipeline from data collection to model evaluation.


## Installation

We strongly recommend using Conda to manage the environment

```bash
conda create -n timebench python=3.11 -y
conda activate timebench
pip install -e .
```

Add path of datasets as GIFT_EVAL in .env

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

[Optional] Put raw time series csv into `data/raw_csv/`. But you can also pass the specific path in the following command.

[Trivial] Some examples to curate datasets can be found in `src/timebench/curation/`

---

### 2. Preprocessing (数据预处理)

**Purpose**: Clean and validate time series data, ensuring quality for downstream tasks.

**Input**: Raw CSV files
**Output**: Cleaned CSV files in `data/processed_csv/{dataset}/{freq}/` and summary reports in `data/processed_summary/{freq}/`

**Processing Steps**:

1. Data type and length validation
2. Missing value detection and imputation
3. Constant series and white noise detection
4. Random walk detection (marked with `[rw]`)
5. Outlier detection and cleaning (marked with `[sp]` for spike presence)
6. High correlation detection between variables

**Usage**:

```bash
# Single file mode
python -m timebench.preprocess --input_path data.csv --dataset MyDataset

# Multi-file mode
python -m timebench.preprocess --input_path csv_folder/ --dataset IMOS --freq 15T
```

**Details**: See [docs/PREPROCESS.md](docs/PREPROCESS.md)

---

### 3. Dataset Building (数据集构建)

**Purpose**: Convert preprocessed CSV files to Arrow format (HuggingFace Datasets) for evaluation. Based on gift-eval. Supports both univariate (UTS) and multivariate (MTS) time series.

**Input**: Cleaned CSV files from preprocessing
**Output**: Arrow dataset saved to `$GIFT_EVAL/{dataset_name}/{freq}/`



**Details**: See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)

---

### 4. Feature Extraction (特征提取)

**Purpose**: Extract statistical and temporal features from preprocessed data for analysis and future benchmarking.

**Input**: Preprocessed CSV files from `data/processed_csv/`
**Output**: Feature CSV files in `output/features/{dataset}/{freq}/{split_mode}.csv`

**Features Extracted**:

- Meta features: random walk flag, spike presence flag
- Trend features: strength, stability, Hurst exponent
- Seasonal features: strength, correlation, lumpiness
- Residual features: ACF, entropy

**Saved but not used**:
- Statistical features: mean, std, missing rate, length
- Periodicity features: FFT-detected periods and strengths


**Details**: See [docs/FEATURES.md](docs/FEATURES.md)

---

### 5. Model Evaluation (模型评估)

**Purpose**: Evaluate forecasting models on test/validation data with comprehensive per-window metrics.

**Input**: Arrow dataset from Dataset Building
**Output**: Predictions and metrics in `output/results/{model_name}/{dataset}_{term}/`

**Evaluation Features**:

- Per-window evaluation (multiple non-overlapping windows per series)
- Comprehensive metrics: MSE, MAE, RMSE, MAPE, sMAPE, MASE, ND, CRPS, QuantileLoss
- Support for both point and probabilistic forecasts
- Test/validation split support


**Configuration**: See [docs/PRED_CONFIG.md](docs/PRED_CONFIG.md) for setting `prediction_length` and `test_split` in `config/datasets.yaml`

**Details**: See [docs/EVALUATION.md](docs/EVALUATION.md)



## Quick Start

### Complete Workflow Example

1. **Preprocess data**:

   ```bash
   python -m timebench.preprocess --input_path raw_data/ --dataset IMOS --freq 15T
   ```

2. **Build dataset**:

   ```python
   from timebench.evaluation.dataset_builder import build_dataset_from_csvs

   build_dataset_from_csvs(
       csv_dir="./data/processed_csv/IMOS/15T",
       output_path="./datasets/GIFT-Eval/IMOS/15T",
       freq="15T",
   )
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

## Dependencies

- Python >= 3.11
- GIFT-Eval (for data loading and evaluation framework)
- PyTorch >= 2.1 (for deep learning models)
- NumPy, Pandas, SciPy (for data processing)
- HuggingFace Datasets (for Arrow format support)

## License

MIT License

## Citation

If you use TIME in your research, please cite:

```bibtex
@software{time2024,
  title = {TIME: Time Series Benchmarking Framework},
  year = {2024},
  url = {https://github.com/yourusername/TIME}
}
```
