# TIME: Time Series Benchmarking Framework

A comprehensive framework for time series forecasting benchmarking, providing a full workflow spanning from data collection to model evaluation.


## Installation

We recommend using Conda to manage the environment

```bash
conda create -n timebench python=3.11 -y
conda activate timebench
pip install -e .
```

Define the output path for processed HF datasets in .env. If you have no specific preference, you can set it to `ROOT_PATH/TIME/data/hf_dataset`. [Used as `storage_env_var` in [`Dataset`](src/timebench/evaluation/data.py#L120). This implementation follows Gift-Eval and may be subject to change.]

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

**Important**: This process is not fully automated! Users need to make final decisions about how to process and formulate the time series based on the summary.


**Details**: See [docs/PREPROCESS.md](docs/PREPROCESS.md)

---

### 3. Feature Extraction (特征提取)

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

### 4. Dataset Building (数据集构建)

**Purpose**: Following gift-eval, convert preprocessed CSV files to Arrow format (HuggingFace Datasets) for evaluation. This step determines whether to save the time series as Univariate (UTS) or Multivariate (MTS).

**Input**: Preprocessed CSV files from `data/processed_csv/`

**Output**: Arrow dataset saved to `$GIFT_EVAL/{dataset_name}/{freq}/`



**Details**: See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)

---

### 5. Model Evaluation (模型评估)

**Purpose**: Evaluate forecasting models and save window-level results for future visualization.

**Input**: Arrow dataset from Dataset Building

**Output**: Window-level predictions and metrics in `output/results/{model_name}/{dataset}_{term}/`

**Evaluation Features**:

- Rolling-window evaluation
- Use multiple quantiles both point and probabilistic forecasts (same as gift-eval)
- Test/val split support. Val split is only for hyperprameter tuning, and no results are saved.


**Configuration**: See [docs/PRED_CONFIG.md](docs/PRED_CONFIG.md) for setting configs in `config/datasets.yaml`

**Details**: See [docs/EVALUATION.md](docs/EVALUATION.md)



## Complete Workflow Example

0. **Download data**

    We use IMOS data as example. Download the data folder from [Google Drive](https://drive.google.com/drive/u/0/folders/1OAKZ0MilP0vtfuzcqq-WONw-PS0s3n3u). The curated data have 8 MTS from different stations and period, each with 8 variates.

1. **Preprocess data**:

   ```bash
   python -m timebench.preprocess --input_path PATH_IMOS --dataset IMOS --freq 15T
   ```

   * `dataset`: Name of the dataset to be saved.
   * `freq`: **Optional**. If not specified, the frequency will be inferred by `pd.infer_freq` automatically. **Note**: Inference may not be 100% accurate; it is recommended to manually provide the correct value to ensure precision.

    After the preprocssing process, the preliminary csv files are be saved and a summary is generated. Since the pipeline detected issues with certain variates, it prompts the user to make a decision.
   ```bash
   ============================================================
    [PreprocessPipeline] 批量处理完成!
      数据集: IMOS
      频率: 15T
      成功: 8/8
      总行数: 117099
      总列数: 47
      输出 CSV 目录: ./data/processed_csv/IMOS/15T
      输出 JSON 目录: ./data/processed_summary/IMOS/15T
    ============================================================

    ============================================================
    [PreprocessPipeline] Variate 汇总统计 (数据集: IMOS)
    ============================================================
    Variate                       保留率          RW率          SP率
    ------------------------------------------------------------
    CNDC                   8/8   (100.0%)   6/8   ( 75.0%)   0/8   (  0.0%)
    CPHL                   8/8   (100.0%)   0/8   (  0.0%)   0/8   (  0.0%)
    DOX2                   8/8   (100.0%)   1/8   ( 12.5%)   0/8   (  0.0%)
    PSAL                   8/8   (100.0%)   5/8   ( 62.5%)   0/8   (  0.0%)
    TEMP                   8/8   (100.0%)   7/8   ( 87.5%)   0/8   (  0.0%)
    TURB                   7/8   ( 87.5%)   1/8   ( 12.5%)   0/8   (  0.0%)
    ============================================================
    [PreprocessPipeline] Variate 汇总已保存至: ./data/processed_summary/IMOS/15T/_variate_summary.json

    ====================================================================================================
    [PreprocessPipeline] 高相关变量对统计 (数据集: IMOS)
    ====================================================================================================
    变量对                                     高相关次数       r均值(高)       r均值(低)       r均值(全)
    ----------------------------------------------------------------------------------------------------
    CNDC <-> TEMP                      5/8     ( 62.5%)     0.9888       0.7181       0.8873
    ====================================================================================================
    说明: r均值(高)=高相关series上的均值, r均值(低)=非高相关series上的均值, r均值(全)=所有series上的均值

    ============================================================
    ⚠️  [决策提示] 需要人工决策!
    ============================================================

    📌 以下 variate 仅在少数 series 上被丢弃，建议移除那些 series:
      - TURB: 在 1/8 个 series 上被丢弃
        被丢弃的 series: item_7.csv

      批量移除命令: python -m timebench.preprocess --remove_series item_7.csv --target_dir ./data/processed_csv/IMOS/15T

    📌 以下变量对在多数 series 上高度相关，考虑移除其中一个:
      - CNDC <-> TEMP: 在 5/8 个 series 上高相关
        r均值: 高相关=0.9888, 非高相关=0.7181, 全部=0.8873
        移除 CNDC: python -m timebench.preprocess --remove_variate CNDC --target_dir ./data/processed_csv/IMOS/15T
        移除 TEMP: python -m timebench.preprocess --remove_variate TEMP --target_dir ./data/processed_csv/IMOS/15T

    💡 提示:
      - 如果某个 variate 在大多数 series 上都被丢弃 → 移除该 variate
      - 如果某个 variate 仅在少数 series 上被丢弃 → 移除那些 series
      - 如果两个变量高度相关 → 根据业务意义选择保留一个
      - 支持逗号分隔的批量操作，如: --remove_variate VAR1,VAR2,VAR3
      - 添加 --dry_run 可预览操作而不实际执行
    ============================================================
   ```
   The summary shows two isses:
   1. Variate `TURB` has excessive missing values on series `item_7`. Since this issue of `TURB` is only observed in one series, we can simply discard series `item_7` and keep `TURB` variate in the dataset.
      ```bash
      python -m timebench.preprocess --remove_series item_7.csv --target_dir ./data/processed_csv/IMOS/15T
      ```

      Note: if `TURB` encounter issues on multiple series, we need to consider discarding variate `TURB` entirely from the dataset.
      ```bash
      # Don't run this command in this example!
      python -m timebench.preprocess --remove_variate TURB --target_dir ./data/processed_csv/IMOS/15T
      ```

      These commands will apply the corresponding changes directly to the processed CSV files.

    2. Two variates are highly correlated (correlation > 95%) on 5 series, raising the question of potential redundancy. We need to consider if we need to discard one of the variates from the dataset. Since the correlaltion is not very high on the remaining 3 series and the schematic meaning of these variates are very different, we decide to keep both variates.

    Therefore, the final processed IMOS dataset has 7 MTS with 8 variates.

2. **Set config**

    The configuration for IMOS data has been established. For new data setup, see the **Configuration** section below.

3. **Build hf_dataset**:

   ```bash
   python -m timebench.evaluation.dataset_builder \
     --csv-dir ./data/processed_csv/IMOS/15T \
     --output-path ./datasets/GIFT-Eval/IMOS/15T \
     --freq 15T
   ```

4. **Extract features**:

   ```bash
   python -m timebench.feature.features_runner --dataset IMOS/15T
   ```

5. **Run evaluation**:

   Here we use moirai to evalute. For other TSFMs, we need to create a new python file for each, following [gift-eval examples](https://github.com/SalesforceAIResearch/gift-eval/tree/main/notebooks).

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

