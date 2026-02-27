# It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks


[![arXiv](https://img.shields.io/badge/arxiv-2602.12147-b31b1b.svg)](https://arxiv.org/abs/2602.12147)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/Real-TSF/TIME/tree/main)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LeaderBoard-FFD21E)](https://huggingface.co/spaces/Real-TSF/TIME-leaderboard)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CSVFiles-FFD21E)](https://huggingface.co/datasets/Real-TSF/TIME-ProcessedCSV)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Results&Features-FFD21E)](https://huggingface.co/datasets/Real-TSF/TIME-Output)
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

TIME is a task-centric time series forecasting benchmark comprising various fresh datasets, tailored for zero-shot TSFM evaluation. This codebase provides a full workflow spanning from data preprocessing to model evaluation.

## 📅 Update Log

### 2026 Feb 27:
* Official release of our TIME codebase.
* Clean features and ProcessedCSV on HuggingFace.

### 2026 Feb 22:
* Leaderboard Results Updates:
  * **Chronos2 & Chronos-bolt**: Integrate updates from [PR#2](https://github.com/zqiao11/TIME/pull/2).
  * **TiRex**: Integrate updates from [PR#3](https://github.com/zqiao11/TIME/pull/3)

### 2026 Feb 12:
* First release of our [arxiv paper](https://arxiv.org/abs/2602.12147) and [leaderboard](https://huggingface.co/spaces/Real-TSF/TIME-leaderboard).


## ⚙️ Installation

1. We recommend using Conda to manage the environment

```bash
conda create -n timebench python=3.11 -y
conda activate timebench
pip install -e .
```

2. Download the dataset from [huggingface](https://huggingface.co/datasets/Real-TSF/TIME)

3. Define the path for HF datasets in `.env`. (Used as `storage_env_var` in [`Dataset`](src/timebench/evaluation/data.py#L120)).

```bash
echo "TIME_DATASET=PATH_TO_DATASET" >> .env
```

## 🚀 Getting Started

### Model Forecasting
We provide the complete codebase and scripts required to reproduce all results from our benchmark.

For each model, use the corresponding script in the `scripts/` directory to automatically set up the Conda environment and run evaluations across all tasks.

⚠️ **Important Note**: Please ensure the script's Conda environment name doesn't conflict with your existing ones..

```
# Example: Running the evaluation for Chronos2
bash scripts/run_chronos2.sh

# We recommand using nohup to run the scripts in the background
nohup bash scripts/run_chronos2.sh > run_chronos2.txt 2>&1 &
```

For each task, window-level predictions (quantiles) and metrics will be saved in `output/results/{model_name}/{dataset}/{freq}/{term}/`.

### Compute Overall Metrics

Once the evaluations are complete, use the following script to aggregate the raw outputs into the overall metrics in leaderboard. This process automatically fetches the Seasonal Naive results from Hugging Face and computes the aggregated metrics across all tasks.

```bash
# Compute Overall Leaderboard based on `output/results` (sorted by MASE)
python scripts/compute_local_leaderboard.py

```

For deeper analysis, including dataset-level breakdowns, pattern-level evaluation and visualizations, you can download and locally run our [Leaderboard App](https://huggingface.co/spaces/Real-TSF/TIME-Leaderboard).


## 💻 Run Your Own Model

To add a new model, follow these steps:

1. **Implement your model in `experiments/`**

   Create a new Python script in the `experiments/` directory (e.g., `experiments/your_model.py`). You can use existing implementations like `experiments/chronos2.py` as a reference template.

-  **Use the Dataset class**

   The `Dataset` class is adapted from [Gift-Eval](https://github.com/SalesforceAIResearch/gift-eval/blob/main/src/gift_eval/data.py) and provides a unified interface for loading time series data:
   ```python
   from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config

   # ⚠️ Important: Set to_univariate based on your model's capabilities
   # If your model only supports univariate forecasting:
   to_univariate = False if Dataset(name=dataset_name, term=term, to_univariate=False).target_dim == 1 else True

   # If your model supports multivariate forecasting natively:
   to_univariate = False

   dataset = Dataset(
       name=dataset_name,
       term=term,  # "short", "medium", or "long"
       to_univariate=to_univariate,
       prediction_length=prediction_length,
       test_length=test_length,
       val_length=val_length,
   )
   ```

-  **Generate predictions and save results**

   TIME uses a flexible evaluation interface that doesn't rely on GluonTS. Simply compute quantile predictions (`fc_quantiles`) externally and pass them to `save_window_predictions`:

   ```python
   from timebench.evaluation.saver import save_window_predictions

   # Generate fc_quantiles with shape:
   # - (num_total_instances, num_quantiles, prediction_length) for univariate
   # - (num_total_instances, num_quantiles, num_variates, prediction_length) for multivariate
   # where num_total_instances = num_series_exp * num_windows

   save_window_predictions(
       dataset=dataset,
       fc_quantiles=fc_quantiles,
       ds_config=f"{dataset_name}/{freq}/{term}",
       output_base_dir="output/results",
       seasonality=season_length,
       model_hyperparams={"model_name": "your_model"},
   )
   ```

   This function automatically computes per-window metrics and saves predictions, metrics, and configuration files to `output/results/{model_name}/{dataset}/{freq}/{term}/`.

2. **Create a run script in `scripts/`**

   Create a shell script (e.g., `scripts/run_your_model.sh`) to run your model across all tasks. The script should:
   - Set up the Conda environment with required dependencies
   - Call your experiment script for each task
   - Include specific hyperparams configuration and ensure reproducibility

### Submit Results to TIME Leaderboard

   Once your evaluation is complete and you are ready to feature on the TIME leaderboard:
   - Open a Pull Request to upload your `output/results/{model_name}/` folder to the [TIME-Output repository](https://huggingface.co/datasets/Real-TSF/TIME-Output/tree/main/results) on Hugging Face.
      ```python
      from huggingface_hub import HfApi

      api = HfApi()

      model_name = "YOUR_MODEL_NAME"

      api.upload_folder(
         folder_path=f"output/results/{model_name}",  # Path to your local results folder
         path_in_repo=f"results/{model_name}",
         repo_id="Real-TSF/TIME-Output",
         repo_type="dataset",
         commit_message=f"Submit evaluation results for {model_name}",
         create_pr=True
      )
      ```
   - The results will be automatically included in the leaderboard after review
   - To ensure reproducibility, we highly recommend contributing your experiment code and execution scripts to this GitHub repository.

## 📊 Datasets and TSfeatures

Our codebase provides utilities for data preprocessing and computing time series features. For detailed instructions, please refer to the documentation in the `docs/` directory:
- [Data Preprocessing Guide](docs/PREPROCESS.md): Screen,preprocess and clean raw CSV datasets
- [Data Format Specification](docs/DATA_FORMAT.md): Convert processed CSV files into the efficient Arrow format
- [Time Series Features](docs/FEATURES.md): Compute TSfeatures from processed csv files

### Adding New Datasets

If you want to add a new dataset to TIME:

1. **Preprocess your data** following the documentation in `docs/`:
   - Generate processed CSV files
   - Create Arrow Datasets (hf_dataset)
   - Compute time series features

2. **Upload processed data to HuggingFace by PR**:
   - Upload processed CSV files to [TIME-ProcessedCSV](https://huggingface.co/datasets/Real-TSF/TIME-ProcessedCSV)
   - Upload hf_dataset to [TIME](https://huggingface.co/datasets/Real-TSF/TIME)
   - Upload features to [TIME-Output](https://huggingface.co/datasets/Real-TSF/TIME-Output/tree/main/features)

3. **Update the configuration**:
   - Update `src/timebench/config/datasets.yaml` on GitHub to include your forecasting tasks
   - Open a Pull Request with your changes

4. **Review and integration**:

   After review and approval, we will:
     - Add your dataset to TIME
     - Evaluate existing models on your new datasets
     - Update the leaderboard with new results

## 🤝 Acknowledgements

The core components of this repository include code adapted from the following excellent projects:
* [Gift-Eval](https://github.com/SalesforceAIResearch/gift-eval)
* [tsfeatures library](https://github.com/Nixtla/tsfeatures)

We also extend our sincere gratitude to the authors of the evaluated TSFMs for open-sourcing their work and driving progress in the time series community.

## Citation

If you find this benchmark useful, please consider citing:
```
@article{qiao2026s,
  title={It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks},
  author={Qiao, Zhongzheng and Pan, Sheng and Wang, Anni and Zhukova, Viktoriya and Liu, Yong and Jiang, Xudong and Wen, Qingsong and Long, Mingsheng and Jin, Ming and Liu, Chenghao},
  journal={arXiv preprint arXiv:2602.12147},
  year={2026}
}
```
