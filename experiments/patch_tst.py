"""
PatchTST model experiments for time series forecasting.

This script supports three modes:
1. tune: Hyperparameter tuning using Optuna
2. train: Train with best hyperparameters (or provided ones)
3. test: Evaluate trained model on test data

Usage:
    # Hyperparameter tuning
    python experiments/patch_tst.py --dataset "SG_Weather/D" --terms short --mode tune

    # Training with best hyperparameters
    python experiments/patch_tst.py --dataset "SG_Weather/D" --terms short --mode train

    # Testing (generate predictions and metrics)
    python experiments/patch_tst.py --dataset "SG_Weather/D" --terms short medium long --mode test

    # Run all datasets
    python experiments/patch_tst.py --dataset all_datasets --mode test
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import optuna
import torch
import yaml
from dotenv import load_dotenv
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
from lightning.pytorch.callbacks import EarlyStopping

from timebench.evaluation import save_window_predictions
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.metrics import compute_per_window_metrics
from timebench.models.patch_tst import PatchTSTEstimator

# Load environment variables
load_dotenv()


def load_model_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load PatchTST model configuration from YAML file."""
    if config_path is None:
        # Default config path
        config_path = Path(__file__).parent.parent / "config" / "models" / "patch_tst.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found at: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_hyperparams_path(output_dir: str, dataset_name: str, term: str) -> Path:
    """Get path for saving/loading best hyperparameters."""
    # Create a safe filename from dataset name
    safe_name = dataset_name.replace("/", "_")
    return Path(output_dir) / "hparams" / f"{safe_name}_{term}_best_params.json"


def save_best_params(params: Dict[str, Any], path: Path):
    """Save best hyperparameters to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved best parameters to {path}")


def load_best_params(path: Path) -> Dict[str, Any]:
    """Load best hyperparameters from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Best parameters not found at: {path}")
    with open(path, "r") as f:
        return json.load(f)


def sample_hyperparams(
    trial: optuna.Trial,
    optuna_config: Dict[str, Any],
    prediction_length: int,
    term: str,
    context_length_multipliers: Dict[str, list],
    min_context_length: int = 32,  # Minimum context length to ensure valid patching
) -> Dict[str, Any]:
    """Sample hyperparameters from Optuna search space."""
    hparams = {}

    for param_name, param_config in optuna_config.items():
        param_type = param_config["type"]

        if param_type == "categorical":
            hparams[param_name] = trial.suggest_categorical(
                param_name, param_config["values"]
            )
        elif param_type == "float":
            # Explicitly convert to float to handle YAML scientific notation as strings
            hparams[param_name] = trial.suggest_float(
                param_name,
                float(param_config["low"]),
                float(param_config["high"]),
                log=param_config.get("log", False),
            )
        elif param_type == "int":
            hparams[param_name] = trial.suggest_int(
                param_name,
                int(param_config["low"]),
                int(param_config["high"]),
                log=param_config.get("log", False),
            )

    # Sample context length based on term
    # Ensure context_length >= min_context_length (to ensure valid patching with patch_len)
    multipliers = context_length_multipliers.get(term, [1])
    context_length_candidates = [
        max(m * prediction_length, min_context_length) for m in multipliers
    ]
    # Remove duplicates and sort
    context_length_candidates = sorted(set(context_length_candidates))
    hparams["context_length"] = trial.suggest_categorical(
        "context_length", context_length_candidates
    )

    return hparams


class PatchTSTObjective:
    """Optuna objective for PatchTST hyperparameter tuning."""

    def __init__(
        self,
        dataset: Dataset,
        model_config: Dict[str, Any],
        term: str,
        max_epochs: int = 50,
    ):
        self.dataset = dataset
        self.model_config = model_config
        self.term = term
        self.max_epochs = max_epochs

    def __call__(self, trial: optuna.Trial) -> float:
        # Sample hyperparameters
        hparams = sample_hyperparams(
            trial,
            self.model_config["optuna"],
            self.dataset.prediction_length,
            self.term,
            self.model_config.get("context_length_multipliers", {"short": [1, 2, 4, 8]}),
        )

        # Get default estimator settings
        estimator_defaults = self.model_config.get("estimator", {})

        # Get patch_len and ensure context_length is valid
        patch_len = hparams.get("patch_len", estimator_defaults.get("patch_len", 16))
        min_context_length = patch_len * 2
        context_length = max(hparams.get("context_length", min_context_length), min_context_length)

        # Create EarlyStopping callback (we'll use its best_score later)
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
        )

        # Create estimator with sampled hyperparameters
        try:
            estimator = PatchTSTEstimator(
                prediction_length=self.dataset.prediction_length,
                patch_len=patch_len,
                context_length=context_length,
                stride=estimator_defaults.get("stride", 8),
                padding_patch=estimator_defaults.get("padding_patch", "end"),
                d_model=hparams.get("d_model", estimator_defaults.get("d_model", 128)),
                nhead=hparams.get("nhead", estimator_defaults.get("nhead", 4)),
                dim_feedforward=hparams.get("d_model", 128) * 2,  # Usually 2x d_model
                num_encoder_layers=hparams.get(
                    "num_encoder_layers", estimator_defaults.get("num_encoder_layers", 2)
                ),
                dropout=hparams.get("dropout", estimator_defaults.get("dropout", 0.1)),
                activation=estimator_defaults.get("activation", "relu"),
                norm_first=estimator_defaults.get("norm_first", False),
                lr=hparams.get("lr", estimator_defaults.get("lr", 1e-3)),
                weight_decay=hparams.get(
                    "weight_decay", estimator_defaults.get("weight_decay", 1e-8)
                ),
                scaling=estimator_defaults.get("scaling", "mean"),
                batch_size=estimator_defaults.get("batch_size", 128),
                num_batches_per_epoch=estimator_defaults.get("num_batches_per_epoch", 100),
                trainer_kwargs={
                    "max_epochs": self.max_epochs,
                    "devices": 1,
                    "accelerator": "auto",
                    "enable_progress_bar": False,
                    "callbacks": [early_stopping],
                },
            )

            # Train with validation data
            estimator.train(
                self.dataset.training_dataset,
                validation_data=self.dataset.validation_dataset,
            )

            # Get the best validation loss from EarlyStopping callback
            # (same approach as gift_eval)
            best_val_loss = early_stopping.best_score
            if best_val_loss is None:
                print(f"Trial {trial.number}: No val_loss recorded")
                return float("inf")

            # Store the number of epochs for later use in evaluation
            stopped_epoch = early_stopping.stopped_epoch or self.max_epochs
            actual_epochs = stopped_epoch - early_stopping.wait_count
            trial.set_user_attr("epochs", actual_epochs)

            return float(best_val_loss)

        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            return float("inf")


def run_tuning(
    dataset_name: str,
    term: str,
    model_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    output_dir: str,
    n_trials: int = 15,
):
    """Run Optuna hyperparameter tuning."""
    print(f"\n{'='*60}")
    print(f"Hyperparameter Tuning: {dataset_name} / {term}")
    print(f"{'='*60}")

    # Get dataset settings
    settings = get_dataset_settings(dataset_name, term, dataset_config)
    prediction_length = settings.get("prediction_length")
    test_length = settings.get("test_length")
    val_length = settings.get("val_length")

    # Load dataset
    # PatchTST operates on univariate series, so we flatten multivariate datasets
    # into multiple independent univariate series (same as gift_eval)
    dataset = Dataset(
        name=dataset_name,
        term=term,
        to_univariate=True,  # PatchTST requires univariate input
        prediction_length=prediction_length,
        test_length=test_length,
        val_length=val_length,
    )

    print(f"Dataset: {dataset_name}")
    print(f"Term: {term}")
    print(f"Prediction length: {prediction_length}")
    print(f"Number of series: {len(dataset.hf_dataset)}")

    # Create Optuna study
    study_name = f"patch_tst_{dataset_name.replace('/', '_')}_{term}"
    storage_path = Path(output_dir) / "optuna" / f"{study_name}.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        direction="minimize",
    )

    # Create objective
    objective = PatchTSTObjective(
        dataset=dataset,
        model_config=model_config,
        term=term,
        max_epochs=model_config.get("estimator", {}).get("max_epochs", 50),
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save best parameters
    best_params = study.best_params
    hparams_path = get_hyperparams_path(output_dir, dataset_name, term)
    save_best_params(best_params, hparams_path)

    print(f"\nBest trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {best_params}")

    return best_params


def run_training(
    dataset_name: str,
    term: str,
    model_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    output_dir: str,
    hparams: Optional[Dict[str, Any]] = None,
):
    """Train PatchTST model with given hyperparameters."""
    print(f"\n{'='*60}")
    print(f"Training: {dataset_name} / {term}")
    print(f"{'='*60}")

    # Get dataset settings
    settings = get_dataset_settings(dataset_name, term, dataset_config)
    prediction_length = settings.get("prediction_length")
    test_length = settings.get("test_length")
    val_length = settings.get("val_length")

    # Load dataset
    # PatchTST operates on univariate series, so we flatten multivariate datasets
    # into multiple independent univariate series (same as gift_eval)
    dataset = Dataset(
        name=dataset_name,
        term=term,
        to_univariate=True,  # PatchTST requires univariate input
        prediction_length=prediction_length,
        test_length=test_length,
        val_length=val_length,
    )

    # Load hyperparameters if not provided
    if hparams is None:
        hparams_path = get_hyperparams_path(output_dir, dataset_name, term)
        if hparams_path.exists():
            hparams = load_best_params(hparams_path)
            print(f"Loaded hyperparameters from {hparams_path}")
        else:
            print("No hyperparameters found, using defaults")
            hparams = {}

    # Get default estimator settings
    estimator_defaults = model_config.get("estimator", {})

    # Get patch_len first (needed for context_length validation)
    patch_len = hparams.get("patch_len", estimator_defaults.get("patch_len", 16))

    # Compute context_length, ensuring it's at least 2 * patch_len for valid patching
    min_context_length = patch_len * 2
    default_context_length = max(dataset.prediction_length * 4, min_context_length)
    context_length = hparams.get("context_length", default_context_length)
    context_length = max(context_length, min_context_length)

    # Get the number of epochs from tuning (if available)
    # If tuned epochs are available, use them; otherwise use default max_epochs
    hparams_path = get_hyperparams_path(output_dir, dataset_name, term)
    max_epochs = estimator_defaults.get("max_epochs", 50)

    # Create estimator
    # Note: No EarlyStopping for final training (no validation data used)
    # This matches gift_eval's evaluate() behavior
    estimator = PatchTSTEstimator(
        prediction_length=dataset.prediction_length,
        patch_len=patch_len,
        context_length=context_length,
        stride=estimator_defaults.get("stride", 8),
        padding_patch=estimator_defaults.get("padding_patch", "end"),
        d_model=hparams.get("d_model", estimator_defaults.get("d_model", 128)),
        nhead=hparams.get("nhead", estimator_defaults.get("nhead", 4)),
        dim_feedforward=hparams.get("d_model", 128) * 2,
        num_encoder_layers=hparams.get(
            "num_encoder_layers", estimator_defaults.get("num_encoder_layers", 2)
        ),
        dropout=hparams.get("dropout", estimator_defaults.get("dropout", 0.1)),
        activation=estimator_defaults.get("activation", "relu"),
        norm_first=estimator_defaults.get("norm_first", False),
        lr=hparams.get("lr", estimator_defaults.get("lr", 1e-3)),
        weight_decay=hparams.get(
            "weight_decay", estimator_defaults.get("weight_decay", 1e-8)
        ),
        scaling=estimator_defaults.get("scaling", "mean"),
        batch_size=estimator_defaults.get("batch_size", 128),
        num_batches_per_epoch=estimator_defaults.get("num_batches_per_epoch", 100),
        trainer_kwargs={
            "max_epochs": max_epochs,
            "devices": 1,
            "accelerator": "auto",
            # No EarlyStopping - train for fixed epochs on full training data
        },
    )

    # Train on validation dataset (train + val combined for final model)
    # No validation_data passed - we use all available data for training
    print(f"Training model for {max_epochs} epochs...")
    predictor = estimator.train(dataset.validation_dataset)

    return predictor, dataset


def run_test(
    dataset_name: str,
    term: str,
    model_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    output_dir: str,
):
    """Test PatchTST model and save predictions."""
    print(f"\n{'='*60}")
    print(f"Testing: {dataset_name} / {term}")
    print(f"{'='*60}")

    # Train model first
    predictor, dataset = run_training(
        dataset_name=dataset_name,
        term=term,
        model_config=model_config,
        dataset_config=dataset_config,
        output_dir=output_dir,
    )

    # Get seasonality for metrics
    season_length = get_seasonality(dataset.freq)

    # Save predictions using TIME's save_window_predictions
    ds_config = f"{dataset_name}/{term}"

    # Get hyperparameters for metadata
    hparams_path = get_hyperparams_path(output_dir, dataset_name, term)
    if hparams_path.exists():
        hparams = load_best_params(hparams_path)
    else:
        hparams = {}

    metadata = save_window_predictions(
        dataset=dataset,
        predictor=predictor,
        ds_config=ds_config,
        output_base_dir=output_dir,
        seasonality=season_length,
        model_hyperparams=hparams,
    )

    print(f"Completed: {metadata['num_series']} series x {metadata['num_windows']} windows")
    print(f"Output: {output_dir}/{ds_config}")

    return metadata



def get_available_terms(dataset_name: str, config: dict) -> list[str]:
    """Get the terms that are actually defined in the config for a dataset."""
    datasets_config = config.get("datasets", {})
    if dataset_name not in datasets_config:
        return []
    dataset_config = datasets_config[dataset_name]
    available_terms = []
    for term in ["short", "medium", "long"]:
        if term in dataset_config and dataset_config[term].get("prediction_length") is not None:
            available_terms.append(term)
    return available_terms

def run_patch_tst_experiment(
    dataset_name: str = "SG_Weather/D",
    terms: list[str] = None,
    mode: str = "test",
    output_dir: str | None = None,
    model_config_path: Path | None = None,
    dataset_config_path: Path | None = None,
    n_trials: int = 15,
):
    """
    Run PatchTST experiments.

    Args:
        dataset_name: Dataset name (e.g., "SG_Weather/D")
        terms: List of terms to evaluate ("short", "medium", "long")
        mode: "tune", "train", or "test"
        output_dir: Output directory for results
        model_config_path: Path to model config YAML
        dataset_config_path: Path to dataset config YAML
        n_trials: Number of Optuna trials for tuning
    """
    # Load configurations
    model_config = load_model_config(model_config_path)
    dataset_config = load_dataset_config(dataset_config_path)

    # Auto-detect available terms from config if not specified
    if terms is None:
        terms = get_available_terms(dataset_name, dataset_config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if output_dir is None:
        output_dir = "./output/results/patch_tst"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# PatchTST Experiment")
    print(f"# Dataset: {dataset_name}")
    print(f"# Terms: {terms}")
    print(f"# Mode: {mode}")
    print(f"{'#'*60}")

    for term in terms:
        try:
            if mode == "tune":
                run_tuning(
                    dataset_name=dataset_name,
                    term=term,
                    model_config=model_config,
                    dataset_config=dataset_config,
                    output_dir=output_dir,
                    n_trials=n_trials,
                )
            elif mode == "train":
                run_training(
                    dataset_name=dataset_name,
                    term=term,
                    model_config=model_config,
                    dataset_config=dataset_config,
                    output_dir=output_dir,
                )
            elif mode == "test":
                run_test(
                    dataset_name=dataset_name,
                    term=term,
                    model_config=model_config,
                    dataset_config=dataset_config,
                    output_dir=output_dir,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

        except Exception as e:
            print(f"ERROR: Failed for {dataset_name}/{term}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("Experiment completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run PatchTST experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["SG_Weather/D"],
        help="Dataset name(s). Use 'all_datasets' to run all datasets from config",
    )
    parser.add_argument(
        "--terms",
        type=str,
        nargs="+",
        default=None,
        choices=["short", "medium", "long"],
        help="Terms to evaluate. If not specified, auto-detect from config.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["tune", "train", "test"],
        help="Experiment mode: tune (hyperparameter search), train, or test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Path to dataset config YAML",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=15,
        help="Number of Optuna trials for hyperparameter tuning",
    )
    parser.add_argument(
        "--cuda-device",
        type=str,
        default="0",
        help="CUDA device ID",
    )

    args = parser.parse_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Handle dataset list or 'all_datasets'
    model_config_path = Path(args.model_config) if args.model_config else None
    dataset_config_path = Path(args.dataset_config) if args.dataset_config else None

    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        # Load all datasets from config
        config = load_dataset_config(dataset_config_path)
        datasets = list(config.get("datasets", {}).keys())
        print(f"Running all {len(datasets)} datasets from config:")
        for ds in datasets:
            print(f"  - {ds}")
    else:
        datasets = args.dataset

    # Iterate over all datasets
    total_datasets = len(datasets)
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        try:
            run_patch_tst_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                mode=args.mode,
                output_dir=args.output_dir,
                model_config_path=model_config_path,
                dataset_config_path=dataset_config_path,
                n_trials=args.n_trials,
            )
        except Exception as e:
            print(f"ERROR: Failed to run experiment for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'#'*60}")
    print(f"# All {total_datasets} dataset(s) completed!")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()

