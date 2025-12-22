# Copyright (c) 2023, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2
# Modified by TimeBench for flexible prediction lengths and YAML config support.

import math
import os
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import datasets
import numpy as np
import pyarrow.compute as pc
import yaml
from dotenv import load_dotenv
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.itertools import Map
from gluonts.time_feature import norm_freq_str
from gluonts.transform import Transformation
from pandas.tseries.frequencies import to_offset
from toolz import compose

# --- Constants ---
DEFAULT_TEST_SPLIT = 0.1
DEFAULT_VAL_SPLIT = 0.1
# MAX_WINDOW = 20

# Default prediction length by frequency (used when no config provided)
M4_PRED_LENGTH_MAP = {
    "A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48,
}

PRED_LENGTH_MAP = {
    "M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60,
}

# Default config path (relative to this file's directory)
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "datasets.yaml"


class Term(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        if self == Term.SHORT:
            return 1
        elif self == Term.MEDIUM:
            return 10
        elif self == Term.LONG:
            return 15
        return 1


def load_dataset_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load dataset configuration from YAML file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_dataset_settings(
    name: str,
    term: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get prediction_length, test_split, and val_split for a specific dataset and term.

    Returns dict with keys: 'prediction_length', 'test_split', 'val_split'
    Values are None if not configured (will use default calculation).
    """
    datasets_config = config.get("datasets", {})
    defaults = config.get("defaults", {})

    if name in datasets_config:
        dataset_config = datasets_config[name]
        term_config = dataset_config.get(term, {})
        return {
            "prediction_length": term_config.get("prediction_length"),
            "test_split": dataset_config.get("test_split", defaults.get("test_split")),
            "val_split": dataset_config.get("val_split", defaults.get("val_split")),
        }

    return {
        "prediction_length": None,
        "test_split": None,
        "val_split": None,
    }

def itemize_start(data_entry: DataEntry) -> DataEntry:
    data_entry["start"] = data_entry["start"].item()
    # Fix target shape: if target is 2D with shape (1, N), squeeze it to 1D for univariate case
    if "target" in data_entry:
        target = data_entry["target"]
        if isinstance(target, np.ndarray) and target.ndim == 2 and target.shape[0] == 1:
            data_entry["target"] = np.squeeze(target, axis=0)
    return data_entry

class MultivariateToUnivariate(Transformation):
    def __init__(self, field):
        self.field = field

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool = False
    ) -> Iterator:
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            for id, val in enumerate(val_ls):
                univariate_entry = data_entry.copy()
                univariate_entry[self.field] = val
                univariate_entry["item_id"] = item_id + "_dim" + str(id)
                yield univariate_entry

class Dataset:
    def __init__(
        self,
        name: str,
        term: Term | str = Term.SHORT,
        to_univariate: bool = False,
        prediction_length: Optional[int] = None,
        test_split: Optional[float] = None,
        val_split: Optional[float] = None,
        storage_env_var: str = "GIFT_EVAL",
    ):
        """
        Initialize a TimeBench Dataset.

        Parameters
        ----------
        name : str
            Dataset name (path relative to storage, e.g., "bitbrains_rnd/5T")
        term : Term or str
            Forecast horizon term: "short", "medium", or "long"
        to_univariate : bool
            Convert multivariate to univariate
        prediction_length : int, optional
            Prediction length. If None, use default calculation based on freq and term.
        test_split : float, optional
            Test set ratio (0~1). If None, use default (0.1).
            Windows is auto-calculated: ceil(test_split * min_series_length / prediction_length)
        val_split : float, optional
            Validation set ratio (0~1). If None, use default (0.1).
            Val windows is auto-calculated: ceil(val_split * min_series_length / prediction_length)
        storage_env_var : str
            Environment variable name for dataset storage path.
        """
        load_dotenv()
        env_path = os.getenv(storage_env_var)
        if not env_path:
            raise ValueError(f"Environment variable '{storage_env_var}' not set.")

        storage_path = Path(env_path)
        dataset_path = storage_path / name

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

        self.hf_dataset = datasets.load_from_disk(str(dataset_path)).with_format("numpy")

        self.term = Term(term) if isinstance(term, str) else term
        self.name = name
        self._custom_prediction_length = prediction_length
        self._test_split = test_split if test_split is not None else DEFAULT_TEST_SPLIT
        self._val_split = val_split if val_split is not None else DEFAULT_VAL_SPLIT

        process = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )

        self.gluonts_dataset = Map(compose(process, itemize_start), self.hf_dataset)
        if to_univariate:
            self.gluonts_dataset = MultivariateToUnivariate("target").apply(
                self.gluonts_dataset
            )

    @cached_property
    def prediction_length(self) -> int:
        """Get prediction length. Uses explicit value if provided, else default calculation."""
        if self._custom_prediction_length is not None:
            return self._custom_prediction_length

        # Default calculation based on frequency and term
        freq = norm_freq_str(to_offset(self.freq).name)
        pred_len = (
            M4_PRED_LENGTH_MAP[freq] if "m4" in self.name else PRED_LENGTH_MAP[freq]
        )
        return self.term.multiplier * pred_len

    @cached_property
    def freq(self) -> str:
        return self.hf_dataset[0]["freq"]

    @cached_property
    def target_dim(self) -> int:
        target = self.hf_dataset[0]["target"]
        return target.shape[0] if len(target.shape) > 1 else 1

    @cached_property
    def past_feat_dynamic_real_dim(self) -> int:
        if "past_feat_dynamic_real" not in self.hf_dataset[0]:
            return 0
        feat = self.hf_dataset[0]["past_feat_dynamic_real"]
        return feat.shape[0] if len(feat.shape) > 1 else 1

    @cached_property
    def test_split(self) -> float:
        """Get test split ratio."""
        return self._test_split

    @cached_property
    def val_split(self) -> float:
        """Get validation split ratio."""
        return self._val_split

    @cached_property
    def windows(self) -> int:
        """
        Get number of test windows.
        Auto-calculated: ceil(test_split * min_series_length / prediction_length)
        """
        if "m4" in self.name:
            return 1
        w = math.ceil(self._test_split * self._min_series_length / self.prediction_length)
        # return min(max(1, w), MAX_WINDOW)
        return max(1, w)

    @cached_property
    def val_windows(self) -> int:
        """
        Get number of validation windows.
        Auto-calculated: ceil(val_split * min_series_length / prediction_length)
        """
        if "m4" in self.name:
            return 1
        w = math.ceil(self._val_split * self._min_series_length / self.prediction_length)
        # return min(max(1, w), MAX_WINDOW)
        return max(1, w)

    @cached_property
    def _series_lengths(self):
        """Get array of all series lengths."""
        target_col = self.hf_dataset.data.column("target")
        if self.hf_dataset[0]["target"].ndim > 1:
            # Multivariate: get length of inner list
            lengths = pc.list_value_length(pc.list_flatten(pc.list_slice(target_col, 0, 1)))
        else:
            lengths = pc.list_value_length(target_col)
        return lengths.to_numpy()

    @cached_property
    def _min_series_length(self) -> int:
        return int(min(self._series_lengths))

    @cached_property
    def _max_series_length(self) -> int:
        return int(max(self._series_lengths))

    @cached_property
    def _avg_series_length(self) -> float:
        return float(self._series_lengths.mean())

    @cached_property
    def sum_series_length(self) -> int:
        target_col = self.hf_dataset.data.column("target")
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(pc.list_flatten(target_col))
        else:
            lengths = pc.list_value_length(target_col)
        return sum(lengths.to_numpy())

    @property
    def training_dataset(self) -> TrainingDataset:
        training_dataset, _ = split(
            self.gluonts_dataset, offset=-self.prediction_length * (self.windows + 1)
        )
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        validation_dataset, _ = split(
            self.gluonts_dataset, offset=-self.prediction_length * self.windows
        )
        return validation_dataset

    @property
    def test_data(self) -> TestData:
        _, test_template = split(
            self.gluonts_dataset, offset=-self.prediction_length * self.windows
        )
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )
        return test_data

    @property
    def val_data(self) -> TestData:
        """
        Get validation data in the same format as test_data.
        Returns TestData with val_windows instances for evaluation.
        """
        # Split at offset that includes both test and val windows
        # Validation set is before test set, so we need to split further back
        total_test_val_windows = self.windows + self.val_windows
        _, val_template = split(
            self.gluonts_dataset, offset=-self.prediction_length * total_test_val_windows
        )
        # Generate validation instances starting from the split point
        val_data = val_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.val_windows,
            distance=self.prediction_length,
        )
        return val_data
