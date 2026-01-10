"""
StatsForecast model wrappers for TIME benchmark.
Adapted from gift_eval implementation.

This module provides predictors that wrap statsforecast models
to work with the TIME evaluation framework.
"""

from typing import Iterator

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive


class SeasonalNaiveForecast:
    """
    Wrapper for SeasonalNaive forecast results.

    Since SeasonalNaive is a deterministic (point) forecast method,
    we replicate the forecast to create multiple samples for compatibility
    with the TIME evaluation framework which expects probabilistic forecasts.

    Attributes:
        samples: Forecast samples with shape (num_samples, num_variates, pred_len)
    """

    def __init__(self, predictions: np.ndarray, num_samples: int = 100):
        """
        Initialize forecast wrapper.

        Args:
            predictions: Point forecast array with shape (num_variates, pred_len)
                        or (pred_len,) for univariate
            num_samples: Number of samples to create (all identical for point forecast)
        """
        # Ensure predictions is 2D: (num_variates, pred_len)
        if predictions.ndim == 1:
            predictions = predictions[np.newaxis, :]

        # Create samples by replicating: (num_samples, num_variates, pred_len)
        self._samples = np.repeat(predictions[np.newaxis, :, :], num_samples, axis=0)

    @property
    def samples(self) -> np.ndarray:
        """Return forecast samples with shape (num_samples, num_variates, pred_len)."""
        return self._samples

    @property
    def mean(self) -> np.ndarray:
        """Return mean forecast with shape (num_variates, pred_len)."""
        return self._samples[0]


class SeasonalNaivePredictor:
    """
    Predictor wrapping statsforecast SeasonalNaive model.

    The Seasonal Naive method forecasts the value from the same season
    in the previous seasonal cycle. For example, with daily data and
    season_length=7, it forecasts Monday with last Monday's value.

    This predictor handles both univariate and multivariate time series
    by processing each variate independently.
    """

    def __init__(
        self,
        prediction_length: int,
        season_length: int,
        freq: str,
        num_samples: int = 100,
    ):
        """
        Initialize SeasonalNaive predictor.

        Args:
            prediction_length: Number of time steps to forecast
            season_length: Seasonal period length (e.g., 7 for weekly, 24 for daily)
            freq: Frequency string (e.g., 'D', 'H', '15T')
            num_samples: Number of samples to generate (all identical for point forecast)
        """
        self.prediction_length = prediction_length
        self.season_length = season_length
        self.freq = freq
        self.num_samples = num_samples

    def _forecast_single_variate(self, history: np.ndarray) -> np.ndarray:
        """
        Generate forecast for a single univariate time series.

        Args:
            history: Historical values with shape (context_len,)

        Returns:
            Forecast array with shape (prediction_length,)
        """
        # Seasonal Naive: repeat the last season's values
        # If history is shorter than season_length, use what we have cyclically
        effective_season = min(self.season_length, len(history))

        # Get the last effective_season values
        last_season = history[-effective_season:]

        # Tile to cover prediction_length
        num_tiles = (self.prediction_length + effective_season - 1) // effective_season
        forecast = np.tile(last_season, num_tiles)[:self.prediction_length]

        return forecast

    def predict(self, dataset_input) -> Iterator[SeasonalNaiveForecast]:
        """
        Generate forecasts for all series in the dataset.

        Args:
            dataset_input: Iterable of data entries, each containing:
                - "target": Historical values with shape (context_len,) for univariate
                           or (num_variates, context_len) for multivariate

        Yields:
            SeasonalNaiveForecast objects with predictions
        """
        for entry in dataset_input:
            target = entry["target"]

            # Handle univariate case
            if target.ndim == 1:
                forecast = self._forecast_single_variate(target)
                yield SeasonalNaiveForecast(forecast, self.num_samples)

            # Handle multivariate case
            else:
                num_variates = target.shape[0]
                forecasts = np.zeros((num_variates, self.prediction_length))

                for v in range(num_variates):
                    forecasts[v] = self._forecast_single_variate(target[v])

                yield SeasonalNaiveForecast(forecasts, self.num_samples)


class NaiveForecast:
    """
    Wrapper for Naive forecast results.

    The Naive method simply repeats the last observed value.
    """

    def __init__(self, predictions: np.ndarray, num_samples: int = 100):
        """
        Initialize forecast wrapper.

        Args:
            predictions: Point forecast array with shape (num_variates, pred_len)
                        or (pred_len,) for univariate
            num_samples: Number of samples to create
        """
        if predictions.ndim == 1:
            predictions = predictions[np.newaxis, :]

        self._samples = np.repeat(predictions[np.newaxis, :, :], num_samples, axis=0)

    @property
    def samples(self) -> np.ndarray:
        return self._samples

    @property
    def mean(self) -> np.ndarray:
        return self._samples[0]


class NaivePredictor:
    """
    Predictor wrapping the Naive forecasting method.

    The Naive method forecasts all future values as the last observed value.
    This is a simple baseline that works well for random walk processes.
    """

    def __init__(
        self,
        prediction_length: int,
        num_samples: int = 100,
    ):
        """
        Initialize Naive predictor.

        Args:
            prediction_length: Number of time steps to forecast
            num_samples: Number of samples to generate
        """
        self.prediction_length = prediction_length
        self.num_samples = num_samples

    def predict(self, dataset_input) -> Iterator[NaiveForecast]:
        """
        Generate forecasts for all series in the dataset.

        Args:
            dataset_input: Iterable of data entries

        Yields:
            NaiveForecast objects with predictions
        """
        for entry in dataset_input:
            target = entry["target"]

            # Handle univariate case
            if target.ndim == 1:
                last_value = target[-1]
                forecast = np.full(self.prediction_length, last_value)
                yield NaiveForecast(forecast, self.num_samples)

            # Handle multivariate case
            else:
                num_variates = target.shape[0]
                forecasts = np.zeros((num_variates, self.prediction_length))

                for v in range(num_variates):
                    forecasts[v] = target[v, -1]

                yield NaiveForecast(forecasts, self.num_samples)

