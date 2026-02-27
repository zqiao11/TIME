"""
TIME benchmark model wrappers.

This module provides predictor wrappers for various forecasting models,
adapting them to work with the TIME evaluation framework.
"""

from timebench.models.statsforecast_predictor import (
    NaiveForecast,
    NaivePredictor,
    SeasonalNaiveForecast,
    SeasonalNaivePredictor,
)

__all__ = [
    "NaiveForecast",
    "NaivePredictor",
    "SeasonalNaiveForecast",
    "SeasonalNaivePredictor",
]

