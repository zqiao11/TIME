"""
Feature extraction module for time series analysis.

This module provides utilities for:
- Computing time series features (trend, seasonality, stationarity, etc.)
- Evaluating feature distributions across datasets
- Running batch feature extraction on multiple datasets
"""

from timebench.feature.features import (
    convert_to_tsfeatures_panel,
    preprocess_for_tsfeatures,
    tsfeatures_with_uid_freq_map,
    extended_stl_features,
    fast_acf_features,
)

__all__ = [
    # Core feature functions
    "convert_to_tsfeatures_panel",
    "preprocess_for_tsfeatures",
    "tsfeatures_with_uid_freq_map",
    # Feature extractors
    "extended_stl_features",
    "fast_acf_features",
]

# Optional: import features_evaluator if available
try:
    from timebench.feature.features_evaluator import (
        evaluate_single_feature_spread,
        load_feature_data,
    )
    __all__.extend([
    "evaluate_single_feature_spread",
    "load_feature_data",
    ])
except ImportError:
    pass

