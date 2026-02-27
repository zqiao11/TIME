# Codes are adapted from https://github.com/Nixtla/tsfeatures

import re
import warnings
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit
from statsmodels.tsa.stattools import adfuller
from tsfeatures import *
from scipy.stats import kurtosis
from scipy.stats import shapiro
from multiprocessing import Pool
from collections import ChainMap

warnings.filterwarnings("ignore")


# ========================
#  Constants
# ========================
# Define candidate periods for different frequency strings
PERIODS = {
    'S': [60, 3600, 43200, 86400],  # T, H, 12H, D
    'T': [60, 720, 1440, 10080],    # H, 12H, D, W
    'H': [12, 24, 168, 720],        # 12H, D, W, M
    'D': [7, 30, 91, 365],          # W, M, Q, A
    'B': [5, 21, 63, 252],          # W, M, Q, A (business days: 5/week, ~21/month, ~63/quarter, ~252/year)
    'W': [4, 13, 52],               # M, Q, A
    'M': [12],                      # A
    'Q': [4],                      # A
    'A': [1]
}

# Define periods margins for fft
PERIODS_MARGINS = {
    'S':   [10, 600, 3600, 7200],     # 10S, 10T, 1H, 2H
    'T':   [10, 60, 120, 720],        # 10T, 1H, 2H, 12H
    'H':   [1, 2, 12, 24],            # 1H, 2H, 12H, 1D
    'D':   [1, 3, 7, 14],             # 1D, 3D, 1W, 2W
    'B':   [1, 3, 5, 10],             # 1B, 3B, 1W, 2W (business days)
    'W':   [1, 2, 4],                 # 1W, 2W, 1M
    'M':   [1],                        # 1M
    'Q':   [1],                        # 1Q
    'A':   [0]
}


# ========================
# Utility Functions
# ========================
def infer_period(freq):
    """
    Infer the candidate periods of a time series based on its frequency string.

    Parameters:
    - freq: Frequency string (e.g., 'H', 'D', '15T', '2A-DEC').

    Returns:
    - List of candidate periods as integers (adjusted if a numeric prefix exists).

    Raises:
    - ValueError: If the frequency is not recognized.
    """
    if '-' in freq:
        freq = freq.split('-')[0]

    if freq in PERIODS:
        return PERIODS[freq], PERIODS_MARGINS[freq]
    elif freq.isalnum():
        pattern = r"(\d+)([a-zA-Z]+)"
        match = re.match(pattern, freq)
        repeat_count, freq_str = match.groups()
        return ([max(p // int(repeat_count), 1) for p in PERIODS[freq_str]],
                [p / int(repeat_count) for p in PERIODS_MARGINS[freq_str]])
    else:
        raise ValueError(f"Frequency {freq} not recognized")


def period_to_freq_window(p: int, delta_p: int) -> float:
    return abs(1 / (p - delta_p) - 1 / (p + delta_p))


def safe_parse_datetime(series: pd.Series) -> pd.Series:
    """
    Robust datetime parser for time series data.
    Tries automatic parsing first, then falls back to known formats.
    """
    try:
        # Step 1: try pandas auto parsing
        return pd.to_datetime(series, infer_datetime_format=True, dayfirst=False)
    except Exception:
        pass

    # Step 2: try known special formats
    known_formats = [
        "%d.%m.%Y %H:%M",   # e.g. "13.01.2020 00:00"
        "%d.%m.%Y",         # e.g. "13.01.2020"
        "%m/%d/%Y %H:%M",   # e.g. "01/13/2020 00:00"
    ]
    for fmt in known_formats:
        try:
            return pd.to_datetime(series, format=fmt)
        except Exception:
            continue

    # Step 3: fallback with coerce
    return pd.to_datetime(series, errors="coerce")


def convert_to_tsfeatures_panel(
    csv_path: str,
    var_cols: List[str] = None,
    test_length: Optional[int] = None,
    mode: Literal["full", "test"] = "full"
) -> pd.DataFrame:
    """
    Convert a CSV file to tsfeatures panel format.

    CSV format requirement: First column must be timestamp, other columns are values.

    Args:
        csv_path: Path to CSV file
        var_cols: List of variable columns to include (default: all except timestamp)
        test_length: Number of timesteps for test portion (required if mode="test")
        mode: Which portion to compute features on:
            - "full": Use entire series
            - "test": Use only the last `test_length` timesteps

    Returns:
        panel_df: DataFrame with columns ['unique_id', 'ds', 'y']
    """
    df = pd.read_csv(csv_path, parse_dates=[0])

    # First column is timestamp (consistent with dataset_builder.py)
    time_col = df.columns[0]

    # Get var columns (all columns except timestamp)
    if not var_cols:
        var_cols = df.columns[1:].tolist()

    # Ensure time is sorted
    df = df.sort_values(time_col).reset_index(drop=True)

    # Filter based on mode
    if mode == "test":
        if test_length is None:
            raise ValueError("test_length must be provided when mode='test'")
        # Keep only the last test_length rows
        df = df.iloc[-test_length:].reset_index(drop=True)

    # Convert to tsfeatures panel
    records = []
    for var in var_cols:
        temp = pd.DataFrame({
            "unique_id": var,
            "ds": safe_parse_datetime(df[time_col]),
            "y": df[var],
        })
        records.append(temp)

    panel_df = pd.concat(records, ignore_index=True)

    return panel_df


def safe_scalets(y:  pd.Series) -> pd.Series:
    """Mean-std scale y (a series), skipping NaNs."""
    mean = np.nanmean(y)
    std = np.nanstd(y, ddof=1)

    if np.isnan(mean) or np.isnan(std) or std < 1e-8:
        return y  # return unchanged if not scalable

    return (y - mean) / std


def preprocess_for_tsfeatures(
    df: pd.DataFrame,
    freq: str = "D"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameters:
    - df: Time series DataFrame with columns ['unique_id', 'ds', 'y']
    - interp_method: Interpolation method: 'linear' or 'stl'
    - freq: Frequency string (e.g., 'D', 'M'), only used when method='stl'

    Returns:
    - preprocessed_df: DataFrame with NaN-safe scaling and interpolation
    - stats_df: DataFrame with extended statistics per series
    """
    stats = []

    def process(group):
        y_raw = group['y']
        uid = group['unique_id'].iloc[0]
        raw_mean = np.nanmean(y_raw)
        raw_std = np.nanstd(y_raw)
        missing_rate = y_raw.isna().mean()
        length = len(y_raw)

        # safe standardization
        y_scaled = safe_scalets(y_raw)

        # fill NaNs only if needed
        if y_scaled.isna().any():
            y_filled = y_scaled.interpolate(method="linear", limit_direction="both")
        else:
            y_filled = y_scaled

        # Compute FFT features
        fft_vals = np.fft.fft(y_filled)
        freqs = np.fft.fftfreq(len(y_filled))
        power = np.abs(fft_vals)

        # Ignore the DC component (index 0)
        power[0] = 0
        N = len(y_filled)
        positive_freqs = freqs[:N//2]
        positive_power = power[:N//2]
        total_power = positive_power.sum()

        candidate_periods, candidate_margins = infer_period(freq)
        candidate_periods = [p for p in candidate_periods if p < N]

        period_power = []
        for p, delta_p in zip(candidate_periods, candidate_margins):
            f_center = 1 / p
            delta_f = period_to_freq_window(p, delta_p)

            f_low = f_center - delta_f / 2
            f_high = f_center + delta_f / 2

            indices = np.where((positive_freqs >= f_low) & (positive_freqs <= f_high))[0]
            strength = positive_power[indices].sum() if len(indices) > 0 else 0.0
            strength = strength / total_power if total_power > 0 else 0.0

            period_power.append((p, strength))


        # Top 3 peaks
        top_periods_power = sorted(period_power, key=lambda x: -x[1])[:3]
        top_periods = [p for p, _ in top_periods_power]
        top_strengths = [round(s, 3) for _, s in top_periods_power]

        stats.append({
            'unique_id': uid,
            'mean': raw_mean,
            'std': raw_std,
            'missing_rate': missing_rate,
            'length': length,
            'period1': top_periods[0] if len(top_periods) > 0 else np.nan,
            'period2': top_periods[1] if len(top_periods) > 1 else np.nan,
            'period3': top_periods[2] if len(top_periods) > 2 else np.nan,
            'p_strength1': top_strengths[0] if len(top_strengths) > 0 else np.nan,
            'p_strength2': top_strengths[1] if len(top_strengths) > 1 else np.nan,
            'p_strength3': top_strengths[2] if len(top_strengths) > 2 else np.nan
        })

        return group.assign(y=y_filled)

    df_processed = df.groupby('unique_id', group_keys=False).apply(process)
    stats_df = pd.DataFrame(stats)

    return df_processed, stats_df

# ========================
# Feature Definition
# ========================
def _get_feats(index,
               ts,
               freq,
               scale = True,
               features = [acf_features, arch_stat, crossing_points,
                          entropy, flat_spots, heterogeneity, holt_parameters,
                          lumpiness, nonlinearity, pacf_features, stl_features,
                          stability, hw_parameters, unitroot_kpss, unitroot_pp,
                          series_length, hurst],
                dict_freqs = FREQS):

    if freq is None:
        inf_freq = pd.infer_freq(ts['ds'])
        if inf_freq is None:
            raise Exception(
                'Failed to infer frequency from the `ds` column, '
                'please provide the frequency using the `freq` argument.'
            )

        freq = dict_freqs.get(inf_freq)
        if freq is None:
            raise Exception(
                'Error trying to convert infered frequency from the `ds` column '
                'to integer. Please provide a dictionary with that frequency '
                'as key and the integer frequency as value. '
                f'Infered frequency: {inf_freq}'
            )


    if isinstance(ts, pd.DataFrame):
        assert 'y' in ts.columns
        ts = ts['y'].values

    if isinstance(ts, pd.Series):
        ts = ts.values

    if scale:
        ts = scalets(ts)

    c_map = ChainMap(*[dict_feat for dict_feat in [func(ts, freq) for func in features]])

    return pd.DataFrame(dict(c_map), index = [index])


def ts_wrapper(uid_group, uid_freq_map, features, dict_freqs, scale):
    uid, group = uid_group
    freq = uid_freq_map.get(uid, None)
    return _get_feats(index=uid, ts=group, freq=freq, scale=scale,
                      features=features, dict_freqs=dict_freqs)


def tsfeatures_with_uid_freq_map(ts: pd.DataFrame,
                                  uid_freq_map: Dict[str, int],
                                  features: List[Callable],
                                  dict_freqs: Optional[Dict[str, int]]=FREQS,
                                  scale: bool = True,
                                  threads: Optional[int] = None):

    """
    Custom tsfeatures wrapper that assigns each unique_id its own freq
    using precomputed uid_freq_map.
    """
    args = [(uid_group, uid_freq_map, features, dict_freqs, scale)
            for uid_group in ts.groupby('unique_id')]

    with Pool(threads) as pool:
        ts_features = pool.starmap(ts_wrapper, args)

    ts_features = pd.concat(ts_features).rename_axis('unique_id').reset_index()
    return ts_features


@njit
def _acf_lags(x: np.ndarray, max_lag: int = 10):
    n = len(x)
    if n < max_lag + 1:
        return np.nan, np.nan

    mean = np.mean(x)
    var = np.var(x)
    if var < 1e-8:
        return np.nan, np.nan

    acfs = np.empty(max_lag)
    for k in range(1, max_lag + 1):
        total = 0.0
        for i in range(n - k):
            total += (x[i] - mean) * (x[i + k] - mean)
        acfs[k - 1] = total / ((n - k) * var)

    return acfs[0], np.sum(acfs ** 2)


def fast_acf_features(x: np.ndarray, freq: int = 1) -> Dict[str, float]:
    """
    Calculates ACF features using the accelerated helper function.
    Now includes diff1_acf features.
    """
    x_acf1, x_acf10 = _acf_lags(x, max_lag=10)

    diff1_x = np.diff(x)
    diff1_acf1, diff1_acf10 = _acf_lags(diff1_x, max_lag=10)


    return {
        'x_acf1': x_acf1,
        'x_acf10': x_acf10,
        'diff1_acf1': diff1_acf1,
        'diff1_acf10': diff1_acf10
    }


def extended_stl_features(x: np.array, freq: int = 1) -> Dict[str, float]:
    """
    Calculates extended seasonal trend features, including entropy, stability,
    linearity, curvature, and spike.
    """
    m = freq
    n = len(x)
    nperiods = int(m > 1)

    # Compute stationarity (ADF test for random walk)
    try:
        x_clean = x[~np.isnan(x)]
        if len(x_clean) > 10000:
            x_clean = x_clean[-10000:]
        result = adfuller(x_clean, autolag="AIC", regression='ct')
        pval = result[1]

        # stationarity: 1 means stationary (p <= 0.05), 0 means non-stationary
        stationarity = 1 if pval <= 0.05 else 0
    except Exception:
        stationarity = 1

    # Compute entropy on raw series
    x_entropy = entropy(x, m)['entropy']

    # --- STL fits ---
    if m > 1:
        try:
            stlfit = STL(x, m, 13).fit()
        except:
            return _get_empty_output(nperiods, m, x_entropy)
        trend0 = stlfit.trend
        remainder = stlfit.resid
        seasonal = stlfit.seasonal
    else:
        deseas = x
        t = np.arange(len(x)) + 1
        try:
            trend0 = SuperSmoother().fit(t, deseas).predict(t)
        except:
            return _get_empty_output(nperiods, m, x_entropy)
        remainder = deseas - trend0
        seasonal = np.zeros(len(x))

    # --- Decomposition stats ---
    detrend = x - trend0
    deseason = x - seasonal
    varx = np.nanvar(x, ddof=1)
    vare = np.nanvar(remainder, ddof=1)
    vardeseason = np.nanvar(deseason, ddof=1)

    # Trend strength
    if varx < 1e-10 or vardeseason / varx < 1e-10:
        trend = 0
    else:
        trend = max(0, min(1, 1 - vare / vardeseason))

    # Seasonal strength
    if m > 1:
        if varx < 1e-10 or np.nanvar(remainder + seasonal, ddof=1) < 1e-10:
            season = 0
        else:
            season = max(0, min(1, 1 - vare / np.nanvar(remainder + seasonal, ddof=1)))

    # --- ### NEW: Spike Calculation ### ---
    # Spike measures the prevalence of "spiky" noise in the remainder
    d = (remainder - np.nanmean(remainder)) ** 2
    varloo = (vare * (n - 1) - d) / (n - 2) # Leave-one-out variance
    spike = np.nanvar(varloo, ddof=1)

    # --- ### Linearity & Curvature Calculation ### ---
    # Based on orthogonal quadratic regression of the trend
    time = np.arange(n) + 1
    try:
        poly_m = poly(time, 2)
        time_x = add_constant(poly_m)
        coefs = OLS(trend0, time_x).fit().params
        linearity = coefs[1]
        curvature = -coefs[2]
    except:
        linearity = np.nan
        curvature = np.nan

    # --- E ACF features & Entropy ---
    e_acf = fast_acf_features(remainder)
    e_entropy = entropy(remainder, m)['entropy']
    e_arch_lm = arch_stat(remainder, m)['arch_lm']
    e_kurtosis = kurtosis(remainder, fisher=True, nan_policy='omit')
    e_shapiro_w = shapiro(remainder)[0]

    # --- Trend stability & properties ---
    trend_stability = stability(trend0, m)['stability']
    trend_hurst = hurst(trend0, m)['hurst']
    trend_nonlinearity = nonlinearity(trend0, m)['nonlinearity']

    # --- Seasonality properties ---
    seasonal_corr = np.nan
    if m > 1:
        try:
            S = seasonal[:len(seasonal) // m * m]
            segments = S.reshape(-1, m)
            corrs = [np.corrcoef(segments[i], segments[j])[0, 1]
                     for i in range(len(segments)) for j in range(i + 1, len(segments))]
            seasonal_corr = np.mean(corrs) if corrs else np.nan
        except:
            seasonal_corr = np.nan

    seasonal_lumpiness = lumpiness(seasonal, m)['lumpiness']
    seasonal_entropy = entropy(seasonal, m)['entropy']

    # --- Output Assembly ---
    output = {
        'stationarity': stationarity,
        'x_entropy': x_entropy,
        'trend_strength': trend,
        'trend_stability': trend_stability,
        'trend_hurst': trend_hurst,
        'trend_nonlinearity': trend_nonlinearity,
        'spike': spike,
        'linearity': linearity,
        'curvature': curvature,
        'e_acf1': e_acf['x_acf1'],
        'e_diff1_acf1': e_acf['diff1_acf1'],
        'e_acf10': e_acf['x_acf10'],
        'e_entropy': e_entropy,
        'e_arch_lm': e_arch_lm,
        'e_kurtosis': e_kurtosis,
        'e_shapiro_w': e_shapiro_w,
    }

    if m > 1:
        output['seasonal_strength'] = season
        output['seasonal_corr'] = seasonal_corr
        output['seasonal_lumpiness'] = seasonal_lumpiness
        output['seasonal_entropy'] = seasonal_entropy

    return output
