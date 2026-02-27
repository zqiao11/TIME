import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


FREQ_MIN_LENGTH = {
    "S": 3600 * 24 * 3,
    "T": 60 * 24 * 28,
    "min": 60 * 24 * 28,
    "H": 24 * 30,
    "h": 24 * 30,
    "D": 183 * 1,
    "B": 130,
    "W": 52 * 2,
    "M": 12 * 10,
    "MS": 12 * 10,
    "ME": 12 * 10,
    "Q": 4 * 10,
    "QS": 4 * 10,
    "QE": 4 * 10,
    "Y": 20,
    "YS": 20,
    "YE": 20,
    "A": 20,
    "default": 20,
}


class PreprocessPipeline:
    def __init__(
            self,
            freq: str | None = None,
            min_length: int | None = None,
            missing_rate_thresh: float = 0.2,
            corr_thresh: float = 0.95,
    ):
        """
        Initializes the time series preprocessing pipeline.

        Args:
            freq: Time series frequency (e.g., '1H'). Inferred automatically if None.
            min_length: Minimum required sequence length. Calculated from freq if None.
            missing_rate_thresh: Maximum allowed missing value ratio.
            corr_thresh: Pearson correlation threshold for detecting highly correlated variables.
        """
        self._freq_override = freq
        self._min_length_override = min_length
        self.min_length = min_length if min_length is not None else FREQ_MIN_LENGTH["default"]
        self.missing_rate_thresh = missing_rate_thresh
        self.corr_thresh = corr_thresh
        self.inferred_freq = None

    def run(self, df: pd.DataFrame, output_path: str | None = None) -> tuple[pd.DataFrame, dict]:
        """Runs the complete preprocessing pipeline on a DataFrame."""
        results = {}
        df = self._normalize_timestamp_column(df)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # 1. Frequency and min_length
        self.inferred_freq = self._freq_override or self._infer_frequency(df)
        if self._min_length_override is None:
            self.min_length = self._get_min_length_for_freq(self.inferred_freq)

        # 2. Univariate checks
        cleaned_columns = {}
        recommended_drop = []

        for col in df.columns:
            col_res = self._run_univariate(df[col])
            results[col] = col_res

            if not col_res["predictable"]:
                recommended_drop.append(col)

            # Use cleaned_ts if available, else fallback to original
            cleaned_ts = col_res.get("cleaned_ts")
            cleaned_columns[col] = cleaned_ts if cleaned_ts is not None else df[col]

        cleaned_df = pd.DataFrame(cleaned_columns)

        # 3. Multivariate checks (Pearson correlation)
        if len(cleaned_df.columns) > 1:
            corr_matrix = cleaned_df.corr(method="pearson")
            cols = corr_matrix.columns
            corr_dups = [
                (cols[i], cols[j], round(corr_matrix.iloc[i, j], 4))
                for i in range(len(cols)) for j in range(i + 1, len(cols))
                if abs(corr_matrix.iloc[i, j]) > self.corr_thresh
            ]
            results["multivariate"] = {
                "correlation_duplicates": corr_dups,
                "correlation_matrix": corr_matrix.to_dict()
            }

        # 4. Compile metadata
        results["_meta"] = {
            "inferred_freq": self.inferred_freq,
            "min_length": self.min_length,
            "n_rows": len(cleaned_df),
            "n_cols": len(cleaned_df.columns),
            "shape": list(cleaned_df.shape),
            "original_columns": list(df.columns),
            "kept_columns": list(cleaned_df.columns),
            "recommended_drop_columns": recommended_drop,
            "num_observations": int(cleaned_df.notna().sum().sum()),
        }

        # 5. Save and return
        if output_path:
            cleaned_df.reset_index().to_csv(output_path, index=False)
            results["_meta"]["output_path"] = output_path

        return cleaned_df, results

    def _compute_cross_series_correlation(
        self,
        series_data: dict[str, pd.DataFrame],
        corr_thresh: float
    ) -> dict:
        """
        计算不同series（CSV文件）之间的相关性
        适用于UTS数据集：每个CSV是单变量时间序列

        要求：所有UTS的长度必须相同才能计算相关性

        Args:
            series_data: {csv_file_name: cleaned_df} 字典
            corr_thresh: 相关性阈值

        Returns:
            dict: 包含相关性矩阵和高相关对的信息（如果所有series都是UTS且长度相同）
        """
        if len(series_data) < 2:
            return {}

        # 检查是否所有series都是单变量的
        all_uts = True
        uts_dict = {}  # {csv_file: series}

        for csv_file, df in series_data.items():
            # 如果timestamp是索引，重置为列
            if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()

            # 确保timestamp列存在
            if 'timestamp' not in df.columns:
                continue

            # 获取唯一的变量列（排除timestamp）
            data_cols = [c for c in df.columns if c != 'timestamp']
            if len(data_cols) != 1:
                all_uts = False
                break

            var_col = data_cols[0]
            ts = df[var_col].values  # 只取数值，不关心时间戳
            uts_dict[csv_file] = ts

        if not all_uts or len(uts_dict) < 2:
            return {}

        # 检查所有UTS的长度是否相同
        lengths = [len(ts) for ts in uts_dict.values()]
        if len(set(lengths)) != 1:
            # 长度不一致，无法计算相关性
            return {}

        # 构建DataFrame用于计算相关性
        all_ts = pd.DataFrame(uts_dict)

        # 计算相关性矩阵
        corr_matrix = all_ts.corr(method='pearson')

        # 找出高度相关的UTS对
        corr_duplicates = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val) and abs(corr_val) > corr_thresh:
                    series1 = corr_matrix.index[i]
                    series2 = corr_matrix.columns[j]
                    corr_duplicates.append((series1, series2, round(corr_val, 4)))

        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "correlation_duplicates": corr_duplicates,
            "num_series": len(uts_dict),
            "series_length": lengths[0] if lengths else 0
        }

    def _run_univariate(self, ts: pd.Series) -> dict:
        """
        Runs univariate checks and cleaning.
        Returns dict with predictability, failure reasons (checks), and cleaned sequence.
        """
        results = {
            "predictable": True,
            "checks": [],  # Only stores failure reasons now
            "cleaned_ts": None,
        }

        # Step 1: Data type
        passed, val = self._check_dtype(ts)
        if not passed:
            results["checks"].append(f"❌ Dtype failed: {val}")
            results["predictable"] = False
            return results

        # Step 2: Length
        passed, val = self._check_length(ts)
        if not passed:
            results["checks"].append(f"❌ Length failed: {val} < {self.min_length}")
            results["predictable"] = False

        # Step 3: Timestamp integrity
        ts, ts_passed, ts_info = self._check_timestamp(ts)
        if not ts_passed:
            results["checks"].append(f"❌ Timestamp failed: {ts_info}")
            results["predictable"] = False

        # Step 4: Missing rate
        passed, val = self._check_missing(ts)
        if not passed:
            results["checks"].append(f"❌ Missing rate failed: {val:.2f}%")
            results["predictable"] = False

        ts_original = ts.copy()
        ts_for_checks = ts.ffill().bfill().fillna(0)

        if not results["predictable"]:
            return results

        # Step 5: Constant check
        passed, topk_dom, entropy = self._check_constant(ts_for_checks)
        if not passed:
            results["checks"].append(f"❌ Constant series: dom={topk_dom:.2f}%, ent={entropy:.2f}")
            results["predictable"] = False
            return results

        # Step 6: White noise check
        is_not_white_noise, wn_pval = self._check_white_noise(ts_for_checks)
        if not is_not_white_noise:
            results["checks"].append(f"❌ White noise Detected: p={wn_pval}")
            results["predictable"] = False
            return results

        # Step 7: Outlier check and cleaning
        keep_variate, cleaned_ts_filled, outlier_stats = self._check_and_clean_outliers(ts_for_checks)
        results["outlier_stats"] = outlier_stats

        if "error" in outlier_stats:
            results["checks"].append(f"❌ Outlier check error: {outlier_stats['error']}")
            results["predictable"] = False
            return results

        if not keep_variate:
            results["checks"].append(f"❌ Extreme outliers: {outlier_stats['extreme_outlier_ratio']:.2f}%")
            results["predictable"] = False
            return results

        # Map cleaned values back, preserving original NaNs
        if cleaned_ts_filled is not None:
            cleaned_ts = ts_original.copy()
            mask = ~ts_original.isna()
            cleaned_ts[mask] = cleaned_ts_filled[mask]
            results["cleaned_ts"] = cleaned_ts
        else:
            results["cleaned_ts"] = ts_original

        return results

    def _infer_frequency(self, df: pd.DataFrame) -> str:
        """Infers time series frequency automatically."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index is not a DatetimeIndex. Please specify 'freq' manually.")

        freq = pd.infer_freq(df.index)

        if freq is None:
            raise ValueError("Cannot infer frequency (e.g., irregular intervals). Please specify 'freq' manually.")

        print(f"[Preprocess] Inferred freq: {freq}")
        return freq

    def _get_min_length_for_freq(self, freq: str) -> int:
        """Calculates minimum required sequence length based on frequency."""
        match = re.match(r'^(\d*)([A-Za-z]+)$', freq)
        if not match:
            print(f"[Preprocess] Cannot parse freq '{freq}', using default min_length.")
            return FREQ_MIN_LENGTH["default"]

        multiplier_str, base_freq = match.groups()
        multiplier = int(multiplier_str) if multiplier_str else 1

        # Lookup exact match or uppercase match as fallback
        base_min = FREQ_MIN_LENGTH.get(base_freq) or FREQ_MIN_LENGTH.get(base_freq.upper())

        if base_min is None:
            print(f"[Preprocess] Unknown base freq '{base_freq}', using default min_length.")
            return FREQ_MIN_LENGTH["default"]

        adjusted_min = max(1, base_min // multiplier)

        return adjusted_min

    def _normalize_timestamp_column(self, df: pd.DataFrame) -> pd.DataFrame:
            """Finds the timestamp column, renames it to 'timestamp', and moves it to the front."""
            df = df.copy()
            time_col = None

            # Identify timestamp column by name or dtype
            for col in df.columns:
                if "time" in col.lower() or "date" in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col]):
                    time_col = col
                    break

            if not time_col:
                raise ValueError("No timestamp column found (requires 'time'/'date' in name or datetime dtype).")

            # Rename to 'timestamp' if necessary
            if time_col != "timestamp":
                df = df.rename(columns={time_col: "timestamp"})

            # Move 'timestamp' to the first column
            cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
            return df[cols]

    def _check_dtype(self, ts):
        return np.issubdtype(ts.dtype, np.number), ts.dtype

    def _check_length(self, ts):
        return len(ts) >= self.min_length, len(ts)

    def _check_missing(self, ts):
        missing_rate = ts.isna().mean()
        return missing_rate <= self.missing_rate_thresh, missing_rate * 100

    def _check_timestamp(self, ts: pd.Series) -> tuple[pd.Series, bool, str]:
        """Checks timestamp integrity and fills missing timestamps with NaN."""
        if not isinstance(ts.index, pd.DatetimeIndex):
            return ts, False, "not_datetime_index"

        if not ts.index.is_monotonic_increasing:
            return ts, False, "not_monotonic"

        if self.inferred_freq and len(ts) > 1:
            freq = self._get_freq_for_date_range(ts.index)
            full_range = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq=freq)

            missing_count = len(full_range) - len(ts)
            if missing_count > 0:
                ts = ts.reindex(full_range)
                ts.index.name = "timestamp"
                print(f"[Preprocess] Filled {missing_count} missing timestamps.")
                return ts, True, f"filled_{missing_count}"

        return ts, True, "ok"

    def _get_freq_for_date_range(self, index: pd.DatetimeIndex) -> str:
        """Adjusts weekly frequency string to match the most common weekday in the data."""
        freq = self.inferred_freq

        if freq and freq.upper().startswith('W') and not index.empty:
            most_common_weekday = index.dayofweek.value_counts().idxmax()
            # 0=Monday, 6=Sunday
            weekdays = ['W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN']
            return weekdays[most_common_weekday]

        return freq

    def _check_constant(self, ts):
        topk = 5
        counts = ts.value_counts(normalize=True)
        topk_dominance = counts.iloc[:topk].sum()

        probs = counts.values
        if len(probs) == 1:
            entropy = 0.0
        else:
            entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs))

        return topk_dominance < 0.5 and entropy > 0.1, topk_dominance * 100, round(entropy, 4)

    def _check_white_noise(self, ts: pd.Series) -> tuple[bool, float | None]:
        """
        Ljung-Box test for white noise.
        Returns (is_not_white_noise, p_value). p <= 0.05 means it has auto-correlation (Passed).
        """
        try:
            ts_clean = ts.dropna()
            # Limit to 10k time steps to avoid O(n^2) performance drop in acorr_ljungbox
            if len(ts_clean) > 10000:
                ts_clean = ts_clean.iloc[-10000:]

            result = acorr_ljungbox(ts_clean, lags=[10, 20], return_df=True)
            pval = result['lb_pvalue'].min()
            return pval <= 0.05, round(pval, 4)
        except Exception:
            return True, None

    def _check_and_clean_outliers(
            self,
            ts: pd.Series,
            window_size: int | None = None,
            k_transient: float = 3.0,
            k_extreme: float = 9.0,
            extreme_thresh: float = 0.05
    ) -> tuple[bool, pd.Series | None, bool, dict]:
        """
        Detects and cleans outliers using a rolling IQR method.
        - Transient Spikes (k_transient < dev < k_extreme): Marks 'spike_presence' if frequent.
        - Extreme Outliers (dev >= k_extreme): Drops variate if frequent, otherwise interpolates.
        """
        if ts.std() == 0 or ts.isna().all():
            return False, None, False, {"error": "constant or all NaN"}

        window_size = window_size or max(20, len(ts) // 10)

        # Compute rolling median and IQR efficiently
        roller = ts.rolling(window=window_size, center=True, min_periods=1)
        rolling_median = roller.median()
        rolling_iqr = roller.quantile(0.75) - roller.quantile(0.25)

        # Fallback for 0 IQR
        global_iqr = ts.quantile(0.75) - ts.quantile(0.25)
        if global_iqr == 0:
            global_iqr = ts.std() * 1.35  # Approx IQR for normal distribution
        rolling_iqr = rolling_iqr.replace(0, global_iqr)

        # Compute deviations
        deviation = np.abs(ts - rolling_median) / rolling_iqr
        is_transient = (deviation > k_transient) & (deviation < k_extreme)
        is_extreme = deviation >= k_extreme

        transient_ratio = is_transient.mean()
        extreme_ratio = is_extreme.mean()

        stats = {
            "transient_spike_ratio": round(transient_ratio * 100, 2),
            "extreme_outlier_ratio": round(extreme_ratio * 100, 2)
        }

        # 1. Drop check: too many extreme outliers
        if extreme_ratio > extreme_thresh:
            stats["error"] = f"extreme_outliers {stats['extreme_outlier_ratio']}% > limit"
            return False, None, False, stats

        # 2. Clean extreme outliers (replace with previous normal value)
        cleaned_ts = ts.copy()
        if is_extreme.any():
            cleaned_ts[is_extreme] = np.nan
            cleaned_ts = cleaned_ts.ffill().bfill()

        return True, cleaned_ts, stats


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, (np.bool_, np.generic)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return None
    else:
        return obj


def save_result_to_json(result: dict, json_path: str) -> None:
    """Saves preprocessing results to JSON, excluding raw series and multivariate data."""
    serializable = {}
    for k, v in result.items():
        if k == "multivariate":
            continue
        if isinstance(v, dict) and k != "_meta":
            serializable[k] = convert_to_serializable({k_in: v_in for k_in, v_in in v.items() if k_in != "cleaned_ts"})
        else:
            serializable[k] = convert_to_serializable(v)

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=4, ensure_ascii=False)
    print(f"[Preprocess] Saved details to: {json_path}")


def process_single_csv(
    csv_path: str,
    output_csv_path: str,
    output_json_path: str,
    freq: str | None = None,
    missing_rate_thresh: float = 0.2,
) -> tuple[pd.DataFrame | None, dict | None]:
    """Processes a single CSV, runs the pipeline, and saves outputs."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Preprocess] ❌ Failed to read CSV: {e}")
        return None, None

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    try:
        pipeline = PreprocessPipeline(
            freq=freq,
            missing_rate_thresh=missing_rate_thresh,
        )
        cleaned_df, result = pipeline.run(df, output_path=output_csv_path)
    except Exception as e:
        print(f"[Preprocess] ❌ Pipeline failed: {e}")
        return None, None

    save_result_to_json(result, output_json_path)
    return cleaned_df, result


def _recalculate_num_observations(json_path: str, csv_path: str) -> None:
    """Recalculates and updates num_observations in a JSON file."""
    try:
        if not os.path.exists(csv_path): return
        df = pd.read_csv(csv_path)
        num_obs = int(df.drop(columns=["timestamp"], errors="ignore").notna().sum().sum())

        with open(json_path, "r") as f: data = json.load(f)
        if "_meta" in data: data["_meta"]["num_observations"] = num_obs
        with open(json_path, "w", encoding="utf-8") as f: json.dump(data, f, indent=4)
    except Exception as e:
        print(f"    ⚠️ Failed to recalc num_observations ({json_path}): {e}")


def _remove_cols_from_json(json_path: str, cols_to_drop: list[str], csv_path: str | None = None) -> None:
    """Removes specified columns from a single JSON metadata file."""
    try:
        with open(json_path, "r") as f: data = json.load(f)
        for col in cols_to_drop: data.pop(col, None)

        if "_meta" in data:
            meta = data["_meta"]
            meta["kept_columns"] = [c for c in meta.get("kept_columns", []) if c not in cols_to_drop]

            dropped = meta.setdefault("dropped_columns", [])
            for col in cols_to_drop:
                if col not in dropped: dropped.append(col)

            meta["n_cols"] = len(meta["kept_columns"])
            if "shape" in meta: meta["shape"][1] = meta["n_cols"]

        with open(json_path, "w", encoding="utf-8") as f: json.dump(data, f, indent=4)
        if csv_path: _recalculate_num_observations(json_path, csv_path)
    except Exception as e:
        print(f"    ⚠️ Failed to update JSON ({json_path}): {e}")


def _recalc_summary_json(summary_json_path: str, csv_dir: str) -> None:
    """Recalculates _summary.json dataset-level stats based on remaining files."""
    try:
        with open(summary_json_path, "r") as f: data = json.load(f)
        csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
        json_dir = os.path.dirname(summary_json_path)

        v_stats, lengths, total_obs = {}, [], 0

        for csv_file in csv_files:
            j_path = os.path.join(json_dir, csv_file.replace(".csv", ".json"))
            if not os.path.exists(j_path): continue

            with open(j_path, "r") as f: s_data = json.load(f)
            meta = s_data.get("_meta", {})
            if meta.get("n_rows", 0) > 0: lengths.append(meta["n_rows"])
            total_obs += meta.get("num_observations", 0)

            for col, res in s_data.items():
                if col.startswith("_") or col == "multivariate": continue
                vs = v_stats.setdefault(col, {"total": 0, "kept": 0})
                vs["total"] += 1
                if res.get("predictable"): vs["kept"] += 1

        data["num_series"] = data["success_count"] = len(csv_files)
        if len(csv_files) > 1 and lengths:
            data.update({
                "num_observations": total_obs,
                "max_series_length": max(lengths),
                "min_series_length": min(lengths),
                "avg_series_length": round(sum(lengths) / len(lengths), 2)
            })
        else:
            for k in ["num_observations", "max_series_length", "min_series_length", "avg_series_length"]:
                data.pop(k, None)

        data["variates"] = {}
        for var, st in v_stats.items():
            tot = st["total"]
            data["variates"][var] = {
                **st,
                "kept_ratio": round(st["kept"]/tot, 4) if tot else 0,
            }

        data.pop("correlation_duplicates", None)
        with open(summary_json_path, "w", encoding="utf-8") as f: json.dump(data, f, indent=4)
    except Exception as e:
        print(f"  ⚠️ Failed to update summary JSON: {e}")


def remove_variate_from_dataset(target_dir: str, variate_name: str, dry_run: bool = False) -> None:
    """Removes a specific variate (column) from all CSVs and updates JSON metadata."""
    if not os.path.isdir(target_dir): raise ValueError(f"Dir not found: {target_dir}")
    csv_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.csv')])
    json_dir = target_dir.replace("processed_csv", "processed_summary")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Removing variate: {variate_name} from {target_dir}")
    mod_count = 0

    for csv_file in csv_files:
        csv_path = os.path.join(target_dir, csv_file)
        df = pd.read_csv(csv_path)
        cols_to_drop = [c for c in df.columns if c == variate_name]

        if cols_to_drop:
            print(f"  {csv_file}: Dropping {cols_to_drop}")
            if not dry_run:
                df.drop(columns=cols_to_drop).to_csv(csv_path, index=False)
                j_path = os.path.join(json_dir, csv_file.replace(".csv", ".json"))
                if os.path.exists(j_path): _remove_cols_from_json(j_path, cols_to_drop, csv_path)
            mod_count += 1

    summary_path = os.path.join(json_dir, "_summary.json")
    if os.path.exists(summary_path) and not dry_run: _recalc_summary_json(summary_path, target_dir)
    print(f"{'Would modify' if dry_run else 'Modified'} {mod_count}/{len(csv_files)} files.")


def remove_drop_marked_variates(target_dir: str, dry_run: bool = False) -> None:
    """Removes all variates that failed checks (recommended_drop) from the dataset."""
    if not os.path.isdir(target_dir): raise ValueError(f"Dir not found: {target_dir}")
    csv_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.csv')])
    json_dir = target_dir.replace("processed_csv", "processed_summary")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Removing failed variates (recommended drops) from {target_dir}")
    mod_count, dropped_vars = 0, set()

    for csv_file in csv_files:
        csv_path = os.path.join(target_dir, csv_file)
        j_path = os.path.join(json_dir, csv_file.replace(".csv", ".json"))

        if not os.path.exists(j_path): continue

        # Read metadata from JSON to find what needs to be dropped
        with open(j_path, "r") as f:
            data = json.load(f)

        cols_to_drop = data.get("_meta", {}).get("recommended_drop_columns", [])

        if cols_to_drop:
            df = pd.read_csv(csv_path)
            valid_cols_to_drop = [c for c in cols_to_drop if c in df.columns]

            if valid_cols_to_drop:
                print(f"  {csv_file}: Dropping {valid_cols_to_drop}")
                dropped_vars.update(valid_cols_to_drop)
                if not dry_run:
                    df.drop(columns=valid_cols_to_drop).to_csv(csv_path, index=False)
                    _remove_cols_from_json(j_path, valid_cols_to_drop, csv_path)
                mod_count += 1

    summary_path = os.path.join(json_dir, "_summary.json")
    if os.path.exists(summary_path) and not dry_run: _recalc_summary_json(summary_path, target_dir)
    print(f"{'Would modify' if dry_run else 'Modified'} {mod_count}/{len(csv_files)} files. Total unique vars dropped: {len(dropped_vars)}")


def remove_series_from_dataset(target_dir: str, series_name: str, dry_run: bool = False) -> None:
    """Removes a specific series (CSV + JSON) from the dataset."""
    if not series_name.endswith('.csv'): series_name += '.csv'
    csv_path = os.path.join(target_dir, series_name)
    json_dir = target_dir.replace("processed_csv", "processed_summary")
    json_path = os.path.join(json_dir, series_name.replace(".csv", ".json"))

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Removing series: {series_name}")
    if not os.path.exists(csv_path):
        print(f"  ❌ File not found: {csv_path}")
        return

    if dry_run:
        print(f"  Would delete CSV: {csv_path} and JSON: {json_path}")
    else:
        os.remove(csv_path)
        if os.path.exists(json_path): os.remove(json_path)
        print(f"  ✅ Deleted {series_name}")

        summary_path = os.path.join(json_dir, "_summary.json")
        if os.path.exists(summary_path): _recalc_summary_json(summary_path, target_dir)


def main():
    parser = argparse.ArgumentParser(description="Time Series Preprocessing Pipeline")
    parser.add_argument("--input_path", type=str, help="Input CSV file or directory")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--freq", type=str, default=None, help="Time series frequency (e.g., '5T')")
    parser.add_argument("--missing_rate_thresh", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, default="./data_cleaned")

    # Cleanup arguments
    parser.add_argument("--remove_variate", type=str, default=None, help="Comma-separated variates to remove")
    parser.add_argument("--remove_series", type=str, default=None, help="Comma-separated series to remove")
    parser.add_argument("--target_dir", type=str, default=None, help="Target directory for cleanup")
    parser.add_argument("--dry_run", action="store_true", help="Dry run for cleanup")
    parser.add_argument("--remove_drop_marked", action="store_true", help="Remove all variates marked with [drop]")

    args = parser.parse_args()

    # --- Cleanup Modes ---
    if args.remove_drop_marked:
        if not args.target_dir:
            parser.error("--remove_drop_marked requires --target_dir")
        remove_drop_marked_variates(args.target_dir, dry_run=args.dry_run)
        return

    if args.remove_variate:
        if not args.target_dir:
            parser.error("--remove_variate requires --target_dir")
        for v in [v.strip() for v in args.remove_variate.split(",") if v.strip()]:
            remove_variate_from_dataset(args.target_dir, v, dry_run=args.dry_run)
        return

    if args.remove_series:
        if not args.target_dir:
            parser.error("--remove_series requires --target_dir")
        for s in [s.strip() for s in args.remove_series.split(",") if s.strip()]:
            remove_series_from_dataset(args.target_dir, s, dry_run=args.dry_run)
        return

    # --- Main Processing Mode ---
    if not args.input_path or not args.dataset:
        parser.error("--input_path and --dataset are required for preprocessing mode.")

    input_path = args.input_path
    if not os.path.exists(input_path):
        raise ValueError(f"Input path not found: {input_path}")

    if os.path.isfile(input_path):
        base_dir = os.path.dirname(input_path)
        csv_files = [os.path.basename(input_path)]
        print(f"[Preprocess] Single-file mode: {input_path}")
    else:
        base_dir = input_path
        csv_files = sorted([f for f in os.listdir(input_path) if f.endswith('.csv')])
        print(f"[Preprocess] Batch mode: {len(csv_files)} files found in {input_path}")

    if not csv_files:
        raise ValueError(f"No CSV files found in: {input_path}")

    # Infer frequency from the first file if not provided
    inferred_freq = args.freq
    if not inferred_freq:
        first_csv = os.path.join(base_dir, csv_files[0])
        df_first = pd.read_csv(first_csv)
        temp_pipeline = PreprocessPipeline(freq=None)
        df_temp = temp_pipeline._normalize_timestamp_column(df_first)
        df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
        try:
            inferred_freq = temp_pipeline._infer_frequency(df_temp.set_index("timestamp"))
            print(f"[Preprocess] Inferred freq: {inferred_freq}")
        except ValueError as e:
            raise ValueError(f"Failed to infer freq from first file. Specify --freq.\nReason: {e}")

    # Setup directories
    csv_output_dir = os.path.join(args.output_dir, "processed_csv", args.dataset, inferred_freq)
    json_output_dir = os.path.join(args.output_dir, "processed_summary", args.dataset, inferred_freq)
    os.makedirs(csv_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)

    # Stats tracking
    stats = {
        "success": 0, "fail": 0, "rows": 0, "cols": 0, "obs": 0,
        "lengths": [], "variates": {}, "corr_dups": {}, "successful_series": {}
    }

    # Process files
    for csv_file in csv_files:
        csv_path = os.path.join(base_dir, csv_file)
        csv_name = os.path.splitext(csv_file)[0]
        out_csv = os.path.join(csv_output_dir, csv_file)
        out_json = os.path.join(json_output_dir, f"{csv_name}.json")

        cleaned_df, result = process_single_csv(
            csv_path=csv_path, output_csv_path=out_csv, output_json_path=out_json,
            freq=inferred_freq, missing_rate_thresh=args.missing_rate_thresh,
        )

        if cleaned_df is None:
            stats["fail"] += 1
            continue

        stats["success"] += 1
        stats["rows"] += len(cleaned_df)
        stats["cols"] += len(cleaned_df.columns)
        stats["lengths"].append(len(cleaned_df))
        stats["obs"] += result.get("_meta", {}).get("num_observations", 0)
        stats["successful_series"][csv_file] = cleaned_df.copy()

        # Track variate stats
        for col, col_res in result.items():
            if col.startswith("_") or col == "multivariate":
                continue
            v_stats = stats["variates"].setdefault(col, {"total": 0, "kept": 0, "dropped_in": [], "kept_in": []})
            v_stats["total"] += 1
            if col_res.get("predictable", False):
                v_stats["kept"] += 1
                v_stats["kept_in"].append(csv_file)
            else:
                failed_checks = [c for c in col_res.get("checks", []) if "❌" in c or "Detected" in c]
                reason_str = " | ".join(failed_checks) if failed_checks else "Unknown reason"
                v_stats["dropped_in"].append((csv_file, reason_str))

    # Cross-series correlation check if inputs are multiple csv with UTS.
    cross_series_corr = {}
    if stats["successful_series"]:
        pipeline = PreprocessPipeline(freq=inferred_freq, missing_rate_thresh=args.missing_rate_thresh)
        cross_series_corr = pipeline._compute_cross_series_correlation(stats["successful_series"], corr_thresh=0.95)

    # Build and save summary JSON
    summary = {
        "dataset": args.dataset,
        "freq": inferred_freq,
        "num_series": len(csv_files),
        "success_count": stats["success"],
        "num_observations": stats["obs"],
        "max_series_length": max(stats["lengths"]) if stats["lengths"] else 0,
        "min_series_length": min(stats["lengths"]) if stats["lengths"] else 0,
        "avg_series_length": round(sum(stats["lengths"]) / len(stats["lengths"]), 2) if stats["lengths"] else 0,
        "variates": {},
        "cross_series_correlation": cross_series_corr.get("correlation_matrix", {}) if cross_series_corr else {}
    }

    for var, v_stats in stats["variates"].items():
        total = v_stats["total"]
        summary["variates"][var] = {
            "total": total, "kept": v_stats["kept"],
            "kept_ratio": round(v_stats["kept"] / total, 4) if total else 0,
            "dropped_series": v_stats["dropped_in"]
        }

    summary_json_path = os.path.join(json_output_dir, "_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)


# === Actionable Summary & Decision Prompts ===
    print("\n" + "="*60)
    print(f"[Finished] Dataset: {args.dataset} | Freq: {inferred_freq}")
    print(f"Success: {stats['success']}/{len(csv_files)} files processed")
    print(f"Total rows: {stats['rows']} | Total cols: {stats['cols']}")
    print(f"Summary saved to: {summary_json_path}")
    print("="*60)

    # 1. Categorize problematic variates
    dropped_variates = []
    partially_dropped = []
    for var, v_stats in stats["variates"].items():
        if v_stats["kept"] < v_stats["total"]:
            drop_ratio = 1 - (v_stats["kept"] / v_stats["total"])
            if drop_ratio >= 0.5:
                dropped_variates.append((var, v_stats["kept"], v_stats["total"], v_stats["dropped_in"]))
            else:
                partially_dropped.append((var, v_stats["kept"], v_stats["total"], v_stats["dropped_in"]))

    # 2. Identify fully dropped series (series where ALL variates failed)
    all_kept_series = set()
    for v_stats in stats["variates"].values():
        all_kept_series.update(v_stats["kept_in"])

    fully_dropped_series = []
    for csv_file in stats["successful_series"].keys():
        if csv_file not in all_kept_series:
            fully_dropped_series.append(csv_file)

    # 3. Print recommendations and commands
    if dropped_variates or partially_dropped or fully_dropped_series:
        print("\n⚠️  [Action Required] Problematic Data Detected")
        print("-" * 60)

        if fully_dropped_series:
            print("\n📌 All variates failed in these series:")
            for s in fully_dropped_series:
                print(f"   - {s}")
            series_str = ",".join(fully_dropped_series)
            print(f"\n   -> Command to remove these series:")
            print(f"      python -m timebench.preprocess --remove_series {series_str} --target_dir {csv_output_dir}")

        if dropped_variates:
            print("\n📌 Variates failed in MOST series (>= 50%):")
            vars_to_remove = []
            series_to_remove = set()
            for var, kept, total, dropped_in in dropped_variates:
                print(f"   - '{var}': Failed in {total-kept}/{total} series.")
                for s_file, reason in dropped_in:
                    print(f"     └─ {s_file}: {reason}")
                    series_to_remove.add(s_file)
                vars_to_remove.append(var)
            vars_str = ",".join(vars_to_remove)
            series_str = ",".join(sorted(series_to_remove))
            print(f"\n   -> Option A (Recommended): Remove these variates from the entire dataset")
            print(f"      python -m timebench.preprocess --remove_variate {vars_str} --target_dir {csv_output_dir}")
            if series_to_remove:
                print(f"   -> Option B: Remove the affected series to keep the variates")
                print(f"      python -m timebench.preprocess --remove_series {series_str} --target_dir {csv_output_dir}")

        if partially_dropped:
            print("\n📌 Variates failed in a FEW series (< 50%):")
            vars_to_remove = []
            series_to_remove = set()
            for var, kept, total, dropped_in in partially_dropped:
                print(f"   - '{var}': Failed in {total-kept}/{total} series.")
                for s_file, reason in dropped_in:
                    print(f"     └─ {s_file}: {reason}")
                    series_to_remove.add(s_file)
                vars_to_remove.append(var)

            vars_str = ",".join(vars_to_remove)
            if series_to_remove:
                series_str = ",".join(sorted(series_to_remove))
                print(f"\n   -> Option A (Recommended): Remove the affected series to keep the variate clean")
                print(f"      python -m timebench.preprocess --remove_series {series_str} --target_dir {csv_output_dir}")
                print(f"   -> Option B: Remove the variates entirely from all series")
                print(f"      python -m timebench.preprocess --remove_variate {vars_str} --target_dir {csv_output_dir}")

        print("\n💡 Tip: Append '--dry_run' to preview deletions without modifying files.")
        print("="*60)

if __name__ == "__main__":
    main()
