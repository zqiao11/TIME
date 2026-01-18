import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


# 全局变量：不同频率对应的最小长度要求
# 基本原则：高频数据需要更多数据点，低频数据相对少一些
FREQ_MIN_LENGTH = {
    # 秒级; 3 Days
    "S": 3600 * 24 * 3,
    # 分钟级; 1 Month  # TODO: 2周？
    "T": 60 * 24 * 28,
    "min": 60 * 24 * 28,
    # 小时级; 1 Month
    "H": 24 * 30,
    "h": 24 * 30,
    # 日级; 0.5 Year
    "D": 183 * 1,
    # 工作日级; 0.5 Year (约130个工作日)
    "B": 130,
    # 周级; 2 Year
    "W": 52 * 2,
    # 月级; 10 Years
    "M": 12 * 10,
    "MS": 12 * 10,
    "ME": 12 * 10,
    # 季度级; 10 Years
    "Q": 4 * 10,
    "QS": 4 * 10,
    "QE": 4 * 10,
    # 年级
    "Y": 20,
    "YS": 20,
    "YE": 20,
    "A": 20,
    # 默认值
    "default": 20,
}


class PreprocessPipeline:
    def __init__(
            self,
            freq: str | None = None,
            min_length: int | None = None,
            missing_rate_thresh: float = 0.2,
            corr_thresh: float = 0.95,
            auto_drop: bool = False
    ):
        """
        Args:
            freq: 时间序列频率。如果为 None，则自动推断；如果自动推断失败，需手动指定
            min_length: 最小长度要求。如果为 None，则根据频率自动设置
            missing_rate_thresh: 允许的最大缺失率
            corr_thresh: 相关性阈值，用于检测高相关变量
            auto_drop: 是否自动删除不符合要求的列。如果为 False（默认），则仅检查不删除，所有列都会保留在输出中
        """
        self._freq_override = freq  # 用户指定的频率（可选）
        self._min_length_override = min_length  # 用户指定的 min_length（可选）
        self.min_length = min_length if min_length is not None else FREQ_MIN_LENGTH["default"]
        self.missing_rate_thresh = missing_rate_thresh
        self.corr_thresh = corr_thresh
        self.auto_drop = auto_drop  # 是否自动删除不符合要求的列
        self.inferred_freq = None  # 推断出的频率

    def run(self, df: pd.DataFrame, output_path: str | None = None) -> tuple[pd.DataFrame, dict]:
        """
        运行预处理 Pipeline

        Args:
            df: 原始时间序列 DataFrame
            output_path: 可选，清洗后 CSV 的保存路径

        Returns:
            (cleaned_df, results):
            - cleaned_df: 清洗后的 DataFrame（时间戳为索引，列名带有特性标记）
            - results: 详细的检查结果字典
        """
        results = {}

        # Step 0: 规范化时间戳列（确保第一列为 timestamp）
        df = self._normalize_timestamp_column(df)

        # 将 timestamp 列转换为 datetime 并设置为索引
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Step 1: 推断频率并设置对应的 min_length
        if self._freq_override is not None:
            self.inferred_freq = self._freq_override
            print(f"[PreprocessPipeline] 使用用户指定的频率: {self.inferred_freq}")
        else:
            self.inferred_freq = self._infer_frequency(df)

        if self._min_length_override is None:
            self.min_length = self._get_min_length_for_freq(self.inferred_freq)

        results["_meta"] = {
            "inferred_freq": self.inferred_freq,
            "min_length": self.min_length
        }

        # --- Step A: 逐列单变量检查和清洗 ---
        cleaned_columns = {}  # 存储清洗后的列 {new_col_name: cleaned_series}
        dropped_columns = []  # 建议删除的列（当 auto_drop=False 时，这些列仍会保留但会被标记）
        recommended_drop_columns = []  # 建议删除的列（用于记录，即使不实际删除）

        for col in df.columns:
            ts = df[col]
            col_result = self._run_univariate(ts)
            results[col] = col_result

            # 确定新列名（添加特性标记后缀）
            new_col_name = col
            tags = []
            if col_result.get("has_spike_presence", False):
                tags.append("sp")
            if col_result.get("is_random_walk", False):
                tags.append("rw")
            if not col_result["predictable"]:
                # 如果不符合要求，添加 [drop] 标记（即使 auto_drop=False 也会标记）
                tags.append("drop")
            if tags:
                new_col_name = f"{col}[{','.join(tags)}]"

            # 根据 auto_drop 决定是否保留该列
            if self.auto_drop and not col_result["predictable"]:
                # 自动删除模式：不符合要求的列不保留
                dropped_columns.append(col)
                recommended_drop_columns.append(col)
            else:
                # 仅检查模式（默认）：所有列都保留，但会标记不符合要求的列
                if not col_result["predictable"]:
                    recommended_drop_columns.append(col)

                # 获取清洗后的时间序列
                cleaned_ts = col_result.get("cleaned_ts")
                if cleaned_ts is not None:
                    cleaned_columns[new_col_name] = cleaned_ts
                else:
                    # 如果没有清洗（无异常），使用原始序列（保留原始NaN）
                    cleaned_columns[new_col_name] = ts

        # 构建清洗后的 DataFrame
        cleaned_df = pd.DataFrame(cleaned_columns)

        # 更新 meta 信息
        results["_meta"]["n_rows"] = len(cleaned_df)
        results["_meta"]["n_cols"] = len(cleaned_df.columns)
        results["_meta"]["shape"] = list(cleaned_df.shape)
        results["_meta"]["original_columns"] = list(df.columns)
        results["_meta"]["kept_columns"] = list(cleaned_df.columns)
        results["_meta"]["dropped_columns"] = dropped_columns  # 实际被删除的列（仅在 auto_drop=True 时）
        results["_meta"]["recommended_drop_columns"] = recommended_drop_columns  # 建议删除的列（检查结果）
        results["_meta"]["auto_drop"] = self.auto_drop
        # 统计非NaN的observations数量
        results["_meta"]["num_observations"] = int(cleaned_df.notna().sum().sum())
        results["_meta"]["spike_presence_columns"] = [
            col for col in cleaned_df.columns if "[sp" in col
        ]
        results["_meta"]["random_walk_columns"] = [
            col for col in cleaned_df.columns if "[rw" in col or ",rw]" in col
        ]
        results["_meta"]["drop_marked_columns"] = [
            col for col in cleaned_df.columns if "[drop" in col or ",drop]" in col
        ]

        if self.auto_drop:
            print(f"[PreprocessPipeline] 模式: 自动删除 (auto_drop=True)")
            print(f"[PreprocessPipeline] 原始列数: {len(df.columns)}, "
                  f"保留列数: {len(cleaned_df.columns)}, "
                  f"丢弃列数: {len(dropped_columns)}")
        else:
            print(f"[PreprocessPipeline] 模式: 仅检查不删除 (auto_drop=False, 默认)")
            print(f"[PreprocessPipeline] 原始列数: {len(df.columns)}, "
                  f"输出列数: {len(cleaned_df.columns)} (全部保留)")
            if recommended_drop_columns:
                print(f"[PreprocessPipeline] ⚠️  建议删除的列 ({len(recommended_drop_columns)} 个): {recommended_drop_columns}")
                print(f"[PreprocessPipeline]   这些列已标记为 [drop]，请根据 summary 决定是否手动删除")

        if results["_meta"]["spike_presence_columns"]:
            print(f"[PreprocessPipeline] 具有 spike_presence [sp] 的列: {results['_meta']['spike_presence_columns']}")
        if results["_meta"]["random_walk_columns"]:
            print(f"[PreprocessPipeline] 具有 random_walk [rw] 的列: {results['_meta']['random_walk_columns']}")

        # --- Step B: 多变量检查（在清洗后的数据上）---
        if len(cleaned_df.columns) > 1:
            corr_matrix = cleaned_df.corr(method="pearson")

            corr_duplicates = []
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > self.corr_thresh:
                        corr_duplicates.append(
                            (corr_matrix.index[i], corr_matrix.columns[j],
                             round(corr_val, 4))
                        )

            results["multivariate"] = {
                "correlation_duplicates": corr_duplicates,
                "correlation_matrix": corr_matrix.to_dict()
            }

        # 保存清洗后的 CSV（可选）
        if output_path is not None:
            # 重置索引，将 timestamp 作为第一列保存
            output_df = cleaned_df.reset_index()
            output_df.to_csv(output_path, index=False)
            print(f"[PreprocessPipeline] 清洗后的数据已保存至: {output_path}")
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
        对单个变量进行检查和清洗

        Returns:
            dict: 包含以下字段：
                - predictable: 是否可预测（保留）
                - checks: 检查结果列表
                - cleaned_ts: 清洗后的时间序列（如果 predictable=True）
                - has_spike_presence: 是否有 spike_presence 特性 (标记 [sp])
                - is_random_walk: 是否为随机游走 (标记 [rw])
                - outlier_stats: 异常值统计信息
        """
        results = {
            "predictable": True,
            "checks": [],
            "cleaned_ts": None,
            "has_spike_presence": False,
            "is_random_walk": False
        }

        # Step 1: Data type check
        passed, val = self._check_dtype(ts)
        results["checks"].append(
            f"Data type check {'✅ Passed' if passed else '❌ Failed'} (dtype={val})"
        )
        if not passed:
            results["predictable"] = False
            return results

        # Step 2: Data integrity
        passed, val = self._check_length(ts)
        results["checks"].append(
            f"Length check {'✅ Passed' if passed else '❌ Failed'} (length={val})"
        )
        if not passed:
            results["predictable"] = False

        # 先检查并补全缺失的时间戳（补全后值为 NaN）
        ts, ts_passed, ts_info = self._check_timestamp(ts)
        results["checks"].append(
            f"Timestamp stability {'✅ Passed' if ts_passed else '❌ Failed'} ({ts_info})"
        )
        if not ts_passed:
            results["predictable"] = False

        # 再检查缺失率（包括原始缺失和补全时间戳产生的缺失）
        passed, val = self._check_missing(ts)
        results["checks"].append(
            f"Missing rate {'✅ Passed' if passed else '❌ Failed'} (missing_rate={val:.2f}%)"
        )
        if not passed:
            results["predictable"] = False

        # 保存原始序列（保留所有NaN）
        ts_original = ts.copy()

        # 为了进行需要完整数据的check（如异常值检测），临时填充NaN
        # 但最终保存时会保留原始NaN值
        ts_for_checks = ts.ffill().bfill().fillna(0)

        if not results["predictable"]:
            return results

        # Step 3: Signal existence
        passed, topk_dominance, entropy = self._check_constant(ts_for_checks)
        results["checks"].append(
            f"Constant series {'✅ Passed' if passed else '❌ Failed'} (topk_dominance={topk_dominance:.2f}%, entropy={entropy:.2f})"
        )
        if not passed:
            results["predictable"] = False
            return results

        # Step 4: White noise check (删除白噪声序列)
        is_not_white_noise, wn_pval = self._check_white_noise(ts_for_checks)
        results["checks"].append(
            f"White noise test {'✅ Passed' if is_not_white_noise else '❌ Detected'} (p={wn_pval})"
        )
        if not is_not_white_noise:
            results["predictable"] = False
            return results

        # Step 5: Random walk check (标记但不删除)
        is_stationary, rw_pval = self._check_random_walk(ts_for_checks)
        is_random_walk = not is_stationary
        results["is_random_walk"] = is_random_walk
        results["checks"].append(
            f"Random walk test {'✅ Stationary' if is_stationary else '⚠️ Random Walk [rw]'} (adf_p={rw_pval})"
        )
        # 注意：随机游走不删除，只标记

        # Step 6: Outlier check and cleaning
        # 对于异常值检测，使用填充后的数据进行检测（需要完整数据才能检测异常值）
        keep_variate, cleaned_ts_filled, has_spike_presence, outlier_stats = self._check_and_clean_outliers(ts_for_checks)

        # 将清洗后的序列映射回原始序列
        # 规则：原始NaN保持不变，异常值替换为前一个正常值（不是NaN）
        if cleaned_ts_filled is not None and keep_variate:
            # 从原始序列开始（保留所有原始NaN）
            cleaned_ts = ts_original.copy()
            # 只更新非NaN位置的值，使用清洗后的值（异常值已被替换为前一个正常值）
            # 这样：原始NaN保持NaN，异常值被替换为前一个正常值
            non_nan_mask = ~ts_original.isna()
            cleaned_ts[non_nan_mask] = cleaned_ts_filled[non_nan_mask]
        else:
            # 如果没有清洗或不符合要求，返回原始序列（保留NaN）
            cleaned_ts = ts_original if keep_variate else None

        if "error" in outlier_stats:
            results["checks"].append(
                f"Outlier check ❌ Failed (error: {outlier_stats['error']})"
            )
            results["predictable"] = False
            return results

        results["outlier_stats"] = outlier_stats

        if not keep_variate:
            results["checks"].append(
                f"Outlier check ❌ Dropped "
                f"(extreme_outlier={outlier_stats['extreme_outlier_ratio']:.2f}% > threshold)"
            )
            results["predictable"] = False
            return results

        # 保留该 variate
        action_str = outlier_stats.get("action", "unchanged")
        spike_flag = " [sp]" if has_spike_presence else ""
        rw_flag = " [rw]" if is_random_walk else ""
        results["checks"].append(
            f"Outlier check ✅ Passed ({action_str}{spike_flag}{rw_flag}) "
            f"(transient_spike={outlier_stats['transient_spike_ratio']:.2f}%, "
            f"extreme_outlier={outlier_stats['extreme_outlier_ratio']:.2f}%)"
        )

        results["cleaned_ts"] = cleaned_ts
        results["has_spike_presence"] = has_spike_presence

        return results

    # --- helper functions ---
    def _infer_frequency(self, df: pd.DataFrame) -> str:
        """
        自动推断时间序列的频率

        Returns:
            推断出的频率字符串，如 'H', 'D', 'M' 等

        Raises:
            ValueError: 如果无法推断频率，提示用户手动指定
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "无法推断频率：索引不是 DatetimeIndex。"
                "请在初始化时手动指定 freq 参数，例如: PreprocessPipeline(freq='H')"
            )

        freq = pd.infer_freq(df.index)
        print(f"[PreprocessPipeline] 推断的频率: {freq}")

        if freq is None:
            raise ValueError(
                "无法自动推断时间序列频率（pd.infer_freq 返回 None）。\n"
                "可能原因：时间间隔不规则、存在缺失时间点、数据量太少等。\n"
                "请在初始化时手动指定 freq 参数，例如: PreprocessPipeline(freq='H')\n"
                "常见频率: 'S'(秒), 'T'(分钟), 'H'(小时), 'D'(天), 'W'(周), 'M'(月), 'Q'(季度), 'Y'(年)"
            )

        return freq

    def _get_min_length_for_freq(self, freq: str) -> int:
        """
        根据频率获取对应的最小长度要求
        支持带数字前缀的频率，如 '10S', '15T', '2H' 等

        Args:
            freq: 频率字符串，如 'H', '10S', '15T' 等

        Returns:
            对应的最小长度（会根据数字前缀调整）
        """
        # 解析频率字符串，提取数字前缀和基础频率
        # 例如: '10S' -> (10, 'S'), 'H' -> (1, 'H'), '15min' -> (15, 'min')
        match = re.match(r'^(\d*)([A-Za-z]+)$', freq)
        if not match:
            print(f"[PreprocessPipeline] 无法解析频率 '{freq}'，使用默认 min_length")
            return FREQ_MIN_LENGTH["default"]

        multiplier_str, base_freq = match.groups()
        multiplier = int(multiplier_str) if multiplier_str else 1

        # 查找基础频率对应的 min_length
        base_min_length = None

        if base_freq in FREQ_MIN_LENGTH:
            base_min_length = FREQ_MIN_LENGTH[base_freq]
        else:
            # 尝试大小写不敏感匹配
            for key in FREQ_MIN_LENGTH:
                if key.upper() == base_freq.upper():
                    base_min_length = FREQ_MIN_LENGTH[key]
                    break

        if base_min_length is None:
            print(f"[PreprocessPipeline] 未知的基础频率 '{base_freq}'，使用默认 min_length")
            return FREQ_MIN_LENGTH["default"]

        # 根据数字前缀调整 min_length
        # 例如: 'S' 的 min_length 是 3600*24*30，那么 '10S' 的 min_length 是 3600*24*30 / 10
        adjusted_min_length = max(1, base_min_length // multiplier)

        if multiplier > 1:
            print(f"[PreprocessPipeline] 频率 '{freq}' = {multiplier} x '{base_freq}', "
                  f"min_length: {base_min_length} / {multiplier} = {adjusted_min_length}")

        return adjusted_min_length

    def _normalize_timestamp_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        规范化时间戳列：
        1. 检查第一列是否为时间戳
        2. 如果不是，寻找名字包含 'time' 或 'date' 的第一个列
        3. 将其重命名为 'timestamp' 并移至第一列
        """
        df = df.copy()
        first_col = df.columns[0]

        # 检查第一列是否已经是时间戳列
        if self._is_timestamp_column(first_col, df[first_col]):
            # 第一列已是时间戳，只需重命名为 timestamp
            if first_col != "timestamp":
                df = df.rename(columns={first_col: "timestamp"})
            return df

        # 第一列不是时间戳，寻找时间戳列
        time_col = self._find_timestamp_column(df)
        if time_col is None:
            raise ValueError("未找到时间戳列：没有列名包含 'time' 或 'date'")

        # 重命名并移至第一列
        df = df.rename(columns={time_col: "timestamp"})
        cols = df.columns.tolist()
        cols.remove("timestamp")
        cols = ["timestamp"] + cols
        df = df[cols]

        return df

    def _is_timestamp_column(self, col_name: str, col_data: pd.Series) -> bool:
        """
        判断一列是否为时间戳列：
        - 列名包含 'time' 或 'date'（不区分大小写）
        - 或者数据类型为 datetime
        """
        name_match = "time" in col_name.lower() or "date" in col_name.lower()
        dtype_match = pd.api.types.is_datetime64_any_dtype(col_data)
        return name_match or dtype_match

    def _find_timestamp_column(self, df: pd.DataFrame) -> str | None:
        """
        寻找第一个名字包含 'time' 或 'date' 的列
        """
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                return col
        return None

    def _check_dtype(self, ts):
        return np.issubdtype(ts.dtype, np.number), ts.dtype

    def _check_length(self, ts):
        return len(ts) >= self.min_length, len(ts)

    def _check_missing(self, ts):
        missing_rate = ts.isna().mean()
        return missing_rate <= self.missing_rate_thresh, missing_rate * 100

    def _check_timestamp(self, ts: pd.Series) -> tuple[pd.Series, bool, str]:
        """
        检查时间戳的完整性，并补全缺失的时间戳（值设为 NaN）

        Returns:
            tuple: (补全后的序列, 是否通过检查, 检查信息)
        """
        # 检查是否为 DatetimeIndex
        if not isinstance(ts.index, pd.DatetimeIndex):
            return ts, False, "not_datetime_index"

        # 检查是否单调递增
        if not ts.index.is_monotonic_increasing:
            return ts, False, "not_monotonic"

        # 使用 inferred_freq 补全缺失的时间戳
        if self.inferred_freq is not None and len(ts) > 1:
            # 获取用于 date_range 的频率字符串
            freq_for_range = self._get_freq_for_date_range(ts.index)

            # 创建完整的时间范围
            full_range = pd.date_range(
                start=ts.index.min(),
                end=ts.index.max(),
                freq=freq_for_range
            )

            # 检查是否有缺失的时间戳
            missing_count = len(full_range) - len(ts)

            if missing_count > 0:
                # 使用 reindex 补全缺失的时间戳，缺失值自动设为 NaN
                ts = ts.reindex(full_range)
                ts.index.name = 'timestamp'
                print(f"[PreprocessPipeline] 补全了 {missing_count} 个缺失的时间戳")
                return ts, True, f"filled_{missing_count}_missing_timestamps"

        return ts, True, "ok"

    def _get_freq_for_date_range(self, index: pd.DatetimeIndex) -> str:
        """
        根据数据的实际日期获取正确的 date_range 频率字符串

        主要处理周频率的情况：pandas 的 'W' 默认是周日，
        但数据可能使用其他日期作为周起始日（如 ISO 标准的周一）

        Returns:
            str: 用于 pd.date_range 的频率字符串
        """
        freq = self.inferred_freq

        # 处理周频率：检测实际的周起始日
        if freq and freq.upper().startswith('W'):
            if len(index) > 0:
                # 获取数据中最常见的星期几
                weekdays = index.dayofweek  # 0=Monday, 6=Sunday
                most_common_weekday = weekdays.value_counts().idxmax()

                # 映射到 pandas 的周频率格式
                weekday_map = {
                    0: 'W-MON',  # Monday
                    1: 'W-TUE',  # Tuesday
                    2: 'W-WED',  # Wednesday
                    3: 'W-THU',  # Thursday
                    4: 'W-FRI',  # Friday
                    5: 'W-SAT',  # Saturday
                    6: 'W-SUN',  # Sunday (default 'W')
                }
                return weekday_map.get(most_common_weekday, 'W')

        return freq

    def _check_constant(self, ts):
        topk = 5
        counts = ts.value_counts(normalize=True)
        topk_dominance = counts.iloc[:topk].sum()

        probs = counts.values
        # 处理常数序列的情况（len(probs) == 1 时，熵为 0）
        if len(probs) == 1:
            entropy = 0.0
        else:
            entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs))

        return topk_dominance < 0.5 and entropy > 0.1, topk_dominance * 100, round(entropy, 4)

    def _check_white_noise(self, ts: pd.Series) -> tuple[bool, float | None]:
        """
        使用 Ljung-Box 检验判断序列是否为白噪声

        Ljung-Box 检验:
        - H0: 序列无自相关（是白噪声）
        - H1: 序列存在自相关（不是白噪声）

        判断逻辑:
        - p ≤ 0.05: 拒绝 H0，存在显著自相关 → 不是白噪声 → 通过
        - p > 0.05: 无法拒绝 H0，可能是白噪声 → 失败（应删除）

        Note:
            acorr_ljungbox 在样本量超过 ~10000 时性能急剧下降（O(n²) 复杂度），
            因此对于长序列，我们使用后 10000 个样本进行检验，这在统计上仍然有效。

        Returns:
            (is_not_white_noise, p_value)
        """
        try:
            ts_clean = ts.dropna()
            # 限制样本量以避免 acorr_ljungbox 的性能问题
            # 10000 样本足以进行可靠的白噪声检验
            max_samples = 10000
            if len(ts_clean) > max_samples:
                ts_clean = ts_clean.iloc[-max_samples:]
            # 使用多个 lag 进行检验，取最小 p 值（更严格）
            result = acorr_ljungbox(ts_clean, lags=[10, 20], return_df=True)
            pval = result['lb_pvalue'].min()
            is_not_white_noise = pval <= 0.05
            return is_not_white_noise, round(pval, 4)
        except Exception:
            # 检验失败时默认通过（保守策略）
            return True, None

    def _check_random_walk(self, ts: pd.Series) -> tuple[bool, float | None]:
        """
        使用 ADF (Augmented Dickey-Fuller) 检验判断序列是否为随机游走

        ADF 检验:
        - H0: 序列有单位根（非平稳，如随机游走）
        - H1: 序列平稳（无单位根）

        使用 regression='ct' (constant + trend):
        - 允许序列包含确定性趋势，避免把趋势序列误判为随机游走
        - 对于 y_t = a + bt + ε_t（趋势平稳），ADF 会正确拒绝 H0

        判断逻辑:
        - p ≤ 0.05: 拒绝 H0，序列平稳 → 不是随机游走 → (True, p)
        - p > 0.05: 无法拒绝 H0，可能是随机游走 → (False, p)

        Note:
            对于长序列，使用后 10000 个样本进行检验以提高效率，
            这在统计上仍然有效且结果一致。

        Returns:
            (is_stationary, p_value): is_stationary=False 表示是随机游走
        """
        try:
            ts_clean = ts.dropna()
            # 限制样本量以提高效率
            max_samples = 10000
            if len(ts_clean) > max_samples:
                ts_clean = ts_clean.iloc[-max_samples:]
            # regression='ct': 包含常数项和趋势项
            # 这样可以区分：
            # - 趋势平稳序列（有确定性趋势但平稳）→ p < 0.05
            # - 随机游走（有单位根）→ p > 0.05
            result = adfuller(ts_clean, autolag="AIC", regression='ct')
            pval = result[1]
            is_stationary = pval <= 0.05
            return is_stationary, round(pval, 4)
        except Exception:
            # 检验失败时默认通过（保守策略）
            return True, None

    def _check_and_clean_outliers(
            self,
            ts: pd.Series,
            window_size: int | None = None,
            k_transient: float = 3.0,
            k_extreme: float = 9.0,
            transient_thresh: float = 0.05,
            extreme_thresh: float = 0.05
    ) -> tuple[bool, pd.Series | None, bool, dict]:
        """
        使用滑动窗口 IQR 方法检测和清洗异常值，区分两类：

        1. Transient Spikes (瞬态尖峰): k_transient < deviation < k_extreme
           - 偏离局部分布但属于正常时间特性的点（如周期性峰值、事件驱动的突变）
           - 如果比例 > transient_thresh，标记该 variate 具有 "spike_presence" 特性

        2. Extreme Outliers (极端异常): deviation >= k_extreme
           - 明显错误的数据点，视为数据错误
           - 如果比例 > extreme_thresh，直接丢弃该 variate
           - 如果比例 <= extreme_thresh，用前一个正常值替换这些错误点

        基于论文描述的 robust local IQR filter:
        - IQR = Q_0.75 - Q_0.25
        - 判断范围: [m_w - k * IQR_w, m_w + k * IQR_w]

        阈值设计原则（保守策略，宁缺勿滥）：
        - k_transient=5.0: 对应约 6.75σ，正态分布下概率 < 0.00001%
          纯正弦波的 max_deviation/IQR ≈ 0.71，远低于此阈值
        - k_extreme=10.0: 对应约 13.5σ，只有真正的数据错误才会触发
        - transient_thresh=0.02: 至少 2% 的点是 spike 才标记 spike_presence
        - extreme_thresh=0.05: 超过 5% 极端异常则认为数据质量太差，丢弃

        Args:
            ts: 时间序列
            window_size: 滑动窗口大小，默认为序列长度的 10% (最小 20)
            k_transient: transient spike 的 IQR 倍数下界阈值 (默认 5.0)
            k_extreme: extreme outlier 的 IQR 倍数阈值 (默认 10.0)
            transient_thresh: spike_presence 标记的比例阈值 (默认 0.02)
            extreme_thresh: extreme outlier 比例上限，超过则丢弃 variate (默认 0.05)

        Returns:
            (keep_variate, cleaned_ts, has_spike_presence, stats):
            - keep_variate: 是否保留该 variate
            - cleaned_ts: 清洗后的时间序列（如果 keep_variate=False 则为 None）
            - has_spike_presence: 是否具有 spike_presence 特性
            - stats: 详细统计信息
        """
        if ts.std() == 0 or ts.isna().all():
            return False, None, False, {"error": "constant or all NaN"}

        n = len(ts)
        if window_size is None:
            window_size = max(20, n // 10)

        # 计算滑动窗口的 median 和 IQR
        rolling_median = ts.rolling(window=window_size, center=True, min_periods=1).median()
        rolling_q25 = ts.rolling(window=window_size, center=True, min_periods=1).quantile(0.25)
        rolling_q75 = ts.rolling(window=window_size, center=True, min_periods=1).quantile(0.75)
        rolling_iqr = rolling_q75 - rolling_q25

        # 处理 IQR 为 0 的情况（用全局 IQR 替代）
        global_iqr = ts.quantile(0.75) - ts.quantile(0.25)
        if global_iqr == 0:
            global_iqr = ts.std() * 1.35  # 近似转换: IQR ≈ 1.35 * std (正态分布)
        rolling_iqr = rolling_iqr.replace(0, global_iqr)

        # 计算每个点与局部中心的偏离程度 (以 IQR 为单位)
        deviation = np.abs(ts - rolling_median) / rolling_iqr

        # 分类异常点
        # Transient spike: k_transient < deviation < k_extreme (正常的尖峰)
        # Extreme outlier: deviation >= k_extreme (数据错误)
        is_transient = (deviation > k_transient) & (deviation < k_extreme)
        is_extreme = deviation >= k_extreme

        transient_ratio = is_transient.mean()
        extreme_ratio = is_extreme.mean()

        stats = {
            "transient_spike_ratio": round(transient_ratio * 100, 2),
            "extreme_outlier_ratio": round(extreme_ratio * 100, 2),
            "transient_spike_count": int(is_transient.sum()),
            "extreme_outlier_count": int(is_extreme.sum()),
            "k_transient": k_transient,
            "k_extreme": k_extreme,
            "window_size": window_size
        }

        # 判断 1: 是否保留 variate（基于 extreme_ratio）
        if extreme_ratio > extreme_thresh:
            stats["action"] = "dropped"
            stats["reason"] = f"extreme_outlier_ratio ({extreme_ratio*100:.2f}%) > threshold ({extreme_thresh*100:.2f}%)"
            return False, None, False, stats

        # 判断 2: 是否标记 spike_presence（基于 transient_ratio）
        has_spike_presence = transient_ratio > transient_thresh

        # 清洗极端异常值：用前一个正常值替换
        cleaned_ts = ts.copy()
        if is_extreme.any():
            # 将极端异常点设为 NaN，然后用前向填充
            cleaned_ts[is_extreme] = np.nan
            cleaned_ts = cleaned_ts.ffill()
            # 如果开头就是异常值，用后向填充补齐
            cleaned_ts = cleaned_ts.bfill()
            stats["action"] = "cleaned"
            stats["cleaned_count"] = int(is_extreme.sum())
        else:
            stats["action"] = "unchanged"
            stats["cleaned_count"] = 0

        if has_spike_presence:
            stats["spike_presence"] = True
            stats["spike_reason"] = f"transient_spike_ratio ({transient_ratio*100:.2f}%) > threshold ({transient_thresh*100:.2f}%)"

        return True, cleaned_ts, has_spike_presence, stats


def convert_to_serializable(obj):
    """递归转换 numpy 类型为 Python 原生类型，用于 JSON 序列化"""
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
        return None  # 排除 Series 对象
    else:
        return obj


def save_result_to_json(result: dict, json_path: str) -> None:
    """将预处理结果保存为 JSON 文件"""
    result_serializable = {}
    for key, value in result.items():
        # 排除 multivariate 字段
        if key == "multivariate":
            continue
        if key == "_meta":
            result_serializable[key] = convert_to_serializable(value)
        elif isinstance(value, dict):
            # 排除 cleaned_ts（Series 对象）
            result_serializable[key] = convert_to_serializable({
                k: v for k, v in value.items()
                if k != "cleaned_ts"
            })
        else:
            result_serializable[key] = convert_to_serializable(value)

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_serializable, f, indent=4, ensure_ascii=False)
    print(f"[PreprocessPipeline] 详细结果已保存至: {json_path}")


def print_result_summary(result: dict, source_name: str = "") -> None:
    """打印预处理结果摘要"""
    header = f"=== {source_name} 各列检查结果摘要 ===" if source_name else "=== 各列检查结果摘要 ==="
    print(f"\n{header}")

    # 获取 auto_drop 模式
    auto_drop = result.get("_meta", {}).get("auto_drop", False)

    for col, col_result in result.items():
        if col.startswith("_") or col == "multivariate":
            continue
        predictable = col_result.get("predictable", False)
        if auto_drop:
            status = "✅ 保留" if predictable else "❌ 丢弃"
        else:
            status = "✅ 通过" if predictable else "⚠️  建议删除"
        tags = []
        if col_result.get("has_spike_presence", False):
            tags.append("sp")
        if col_result.get("is_random_walk", False):
            tags.append("rw")
        tag_str = f" [{','.join(tags)}]" if tags else ""
        print(f"  {col}: {status}{tag_str}")

        # 打印 check 信息
        checks = col_result.get("checks", [])
        if checks:
            if predictable:
                # 通过的列：只打印最后一个 check（通常是 outlier check）
                print(f"    └─ {checks[-1]}")
            else:
                # 未通过的列：打印所有失败的 checks（带 ❌ 或 Detected 的）
                failed_checks = [c for c in checks if "❌" in c or "Detected" in c]
                if failed_checks:
                    for i, check in enumerate(failed_checks):
                        prefix = "└─" if i == len(failed_checks) - 1 else "├─"
                        print(f"    {prefix} {check}")
                else:
                    # 如果没找到明确失败的，打印最后一个 check
                    print(f"    └─ {checks[-1]}")


def process_single_csv(
    csv_path: str,
    output_csv_path: str,
    output_json_path: str,
    freq: str | None = None,
    missing_rate_thresh: float = 0.2,
    auto_drop: bool = False,
    verbose: bool = True
) -> tuple[pd.DataFrame | None, dict | None]:
    """
    处理单个 CSV 文件

    Args:
        csv_path: 输入 CSV 文件路径
        output_csv_path: 输出清洗后 CSV 的路径
        output_json_path: 输出 JSON 结果的路径
        freq: 时间序列频率（可选）
        missing_rate_thresh: 允许的最大缺失率
        auto_drop: 是否自动删除不符合要求的列（默认 False，仅检查不删除）
        verbose: 是否打印详细信息

    Returns:
        (cleaned_df, result): 清洗后的 DataFrame 和结果字典，如果处理失败则返回 (None, None)
    """
    if verbose:
        print("\n" + "=" * 60)
        print(f"[PreprocessPipeline] 处理文件: {csv_path}")
        print("=" * 60)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[PreprocessPipeline] ❌ 读取文件失败: {e}")
        return None, None

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # 调用 PreprocessPipeline
    pipeline = PreprocessPipeline(
        freq=freq,
        missing_rate_thresh=missing_rate_thresh,
        auto_drop=auto_drop
    )

    try:
        cleaned_df, result = pipeline.run(df, output_path=output_csv_path)
    except Exception as e:
        print(f"[PreprocessPipeline] ❌ 处理失败: {e}")
        return None, None

    if verbose:
        print(f"\n推断的频率: {pipeline.inferred_freq}")
        print(f"对应的 min_length: {pipeline.min_length}")
        print(f"清洗后的 DataFrame shape: {cleaned_df.shape}")

    # 保存 JSON 结果
    save_result_to_json(result, output_json_path)

    if verbose:
        print_result_summary(result, os.path.basename(csv_path))

    return cleaned_df, result




def remove_variate_from_dataset(target_dir: str, variate_name: str, dry_run: bool = False) -> None:
    """
    从数据集的所有 CSV 文件中移除指定的 variate（列），并更新对应的 JSON 文件

    Args:
        target_dir: 包含处理后 CSV 文件的目录
        variate_name: 要移除的 variate 名称（支持带标记如 'TEMP[rw]' 或不带标记 'TEMP'）
        dry_run: 如果为 True，只显示将要执行的操作而不实际修改文件
    """
    if not os.path.isdir(target_dir):
        raise ValueError(f"目录不存在: {target_dir}")

    csv_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.csv')])
    if not csv_files:
        print(f"[remove_variate] 目录中没有 CSV 文件: {target_dir}")
        return

    # 对应的 JSON 目录
    json_dir = target_dir.replace("processed_csv", "processed_summary")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}从数据集中移除 variate: {variate_name}")
    print(f"目标 CSV 目录: {target_dir}")
    print(f"目标 JSON 目录: {json_dir}")
    print("-" * 60)

    modified_count = 0
    for csv_file in csv_files:
        csv_path = os.path.join(target_dir, csv_file)
        df = pd.read_csv(csv_path)

        # 查找匹配的列（支持带标记或不带标记的名称）
        cols_to_drop = []
        for col in df.columns:
            # 精确匹配或者匹配基础名称（去掉 [xx] 标记后）
            base_col = re.sub(r'\[.*?\]', '', col)
            if col == variate_name or base_col == variate_name:
                cols_to_drop.append(col)

        if cols_to_drop:
            print(f"  {csv_file}: 移除列 {cols_to_drop}")
            if not dry_run:
                df = df.drop(columns=cols_to_drop)
                df.to_csv(csv_path, index=False)

                # 更新对应的 JSON 文件
                json_name = csv_file.replace(".csv", ".json")
                json_path = os.path.join(json_dir, json_name)
                if os.path.exists(json_path):
                    _update_json_remove_variate(json_path, variate_name, csv_path)

            modified_count += 1
        else:
            print(f"  {csv_file}: 未找到匹配的列")

    # 更新 _summary.json
    summary_json_path = os.path.join(json_dir, "_summary.json")
    if os.path.exists(summary_json_path):
        print(f"\n更新汇总文件: {summary_json_path}")
        if not dry_run:
            _update_summary_json_remove_variate(summary_json_path, variate_name, target_dir)

    print("-" * 60)
    action = "将修改" if dry_run else "已修改"
    print(f"{action} {modified_count}/{len(csv_files)} 个 CSV 文件")
    if dry_run:
        print("提示: 移除 --dry_run 参数以实际执行修改")


def _recalculate_num_observations(json_path: str, csv_path: str) -> None:
    """重新计算并更新 JSON 文件中的 num_observations"""
    try:
        if not os.path.exists(csv_path):
            return
        df = pd.read_csv(csv_path)
        # 跳过timestamp列（如果存在）
        if "timestamp" in df.columns:
            df = df.drop(columns=["timestamp"])
        num_obs = int(df.notna().sum().sum())

        with open(json_path, "r") as f:
            data = json.load(f)

        if "_meta" in data:
            data["_meta"]["num_observations"] = num_obs

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"    ⚠️ 重新计算 num_observations 失败 ({json_path}): {e}")


def _update_json_remove_variate(json_path: str, variate_name: str, csv_path: str | None = None) -> None:
    """更新单个 JSON 文件，移除指定 variate 的记录"""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # 找到并移除匹配的 variate
        keys_to_remove = []
        for key in data.keys():
            if key.startswith("_") or key == "multivariate":
                continue
            base_key = re.sub(r'\[.*?\]', '', key)
            if key == variate_name or base_key == variate_name:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del data[key]

        # 更新 _meta 信息
        if "_meta" in data:
            meta = data["_meta"]
            # 更新 kept_columns
            if "kept_columns" in meta:
                meta["kept_columns"] = [c for c in meta["kept_columns"]
                                        if c != variate_name and re.sub(r'\[.*?\]', '', c) != variate_name]
            # 更新 dropped_columns
            if "dropped_columns" in meta and variate_name not in meta["dropped_columns"]:
                meta["dropped_columns"].append(variate_name)
            # 更新 n_cols
            if "kept_columns" in meta:
                meta["n_cols"] = len(meta["kept_columns"])
            # 更新 shape
            if "shape" in meta and "n_cols" in meta:
                meta["shape"][1] = meta["n_cols"]
            # 更新 spike_presence_columns 和 random_walk_columns
            for list_key in ["spike_presence_columns", "random_walk_columns"]:
                if list_key in meta:
                    meta[list_key] = [c for c in meta[list_key]
                                      if re.sub(r'\[.*?\]', '', c) != variate_name]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # 重新计算 num_observations（如果提供了 CSV 路径）
        if csv_path is not None:
            _recalculate_num_observations(json_path, csv_path)

    except Exception as e:
        print(f"    ⚠️ 更新 JSON 失败 ({json_path}): {e}")


def _update_summary_json_remove_variate(summary_json_path: str, variate_name: str, csv_dir: str | None = None) -> None:
    """更新 _summary.json，移除指定 variate，并重新计算数据集级别的统计信息"""
    try:
        with open(summary_json_path, "r") as f:
            data = json.load(f)

        if "variates" in data:
            # 找到并移除匹配的 variate
            keys_to_remove = []
            for key in data["variates"].keys():
                base_key = re.sub(r'\[.*?\]', '', key)
                if key == variate_name or base_key == variate_name:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del data["variates"][key]
                print(f"  已从汇总中移除: {key}")

        # 如果提供了 csv_dir，重新计算数据集级别的统计信息
        if csv_dir is not None:
            _update_summary_json_remove_series(summary_json_path, csv_dir)
        else:
            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

    except Exception as e:
        print(f"  ⚠️ 更新汇总 JSON 失败: {e}")


def remove_series_from_dataset(target_dir: str, series_name: str, dry_run: bool = False) -> None:
    """
    从数据集中移除指定的 series（CSV 文件），并更新对应的 JSON 文件

    Args:
        target_dir: 包含处理后 CSV 文件的目录
        series_name: 要移除的 series 文件名（如 'item_0.csv'）
        dry_run: 如果为 True，只显示将要执行的操作而不实际删除文件
    """
    if not os.path.isdir(target_dir):
        raise ValueError(f"目录不存在: {target_dir}")

    # 确保文件名有 .csv 后缀
    if not series_name.endswith('.csv'):
        series_name = series_name + '.csv'

    csv_path = os.path.join(target_dir, series_name)
    json_dir = target_dir.replace("processed_csv", "processed_summary")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}从数据集中移除 series: {series_name}")
    print(f"目标 CSV 目录: {target_dir}")
    print(f"目标 JSON 目录: {json_dir}")
    print("-" * 60)

    if not os.path.exists(csv_path):
        print(f"  ❌ 文件不存在: {csv_path}")
        return

    if dry_run:
        print(f"  将删除 CSV: {csv_path}")
        json_name = series_name.replace(".csv", ".json")
        json_path = os.path.join(json_dir, json_name)
        if os.path.exists(json_path):
            print(f"  将删除 JSON: {json_path}")
        summary_json_path = os.path.join(json_dir, "_variate_summary.json")
        if os.path.exists(summary_json_path):
            print(f"  将更新汇总: {summary_json_path}")
        print("-" * 60)
        print("提示: 移除 --dry_run 参数以实际执行删除")
    else:
        # 删除 CSV 文件
        os.remove(csv_path)
        print(f"  ✅ 已删除 CSV: {csv_path}")

        # 删除对应的 JSON 文件（如果存在）
        json_name = series_name.replace(".csv", ".json")
        json_path = os.path.join(json_dir, json_name)
        if os.path.exists(json_path):
            os.remove(json_path)
            print(f"  ✅ 已删除 JSON: {json_path}")

        # 更新 _summary.json
        summary_json_path = os.path.join(json_dir, "_summary.json")
        if os.path.exists(summary_json_path):
            _update_summary_json_remove_series(summary_json_path, target_dir)
            print(f"  ✅ 已更新汇总: {summary_json_path}")

        print("-" * 60)
        print("删除完成")


def _update_summary_json_remove_series(summary_json_path: str, csv_dir: str) -> None:
    """
    重新计算 _summary.json
    基于当前存在的 CSV 文件重新统计各 variate 的信息
    """
    try:
        # 读取现有汇总
        with open(summary_json_path, "r") as f:
            data = json.load(f)

        # 获取当前存在的 CSV 文件
        csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])

        # 重新统计 variates
        variate_stats = {}
        # 重新统计 correlation_duplicates
        corr_dup_stats = {}
        json_dir = os.path.dirname(summary_json_path)

        # 数据集级别的统计
        series_lengths = []
        total_observations = 0

        # 辅助函数：去掉变量名的后缀标记（如 [rw], [sp]）
        def strip_var_suffix(var_name):
            return re.sub(r'\[.*?\]', '', var_name)

        for csv_file in csv_files:
            json_name = csv_file.replace(".csv", ".json")
            json_path = os.path.join(json_dir, json_name)

            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    series_data = json.load(f)

                # 收集数据集级别的统计信息
                meta = series_data.get("_meta", {})
                series_length = meta.get("n_rows", 0)
                num_obs = meta.get("num_observations", 0)
                if series_length > 0:
                    series_lengths.append(series_length)
                total_observations += num_obs

                # 统计 variates
                for col, col_result in series_data.items():
                    if col.startswith("_") or col == "multivariate":
                        continue
                    if col not in variate_stats:
                        variate_stats[col] = {"total": 0, "kept": 0, "rw": 0, "sp": 0}
                    variate_stats[col]["total"] += 1
                    if col_result.get("predictable", False):
                        variate_stats[col]["kept"] += 1
                    if col_result.get("is_random_walk", False):
                        variate_stats[col]["rw"] += 1
                    if col_result.get("has_spike_presence", False):
                        variate_stats[col]["sp"] += 1

                # 统计 correlation_duplicates
                if "multivariate" in series_data:
                    corr_matrix = series_data["multivariate"].get("correlation_matrix", {})
                    corr_duplicates = series_data["multivariate"].get("correlation_duplicates", [])

                    # 记录哪些 pair 在此 series 中是高相关的
                    high_corr_pairs_in_series = set()
                    for dup in corr_duplicates:
                        var1, var2, corr_val = dup
                        base_var1, base_var2 = strip_var_suffix(var1), strip_var_suffix(var2)
                        pair = tuple(sorted([base_var1, base_var2]))

                        if pair not in corr_dup_stats:
                            corr_dup_stats[pair] = {
                                "high_count": 0, "high_series": [], "high_corr_values": [],
                                "all_corr_values": [], "all_series": []
                            }

                        if pair not in high_corr_pairs_in_series:
                            high_corr_pairs_in_series.add(pair)
                            corr_dup_stats[pair]["high_count"] += 1
                            corr_dup_stats[pair]["high_series"].append(csv_file)
                            corr_dup_stats[pair]["high_corr_values"].append(corr_val)

                    # 从完整相关矩阵中提取所有 pair 的相关系数
                    if corr_matrix:
                        vars_list = list(corr_matrix.keys())
                        for i, var1 in enumerate(vars_list):
                            for var2 in vars_list[i+1:]:
                                base_var1, base_var2 = strip_var_suffix(var1), strip_var_suffix(var2)
                                pair = tuple(sorted([base_var1, base_var2]))
                                corr_val = corr_matrix.get(var1, {}).get(var2)
                                if corr_val is not None:
                                    if pair not in corr_dup_stats:
                                        corr_dup_stats[pair] = {
                                            "high_count": 0, "high_series": [], "high_corr_values": [],
                                            "all_corr_values": [], "all_series": []
                                        }
                                    if csv_file not in corr_dup_stats[pair]["all_series"]:
                                        corr_dup_stats[pair]["all_corr_values"].append(corr_val)
                                        corr_dup_stats[pair]["all_series"].append(csv_file)

        # 更新 variates 数据
        data["num_series"] = len(csv_files)
        data["success_count"] = len(csv_files)
        data["variates"] = {}

        # 更新数据集级别的统计信息（仅当有多个series时）
        if len(csv_files) > 1 and series_lengths:
            data["num_observations"] = total_observations
            data["max_series_length"] = max(series_lengths)
            data["min_series_length"] = min(series_lengths)
            data["avg_series_length"] = round(sum(series_lengths) / len(series_lengths), 2)
        elif len(csv_files) <= 1:
            # 如果只有一个或没有series，删除这些字段
            data.pop("num_observations", None)
            data.pop("max_series_length", None)
            data.pop("min_series_length", None)
            data.pop("avg_series_length", None)

        for variate, stats in variate_stats.items():
            total = stats["total"]
            data["variates"][variate] = {
                "total": total,
                "kept": stats["kept"],
                "rw": stats["rw"],
                "sp": stats["sp"],
                "kept_ratio": round(stats["kept"] / total, 4) if total > 0 else 0,
                "rw_ratio": round(stats["rw"] / total, 4) if total > 0 else 0,
                "sp_ratio": round(stats["sp"] / total, 4) if total > 0 else 0
            }

        # 注意：不再保存 correlation_duplicates 字段，但保留计算用于打印
        # 如果存在 correlation_duplicates 字段，删除它
        if "correlation_duplicates" in data:
            del data["correlation_duplicates"]

        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    except Exception as e:
        print(f"  ⚠️ 更新汇总 JSON 失败: {e}")


def remove_drop_marked_variates(target_dir: str, dry_run: bool = False) -> None:
    """
    从数据集的所有 CSV 文件中移除所有带有 [drop] 标记的 variate（列），并更新对应的 JSON 文件

    Args:
        target_dir: 包含处理后 CSV 文件的目录
        dry_run: 如果为 True，只显示将要执行的操作而不实际修改文件
    """
    if not os.path.isdir(target_dir):
        raise ValueError(f"目录不存在: {target_dir}")

    csv_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.csv')])
    if not csv_files:
        print(f"[remove_drop_marked_variates] 目录中没有 CSV 文件: {target_dir}")
        return

    # 对应的 JSON 目录
    json_dir = target_dir.replace("processed_csv", "processed_summary")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}从数据集中移除所有带 [drop] 标记的 variate")
    print(f"目标 CSV 目录: {target_dir}")
    print(f"目标 JSON 目录: {json_dir}")
    print("-" * 60)

    all_dropped_variates = set()  # 记录所有被删除的 variate（基础名称）
    modified_count = 0

    for csv_file in csv_files:
        csv_path = os.path.join(target_dir, csv_file)
        df = pd.read_csv(csv_path)

        # 查找所有带 [drop] 标记的列
        cols_to_drop = []
        for col in df.columns:
            if "[drop" in col or ",drop]" in col:
                cols_to_drop.append(col)
                # 提取基础名称（去掉所有标记）
                base_col = re.sub(r'\[.*?\]', '', col)
                all_dropped_variates.add(base_col)

        if cols_to_drop:
            print(f"  {csv_file}: 移除列 {cols_to_drop}")
            if not dry_run:
                df = df.drop(columns=cols_to_drop)
                df.to_csv(csv_path, index=False)

                # 更新对应的 JSON 文件
                json_name = csv_file.replace(".csv", ".json")
                json_path = os.path.join(json_dir, json_name)
                if os.path.exists(json_path):
                    _update_json_remove_drop_variates(json_path, cols_to_drop, csv_path)

            modified_count += 1
        else:
            print(f"  {csv_file}: 未找到带 [drop] 标记的列")

    # 更新 _summary.json
    summary_json_path = os.path.join(json_dir, "_summary.json")
    if os.path.exists(summary_json_path):
        print(f"\n更新汇总文件: {summary_json_path}")
        if not dry_run:
            # 重新计算所有统计信息（包括数据集级别的统计）
            _update_summary_json_remove_series(summary_json_path, target_dir)

    print("-" * 60)
    action = "将修改" if dry_run else "已修改"
    print(f"{action} {modified_count}/{len(csv_files)} 个 CSV 文件")
    if all_dropped_variates:
        print(f"共移除 {len(all_dropped_variates)} 个不同的 variate: {sorted(all_dropped_variates)}")
    if dry_run:
        print("提示: 移除 --dry_run 参数以实际执行修改")


def _update_json_remove_drop_variates(json_path: str, cols_to_drop: list[str], csv_path: str | None = None) -> None:
    """更新单个 JSON 文件，移除所有带 [drop] 标记的 variate"""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # 找到并移除所有带 [drop] 标记的 variate
        keys_to_remove = []
        for key in data.keys():
            if key.startswith("_") or key == "multivariate":
                continue
            # 检查是否在要删除的列列表中，或者包含 [drop] 标记
            if key in cols_to_drop or "[drop" in key or ",drop]" in key:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del data[key]

        # 更新 _meta 信息
        if "_meta" in data:
            meta = data["_meta"]
            # 更新 kept_columns（移除所有带 [drop] 标记的列）
            if "kept_columns" in meta:
                meta["kept_columns"] = [
                    c for c in meta["kept_columns"]
                    if c not in cols_to_drop and "[drop" not in c and ",drop]" not in c
                ]
            # 更新 dropped_columns（添加被删除的列的基础名称）
            if "dropped_columns" in meta:
                for col in cols_to_drop:
                    base_col = re.sub(r'\[.*?\]', '', col)
                    if base_col not in meta["dropped_columns"]:
                        meta["dropped_columns"].append(base_col)
            # 更新 n_cols
            if "kept_columns" in meta:
                meta["n_cols"] = len(meta["kept_columns"])
            # 更新 shape
            if "shape" in meta and "n_cols" in meta:
                meta["shape"][1] = meta["n_cols"]
            # 更新其他列列表
            for list_key in ["spike_presence_columns", "random_walk_columns", "drop_marked_columns"]:
                if list_key in meta:
                    meta[list_key] = [
                        c for c in meta[list_key]
                        if c not in cols_to_drop and "[drop" not in c and ",drop]" not in c
                    ]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # 重新计算 num_observations（如果提供了 CSV 路径）
        if csv_path is not None:
            _recalculate_num_observations(json_path, csv_path)

    except Exception as e:
        print(f"    ⚠️ 更新 JSON 失败 ({json_path}): {e}")


def main():
    parser = argparse.ArgumentParser(description="时间序列预处理 Pipeline")

    # 主要处理模式的参数
    parser.add_argument(
        "--input_path",
        type=str,
        default='data/raw_csv/volicity/5T',
        help="输入路径：可以是单个 CSV 文件或包含多个 CSV 文件的文件夹"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='current_velocity',
        help="数据集名称（用于输出文件命名）"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default='5T',
        help="时间序列频率（如 'H', 'D', '15T' 等）。单文件模式可选（自动推断）；多文件模式建议指定"
    )
    parser.add_argument(
        "--missing_rate_thresh",
        type=float,
        default=0.3,
        help="允许的最大缺失率（默认: 0.3）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="输出根目录（默认: ./data）"
    )
    parser.add_argument(
        "--auto_drop",
        action="store_true",
        help="自动删除不符合要求的列。默认 False（仅检查不删除），所有列都会保留在输出中，但会标记建议删除的列"
    )

    # 数据集清理模式的参数
    parser.add_argument(
        "--remove_variate",
        type=str,
        default=None,
        help="从数据集中移除指定的 variate（列名），支持逗号分隔的多个值，如: VAR1,VAR2,VAR3"
    )
    parser.add_argument(
        "--remove_series",
        type=str,
        default=None,
        help="从数据集中移除指定的 series（CSV 文件名），支持逗号分隔的多个值，如: item_0.csv,item_1.csv"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default=None,
        help="清理操作的目标目录（包含处理后 CSV 文件的目录）"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只显示将要执行的操作，不实际修改文件"
    )
    parser.add_argument(
        "--remove_drop_marked",
        action="store_true",
        help="删除所有带 [drop] 标记的 variate（列），并更新对应的 JSON 文件"
    )

    args = parser.parse_args()

    # 清理模式：删除所有带 [drop] 标记的 variate
    if args.remove_drop_marked:
        if args.target_dir is None:
            parser.error("--remove_drop_marked 需要同时指定 --target_dir")
        remove_drop_marked_variates(args.target_dir, dry_run=args.dry_run)
        return

    # 清理模式：移除 variate（支持逗号分隔的多个值）
    if args.remove_variate is not None:
        if args.target_dir is None:
            parser.error("--remove_variate 需要同时指定 --target_dir")
        variate_names = [v.strip() for v in args.remove_variate.split(",") if v.strip()]
        for variate_name in variate_names:
            remove_variate_from_dataset(args.target_dir, variate_name, dry_run=args.dry_run)
        return

    # 清理模式：移除 series（支持逗号分隔的多个值）
    if args.remove_series is not None:
        if args.target_dir is None:
            parser.error("--remove_series 需要同时指定 --target_dir")
        series_names = [s.strip() for s in args.remove_series.split(",") if s.strip()]
        for series_name in series_names:
            remove_series_from_dataset(args.target_dir, series_name, dry_run=args.dry_run)
        return

    # 主处理模式：需要 input_path 和 dataset
    if args.input_path is None or args.dataset is None:
        parser.error("主处理模式需要同时指定 --input_path 和 --dataset")

    input_path = args.input_path
    dataset = args.dataset
    freq = args.freq
    output_dir = args.output_dir

    # 判断输入是文件还是文件夹
    is_directory = os.path.isdir(input_path)
    is_file = os.path.isfile(input_path)

    if not is_directory and not is_file:
        raise ValueError(f"输入路径不存在: {input_path}")

    if is_file:
        # === 单文件模式 ===
        print(f"[PreprocessPipeline] 单文件模式: {input_path}")

        # 先处理文件以获取推断的频率
        df = pd.read_csv(input_path)

        # 临时创建 pipeline 来推断频率（如果未指定）
        temp_pipeline = PreprocessPipeline(freq=freq)
        df_temp = df.copy()
        df_temp = temp_pipeline._normalize_timestamp_column(df_temp)
        df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
        df_temp = df_temp.set_index("timestamp")

        if freq is None:
            inferred_freq = temp_pipeline._infer_frequency(df_temp)
        else:
            inferred_freq = freq

        # 构造输出路径: ./data/processed_csv/{dataset}/{freq}/item0.csv（与多文件模式对齐）
        csv_output_dir = os.path.join(output_dir, "processed_csv", dataset, inferred_freq)
        json_output_dir = os.path.join(output_dir, "processed_summary", dataset, inferred_freq)

        os.makedirs(csv_output_dir, exist_ok=True)
        os.makedirs(json_output_dir, exist_ok=True)

        output_csv_path = os.path.join(csv_output_dir, "item0.csv")
        output_json_path = os.path.join(json_output_dir, "item0.json")

        cleaned_df, result = process_single_csv(
            csv_path=input_path,
            output_csv_path=output_csv_path,
            output_json_path=output_json_path,
            freq=freq,
            missing_rate_thresh=args.missing_rate_thresh,
            auto_drop=args.auto_drop,
            verbose=True
        )

        if cleaned_df is not None:
            # === 生成 Summary（与多文件模式对齐）===
            variate_stats = {}
            dropped_variates = []
            rw_variates = []
            sp_variates = []

            for col, col_result in result.items():
                if col.startswith("_") or col == "multivariate":
                    continue
                is_kept = col_result.get("predictable", False)
                is_rw = col_result.get("is_random_walk", False)
                is_sp = col_result.get("has_spike_presence", False)

                variate_stats[col] = {
                    "total": 1,
                    "kept": 1 if is_kept else 0,
                    "rw": 1 if is_rw else 0,
                    "sp": 1 if is_sp else 0
                }

                if not is_kept:
                    dropped_variates.append(col)
                if is_rw:
                    rw_variates.append(col)
                if is_sp:
                    sp_variates.append(col)

            # 构建 summary
            variate_summary = {
                "dataset": dataset,
                "freq": inferred_freq,
                "num_series": 1,
                "success_count": 1,
                "num_observations": result.get("_meta", {}).get("num_observations", 0),
                "series_length": len(cleaned_df),
                "variates": {}
            }

            for variate, stats in variate_stats.items():
                variate_summary["variates"][variate] = {
                    "total": stats["total"],
                    "kept": stats["kept"],
                    "rw": stats["rw"],
                    "sp": stats["sp"],
                    "kept_ratio": stats["kept"],
                    "rw_ratio": stats["rw"],
                    "sp_ratio": stats["sp"]
                }

            # 添加 correlation 信息到 summary
            corr_duplicates = []
            if "multivariate" in result:
                corr_matrix = result["multivariate"].get("correlation_matrix", {})
                corr_duplicates = result["multivariate"].get("correlation_duplicates", [])
                if corr_matrix:
                    variate_summary["correlation_matrix"] = corr_matrix
                if corr_duplicates:
                    variate_summary["correlation_duplicates"] = [
                        {"var1": v1, "var2": v2, "corr": c} for v1, v2, c in corr_duplicates
                    ]

            # 保存 summary JSON
            summary_json_path = os.path.join(json_output_dir, "_summary.json")
            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump(variate_summary, f, indent=4, ensure_ascii=False)

            # === 打印汇总信息（与多文件模式对齐）===
            print("\n" + "=" * 60)
            print("[PreprocessPipeline] 处理完成!")
            print(f"  数据集: {dataset}")
            print(f"  频率: {inferred_freq}")
            print(f"  成功: 1/1")
            print(f"  总行数: {len(cleaned_df)}")
            print(f"  总列数: {len(cleaned_df.columns)}")
            print(f"  输出 CSV: {output_csv_path}")
            print(f"  输出 JSON: {output_json_path}")
            print(f"  输出 Summary: {summary_json_path}")
            print("=" * 60)

            # === 打印 Variate 汇总统计 ===
            total_variates = len(variate_stats)
            kept_count = sum(1 for s in variate_stats.values() if s["kept"])
            rw_count = len(rw_variates)
            sp_count = len(sp_variates)

            print("\n" + "=" * 60)
            print(f"[PreprocessPipeline] Variate 汇总统计 (数据集: {dataset})")
            print("=" * 60)
            print(f"  总 Variate 数: {total_variates}")
            print(f"  保留: {kept_count} ({kept_count/total_variates*100:.1f}%)")
            print(f"  建议删除: {total_variates - kept_count} ({(total_variates - kept_count)/total_variates*100:.1f}%)")
            print(f"  Random Walk [rw]: {rw_count} ({rw_count/total_variates*100:.1f}%)")
            print(f"  Spike Presence [sp]: {sp_count} ({sp_count/total_variates*100:.1f}%)")
            print("=" * 60)

            # === 打印 Correlation 统计 ===
            if corr_duplicates:
                print("\n" + "=" * 100)
                print(f"[PreprocessPipeline] 高相关变量对统计 (|r| > 0.95)")
                print("=" * 100)
                print(f"{'变量对':<50} {'相关系数':>12}")
                print("-" * 100)

                for var1, var2, corr_val in corr_duplicates:
                    pair_str = f"{var1} <-> {var2}"
                    print(f"{pair_str:<50} {corr_val:>12.4f}")

                print("=" * 100)
                print(f"共发现 {len(corr_duplicates)} 对高相关变量")
            else:
                print("\n[PreprocessPipeline] ✅ 未发现高度相关的变量对 (|r| > 0.95)")

            # === 决策提示 ===
            has_decisions = dropped_variates or corr_duplicates

            if has_decisions:
                print("\n" + "=" * 60)
                print("⚠️  [决策提示] 需要人工决策!")
                print("=" * 60)

                if dropped_variates:
                    print("\n📌 以下 variate 建议删除:")
                    for var in dropped_variates:
                        print(f"   - {var}")
                    # 生成批量删除命令
                    base_vars = [re.sub(r'\[.*?\]', '', v) for v in dropped_variates]
                    variates_str = ",".join(base_vars)
                    print(f"\n   批量移除命令: python -m timebench.preprocess --remove_variate {variates_str} --target_dir {csv_output_dir}")
                    print(f"   或一键删除所有 [drop] 标记: python -m timebench.preprocess --remove_drop_marked --target_dir {csv_output_dir}")

                if corr_duplicates:
                    print("\n📌 以下变量对高度相关，考虑移除其中一个:")
                    for var1, var2, corr_val in corr_duplicates:
                        base_var1 = re.sub(r'\[.*?\]', '', var1)
                        base_var2 = re.sub(r'\[.*?\]', '', var2)
                        print(f"   - {base_var1} <-> {base_var2}: r = {corr_val}")
                        print(f"     移除 {base_var1}: python -m timebench.preprocess --remove_variate {base_var1} --target_dir {csv_output_dir}")
                        print(f"     移除 {base_var2}: python -m timebench.preprocess --remove_variate {base_var2} --target_dir {csv_output_dir}")

                print("\n💡 提示:")
                if dropped_variates:
                    print("   - 建议删除的 variate 已标记为 [drop]，可使用 --remove_drop_marked 一键删除")
                if corr_duplicates:
                    print("   - 如果两个变量高度相关 → 根据业务意义选择保留一个")
                print("   - 支持逗号分隔的批量操作，如: --remove_variate VAR1,VAR2,VAR3")
                print("   - 添加 --dry_run 可预览操作而不实际执行")
                print("=" * 60)

            print(f"\n[PreprocessPipeline] 汇总已保存至: {summary_json_path}")

    else:
        # === 多文件模式 ===
        csv_files = sorted([
            f for f in os.listdir(input_path)
            if f.endswith('.csv')
        ])

        if not csv_files:
            raise ValueError(f"文件夹中没有找到 CSV 文件: {input_path}")

        print(f"[PreprocessPipeline] 多文件模式: {input_path}")
        print(f"[PreprocessPipeline] 找到 {len(csv_files)} 个 CSV 文件")

        # 多文件模式下，需要确定频率
        # 如果用户未指定，尝试从第一个文件推断
        if freq is None:
            first_csv = os.path.join(input_path, csv_files[0])
            df_first = pd.read_csv(first_csv)
            temp_pipeline = PreprocessPipeline(freq=None)
            df_temp = temp_pipeline._normalize_timestamp_column(df_first)
            df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
            df_temp = df_temp.set_index("timestamp")
            try:
                inferred_freq = temp_pipeline._infer_frequency(df_temp)
                print(f"[PreprocessPipeline] 从第一个文件推断的频率: {inferred_freq}")
            except ValueError as e:
                raise ValueError(
                    f"无法从第一个文件推断频率，请使用 --freq 参数手动指定。\n原因: {e}"
                )
        else:
            inferred_freq = freq

        # 构造输出路径: ./data/processed_csv/{dataset}/{freq}/{原始csv名}.csv
        csv_output_dir = os.path.join(output_dir, "processed_csv", dataset, inferred_freq)
        json_output_dir = os.path.join(output_dir, "processed_summary", dataset, inferred_freq)

        os.makedirs(csv_output_dir, exist_ok=True)
        os.makedirs(json_output_dir, exist_ok=True)

        # 统计信息
        success_count = 0
        fail_count = 0
        total_rows = 0
        total_cols = 0

        # 用于汇总每个 variate 在所有 series 上的统计
        # {variate_name: {"total": 0, "kept": 0, "rw": 0, "sp": 0, "dropped_series": [], "kept_series": []}}
        variate_stats = {}

        # 用于汇总 correlation_duplicates
        # {(var1, var2): {"high_count": 0, "high_series": [], "high_corr_values": [], "all_corr_values": [], "all_series": []}}
        corr_dup_stats = {}

        # 用于数据集级别的统计
        series_lengths = []  # 每个series的长度
        total_observations = 0  # 所有series的observations总和

        # 记录所有 variate 都建议删除的 series
        fully_dropped_series = []  # 所有 variate 都建议删除的 series

        # 用于存储所有成功处理的series数据（用于跨series相关性检查）
        successful_series_data = {}  # {csv_file: cleaned_df}

        for csv_file in csv_files:
            csv_path = os.path.join(input_path, csv_file)
            csv_name = os.path.splitext(csv_file)[0]

            output_csv_path = os.path.join(csv_output_dir, csv_file)
            output_json_path = os.path.join(json_output_dir, f"{csv_name}.json")

            cleaned_df, result = process_single_csv(
                csv_path=csv_path,
                output_csv_path=output_csv_path,
                output_json_path=output_json_path,
                freq=inferred_freq,  # 使用统一的频率
                missing_rate_thresh=args.missing_rate_thresh,
                auto_drop=args.auto_drop,
                verbose=True
            )

            if cleaned_df is not None:
                success_count += 1
                total_rows += len(cleaned_df)
                total_cols += len(cleaned_df.columns)

                # 保存cleaned_df用于跨series相关性检查
                successful_series_data[csv_file] = cleaned_df.copy()

                # 收集数据集级别的统计信息
                series_length = len(cleaned_df)
                series_lengths.append(series_length)
                # 从result中获取num_observations（如果存在）
                num_obs = result.get("_meta", {}).get("num_observations", 0)
                total_observations += num_obs

                # 收集每个 variate 的统计信息
                for col, col_result in result.items():
                    if col.startswith("_") or col == "multivariate":
                        continue
                    # 初始化该 variate 的统计
                    if col not in variate_stats:
                        variate_stats[col] = {
                            "total": 0, "kept": 0, "rw": 0, "sp": 0,
                            "dropped_series": [], "kept_series": []
                        }
                    variate_stats[col]["total"] += 1
                    if col_result.get("predictable", False):
                        variate_stats[col]["kept"] += 1
                        variate_stats[col]["kept_series"].append(csv_file)
                    else:
                        variate_stats[col]["dropped_series"].append(csv_file)
                    if col_result.get("is_random_walk", False):
                        variate_stats[col]["rw"] += 1
                    if col_result.get("has_spike_presence", False):
                        variate_stats[col]["sp"] += 1

                # 检查该 series 是否所有 variate 都建议删除
                all_variates = [col for col in result.keys()
                                if not col.startswith("_") and col != "multivariate"]
                if all_variates:  # 如果有 variate
                    all_dropped = all(not result[col].get("predictable", False)
                                      for col in all_variates)
                    if all_dropped:
                        fully_dropped_series.append(csv_file)

                # 收集 correlation 统计（从完整相关矩阵）
                if "multivariate" in result:
                    corr_matrix = result["multivariate"].get("correlation_matrix", {})
                    corr_duplicates = result["multivariate"].get("correlation_duplicates", [])

                    # 辅助函数：去掉变量名的后缀标记（如 [rw], [sp]）
                    def strip_var_suffix(var_name):
                        return re.sub(r'\[.*?\]', '', var_name)

                    # 记录哪些 pair（基础名称）在此 series 中是高相关的
                    high_corr_pairs_in_series = set()
                    for dup in corr_duplicates:
                        var1, var2, corr_val = dup
                        # 使用基础名称作为 key，确保 CNDC[rw] 和 CNDC 被视为同一变量
                        base_var1, base_var2 = strip_var_suffix(var1), strip_var_suffix(var2)
                        pair = tuple(sorted([base_var1, base_var2]))

                        if pair not in corr_dup_stats:
                            corr_dup_stats[pair] = {
                                "high_count": 0, "high_series": [], "high_corr_values": [],
                                "all_corr_values": [], "all_series": []
                            }

                        # 避免同一 series 的同一 pair 被重复计数
                        if pair not in high_corr_pairs_in_series:
                            high_corr_pairs_in_series.add(pair)
                            corr_dup_stats[pair]["high_count"] += 1
                            corr_dup_stats[pair]["high_series"].append(csv_file)
                            corr_dup_stats[pair]["high_corr_values"].append(corr_val)

                    # 从完整相关矩阵中提取所有 pair 的相关系数
                    if corr_matrix:
                        vars_list = list(corr_matrix.keys())
                        for i, var1 in enumerate(vars_list):
                            for var2 in vars_list[i+1:]:
                                # 使用基础名称作为 key
                                base_var1, base_var2 = strip_var_suffix(var1), strip_var_suffix(var2)
                                pair = tuple(sorted([base_var1, base_var2]))
                                corr_val = corr_matrix.get(var1, {}).get(var2)
                                if corr_val is not None:
                                    if pair not in corr_dup_stats:
                                        corr_dup_stats[pair] = {
                                            "high_count": 0, "high_series": [], "high_corr_values": [],
                                            "all_corr_values": [], "all_series": []
                                        }
                                    # 避免同一 series 的同一 pair 被重复添加
                                    if csv_file not in corr_dup_stats[pair]["all_series"]:
                                        corr_dup_stats[pair]["all_corr_values"].append(corr_val)
                                        corr_dup_stats[pair]["all_series"].append(csv_file)
            else:
                fail_count += 1

        # === 跨series相关性检查（适用于UTS数据集）===
        cross_series_corr = {}
        if successful_series_data:
            pipeline = PreprocessPipeline(
                freq=inferred_freq,
                missing_rate_thresh=args.missing_rate_thresh,
                corr_thresh=0.95  # 使用默认阈值
            )
            cross_series_corr = pipeline._compute_cross_series_correlation(
                successful_series_data,
                corr_thresh=0.95
            )

            if cross_series_corr:
                print("\n" + "=" * 60)
                print(f"[PreprocessPipeline] 跨Series相关性检查 (UTS数据集)")
                print("=" * 60)
                print(f"检测到 {cross_series_corr['num_series']} 个UTS series")
                print(f"Series长度: {cross_series_corr['series_length']}")

                if cross_series_corr['correlation_duplicates']:
                    print(f"\n⚠️  发现 {len(cross_series_corr['correlation_duplicates'])} 对高度相关的UTS:")
                    for series1, series2, corr_val in cross_series_corr['correlation_duplicates']:
                        print(f"   - {series1} <-> {series2}: r = {corr_val}")
                    print("\n💡 建议: 考虑移除其中一个高度相关的series以减少冗余")
                else:
                    print("✅ 未发现高度相关的UTS对")
                print("=" * 60)

        # 打印汇总信息
        print("\n" + "=" * 60)
        print("[PreprocessPipeline] 批量处理完成!")
        print(f"  数据集: {dataset}")
        print(f"  频率: {inferred_freq}")
        print(f"  成功: {success_count}/{len(csv_files)}")
        if fail_count > 0:
            print(f"  失败: {fail_count}/{len(csv_files)}")
        print(f"  总行数: {total_rows}")
        print(f"  总列数: {total_cols}")
        print(f"  输出 CSV 目录: {csv_output_dir}")
        print(f"  输出 JSON 目录: {json_output_dir}")
        print("=" * 60)

        # 打印 variate 汇总统计
        if variate_stats:
            print("\n" + "=" * 60)
            print(f"[PreprocessPipeline] Variate 汇总统计 (数据集: {dataset})")
            print("=" * 60)
            print(f"{'Variate':<20} {'保留率':>12} {'RW率':>12} {'SP率':>12}")
            print("-" * 60)

            # 按 variate 名称排序
            for variate in sorted(variate_stats.keys()):
                stats = variate_stats[variate]
                total = stats["total"]
                kept_ratio = stats["kept"] / total if total > 0 else 0
                rw_ratio = stats["rw"] / total if total > 0 else 0
                sp_ratio = stats["sp"] / total if total > 0 else 0

                print(f"{variate:<20} {stats['kept']:>3}/{total:<3} ({kept_ratio*100:>5.1f}%) "
                      f"{stats['rw']:>3}/{total:<3} ({rw_ratio*100:>5.1f}%) "
                      f"{stats['sp']:>3}/{total:<3} ({sp_ratio*100:>5.1f}%)")

            print("=" * 60)

            # 保存 variate 汇总统计到 JSON
            variate_summary = {
                "dataset": dataset,
                "freq": inferred_freq,
                "num_series": len(csv_files),
                "success_count": success_count,
                "variates": {}
            }

            # 添加数据集级别的统计信息（仅当有多个series时）
            if len(csv_files) > 1 and series_lengths:
                variate_summary["num_observations"] = total_observations
                variate_summary["max_series_length"] = max(series_lengths)
                variate_summary["min_series_length"] = min(series_lengths)
                variate_summary["avg_series_length"] = round(sum(series_lengths) / len(series_lengths), 2)
            for variate, stats in variate_stats.items():
                total = stats["total"]
                variate_summary["variates"][variate] = {
                    "total": total,
                    "kept": stats["kept"],
                    "rw": stats["rw"],
                    "sp": stats["sp"],
                    "kept_ratio": round(stats["kept"] / total, 4) if total > 0 else 0,
                    "rw_ratio": round(stats["rw"] / total, 4) if total > 0 else 0,
                    "sp_ratio": round(stats["sp"] / total, 4) if total > 0 else 0,
                    "dropped_series": stats["dropped_series"],
                    "kept_series": stats["kept_series"]
                }

            # 注意：不再保存 correlation_duplicates 字段，但保留计算用于打印

            # 添加跨series相关性信息（仅保存correlation_matrix，不保存correlation_duplicates）
            if cross_series_corr:
                cross_series_info = {
                    "num_series": cross_series_corr["num_series"],
                    "series_length": cross_series_corr["series_length"],
                    "correlation_matrix": cross_series_corr["correlation_matrix"]
                }
                variate_summary["cross_series_correlation"] = cross_series_info

            summary_json_path = os.path.join(json_output_dir, "_summary.json")
            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump(variate_summary, f, indent=4, ensure_ascii=False)
            print(f"[PreprocessPipeline] 汇总已保存至: {summary_json_path}")

            # 打印 correlation_duplicates 统计（如果有高相关的）
            high_corr_exists = any(stats["high_count"] > 0 for stats in corr_dup_stats.values())
            if high_corr_exists:
                print("\n" + "=" * 100)
                print(f"[PreprocessPipeline] 高相关变量对统计 (数据集: {dataset})")
                print("=" * 100)
                print(f"{'变量对':<30} {'高相关次数':>14} {'r均值(高)':>12} {'r均值(低)':>12} {'r均值(全)':>12}")
                print("-" * 100)

                for pair in sorted(corr_dup_stats.keys()):
                    stats = corr_dup_stats[pair]
                    if stats["high_count"] == 0:
                        continue  # 只显示有高相关的 pair

                    high_corr_values = stats["high_corr_values"]
                    all_corr_values = stats["all_corr_values"]
                    total_count = len(all_corr_values)

                    avg_high = sum(high_corr_values) / len(high_corr_values) if high_corr_values else 0

                    # 计算非高相关的均值
                    high_series_set = set(stats["high_series"])
                    low_corr_values = [all_corr_values[i] for i, s in enumerate(stats["all_series"]) if s not in high_series_set]
                    avg_low = sum(low_corr_values) / len(low_corr_values) if low_corr_values else None

                    avg_all = sum(all_corr_values) / len(all_corr_values) if all_corr_values else 0

                    pair_str = f"{pair[0]} <-> {pair[1]}"
                    avg_low_str = f"{avg_low:.4f}" if avg_low is not None else "N/A"
                    print(f"{pair_str:<30} {stats['high_count']:>5}/{total_count:<5} ({stats['high_count']/total_count*100 if total_count else 0:>5.1f}%) "
                          f"{avg_high:>10.4f} {avg_low_str:>12} {avg_all:>12.4f}")

                print("=" * 100)
                print("说明: r均值(高)=高相关series上的均值, r均值(低)=非高相关series上的均值, r均值(全)=所有series上的均值")

            # 检查是否有需要人工决策的情况
            dropped_variates = []
            partially_dropped = []
            for variate, stats in variate_stats.items():
                if stats["kept"] < stats["total"]:
                    drop_ratio = 1 - stats["kept"] / stats["total"]
                    if drop_ratio >= 0.5:
                        dropped_variates.append((variate, stats["kept"], stats["total"], stats["dropped_series"]))
                    else:
                        partially_dropped.append((variate, stats["kept"], stats["total"], stats["dropped_series"]))

            # 检查高相关变量对
            high_corr_pairs = []
            for pair, stats in corr_dup_stats.items():
                total_count = len(stats["all_corr_values"])
                if total_count == 0:
                    continue
                ratio = stats["high_count"] / total_count
                if ratio >= 0.5:  # 在超过一半的 series 中高相关
                    high_corr_values = stats["high_corr_values"]
                    all_corr_values = stats["all_corr_values"]

                    avg_high = sum(high_corr_values) / len(high_corr_values) if high_corr_values else 0
                    avg_all = sum(all_corr_values) / len(all_corr_values) if all_corr_values else 0

                    # 计算非高相关的均值
                    high_series_set = set(stats["high_series"])
                    low_corr_values = [all_corr_values[i] for i, s in enumerate(stats["all_series"]) if s not in high_series_set]
                    avg_low = sum(low_corr_values) / len(low_corr_values) if low_corr_values else None

                    high_corr_pairs.append((pair, stats["high_count"], total_count, avg_high, avg_low, avg_all))

            # 检查跨series相关性（UTS数据集）
            cross_series_high_corr = []
            if cross_series_corr and cross_series_corr.get("correlation_duplicates"):
                for series1, series2, corr_val in cross_series_corr["correlation_duplicates"]:
                    cross_series_high_corr.append((series1, series2, corr_val))

            if dropped_variates or partially_dropped or high_corr_pairs or fully_dropped_series or cross_series_high_corr:
                print("\n" + "=" * 60)
                print("⚠️  [决策提示] 需要人工决策!")
                print("=" * 60)

                if fully_dropped_series:
                    print("\n📌 以下 series 的所有 variate 都建议删除，建议直接删除整个 series:")
                    for series in fully_dropped_series:
                        print(f"   - {series}")
                    # 生成批量删除命令
                    series_str = ",".join(sorted(fully_dropped_series))
                    print(f"\n   批量删除命令: python -m timebench.preprocess --remove_series {series_str} --target_dir {csv_output_dir}")

                if dropped_variates:
                    print("\n📌 以下 variate 在多数 series 上被丢弃，建议从整个数据集中移除该 variate:")
                    variates_to_remove = []
                    for variate, kept, total, dropped_series in dropped_variates:
                        print(f"   - {variate}: 仅在 {kept}/{total} 个 series 上保留 (丢弃率 {(1-kept/total)*100:.1f}%)")
                        print(f"     被丢弃的 series: {', '.join(dropped_series)}")
                        variates_to_remove.append(variate)
                    # 生成批量删除命令
                    variates_str = ",".join(variates_to_remove)
                    print(f"\n   批量移除命令: python -m timebench.preprocess --remove_variate {variates_str} --target_dir {csv_output_dir}")

                if partially_dropped:
                    print("\n📌 以下 variate 仅在少数 series 上被丢弃，建议移除那些 series:")
                    all_series_to_remove = set()
                    for variate, kept, total, dropped_series in partially_dropped:
                        dropped_count = total - kept
                        print(f"   - {variate}: 在 {dropped_count}/{total} 个 series 上被丢弃")
                        print(f"     被丢弃的 series: {', '.join(dropped_series)}")
                        all_series_to_remove.update(dropped_series)
                    # 生成批量删除命令
                    if all_series_to_remove:
                        series_str = ",".join(sorted(all_series_to_remove))
                        print(f"\n   批量移除命令: python -m timebench.preprocess --remove_series {series_str} --target_dir {csv_output_dir}")

                if high_corr_pairs:
                    print("\n📌 以下变量对在多数 series 上高度相关，考虑移除其中一个:")
                    for pair, high_count, total, avg_high, avg_low, avg_all in high_corr_pairs:
                        avg_low_str = f"{avg_low:.4f}" if avg_low is not None else "N/A"
                        print(f"   - {pair[0]} <-> {pair[1]}: 在 {high_count}/{total} 个 series 上高相关")
                        print(f"     r均值: 高相关={avg_high:.4f}, 非高相关={avg_low_str}, 全部={avg_all:.4f}")
                        print(f"     移除 {pair[0]}: python -m timebench.preprocess --remove_variate {pair[0]} --target_dir {csv_output_dir}")
                        print(f"     移除 {pair[1]}: python -m timebench.preprocess --remove_variate {pair[1]} --target_dir {csv_output_dir}")

                if cross_series_high_corr:
                    print("\n📌 以下UTS series之间高度相关（跨series相关性），考虑移除其中一个:")
                    for series1, series2, corr_val in cross_series_high_corr:
                        print(f"   - {series1} <-> {series2}: r = {corr_val}")
                        print(f"     移除 {series1}: python -m timebench.preprocess --remove_series {series1} --target_dir {csv_output_dir}")
                        print(f"     移除 {series2}: python -m timebench.preprocess --remove_series {series2} --target_dir {csv_output_dir}")

                print("\n💡 提示:")
                if fully_dropped_series:
                    print("   - 如果某个 series 的所有 variate 都建议删除 → 删除整个 series（删除后会自动更新 summary）")
                if dropped_variates or partially_dropped:
                    print("   - 如果某个 variate 在大多数 series 上都被丢弃 → 移除该 variate")
                    print("   - 如果某个 variate 仅在少数 series 上被丢弃 → 移除那些 series")
                if high_corr_pairs:
                    print("   - 如果两个变量高度相关 → 根据业务意义选择保留一个")
                if cross_series_high_corr:
                    print("   - 如果两个UTS series高度相关 → 根据业务意义选择保留一个")
                print("   - 支持逗号分隔的批量操作，如: --remove_variate VAR1,VAR2,VAR3 或 --remove_series file1.csv,file2.csv")
                print("   - 添加 --dry_run 可预览操作而不实际执行")
                print("=" * 60)


if __name__ == "__main__":
    main()
