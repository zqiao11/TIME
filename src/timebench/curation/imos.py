import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def select_valid_depth_period(df, depth_col="DEPTH", nominal_col="NOMINAL_DEPTH", tol=0.8):
    """根据 Depth 选择有效时间段，只保留传感器在正常深度时的数据。"""
    if depth_col not in df.columns:
        return df  # 没有深度信息则直接返回

    nominal_depth = df[depth_col].median()
    valid_mask = df[depth_col] > nominal_depth * tol

    if valid_mask.any():
        start_idx = valid_mask.idxmax()
        end_idx = valid_mask[::-1].idxmax()
        df_valid = df.loc[start_idx:end_idx]
    else:
        df_valid = pd.DataFrame(columns=df.columns)

    return df_valid


def clean_mts_by_flags_dynamic(df, vars, flag_suffix="_quality_control",
                               window_size=24, stable_ratio=0.95):
    """
    自动检测首尾异常区间并裁剪 + 对中间孤立异常点进行前向填充。

    参数：
        df : DataFrame，含所有变量和对应flag列（已按时间排序）
        vars : list[str]，变量名
        flag_suffix : str，flag列后缀
        window_size : int，滑动窗口大小（连续多少个点内大部分正常才算稳定）
        stable_ratio : float，窗口内正常比例超过多少才算稳定（默认0.7）
    返回：
        裁剪并修正后的 DataFrame
    """

    df_clean = df.copy()
    n = len(df)
    if n == 0:
        return df_clean

    # 1️⃣ 计算全局异常掩码（任一变量异常即异常）
    invalid_any = None
    for var in vars:
        flag_col = f"{var}{flag_suffix}"
        if flag_col not in df.columns:
            continue
        invalid_mask = ~df[flag_col].isin([0, 1])
        invalid_any = invalid_mask if invalid_any is None else (invalid_any | invalid_mask)

    if invalid_any is None:
        return df_clean

    # 2️⃣ 计算滑动“正常比例”
    normal_ratio = (~invalid_any).rolling(window=window_size, min_periods=1).mean()

    # 3️⃣ 从前向后找到第一个“稳定段”开始
    start_idx = 0
    for i in range(n - window_size):
        if normal_ratio.iloc[i:i + window_size].mean() >= stable_ratio:
            start_idx = i
            break

    # 4️⃣ 从后往前找到最后一个“稳定段”结束
    end_idx = n
    for i in range(n - 1, window_size, -1):
        if normal_ratio.iloc[i - window_size:i].mean() >= stable_ratio:
            end_idx = i
            break

    df_trimmed = df.iloc[start_idx:end_idx].reset_index(drop=True)

    print(f"✂️ 全局裁剪范围: start={start_idx}, end={end_idx}, 保留长度={len(df_trimmed)}")

    # 5️⃣ 对裁剪后的数据逐变量修正中间零散异常点
    for var in vars:
        val_col = var
        flag_col = f"{var}{flag_suffix}"
        if val_col not in df_trimmed.columns or flag_col not in df_trimmed.columns:
            continue

        bad_mask = ~df_trimmed[flag_col].isin([0, 1])
        if bad_mask.any():
            s = df_trimmed[val_col].astype(float).copy()
            s[bad_mask] = np.nan
            s = s.ffill()  # 前向填充
            df_trimmed[val_col] = s

    return df_trimmed

# ========= 文件路径设置 =========
folder_path = "/home/zhongzheng/TSBench/Build-TSBench/data/industry/IMOS"
output_dir = Path(folder_path) / "processed_15min"
plot_dir = output_dir / "plots"
output_dir.mkdir(exist_ok=True)
plot_dir.mkdir(exist_ok=True)

# ========= 文件筛选 =========
csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
print(f"📂 一共发现 {len(csv_files)} 个 CSV 文件")

# 删除最后一个和倒数第五个
removed_files = [csv_files[-1], csv_files[-5]]
selected_files = [f for f in csv_files[-10:] if f not in removed_files]

vars = ["CNDC", "DOX2", "PSAL", "TEMP", "TURB", "CPHL"]

# ========= 主循环 =========
for i, file_name in enumerate(selected_files):
    file_path = Path(folder_path) / file_name
    print(f"\n=== 处理文件: {file_name} ===")

    # 读取 CSV
    df = pd.read_csv(file_path, comment="#", low_memory=False)

    if "DateTime" not in df.columns:
        print("⚠️ 未找到 DateTime 列，跳过")
        continue

    # 转换时间 & 排序
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime")

    # 先用“原始分辨率 + 原始 flags”做全局首尾裁剪 & 中间异常修复
    df_clean = clean_mts_by_flags_dynamic(
        df,
        vars=vars,
        flag_suffix="_quality_control",
        window_size=24,  # 连续24个点为检测窗口，可根据采样间隔调整
        stable_ratio=0.95 # 窗口内95%以上为正常即视为稳定
    )

    # ====== 15min 重采样 ======
    df_15min = df_clean.set_index("DateTime").resample("15T").mean()
    df_15min = select_valid_depth_period(df_15min, depth_col="DEPTH", nominal_col="NOMINAL_DEPTH", tol=0.95)

    # 缺失统计
    all_nan_ratio = (df_15min.isna().all(axis=1)).mean()
    print(f"剔除后剩余: {len(df_15min)}")
    print(f"剔除后数据缺失比例: {all_nan_ratio:.2%}")

    # ====== Flag 检查 + 可视化 ======
    print("\n🧭 Flag 异常比例检查 + 可视化（基于 df_15min）：")

    fig, axes = plt.subplots(len(vars), 1, figsize=(12, 3 * len(vars)), sharex=True)
    if len(vars) == 1:
        axes = [axes]

    for ax, var in zip(axes, vars):
        flag_col = f"{var}_quality_control"
        if var not in df_15min.columns:
            ax.set_title(f"{var} (未找到)")
            continue

        # 绘制时间序列
        ax.plot(df_15min.index, df_15min[var], label=var, color="tab:blue")

        # flag 列来自原始 df（因为 df_15min 重采样后 flag 可能被平均掉）
        if flag_col in df.columns:
            invalid_mask = ~df[flag_col].isin([0, 1])
            invalid_ratio = invalid_mask.mean()
            print(f"  - {flag_col}: {invalid_ratio:.2%} (非 0/1 比例)")

            # 找出异常点对应的时间戳
            bad_points = df.loc[invalid_mask, ["DateTime", var]].dropna()
            if not bad_points.empty:
                ax.scatter(bad_points["DateTime"], bad_points[var],
                           color="red", s=10, label="Invalid flag")
        else:
            print(f"  ⚠️ 未找到 {flag_col}")

        ax.set_ylabel(var)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time")
    plt.suptitle(f"{file_name} - Flag Check Visualization (15min cleaned)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plot_path = plot_dir / f"{file_name.replace('.csv', '_flags_15min.png')}"
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"📊 图像已保存至: {plot_path}")



    # # ====== 仅绘制干净的 df_15min（无异常点） ======
    # print("\n🎨 绘制无异常点的清洗后序列...")
    #
    # fig2, axes2 = plt.subplots(len(vars), 1, figsize=(12, 3 * len(vars)), sharex=True)
    # if len(vars) == 1:
    #     axes2 = [axes2]
    #
    # for ax, var in zip(axes2, vars):
    #     if var not in df_15min.columns:
    #         ax.set_title(f"{var} (未找到)")
    #         continue
    #     ax.plot(df_15min.index, df_15min[var], color="tab:blue", linewidth=1.5)
    #     ax.set_ylabel(var)
    #     ax.legend([var], loc="upper right")
    #
    # axes2[-1].set_xlabel("Time")
    # plt.suptitle(f"{file_name} - Cleaned 15min Time Series", fontsize=14)
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    #
    # clean_plot_path = plot_dir / f"{file_name.replace('.csv', '_clean_15min.png')}"
    # plt.savefig(clean_plot_path, dpi=200)
    # plt.close(fig2)
    # print(f"✅ 无异常点图像已保存至: {clean_plot_path}")


    # # ====== 保留指定列 ======
    # df_15min = df_15min.reset_index()  # 恢复 DateTime 为列
    # keep_cols = [col for col in ["DateTime"]+ vars if col in df_15min.columns]
    # df_15min = df_15min[keep_cols]
    #
    # # ====== 保存结果 ======
    # output_path = output_dir / f"item_{i}.csv"
    # df_15min.to_csv(output_path, index=False)
    # print(f"✅ 已保存至: {output_path}")
