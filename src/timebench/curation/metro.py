# -*- coding: utf-8 -*-
"""
Downsample and visualize only selected variables (2nd–8th columns)
from MetroPT3(AirCompressor).csv, aggregated to 15-minute resolution.
"""

from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt

# ========= 1) Load =========
csv_path = Path("/home/zhongzheng/TSBench/Build-TSBench/data/curated/MetroPT3(AirCompressor).csv")
df = pd.read_csv(csv_path)

# ========= 2) Parse time column =========
def parse_time(df):
    if "timestamp" in df.columns:
        try:
            t = pd.to_datetime(df["timestamp"], errors="raise", infer_datetime_format=True)
            df["datetime"] = t
        except Exception:
            s = pd.to_numeric(df["timestamp"], errors="coerce")
            df["datetime"] = pd.to_datetime(s, unit="s", origin="unix", errors="coerce")
    else:
        df["datetime"] = pd.date_range("2000-01-01", periods=len(df), freq="S")

    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()
    return df

df = parse_time(df)

# ========= 3) Resample to 15-minute frequency =========
df_15min = df.resample("15min").mean(numeric_only=True)

# ========= 4) Select 2nd–8th columns for visualization =========
# 注意：df.columns[1:8] 是第二到第八列（Python索引从0开始）
cols_to_plot = df_15min.columns[1:8].tolist()

# ========= 5) Plot =========
n = len(cols_to_plot)
n_cols = 3
n_rows = math.ceil(n / n_cols)
fig_height = max(6, 2.5 * n_rows)

fig = plt.figure(figsize=(18, fig_height))
for i, c in enumerate(cols_to_plot, 1):
    ax = fig.add_subplot(n_rows, n_cols, i)
    ax.plot(df_15min.index, df_15min[c], linewidth=0.9)
    ax.set_title(c, fontsize=10)
    ax.set_xlabel("Time (15min)")
    ax.set_ylabel("Value")
    ax.grid(True, linewidth=0.4, alpha=0.5)

plt.suptitle("MetroPT3 (Air Compressor) — 15min Downsampled (Columns 2–8)", y=1.02, fontsize=14)
plt.tight_layout()

subplots_path = csv_path.with_name("MetroPT3_AirCompressor_15min_selected.png")
plt.savefig(subplots_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"[OK] Saved selected-variable plot -> {subplots_path}")
print(f"[INFO] Plotted columns: {cols_to_plot}")
print(f"[INFO] Downsampled shape: {df_15min.shape}")
