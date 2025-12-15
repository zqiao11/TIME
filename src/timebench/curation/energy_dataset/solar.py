# # -*- coding: utf-8 -*-
# """
# Downsample each solar dataset to hourly frequency,
# select time range [2023-04-01, 2025-04-01],
# and merge as a multivariate time series (MTS).
# """

from pathlib import Path
import pandas as pd

# ========= 1) File paths =========
paths = {
    "solar_63726": Path("/home/zhongzheng/TSBench/Build-TSBench/data/curated/solar/solar_id_63726.csv"),
    "solar_51616": Path("/home/zhongzheng/TSBench/Build-TSBench/data/curated/solar/solar_id_51616.csv"),
    "solar_31378": Path("/home/zhongzheng/TSBench/Build-TSBench/data/curated/solar/solar_id_31378.csv"),
}

# ========= 2) Target time range =========
start_date = pd.Timestamp("2021-04-01")
end_date = pd.Timestamp("2025-04-01")

# ========= 3) Load, parse, downsample =========
dfs = {}
for name, path in paths.items():
    print(f"[INFO] Loading {path.name} ...")
    df = pd.read_csv(path)

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    # Select range
    df = df[(df["datetime"] >= start_date) & (df["datetime"] < end_date)]

    # Resample to hourly mean
    df = df.set_index("datetime").resample("1H").mean(numeric_only=True)

    # Keep only energy_generation
    if "energy_generation" not in df.columns:
        raise ValueError(f"'energy_generation' column not found in {path.name}")
    df = df[["energy_generation"]].rename(columns={"energy_generation": name})

    dfs[name] = df
    print(f"  → Time range: {df.index.min()} → {df.index.max()}, samples = {len(df)}")

# # ========= 4) Merge on datetime =========
# merged = pd.concat(dfs.values(), axis=1, join="inner")
#
# print("\n[RESULT]")
# print(f"Common range: {merged.index.min()} → {merged.index.max()}")
# print(f"Merged shape: {merged.shape}")
# print(merged.head())
#
# # ========= 5) Save result =========
# save_path = Path("/home/zhongzheng/TSBench/Build-TSBench/data/curated/solar/solar_MTS_energy_hourly.csv")
# merged.to_csv(save_path, index_label="datetime")
# print(f"[OK] Saved merged hourly MTS -> {save_path}")
#
#
#
#
#
# import matplotlib.pyplot as plt
# # ========= 1) Load data =========
# csv_path = Path("/home/zhongzheng/TSBench/Build-TSBench/data/curated/solar/solar_MTS_energy_hourly.csv")
# df = pd.read_csv(csv_path, parse_dates=["datetime"])
#
# # ========= 2) Plot =========
# plt.figure(figsize=(14, 6))
# for col in df.columns[1:]:
#     plt.plot(df["datetime"], df[col], label=col, linewidth=1)
#
# plt.title("Hourly Energy Generation — Solar MTS (2023-04 to 2025-04)")
# plt.xlabel("Time")
# plt.ylabel("Energy Generation (kWh)")
# plt.legend(loc="upper right", ncol=3, fontsize=8)
# plt.grid(True, alpha=0.5)
# plt.tight_layout()
# plt.show()
