# -*- coding: utf-8 -*-
"""
Extract and merge selected EWELD City2 users into one multivariate time series (MTS)
Range: 2019-06-01 ~ 2020-01-01
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ========= 1) File path =========
csv_path = Path("/home/zhongzheng/TSBench/Build-TSBench/data/curated/energy-ts/eweld/EWELD_City2_Cleaned.csv")

# ========= 2) Load data =========
print(f"[INFO] Loading {csv_path.name} ...")
df = pd.read_csv(csv_path)

# Parse time column
if "Time" not in df.columns:
    raise ValueError("'Time' column not found in the CSV.")
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df = df.dropna(subset=["Time"]).sort_values("Time")

# ========= 3) Filter time range =========
start_date = pd.Timestamp("2019-06-01")
end_date = pd.Timestamp("2020-01-01")
df = df[(df["Time"] >= start_date) & (df["Time"] < end_date)]
print(f"[INFO] Filtered rows: {len(df)} from {start_date.date()} to {end_date.date()}")

# ========= 4) Select target columns =========
target_cols = [
    "User_U9", "User_U26", "User_U62", "User_U74", "User_U142", "User_U143",
    "User_U148", "User_U158", "User_U260", "User_U293"
]

missing = [c for c in target_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# ========= 5) Visualization =========
plot_dir = csv_path.parent / "plots"
plot_dir.mkdir(exist_ok=True)

n_vars = len(target_cols)
fig, axes = plt.subplots(
    nrows=n_vars, ncols=1,
    figsize=(12, 3 * n_vars),
    sharex=True
)

if n_vars == 1:
    axes = [axes]  # 确保axes是list

for ax, col in zip(axes, target_cols):
    ax.plot(df["Time"], df[col], lw=1.2, color="tab:blue")
    ax.set_title(col, fontsize=10)
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.5)

axes[-1].set_xlabel("Time")
plt.tight_layout()

save_path = plot_dir / "EWELD_City2_SelectedUsers_20190601_20200101.png"
plt.savefig(save_path, dpi=150)
print(f"\n[OK] Saved combined plot -> {save_path}")

# ========= 5) Merge into multivariate time series =========
mts_df = df[["Time"] + target_cols].reset_index(drop=True)

# ========= 6) Print summary =========
print(f"\n[INFO] MTS created with shape: {mts_df.shape}")
print("[INFO] Each variate valid sample count:")
for col in target_cols:
    print(f"  {col:<10} → {mts_df[col].notna().sum():>8}")

# ========= 7) Save merged MTS =========
save_path = csv_path.with_name("EWELD_City2_SelectedUsers_20190601_20200101_MTS.csv")
mts_df.to_csv(save_path, index=False)
print(f"\n[OK] Saved merged MTS -> {save_path}")

