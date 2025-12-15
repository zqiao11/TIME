# -*- coding: utf-8 -*-
"""
Extract the last 1/5 of uv_zt_x0_y0.csv and save as a new CSV file.
"""

from pathlib import Path
import pandas as pd

# ========= 1) File path =========
csv_path = Path("/home/zhongzheng/TSBench/Build-TSBench/data/curated/wind/mfwpn/uv_zt_x0_y0.csv")

# ========= 2) Load data =========
print(f"[INFO] Loading {csv_path.name} ...")
df = pd.read_csv(csv_path)

# ========= 3) Check time column =========
if "time_index" not in df.columns:
    raise ValueError("'time_index' column not found in the CSV.")

# ========= 4) Compute tail portion =========
n_total = len(df)
n_last = n_total // 5
df_tail = df.tail(n_last).reset_index(drop=True)
print(f"[INFO] Total rows: {n_total}, extracted last 1/5 = {n_last} rows.")

# ========= 5) Save new CSV =========
save_path = csv_path.with_name(csv_path.stem + "_last1of5.csv")
df_tail.to_csv(save_path, index=False)

print(f"[OK] Saved truncated dataset -> {save_path}")
print(f"[INFO] Final shape: {df_tail.shape}")
