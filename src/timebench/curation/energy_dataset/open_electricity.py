# -*- coding: utf-8 -*-
"""
Merge all NEM regions into one 10-variable MTS (demand & price per region)
Range: 2025-01-01 ~ 2025-06-01
Keep original 5min frequency (no resampling)
"""

from pathlib import Path
import pandas as pd

# ========= 1) File path =========
csv_path = Path("/home/zhongzheng/TSBench/Build-TSBench/data/curated/price/NEM_price_2023-2025_5m_all_regions.csv")

# ========= 2) Load data =========
print(f"[INFO] Loading {csv_path.name} ...")
df = pd.read_csv(csv_path)

# Parse timestamp & remove timezone if present
if "timestamp" not in df.columns:
    raise ValueError("'timestamp' column not found in the CSV.")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["timestamp"] = df["timestamp"].dt.tz_localize(None)
df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

# ========= 3) Define regions & target columns =========
regions = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]
target_cols = [f"demand__{r}" for r in regions] + [f"price__{r}" for r in regions]
missing = [c for c in target_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

# ========= 4) Select time range =========
start_date = pd.Timestamp("2025-01-01")
end_date = pd.Timestamp("2025-06-01")
df = df[(df["timestamp"] >= start_date) & (df["timestamp"] < end_date)]
print(f"[INFO] Filtered rows: {len(df)} from {start_date.date()} to {end_date.date()}")

# ========= 5) Keep target columns =========
df_mts = df[["timestamp"] + target_cols].reset_index(drop=True)

# ========= 6) Print summary =========
print("\n[INFO] MTS column summary:")
for c in target_cols:
    print(f"{c:<20} → {df_mts[c].notna().sum():>8} samples")

# ========= 7) Save merged 5min MTS =========
save_path = csv_path.with_name("NEM_MTS_price_demand_2025H1_5min.csv")
df_mts.to_csv(save_path, index=False)

print(f"\n[OK] Saved merged 10-variable (5min) MTS -> {save_path}")
print(f"[INFO] Final shape: {df_mts.shape}")
