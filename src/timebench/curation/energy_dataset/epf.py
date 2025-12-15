# -*- coding: utf-8 -*-
"""
Extract only 'Date' and 'Price'-related columns from all EPF CSVs,
rename price columns to 'Price', and save cleaned versions.
"""

from pathlib import Path
import pandas as pd
import re

# ========= 1) Root directory =========
root = Path("/home/zhongzheng/TSBench/Build-TSBench/data/curated/epf")

# ========= 2) Get all CSV files =========
csv_paths = sorted(root.glob("*.csv"))
print(f"[INFO] Found {len(csv_paths)} CSV files\n")

# ========= 3) Process each CSV =========
for path in csv_paths:
    df = pd.read_csv(path)

    # Check for datetime column
    if "Date" not in df.columns:
        raise ValueError(f"'Date' column not found in {path.name}")

    # Find all columns containing 'Price'
    price_cols = [col for col in df.columns if re.search("price", col, re.IGNORECASE)]
    if not price_cols:
        print(f"[WARN] No Price columns found in {path.name}, skipped.")
        continue

    # If multiple Price columns exist, merge them into one (prefer the first non-null)
    if len(price_cols) > 1:
        df["Price"] = df[price_cols].bfill(axis=1).iloc[:, 0]
    else:
        df["Price"] = df[price_cols[0]]

    # Keep only Date and Price
    df_new = df[["Date", "Price"]].copy()

    # Save new version
    save_path = path.with_name(path.stem + "_Price.csv")
    df_new.to_csv(save_path, index=False)
    print(f"[OK] Saved -> {save_path.name:<35} | rows={len(df_new)} | price_cols={price_cols}")
