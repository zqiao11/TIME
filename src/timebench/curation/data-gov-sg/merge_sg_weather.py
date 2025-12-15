import pandas as pd
import numpy as np
import os


stations = ['S43', 'S50', 'S104', 'S107', 'S115', 'S116']
metrics = ['rainfall', 'humidity', 'air_temp', 'wind_speed', 'wind_direction']
start_date = "2017-01-01"
end_date = "2025-05-31"

csv_paths = {
    "rainfall": "/home/zhongzheng/TSBench/Build-TSBench/data/curated/rainfall/valid_rainfall_daily_2016-05-01_to_2025-05-31.csv",
    "humidity": "/home/zhongzheng/TSBench/Build-TSBench/data/curated/humidity/valid_humidity_daily_2016-11-01_to_2025-05-31.csv",
    "air_temp": "/home/zhongzheng/TSBench/Build-TSBench/data/curated/air-temperature/valid_air_temperature_daily_2016-05-01_to_2025-05-31.csv",
    "wind_speed": "/home/zhongzheng/TSBench/Build-TSBench/data/curated/wind-speed/valid_windspeed_daily_2016-01-01_to_2025-05-31.csv",
    "wind_direction": "/home/zhongzheng/TSBench/Build-TSBench/data/curated/wind-direction/valid_winddirection_daily_2016-11-01_to_2025-05-31.csv",
}

# ==== 读取并预处理 ====
def load_and_prepare(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date  # 只保留日期
    return df.drop(columns=["timestamp"])

datasets = {name: load_and_prepare(path) for name, path in csv_paths.items()}

# ==== 按站点合并 ====
mts_dict = {}

for station in stations:
    dfs = []
    for name, df in datasets.items():
        if station in df.columns:
            tmp = df[["date", station]].rename(columns={station: name})
            dfs.append(tmp)
        else:
            print(f"⚠️ Station {station} not found in {name}")

    # 逐个 merge
    merged = dfs[0]
    for tmp in dfs[1:]:
        merged = pd.merge(merged, tmp, on="date", how="outer")

    # 日期范围过滤
    merged = merged[(merged["date"] >= pd.to_datetime(start_date).date()) &
                    (merged["date"] <= pd.to_datetime(end_date).date())]

    # 排序
    merged = merged.sort_values("date").reset_index(drop=True)

    # 保存
    mts_dict[station] = merged
    merged.to_csv(f"merged_MTS_{station}.csv", index=False)

print("✅ Done! 生成文件：merged_MTS_S43.csv, merged_MTS_S50.csv ...")