import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from requests.exceptions import HTTPError
import time
import os
import math


start = "2025-01-01T00:00:00"
end = "2025-06-01T00:00:00"

start_datetime = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
end_datetime = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S")
output_dir = "/home/zhongzheng/TSBench/Build-TSBench/data/curated/carpark"
os.makedirs(output_dir, exist_ok=True)

records = []
not_found_dates = []  # 存储 404 日期

# 读取筛选后的 carpark 编号
filtered_df = pd.read_csv("/home/zhongzheng/TSBench/Build-TSBench/curation/data-gov-sg/filtered_storey_carparks.csv")
valid_carparks = set(filtered_df["car_park_no"].str.strip())  # 转为集合加速查找

current_datetime = start_datetime
# while current_datetime <= end_datetime:
#     datetime_str = current_datetime.strftime("%Y-%m-%dT%H:%M:%S")
#     url = "https://api.data.gov.sg/v1/transport/carpark-availability"
#     params = {"date_time": datetime_str}
#     retries = 0
#     MAX_RETRIES = 3
#
#     while retries < MAX_RETRIES:
#         try:
#             print(f"Fetching {datetime_str} ...")
#             resp = requests.get(url, params=params, timeout=10)
#             if resp.status_code == 404:
#                 print(f"404 Not Found on {datetime_str}")
#                 not_found_dates.append(datetime_str)
#                 break
#             resp.raise_for_status()
#             data = resp.json().get("items", [])
#
#             for item in data:
#                 row = {"timestamp": current_datetime,}
#                 for cp in item["carpark_data"]:
#                     carpark_id = cp["carpark_number"].strip()
#                     if carpark_id not in valid_carparks:
#                         continue
#
#                     # 只取 lot_type == 'C'（可选）
#                     for info in cp["carpark_info"]:
#                         # 每个 timestamp 只保留 carpark 的 lots_available
#                         row[carpark_id] = int(info["lots_available"])
#                         break
#                 records.append(row)
#             break
#         except requests.exceptions.HTTPError as e:
#             if resp.status_code == 429:
#                 print(f"429 Too Many Requests on {datetime_str}, retrying after 60s...")
#                 time.sleep(60)
#                 retries += 1
#             else:
#                 print(f"Failed on {datetime_str}: {e}")
#                 break
#         except Exception as e:
#             print(f"Error on {datetime_str}: {e}")
#             break
#
#     current_datetime += timedelta(minutes=15)
#
# # 转为 DataFrame 并保存
# df = pd.DataFrame(records)
# df["timestamp"] = pd.to_datetime(df["timestamp"])
# df = df.sort_values("timestamp")
#
# # 将 timestamp 还原为列再保存
# df.to_csv(
#     os.path.join(output_dir, f"carpark_15T_{start}_to_{end}.csv"),
#     index=False
# )
#
# # 保存 404 失败记录
# with open(os.path.join(output_dir, "carpark_not_found_dates.txt"), "w") as f:
#     for d in not_found_dates:
#         f.write(d + "\n")
#
# print("✅ Saved")


# 读取数据
df = pd.read_csv(os.path.join(output_dir, f"carpark_15T_{start}_to_{end}.csv"), parse_dates=["timestamp"])
df = df.set_index("timestamp").sort_index()

# 画图
plt.figure(figsize=(14, 6))
for col in df.columns[:3]:
    plt.plot(df.index, df[col], label=col)

plt.xlabel("Time")
plt.ylabel("")
plt.title(f"Per 15T Carpark Available Lots ({start} to {end})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

end = 1