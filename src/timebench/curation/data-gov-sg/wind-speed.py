import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from requests.exceptions import HTTPError
import time
from collections import defaultdict
import os
import math

start = "2016-01-01"
end = "2025-05-31"
start_date = datetime.strptime(start, "%Y-%m-%d")
end_date = datetime.strptime(end, "%Y-%m-%d")
output_dir = "/home/zhongzheng/TSBench/Build-TSBench/data/curated/wind-speed"
os.makedirs(output_dir, exist_ok=True)

records = []
not_found_dates = []  # 存储 404 日期
non_2359_dates = []

###################### Curate data ######################
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime("%Y-%m-%d")
    url = "https://api-open.data.gov.sg/v2/real-time/api/wind-speed"
    params = {"date": date_str}
    retries = 0
    MAX_RETRIES = 3

    while retries < MAX_RETRIES:
        try:
            print(f"Fetching {date_str} ...")
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 404:
                print(f"404 Not Found on {date_str}")
                not_found_dates.append(date_str)
                break
            resp.raise_for_status()
            data = resp.json().get("data", [])

            # 汇总每个 station 的 readings
            station_sum = defaultdict(float)
            station_count = defaultdict(int)

            for item in data["readings"]:
                for entry in item['data']:
                    sid = entry['stationId']
                    val = entry['value']
                    station_sum[sid] += val
                    station_count[sid] += 1

            # 求均值，保留一位小数
            reading_daily_mean = {
                sid: round(station_sum[sid] / station_count[sid], 1)
                for sid in station_sum
            }

            ts = data["readings"][0]["timestamp"]
            reading_daily_mean['timestamp'] = ts

            # 检查是否为 23:59:00（按新加坡时间格式判断）
            if not ts.endswith("23:59:00+08:00"):
                non_2359_dates.append(current_date.strftime("%Y-%m-%d"))

            records.append(reading_daily_mean)

            break  # 成功就退出重试
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                print(f"429 Too Many Requests on {date_str}, retrying after 60s...")
                time.sleep(60)
                retries += 1
            else:
                print(f"Failed on {date_str}: {e}")
                break
        except Exception as e:
            print(f"Error on {date_str}: {e}")
            break

    current_date += timedelta(days=1)

# 保存数据为 CSV
df = pd.DataFrame(records)
df["timestamp"] = pd.to_datetime(df["timestamp"])
columns = ["timestamp"] + [col for col in df.columns if col != "timestamp"]
df = df[columns]
df.to_csv(os.path.join(output_dir,f"windspeed_daily_{start}_to_{end}.csv"), index=False)

# 保存 404 日期列表
with open(os.path.join(output_dir,"windspeed_not_found_dates.txt"), "w") as f:
    for d in not_found_dates:
        f.write(d + "\n")

with open(os.path.join(output_dir,"windspeed_non_2359_dates.txt"), "w") as f:
    for d in non_2359_dates:
        f.write(d + "\n")


print(f"✅ Saved to windspeed_daily_{start}_to_{end}.csv")


###################### Filter Variates with too many missing values ######################
df = pd.read_csv(os.path.join(output_dir,f"windspeed_daily_{start}_to_{end}.csv"), parse_dates=["timestamp"])

data_columns = [col for col in df.columns if col != "timestamp"]
missing_rate = df[data_columns].isna().mean()
valid_stations = missing_rate[missing_rate < 0.1].index.tolist()

df_valid = df[["timestamp"] + valid_stations]
df_valid.to_csv(os.path.join(output_dir, f"valid_windspeed_daily_{start}_to_{end}.csv"), index=False)


###################### Plot ######################
plt.figure(figsize=(14, 6))
for station in valid_stations:
    plt.plot(df.index, df[station], label=station)

plt.xlabel("Date")
plt.ylabel("Wind Speed (daily mean)")
plt.title(f"Wind Speed Per Station ({start} to {end})")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
# plt.show()
plt.savefig(os.path.join(output_dir, "all_valid.png"))


# 遍历并单独绘图保存
for station in valid_stations:
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df[station])
    plt.xlabel("Date")
    plt.ylabel("Wind Speed")
    plt.title(f"{station} Wind Speed ({start} to {end})")
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)

    # plt.show()
    plt.savefig(os.path.join(output_dir, f"{station}.png"))


cols = 2
rows = math.ceil(len(valid_stations) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), sharex=True)
axes = axes.flatten()

for i, station in enumerate(valid_stations):
    ax = axes[i]
    ax.plot(df.index, df[station])
    ax.set_title(f"{station}")
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)

# 处理多余空子图
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle(f"Wind Speed per Station ({start} to {end})\n(Missing rate < 10%)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()
plt.savefig(os.path.join(output_dir, "sub_valid.png"))