import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from requests.exceptions import HTTPError
import time
from collections import defaultdict
import os
import math

start = "2021-01-01"
end = "2025-5-31"
start_date = datetime.strptime(start, "%Y-%m-%d")
end_date = datetime.strptime(end, "%Y-%m-%d")
output_dir = f"/home/zhongzheng/TSBench/Build-TSBench/data/curated/pm25"
os.makedirs(output_dir, exist_ok=True)

records = []
not_found_dates = []

###################### Curate data ######################
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime("%Y-%m-%d")
    url = "https://api-open.data.gov.sg/v2/real-time/api/pm25"
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
            for item in data["items"]:
                ts = item["timestamp"]
                readings = item["readings"]["pm25_one_hourly"]
                readings["timestamp"] = ts
                records.append(readings)
            # time.sleep(1)  # 限速
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
df = df.sort_values("timestamp")[["timestamp", "central", "east", "north", "south", "west"]]
df.to_csv(os.path.join(output_dir, f"pm25_hourly_{start}_to_{end}.csv"), index=False)

# 保存 404 日期列表
with open(os.path.join(output_dir, "pm25_not_found_dates.txt"), "w") as f:
    for d in not_found_dates:
        f.write(d + "\n")

print(f"✅ Saved to pm25_hourly_{start}_to_{end}.csv")


###################### Filter Variates with too many missing values ######################
df = pd.read_csv(os.path.join(output_dir,f"pm25_hourly_{start}_to_{end}.csv"), parse_dates=["timestamp"])
df = df.set_index("timestamp").sort_index()

###################### Plot ######################
plt.figure(figsize=(14, 6))
for region in ["central", "east", "north", "south", "west"]:
    plt.plot(df.index, df[region], label=region)

plt.xlabel("Date")
plt.ylabel("PM2.5")
plt.title(f"PM2.5 Per Region ({start} to {end})")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
# plt.show()
plt.savefig(os.path.join(output_dir, "all.png"))


# 遍历并单独绘图保存
for region in ["central", "east", "north", "south", "west"]:
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df[region])
    plt.xlabel("Date")
    plt.ylabel("PM2.5")
    plt.title(f"{region} PM2.5 ({start} to {end})")
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)

    # plt.show()
    plt.savefig(os.path.join(output_dir, f"{region}.png"))


cols = 1
rows = 5
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), sharex=True)
axes = axes.flatten()

for i, region in enumerate(["central", "east", "north", "south", "west"]):
    ax = axes[i]
    ax.plot(df.index, df[region])
    ax.set_title(f"{region}")
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)

# 处理多余空子图
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle(f"PM2.5 per Region ({start} to {end})\n(Missing rate < 10%)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()
plt.savefig(os.path.join(output_dir, "sub.png"))

