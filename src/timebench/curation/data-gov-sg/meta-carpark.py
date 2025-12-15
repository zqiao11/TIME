import pandas as pd

# 读取 carpark 元数据
df = pd.read_csv("/home/zhongzheng/TSBench/Build-TSBench/curation/data-gov-sg/meta/HDBCarparkInformation.csv")

# 筛选符合条件的 carpark
filtered_df = df[
    (df["car_park_type"].str.upper() == "MULTI-STOREY CAR PARK") &
    (df["type_of_parking_system"].str.upper() == "ELECTRONIC PARKING") &
    (df["short_term_parking"].str.upper() == "WHOLE DAY") &
    (df["free_parking"].str.upper() == "SUN & PH FR 7AM-10.30PM") &
    (df["night_parking"].str.upper() == "YES")
]

# 保存筛选后的数据
filtered_df.to_csv("filtered_storey_carparks.csv", index=False)

end = 1