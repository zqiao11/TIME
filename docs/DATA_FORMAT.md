# 时间序列数据集格式说明

本文档详细说明时间序列数据集格式，以及如何将 CSV/Pandas 数据转换为 Arrow 格式。

---

##  CSV 文件格式要求

### 格式规范

**时间序列的CSV文件必须满足以下要求：**

| 要求 | 说明 |
|------|------|
| **第一列** | 必须是 timestamp（时间戳） |
| **其他列** | 数值数据（时间序列的各个变量） |
| **时间格式** | 可被 `pd.to_datetime()` 解析的格式 |

### 正确的 CSV 格式示例

```csv
timestamp,temperature,humidity,pressure
2024-01-01 00:00:00,25.3,60.2,1013.25
2024-01-01 00:15:00,25.4,60.1,1013.30
2024-01-01 00:30:00,25.5,59.8,1013.28
```

### 从 DataFrame 保存 CSV 的正确方式

```python
import pandas as pd

# 创建 DataFrame
df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=100, freq="15T"),
    "temperature": [...],
    "humidity": [...],
})

# ✅ 正确：timestamp 作为普通列保存（推荐）
df.to_csv("data.csv", index=False)

# ✅ 也正确：timestamp 作为 index 保存
df.set_index("timestamp").to_csv("data.csv")
```

---

## 核心概念：Dataset 与 Time Series 的关系

```
┌─────────────────────────────────────────────────────────────────┐
│                         Dataset (数据集)                         │
│  ┌───────────────┐ ┌───────────────┐     ┌───────────────┐     │
│  │  Sample 0     │ │  Sample 1     │ ... │  Sample N-1   │     │
│  │  (1个时间序列) │ │  (1个时间序列) │     │  (1个时间序列) │     │
│  │               │ │               │     │               │     │
│  │  可以是 UTS   │ │  可以是 MTS   │     │  可以是 MTS   │     │
│  │  或 MTS       │ │  (多变量)     │     │  (多变量)     │     │
│  └───────────────┘ └───────────────┘     └───────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### 关键点

| 概念 | 说明 |
|------|------|
| **Dataset** | 包含 **多个** 时间序列 (samples/items) |
| **Sample/Item** | **一个** 时间序列(series)，可以是 UTS 或 MTS |
| **UTS** | 单变量时间序列，`target` 形状为 `[1, T]` |
| **MTS** | 多变量时间序列，`target` 形状为 `[D, T]` (D > 1) |

### Pandas 与 Dataset 的对应关系

```
1 个 Pandas DataFrame  =  1 个 Time Series (Sample)
List[DataFrame]        =  Dataset (包含多个 Time Series)
```


### UTS vs MTS 模式

```python
# MTS 模式 (to_univariate=False) - 默认
# 每个 DataFrame 保持为一个多变量时间序列
# DataFrame 有 D 列 → target 形状 [D, T]

df = pd.DataFrame({
    "温度": [...],
    "湿度": [...],
    "风速": [...]
})  # 3列，T行

# 转换后: target 形状 = [3, T]

# -------------------------------------------

# UTS 模式 (to_univariate=True)
# 每个 DataFrame 的每一列拆分成独立的单变量时间序列
# DataFrame 有 D 列 → 生成 D 个 samples

df = pd.DataFrame({
    "温度": [...],
    "湿度": [...],
    "风速": [...]
})  # 3列，T行

# 转换后: 3 个 samples，每个 target 形状 = [T]
```

---

## 1. 数据集存储格式

GIFT-Eval 使用 **HuggingFace Datasets** 的 Arrow 格式存储数据。

### 目录结构

```
dataset_name/
├── data-00000-of-00001.arrow   # 实际数据 (Apache Arrow 格式)
├── dataset_info.json           # 数据集元信息
└── state.json                  # 数据集状态信息
```

---

## 2. 数据集字段说明

每条样本（sample）代表一条时间序列，包含以下字段：

| 字段名 | 数据类型 | 必需 | 说明 |
|--------|----------|------|------|
| `item_id` | `string` | ✅ | 时间序列的唯一标识符 |
| `start` | `timestamp[s]` | ✅ | 时间序列的起始时间戳 |
| `freq` | `string` | ✅ | 采样频率 |
| `target` | `Sequence[Sequence[float32]]` | ✅ | 目标时间序列数据 |
| `past_feat_dynamic_real` | `Sequence[Sequence[float32]]` | ❌ | 历史动态特征（可选） |

### 2.1 `item_id` (必需)

**类型**: `string`

时间序列的唯一标识符，用于区分数据集中的不同序列。

```python
# 示例
"item_id": "sensor_001"
"item_id": "stock_AAPL"
"item_id": "weather_station_42"
```

### 2.2 `start` (必需)

**类型**: `timestamp[s]` (秒级时间戳)

时间序列第一个数据点的时间戳。结合 `freq` 可以推断出整个序列的时间索引。

```python
# 示例
"start": datetime.datetime(2024, 1, 1, 0, 0, 0)
"start": pd.Timestamp("2024-01-01 00:00:00")
```

### 2.3 `freq` (必需)

**类型**: `string`

时间序列的采样频率，使用 Pandas 频率字符串格式。

| 频率字符串 | 含义 |
|-----------|------|
| `"T"` 或 `"min"` | 分钟 |
| `"5T"` | 5 分钟 |
| `"15T"` | 15 分钟 |
| `"H"` | 小时 |
| `"D"` | 天 |
| `"W"` | 周 |
| `"M"` | 月 |
| `"Q"` | 季度 |
| `"Y"` | 年 |


### 2.4 `target` (必需) ⭐

**类型**: `Sequence[Sequence[float32]]`

**形状**: `[num_dimensions, sequence_length]`

这是最重要的字段，存储实际的时间序列数据。

#### 单变量时间序列 (Univariate)

```python
# 形状: [1, T] 或直接 [T]
"target": [[1.2, 3.4, 5.6, 7.8, 9.0, ...]]  # 1维，T个时间步
```

#### 多变量时间序列 (Multivariate)

```python
# 形状: [D, T]，D=维度数，T=时间步数
"target": [
    [1.2, 1.3, 1.4, ...],  # 维度 0：如温度
    [2.5, 2.6, 2.7, ...],  # 维度 1：如湿度
    [3.8, 3.9, 4.0, ...],  # 维度 2：如风速
]
# 形状: [3, T]
```

**重要说明**：
- 第一维是**变量/特征维度**，第二维是**时间维度**
- 数据类型为 `float32`
- 允许包含 `NaN` 值表示缺失数据

### 2.5 `past_feat_dynamic_real` (可选)

**类型**: `Sequence[Sequence[float32]]`

**形状**: `[num_features, sequence_length]`

目前版本的Benchmark不需要添加


---

## 3. 数据转换示例

### 3.1 CSV 文件格式要求

`build_dataset_from_csvs` 函数要求的 CSV 格式：

```
┌─────────────────────────────────────────────────────────┐
│  timestamp (第1列)  │  var_1  │  var_2  │ ... │  var_D  │
├─────────────────────────────────────────────────────────┤
│  2024-01-01 00:00   │  1.23   │  4.56   │ ... │  7.89   │
│  2024-01-01 01:00   │  1.34   │  4.67   │ ... │  7.90   │
│  2024-01-01 02:00   │  1.45   │  4.78   │ ... │  7.91   │
│  ...                │  ...    │  ...    │ ... │  ...    │
└─────────────────────────────────────────────────────────┘
```

| 列 | 要求 | 说明 |
|----|------|------|
| **第1列** | 时间戳 | 必须可被 `pd.to_datetime()` 解析 |
| **第2~N列** | 数值 | 时间序列的各个变量/维度，`float` 类型 |



### 3.2 转换流程

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      build_dataset_from_csvs 流程                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. 扫描目录，按 pattern 匹配 CSV 文件                                      │
│    csv_paths = sorted(csv_dir.glob(pattern))                             │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. 读取所有 CSV 为 DataFrame (第1列解析为时间)                              │
│    dfs = [pd.read_csv(path, parse_dates=[0]) for path in csv_paths]      │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 3. 调用 dataframes_to_generator 创建生成器                                │
│    - 将第1列设为时间索引                                                   │
│    - 推断频率 (pd.infer_freq)                                             │
│    - 根据 to_univariate 参数决定转换模式                                   │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │  MTS 模式 (默认)     │         │  UTS 模式            │
        │  to_univariate=False│         │  to_univariate=True │
        ├─────────────────────┤         ├─────────────────────┤
        │ 1个CSV = 1个Sample  │         │ 1个CSV = D个Sample  │
        │ target: [D, T]      │         │ (D=列数)            │
        │                     │         │ target: [T]         │
        └─────────────────────┘         └─────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 4. 创建 HuggingFace Dataset 并保存                                        │
│    Dataset.from_generator(gen_func, features=features)                   │
│    dataset.save_to_disk(output_path)                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.3 MTS 模式 (多变量时间序列)

**场景**：每个 CSV 文件包含一个多变量时间序列（如一个传感器的多个指标）

```python
from timebench.evaluation.dataset_builder import build_dataset_from_csvs

# 目录结构:
# csv_dir/
# ├── sensor_001.csv  (3列数据 × 1000行) → 1个 Sample, target: [3, 1000]
# ├── sensor_002.csv  (3列数据 × 1200行) → 1个 Sample, target: [3, 1200]
# └── sensor_003.csv  (3列数据 × 800行)  → 1个 Sample, target: [3, 800]

dataset = build_dataset_from_csvs(
    csv_dir="/path/to/csv/files",
    output_path="/path/to/output/dataset",
    pattern="*.csv",
    freq="15T",
    to_univariate=False,  # MTS 模式
    item_prefix="sensor_", # Prefix of each item/sample' name
)
# 结果: Dataset 包含 3 个 samples
```


### 3.4 UTS 模式 (单变量时间序列)

**场景**：将每个变量拆分成独立的单变量时间序列

```python
# 目录结构:
# csv_dir/
# ├── sensor_001.csv  (3列数据) → 3个 Samples (temp, humid, press)
# ├── sensor_002.csv  (3列数据) → 3个 Samples
# └── sensor_003.csv  (3列数据) → 3个 Samples

dataset = build_dataset_from_csvs(
    csv_dir="/path/to/csv/files",
    output_path="/path/to/output/dataset",
    pattern="*.csv",
    freq="15T",
    to_univariate=True,  # UTS 模式
    item_prefix="sensor_",
)
# 结果: Dataset 包含 9 个 samples (3 files × 3 columns)
```


### 3.5 自定义转换

如果 CSV 格式不同（如时间列不在第1列），需手动适配后使用 `dataframes_to_generator` 进行转换。


## 4. 读取数据集

```python
from datasets import load_from_disk
import numpy as np

# 加载数据集
dataset = load_from_disk("/path/to/dataset")

# 查看基本信息
print(f"样本数量: {len(dataset)}")
print(f"字段: {dataset.column_names}")
print(f"Features: {dataset.features}")

# 访问单个样本
sample = dataset[0]
print(f"item_id: {sample['item_id']}")
print(f"start: {sample['start']}")
print(f"freq: {sample['freq']}")
print(f"target shape: {np.array(sample['target']).shape}")

# 批量访问
batch = dataset[:10]  # 前10个样本
targets = [np.array(t) for t in batch['target']]
```

---

## 5. 数据格式验证

使用 `analyze_dataset.py` 脚本验证转换后的数据集：

```bash
python src/timebench/evaluation/analyze_dataset.py
```

---

## 6. 下一步

数据转换完成后，参考 [PRED_CONFIG.md](./PRED_CONFIG.md) 了解如何配置预测任务参数（`prediction_length`, `test_split`, `val_split`）。
