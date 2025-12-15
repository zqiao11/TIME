# 预测任务配置指南

本文档说明如何配置预测任务参数（`prediction_length`、`test_split`）以及如何使用 `Dataset` 类进行实验。

---

## 1. 设计理念

### 为什么使用 test_split 而不是 windows？

```
序列: [===================|==========]
                          ↑
                     固定的 split 点 (如 90%)

short:  [---train---][w1][w2][w3]...[wN]  (多个小窗口)
long:   [---train---][====w1====][==w2==]  (少数大窗口)
```

- **相同的 test_split** = 不同 term 在 **相同的测试区间** 上评估
- **windows 自动计算** = `ceil(test_split × min_series_length / prediction_length)`
- **公平比较**：不同 term 的模型使用相同的训练数据，只是预测长度不同

---

## 2. 配置文件结构

`config/datasets.yaml`:

```yaml
datasets:
  # 数据集名称 (与存储路径一致)
  bitbrains_rnd/5T:
    test_split: 0.1              # 测试集占比 (所有 term 共享)
    short:
      prediction_length: 48      # 48 × 5min = 4 hours
    medium:
      prediction_length: 288     # 288 × 5min = 24 hours
    long:
      prediction_length: 576     # 576 × 5min = 48 hours

  TSBench_IMOS_v2/15T:
    test_split: 0.1
    short:
      prediction_length: 96      # 96 × 15min = 24 hours
    medium:
      prediction_length: 672     # 7 days
    long:
      prediction_length: 2880    # 30 days
```

### 参数说明

| 参数 | 层级 | 说明 |
|------|------|------|
| `test_split` | 数据集级别 | 测试集占比 (0~1)，所有 term 共享 |
| `prediction_length` | term 级别 | 预测长度（时间步数） |
| `windows` | 自动计算 | 无需手动设置 |

---

## 3. 读取配置并初始化 Dataset

```python
from timebench.evaluation.data import Dataset, load_dataset_config, get_dataset_settings

# 1. 加载配置文件
config = load_dataset_config()  # 默认路径: config/datasets.yaml

# 2. 获取特定数据集和 term 的设置
dataset_name = "TSBench_IMOS_v2/15T"
term = "short"
settings = get_dataset_settings(dataset_name, term, config)

print(settings)
# {'prediction_length': 96, 'test_split': 0.1}

# 3. 初始化 Dataset
dataset = Dataset(
    name=dataset_name,
    term=term,
    prediction_length=settings["prediction_length"],
    test_split=settings["test_split"],
)

# windows 会自动计算
print(f"prediction_length: {dataset.prediction_length}")
print(f"test_split: {dataset.test_split}")
print(f"windows: {dataset.windows}")  # 自动计算
```

---

## 4. Dataset 类参数

```python
class Dataset:
    def __init__(
        self,
        name: str,                          # 数据集名称
        term: str = "short",                # 预测期限
        to_univariate: bool = False,        # 是否转为单变量
        prediction_length: int = None,      # 预测长度 (显式传入)
        test_split: float = None,           # 测试集占比 (默认 0.1)
        storage_env_var: str = "GIFT_EVAL", # 数据存储路径环境变量
    )
```

### 参数优先级

```
显式传入 > 默认值
```

| 参数 | 显式传入 | 未传入 |
|------|---------|--------|
| `prediction_length` | 使用传入值 | 根据 freq 和 term 自动计算 |
| `test_split` | 使用传入值 | 默认 0.1 |
| `windows` | **自动计算** | `ceil(test_split × min_len / pred_len)` |

### windows 计算公式

```python
windows = min(max(1, ceil(test_split * min_series_length / prediction_length)), 20)
```

**示例**：假设 `min_series_length = 10000`，`test_split = 0.1`

| term | prediction_length | windows |
|------|-------------------|---------|
| short | 48 | ceil(0.1 × 10000 / 48) = 21 → **20** (上限) |
| medium | 480 | ceil(0.1 × 10000 / 480) = **3** |
| long | 720 | ceil(0.1 × 10000 / 720) = **2** |

所有 term 的 `test_split × min_series_length = 1000` 相同，确保在相同测试区间评估。

---

## 5. 完整实验示例

```python
from timebench.evaluation.data import Dataset, load_dataset_config, get_dataset_settings

# ========== 配置 ==========
DATASET_NAME = "TSBench_IMOS_v2/15T"
TERM = "short"

# ========== 读取配置 ==========
config = load_dataset_config()
settings = get_dataset_settings(DATASET_NAME, TERM, config)

# ========== 初始化数据集 ==========
dataset = Dataset(
    name=DATASET_NAME,
    term=TERM,
    prediction_length=settings["prediction_length"],
    test_split=settings["test_split"],
)

# ========== 查看数据集信息 ==========
print(f"Dataset: {dataset.name}")
print(f"Frequency: {dataset.freq}")
print(f"Target dimensions: {dataset.target_dim}")
print(f"Prediction length: {dataset.prediction_length}")
print(f"Test split: {dataset.test_split}")
print(f"Test windows: {dataset.windows}")  # 自动计算

# ========== 获取数据 ==========
train_data = dataset.training_dataset
val_data = dataset.validation_dataset
test_data = dataset.test_data
```

---

## 6. 批量运行多个配置

```python
from timebench.evaluation.data import Dataset, load_dataset_config, get_dataset_settings

config = load_dataset_config()

# 定义要运行的实验
experiments = [
    ("TSBench_IMOS_v2/15T", "short"),
    ("TSBench_IMOS_v2/15T", "medium"),
    ("TSBench_IMOS_v2/15T", "long"),
]

for dataset_name, term in experiments:
    settings = get_dataset_settings(dataset_name, term, config)

    dataset = Dataset(
        name=dataset_name,
        term=term,
        prediction_length=settings["prediction_length"],
        test_split=settings["test_split"],
    )

    print(f"\n=== {dataset_name} ({term}) ===")
    print(f"  pred_len: {dataset.prediction_length}")
    print(f"  test_split: {dataset.test_split}")
    print(f"  windows: {dataset.windows}")  # 自动计算，不同 term 不同

    # 运行你的模型...
```

---

## 7. 添加新数据集配置

在 `config/datasets.yaml` 中添加：

```yaml
datasets:
  # ... 已有配置 ...

  # 添加新数据集
  my_new_dataset/H:
    test_split: 0.15             # 该数据集使用 15% 作为测试集
    short:
      prediction_length: 24      # 24 hours
    medium:
      prediction_length: 168     # 7 days
    long:
      prediction_length: 720     # 30 days
```

---

## 8. 默认计算逻辑

如果配置文件中未设置某个数据集，将使用默认计算：

**prediction_length**：

```python
PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
term_multipliers = {"short": 1, "medium": 10, "long": 15}

prediction_length = PRED_LENGTH_MAP[freq] * term_multipliers[term]
```

**test_split**：默认 `0.1` (10%)
