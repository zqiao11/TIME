# Time Series Features Extraction 模块文档

## 概述

`features_runner.py` 模块用于从预处理后的时间序列数据中提取统计特征和时序特征。这些特征可用于：
- 数据集分析和可视化
- 模型选择和超参数调优
- 时间序列分类和聚类

## 数据流

### 输入数据格式

**输入位置**：`./data/processed_csv/{dataset}/{freq}/*.csv`

**格式要求**：
- 第一列：`timestamp`（时间戳，datetime 格式）
- 其他列：变量值（variate），可能包含特性标记：
  - `[rw]`：随机游走标记（random walk）
  - `[sp]`：尖峰存在标记（spike presence）
  - 示例：`CNDC[rw]`, `TEMP[rw]`, `DOX2`

**示例**：
```csv
timestamp,CNDC[rw],DOX2,PSAL[rw],TEMP[rw],TURB[rw],CPHL
2015-08-04 08:15:00,5.379663,202.457763,34.833,25.911167,3.016,0.62
2015-08-04 08:30:00,5.375606,201.869071,34.832,25.873860,3.039,0.594
...
```

### 输出数据格式

**输出位置**：`./output/features/{dataset}/{freq}/{split_mode}.csv`

**格式**：CSV 文件，每行代表一个时间序列（一个 series 的一个 variate）

**`split_mode` 参数说明**：
- `test`：只在测试集（test split）上计算特征
- `train`：只在训练集（train split，包含 val）上计算特征
- `full`：在完整数据上计算特征

**重要**：所有特征（包括标准化的 mean/std）都是在**指定的 split 上**计算的。例如：
- 如果 `split_mode="test"`，则 mean/std 和所有特征都在 test split 上计算
- 如果 `split_mode="train"`，则 mean/std 和所有特征都在 train split 上计算
- 如果 `split_mode="full"`，则 mean/std 和所有特征都在完整数据上计算

**列说明**：

| 列名 | 类型 | 说明 | 来源 |
|------|------|------|------|
| `unique_id` | string | 序列标识，格式：`{series_name}_{variate_name}` | 从文件名和列名生成 |
| `is_random_walk` | int | 是否为随机游走（0/1） | 从 `[rw]` 标记提取 |
| `has_spike_presence` | int | 是否有尖峰特性（0/1） | 从 `[sp]` 标记提取 |
| `trend_strength` | float | 趋势强度 [0, 1] | STL 分解 |
| `trend_stability` | float | 趋势稳定性 | STL 分解 |
| `trend_hurst` | float | 趋势 Hurst 指数 | STL 分解 |
| `seasonal_strength` | float | 季节性强度 [0, 1] | STL 分解 |
| `seasonal_corr` | float | 季节性相关性 | STL 分解 |
| `seasonal_lumpiness` | float | 季节性块状性 | STL 分解 |
| `e_acf1` | float | 残差一阶自相关 | STL 残差 |
| `e_acf10` | float | 残差十阶自相关和 | STL 残差 |
| `e_entropy` | float | 残差熵 | STL 残差 |
| `mean` | float | 原始均值 | 统计信息 |
| `std` | float | 原始标准差 | 统计信息 |
| `missing_rate` | float | 缺失率 [0, 1] | 统计信息 |
| `length` | int | 序列长度 | 统计信息 |
| `period1/2/3` | int | FFT 检测的前 3 个主要周期 | FFT 分析 |
| `p_strength1/2/3` | float | 对应周期的强度 [0, 1] | FFT 分析 |

**示例输出**：
```csv
unique_id,is_random_walk,has_spike_presence,trend_strength,...,mean,std,...
item_0_CNDC[rw],1,0,0.802521,...,5.772001,0.051031,...
item_0_CPHL,0,0,0.989745,...,0.735179,0.222357,...
```

## 数据预处理

### 标准化（Standardization）

**方法**：均值-标准差标准化（Z-score normalization）

**公式**：
```
y_scaled = (y - mean) / std
```

其中：
- `mean = nanmean(y)`：序列的均值（忽略 NaN）
- `std = nanstd(y, ddof=1)`：序列的标准差（样本标准差，ddof=1）
- **在每个时间序列内部独立进行**（按 `unique_id` 分组）
- mean 和 std 是在**指定的 split 上**计算的（由 `split_mode` 参数决定）

### 插值（Interpolation）

**方法**：线性插值（linear interpolation）

**处理逻辑**：
1. 先进行标准化
2. 如果标准化后的序列仍有 NaN 值，使用线性插值填充
3. 插值方向：双向（`limit_direction="both"`），即前向填充和后向填充

**注意**：插值只在标准化后的数据上进行，确保插值后的值在合理的范围内。

### 重要说明

⚠️ **标准化和插值都是临时处理，不会影响原始数据**

- 标准化和插值仅在**计算特征时**在内存中进行
- **不会修改**输入的 CSV 文件
- **不会保存**标准化后的数据
- 后续的训练/评估阶段，输入的数据仍然是**原始的、未标准化的**数据


## 使用方法

### 基本用法

```bash
# 处理单个数据集（默认使用 test split）
python -m timebench.feature.features_runner --dataset IMOS/15T

# 指定 split 模式
python -m timebench.feature.features_runner --dataset IMOS/15T --split full

# 处理所有在 datasets.yaml 中配置的数据集
python -m timebench.feature.features_runner --all
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | string | None | 数据集 key，格式：`{name}/{freq}`（如 `IMOS/15T`） |
| `--all` | flag | False | 处理所有在配置文件中定义的数据集 |
| `--split` | string | `test` | 数据分割模式：`full`（全部）、`test`（测试集）、`train`（训练集） |
| `--config` | string | `./config/datasets.yaml` | 数据集配置文件路径 |
| `--csv_dir` | string | `./data/processed_csv` | 预处理后的 CSV 文件根目录 |
| `--output_dir` | string | `./output` | 特征输出根目录 |

### 配置文件要求

`datasets.yaml` 中需要包含数据集的 `test_split` 配置：

```yaml
datasets:
  IMOS/15T:
    test_split: 0.1
    short:
      prediction_length: 16
    medium:
      prediction_length: 96
    long:
      prediction_length: 672
```

## 特征说明

### Meta 特征（从预处理标记提取）

- **is_random_walk**：从 `unique_id` 中的 `[rw]` 标记提取，表示该序列是否为随机游走
- **has_spike_presence**：从 `unique_id` 中的 `[sp]` 标记提取，表示该序列是否具有尖峰特性

### Trend 特征（趋势特征）

通过 STL 分解提取趋势成分（trend），然后计算：
- **trend_strength**：趋势强度，衡量趋势在总方差中的占比 [0, 1]
- **trend_stability**：趋势稳定性，衡量趋势的平滑程度
- **trend_hurst**：趋势的 Hurst 指数，衡量趋势的长期记忆性

### Seasonal 特征（季节性特征）

通过 STL 分解提取季节性成分（seasonal），然后计算：
- **seasonal_strength**：季节性强度，衡量季节性在总方差中的占比 [0, 1]
- **seasonal_corr**：季节性相关性，衡量不同周期之间的相关性
- **seasonal_lumpiness**：季节性块状性，衡量季节性的不均匀程度

### Residual 特征（残差特征）

对 STL 分解后的残差（remainder）计算：
- **e_acf1**：残差的一阶自相关系数
- **e_acf10**：残差的前十阶自相关系数的和
- **e_entropy**：残差的熵，衡量残差的随机性

### 统计特征

- **mean, std**：原始序列的均值和标准差（**未标准化**）
- **missing_rate**：缺失率
- **length**：序列长度
- **period1/2/3**：通过 FFT 检测到的前 3 个主要周期
- **p_strength1/2/3**：对应周期的强度（归一化后的功率谱密度）

## 注意事项

1. **数据分割**：特征提取时使用的 `test_split` 来自 `datasets.yaml` 配置，确保与后续训练/评估时的分割一致

2. **内存使用**：对于大型数据集，所有 CSV 文件会被加载到内存中。如果内存不足，考虑分批处理

3. **运行时间**：主要时间消耗在 STL 分解上，对于每个序列大约需要 0.1-0.5 秒


## 相关文档

- [PREPROCESS.md](./PREPROCESS.md)：数据预处理模块文档
- [PRED_CONFIG.md](./PRED_CONFIG.md)：预测配置文档
- [DATA_FORMAT.md](./DATA_FORMAT.md)：数据格式说明

