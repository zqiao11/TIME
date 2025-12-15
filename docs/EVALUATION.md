# 评估模块使用指南

本文档介绍如何使用 TIME 框架的评估模块进行时序预测评估。

---

## 1. 数据集准备

评估使用的数据集需要按照 [DATA_FORMAT.md](DATA_FORMAT.md) 所示流程进行预处理。预处理后的数据集应存储在环境变量 `GIFT_EVAL` 指定的路径下，格式为 HuggingFace Datasets 格式。

---

## 2. 验证集与测试集模式

TIME 支持两种评估模式：

### 2.1 测试集模式（默认）

- **用途**: 最终模型评估，保存所有结果
- **数据**: 使用 `dataset.test_data`
- **行为**: 生成预测、计算指标、保存结果到文件

```bash
python experiments/moirai.py --dataset "IMOS/15T" --terms short
```

### 2.2 验证集模式

- **用途**: 超参数选择，不保存结果
- **数据**: 使用 `dataset.val_data`
- **行为**: 生成预测、计算指标、仅打印摘要到控制台

```bash
python experiments/moirai.py --dataset "IMOS/15T" --terms short --val
```

### 2.3 数据构建方式

#### 窗口数计算

窗口数由 `test_split`/`val_split` 和 `prediction_length` 自动计算：

```python
# 测试集窗口数
windows = ceil(test_split * min_series_length / prediction_length)

# 验证集窗口数
val_windows = ceil(val_split * min_series_length / prediction_length)
```

其中 `min_series_length` 是数据集中最短序列的长度。

**最后一个窗口的处理**:

当使用 `ceil()` 向上取整时，最后一个窗口可能会超出实际可用的测试/验证集长度。例如：

- 实际测试集长度 = `test_split * min_series_length = 100` 个时间步
- `prediction_length = 48`
- `windows = ceil(100 / 48) = 3`
- 总窗口长度 = `3 * 48 = 144` 个时间步
- 超出长度 = `144 - 100 = 44` 个时间步

在这种情况下：

- **预测时**: 模型仍然使用完整的 `prediction_length` 进行预测，生成长度为 `prediction_length` 的预测序列
- **Ground truth**: GluonTS 的 `generate_instances` 会自动截断到实际可用长度，最后一个窗口的 ground truth 实际长度可能小于 `prediction_length`
- **指标计算**: 计算指标时，只使用 ground truth 的有效部分（实际长度）与对应的预测部分进行比较，超出部分会被忽略。例如，如果最后一个窗口的 ground truth 只有 4 个时间步，则只计算这 4 个时间步的指标，预测序列的前 4 个值与 ground truth 进行比较

#### 数据分割

- **训练集**: 从开始到 `-(windows + val_windows) * prediction_length`
- **验证集**: 从 `-windows * prediction_length` 到 `-val_windows * prediction_length`
- **测试集**: 最后 `windows * prediction_length` 个时间步

#### 窗口生成

`test_data` 和 `val_data` 都使用相同的生成方式。
每个窗口包含：

- **Context**: 历史数据（用于预测）
- **Target**: 未来 `prediction_length` 个时间步（用于评估）

窗口之间不重叠，每个窗口间隔 `prediction_length` 个时间步。

---

## 3. 自定义模型评估

要为自己的模型创建评估脚本，只需要修改模型初始化部分，其他流程保持不变。

参考 `experiments/moirai.py` 的模型初始化部分（第 78-89 行）：

```python
if output_dir is None:
        output_dir = f"./output/results/moirai_{model_size}"

    os.makedirs(output_dir, exist_ok=True)

# Initialize model
print(f"Loading Moirai-{model_size} model...")
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{model_size}"),
    prediction_length=1,  # Will be updated per dataset
    context_length=context_length,
    patch_size=32,
    num_samples=num_samples,
    target_dim=1,  # Will be updated per dataset
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
)
```

**替换为你的模型的output_dir和初始化代码即可**，其他部分（数据加载、预测生成、指标计算、结果保存）都保持不变。

---

## 4. 输出结果格式

### 4.1 测试集评估输出

测试集评估结果保存在以下目录结构：

```text
output_dir/
  {dataset_name}_{term}/
    predictions.npz      # 预测结果和真实值
    metrics.npz           # 所有指标
    metadata.json         # 元数据信息
```

#### predictions.npz

包含以下数组：

- `predictions_mean`: `(num_series, num_windows, num_variates, prediction_length)` - 预测的均值
- `predictions_samples`: `(num_series, num_windows, num_samples, num_variates, prediction_length)` - 概率预测的采样样本
- `ground_truth`: `(num_series, num_windows, num_variates, prediction_length)` - 真实值
- `context`: `(num_series, num_windows, num_variates, max_context_length)` - 历史上下文（较短上下文用 NaN 填充）

#### metrics.npz

包含所有评估指标，每个指标数组形状为 `(num_series, num_windows, num_variates)`：

- `MSE`: Mean Squared Error
- `MAE`: Mean Absolute Error
- `RMSE`: Root Mean Squared Error
- `MAPE`: Mean Absolute Percentage Error
- `sMAPE`: Symmetric MAPE
- `MASE`: Mean Absolute Scaled Error
- `ND`: Normalized Deviation
- `CRPS`: Continuous Ranked Probability Score
- `QuantileLoss_0.1`, `QuantileLoss_0.5`, `QuantileLoss_0.9`: 分位数损失

#### metadata.json

包含数据集和实验的元信息：

```json
{
  "dataset_config": "IMOS_15T/short",
  "num_series": 100,
  "num_windows": 3,
  "num_variates": 1,
  "prediction_length": 16,
  "num_samples": 100,
  "freq": "15T",
  "seasonality": 96,
  "max_context_length": 1000,
  "shapes": {
    "predictions_mean": [100, 3, 1, 16],
    "predictions_samples": [100, 3, 100, 1, 16],
    "ground_truth": [100, 3, 1, 16],
    "context": [100, 3, 1, 1000]
  },
  "metric_names": ["MSE", "MAE", "RMSE", ...],
  "metric_shape": "(num_series, num_windows, num_variates)"
}
```

### 4.2 验证集评估输出

验证集评估不保存任何文件，只打印指标摘要到控制台：

```text
    Metrics summary (averaged over all series/windows/variates):
      MSE: 0.1234
      MAE: 0.2345
      RMSE: 0.3456
      ...
    (No results saved - validation data used for hyperparameter selection)
```

---

## 5. 使用示例

### 加载和分析结果

```python
import numpy as np
import json

# 加载预测结果
data = np.load('./results/moirai_small/IMOS_15T/short/predictions.npz')
predictions_mean = data['predictions_mean']
ground_truth = data['ground_truth']

# 加载指标
metrics = np.load('./results/moirai_small/IMOS_15T/short/metrics.npz')
mse = metrics['MSE']
mae = metrics['MAE']

# 计算平均指标
mean_mse = np.nanmean(mse)
mean_mae = np.nanmean(mae)
```

---

## 6. 相关文档

- [数据格式说明](DATA_FORMAT.md): 数据集准备流程
- [预测任务配置指南](PRED_CONFIG.md): 如何配置 `prediction_length` 和 `test_split`
