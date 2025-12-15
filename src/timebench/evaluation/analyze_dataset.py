"""
分析GIFT-Eval数据集格式的脚本
"""

import numpy as np
from datasets import load_from_disk

# 加载数据集
dataset_path = "/home/zhongzheng/datasets/GIFT-Eval/TSBench_IMOS_v2/15T"
print(f"正在加载数据集: {dataset_path}")
dataset = load_from_disk(dataset_path)

print("\n" + "="*60)
print("1. 数据集基本信息")
print("="*60)
print(f"数据集类型: {type(dataset)}")
print(f"样本数量: {len(dataset)}")
print(f"列名 (features): {dataset.column_names}")
print(f"\n数据集Features定义:")
print(dataset.features)

print("\n" + "="*60)
print("2. 查看第一条样本的详细内容")
print("="*60)
sample = dataset[0]
for key, value in sample.items():
    print(f"\n字段: {key}")
    print(f"  类型: {type(value)}")
    if isinstance(value, (list, np.ndarray)):
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (list, np.ndarray)):
            # 嵌套列表/数组
            print(f"  外层长度 (维度数): {len(value)}")
            print(f"  内层长度 (时间步数): {len(value[0])}")
            print(f"  数据类型: {type(value[0][0]) if len(value[0]) > 0 else 'N/A'}")
            print(f"  前3个时间步的值: {[v[:3] for v in value]}")
        else:
            print(f"  长度: {len(value)}")
            print(f"  前5个值: {value[:5] if len(value) > 5 else value}")
    else:
        print(f"  值: {value}")

print("\n" + "="*60)
print("3. target字段详细分析 (时间序列主要数据)")
print("="*60)
target = sample['target']
target_array = np.array(target)
print(f"target 形状: {target_array.shape}")
print(f"  - 维度数 (变量数): {target_array.shape[0]}")
print(f"  - 时间步长: {target_array.shape[1]}")
print(f"数据类型: {target_array.dtype}")
print(f"统计信息:")
for i, dim in enumerate(target_array):
    print(f"  维度{i}: min={dim.min():.4f}, max={dim.max():.4f}, mean={dim.mean():.4f}, std={dim.std():.4f}")

print("\n" + "="*60)
print("4. past_feat_dynamic_real字段分析 (动态特征)")
print("="*60)
if 'past_feat_dynamic_real' in sample:
    past_feat = sample['past_feat_dynamic_real']
    past_feat_array = np.array(past_feat)
    print(f"past_feat_dynamic_real 形状: {past_feat_array.shape}")
    print(f"  - 特征数: {past_feat_array.shape[0]}")
    print(f"  - 时间步长: {past_feat_array.shape[1]}")
else:
    print("该数据集不包含 past_feat_dynamic_real 字段")

print("\n" + "="*60)
print("5. 检查多条样本的一致性")
print("="*60)
# 检查前10条样本的target形状
has_past_feat = 'past_feat_dynamic_real' in dataset.column_names
print("前10条样本的target形状:")
for i in range(min(10, len(dataset))):
    target_shape = np.array(dataset[i]['target']).shape
    info = f"  样本{i}: item_id={dataset[i]['item_id']}, target={target_shape}"
    if has_past_feat:
        past_feat_shape = np.array(dataset[i]['past_feat_dynamic_real']).shape
        info += f", past_feat={past_feat_shape}"
    info += f", freq={dataset[i]['freq']}"
    print(info)

print("\n" + "="*60)
print("6. 所有样本的统计汇总")
print("="*60)
# 统计所有样本的时间序列长度
target_lengths = []
for i in range(len(dataset)):
    target_lengths.append(len(dataset[i]['target'][0]))

target_lengths = np.array(target_lengths)
print(f"时间序列长度统计:")
print(f"  最小: {target_lengths.min()}")
print(f"  最大: {target_lengths.max()}")
print(f"  平均: {target_lengths.mean():.2f}")
print(f"  是否等长: {target_lengths.min() == target_lengths.max()}")

# 收集所有item_id
item_ids = [dataset[i]['item_id'] for i in range(len(dataset))]
unique_ids = set(item_ids)
print(f"\n唯一item_id数量: {len(unique_ids)}")
print(f"示例item_ids (前5个): {item_ids[:5]}")

# 收集所有freq
freqs = [dataset[i]['freq'] for i in range(len(dataset))]
unique_freqs = set(freqs)
print(f"\n频率类型: {unique_freqs}")

# 收集所有start时间
starts = [dataset[i]['start'] for i in range(len(dataset))]
print(f"\n起始时间示例 (前5个): {starts[:5]}")
print(f"起始时间类型: {type(starts[0])}")

print("\n" + "="*60)
print("7. 数据结构总结")
print("="*60)
target_shape = np.array(dataset[0]['target']).shape
summary = f"""
GIFT-Eval数据集结构:
---------------------
格式: HuggingFace Datasets (Arrow格式)

该数据集包含的字段: {dataset.column_names}

字段说明:
1. item_id (str): 时间序列的唯一标识符
2. start (timestamp): 时间序列的起始时间戳
3. target (list[list[float]]): 主要时间序列数据
   - 形状: [num_dimensions, sequence_length]
   - 本数据集: [{target_shape[0]}, {target_shape[1]}]
4. freq (str): 时间频率 (如 "5T" 表示5分钟)"""

if has_past_feat:
    past_feat_shape = np.array(dataset[0]['past_feat_dynamic_real']).shape
    summary += f"""
5. past_feat_dynamic_real (list[list[float]]): 过去的动态特征
   - 形状: [num_features, sequence_length]
   - 本数据集: [{past_feat_shape[0]}, {past_feat_shape[1]}]"""
else:
    summary += """
(注: 该数据集不包含 past_feat_dynamic_real 字段)"""

summary += """

转换自己数据时需要的格式:
- 创建一个包含上述字段的字典列表
- 使用 datasets.Dataset.from_dict() 或类似方法创建数据集
- 保存为Arrow格式
"""
print(summary)

