#!/usr/bin/env python3
"""
检查 output/results 中每个模型是否运行了 datasets.yaml 中定义的全部实验。
"""

import os
import yaml
from pathlib import Path
from collections import defaultdict


def load_expected_experiments(yaml_path: str) -> set:
    """
    从 datasets.yaml 加载所有预期的 (dataset, freq, term) 组合。
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    experiments = set()
    datasets = config.get('datasets', {})

    for dataset_freq, settings in datasets.items():
        # dataset_freq 格式为 "dataset_name/freq"
        parts = dataset_freq.split('/')
        if len(parts) != 2:
            print(f"警告: 无法解析 '{dataset_freq}'，跳过")
            continue

        dataset_name, freq = parts

        # 获取所有 term (short, medium, long)
        for term in ['short', 'medium', 'long']:
            if term in settings:
                experiments.add((dataset_name, freq, term))

    return experiments


def get_completed_experiments(model_path: Path) -> set:
    """
    获取模型已完成的实验 (dataset, freq, term) 组合。
    检查是否存在 results.json 或 metrics.json 文件。
    """
    completed = set()

    if not model_path.exists():
        return completed

    # 遍历 dataset 目录
    for dataset_dir in model_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name

        # 遍历 freq 目录
        for freq_dir in dataset_dir.iterdir():
            if not freq_dir.is_dir():
                continue
            freq = freq_dir.name

            # 遍历 term 目录
            for term_dir in freq_dir.iterdir():
                if not term_dir.is_dir():
                    continue
                term = term_dir.name

                # 检查是否有结果文件（results.json 或 metrics.json）
                results_file = term_dir / 'results.json'
                metrics_file = term_dir / 'metrics.json'

                if results_file.exists() or metrics_file.exists():
                    completed.add((dataset_name, freq, term))
                else:
                    # 如果目录存在但没有结果文件，也认为完成（可能是其他格式）
                    # 检查目录是否有任何文件
                    has_files = any(term_dir.iterdir())
                    if has_files:
                        completed.add((dataset_name, freq, term))

    return completed


def main():
    # 路径设置
    base_dir = Path(__file__).parent
    yaml_path = base_dir / 'src/timebench/config/datasets.yaml'
    results_dir = base_dir / 'output/results'

    # 加载预期实验
    print("=" * 70)
    print("检查实验完成状态")
    print("=" * 70)

    expected = load_expected_experiments(yaml_path)
    print(f"\n📋 datasets.yaml 中定义的实验总数: {len(expected)}")

    # 按 dataset/freq 分组显示
    by_dataset_freq = defaultdict(list)
    for dataset, freq, term in sorted(expected):
        by_dataset_freq[(dataset, freq)].append(term)

    print(f"   涵盖 {len(by_dataset_freq)} 个 dataset/freq 组合")

    # 获取所有模型
    if not results_dir.exists():
        print(f"\n❌ 结果目录不存在: {results_dir}")
        return

    models = sorted([d.name for d in results_dir.iterdir() if d.is_dir()])
    print(f"\n🤖 发现 {len(models)} 个模型: {', '.join(models)}")

    # 检查每个模型
    print("\n" + "=" * 70)
    print("各模型完成状态")
    print("=" * 70)

    all_complete = []
    incomplete = []

    for model in models:
        model_path = results_dir / model
        completed = get_completed_experiments(model_path)
        missing = expected - completed

        completion_rate = len(completed) / len(expected) * 100 if expected else 0

        if not missing:
            all_complete.append(model)
            print(f"\n✅ {model}: {len(completed)}/{len(expected)} ({completion_rate:.1f}%) - 全部完成!")
        else:
            incomplete.append((model, missing))
            print(f"\n❌ {model}: {len(completed)}/{len(expected)} ({completion_rate:.1f}%) - 缺少 {len(missing)} 个实验")

            # 按 dataset/freq 分组显示缺失的实验
            missing_by_df = defaultdict(list)
            for dataset, freq, term in sorted(missing):
                missing_by_df[(dataset, freq)].append(term)

            for (dataset, freq), terms in sorted(missing_by_df.items()):
                print(f"   - {dataset}/{freq}: {', '.join(sorted(terms))}")

    # 汇总
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"✅ 完成全部实验的模型 ({len(all_complete)}): {', '.join(all_complete) if all_complete else '无'}")
    print(f"❌ 有缺失实验的模型 ({len(incomplete)}): {', '.join([m for m, _ in incomplete]) if incomplete else '无'}")

    # 如果需要，生成缺失实验的详细报告
    if incomplete:
        print("\n" + "=" * 70)
        print("缺失实验详细列表")
        print("=" * 70)
        for model, missing in incomplete:
            print(f"\n### {model} ###")
            for dataset, freq, term in sorted(missing):
                print(f"  {dataset}/{freq}/{term}")


if __name__ == '__main__':
    main()
