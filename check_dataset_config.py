"""
检查 data/hf_dataset 里的数据集配置是否与 datasets.yaml 中的设定一致。

功能：
1. 遍历检查 data/hf_dataset 里的全部数据集
2. 通过 Dataset 类加载数据集，检查 test_length, val_length, prediction_length (每个 term 一个)
   是否与 datasets.yaml 里的设定一致
3. 支持指定单一的 'dataset/freq' 或 'dataset/freq' 列表进行检查

使用方式：
    # 检查所有数据集
    python check_dataset_config.py

    # 检查指定数据集
    python check_dataset_config.py --datasets "Coastal_T_S/H" "SG_PM25/H"

    # 检查指定数据集（也可以用逗号分隔）
    python check_dataset_config.py --datasets "Coastal_T_S/H,SG_PM25/H"
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add src to path for importing timebench
sys.path.insert(0, str(Path(__file__).parent / "src"))

from timebench.evaluation.data import Dataset, load_dataset_config


def get_all_hf_datasets(hf_root: Path) -> List[str]:
    """Get all dataset/freq paths from hf_dataset directory."""
    datasets = []
    if not hf_root.exists():
        return datasets

    for dataset_dir in hf_root.iterdir():
        if dataset_dir.is_dir():
            for freq_dir in dataset_dir.iterdir():
                if freq_dir.is_dir() and (freq_dir / "dataset_info.json").exists():
                    # Format: dataset_name/freq
                    datasets.append(f"{dataset_dir.name}/{freq_dir.name}")

    return sorted(datasets)


def get_terms_from_config(dataset_config: Dict[str, Any]) -> List[str]:
    """Extract available terms (short, medium, long) from dataset config."""
    terms = []
    for term in ["short", "medium", "long"]:
        if term in dataset_config:
            terms.append(term)
    return terms


def check_dataset(
    dataset_name: str,
    yaml_config: Dict[str, Any],
    hf_root: Path,
    verbose: bool = True
) -> Tuple[bool, List[str]]:
    """
    Check a single dataset's configuration against the YAML settings.

    Loads the Dataset class with YAML config and verifies:
    - _test_length matches YAML test_length
    - _val_length matches YAML val_length
    - prediction_length matches YAML term-specific prediction_length

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    datasets_config = yaml_config.get("datasets", {})

    # Check if dataset exists in YAML config
    if dataset_name not in datasets_config:
        issues.append(f"数据集 '{dataset_name}' 在 datasets.yaml 中未配置")
        return False, issues

    dataset_config = datasets_config[dataset_name]

    # Get YAML configured values
    yaml_test_length = dataset_config.get("test_length")
    yaml_val_length = dataset_config.get("val_length", 0)  # val_length is optional

    if yaml_test_length is None:
        issues.append("YAML 配置缺少 test_length")
        return False, issues

    # Get available terms
    terms = get_terms_from_config(dataset_config)
    if not terms:
        issues.append("YAML 配置没有任何 term (short/medium/long)")
        return False, issues

    if verbose:
        print(f"\n{'='*60}")
        print(f"数据集: {dataset_name}")
        print("=" * 60)
        print("  YAML 配置:")
        print(f"    test_length: {yaml_test_length}")
        print(f"    val_length: {yaml_val_length}")
        print(f"    terms: {terms}")

    # Check each term
    for term in terms:
        yaml_pred_length = dataset_config.get(term, {}).get("prediction_length")

        if yaml_pred_length is None:
            issues.append(f"term '{term}' 在 YAML 中缺少 prediction_length")
            continue

        if verbose:
            print(f"      {term}.prediction_length: {yaml_pred_length}")

        # Load Dataset with YAML config values
        try:
            ds = Dataset(
                name=dataset_name,
                term=term,
                prediction_length=yaml_pred_length,
                test_length=yaml_test_length,
                val_length=yaml_val_length,
                storage_path=hf_root,
            )

            # Verify the Dataset's internal values match YAML config
            actual_test_length = ds._test_length
            actual_val_length = ds._val_length
            actual_pred_length = ds.prediction_length

            term_issues = []

            if actual_test_length != yaml_test_length:
                term_issues.append(
                    f"test_length 不匹配: Dataset={actual_test_length}, YAML={yaml_test_length}"
                )

            if actual_val_length != yaml_val_length:
                term_issues.append(
                    f"val_length 不匹配: Dataset={actual_val_length}, YAML={yaml_val_length}"
                )

            if actual_pred_length != yaml_pred_length:
                term_issues.append(
                    f"prediction_length 不匹配: Dataset={actual_pred_length}, YAML={yaml_pred_length}"
                )

            if term_issues:
                for ti in term_issues:
                    issues.append(f"term '{term}': {ti}")

            if verbose:
                test_match = "✅" if actual_test_length == yaml_test_length else "❌"
                val_match = "✅" if actual_val_length == yaml_val_length else "❌"
                pred_match = "✅" if actual_pred_length == yaml_pred_length else "❌"
                print(f"\n  Term '{term}' 验证结果:")
                print(f"    Dataset._test_length = {actual_test_length} (YAML: {yaml_test_length}) {test_match}")
                print(f"    Dataset._val_length = {actual_val_length} (YAML: {yaml_val_length}) {val_match}")
                print(f"    Dataset.prediction_length = {actual_pred_length} (YAML: {yaml_pred_length}) {pred_match}")
                print(f"    windows = {ds.windows}")
                if yaml_val_length > 0:
                    print(f"    val_windows = {ds.val_windows}")

        except Exception as e:
            issues.append(f"term '{term}': 加载数据集失败 - {str(e)}")
            if verbose:
                print(f"\n  Term '{term}' ❌ 加载失败: {str(e)}")

    if verbose:
        if issues:
            print("\n  ❌ 发现问题:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("\n  ✅ 所有配置验证通过")

    return len(issues) == 0, issues


def check_yaml_coverage(yaml_config: Dict[str, Any], hf_datasets: List[str]) -> List[str]:
    """Check if all YAML configured datasets exist in hf_dataset directory."""
    issues = []
    datasets_config = yaml_config.get("datasets", {})

    for yaml_dataset in datasets_config.keys():
        if yaml_dataset not in hf_datasets:
            issues.append(f"YAML 配置的数据集 '{yaml_dataset}' 在 hf_dataset 目录中不存在")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="检查 hf_dataset 数据集配置是否与 datasets.yaml 一致"
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="*",
        default=None,
        help="指定要检查的数据集，格式: 'dataset/freq'。可以用空格或逗号分隔多个。不指定则检查全部。"
    )
    parser.add_argument(
        "--config-path", "-c",
        type=str,
        default=None,
        help="datasets.yaml 配置文件路径"
    )
    parser.add_argument(
        "--hf-root",
        type=str,
        default=None,
        help="hf_dataset 根目录路径"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="只显示有问题的数据集"
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="只显示摘要统计"
    )

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    config_path = Path(args.config_path) if args.config_path else script_dir / "src" / "timebench" / "config" / "datasets.yaml"
    hf_root = Path(args.hf_root) if args.hf_root else script_dir / "data" / "hf_dataset"

    print(f"配置文件: {config_path}")
    print(f"数据集目录: {hf_root}")

    # Load YAML config
    yaml_config = load_dataset_config(config_path)

    # Get all HF datasets
    all_hf_datasets = get_all_hf_datasets(hf_root)
    print(f"找到 {len(all_hf_datasets)} 个数据集目录")

    # Parse specified datasets
    datasets_to_check = []
    if args.datasets:
        for d in args.datasets:
            # Support comma-separated values
            datasets_to_check.extend([x.strip() for x in d.split(",") if x.strip()])
    else:
        datasets_to_check = all_hf_datasets

    print(f"将检查 {len(datasets_to_check)} 个数据集\n")

    # Check YAML coverage (only for full check)
    if not args.datasets:
        coverage_issues = check_yaml_coverage(yaml_config, all_hf_datasets)
        if coverage_issues:
            print("⚠️  YAML 配置覆盖检查:")
            for issue in coverage_issues:
                print(f"  - {issue}")
            print()

    # Check each dataset
    results = {
        "valid": [],
        "invalid": [],
        "missing_in_yaml": [],
        "missing_in_hf": []
    }

    for dataset_name in datasets_to_check:
        # Check if dataset exists in hf_dataset
        if dataset_name not in all_hf_datasets:
            results["missing_in_hf"].append(dataset_name)
            if not args.summary:
                print(f"\n❌ 数据集 '{dataset_name}' 在 hf_dataset 目录中不存在")
            continue

        verbose = not args.quiet and not args.summary
        is_valid, issues = check_dataset(dataset_name, yaml_config, hf_root, verbose=verbose)

        if is_valid:
            results["valid"].append(dataset_name)
        else:
            if any("未配置" in str(i) for i in issues):
                results["missing_in_yaml"].append(dataset_name)
            else:
                results["invalid"].append((dataset_name, issues))

            # Show issues in quiet mode
            if args.quiet and not args.summary:
                print(f"\n❌ {dataset_name}:")
                for issue in issues:
                    print(f"   - {issue}")

    # Print summary
    print(f"\n{'='*60}")
    print("检查结果摘要")
    print("=" * 60)
    print(f"✅ 配置有效: {len(results['valid'])} 个数据集")
    print(f"❌ 配置无效: {len(results['invalid'])} 个数据集")
    print(f"⚠️  YAML 中未配置: {len(results['missing_in_yaml'])} 个数据集")
    print(f"⚠️  hf_dataset 中不存在: {len(results['missing_in_hf'])} 个数据集")

    if results['missing_in_yaml']:
        print("\nYAML 中未配置的数据集:")
        for d in results['missing_in_yaml']:
            print(f"  - {d}")

    if results['missing_in_hf']:
        print("\nhf_dataset 中不存在的数据集:")
        for d in results['missing_in_hf']:
            print(f"  - {d}")

    if results['invalid']:
        print("\n配置无效的数据集:")
        for d, issues in results['invalid']:
            print(f"  - {d}:")
            for issue in issues:
                print(f"      {issue}")

    # Return exit code based on results
    total_issues = len(results['invalid']) + len(results['missing_in_yaml']) + len(results['missing_in_hf'])
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    exit(main())
