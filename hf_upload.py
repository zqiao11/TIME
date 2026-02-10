import os

from huggingface_hub import HfApi

# ================= 配置区域 =================
REPO_ID = "Real-TSF/TIME"

# 定义上传任务列表
# 格式: ("本地文件夹路径", "上传到仓库后的文件夹路径")
# 建议保持一致，这样仓库里的结构就和本地一样
FOLDERS_TO_UPLOAD = [
    # ("data/hf_dataset/epf_electricity_price", "data/hf_dataset/epf_electricity_price"),       # 将本地 data 文件夹的内容 -> 传到仓库的 /data 目录下
    # ("output", "output")    # 将本地 output 文件夹的内容 -> 传到仓库的 /output 目录下
    # ("output/results", "output/results")
    # ("output/features", "output/features")
    ("data/processed_csv/Oil_Price/B", "data/processed_csv/Oil_Price/B"),
    ("data/processed_csv/Global_Price/Q", "data/processed_csv/Global_Price/Q"),
    ("output/features/Oil_Price/B", "output/features/Oil_Price/B"),
    ("output/features/Global_Price/Q", "output/features/Global_Price/Q"),
]

# 是否分批上传子文件夹（解决大文件夹超时问题）
BATCH_UPLOAD_SUBFOLDERS = False

# ===========================================

IGNORE_PATTERNS = [
    ".DS_Store",    # Mac 系统垃圾文件
    "__pycache__",  # Python 缓存
    ".cache",       # 👈 必须加：忽略 .cache 文件夹
    "download",     # 👈 建议加：看你截图里有 download，通常也是临时文件
    "*.lock",       # 可选：忽略锁文件
]

EXCLUDE_DIRS = ["hparams", "optuna"]

def upload_single_folder(api, local_path, repo_path):
    """上传单个文件夹"""
    api.upload_folder(
        folder_path=local_path,
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=repo_path,
        commit_message=f"Fix csv cloumn name",  # TODO
        ignore_patterns=IGNORE_PATTERNS
    )


def upload_project():
    api = HfApi()

    print(f"🚀 开始向仓库 {REPO_ID} 上传文件...")

    for local_path, repo_path in FOLDERS_TO_UPLOAD:
        # 检查本地文件夹是否存在
        if not os.path.exists(local_path):
            print(f"⚠️  跳过: 找不到本地文件夹 '{local_path}'")
            continue

        if BATCH_UPLOAD_SUBFOLDERS:
            # 分批上传模式：遍历子文件夹，逐个上传
            subfolders = [f for f in os.listdir(local_path) if os.path.isdir(os.path.join(local_path, f))]
            total = len(subfolders)

            print(f"\n📂 分批上传模式: '{local_path}' 下共有 {total} 个子文件夹")

            for idx, subfolder in enumerate(subfolders, 1):
                if subfolder in EXCLUDE_DIRS:
                    print(f" 🚫 [{idx}/{total}] 跳过忽略的文件夹: {subfolder}")
                    continue

                sub_local_path = os.path.join(local_path, subfolder)
                sub_repo_path = os.path.join(repo_path, subfolder)

                print(f"\n  [{idx}/{total}] 正在上传: {subfolder} ...")

                try:
                    upload_single_folder(api, sub_local_path, sub_repo_path)
                    print(f"  ✅ [{idx}/{total}] '{subfolder}' 上传成功！")
                except Exception as e:
                    print(f"  ❌ [{idx}/{total}] '{subfolder}' 上传失败: {e}")
        else:
            # 原有模式：直接上传整个文件夹
            print(f"\n📂 正在处理: 本地 '{local_path}' -> 仓库 '{repo_path}' ...")

            try:
                upload_single_folder(api, local_path, repo_path)
                print(f"✅ 完成: '{local_path}' 上传成功！")
            except Exception as e:
                print(f"❌ 错误: 上传 '{local_path}' 时发生异常:\n{e}")

    print("\n🎉 所有任务处理完毕！")
    print(f"查看你的仓库: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    upload_project()
