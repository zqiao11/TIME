from huggingface_hub import HfApi
import os

# ================= 配置区域 =================
REPO_ID = "TIME-benchmark/TIME-1.0"

# 定义上传任务列表
# 格式: ("本地文件夹路径", "上传到仓库后的文件夹路径")
# 建议保持一致，这样仓库里的结构就和本地一样
FOLDERS_TO_UPLOAD = [
    ("data", "data"),       # 将本地 data 文件夹的内容 -> 传到仓库的 /data 目录下
    ("output", "output")    # 将本地 output 文件夹的内容 -> 传到仓库的 /output 目录下
]
# ===========================================

def upload_project():
    api = HfApi()

    print(f"🚀 开始向仓库 {REPO_ID} 上传文件...")

    for local_path, repo_path in FOLDERS_TO_UPLOAD:
        # 检查本地文件夹是否存在
        if not os.path.exists(local_path):
            print(f"⚠️  跳过: 找不到本地文件夹 '{local_path}'")
            continue

        print(f"\n📂 正在处理: 本地 '{local_path}' -> 仓库 '{repo_path}' ...")

        try:
            api.upload_folder(
                folder_path=local_path,
                repo_id=REPO_ID,
                repo_type="dataset",  # 必须指定 dataset

                # 【关键点】path_in_repo 指定了文件在仓库里的存放位置
                # 如果不写这个，文件夹里的内容会直接散落在仓库根目录
                path_in_repo=repo_path,

                commit_message=f"Upload {repo_path} folder",
                ignore_patterns=[".DS_Store", "__pycache__"] # 可选：忽略系统垃圾文件
            )
            print(f"✅ 完成: '{local_path}' 上传成功！")

        except Exception as e:
            print(f"❌ 错误: 上传 '{local_path}' 时发生异常:\n{e}")

    print("\n🎉 所有任务处理完毕！")
    print(f"查看你的仓库: https://huggingface.co/datasets/{REPO_ID}")
    # 也可以直接看文件树: https://huggingface.co/datasets/{REPO_ID}/tree/main

if __name__ == "__main__":
    upload_project()