from huggingface_hub import HfApi

# ================= 配置区域 =================
REPO_ID = "TIME-benchmark/TIME-1.0"

# 定义删除任务列表
# 格式: 只需要填写 "仓库里的文件夹路径" (字符串列表)
FOLDERS_TO_DELETE = [
    "data/hf_dataset",      # 将删除仓库根目录下的 /data 文件夹
    # "output/results"
]
# ===========================================

def delete_project_folders():
    api = HfApi()

    print(f"🚀 开始从仓库 {REPO_ID} 删除文件夹...")

    for repo_path in FOLDERS_TO_DELETE:
        print(f"\n🗑️  正在处理: 删除仓库路径 '{repo_path}' ...")

        try:
            # 执行删除操作
            api.delete_folder(
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="dataset",  # 必须指定 dataset
                commit_message=f"Delete folder {repo_path}"
            )
            print(f"✅ 完成: '{repo_path}' 删除成功！")

        except Exception as e:
            # 常见的错误可能是文件夹不存在，或者网络问题
            print(f"❌ 错误: 删除 '{repo_path}' 时发生异常 (可能文件夹已不存在):\n{e}")

    print("\n🎉 所有删除任务处理完毕！")
    print(f"检查你的仓库: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    # 二次确认，防止误删 (可选)
    confirm = input(f"⚠️  警告: 你即将从 {REPO_ID} 删除 {FOLDERS_TO_DELETE}。\n操作不可逆！确认请输入 'y': ")
    if confirm.lower() == 'y':
        delete_project_folders()
    else:
        print("操作已取消。")