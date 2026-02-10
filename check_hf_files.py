from huggingface_hub import HfApi
import os

# ---------------- 配置信息 ----------------
REPO_ID = "TIME-benchmark/TIME-1.0"      # 你的仓库ID
# 注意：LOCAL_FOLDER 对应的是仓库里的 output/results 这一层
LOCAL_FOLDER = "/home/eee/qzz/TIME/output/results"
REPO_TYPE = "dataset"

# 这里定义你要对比的远程子目录，必须不包含开头的 /
REMOTE_SUBDIR = "output/results"
# ----------------------------------------

def get_local_files(folder_path):
    file_set = set()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取相对路径 (例如: "subdir/data.csv")
            rel_path = os.path.relpath(os.path.join(root, file), folder_path)
            # 排除干扰文件
            if ".git" not in rel_path and ".DS_Store" not in rel_path:
                # 统一转为 Linux 风格的斜杠（防止 Windows 环境下路径报错）
                file_set.add(rel_path.replace("\\", "/"))
    return file_set

def check_upload_status():
    api = HfApi()

    print(f"正在获取远程仓库 {REPO_ID} 的完整文件列表...")
    all_remote_files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

    # 【关键修改】：过滤并清洗远程路径
    # 只保留 output/results 下的文件，并把 "output/results/" 这个前缀切掉
    # 这样剩下的部分就是 "subdir/data.csv"，能和本地对上了
    target_prefix = REMOTE_SUBDIR + "/"
    remote_files_in_subdir = set()

    for f in all_remote_files:
        if f.startswith(target_prefix):
            # 切掉前缀，得到相对路径
            stripped_path = f[len(target_prefix):]
            if stripped_path: # 防止把文件夹本身算进去
                remote_files_in_subdir.add(stripped_path)

    print(f"远程 {REMOTE_SUBDIR} 目录下共有 {len(remote_files_in_subdir)} 个文件。")

    print(f"正在扫描本地目录 {LOCAL_FOLDER}...")
    local_files = get_local_files(LOCAL_FOLDER)
    print(f"本地目录下共有 {len(local_files)} 个文件。")

    # 计算差集：在本地但不在远程的文件
    missing_files = local_files - remote_files_in_subdir

    print("-" * 30)
    if missing_files:
        print(f"❌ 发现 {len(missing_files)} 个文件未上传成功（或路径不匹配）：")
        # 排序后打印，方便查看
        for f in sorted(list(missing_files))[:15]:
            print(f" - {f}")
        if len(missing_files) > 15:
            print(f"... (还有 {len(missing_files)-15} 个文件)")
    else:
        print(f"✅ 完美匹配！本地 {len(local_files)} 个文件已全部存在于远程的 {REMOTE_SUBDIR} 文件夹中。")

if __name__ == "__main__":
    check_upload_status()