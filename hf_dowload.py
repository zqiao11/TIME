from huggingface_hub import snapshot_download
import os

# ================= 配置区域 =================
REPO_ID = "Real-TSF/TIME"

# 【重要修改】
# 因为你的仓库里已经有了 'data' and 'output' 文件夹结构，
# 所以这里建议设为 "." (当前目录)，或者你项目的根目录。
# 效果：
#   远程的 data/   --> 下载到 本地 ./data/
#   远程的 output/ --> 下载到 本地 ./output/
LOCAL_ROOT_DIR = "."

# [可选] 开启 HF Transfer 加速 (使用 Rust 高速传输)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# ===========================================

def download_project():
    # 获取绝对路径方便查看
    abs_path = os.path.abspath(LOCAL_ROOT_DIR)
    print(f"🚀 开始从 Hugging Face 仓库 '{REPO_ID}' 同步所有数据...")
    print(f"📂 本地保存根目录: {abs_path}")
    print(f"   (预期会更新/创建 {os.path.join(abs_path, 'data')} 和 {os.path.join(abs_path, 'output')})")

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_ROOT_DIR, # 下载到这个根目录

            # 关键参数：
            local_dir_use_symlinks=False, # 下载真实文件
            token=True,                   # 读取本地 Token
            resume_download=True,         # 断点续传

            # 【进阶用法】如果你只想下载 data 文件夹，取消下面这行的注释：
            # allow_patterns=["data/*"],

            # 【进阶用法】如果你只想下载 output 文件夹，取消下面这行的注释：
            # allow_patterns=["output/*"],
        )

        print("\n✅ 同步完成！")
        print(f"你的数据和结果已就位：")
        print(f"  - 数据: {os.path.join(abs_path, 'data')}")
        print(f"  - 结果: {os.path.join(abs_path, 'output')}")

    except Exception as e:
        print(f"\n❌ 下载过程中出错: {e}")

if __name__ == "__main__":
    download_project()