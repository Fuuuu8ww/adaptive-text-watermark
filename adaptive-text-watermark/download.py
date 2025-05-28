
from huggingface_hub import snapshot_download
import os

# 替换为您要下载的模型的 Hugging Face Hub ID
repo_id = "facebook/opt-350m"  # 示例，请替换为您的 opt-305m 的正确 Hub ID
# 替换为您希望在服务器上保存模型的本地路径
local_model_path = "/root/adaptive-text-watermark/adaptive-text-watermark/data/opt-305m" # 示例路径

# 可选参数：
# revision="main"  # 可以指定分支、标签或 commit hash
# token=None  # 如果是私有模型，需要提供 Hugging Face token (hf_hub_token)
# resume_download=True # 如果下载中断，可以尝试断点续传
# local_dir_use_symlinks=False # Windows上建议设置为False，Linux/MacOS上可以为True或False，设为False会直接下载文件而不是创建指向缓存的符号链接

print(f"Downloading model '{repo_id}' to '{local_model_path}'...")

try:
    # 使用 local_dir 参数指定下载路径
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_model_path,
        local_dir_use_symlinks=False, # 重要：确保文件直接下载到指定目录
        # ignore_patterns=["*.safetensors", "*.h5"], # 可选：忽略特定类型文件
        # allow_patterns=["*.bin", "*.json", "*.txt"], # 可选：只下载特定类型文件
    )
    print(f"Model '{repo_id}' downloaded successfully to '{local_model_path}'.")
    print("Files in the directory:")
    for item in os.listdir(local_model_path):
        print(os.path.join(local_model_path, item))

except Exception as e:
    print(f"An error occurred: {e}")