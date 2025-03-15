from huggingface_hub import snapshot_download
import transformers
print(transformers.__version__)

# snapshot_download(repo_id="huggyllama/llama-13b",
#  local_dir="/remote-home1/yangshuo/N-LoRA/initial_model/llama-13b/",
#  local_dir_use_symlinks=False, max_workers=1 )

# T5-small 
snapshot_download(repo_id="google/vit-base-patch16-224",
 local_dir="initial_model/vit-base-patch16-224/",
 local_dir_use_symlinks=False, max_workers=1 )






# vit-base-patch16-224-in21k
# snapshot_download(repo_id="google/vit-base-patch16-224-in21k",
#  local_dir="initial_model/vit-base-patch16-224-in21k/",
#  local_dir_use_symlinks=False, max_workers=1 )
