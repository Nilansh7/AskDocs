import os
from huggingface_hub import snapshot_download

# Define the model name and the local directory to save it
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
local_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

print(f"Downloading model '{model_name}' to cache directory...")

try:
    # This command will download all model files to your Hugging Face cache
    snapshot_download(repo_id=model_name, local_dir_use_symlinks=False)
    print("Download complete!")
except Exception as e:
    print(f"An error occurred during download: {e}")
    print("Make sure you are logged into Hugging Face CLI and have accepted the model's terms.")
