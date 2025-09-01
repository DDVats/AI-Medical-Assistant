from huggingface_hub import hf_hub_download
from pathlib import Path

LUNG_MODEL_PATH = Path.home() / 'medical_ai_project' / 'best_lung_model.keras'

hf_hub_download(
    repo_id="your_username/lung_model",  # replace with your Hugging Face model repo
    filename="best_lung_model.keras",
    local_dir=str(LUNG_MODEL_PATH.parent)
)
print("Downloaded lung model to:", LUNG_MODEL_PATH)
