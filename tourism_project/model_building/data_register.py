
# ─────────────────────────────────────────────────────────────
# data_register.py
# Purpose: Upload the tourism dataset to Hugging Face Hub
# This runs as Step 1 in our automated MLOps pipeline
# ─────────────────────────────────────────────────────────────

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ── Your Hugging Face credentials ──────────────────────────
HF_TOKEN  = os.getenv("HF_TOKEN")   # injected securely from GitHub secrets
HF_USER   = "kumard28"
REPO_NAME = "Tourism-package-data"
REPO_ID   = f"{HF_USER}/{REPO_NAME}"
REPO_TYPE = "dataset"

# ── Initialize the HF API ──────────────────────────────────
api = HfApi(token=HF_TOKEN)

# ── Create the dataset repo if it doesn't already exist ───
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Repository '{REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repository '{REPO_ID}' not found. Creating it now...")
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False)
    print(f"Repository '{REPO_ID}' created successfully.")

# ── Upload the data folder to Hugging Face ─────────────────
api.upload_folder(
    folder_path="tourism_project/data",   # local folder containing tourism.csv
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)

print("✅ Data registration complete. tourism.csv uploaded to Hugging Face.")
