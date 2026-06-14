from huggingface_hub import HfApi
import os

# ── Initialise HF API using token from environment variable ───────────────────
# In Colab we pass the token directly; in GitHub Actions it comes from secrets
api = HfApi(token=os.getenv("HF_TOKEN"))

# ── Upload all deployment files to the HuggingFace Space ─────────────────────
# folder_path: the local folder containing Dockerfile, app.py, requirements.txt
# repo_id: your HuggingFace Space (must already exist and be set to Docker/Streamlit)
# repo_type: "space" tells HF this is a Space, not a model or dataset repo
# path_in_repo: "" means files go to the root of the Space repo
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id="kumard28/Tourism-Package-Prediction",
    repo_type="space",
    path_in_repo="",
)

print("✅ Deployment files uploaded to HuggingFace Space successfully!")
print("🔗 View Space at: https://huggingface.co/spaces/kumard28/Tourism-Package-Prediction")
