# ── prep.py: Data Preparation Script for GitHub Actions Pipeline ──────────────
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# ── Initialise HuggingFace API ────────────────────────────────────────────────
api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_REPO = "kumard28/Tourism-package-data"

# ── Load raw tourism data from HuggingFace ────────────────────────────────────
print("Loading raw dataset from HuggingFace...")
df = pd.read_csv(f"hf://datasets/{DATASET_REPO}/tourism.csv")
print(f"Dataset loaded. Shape: {df.shape}")

# ── Drop unnecessary columns ──────────────────────────────────────────────────
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True, errors="ignore")

# ── Fix known data quality issues ─────────────────────────────────────────────
df["Gender"] = df["Gender"].replace("Fe Male", "Female")
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")

# ── Drop rows with missing values ─────────────────────────────────────────────
df.dropna(inplace=True)
print(f"Shape after cleaning: {df.shape}")

# ── One Hot Encoding ──────────────────────────────────────────────────────────
df = pd.get_dummies(df, drop_first=True)
print(f"Shape after encoding: {df.shape}")

# ── Split into features and target ────────────────────────────────────────────
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Combine back for saving ───────────────────────────────────────────────────
train = pd.concat([X_train, y_train], axis=1)
test  = pd.concat([X_test,  y_test],  axis=1)

# ── Save locally ──────────────────────────────────────────────────────────────
train.to_csv("train.csv", index=False)
test.to_csv("test.csv",   index=False)
print("✅ train.csv and test.csv saved locally")

# ── Upload to HuggingFace ─────────────────────────────────────────────────────
for file in ["train.csv", "test.csv"]:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"✅ {file} uploaded to HuggingFace")

print("✅ Data preparation complete.")
