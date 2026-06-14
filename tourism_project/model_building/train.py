# ── train.py: Model Training Script for GitHub Actions Pipeline ───────────────
import pandas as pd
import os
import joblib
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ── Setup ─────────────────────────────────────────────────────────────────────
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_REPO = "kumard28/Tourism-package-data"
MODEL_REPO   = "kumard28/Tourism-Package-Prediction-Model"

# ── MLflow setup ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism-Package-Prediction-Experiment")

# ── Load train/test from HuggingFace ─────────────────────────────────────────
print("Loading train and test data...")
train = pd.read_csv(f"hf://datasets/{DATASET_REPO}/train.csv")
test  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/test.csv")

target_col = "ProdTaken"
Xtrain = train.drop(columns=[target_col])
ytrain = train[target_col]
Xtest  = test.drop(columns=[target_col])
ytest  = test[target_col]

print(f"✅ Train: {Xtrain.shape}, Test: {Xtest.shape}")

# ── Handle class imbalance ────────────────────────────────────────────────────
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# ── Define XGBoost model ──────────────────────────────────────────────────────
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss",
    verbosity=0
)

# ── Hyperparameter grid ───────────────────────────────────────────────────────
param_grid = {
    "n_estimators":     [50, 100],
    "max_depth":        [3, 4],
    "learning_rate":    [0.05, 0.1],
    "colsample_bytree": [0.6, 0.8],
    "reg_lambda":       [0.5, 1.0],
}

# ── GridSearchCV with MLflow tracking ────────────────────────────────────────
print("Starting GridSearchCV with MLflow tracking...")
with mlflow.start_run(run_name="XGBoost_GridSearch"):
    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=5,
        scoring="f1", n_jobs=-1, verbose=1
    )
    grid_search.fit(Xtrain, ytrain)

    # Log all combinations
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(run_name=f"combo_{i+1}", nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_f1",  results["std_test_score"][i])

    # Log best params and metrics
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    y_pred_test = best_model.predict(Xtest)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "test_accuracy":  test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall":    test_report["1"]["recall"],
        "test_f1":        test_report["1"]["f1-score"],
    })

    print(f"✅ Best params: {grid_search.best_params_}")
    print(f"✅ Best CV F1:  {grid_search.best_score_:.4f}")

# ── Save and upload model ─────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/tourism_model.pkl")
print("✅ Model saved locally")

try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO, repo_type="model",
                private=False, token=os.getenv("HF_TOKEN"))

api.upload_file(
    path_or_fileobj="model/tourism_model.pkl",
    path_in_repo="tourism_model.pkl",
    repo_id=MODEL_REPO,
    repo_type="model",
)
print("✅ Model uploaded to HuggingFace!")
print("✅ Training pipeline complete.")
