# Model training: load splits from HF, tune XGBoost and a Decision Tree with
# cross-validation, log every run to MLflow, evaluate on the test set, and
# register the best model on the Hugging Face model hub
import os
import joblib
import mlflow
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

DATA_PATH = "hf://datasets/MatthewSimon/tourism-dataset"
MODEL_REPO = "MatthewSimon/tourism-model"

mlflow.set_experiment("tourism-wellness-package")

# --- Load the train/test splits from the Hugging Face dataset repo ---
Xtrain = pd.read_csv(f"{DATA_PATH}/Xtrain.csv")
Xtest = pd.read_csv(f"{DATA_PATH}/Xtest.csv")
ytrain = pd.read_csv(f"{DATA_PATH}/ytrain.csv").squeeze()
ytest = pd.read_csv(f"{DATA_PATH}/ytest.csv").squeeze()

# --- Preprocessing ---
# Trees don't need scaling, so numeric columns pass through unchanged.
# Categorical columns are one-hot encoded; handle_unknown="ignore" stops rare
# categories (e.g. the 2 'Free Lancer' rows) raising errors if unseen in training.
categorical_features = Xtrain.select_dtypes(include=["object", "string"]).columns.tolist()
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
    remainder="passthrough",
    verbose_feature_names_out=False,
)

# Ratio of non-buyers to buyers, used to up-weight the minority class in XGBoost
scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()

# --- Candidate models and their hyperparameter grids ---
models = {
    "xgboost": (
        XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42,
                      eval_metric="logloss", n_jobs=-1),
        {
            "xgbclassifier__n_estimators": [100, 200],
            "xgbclassifier__max_depth": [3, 5],
            "xgbclassifier__learning_rate": [0.05, 0.1],
            "xgbclassifier__subsample": [0.8, 1.0],
            "xgbclassifier__colsample_bytree": [0.8, 1.0],
        },
    ),
    "decision_tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        {
            "decisiontreeclassifier__max_depth": [3, 5, 7, 10],
            "decisiontreeclassifier__min_samples_leaf": [1, 5, 10],
            "decisiontreeclassifier__criterion": ["gini", "entropy"],
        },
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, (estimator, param_grid) in models.items():
    pipeline = make_pipeline(preprocessor, estimator)
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1)

    with mlflow.start_run(run_name=name):
        search.fit(Xtrain, ytrain)

        # Log every parameter combination tried as a nested run
        cvr = search.cv_results_
        for i, params in enumerate(cvr["params"]):
            with mlflow.start_run(run_name=f"{name}_combo_{i}", nested=True):
                mlflow.log_params({k.split("__")[1]: v for k, v in params.items()})
                mlflow.log_metric("cv_f1_mean", cvr["mean_test_score"][i])
                mlflow.log_metric("cv_f1_std", cvr["std_test_score"][i])

        # Log the best parameters and evaluate the best model on the test set
        best = search.best_estimator_
        y_pred = best.predict(Xtest)
        y_prob = best.predict_proba(Xtest)[:, 1]
        test_metrics = {
            "test_accuracy": accuracy_score(ytest, y_pred),
            "test_precision": precision_score(ytest, y_pred),
            "test_recall": recall_score(ytest, y_pred),
            "test_f1": f1_score(ytest, y_pred),
            "test_roc_auc": roc_auc_score(ytest, y_prob),
        }
        mlflow.log_params({k.split("__")[1]: v for k, v in search.best_params_.items()})
        mlflow.log_metric("best_cv_f1", search.best_score_)
        mlflow.log_metrics(test_metrics)

        # Save the model locally and attach it to the MLflow run
        model_file = f"tourism_project/model_building/{name}_model.joblib"
        joblib.dump(best, model_file)
        mlflow.log_artifact(model_file)

    results[name] = {"cv_f1": search.best_score_, "file": model_file, **test_metrics}
    print(f"\n{name}: best CV F1 = {search.best_score_:.3f}")
    print(f"  best params: {search.best_params_}")
    print("  test:", {k: round(float(v), 3) for k, v in test_metrics.items()})

# --- Select the best model by CV score (not test score, to keep the test set unbiased) ---
best_name = max(results, key=lambda n: results[n]["cv_f1"])
print(f"\nBest model: {best_name}")

# --- Register the best model on the Hugging Face model hub ---
api = HfApi(token=os.getenv("HF_TOKEN"))
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print(f"Model repo '{MODEL_REPO}' exists. Using it.")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False,
                token=os.getenv("HF_TOKEN"))
    print(f"Model repo '{MODEL_REPO}' created.")

# Upload under a fixed filename so the app always loads the current best model
api.upload_file(
    path_or_fileobj=results[best_name]["file"],
    path_in_repo="best_model.joblib",
    repo_id=MODEL_REPO,
    repo_type="model",
)
print("Best model uploaded to the Hugging Face model hub.")
