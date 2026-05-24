import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import catboost as cb
import joblib
from clearml import PipelineController, OutputModel, Task
import s3fs
import os

# ============================================
# STEP 1: LOAD DATA FROM S3
# ============================================
def load_data(
    train_data_path: str,
    bucket_name: str,
    endpoint_url: str
):
    s3_url = f"s3://{bucket_name}/{train_data_path}"
    
    df = pd.read_parquet(
        s3_url,
        storage_options={
            'client_kwargs': {'endpoint_url': endpoint_url}
        }
    )
    
    feature_cols = [
        "views", "purchases", "ctr", "hour", "weekday", "categoryid", "available"
    ]
    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val


# ============================================
# STEP 2: TRAIN CATBOOST WITH SPECIFIC PARAMS
# ============================================
def train_catboost(
    X_train, y_train,
    depth: int,
    learning_rate: float,
    iterations: int
):
    model = cb.CatBoostClassifier(
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        loss_function="Logloss",
        verbose=False,
        random_seed=42
    )
    
    model.fit(X_train, y_train)
    return model


# ============================================
# STEP 3: EVALUATE MODEL (PR-AUC)
# ============================================
def evaluate_model(model, X_val, y_val, params: dict):
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_pred_proba)
    
    task = Task.current_task()
    if task:
        task.get_logger().report_scalar(
            "metrics", "pr_auc", value=pr_auc, iteration=0
        )
        for param_name, param_value in params.items():
            task.get_logger().report_single_value(f"param_{param_name}", param_value)
    
    return {
        "pr_auc": pr_auc,
        "model": model,
        "params": params
    }


# ============================================
# STEP 4: SELECT BEST MODEL BY PR-AUC
# ============================================
def select_and_save_best_model(
    experiment_results,
    bucket_name: str,
    model_key: str,
    endpoint_url: str
):
    best_result = max(experiment_results, key=lambda x: x["pr_auc"])
    best_model = best_result["model"]
    best_pr_auc = best_result["pr_auc"]
    best_params = best_result["params"]
    
    print(f"Best model PR-AUC = {best_pr_auc:.4f}")
    print(f"Best params: {best_params}")
    
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": endpoint_url}
    )
    
    model_path = f"s3://{bucket_name}/{model_key}"
    with fs.open(model_path, "wb") as f:
        joblib.dump(best_model, f)
    
    task = Task.current_task()
    output_model = OutputModel(
        task=task,
        framework="CatBoost",
        name="catboost_ranker",
        comment=f"Best model with PR-AUC = {best_pr_auc:.4f}",
        tags=["best_model", "catboost"]
    )
    
    temp_path = "/tmp/best_catboost_model.pkl"
    joblib.dump(best_model, temp_path, compress=True)
    output_model.update_weights(temp_path)
    os.remove(temp_path)
    
    print(f"Model saved to S3: {model_path}")
    print(f"Model registered in ClearML with ID: {output_model.id}")
    
    return {
        "best_pr_auc": best_pr_auc,
        "best_params": best_params,
        "s3_path": model_path,
        "clearml_model_id": output_model.id
    }


# ============================================
# CREATE PIPELINE
# ============================================
def create_pipeline():
    pipe = PipelineController(
        name="CatBoost Hyperparameter Tuning",
        project="mlops",
        version="1.0.0",
        add_pipeline_tags=True
    )
    
    pipe.set_default_execution_queue(default_execution_queue="default")
    
    # Pipeline parameters
    pipe.add_parameter(
        name="train_data_path",
        default="nil_project/processed_data/data_for_training.parquet"
    )
    pipe.add_parameter(
        name="bucket_name",
        default="r-mlops-bucket-12-1-1-22209764"
    )
    pipe.add_parameter(
        name="model_key",
        default="nil_project/models/ranker.pkl"
    )
    pipe.add_parameter(
        name="endpoint_url",
        default="https://storage.yandexcloud.net"
    )
    
    # Step 1: Load data
    pipe.add_function_step(
        name="load_data",
        function=load_data,
        function_kwargs=dict(
            train_data_path="${pipeline.train_data_path}",
            bucket_name="${pipeline.bucket_name}",
            endpoint_url="${pipeline.endpoint_url}"
        ),
        function_return=["X_train", "X_val", "y_train", "y_val"],
        packages=[
            "clearml[s3]==2.0.2",
            "pandas==2.2.2",
            "scikit-learn==1.5.1",
            "catboost==1.2.8",
            "joblib==1.5.2",
            "numpy==1.26.3",
            "s3fs==2024.10.0",
            "pyarrow==22.0.0"
        ]
    )
    
    # Experiment 1: depth=4, lr=0.1, iter=500
    pipe.add_function_step(
        name="train_exp1",
        function=train_catboost,
        function_kwargs=dict(
            X_train="${load_data.X_train}",
            y_train="${load_data.y_train}",
            depth=4,
            learning_rate=0.1,
            iterations=500
        ),
        function_return=["model_exp1"],
        packages=[
            "clearml[s3]==2.0.2",
            "pandas==2.2.2",
            "scikit-learn==1.5.1",
            "catboost==1.2.8",
            "joblib==1.5.2",
            "numpy==1.26.3",
            "s3fs==2024.10.0",
            "pyarrow==22.0.0"
        ],
        parents=["load_data"]
    )
    
    # Experiment 2: depth=6, lr=0.05, iter=300
    pipe.add_function_step(
        name="train_exp2",
        function=train_catboost,
        function_kwargs=dict(
            X_train="${load_data.X_train}",
            y_train="${load_data.y_train}",
            depth=6,
            learning_rate=0.05,
            iterations=300
        ),
        function_return=["model_exp2"],
        packages=[
            "clearml[s3]==2.0.2",
            "pandas==2.2.2",
            "scikit-learn==1.5.1",
            "catboost==1.2.8",
            "joblib==1.5.2",
            "numpy==1.26.3",
            "s3fs==2024.10.0",
            "pyarrow==22.0.0"
        ],
        parents=["load_data"]
    )
    
    # Evaluate Experiment 1
    pipe.add_function_step(
        name="evaluate_exp1",
        function=evaluate_model,
        function_kwargs=dict(
            model="${train_exp1.model_exp1}",
            X_val="${load_data.X_val}",
            y_val="${load_data.y_val}",
            params={"depth": 4, "learning_rate": 0.1, "iterations": 500}
        ),
        function_return=["result_exp1"],
        packages=[
            "clearml[s3]==2.0.2",
            "pandas==2.2.2",
            "scikit-learn==1.5.1",
            "catboost==1.2.8",
            "joblib==1.5.2",
            "numpy==1.26.3",
            "s3fs==2024.10.0",
            "pyarrow==22.0.0"
        ],
        parents=["train_exp1"]
    )
    
    # Evaluate Experiment 2
    pipe.add_function_step(
        name="evaluate_exp2",
        function=evaluate_model,
        function_kwargs=dict(
            model="${train_exp2.model_exp2}",
            X_val="${load_data.X_val}",
            y_val="${load_data.y_val}",
            params={"depth": 6, "learning_rate": 0.05, "iterations": 300}
        ),
        function_return=["result_exp2"],
        packages=[
            "clearml[s3]==2.0.2",
            "pandas==2.2.2",
            "scikit-learn==1.5.1",
            "catboost==1.2.8",
            "joblib==1.5.2",
            "numpy==1.26.3",
            "s3fs==2024.10.0",
            "pyarrow==22.0.0"
        ],
        parents=["train_exp2"]
    )
    
    # Select and save best model
    pipe.add_function_step(
        name="select_best",
        function=select_and_save_best_model,
        function_kwargs=dict(
            experiment_results=[
                "${evaluate_exp1.result_exp1}",
                "${evaluate_exp2.result_exp2}"
            ],
            bucket_name="${pipeline.bucket_name}",
            model_key="${pipeline.model_key}",
            endpoint_url="${pipeline.endpoint_url}"
        ),
        function_return=["best_info"],
        packages=[
            "clearml[s3]==2.0.2",
            "pandas==2.2.2",
            "scikit-learn==1.5.1",
            "catboost==1.2.8",
            "joblib==1.5.2",
            "numpy==1.26.3",
            "s3fs==2024.10.0",
            "pyarrow==22.0.0"
        ],
        parents=["evaluate_exp1", "evaluate_exp2"]
    )
    
    return pipe


if __name__ == "__main__":
    print("ClearML Environment:")
    print(f"  CLEARML_API_HOST: {os.environ.get('CLEARML_API_HOST', 'NOT SET')}")
    print(f"  BUCKET_NAME: {os.environ.get('BUCKET_NAME', 'r-mlops-bucket-12-1-1-22209764')}")
    
    pipe = create_pipeline()
    pipe.start(queue="default")
    print("Pipeline submitted successfully")