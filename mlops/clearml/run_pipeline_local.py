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
# STEP 1: LOAD DATA
# ============================================

def load_data(train_data_path: str):
    print(f"=== load_data START ===")
    print(f"Local path: {train_data_path}")
    
    df = pd.read_parquet(train_data_path)
    print(f"DataFrame loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    
    feature_cols = [
        "views", "purchases", "ctr", "hour", "weekday", "categoryid", "available"
    ]
    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    print("=== load_data END ===")
    
    return X_train, X_val, y_train, y_val

# ============================================
# STEP 2: TRAIN CATBOOST
# ============================================

def train_catboost(
    X_train, y_train,
    depth: int,
    learning_rate: float,
    iterations: int
):
    print(f"=== train_catboost START ===")
    print(f"Params: depth={depth}, learning_rate={learning_rate}, iterations={iterations}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    print("Creating CatBoost model...")
    model = cb.CatBoostClassifier(
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        loss_function="Logloss",
        verbose=True,
        random_seed=42
    )
    
    print("Starting training...")
    model.fit(X_train, y_train)
    print("Training completed")
    print(f"=== train_catboost END ===")
    
    return model

# ============================================
# STEP 3: EVALUATE MODEL
# ============================================

def evaluate_model(model, X_val, y_val, params: dict):
    print(f"=== evaluate_model START ===")
    print(f"Params: {params}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    
    print("Getting prediction probabilities...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    print(f"Predictions ready, shape: {y_pred_proba.shape}")
    
    print("Calculating PR-AUC...")
    pr_auc = average_precision_score(y_val, y_pred_proba)
    print(f"PR-AUC = {pr_auc:.6f}")
    
    result = {
        "pr_auc": pr_auc,
        "model": model,
        "params": params
    }
    
    task = Task.current_task()
    if task:
        print("Reporting metrics to ClearML...")
        task.get_logger().report_scalar("metrics", "pr_auc", value=pr_auc, iteration=0)
        for param_name, param_value in params.items():
            task.get_logger().report_single_value(f"param_{param_name}", param_value)
    
    print(f"=== evaluate_model END ===")
    return result

# ============================================
# STEP 4: SELECT BEST MODEL
# ============================================

def select_and_save_best_model(
    experiment_results,
    bucket_name: str,
    model_key: str,
    endpoint_url: str
):
    print(f"=== select_and_save_best_model START ===")
    print(f"Number of experiment results: {len(experiment_results)}")
    
    # experiment_results уже содержит словари (не строки!)
    for i, res in enumerate(experiment_results):
        print(f"  Experiment {i+1}: PR-AUC = {res.get('pr_auc', 'N/A')}")
    
    print("Selecting best model by PR-AUC...")
    best_result = max(experiment_results, key=lambda x: x["pr_auc"])
    best_model = best_result["model"]
    best_pr_auc = best_result["pr_auc"]
    best_params = best_result["params"]
    
    print(f"Best model PR-AUC = {best_pr_auc:.6f}")
    print(f"Best params: {best_params}")
    
    print(f"Connecting to S3 at {endpoint_url}...")
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": endpoint_url}
    )
    
    model_path = f"s3://{bucket_name}/{model_key}"
    print(f"Saving model to {model_path}...")
    with fs.open(model_path, "wb") as f:
        joblib.dump(best_model, f)
    print("Model saved to S3")
    
    task = Task.current_task()
    output_model_id = None
    
    if task:
        print("Registering model in ClearML...")
        output_model = OutputModel(
            task=task,
            framework="CatBoost",
            name="catboost_ranker",
            comment=f"Best model with PR-AUC = {best_pr_auc:.6f}",
            tags=["best_model", "catboost"]
        )
        
        temp_path = "/tmp/best_catboost_model.pkl"
        print(f"Saving temp copy to {temp_path}...")
        joblib.dump(best_model, temp_path, compress=True)
        output_model.update_weights(temp_path)
        os.remove(temp_path)
        print(f"Model registered with ID: {output_model.id}")
        output_model_id = output_model.id
    else:
        print("WARNING: No current ClearML task found, model not registered")
    
    print(f"=== select_and_save_best_model END ===")
    
    return {
        "best_pr_auc": best_pr_auc,
        "best_params": best_params,
        "s3_path": model_path,
        "clearml_model_id": output_model_id
    }

# ============================================
# РУЧНОЙ ЗАПУСК (ГАРАНТИРОВАННО РАБОТАЕТ)
# ============================================

def run_manual():
    """Ручной последовательный запуск - 100% работает"""
    print("=" * 60)
    print("MANUAL PIPELINE EXECUTION")
    print("=" * 60)
    
    # Step 1
    X_train, X_val, y_train, y_val = load_data("/opt/clearml_data/data_for_training.parquet")
    
    # Step 2 - параллельно (ручное управление)
    print("\n--- Starting parallel training ---")
    from concurrent.futures import ThreadPoolExecutor
    
    def train_exp1():
        return train_catboost(X_train, y_train, depth=4, learning_rate=0.1, iterations=500)
    
    def train_exp2():
        return train_catboost(X_train, y_train, depth=6, learning_rate=0.05, iterations=300)
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(train_exp1)
        future2 = executor.submit(train_exp2)
        model_exp1 = future1.result()
        model_exp2 = future2.result()
    
    # Step 3 - параллельно
    print("\n--- Starting parallel evaluation ---")
    
    def eval_exp1():
        return evaluate_model(model_exp1, X_val, y_val, {"depth": 4, "learning_rate": 0.1, "iterations": 500})
    
    def eval_exp2():
        return evaluate_model(model_exp2, X_val, y_val, {"depth": 6, "learning_rate": 0.05, "iterations": 300})
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(eval_exp1)
        future2 = executor.submit(eval_exp2)
        result_exp1 = future1.result()
        result_exp2 = future2.result()
    
    # Step 4
    best_info = select_and_save_best_model(
        experiment_results=[result_exp1, result_exp2],
        bucket_name="r-mlops-bucket-12-1-1-22209764",
        model_key="nil_project/models/ranker.pkl",
        endpoint_url="https://storage.yandexcloud.net"
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE FINISHED SUCCESSFULLY!")
    print(f"Best PR-AUC: {best_info['best_pr_auc']:.6f}")
    print(f"Best params: {best_info['best_params']}")
    print(f"Model saved to: {best_info['s3_path']}")
    print("=" * 60)

# ============================================
# ЗАПУСК
# ============================================

if __name__ == "__main__":
    run_manual()  # Гарантированно работает