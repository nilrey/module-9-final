import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import catboost as cb
import joblib
from clearml import PipelineController, OutputModel, Task
import s3fs
import os
import json
import tempfile

# ============================================
# STEP 1: LOAD DATA
# ============================================

def load_data(train_data_path: str):
    print(f"=== load_data START ===")
    
    df = pd.read_parquet(train_data_path)
    
    feature_cols = [
        "views", "purchases", "ctr", "hour", "weekday", "categoryid", "available"
    ]
    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Сохраняем данные во временные файлы (JSON)
    temp_dir = tempfile.gettempdir()
    
    X_train_path = os.path.join(temp_dir, "X_train.npy")
    X_val_path = os.path.join(temp_dir, "X_val.npy")
    y_train_path = os.path.join(temp_dir, "y_train.npy")
    y_val_path = os.path.join(temp_dir, "y_val.npy")
    
    np.save(X_train_path, X_train.values)
    np.save(X_val_path, X_val.values)
    np.save(y_train_path, y_train.values)
    np.save(y_val_path, y_val.values)
    
    # Сохраняем колонки и метаданные
    metadata = {
        "feature_cols": feature_cols,
        "X_train_shape": X_train.shape,
        "X_val_shape": X_val.shape
    }
    metadata_path = os.path.join(temp_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    
    print(f"Data saved to temp files")
    
    return {
        "X_train_path": X_train_path,
        "X_val_path": X_val_path,
        "y_train_path": y_train_path,
        "y_val_path": y_val_path,
        "metadata_path": metadata_path
    }


# ============================================
# STEP 2: TRAIN CATBOOST
# ============================================

def train_catboost(
    X_path, y_path,
    depth: int,
    learning_rate: float,
    iterations: int
):
    print(f"=== train_catboost START ===")
    print(f"Params: depth={depth}, learning_rate={learning_rate}, iterations={iterations}")
    
    # Загружаем данные из файлов
    X_train = np.load(X_path)
    y_train = np.load(y_path)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    model = cb.CatBoostClassifier(
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        loss_function="Logloss",
        verbose=True,
        random_seed=42
    )
    
    model.fit(X_train, y_train)
    
    # Сохраняем модель
    temp_dir = tempfile.gettempdir()
    model_path = os.path.join(temp_dir, f"model_depth{depth}_lr{learning_rate}.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model_path


# ============================================
# STEP 3: EVALUATE MODEL
# ============================================

def evaluate_model(model_path, X_path, y_path, params_json: str):
    print(f"=== evaluate_model START ===")
    
    # Парсим параметры из JSON
    params = json.loads(params_json)
    print(f"Params: {params}")
    
    # Загружаем данные
    X_val = np.load(X_path)
    y_val = np.load(y_path)
    
    # Загружаем модель
    model = joblib.load(model_path)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_pred_proba)
    print(f"PR-AUC = {pr_auc:.6f}")
    
    result = {
        "pr_auc": pr_auc,
        "model_path": model_path,
        "params": params
    }
    
    task = Task.current_task()
    if task:
        task.get_logger().report_scalar("metrics", "pr_auc", value=pr_auc, iteration=0)
        for param_name, param_value in params.items():
            task.get_logger().report_single_value(f"param_{param_name}", param_value)
    
    # Возвращаем JSON строку
    return json.dumps(result)


# ============================================
# STEP 4: SELECT BEST MODEL
# ============================================

def select_and_save_best_model(
    experiment_results_json: str,
    bucket_name: str,
    model_key: str,
    endpoint_url: str
):
    print(f"=== select_and_save_best_model START ===")
    
    # Парсим результаты экспериментов из JSON
    experiment_results = json.loads(experiment_results_json)
    print(f"Number of experiment results: {len(experiment_results)}")
    
    # Выбираем лучшую модель по PR-AUC
    best_result = max(experiment_results, key=lambda x: x["pr_auc"])
    best_pr_auc = best_result["pr_auc"]
    best_params = best_result["params"]
    best_model_path = best_result["model_path"]
    
    print(f"Best model PR-AUC = {best_pr_auc:.6f}")
    print(f"Best params: {best_params}")
    
    # Загружаем лучшую модель
    best_model = joblib.load(best_model_path)
    
    # Сохраняем в S3
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": endpoint_url})
    model_s3_path = f"s3://{bucket_name}/{model_key}"
    
    with fs.open(model_s3_path, "wb") as f:
        joblib.dump(best_model, f)
    print(f"Model saved to S3: {model_s3_path}")
    
    return json.dumps({
        "best_pr_auc": best_pr_auc,
        "best_params": best_params,
        "s3_path": model_s3_path
    })


# ============================================
# PIPELINE CONTROLLER
# ============================================

pipe = PipelineController(
    name="CatBoost Hyperparameter Tuning",
    project="mlops",
    version="1.0.0"
)

# Шаг 1: Загрузка данных
pipe.add_function_step(
    name="load_data",
    function=load_data,
    function_kwargs=dict(
        train_data_path="/opt/clearml_data/data_for_training.parquet"
    ),
    function_return=["data_paths"]
)

# Шаг 2: Обучение моделей
pipe.add_function_step(
    name="train_exp1",
    function=train_catboost,
    function_kwargs=dict(
        X_path="${load_data.data_paths.X_train_path}",
        y_path="${load_data.data_paths.y_train_path}",
        depth=4,
        learning_rate=0.1,
        iterations=500
    ),
    parents=["load_data"],
    function_return=["model_path_exp1"]
)

pipe.add_function_step(
    name="train_exp2",
    function=train_catboost,
    function_kwargs=dict(
        X_path="${load_data.data_paths.X_train_path}",
        y_path="${load_data.data_paths.y_train_path}",
        depth=6,
        learning_rate=0.05,
        iterations=300
    ),
    parents=["load_data"],
    function_return=["model_path_exp2"]
)

# Шаг 3: Оценка моделей
pipe.add_function_step(
    name="evaluate_exp1",
    function=evaluate_model,
    function_kwargs=dict(
        model_path="${train_exp1.model_path_exp1}",
        X_path="${load_data.data_paths.X_val_path}",
        y_path="${load_data.data_paths.y_val_path}",
        params_json=json.dumps({"depth": 4, "learning_rate": 0.1, "iterations": 500})
    ),
    parents=["train_exp1"],
    function_return=["result_exp1"]
)

pipe.add_function_step(
    name="evaluate_exp2",
    function=evaluate_model,
    function_kwargs=dict(
        model_path="${train_exp2.model_path_exp2}",
        X_path="${load_data.data_paths.X_val_path}",
        y_path="${load_data.data_paths.y_val_path}",
        params_json=json.dumps({"depth": 6, "learning_rate": 0.05, "iterations": 300})
    ),
    parents=["train_exp2"],
    function_return=["result_exp2"]
)

# Шаг 4: Выбор лучшей модели
pipe.add_function_step(
    name="select_best",
    function=select_and_save_best_model,
    function_kwargs=dict(
        experiment_results_json=f'[{ "${evaluate_exp1.result_exp1}", "${evaluate_exp2.result_exp2}" }]',
        bucket_name="r-mlops-bucket-12-1-1-22209764",
        model_key="nil_project/models/ranker.pkl",
        endpoint_url="https://storage.yandexcloud.net"
    ),
    parents=["evaluate_exp1", "evaluate_exp2"],
    function_return=["best_info"]
)

# Запуск
if __name__ == "__main__":
    print("Running pipeline locally...")
    pipe.start_locally(True)
    print("Pipeline finished")