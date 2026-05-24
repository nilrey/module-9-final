import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import catboost as cb
import joblib
from clearml import PipelineController, Task
import s3fs
import os
import json
import tempfile


# STEP 1: LOAD DATA
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
    
    # Сохраняем данные
    temp_dir = tempfile.gettempdir()
    
    X_train_path = os.path.join(temp_dir, "X_train.npy")
    X_val_path = os.path.join(temp_dir, "X_val.npy")
    y_train_path = os.path.join(temp_dir, "y_train.npy")
    y_val_path = os.path.join(temp_dir, "y_val.npy")
    
    np.save(X_train_path, X_train.values)
    np.save(X_val_path, X_val.values)
    np.save(y_train_path, y_train.values)
    np.save(y_val_path, y_val.values)
    
    # Возвращаем JSON строку со словарем
    data_paths = {
        "X_train_path": X_train_path,
        "X_val_path": X_val_path,
        "y_train_path": y_train_path,
        "y_val_path": y_val_path
    }
    
    return json.dumps(data_paths) 


# STEP 2: НАЧИНАЕМ ОБУЧЕНИЕ
def train_catboost(
    data_paths_json: str,  # принимаем JSON строку (т.к. передача через внутренние переменные не происходт для локального запуска)
    depth: int,
    learning_rate: float,
    iterations: int
):
    print(f"Start train_catboost")
    print(f"Params: depth={depth}, learning_rate={learning_rate}, iterations={iterations}")
    
    # Парсим JSON
    print(f"Received data_paths_json: {data_paths_json}")
    print(f"Type of data_paths_json: {type(data_paths_json)}")
    
    data_paths = json.loads(data_paths_json)
    print(f"Parsed data_paths: {data_paths}")
    
    X_path = data_paths["X_train_path"]
    y_path = data_paths["y_train_path"]
    
    print(f"X_path: {X_path}")
    print(f"y_path: {y_path}")
    print(f"X_path exists: {os.path.exists(X_path)}")
    print(f"y_path exists: {os.path.exists(y_path)}")
    
    # Загружаем данные
    X_train = np.load(X_path)
    y_train = np.load(y_path)
    
    print(f"X_train shape: {X_train.shape}, type: {type(X_train)}")
    print(f"y_train shape: {y_train.shape}, type: {type(y_train)}")
    
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
    
    # Сохраняем модель
    temp_dir = tempfile.gettempdir()
    model_path = os.path.join(temp_dir, f"model_depth{depth}_lr{learning_rate}.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    print(f"Model file exists: {os.path.exists(model_path)}")
    
    print(f"Finish train_catboost success")
    return model_path


# STEP 3: считаем PR_AUC
def evaluate_model(model_path, data_paths_json: str, params_json: str, output_file: str = None):
    print(f"Start evaluate_model")
    
    # Парсим пути к данным из JSON
    data_paths = json.loads(data_paths_json)
    X_path = data_paths["X_val_path"]
    y_path = data_paths["y_val_path"]
    
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
    
    # Сохраняем результат во временный файл
    if output_file is None:
        temp_dir = tempfile.gettempdir()
        output_file = os.path.join(temp_dir, f"result_{params['depth']}_{params['learning_rate']}.json")
    
    with open(output_file, "w") as f:
        json.dump(result, f)
    print(f"Result saved to: {output_file}")
    
    task = Task.current_task()
    if task:
        task.get_logger().report_scalar("metrics", "pr_auc", value=pr_auc, iteration=0)
        for param_name, param_value in params.items():
            task.get_logger().report_single_value(f"param_{param_name}", param_value)
    
    print(f"Finished evaluate_model success")
    return output_file  # возвращаем путь к файлу 



# STEP 4: ВЫБОР ЛУЧШЕЙ МОДЕЛИ
def save_best_model(
    result_files_json: str,
    bucket_name: str,
    model_key: str,
    endpoint_url: str
):
    print(f"Srart save_best_model ")
    
    # Парсим пути к файлам
    result_files = json.loads(result_files_json)
    print(f"Result files: {result_files}")
    
    # Загружаем результаты из файлов
    experiment_results = []
    for file_path in result_files:
        with open(file_path, "r") as f:
            result = json.load(f)
            experiment_results.append(result)
            print(f"Loaded result: PR-AUC = {result['pr_auc']:.6f}")
    
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
    
    # Очищаем временные файлы
    for file_path in result_files:
        try:
            os.remove(file_path)
            print(f"Cleaned up: {file_path}")
        except:
            pass
    
    print(f"Finished save_best_model success")
    
    return json.dumps({
        "best_pr_auc": best_pr_auc,
        "best_params": best_params,
        "s3_path": model_s3_path
    })



# PIPELINE CONTROLLER
pipe = PipelineController(
    name="CatBoost Hyperparameter Tuning",
    project="mlops",
    version="1.0.0"
)

# Загрузка данных
pipe.add_function_step(
    name="load_data",
    function=load_data,
    function_kwargs=dict(
        train_data_path="/opt/clearml_data/data_for_training.parquet"
    ),
    function_return=["data_paths_json"]  
)

# Параллельное обучение моделей с разными гиперпараметрами
pipe.add_function_step(
    name="train_exp1",
    function=train_catboost,
    function_kwargs=dict(
        data_paths_json="${load_data.data_paths_json}",  
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
        data_paths_json="${load_data.data_paths_json}",  
        depth=6,
        learning_rate=0.05,
        iterations=300
    ),
    parents=["load_data"],
    function_return=["model_path_exp2"]
)

# Оценка моделей
pipe.add_function_step(
    name="evaluate_exp1",
    function=evaluate_model,
    function_kwargs=dict(
        model_path="${train_exp1.model_path_exp1}",
        data_paths_json="${load_data.data_paths_json}",
        params_json='{"depth": 4, "learning_rate": 0.1, "iterations": 500}',
        output_file="/tmp/result_exp1.json"  # фиксированный путь
    ),
    parents=["train_exp1"],
    function_return=["result_file_exp1"]
)

pipe.add_function_step(
    name="evaluate_exp2",
    function=evaluate_model,
    function_kwargs=dict(
        model_path="${train_exp2.model_path_exp2}",
        data_paths_json="${load_data.data_paths_json}",
        params_json='{"depth": 6, "learning_rate": 0.05, "iterations": 300}',
        output_file="/tmp/result_exp2.json"  # фиксированный путь
    ),
    parents=["train_exp2"],
    function_return=["result_file_exp2"]
)

# Выбор лучшей модели
pipe.add_function_step(
    name="select_best",
    function=save_best_model,
    function_kwargs=dict(
        result_files_json='["/tmp/result_exp1.json", "/tmp/result_exp2.json"]',
        bucket_name="r-mlops-bucket-12-1-1-22209764",
        model_key="nil_project/models/ranker.pkl",
        endpoint_url="https://storage.yandexcloud.net"
    ),
    parents=["evaluate_exp1", "evaluate_exp2"],
    function_return=["best_info"]
)


if __name__ == "__main__":
    print("Pipeline start")
    pipe.start_locally(True)
    print("Pipeline finished")