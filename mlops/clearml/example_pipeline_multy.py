import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from clearml import PipelineController, OutputModel, Task
import s3fs
import os
# import ast

# ============================================
# STEP 1: LOAD DATA
# ============================================
def load_data(dataset_path: str, bucket_name: str, endpoint_url: str):

    s3_url = f"s3://{bucket_name}/{dataset_path}"

    df = pd.read_csv(
        s3_url,
        storage_options={
            'client_kwargs': {'endpoint_url': endpoint_url},
            'config_kwargs': {
                's3': {
                    'addressing_style': 'path'
                }
            }
        }
    )
    
    df["distance"] = np.sqrt(
        (df["dropoff_longitude"] - df["pickup_longitude"]) ** 2
        + (df["dropoff_latitude"] - df["pickup_latitude"]) ** 2
    )
    
    X = df[["distance", "passenger_count"]]
    y = df["fare_amount"]
    
    mask = (y > 0) & (df["passenger_count"] > 0) & (df["passenger_count"] <= 6)
    X = X[mask]
    y = y[mask]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

# ============================================
# STEP 2: TRAIN MODEL
# ============================================
def train_model(X_train, y_train, model_type: str, **kwargs):
    if model_type == "LinearRegression":
        model = LinearRegression()
    elif model_type == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 50),
            max_depth=kwargs.get("max_depth", 10),
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model

# ============================================
# STEP 3: EVALUATE MODEL
# ============================================
def evaluate_model(model, X_test, y_test, model_name: str):
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    task = Task.current_task()
    if task:
        task.get_logger().report_scalar("metrics", f"{model_name}_rmse", value=rmse, iteration=0)
        task.get_logger().report_scalar("metrics", f"{model_name}_mae", value=mae, iteration=0)
        task.get_logger().report_scalar("metrics", f"{model_name}_r2", value=r2, iteration=0)
    
    return {
        "model_name": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "model": model
    }

# ============================================
# STEP 4: SELECT AND SAVE BEST MODEL
# ============================================
def select_and_save_best_model(
    lr_results,
    rf_results,
    bucket_name: str,
    model_key: str,
    endpoint_url: str
):
    results_list = [lr_results, rf_results]

    # Находим модель с минимальным RMSE
    best_result = min(results_list, key=lambda x: x["rmse"])
    best_model = best_result["model"]
    best_name = best_result["model_name"]
    best_rmse = best_result["rmse"]
    
    print(f"Best model: {bucket_name} with RMSE = {best_rmse:.4f}")
            
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": endpoint_url},
        config_kwargs={
            "s3": {
                "addressing_style": "path"
            }
        }
    )

    model_path = "s3://{bucket_name}/models/best_model.pkl"

    with fs.open(model_path, "wb") as f:
        joblib.dump(best_model, f)

    # Регистрируем в ClearML
    task = Task.current_task()
    output_model = OutputModel(
        task=task,
        framework="ScikitLearn",
        name=best_name,
        comment=f"Best model with RMSE = {best_rmse:.4f}",
        tags=["best_model", "linear_regression" if "Linear" in best_name else "random_forest"]
    )
    
    # Save temporary file for ClearML
    temp_path = f"/tmp/best_model_{best_name}.pkl"
    joblib.dump(best_model, temp_path, compress=True)
    output_model.update_weights(temp_path)
    os.remove(temp_path)
    
    print(f"Model registered in ClearML with ID: {output_model.id}")
    
    return {
        "best_model_name": best_name,
        "best_model_rmse": best_rmse,
        "s3_path": f"s3://{model_path}",
        "clearml_model_id": output_model.id
    }

# ============================================
# MAIN PIPELINE
# ============================================
def create_pipeline():
    """Create and configure ClearML pipeline"""
    
    pipe = PipelineController(
        name="Uber Price Prediction Pipeline",
        project="mlops",
        version="1.0.0",
        docker="python:3.12",
        add_pipeline_tags=True
    )
    
    # Default queue
    pipe.set_default_execution_queue(default_execution_queue="default")
    
    # Pipeline parameters
    pipe.add_parameter(
        name="dataset_path",
        default="uber_processed.csv"
    )
    
    pipe.add_parameter(
        name="bucket_name",
        default="r-mlops-bucket-8-1-11-22209764"
    )
    
    pipe.add_parameter(
        name="model_key",
        default="models/best_model.pkl"
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
            dataset_path="${pipeline.dataset_path}",
            bucket_name="${pipeline.bucket_name}",
            endpoint_url="${pipeline.endpoint_url}"
        ),
        function_return=["X_train", "X_test", "y_train", "y_test"],
        docker="python:3.12",
        packages=[
            "clearml[s3]==2.0.2",
            "boto3==1.34.0",
            "pandas==2.2.2",
            "scikit-learn==1.5.1",
            "s3fs==2024.10.0",
            "fsspec==2024.10.0",
            "numpy==1.26.3"
        ]
    )
    # Step 2a: Train LinearRegression
    pipe.add_function_step(
        name="train_lr",
        function=train_model,
        function_kwargs=dict(
            X_train="${load_data.X_train}",
            y_train="${load_data.y_train}",
            model_type="LinearRegression"
        ),
        function_return=["lr_model"],
        docker="python:3.12",
        packages=[
            "clearml[s3]==2.0.2",
            "boto3==1.34.0",
            "pandas==2.2.2",
            "scikit-learn==1.5.1",
            "s3fs==2024.10.0",
            "fsspec==2024.10.0",
            "numpy==1.26.3"
        ],
        parents=["load_data"]
    )
    
    # Step 2b: Train RandomForest
    pipe.add_function_step(
        name="train_rf",
        function=train_model,
        function_kwargs=dict(
            X_train="${load_data.X_train}",
            y_train="${load_data.y_train}",
            model_type="RandomForest",
            n_estimators=50,
            max_depth=10
        ),
        function_return=["rf_model"],
        docker="python:3.12",
        packages=[
            "clearml[s3]==2.0.2",
            "boto3==1.34.0",
            "pandas==2.2.2",
            "scikit-learn==1.5.1",
            "s3fs==2024.10.0",
            "fsspec==2024.10.0",
            "numpy==1.26.3"
        ],
        parents=["load_data"]
    )
        
    # Step 3a: Evaluate LinearRegression
    pipe.add_function_step(
        name="evaluate_lr",
        function=evaluate_model,
        function_kwargs=dict(
            model="${train_lr.lr_model}",
            X_test="${load_data.X_test}",
            y_test="${load_data.y_test}",
            model_name="LinearRegression"
        ),
        function_return=["lr_results"],
        docker="python:3.12",
        packages=[
            "clearml[s3]==2.0.2",
            "boto3==1.34.0",
            "pandas==2.2.2",
            "scikit-learn==1.5.1",
            "s3fs==2024.10.0",
            "fsspec==2024.10.0",
            "numpy==1.26.3"
        ],
        parents=["train_lr"],
    )
    
    # Step 3b: Evaluate RandomForest
    pipe.add_function_step(
        name="evaluate_rf",
        function=evaluate_model,
        function_kwargs=dict(
            model="${train_rf.rf_model}",
            X_test="${load_data.X_test}",
            y_test="${load_data.y_test}",
            model_name="RandomForest_100_20"
        ),
        function_return=["rf_results"],
        docker="python:3.12",
        packages=[
            "clearml[s3]==2.0.2",
            "boto3==1.34.0",
            "pandas==2.2.2",
            "scikit-learn==1.5.1",
            "s3fs==2024.10.0",
            "fsspec==2024.10.0",
            "numpy==1.26.3"
        ],
        parents=["train_rf"]
    )
    
    # Step 4: Select and save best model
    pipe.add_function_step(
        name="select_best",
        function=select_and_save_best_model,
        function_kwargs=dict(
            lr_results="${evaluate_lr.lr_results}",
            rf_results="${evaluate_rf.rf_results}",
            bucket_name="${pipeline.bucket_name}",
            model_key="${pipeline.model_key}",
            endpoint_url="${pipeline.endpoint_url}"
        ),        
        function_return=["best_info"],
        docker="python:3.12",
        packages=[
            "clearml[s3]==2.0.2",
            "boto3==1.34.0",
            "pandas==2.2.2",
            "scikit-learn==1.5.1",
            "s3fs==2024.10.0",
            "fsspec==2024.10.0",
            "numpy==1.26.3"
        ],
        parents=["evaluate_lr", "evaluate_rf"]
    )    

    return pipe


if __name__ == "__main__":
    
    # Вывод всех переменных окружения ClearML для контроля
    print("=" * 60)
    print("ClearML Environment Variables:")
    print("=" * 60)
    
    clearlm_vars = {
        "CLEARML_API_HOST": os.environ.get("CLEARML_API_HOST", "NOT SET"),
        "CLEARML_WEB_HOST": os.environ.get("CLEARML_WEB_HOST", "NOT SET"),
        "CLEARML_FILES_HOST": os.environ.get("CLEARML_FILES_HOST", "NOT SET"),
        "CLEARML_API_ACCESS_KEY": os.environ.get("CLEARML_API_ACCESS_KEY", "NOT SET"),
        "CLEARML_API_SECRET_KEY": os.environ.get("CLEARML_API_SECRET_KEY", "NOT SET")
    }
    
    for key, value in clearlm_vars.items():
        if "SECRET" in key or "KEY" in key and "ACCESS" not in key:
            # Маскируем секретные ключи
            if value != "NOT SET":
                masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"  {key}: {masked}")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    print("=" * 60)
    
    # Вывод S3 параметров
    print("S3/Storage Configuration:")
    print(f"  AWS_ACCESS_KEY_ID: {os.environ.get('AWS_ACCESS_KEY_ID', 'NOT SET')[:10]}...")
    print(f"  AWS_SECRET_ACCESS_KEY: {'*' * 10} (length: {len(os.environ.get('AWS_SECRET_ACCESS_KEY', ''))})")
    print(f"  BUCKET_NAME: {os.environ.get('BUCKET_NAME', 'NOT SET')}")
    print(f"  DATASET_PATH: {os.environ.get('DATASET_PATH', 'NOT SET')}")
    print(f"  MODEL_KEY: {os.environ.get('MODEL_KEY', 'NOT SET')}")
    print(f"  S3_ENDPOINT_URL: {os.environ.get('S3_ENDPOINT_URL', 'NOT SET')}")
    print("=" * 60)
    
    # Проверка наличия ключей S3
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        print("ERROR: AWS credentials not set!")
        exit(1)
    
    # Проверка наличия ключей ClearML
    if not os.environ.get("CLEARML_API_ACCESS_KEY") or not os.environ.get("CLEARML_API_SECRET_KEY"):
        print("WARNING: ClearML credentials not fully set!")
    else:
        print("ClearML credentials found") 
    # Запуск пайплайна
    pipe = create_pipeline()
    
    # Остальной код запуска
    print("\nStarting Uber Price Prediction Pipeline...")
    print("=" * 60)
    
    # Передаём параметры в start
    pipe.start(queue="default")
    
    print("\nPipeline submitted successfully!")