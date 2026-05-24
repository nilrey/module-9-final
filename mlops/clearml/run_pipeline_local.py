import pandas as pd
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
    print("=== load_data START ===")

    df = pd.read_parquet(train_data_path)

    feature_cols = [
        "views",
        "purchases",
        "ctr",
        "hour",
        "weekday",
        "categoryid",
        "available"
    ]

    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("=== load_data END ===")

    return X_train, X_val, y_train, y_val


# ============================================
# STEP 2: TRAIN
# ============================================

def train_catboost(
    X_train,
    y_train,
    depth,
    learning_rate,
    iterations
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
        verbose=False,
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

def evaluate_model(model, X_val, y_val, params):
    print("=== evaluate_model START ===")
    print(f"Params: {params}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    
    print("Getting prediction probabilities...")

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    print(f"Predictions ready, shape: {y_pred_proba.shape}")
    
    print("Calculating PR-AUC...")

    pr_auc = average_precision_score(
        y_val,
        y_pred_proba
    )

    print(f"PR-AUC = {pr_auc:.6f}")

    task = Task.current_task()

    if task:
        print("Reporting metrics to ClearML...")
        task.get_logger().report_scalar(
            "metrics",
            "pr_auc",
            value=pr_auc,
            iteration=0
        )

    print("=== evaluate_model END ===")

    return pr_auc, params, model


# ============================================
# STEP 4: SELECT BEST
# ============================================

def select_and_save_best_model(
    experiment_results,
    bucket_name,
    model_key,
    endpoint_url
):
    print("=== select_and_save_best_model START ===")
    print(f"Number of experiment results: {len(experiment_results)}")

    for i, res in enumerate(experiment_results):
        print(
            f"Experiment {i+1}: "
            f"PR-AUC={res['pr_auc']:.6f}, "
            f"params={res['params']}"
        )

    best_result = max(
        experiment_results,
        key=lambda x: x["pr_auc"]
    )

    best_model = best_result["model"]
    best_pr_auc = best_result["pr_auc"]
    best_params = best_result["params"]
    
    print(f"Best model PR-AUC = {best_pr_auc:.6f}")
    print(f"Best params: {best_params}")
    
    print(f"Connecting to S3 at {endpoint_url}")

    fs = s3fs.S3FileSystem(
        client_kwargs={
            "endpoint_url": endpoint_url
        }
    )

    model_path = f"s3://{bucket_name}/{model_key}"

    with fs.open(model_path, "wb") as f:
        joblib.dump(best_model, f)

    output_model_id = None

    task = Task.current_task()

    if task:
        output_model = OutputModel(
            task=task,
            framework="CatBoost",
            name="catboost_ranker",
            comment=f"Best PR-AUC={best_pr_auc:.6f}",
            tags=["best_model"]
        )

        temp_path = "/tmp/best_model.pkl"

        joblib.dump(best_model, temp_path)

        output_model.update_weights(temp_path)

        os.remove(temp_path)

        output_model_id = output_model.id

    print("=== select_and_save_best_model END ===")

    return {
        "best_pr_auc": best_pr_auc,
        "best_params": best_params,
        "s3_path": model_path,
        "clearml_model_id": output_model_id
    }


# ============================================
# PIPELINE
# ============================================

# Создаём контроллер
pipe = PipelineController(
    name="CatBoost Hyperparameter Tuning",
    project="mlops",
    version="1.0.0"
)

# Добавляем шаги
pipe.add_function_step(
    name="load_data",
    function=load_data,
    function_kwargs=dict(
        train_data_path="/opt/clearml_data/data_for_training.parquet"
    ),
    function_return=[
        "X_train",
        "X_val",
        "y_train",
        "y_val"
    ]
)


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
    parents=["load_data"],
    function_return=["model_exp1"]
)


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
    parents=["load_data"],
    function_return=["model_exp2"]
)


pipe.add_function_step(
    name="evaluate_exp1",
    function=evaluate_model,
    function_kwargs=dict(
        model="${train_exp1.model_exp1}",
        X_val="${load_data.X_val}",
        y_val="${load_data.y_val}",
        params=dict(
            depth=4,
            learning_rate=0.1,
            iterations=500
        )
    ),
    parents=["train_exp1"],
    function_return=[
        "pr_auc_exp1",
        "params_exp1",
        "model_exp1_eval"
    ]
)


pipe.add_function_step(
    name="evaluate_exp2",
    function=evaluate_model,
    function_kwargs=dict(
        model="${train_exp2.model_exp2}",
        X_val="${load_data.X_val}",
        y_val="${load_data.y_val}",
        params=dict(
            depth=6,
            learning_rate=0.05,
            iterations=300
        )
    ),
    parents=["train_exp2"],
    function_return=[
        "pr_auc_exp2",
        "params_exp2",
        "model_exp2_eval"
    ]
)


pipe.add_function_step(
    name="select_best",
    function=select_and_save_best_model,
    function_kwargs=dict(
        experiment_results=[
            {
                "pr_auc": "${evaluate_exp1.pr_auc_exp1}",
                "params": "${evaluate_exp1.params_exp1}",
                "model": "${evaluate_exp1.model_exp1_eval}",
            },
            {
                "pr_auc": "${evaluate_exp2.pr_auc_exp2}",
                "params": "${evaluate_exp2.params_exp2}",
                "model": "${evaluate_exp2.model_exp2_eval}",
            }
        ],
        bucket_name="r-mlops-bucket-12-1-1-22209764",
        model_key="nil_project/models/ranker.pkl",
        endpoint_url="https://storage.yandexcloud.net"
    ),
    parents=[
        "evaluate_exp1",
        "evaluate_exp2"
    ],
    function_return=["best_info"]
)


if __name__ == "__main__":
    print("Running pipeline locally...")
    pipe.start_locally(True)
    print("Pipeline finished")