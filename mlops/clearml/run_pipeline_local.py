from clearml import PipelineController

# Импортируем твои функции из основного файла
from clearml_pipeline import load_data, train_catboost, evaluate_model, select_and_save_best_model

# Создаём контроллер без очереди
pipe = PipelineController(
    name="CatBoost Hyperparameter Tuning (Local)",
    project="mlops",
    version="1.0.0"
)

# Добавляем шаги (без queue, без parents)
pipe.add_function_step(
    name="load_data",
    function=load_data,
    function_kwargs=dict(
        train_data_path="/opt/clearml_data/data_for_training.parquet"
    ),
    function_return=["X_train", "X_val", "y_train", "y_val"]
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
    function_return=["model_exp2"]
)

pipe.add_function_step(
    name="evaluate_exp1",
    function=evaluate_model,
    function_kwargs=dict(
        model="${train_exp1.model_exp1}",
        X_val="${load_data.X_val}",
        y_val="${load_data.y_val}",
        params={"depth": 4, "learning_rate": 0.1, "iterations": 500}
    ),
    function_return=["result_exp1"]
)

pipe.add_function_step(
    name="evaluate_exp2",
    function=evaluate_model,
    function_kwargs=dict(
        model="${train_exp2.model_exp2}",
        X_val="${load_data.X_val}",
        y_val="${load_data.y_val}",
        params={"depth": 6, "learning_rate": 0.05, "iterations": 300}
    ),
    function_return=["result_exp2"]
)

pipe.add_function_step(
    name="select_best",
    function=select_and_save_best_model,
    function_kwargs=dict(
        experiment_results=[
            "${evaluate_exp1.result_exp1}",
            "${evaluate_exp2.result_exp2}"
        ],
        bucket_name="r-mlops-bucket-12-1-1-22209764",
        model_key="nil_project/models/ranker.pkl",
        endpoint_url="https://storage.yandexcloud.net"
    ),
    function_return=["best_info"]
)

# Запуск локально (без агента)
if __name__ == "__main__":
    print("Running pipeline locally...")
    pipe.run_locally()
    print("Pipeline finished")