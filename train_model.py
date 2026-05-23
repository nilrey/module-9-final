# Подсказка №1. Попробуйте повторить данный скрипт обучения при помощи ClearML. Используйте optuna для подбора гиперпараметров. 
# Подсказка №2. Вам необходимо дополнительно создать тестовую выборку для валидации модели.
# Подсказка №3. Валидацию делаем по PR-AUC (выборка содержит очень мало положительных классов).
# Подсказка №4. Если ClearML крутиться на ВМ, то может быть неудобно работать с Model Registry. Однако, модель обязательно сохраняем на S3.

import pandas as pd
import pickle

from catboost import CatBoostClassifier, Pool
from pathlib import Path


def main():
    """Самый базовый скрипт для обучения модели.

    Note:
        Не забываем что лучше использовать ClearML\MLFlow
    """
    df = pd.read_parquet(Path("./data_for_training.parquet"))

    # Target
    y = df["target"]

    # Features
    feature_cols = [
        "views",
        "purchases",
        "ctr",
        "hour",
        "weekday",
        "categoryid",
        "available",
    ]
    X = df[feature_cols].astype(float)

    model = CatBoostClassifier(
        depth=4, learning_rate=0.1, loss_function="Logloss", verbose=False
    )

    train_pool = Pool(X, y)
    model.fit(train_pool)

    with open(Path("./ranker.pkl"), "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
