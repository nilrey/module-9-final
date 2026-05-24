


Preprocessing в контексте проекта — это этап, который выполняется в Airflow DAG. Он делает:

    Загружает сырые данные из S3 (events.csv, item_properties_part1.csv, item_properties_part2.csv)

    Объединяет два файла свойств в один (item_properties_merged.csv)

    Выполняет трансформации через prepare_features.py:

        вычисляет популярность товаров (views, purchases, CTR)

        извлекает признаки categoryid, available

        создаёт целевую переменную target (купил ли пользователь товар в течение 24 часов после просмотра)

        добавляет контекстные признаки (hour, weekday)

    Сохраняет обработанные данные в S3:

        item_features.parquet — признаки товаров

        data_for_training.parquet — обучающая выборка с таргетом

В ClearML пайплайне нужно загрузить готовые data_for_training.parquet из S3 и сразу перейти к обучению и перебору гиперпараметров.

Таким образом, получено разделение:

    Airflow — подготовка данных (тяжёлые, однократные операции)

    ClearML — эксперименты и обучение (итеративно, много запусков)




У нас есть data_for_training.parquet и item_features.parquet для чего item_features.parquet ?
item_features.parquet нужен для инференса (работы сервиса), а не для обучения.

В обучении (train_catboost) используются:

    data_for_training.parquet — содержит признаки товаров + таргет

В инференсе (FastAPI сервис):

    item_features.parquet — загружается в loaded_models.py, чтобы по itemid получить признаки товара (views, purchases, ctr, categoryid, available)

    ranker.pkl — модель

Таким образом:

    Airflow DAG создаёт оба файла

    ClearML пайплайн использует только data_for_training.parquet

    FastAPI сервис использует item_features.parquet + ranker.pkl