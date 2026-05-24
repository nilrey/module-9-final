# Заполни меня правильно - тут основная логика сервиса.

import logging

from fastapi import FastAPI

from src.loaded_models import item_features, ranker

# Возможно надо еще импортировать prometheus, pydantic (схемы)
# from schemas import PredictPurchaseRequest, PredictPurchaseResponse, ErrorResponse
# from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

APP_VERSION = "0.0.1"

app = FastAPI(
    title="Retail Recommendation API",
    version=APP_VERSION,
    description="API для выдачи рекомендаций и вероятностей покупки.",
)


@app.get("/health", tags=["system"])
def health_check():
    """
    Проверка состояния сервиса и доступности моделей.
    """
    from loaded_models import item_features, ranker

    ok = {
        "ranker_loaded": ranker is not None,
        "items_in_feature_store": len(item_features),
    }
    logger.info(f"Health check: {ok}")
    return {"status": "ok", "details": ok}


@app.get("/version", tags=["system"])
def version():
    """
    Возвращает текущую версию сервиса.
    """
    return {"version": APP_VERSION, "service": "retail_recsys"}


@app.get("/", tags=["system"])
def root():
    return {"message": "Retail recommender API running", "version": APP_VERSION}


@app.get("/predict_purchase")
def predict_purchase(itemid: int = 98113, hour: int = 12, weekday: int = 3) -> dict:
    """Предсказываем вероятность покупки товара

    Note:
        Предсказания вероятности покупки товара в контретный день недели и время.

    Args:
        itemid: номер товара в базе данных.
        hour: час текущий.
        weekday: день недели текущий.
    """
    # Подсказка №1: pydantic схемы на входе нехватает
    row = item_features[item_features["itemid"] == itemid].copy()
    if row.empty:
        return {"error": "item not found"}

    row["hour"] = hour
    row["weekday"] = weekday
    x = row[
        ["views", "purchases", "ctr", "hour", "weekday", "categoryid", "available"]
    ].astype(float)
    prob = float(ranker.predict_proba(x)[0][1])
    # Подсказка №2: pydantic схемы в return нехватает
    return {"itemid": itemid, "purchase_probability": prob}


@app.get("/metrics", tags=["system"])
def metrics():
    """Надо как-то все логировать"""
    # Заполни меня - основная логика по которой мы отдаем метрики в prometheus.
    ...


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=32000, workers=1)
