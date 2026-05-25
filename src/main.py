import logging
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
from src.schemas import PredictPurchaseRequest, PredictPurchaseResponse

from src.loaded_models import ranker, item_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

APP_VERSION = "0.0.1"


REQUESTS = Counter(
    'http_requests_total', 
    'Total HTTP requests', 
    ['method', 'endpoint', 'status']
)
LATENCY = Histogram(
    'http_request_duration_seconds', 
    'HTTP request latency', 
    ['method', 'endpoint']
)
ERRORS = Counter(
    'http_errors_total', 
    'Total HTTP errors', 
    ['method', 'endpoint', 'error_type']
)
PREDICTION_SCORES = Histogram(
    'prediction_scores', 
    'Distribution of prediction scores (purchase probability)',
    buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
SERVICE_UP = Gauge('service_up', 'Service health status (1=up, 0=down)')

app = FastAPI(
    title="Retail Recommendation API",
    version=APP_VERSION,
    description="API для выдачи рекомендаций и вероятностей покупки.",
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration)
        REQUESTS.labels(
            method=request.method, 
            endpoint=request.url.path, 
            status=response.status_code
        ).inc()
        
        if response.status_code >= 400:
            ERRORS.labels(
                method=request.method, 
                endpoint=request.url.path, 
                error_type=str(response.status_code)
            ).inc()
        
        return response
    except Exception as e:
        duration = time.time() - start_time
        LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration)
        ERRORS.labels(
            method=request.method, 
            endpoint=request.url.path, 
            error_type=type(e).__name__
        ).inc()
        raise


@app.on_event("startup")
async def startup_event():
    SERVICE_UP.set(1)
    logger.info("Service started, metrics initialized")

@app.on_event("shutdown")
async def shutdown_event():
    SERVICE_UP.set(0)
    logger.info("Service stopped")

@app.get("/health", tags=["system"])
def health_check():
    # Проверка состояния сервиса 
    ok = {
        "ranker_loaded": ranker is not None,
        "items_in_feature_store": len(item_features),
    }
    logger.info(f"Health check: {ok}")
    return {"status": "ok", "details": ok}

@app.get("/version", tags=["system"])
def version():
    # текущая версия сервиса
    return {"version": APP_VERSION, "service": "retail_recsys"}

@app.get("/", tags=["system"])
def root():
    return {"message": "Retail recommender API running", "version": APP_VERSION}



@app.post("/predict_purchase", response_model=PredictPurchaseResponse)
def predict_purchase(request: PredictPurchaseRequest):
    """Предсказываем вероятность покупки товара

    Note:
        Предсказания вероятности покупки товара в контретный день недели и время.

    Args:
        itemid: номер товара в базе данных.
        hour: час текущий.
        weekday: день недели текущий.
    """
    row = item_features[item_features["itemid"] == request.itemid].copy()
    
    if row.empty:
        logger.warning(f"Item {request.itemid} not found")
        return JSONResponse(status_code=404, content={"error": "item not found"})

    row["hour"] = request.hour
    row["weekday"] = request.weekday
    x = row[
        ["views", "purchases", "ctr", "hour", "weekday", "categoryid", "available"]
    ].astype(float)
    
    prob = float(ranker.predict_proba(x)[0][1])
    
    PREDICTION_SCORES.observe(prob)
    
    logger.info(f"Prediction: itemid={request.itemid}, hour={request.hour}, weekday={request.weekday}, prob={prob:.4f}")
    
    return PredictPurchaseResponse(itemid=request.itemid, purchase_probability=prob)


@app.get("/metrics", tags=["system"])
def metrics():
    #отдаем метрики в prometheus.
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=32000, workers=1)
