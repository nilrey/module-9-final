from pydantic import BaseModel, Field

class PredictPurchaseRequest(BaseModel):
    itemid: int = Field(..., description="ID товара", example=356475)
    hour: int = Field(..., ge=0, le=23, description="Час (0-23)", example=14)
    weekday: int = Field(..., ge=0, le=6, description="День недели (0-6)", example=3)

class PredictPurchaseResponse(BaseModel):
    itemid: int = Field(..., description="ID товара")
    purchase_probability: float = Field(..., ge=0, le=1, description="Вероятность покупки")