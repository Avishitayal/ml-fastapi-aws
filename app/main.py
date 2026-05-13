from fastapi import FastAPI
from pydantic import BaseModel
from ml.predictor import predict_house_price

app = FastAPI(
    title="House Price Prediction API",
    description="ML Regression API using FastAPI",
    version="1.0.0"
)

# Request schema
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: HouseFeatures):

    features = [
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]

    prediction = predict_house_price(features)

    return {
        "predicted_house_price": prediction
    }