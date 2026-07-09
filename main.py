import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="California Housing Price Prediction API")

# Enable CORS so your HTML can talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("california_housing_model.joblib")
features = joblib.load("california_housing_features.joblib")


class HouseData(BaseModel):
    MedInc: float = Field(gt=0)
    HouseAge: float = Field(gt=0)
    AveRooms: float = Field(gt=0)
    AveBedrms: float = Field(gt=0)
    Population: float = Field(gt=0)
    AveOccup: float = Field(gt=0)
    Latitude: float = Field(ge=-90, le=90)
    Longitude: float = Field(ge=-180, le=180)


@app.get("/")
def home():
    return {
        "message": "California Housing Price Prediction API",
        "status": "Running"
    }


@app.get("/description")
def description():
    return {
        "model": "RandomForestRegressor",
        "features": features
    }


@app.post("/predict")
def predict(house: HouseData):
    try:
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])

        prediction = model.predict(input_data)[0]

        price = prediction * 100000
        return {
            "prediction": float(price),
            "confidence_low": float(price - 39000),
            "confidence_high": float(price + 39000)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
