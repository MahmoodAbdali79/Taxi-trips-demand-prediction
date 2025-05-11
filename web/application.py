import os
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conint

from src.config_reader import read_config
from src.logger import get_logger

"""
FastAPI web application for predicting taxi trip demand using a trained machine learning model.

This service:
- Loads a trained RandomForestRegressor model.
- Provides a REST API endpoint to receive request features and return predicted trip demand.
- Logs API activity using a custom logger.
"""


config_path = "config/config.yaml"
web_config = read_config(config_path)["web"]

model_dir = web_config["model_output_dir"]
model_name = web_config["model_name"]
model = joblib.load(Path(model_dir) / model_name)

app = FastAPI()
logger = get_logger(__name__)


class DemandRequest(BaseModel):
    """
    Input schema for taxi demand prediction.

    Attributes:
        pickup_hour (int): Hour of pickup time (0–23).
        pickup_day (int): Day of the month (1–31).
        pickup_weekday (int): Day of the week (0 = Monday, 6 = Sunday).
        is_weekend (int): Indicator whether the day is a weekend (0 or 1).
        PUlocationID (int): Pickup location ID (1–265).
    """

    pickup_hour: conint(ge=0, le=23)
    pickup_day: conint(ge=0, le=31)
    pickup_weekday: conint(ge=0, le=6)
    is_weekend: conint(le=1)
    PUlocationID: conint(ge=1, le=265)


class DemandResponse(BaseModel):
    """
    Output schema for taxi demand prediction.

    Attributes:
        trip_count (int): Predicted number of trips for the specified time and location.
    """

    trip_count: int


@app.post("/predict", response_model=DemandResponse)
def predict_demand(request: DemandRequest):
    """
    Predict the number of taxi trips given pickup time and location features.

    Receives a JSON payload with the following fields:
    - pickup_hour
    - pickup_day
    - pickup_weekday
    - is_weekend
    - PUlocationID

    Returns:
        JSON response containing:
        - trip_count (rounded predicted number of trips)
    """
    logger.info("The predict link touched")
    features = pd.DataFrame(
        [
            {
                "pickup_day": request.pickup_day,
                "pickup_hour": request.pickup_hour,
                "pickup_weekday": request.pickup_weekday,
                "is_weekend": request.is_weekend,
                "PUlocationID": request.PUlocationID,
            }
        ]
    )
    prediction = model.predict(features)[0]

    return {"trip_count": round(prediction)}


if __name__ == "__main__":
    logger.info("Application loaded")
    uvicorn.run(
        "web.application:app",
        host=os.environ.get("WEB_HOST", web_config["host"]),
        port=int(os.environ.get("WEB_PORT", web_config["port"])),
        reload=True,
    )
