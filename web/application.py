import os
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conint

from src.config_reader import read_config
from src.logger import get_logger

config_path = "config/config.yaml"
web_config = read_config(config_path)["web"]

model_dir = web_config["model_output_dir"]
model_name = web_config["model_name"]
model = joblib.load(Path(model_dir) / model_name)

app = FastAPI()
logger = get_logger(__name__)


class DemandRequest(BaseModel):
    hour_of_day: conint(ge=0, le=23)
    day: conint(ge=0)
    row: conint(ge=0, le=7)
    col: conint(ge=0, le=7)


class DemandResponse(BaseModel):
    demand: int


@app.post("/predict", response_model=DemandResponse)
def predict_demand(request: DemandRequest):
    logger.info("The predict link touched")
    features = pd.DataFrame(
        [
            {
                "hour_of_day": request.hour_of_day,
                "day": request.day,
                "row": request.row,
                "col": request.col,
            }
        ]
    )
    prediction = model.predict(features)[0]

    return {"demand": round(prediction)}


if __name__ == "__main__":
    logger.info("Application loaded")
    uvicorn.run(
        "web.application:app",
        host=os.environ.get("WEB_HOST", web_config["host"]),
        port=int(os.environ.get("WEB_PORT", web_config["port"])),
        # reload=True
    )
