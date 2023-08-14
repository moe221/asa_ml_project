from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))

from registry import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Preload the model to accelerate the predictions
# Load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes
app.state.model = load_model()

class PredictionInput(BaseModel):
    cpa_7_days: List[float]
    cpt_7_days: List[float]
    ttr_7_days: List[float]
    cvr_7_days: List[float]
    impressions_7_days:List[float]
    country_code: str
    latest_cpt: float
    daily_budget: float

class PredictionOutput(BaseModel):
    result1: float

@app.post("/predict")
async def predict(input_data: PredictionInput):

    """
    Make a single cpi target prediction.
    Assumes 7 day values provided as a list of floats is provided by the user
    Assumes `latest_cpt` implicitly refers to the daily average cpt on the day the prediciton is made
    """

    # calculate averages
    avg_cpa_rolling_mean_7d = np.mean(input_data.cpa_7_days)
    avg_cpt_rolling_mean_7d = np.mean(input_data.cpt_7_days)
    ttr_rolling_mean_7d = np.mean(input_data.ttr_7_days)
    impressions_rolling_mean_7d = np.mean(input_data.impressions_7_days)
    conversion_rate_rolling_mean_7d = np.mean(input_data.cvr_7_days)
    country_code = input_data.country_code
    avg_cpt = input_data.latest_cpt
    daily_budget = input_data.daily_budget


    data = {
    'avg_cpa_rolling_mean_7d':[avg_cpa_rolling_mean_7d],
    'avg_cpt_rolling_mean_7d':[avg_cpt_rolling_mean_7d],
    'ttr_rolling_mean_7d':[ttr_rolling_mean_7d],
    'impressions_rolling_mean_7d':[impressions_rolling_mean_7d],
    'conversion_rate_rolling_mean_7d':[conversion_rate_rolling_mean_7d],
    'country_code':[country_code],
    'avg_cpt':[avg_cpt],
    'daily_budget':[daily_budget]
    }

    rf_model = app.state.model
    assert rf_model is not None

    y_pred = rf_model.predict(pd.DataFrame(data=data))
    y_pred = round(y_pred[0], 2)

    return dict(cpi_target=float(y_pred))


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Project Ravioli - ASA campaign CPI target prediction API.")
    # $CHA_END
