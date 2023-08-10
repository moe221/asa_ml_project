from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import numpy as np
from typing import List, Dict
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from registry import load_model


app = FastAPI()

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
    # Add more fields as needed

@app.post("/predict")
async def predict(input_data: PredictionInput):

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

    rf_model = load_model()

    prediction = rf_model.predict(pd.DataFrame(data=data))
    res = round(prediction[0], 2)

    return {"prediction":res}
