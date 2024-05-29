# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:23:41 2024

@author: menna
"""
import uvicorn
from fastapi import FastAPI
#import numpy as np
import pickle
import re
import pandas as pd

app=FastAPI()
pickle_in=open("model4.pkl","rb")
model=pickle.load(pickle_in)


def extract_temp(x):
    matches = re.findall(r'\d+\.\d+', x)
    return float(matches[0]) if matches else None


@app.get('/')
def index():
    return {'message':'Hello,world'}

@app.get('/{name}')
def get_name(name: str):
    return {'welcome to my model': f'Hello,{name}'}

@app.post("/predict/")
async def predict_consumption(datetime: str, weather: str):
    # Data preprocessing for prediction
    new_data = pd.DataFrame({'datetime': [datetime], 'weather': [weather]})
    new_data['datetime'] = pd.to_datetime(new_data['datetime'])
    new_data['hour'] = new_data['datetime'].dt.hour
    new_data['dayofweek'] = new_data['datetime'].dt.dayofweek
    new_data['weather'] = new_data['weather'].apply(extract_temp)
    new_data = new_data.dropna()  # Drop rows with missing weather data
    new_X = new_data[['hour', 'dayofweek', 'weather']]
    
    # Prediction
    predicted_consumption = model.predict(new_X)
    return {"predicted_consumption": predicted_consumption[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
