from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import re
import pandas as pd

app = FastAPI()
pickle_in = open("model4.pkl", "rb")
model = pickle.load(pickle_in)

class ConsumptionRequest(BaseModel):
    datetime: str
    weather: str

def extract_temp(x):
    matches = re.findall(r'\d+\.\d+', x)
    return float(matches[0]) if matches else None

@app.get('/')
def index():
    return {'message': 'Hello, world'}

@app.get('/{name}')
def get_name(name: str):
    return {'welcome to my model': f'Hello, {name}'}

@app.post("/predict/")
async def predict_consumption(request: ConsumptionRequest):
    # Data preprocessing for prediction
    try:
        new_data = pd.DataFrame({'datetime': [request.datetime], 'weather': [request.weather]})
        new_data['datetime'] = pd.to_datetime(new_data['datetime'])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {e}")

    new_data['hour'] = new_data['datetime'].dt.hour
    new_data['dayofweek'] = new_data['datetime'].dt.dayofweek
    new_data['weather'] = new_data['weather'].apply(extract_temp)
    new_data = new_data.dropna()  # Drop rows with missing weather data

    if new_data.empty:
        raise HTTPException(status_code=400, detail="Missing or invalid weather data")

    new_X = new_data[['hour', 'dayofweek', 'weather']]
    
    # Prediction
    try:
        predicted_consumption = model.predict(new_X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {"predicted_consumption": predicted_consumption[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
