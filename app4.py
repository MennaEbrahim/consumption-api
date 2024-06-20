# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 03:33:03 2024

@author: menna
"""

from flask import Flask, request, jsonify
import pickle
import re
import pandas as pd
from datetime import datetime, timedelta
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained model
model_file_path = "model4.pkl"

if not os.path.exists(model_file_path):
    logging.error(f"Pickle file not found: {model_file_path}")
    raise Exception(f"Pickle file not found: {model_file_path}")

try:
    with open(model_file_path, "rb") as pickle_in:
        model = pickle.load(pickle_in)
        logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading pickle file: {e}")
    raise Exception(f"Error loading pickle file: {e}")

def extract_temp(x):
    matches = re.findall(r'\d+\.\d+', x)
    return float(matches[0]) if matches else None

def generate_future_dates(start_date, days):
    return [start_date + timedelta(days=i) for i in range(days)]

def prepare_input_data(start_date, weather):
    future_dates = generate_future_dates(start_date, 7)
    future_data = pd.DataFrame({'datetime': future_dates})
    future_data['datetime'] = pd.to_datetime(future_data['datetime'])
    future_data['hour'] = future_data['datetime'].dt.hour
    future_data['dayofweek'] = future_data['datetime'].dt.dayofweek
    future_data['weather'] = weather  # Assuming the weather is constant for simplicity
    future_data['weather'] = future_data['weather'].apply(extract_temp)
    future_data = future_data.dropna()  # Drop rows with missing weather data
    return future_data[['hour', 'dayofweek', 'weather']], future_dates

@app.route('/')
def index():
    return jsonify({'message': 'Hello, world'})

@app.route('/<name>')
def get_name(name: str):
    return jsonify({'welcome to my model': f'Hello, {name}'})

@app.route('/predict/', methods=['POST'])
def predict_next_week_consumption():
    # Data preprocessing for prediction
    data = request.get_json()
    try:
        start_date = datetime.fromisoformat(data['datetime'])
        weather = data['weather']
        
        future_data, future_dates = prepare_input_data(start_date, weather)
    except ValueError as e:
        return jsonify({'detail': f"Invalid input: {e}"}), 400

    if future_data.empty:
        return jsonify({'detail': "Missing or invalid weather data"}), 400

    # Prediction
    try:
        predicted_consumption = model.predict(future_data)
    except Exception as e:
        return jsonify({'detail': f"Prediction error: {e}"}), 500

    results = [
        {
            'datetime': date.isoformat(),
            'predicted_consumption': prediction
        } for date, prediction in zip(future_dates, predicted_consumption)
    ]

    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

