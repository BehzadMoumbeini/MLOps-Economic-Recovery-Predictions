import os
import json
import boto3
import pickle
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from xgboost import XGBRegressor



# Initialize Flask app
app = Flask(__name__)

# Load model (LSTM or XGBoost depending on choice)
model_lstm = tf.keras.models.load_model('lstm_model.h5')
model_xgb = pickle.load(open('xgboost_model.pkl', 'rb'))

# S3 initialization (optional if model is stored in S3)
s3_client = boto3.client('s3')

# Function to make predictions
def make_prediction(model, data):
    if isinstance(model, tf.keras.Model):
        # LSTM prediction
        data = np.array(data).reshape(1, data.shape[0], data.shape[1])
        return model.predict(data).tolist()
    elif isinstance(model, XGBRegressor):
        # XGBoost prediction
        return model.predict([data]).tolist()

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    model_type = request.json.get('model_type', 'lstm')  # Default to LSTM

    if model_type == 'lstm':
        prediction = make_prediction(model_lstm, np.array(data))
    elif model_type == 'xgboost':
        prediction = make_prediction(model_xgb, np.array(data))
    
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
