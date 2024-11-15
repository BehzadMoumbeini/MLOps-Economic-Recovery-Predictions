import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json

# Load dataset
data = pd.read_csv('economic_healthcare_data.csv')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Feature scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['economic_indicator', 'healthcare_expenditure', 'cases', 'recovery_rate']])

# Convert data to JSON
data_json = data.to_json(orient='records', lines=True)

with open('processed_data.json', 'w') as f:
    f.write(data_json)

print("Data preprocessing complete and saved to JSON format.")


