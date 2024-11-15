import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Function to create sequences
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length, :-1])  # All features except target
        labels.append(data[i+sequence_length, -1])  # The target value (next period)
    
    return np.array(sequences), np.array(labels)

# Load processed data
data = pd.read_json('processed_data.json')

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['economic_indicator', 'healthcare_expenditure', 'cases', 'recovery_rate']])

# Define sequence length (e.g., 30 days)
sequence_length = 30

# Create sequences for LSTM
X, y = create_sequences(scaled_data, sequence_length)

print(f"Created sequences of length {sequence_length}.")
