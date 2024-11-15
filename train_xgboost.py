import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load processed data
data = pd.read_json('processed_data.json')

# Prepare features and target
X = data[['economic_indicator', 'healthcare_expenditure', 'cases']]
y = data['recovery_rate']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define and train XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)

print(f"XGBoost Model evaluation - MSE: {mse}")
