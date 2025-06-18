import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load Excel file
file_path = "data/market_data.xlsx"  # Update path if needed
xls = pd.ExcelFile(file_path)
df_us = xls.parse('US')

# Select relevant columns
selected_columns = ['Date', 'PE', 'CAPE', 'DY', 'EMP', '_MKT']
df_kf = df_us[selected_columns].copy()
df_kf.dropna(inplace=True)

# Set datetime index
df_kf['Date'] = pd.to_datetime(df_kf['Date'])
df_kf.set_index('Date', inplace=True)

# Normalize input features
features = ['PE', 'CAPE', 'DY', 'EMP']
scaler = StandardScaler()
df_kf[features] = scaler.fit_transform(df_kf[features])

# Prepare data
X = df_kf[features].values
Z = df_kf['_MKT'].values.reshape(-1, 1)

# Initialize Kalman Filter
n_timesteps = len(Z)
n_features = X.shape[1]

x_est = np.zeros((n_features, 1))          # Initial state estimate
P = np.eye(n_features)                     # Initial covariance
F = np.eye(n_features)                     # State transition matrix
Q = 1e-5 * np.eye(n_features)              # Process noise covariance
R = np.var(Z) * np.eye(1)                  # Measurement noise covariance
I = np.eye(n_features)

predicted_mkt = []

# Run Kalman Filter
for k in range(n_timesteps):
    # Predict
    x_pred = F @ x_est
    P = F @ P @ F.T + Q

    # Update
    H = X[k].reshape(1, -1)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    y = Z[k] - H @ x_pred
    x_est = x_pred + K @ y
    P = (I - K @ H) @ P

    # Save prediction
    predicted = H @ x_est
    predicted_mkt.append(predicted.item())

# Add predictions to DataFrame
df_kf['Predicted_MKT'] = predicted_mkt

# Optional: Save to CSV
# df_kf.to_csv("kalman_predictions.csv")

# Display the first few rows
print(df_kf.head())
