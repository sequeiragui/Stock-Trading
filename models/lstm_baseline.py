import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        self.dropout = nn.Dropout(0.2)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

# Load your dataset
df = pd.read_excel("data/market_data.xlsx", sheet_name="US")

# Select and preprocess data
features = ['PE', 'CAPE', 'DY', 'EMP']
target = '_MKT'
df = df[['Date'] + features + [target]].dropna()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Normalize
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(df[features])
y = scaler_y.fit_transform(df[[target]])

# Create sequences
sequence_length = 4
X_seq, y_seq = [], []
for i in range(len(X) - sequence_length):
    X_seq.append(X[i:i+sequence_length])
    y_seq.append(y[i+sequence_length])

X_seq = torch.tensor(X_seq, dtype=torch.float32)
y_seq = torch.tensor(y_seq, dtype=torch.float32)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Model setup
input_size = len(features)
hidden_size = 50
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(1000):  
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Inference
model.eval()
with torch.no_grad():
    predictions = model(X_test)

# Inverse transform
predicted = scaler_y.inverse_transform(predictions.numpy())
actual = scaler_y.inverse_transform(y_test.numpy())

# Save results
result_df = pd.DataFrame({
    'Actual_MKT': actual.flatten(),
    'Predicted_MKT_LSTM': predicted.flatten()
}, index=df.index[-len(predicted):])

print(result_df.head())


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Flatten arrays just in case
actual = actual.flatten()
predicted = predicted.flatten()

# === Regression Metrics ===
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print(f"\n📊 Regression Metrics:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")

# === Directional Accuracy ===
prev_actual = actual[:-1]
actual_change = actual[1:] - prev_actual
predicted_change = predicted[1:] - prev_actual

direction_correct = np.sign(actual_change) == np.sign(predicted_change)
direction_accuracy = np.mean(direction_correct) * 100

print(f"\n🔁 Directional Accuracy: {direction_accuracy:.2f}%")
