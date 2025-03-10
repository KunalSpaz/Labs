import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

# Load dataset from a local CSV file
df = pd.read_csv('/home/student/Downloads/daily_csv.csv')

# Clean NaN values from the dataset
df = df.dropna(subset=['Price'])  # Remove rows where 'Price' is NaN

# Check if there are still any NaN values
if df['Price'].isnull().any():
    print("Data still contains NaN values after cleaning. Please check the dataset.")
    exit()

sequence_length = 10

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

# Assuming 'Price' is the column name for natural gas prices
X, y = create_sequences(df['Price'].values, sequence_length)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Normalize data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the last time step's output
        return out

# Initialize the model, define the loss function and the optimizer
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate

# Train the model
epochs = 50
losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor.view(-1, 1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Plot training loss
plt.figure(figsize=(12, 6))
plt.plot(losses, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict price for the next day (11th day)
model.eval()
with torch.no_grad():
    # Use the last sequence from the test set to predict the next price
    last_sequence = X_test_tensor[-1].view(1, sequence_length, 1)
    predicted_price = model(last_sequence).numpy()
    predicted_price_inverse = scaler.inverse_transform(predicted_price)

# Print the predicted price for the 11th day
print(f"Predicted Price for the 11th Day: {predicted_price_inverse[0][0]}")

# Predict prices for the test set
with torch.no_grad():
    predicted_prices = model(X_test_tensor).numpy()
    predicted_prices_inverse = scaler.inverse_transform(predicted_prices)
    y_test_inverse = scaler.inverse_transform(y_test_tensor.view(-1, 1).numpy())

# Plot predicted vs actual prices
plt.figure(figsize=(12, 6))
plt.plot(y_test_inverse, label='Actual Prices', color='blue')
plt.plot(predicted_prices_inverse, label='Predicted Prices', color='orange')
plt.title('Predicted vs Actual Prices')
plt.xlabel('Samples')
plt.ylabel('Price')
plt.legend()
plt.show()







