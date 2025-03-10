import torch
import torch.nn as nn
import numpy as np

# Sample text data
text = "kunal verma"

# Create a set of unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Prepare the input and target sequences
seq_length = 3
input_sequences = []
target_sequences = []

for i in range(len(text) - seq_length):
    input_seq = text[i:i + seq_length]
    target_seq = text[i + seq_length]
    input_sequences.append([char_to_idx[ch] for ch in input_seq])
    target_sequences.append(char_to_idx[target_seq])

# Convert to numpy arrays
X = np.array(input_sequences)
y = np.array(target_sequences)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)


# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Get the output of the last time step
        return out


# Hyperparameters
embedding_dim = 10
hidden_dim = 20

# Initialize the model
model = RNNModel(vocab_size, embedding_dim, hidden_dim)

# Training parameters
num_epochs = 1000
learning_rate = 0.01

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# Function to predict the next character
def predict_next_char(model, input_seq):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([char_to_idx[ch] for ch in input_seq], dtype=torch.long).unsqueeze(
            0)  # Add batch dimension
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        return idx_to_char[predicted_idx.item()]


# Test the model
test_input = "ve"
predicted_char = predict_next_char(model, test_input)
print(f'Input: "{test_input}", Predicted next character: "{predicted_char}"')



