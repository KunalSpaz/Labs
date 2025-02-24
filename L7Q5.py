import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define the neural network
class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # Adjusted for input size
        self.fc2 = nn.Linear(128, 2)  # Output layer for binary classification

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Data preparation (split into training and validation sets)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Using FakeData for demonstration (replace with your actual dataset)
train_dataset = datasets.FakeData(transform=transform, size=800, num_classes=2)  # Ensure num_classes=2
val_dataset = datasets.FakeData(transform=transform, size=200, num_classes=2)  # Ensure num_classes=2
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize model and optimizer
model_for_early_stopping = CatDogClassifier()
criterion = nn.CrossEntropyLoss()
optimizer_for_early_stopping = optim.SGD(model_for_early_stopping.parameters(), lr=0.01)

# Early stopping parameters
best_val_loss = float('inf')  # Initialize best validation loss to infinity
patience_counter = 0  # Counter for early stopping patience
patience_limit = 3  # Number of epochs to wait before stopping

# Training loop with early stopping
for epoch in range(50):  # Train for many epochs to monitor early stopping
    model_for_early_stopping.train()  # Set model to training mode

    total_train_loss = 0.0
    for images, labels in train_loader:
        # Debugging: Check label values
        if labels.max().item() >= 2 or labels.min().item() < 0:
            raise ValueError(f"Invalid label found: {labels}")

        optimizer_for_early_stopping.zero_grad()

        outputs_train = model_for_early_stopping(images)

        loss_train = criterion(outputs_train, labels)

        loss_train.backward()

        optimizer_for_early_stopping.step()

        total_train_loss += loss_train.item()

    # Validation phase
    model_for_early_stopping.eval()  # Set model to evaluation mode

    total_val_loss = 0.0
    with torch.no_grad():
        for images_val, labels_val in val_loader:
            outputs_val = model_for_early_stopping(images_val)
            loss_val = criterion(outputs_val, labels_val)
            total_val_loss += loss_val.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch + 1}, Train Loss: {total_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Check early stopping condition
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0  # Reset patience counter if validation loss improves
    else:
        patience_counter += 1

    if patience_counter >= patience_limit:
        print("Early stopping triggered.")
        break

print("Training complete.")


