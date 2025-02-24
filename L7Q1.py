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
        self.fc2 = nn.Linear(128, 2)  # Output layer for 2 classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Data preparation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Use FakeData with 2 classes for demonstration
dataset = datasets.FakeData(transform=transform, size=1000, num_classes=2)  # Ensure num_classes=2
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = CatDogClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(5):
    total_loss = 0.0
    for images, labels in train_loader:
        # Debugging: Check label values
        if labels.max().item() >= 2 or labels.min().item() < 0:
            raise ValueError(f"Invalid label found: {labels}")

        optimizer.zero_grad()
        outputs = model(images)

        # Debugging: Check output shape and label shape
        if outputs.shape[1] != 2:
            raise ValueError(f"Output shape mismatch: {outputs.shape}")

        loss = criterion(outputs, labels)  # Ensure labels are in [0, C-1]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


