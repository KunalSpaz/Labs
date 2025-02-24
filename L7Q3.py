import torch
import torch.nn as nn  # Import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define the neural network with dropout
class CatDogClassifierWithDropout(nn.Module):
    def __init__(self):
        super(CatDogClassifierWithDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # Adjusted for input size
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with probability 0.5
        self.fc2 = nn.Linear(128, 2)  # Output layer for binary classification

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout during training
        x = self.fc2(x)
        return x


# Data preparation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Use FakeData with 2 classes for demonstration (replace with your actual dataset)
dataset = datasets.FakeData(transform=transform, size=1000, num_classes=2)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model_with_dropout = CatDogClassifierWithDropout()
criterion = nn.CrossEntropyLoss()
optimizer_with_dropout = optim.SGD(model_with_dropout.parameters(), lr=0.01)

# Training loop with dropout regularization
for epoch in range(5):
    total_loss_with_dropout = 0.0
    for images, labels in train_loader:
        optimizer_with_dropout.zero_grad()

        outputs_with_dropout = model_with_dropout(images)

        loss_with_dropout = criterion(outputs_with_dropout, labels)

        loss_with_dropout.backward()

        optimizer_with_dropout.step()

        total_loss_with_dropout += loss_with_dropout.item()

    print(f"Epoch {epoch + 1}, Loss with Dropout: {total_loss_with_dropout:.4f}")

