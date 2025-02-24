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


# Data preparation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.FakeData(transform=transform, size=1000, num_classes=2)  # Replace with actual dataset
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model and optimizer
model = CatDogClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop with L1 regularization
l1_lambda = 0.01

for epoch in range(5):
    total_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)

        # Standard loss
        loss = criterion(outputs, labels)

        # Add L1 penalty to loss
        l1_penalty = sum(torch.norm(param, p=1) for param in model.parameters())
        loss += l1_lambda * l1_penalty

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# Calculate L1 norm of weights manually (loop-based approach)
l1_norm = sum(torch.norm(param, p=1).item() for param in model.parameters())
print(f"L1 Norm of weights: {l1_norm}")

