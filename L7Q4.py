import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Custom Dropout Layer using Bernoulli Distribution
class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super(CustomDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training:  # No dropout during evaluation
            return x
        # Generate a binary mask using Bernoulli distribution
        mask = (torch.rand_like(x) > self.p).float() / (1 - self.p)  # Scale by keep probability
        return mask * x


# Define the neural network with custom dropout
class CatDogClassifierWithCustomDropout(nn.Module):
    def __init__(self):
        super(CatDogClassifierWithCustomDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # Adjusted for input size
        self.custom_dropout = CustomDropout(p=0.5)  # Custom dropout layer
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.custom_dropout(x)  # Apply custom dropout during training
        x = self.fc2(x)
        return x


# Data preparation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Use FakeData for demonstration (replace with your actual dataset)
dataset = datasets.FakeData(transform=transform, size=1000, num_classes=2)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model with custom dropout and optimizer
model_with_custom_dropout = CatDogClassifierWithCustomDropout()
criterion = nn.CrossEntropyLoss()
optimizer_with_custom_dropout = optim.SGD(model_with_custom_dropout.parameters(), lr=0.01)

# Training loop with custom dropout
for epoch in range(5):
    total_loss_custom_dropout = 0.0
    for images, labels in train_loader:
        optimizer_with_custom_dropout.zero_grad()

        outputs_with_custom_dropout = model_with_custom_dropout(images)

        loss_with_custom_dropout = criterion(outputs_with_custom_dropout, labels)

        loss_with_custom_dropout.backward()

        optimizer_with_custom_dropout.step()

        total_loss_custom_dropout += loss_with_custom_dropout.item()

    print(f"Epoch {epoch + 1}, Loss with Custom Dropout: {total_loss_custom_dropout:.4f}")
