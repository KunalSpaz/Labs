import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Define the CNN architecture with correct dimensions
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 28x28 -> 26x26
        self.conv2 = nn.Conv2d(32, 64, 3)  # 13x13 -> 11x11
        self.conv3 = nn.Conv2d(64, 64, 3)  # 5x5 -> 3x3
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)  # Corrected input dimension
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Add print statements to debug dimensions
        x = self.pool(self.relu(self.conv1(x)))  # 26x26 -> 13x13
        x = self.pool(self.relu(self.conv2(x)))  # 11x11 -> 5x5
        x = self.relu(self.conv3(x))  # 3x3
        x = x.view(-1, 64 * 3 * 3)  # Flatten with correct dimensions
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_and_evaluate():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load Fashion-MNIST dataset
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Initialize the model
    model = ConvNet().to(device)

    # First train on MNIST
    mnist_train = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    mnist_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train on MNIST first
    print("Pre-training on MNIST...")
    model.train()
    for epoch in range(5):  # Fewer epochs for pre-training
        running_loss = 0.0
        for i, (images, labels) in enumerate(mnist_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'MNIST Epoch {epoch + 1}, Loss: {running_loss / len(mnist_loader):.3f}')

    # Now fine-tune on Fashion-MNIST
    print("\nFine-tuning on Fashion-MNIST...")
    # Reset optimizer for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for fine-tuning

    train_losses = []
    test_accuracies = []

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)

        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.3f}, Test Accuracy: {accuracy:.2f}%')

    return model, train_losses, test_accuracies


def plot_results(train_losses, test_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    # Plot test accuracy
    ax2.plot(test_accuracies)
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.show()


# Class labels for Fashion-MNIST
fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

if __name__ == "__main__":
    model, train_losses, test_accuracies = train_and_evaluate()
    plot_results(train_losses, test_accuracies)