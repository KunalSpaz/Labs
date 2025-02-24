import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

# Define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(-1, 64 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename="checkpoint.pth"):
    """Save checkpoint with model state, optimizer state, epoch, and metrics."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    """Load checkpoint and return epoch and metrics."""
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded: {filename}")
        return (checkpoint['epoch'], checkpoint['loss'],
                checkpoint['accuracy'])
    return 0, float('inf'), 0

def train_model(num_epochs=5, resume=False):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                            shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                           shuffle=False)

    # Initialize model and optimizer
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Initialize tracking variables
    best_accuracy = 0
    start_epoch = 0
    best_loss = float('inf')

    # Load checkpoint if resuming training
    if resume:
        start_epoch, best_loss, best_accuracy = load_checkpoint(
            model, optimizer, "checkpoint.pth")
        start_epoch += 1  # Start from next epoch

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate epoch loss
        epoch_loss = running_loss / len(trainloader)

        # Evaluate accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Loss: {epoch_loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}%')

        # Save regular checkpoint
        save_checkpoint(model, optimizer, epoch, epoch_loss, accuracy)

        # Save best model if accuracy improves
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(model, optimizer, epoch, epoch_loss, accuracy,
                          "best_model.pth")
            print(f"Best model saved with accuracy: {accuracy:.2f}%")

def main():
    # Train from scratch
    print("Starting training from scratch...")
    train_model(num_epochs=10, resume=False)

    # Demonstrate resuming training
    print("\nResuming training from checkpoint...")
    train_model(num_epochs=15, resume=True)

if __name__ == "__main__":
    main()