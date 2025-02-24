import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import requests
import zipfile
from io import BytesIO


def download_dataset(url):
    # Create data directory
    os.makedirs('data', exist_ok=True)

    # Download and extract dataset
    print("Downloading dataset...")
    response = requests.get(url)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall('data')
    print("Dataset downloaded and extracted")


class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        # Get paths for both cats and dogs
        cat_dir = os.path.join(root_dir, 'cats')
        dog_dir = os.path.join(root_dir, 'dogs')

        # Add cat images (label 0)
        for img_name in os.listdir(cat_dir):
            self.images.append(os.path.join(cat_dir, img_name))
            self.labels.append(0)

        # Add dog images (label 1)
        for img_name in os.listdir(dog_dir):
            self.images.append(os.path.join(dog_dir, img_name))
            self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Download dataset
    url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    download_dataset(url)

    # Setup data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_data = CatDogDataset('data/cats_and_dogs_filtered/train', transform)
    val_data = CatDogDataset('data/cats_and_dogs_filtered/validation', transform)

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    # Load and modify AlexNet
    model = models.alexnet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, 2)
    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier[6].parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Train
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Loss: {running_loss / len(train_loader):.3f}')
        print(f'Validation Accuracy: {100 * correct / total:.2f}%\n')


if __name__ == "__main__":
    main()