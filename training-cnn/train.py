import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch

# ------------------
# MLflow config
# ------------------
mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("FashionMNIST-CNN")

# ------------------
# Device
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------
# Dataset
# ------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ------------------
# CNN Model
# ------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = CNN().to(device)

# ------------------
# Training setup
# ------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

# ------------------
# Training loop
# ------------------
with mlflow.start_run():
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("optimizer", "Adam")

    for epoch in range(epochs):
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
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # ------------------
    # Evaluation
    # ------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    mlflow.log_metric("accuracy", accuracy)

    print(f"Accuracy: {accuracy:.4f}")

    # ------------------
    # Save model
    # ------------------
    mlflow.pytorch.log_model(model, "cnn_fashion_mnist")
