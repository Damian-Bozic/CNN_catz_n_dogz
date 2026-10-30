import torch
import torch.nn as nn
import torch.optim as optim
from collections.abc import Callable
from dataset import get_data_loaders
from model import CatDogCNN


def train_model(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.001,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    model = CatDogCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting CNN training...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_train
        epoch_acc = (correct_train / total_train) * 100

        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = (correct_test / total_test) * 100
        print(

            f"Epoch [{epoch + 1}/{epochs}] -> Loss: {epoch_loss:.4f}",
            f"Training accuracy: {epoch_acc:.2f}%",
            f"Test accuracy: {test_acc:.2f}%",
            sep=" | "
        )

        if progress_callback is not None:
            progress_callback(epoch + 1, epochs)

    torch.save(model.state_dict(), "/tmp/cat_dog_cnn.pth")
    print("Training finished. Model saved to '/tmp/cat_dog_cnn.pth'.")


if __name__ == "__main__":
    train_model(epochs=10)
