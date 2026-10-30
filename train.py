import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from model import CatDogCNN


def train_model(epochs=5, batch_size=32, lr=0.001):
    # Wybór urządzenia (GPU/CUDA jeśli dostępne, inaczej CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie do treningu: {device}")

    # Załadowanie danych i inicjalizacja modelu
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    model = CatDogCNN().to(device)

    # Definicja funkcji straty i optymalizatora
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Rozpoczynam trening sieci CNN...")

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

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (correct_train / total_train) * 100

        # Ewaluacja na zbiorze testowym
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
            f"Epoka [{epoch + 1}/{epochs}] -> Strata: {epoch_loss:.4f} | Celność Train: {epoch_acc:.2f}% | Celność Test: {test_acc:.2f}%")

    torch.save(model.state_dict(), "cat_dog_cnn.pth")
    print("Trening zakończony! Model zapisano jako 'cat_dog_cnn.pth'")


if __name__ == "__main__":
    train_model(epochs=10)
