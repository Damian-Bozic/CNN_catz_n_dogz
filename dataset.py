import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_data_loaders(batch_size=32):
    # Zaawansowane transformacje dla zbioru treningowego (Augmentacja w locie)
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(20),       # Losowy obrót o +/- 20 stopni
        transforms.RandomHorizontalFlip(),   # Odbicie lustrzane
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Przesunięcia i zoom
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Losowa zmiana jasności i kontrastu
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transformacje dla zbioru testowego (czyste dane, bez modyfikacji)
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Automatyczne wykrywanie ścieżki (w tym samym folderze lub poziom wyżej)
    root_train = 'training_data' if os.path.exists('training_data') else '../training_data'
    root_test = 'testing_data' if os.path.exists('testing_data') else '../testing_data'

    # Sprawdzenie czy foldery fizycznie istnieją przed załadowaniem
    if not os.path.exists(root_train) or not os.path.exists(root_test):
        raise FileNotFoundError(
            f"Nie znaleziono folderów z danymi! Upewnij się, że foldery "
            f"'training_data' oraz 'testing_data' są w odpowiednim miejscu.\n"
            f"Szukano w: {os.path.abspath(root_train)} oraz {os.path.abspath(root_test)}"
        )

    train_dataset = datasets.ImageFolder(root=root_train, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=root_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    images, labels = next(iter(train_loader))
    print(f"Paczka obrazów ma kształt (shape): {images.shape}")
    print("Wykryte klasy w projekcie:", train_loader.dataset.classes)
