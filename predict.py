import sys
import os
import torch
from PIL import Image
from torchvision import transforms
from model import CatDogCNN


def predict_image(image_path):
    # 1. Sprawdzenie czy plik istnieje
    if not os.path.exists(image_path):
        print(f"Błąd: Plik '{image_path}' nie istnieje!")
        return

    # 2. Definicja transformacji identycznych jak przy zbiorze testowym
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. Wczytanie i przygotowanie obrazu
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Błąd podczas otwierania pliku: {e}")
        return

    image_tensor = transform(image).unsqueeze(
        0)  # Dodanie wymiaru paczki (batch size = 1)

    # 4. Inicjalizacja modelu i wczytanie wag
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatDogCNN().to(device)

    if not os.path.exists("cat_dog_cnn.pth"):
        print(
            "Błąd: Brak pliku wag 'cat_dog_cnn.pth'. Poczekaj na zakończenie treningu!")
        return

    model.load_state_dict(torch.load("cat_dog_cnn.pth", map_location=device))
    model.eval()

    # 5. Predykcja
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    # 6. Wyświetlenie wyniku
    classes = ['kot', 'pies']
    result = classes[predicted_idx.item()]
    percentage = confidence.item() * 100

    print(
        f"\n[WYNIK]: Model uważa na {percentage:.2f}%, że to zdjęcie przedstawia: **{result.upper()}**")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: uv run python3 predict.py <sciezka_do_zdjecia.jpg>")
    else:
        predict_image(sys.argv[1])
