import sys
import os
import torch
from PIL import Image
from torchvision import transforms
from model import CatDogCNN


def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Błąd: Plik '{image_path}' nie istnieje!")
        return

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Błąd podczas otwierania pliku: {e}")
        return

    image_tensor = transform(image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatDogCNN().to(device)

    # Wskaż nową ścieżkę w /tmp/
    model_path = "/tmp/cat_dog_cnn.pth"

    if not os.path.exists(model_path):
        print(
            f"Błąd: Brak pliku wag '{model_path}'.",
            "Poczekaj na zakończenie treningu!"
        )
        return

    model.load_state_dict(torch.load(model_path, map_location=device))

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    classes = ['kot', 'pies']
    result = classes[predicted_idx.item()]
    percentage = confidence.item() * 100

    print(
        f"\n[WYNIK]: Model uważa na {percentage:.2f}%,",
        f"że to zdjęcie przedstawia: **{result.upper()}**"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: uv run python3 predict.py <sciezka_do_zdjecia.jpg>")
    else:
        predict_image(sys.argv[1])
