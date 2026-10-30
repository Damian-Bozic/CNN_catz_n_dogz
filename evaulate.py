import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from collections.abc import Callable
from sklearn.metrics import classification_report, confusion_matrix, \
    ConfusionMatrixDisplay
from dataset import get_data_loaders
from model import CatDogCNN


def evaluate_model(
    progress_callback: Callable[[int, int], None] | None = None,
):
    # 1. Ustawienie urządzenia
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ewaluacja na urządzeniu: {device}")

    # 2. Załadowanie danych testowych i modelu
    _, test_loader = get_data_loaders(batch_size=64)
    model = CatDogCNN().to(device)

    # Wskaż nową ścieżkę w /tmp/
    model_path = "/tmp/cat_dog_cnn.pth"

    if not os.path.exists(model_path):
        print(
            f"Błąd: Nie znaleziono pliku wag '{model_path}'. Najpierw wytrenuj model!")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    all_preds = []
    all_labels = []

    print("Zbieranie predykcji ze zbioru testowego...")
    total_batches = len(test_loader)
    with torch.no_grad():
        for batch_number, (images, labels) in enumerate(test_loader, start=1):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            if progress_callback is not None:
                progress_callback(batch_number, total_batches)

    # 3. Generowanie raportu klasyfikacji
    classes = ['Kot', 'Pies']
    print("\n--- SZCZEGÓŁOWE STATYSTYKI TESTOWE ---")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 4. Generowanie i zapisywanie Macierzy Pomyłek
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    # Rysowanie wykresu
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title('Macierz Pomyłek - Kot vs Pies')

    # Zapis do pliku tekstowego/graficznego
    output_graph = "confusion_matrix.png"
    plt.savefig(output_graph)
    print(
        f"\n[SUKCES] Macierz pomyłek została zapisana jako plik: '{output_graph}'")


if __name__ == "__main__":
    evaluate_model()
