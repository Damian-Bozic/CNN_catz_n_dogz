import torch
import matplotlib.pyplot as plt
from collections.abc import Callable
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, \
    ConfusionMatrixDisplay

from dataset import get_data_loaders
from model import CatDogCNN

MODEL_PATH = Path("/tmp/cat_dog_cnn.pth")


def evaluate_model(
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    # 1. Select the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    # 2. Check for the trained model before downloading the test data.
    if not MODEL_PATH.is_file():
        print(
            f"Error: model weights '{MODEL_PATH}' were not found.",
            "Train the model first!"
        )
        return

    # 3. Load the test data and model.
    _, test_loader = get_data_loaders(batch_size=64)
    model = CatDogCNN().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []

    print("Collecting predictions from the test set...")
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

    # 4. Generate the classification report.
    classes = ["Cat", "Dog"]
    print("\n--- TEST CLASSIFICATION REPORT ---")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 5. Generate and save the confusion matrix.
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    # Draw the chart.
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion matrix: cat vs. dog")
    fig.tight_layout()

    # Save the chart.
    output_graph = "confusion_matrix.png"
    fig.savefig(output_graph)
    plt.close(fig)
    print(
        f"\nSaved the confusion matrix to '{output_graph}'.")


if __name__ == "__main__":
    evaluate_model()
