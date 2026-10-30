import sys
from pathlib import Path
from typing import cast

import torch
from PIL import Image
from PIL import UnidentifiedImageError
from torchvision import transforms

from model import CatDogCNN


MODEL_PATH = Path("/tmp/cat_dog_cnn.pth")


def predict_image(image_path: str) -> None:
    image_file = Path(image_path)
    if not image_file.is_file():
        print(f"Error: file '{image_path}' does not exist.")
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
        with Image.open(image_file) as source_image:
            image = source_image.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        print(f"Error while opening the file: {error}")
        return

    image_tensor = cast(torch.Tensor, transform(image)).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatDogCNN().to(device)
    model.eval()

    if not MODEL_PATH.is_file():
        print(
            f"Error: model weights '{MODEL_PATH}' are missing.",
            "Wait for training to finish."
        )
        return

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    classes = ["cat", "dog"]
    result = classes[int(predicted_idx.item())]
    percentage = confidence.item() * 100

    print(
        f"\n[RESULT]: The model is {percentage:.2f}% confident",
        f"that this image is a {result.upper()}."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python3 predict.py <path_to_image.jpg>")
    else:
        predict_image(sys.argv[1])
