import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from datasets import load_dataset


class HuggingFaceCatsDogs(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['labels']

        # Convert non-RGB images before applying transforms.
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(batch_size=32):
    print("Ładowanie zbioru microsoft/cats_vs_dogs z Hugging Face...")
    raw_dataset = load_dataset(
        "microsoft/cats_vs_dogs",
        split="train",
        cache_dir="/tmp/ai-ml-piscine-rush"
    )

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_size = int(0.8 * len(raw_dataset))
    test_size = len(raw_dataset) - train_size
    raw_train, raw_test = random_split(
        raw_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = HuggingFaceCatsDogs(raw_train, transform=train_transform)
    test_dataset = HuggingFaceCatsDogs(raw_test, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    print(
        f"Załadowano! Trening: {len(train_dataset)} zdjęć,",
        f"Test: {len(test_dataset)} zdjęć."
    )
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    images, labels = next(iter(train_loader))
    print(f"Kształt paczki danych: {images.shape}")
